import logging
import re
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import register_model, register_model_architecture, \
    FairseqDecoder
from fairseq.models.bart import mbart_base_architecture
from fairseq.models.transformer import TransformerModel, \
    TransformerEncoder
from fairseq.modules import LayerNorm
from fairseq.modules.transformer_sentence_encoder import \
    init_bert_params

logger = logging.getLogger(__name__)


@register_model('marss')
class MaskReplaceModel(TransformerModel):

    def __init__(self, args, translator, generator, discriminator):
        super().__init__(args, translator.encoder, translator.decoder)
        self.args = args
        self.generator = generator
        self.discriminator = discriminator
        self.classifier = None

        if args.task == "weighted_backtranslation":
            for n, p in self.discriminator.named_parameters():
                logger.warning(f'freezing layer: ({n})')
                p.requires_grad = False
            for n, p in self.encoder.named_parameters():
                # if n.startswith('layers'):
                logger.warning(f'freezing layer: ({n})')
                p.requires_grad = False

        if args.task == "mask_replace_token_prediction":
            self.classifier = ProjectionHead(
                embed_dim=translator.encoder.embed_tokens.weight.shape[1],
                output_dim=len(translator.encoder.dictionary),
                non_linearity=False, norm_input=False
            )

        for frozen_layer_prefix in eval(self.args.freeze_layers):
            for name, param in self.named_parameters():
                if re.match("^" + frozen_layer_prefix, name):
                    logger.warning(f'freezing pretraining layer : ({name})')
                    param.requires_grad = False

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        print()

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)

        parser.add_argument('--tie-generator-translator-embeddings',
                            action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 'between the generator and translator models'
                                 ' (requires shared dictionary and embed dim)')

        parser.add_argument('--tie-generator-encoder', action='store_true',
                            help='tie the weights of the generator and the encoder')

        parser.add_argument('--tie-generator-decoder', action='store_true',
                            help='tie the weights of the generator and the decoder')

        parser.add_argument('--tie-generator-projections',
                            action='store_true',
                            help='Applicable online when `generator-ratio`<1`, '
                                 'which requires to use down-up '
                                 'projections from the embeddings to the '
                                 'transformer\'s body.')

        parser.add_argument('--generator-ratio', type=float, metavar='D',
                            help='multiplier for generator size based on encoder '
                                 'size for: hidden-size, FFN-size, '
                                 'and num-attention-heads')

        parser.add_argument('--replacement-detection-encoder',
                            type=int, metavar='D', default=0,
                            help="add a replaced-token-detection (discriminator)"
                                 " over the output of the k-th layer of the encoder."
                                 " K=0 disables the RTD head.")

        parser.add_argument('--replacement-detection-decoder',
                            type=int, metavar='D', default=0,
                            help="add a replaced-token-detection (discriminator)"
                                 " over the output of the k-th layer of the decoder."
                                 " K=0 disables the RTD head.")

        parser.add_argument('--freeze-layers',
                            metavar='LIST',
                            default='[]',
                            help='(Key prefixes of) layers to freeze from a'
                                 'pretrained checkpoint.')

        parser.add_argument('--reset-pretrained-layers', type=str,
                            metavar='LIST',
                            default='[]',
                            help='(Key prefixes of) layers to remove from a'
                                 'pretrained checkpoint.')

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        translator = TransformerModel.build_model(args, task)
        discriminator = None
        generator = None

        # Generator definition
        if (args.task in ["marss_pretraining", "mask_replace_token_prediction"]
                and (args.mask_replace or args.mask_replace_decoder)):
            # ----------------------------------------------------------
            # 1. Express generator hyper-parameters (arguments),
            # as a ratio of the encoder's hyper-parameters
            # ----------------------------------------------------------
            gen_hps = {
                "encoder_embed_dim",
                "encoder_ffn_embed_dim",
                "encoder_attention_heads"
            }
            gen_args = {k: type(v)(v * float(args.generator_ratio))
            if k in gen_hps else v
                        for k, v in vars(args).items()}
            gen_args = Namespace(**gen_args)

            # ----------------------------------------------------------
            # 2. Initialize the generator
            # ----------------------------------------------------------
            if args.tie_generator_translator_embeddings:
                logger.info("Using *shared* (input/output) embeddings for the "
                            "generator and the autoencoder.")
                generator_embed_tokens = translator.encoder.embed_tokens
            else:
                logger.info("Using *separate*  embeddings for the generator "
                            "and the autoencoder.")
                generator_embed_tokens = TransformerModel.build_embedding(
                    gen_args, translator.encoder.dictionary,
                    gen_args.encoder_embed_dim, gen_args.encoder_embed_path
                )

            if args.tie_generator_encoder:
                assert not args.tie_generator_decoder
                logger.info("Using *shared* weights for the Transformer body "
                            "of the generator and the encoder.")
                generator_body = translator.encoder

            else:
                logger.info("Using *separate* weights for the Transformer body "
                            "of the generator and the encoder.")
                generator_body = TransformerEncoder(gen_args,
                                                    translator.encoder.dictionary,
                                                    generator_embed_tokens)

                if args.tie_generator_decoder:
                    raise NotImplementedError
                    # assert not args.tie_generator_encoder
                    # generator_body = translator.decoder

            generator = MaskedGenerator(gen_args,
                                        translator.encoder.dictionary,
                                        generator_body)

        # ----------------------------------------------------------
        # Replacement Detection Discriminator (Head)
        # ----------------------------------------------------------
        use_discriminator = False

        if args.task == "marss_pretraining":
            if args.replacement_detection_encoder != 0:
                msg = "You must enable '--mask-replace', " \
                      "in order to use a replacement detection discriminator."
                assert args.mask_replace, msg
                use_discriminator = True

            if args.replacement_detection_decoder != 0:
                msg = "Cannot use a replacement detection discriminator " \
                      "over the decoder, without '--mask-replace-decoder'" \
                      " or '--mask-decoder-paired'!"
                assert args.mask_replace_decoder or args.mask_decoder_paired, msg
                use_discriminator = True

        # load the pretrained discriminator for scoring the synthetic data
        elif args.task == "weighted_backtranslation":
            use_discriminator = True

        if use_discriminator:
            logger.info("Using Replacement Detection Discriminator.")
            discriminator = ProjectionHead(
                embed_dim=args.encoder_embed_dim,
                output_dim=1,
                activation_fn=args.activation_fn,
                norm_input=True,
            )

        return cls(args, translator, generator, discriminator)

    def assert_weight_sharing(self):
        if self.args.tie_generator_translator_embeddings:
            assert self.encoder.embed_tokens.weight is self.generator.encoder.embed_tokens.weight
            assert torch.all(torch.eq(self.encoder.embed_tokens.weight,
                                      self.generator.encoder.embed_tokens.weight)).item()

    def replacement_detection(self, features):
        """
        Given the encoder's outputs,
        we predict whether each of its input tokens is original or fake
        (i.e., sample from generator).

        """
        # self.assert_weight_sharing()

        # T x B x C -> B x T x C
        x = features.transpose(0, 1)
        logits = self.discriminator(x)
        return logits

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + '.' if name != '' else ''

        # remove generator and rtd layers when finetuning
        keys_to_delete = []
        if self.args.task in ('translation',
                              'translation_from_pretrained_bart',
                              'backtranslation_from_pretrained_bart',
                              'backtranslation',
                              'unsupervised_translation_from_pretrained'):
            for k in state_dict.keys():
                if k.startswith(prefix + 'generator.') or k.startswith(
                        prefix + 'discriminator.'):
                    logger.warning(f'deleting pretraining layer '
                                   f'from checkpoint: ({k})')
                    keys_to_delete.append(k)
        elif self.args.task == 'weighted_backtranslation':
            for k in state_dict.keys():
                if k.startswith(prefix + 'generator.'):
                    logger.warning(f'deleting pretraining layer '
                                   f'from checkpoint: ({k})')
                    keys_to_delete.append(k)
        elif self.args.task == 'mask_replace_token_prediction':
            for k in state_dict.keys():
                if k.startswith(prefix + 'discriminator.'):
                    logger.warning(f'deleting pretraining layer '
                                   f'from checkpoint: ({k})')
                    keys_to_delete.append(k)

        for k in keys_to_delete:
            del state_dict[k]

        if self.generator is not None:
            if 'generator.output_projection.layer_norm_ffn.weight' not in state_dict:
                w = state_dict['generator.output_projection.layer_norm.weight']
                b = state_dict['generator.output_projection.layer_norm.bias']
                state_dict[
                    'generator.output_projection.layer_norm_ffn.weight'] = w
                state_dict[
                    'generator.output_projection.layer_norm_ffn.bias'] = b
                del state_dict['generator.output_projection.layer_norm.weight']
                del state_dict['generator.output_projection.layer_norm.bias']

                logger.info(" ** Loading state_dict from an old model!")

        # remove pretrained weights from checkpoint
        if self.args.task in ('translation',
                              'translation_from_pretrained_bart',
                              'backtranslation_from_pretrained_bart',
                              'backtranslation',
                              'unsupervised_translation_from_pretrained'):
            model_state = self.state_dict()
            for k in list(state_dict.keys()):
                for prefix in eval(self.args.reset_pretrained_layers):
                    if re.match("^" + prefix, k):
                        logger.warning(f'randomizing pretraining layer : ({k})')
                        state_dict[k].data = model_state[k].data
        print()


class MaskedGenerator(FairseqDecoder):
    """
    ELECTRA-inspired generator

    Implements the :class:`~fairseq.models.FairseqDecoder` interface required
    by :class:`~fairseq.models.FairseqLanguageModel`.
    """

    def __init__(self, args, dictionary, encoder):
        super().__init__(dictionary)
        self.args = args
        self.encoder = encoder

        # projection to vocabulary - reuse the embedding layer's weights
        self.output_projection = ProjectionHead(
            embed_dim=self.encoder.embed_tokens.weight.shape[1],
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            # non_linearity=False,
            weight=self.encoder.embed_tokens.weight
        )

        # if the embedding size and the transformer size are different,
        # then add a projection in-between
        if args.generator_ratio < 1:
            assert not args.tie_generator_encoder
            assert not args.tie_generator_decoder

            gen_dim = args.encoder_embed_dim
            enc_dim = int(args.encoder_embed_dim * args.generator_ratio ** -1)

            self.W_down = nn.Linear(enc_dim, gen_dim, bias=False)
            self.W_up = nn.Linear(gen_dim, enc_dim, bias=False)

            if args.tie_generator_projections:
                self.W_up.weight = self.W_down.weight

            # self.encoder.layer_norm was initialized based on the dimensions
            # of the encoder's embeddings, therefore we re-initialize it here
            if self.encoder.layer_norm is not None:
                self.encoder.layer_norm = LayerNorm(args.encoder_embed_dim)

            # replace the forward_embedding() of the encoder with another
            # that projects the embeddings to the appropriate dimensionality
            self.encoder.forward_embedding = self.forward_embedding

        else:
            self.W_down = None
            self.W_up = None

    def forward_embedding(self, src_tokens):
        """
        Hacky solution for overriding  TransformerEncoder.forward_embedding
        in order to add the projection between the embedding and TransformerEncoder
        """
        x, embed = TransformerEncoder.forward_embedding(self.encoder,
                                                        src_tokens)
        if self.W_down is not None:
            x = self.W_down(x)

        return x, embed

    def forward(self, src_tokens, src_lengths, masked_tokens=None, **unused):

        outputs = self.extract_features(src_tokens, src_lengths)

        # T x B x C -> B x T x C
        features = outputs.encoder_out.transpose(0, 1)
        logits = self.output_layer(features, masked_tokens=masked_tokens)

        return logits, outputs

    def extract_features(self, src_tokens, src_lengths, **kwargs):
        features = self.encoder(src_tokens, src_lengths)
        return features

    def output_layer(self, features, masked_tokens=None, **unused):
        # Only project the unmasked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        if self.W_up is not None:
            if self.args.tie_generator_projections:
                features = F.linear(features, self.W_down.weight.t())
            else:
                features = self.W_up(features)

        # project back to size of vocabulary with bias
        x = self.output_projection(features)
        del features
        return x


class ProjectionHead(nn.Module):

    def __init__(self, embed_dim, output_dim, activation_fn=None,
                 weight=None, non_linearity=True, norm_input=False):
        super().__init__()
        self.non_linearity = non_linearity
        self.norm_input = norm_input

        if self.norm_input:
            self.layer_norm_in = LayerNorm(embed_dim)

        if self.non_linearity:
            self.dense = nn.Linear(embed_dim, embed_dim)
            self.activation_fn = utils.get_activation_fn(activation_fn)
            self.layer_norm_ffn = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the unmasked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        if self.norm_input:
            features = self.layer_norm_in(features)

        if self.non_linearity:
            x = self.dense(features)
            x = self.activation_fn(x)
            x = self.layer_norm_ffn(x)
        else:
            x = features

        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias

        del features
        return x


@register_model_architecture("marss", "marss_base")
def base_architecture(args):
    # --------------------------------------------------------------------
    # Encoder-Decoder dimensionality
    # --------------------------------------------------------------------
    heads = 12
    layers = 6
    emb_dim = 768
    ffn_dim = 4 * emb_dim

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', emb_dim)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', ffn_dim)
    args.encoder_layers = getattr(args, 'encoder_layers', layers)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads',
                                           heads)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', emb_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', ffn_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', layers)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads',
                                           heads)
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', emb_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', emb_dim)

    # --------------------------------------------------------------------
    # Regularization, Normalization, Initialization
    # --------------------------------------------------------------------

    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)

    # args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', True)
    # args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', True)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)  # !!
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)  # !!

    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', True)

    # Similar to T2T: research has shown that prenorm is better
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before',
                                            True)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before',
                                            True)

    # Tie all embeddings
    args.share_decoder_input_output_embed = getattr(
        args, 'share_decoder_input_output_embed', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', True)

    # --------------------------------------------------------------------
    # Generator
    # --------------------------------------------------------------------
    args.tie_generator_translator_embeddings = getattr(
        args, "tie_generator_translator_embeddings", True)

    args.tie_generator_encoder = getattr(args, "tie_generator_encoder", False)
    args.tie_generator_decoder = getattr(args, "tie_generator_decoder", False)
    args.tie_generator_projections = getattr(args, "tie_generator_projections",
                                             False)
    args.generator_ratio = getattr(args, 'generator_ratio', 0.25)

    # in case we've missed anything
    mbart_base_architecture(args)


@register_model_architecture("marss", "marss_analysis")
def marss_analysis_architecture(args):
    heads = 8
    layers = 6
    emb_dim = 512
    ffn_dim = 4 * emb_dim

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', emb_dim)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', ffn_dim)
    args.encoder_layers = getattr(args, 'encoder_layers', layers)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads',
                                           heads)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', emb_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', ffn_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', layers)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads',
                                           heads)
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', emb_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', emb_dim)

    base_architecture(args)


@register_model_architecture("marss", "marss_medium")
def marss_medium_architecture(args):
    heads = 8
    layers = 6
    emb_dim = 768
    ffn_dim = 4 * emb_dim

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', emb_dim)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', ffn_dim)
    args.encoder_layers = getattr(args, 'encoder_layers', layers)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads',
                                           heads)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', emb_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', ffn_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', layers)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads',
                                           heads)
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', emb_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', emb_dim)

    base_architecture(args)


@register_model_architecture("marss", "marss_large")
def marss_large_architecture(args):
    heads = 12
    layers = 6
    emb_dim = 1024
    ffn_dim = 4 * emb_dim

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', emb_dim)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', ffn_dim)
    args.encoder_layers = getattr(args, 'encoder_layers', layers)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads',
                                           heads)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', emb_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', ffn_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', layers)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads',
                                           heads)
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', emb_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', emb_dim)

    base_architecture(args)


@register_model_architecture("marss", "marss_xlm")
def marss_xlm_architecture(args):
    heads = 8
    layers = 6
    emb_dim = 1024
    ffn_dim = 4 * emb_dim

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', emb_dim)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', ffn_dim)
    args.encoder_layers = getattr(args, 'encoder_layers', layers)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads',
                                           heads)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', emb_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', ffn_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', layers)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads',
                                           heads)
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', emb_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', emb_dim)

    base_architecture(args)
