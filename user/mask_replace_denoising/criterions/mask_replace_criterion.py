import logging
import math
import os
from itertools import permutations
from typing import Optional

import numpy
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import metrics, utils, modules
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion
from fairseq.criterions.label_smoothed_cross_entropy import \
    LabelSmoothedCrossEntropyCriterion
from torch.distributions import Categorical

from user.mask_replace_denoising.models.mask_replace_model import \
    MaskReplaceModel

logger = logging.getLogger(__name__)


@register_criterion('marss')
class MaskReplaceCriterion(FairseqCriterion):
    """
    This objective combines replaced token detection and reconstruction.
    """

    def __init__(self, task, sentence_avg,
                 label_smoothing,
                 mlm_loss_coefficient,
                 rtd_loss_coefficient,
                 ae_loss_coefficient):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.label_smoothing = label_smoothing
        self.mlm_loss_coefficient = mlm_loss_coefficient
        self.rtd_loss_coefficient = rtd_loss_coefficient
        self.ae_loss_coefficient = ae_loss_coefficient
        self.steps = 0

        self.sampling_temperature = task.args.mask_replace_sampling_temperature
        self.sampling_topp = task.args.mask_replace_sampling_topp
        self.sampling_topk = task.args.mask_replace_sampling_topk

        if label_smoothing > 0:
            _criterion = LabelSmoothedCrossEntropyCriterion
        else:
            _criterion = CrossEntropyCriterion

        self.seq_criterion = _criterion.build_criterion(task.args, task)

        # --------------------------------------------------------------
        # Cross-lingual token replacement - Logit Masking
        # --------------------------------------------------------------
        self.xreplace_prob = task.args.xreplace_prob
        self.xreplace_prob_threshold = task.args.xreplace_prob_threshold
        languages = task.langs.split(',')

        if task.args.xreplace_prob:
            # for each language, point to the list of probabilities that each
            # token has in the target language. During sampling, we zero tokens
            # whose probability in the TARGET language is less than the threshold
            self.lang_token_filters = {}
            for i, src_lang in enumerate(languages):
                # find the token-probs of the target language
                trg_lang = languages[::-1][i]
                trg_probs = numpy.array([task.token_probs[trg_lang][symbol]
                                         for symbol in task.dictionary.symbols])

                self.lang_token_filters[f"{src_lang}_{trg_lang}"] = trg_probs
                # self.target_token_probs[trg_lang] = trg_probs

                # map the source lang_id to the prob list of the target language
                src_lang_id = task.dictionary.indices[f"[{src_lang}]"]
                self.lang_token_filters[src_lang_id] = numpy.array(trg_probs)
                pass

        # --------------------------------------------------------------
        # Cross-lingual token replacement - Language Embedding EMA
        # --------------------------------------------------------------
        self.xreplace_ema = task.xreplace_ema
        self.xreplace_ema_rate = task.xreplace_ema_rate
        if self.xreplace_ema:
            embed_dim = task.args.encoder_embed_dim * task.args.generator_ratio
            centroids = {}
            for lang in languages:
                emb = nn.Parameter(torch.randn(int(embed_dim)))
                emb.requires_grad = False
                emb.data.normal_(mean=0.0, std=0.02)
                centroids[f'[{lang}]'] = emb
            self.centroids = nn.ParameterDict(centroids)

            self.xreplace_ema_log = {}

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float,
                            metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

        parser.add_argument('--mlm-loss-coefficient', type=float,
                            metavar='D', default=1,
                            help='multiplier for the MLM loss of the generator')

        parser.add_argument('--rtd-loss-coefficient', type=float,
                            metavar='D', default=50,
                            help='multiplier for the replaced token detection '
                                 '(RTD) loss, of the encoder.')

        parser.add_argument('--ae-loss-coefficient', type=float,
                            metavar='D', default=1,
                            help='multiplier for the autoencoder/reconstruction'
                                 ' loss.')

        parser.add_argument('--rtd-loss-class-weight',
                            action='store_true', default=False,
                            help='Re-weight the positive examples in the '
                                 'binary-cross-entropy to increase the recall.')

        parser.add_argument('--test-multitask',
                            action='store_true', default=False,
                            help='Hack that does NOT replace the encoder\'s '
                                 'input to test the performance of multitasking.')

    def _detok(self, tensor):
        return numpy.array([self.task.dictionary.symbols[t] for t in tensor])

    def _debug_batch(self, seqs):
        return numpy.array([numpy.array(list([[self.task.dictionary[x]
                                               for x in s]
                                              for s in zip(*r)])).T
                            for r in zip(*seqs)])

    def _debug_sample(self, seqs):
        return numpy.array(list([[self.task.dictionary[x] for x in s]
                                 for s in zip(*seqs)])).T

    def log_batch(self, batch):
        if self.steps % self.task.args.log_interval == 0:
            path = self.task.args.save_dir
            batch = self._debug_batch(batch)

            with open(os.path.join(path, "samples.html"), "w") as f:
                f.write(
                    '<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css">' + '\n')
                for b in batch[:10]:
                    f.write(pandas.DataFrame(b,
                                             index=["x_1", "x_2",
                                                    "y_1", "y_2",
                                                    "target"]).to_html(
                        classes='table table-hover',
                        header=False) + "</br>")

    @staticmethod
    def top_k_top_p_filtering(
            logits: torch.Tensor,
            top_k: int = 0,
            top_p: float = 1.0,
            filter_value: float = -float("Inf"),
            min_tokens_to_keep: int = 1,
    ) -> torch.Tensor:
        """
        Filter a distribution of logits using top-k
        and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            top_k: if top_k > 0, keep only top k tokens with highest probability
                (top-k filtering).
            top_p: if top_p < 1.0, keep the top tokens with cumulative
                probability >= top_p (nucleus filtering). Nucleus filtering is
                described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            min_tokens_to_keep: Make sure we keep at least min_tokens_to_keep
                per batch example in the output

        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

        Returns: logits (batch size, vocabulary size)

        """

        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep),
                        logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][
                ..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1),
                                            dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[...,
                                                :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1,
                                                                 sorted_indices,
                                                                 sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

    def _xmask(self, tokens, lengths, logits, masked_tokens):
        # find the language (id) of each sentence
        sent_langid = torch.gather(tokens, 1, (lengths - 1).unsqueeze(1))
        # find language (id) of each (to be replaced) masked token
        token_langid = sent_langid.expand(-1, tokens.size(1))[masked_tokens]

        # map language-ids to language-masks, todo: can be made more efficient
        u, inv = numpy.unique(token_langid.cpu(), return_inverse=True)
        tok_filters = numpy.array([self.lang_token_filters[x] for x in u])[inv]
        tok_filters = torch.tensor(tok_filters,
                                   dtype=logits.dtype,
                                   device=logits.device)
        token_masks = tok_filters < self.xreplace_prob_threshold
        logits = logits.masked_fill(token_masks, -numpy.inf)

        return logits

    def _xshift(self, tokens, lengths, encoder_out):
        # gather the language (id) tokens of each sentence
        lang_ids = torch.gather(tokens, 1, (lengths - 1).unsqueeze(1))
        # langs = self._detok(lang_ids)
        features = encoder_out.encoder_out.transpose(0, 1)

        # ------------------------------------------------------------------
        # 1. Compute the (language) centroids of each sentence in the  batch
        # ------------------------------------------------------------------
        # We exclude the special tokens (e.g., <s>, ['LANG'], etc.) from the
        # the calculation of the language average (centroid)
        # ------------------------------------------------------------------

        # mask all the tokens after (including) the </s>
        non_special_mask = (torch.arange(max(lengths), device=tokens.device)
                            .unsqueeze(0)
                            .expand_as(tokens)) < (lengths - 2).unsqueeze(1)
        # mask the <s> tokens
        non_special_mask[:, 0] = False

        # obtain the clean sentence-level averages
        non_special_outputs = (non_special_mask.unsqueeze(-1) * features)
        mu = non_special_outputs.sum(1) / non_special_mask.sum(-1).unsqueeze(-1)

        # ------------------------------------------------------------------
        # 2. Update each language's EMA
        # ------------------------------------------------------------------
        for lang_id in lang_ids.unique():  # for each language in the batch
            # for the given language, identify the sentences that belong to it
            # and obtain the language average for the batch
            mu_current = mu[(lang_ids == lang_id).squeeze()].mean(0)
            langid = self.task.dictionary.symbols[lang_id.item()]

            # update the value of the EMA
            mu_new = (self.xreplace_ema_rate * self.centroids[langid]
                      + (1 - self.xreplace_ema_rate) * mu_current)

            # logging
            cos = F.cosine_similarity(self.centroids[langid], mu_current, dim=0)
            l2 = torch.norm(self.centroids[langid] - mu_current)
            self.xreplace_ema_log[f"ema_cosine_self-{langid[1:3]}"] = cos.item()
            self.xreplace_ema_log[f"ema_euclid_self-{langid[1:3]}"] = l2.item()

            # update ema
            self.centroids[langid].data = mu_new

        # ------------------------------------------------------------------
        # 3. Shift outputs according to the difference of the language EMAs
        # ------------------------------------------------------------------
        for src_id, trg_id in permutations(lang_ids.unique(), 2):
            src_token = self.task.dictionary.symbols[src_id.item()]
            trg_token = self.task.dictionary.symbols[trg_id.item()]

            # diff between target and source language EMAs
            d = self.centroids[trg_token] - self.centroids[src_token]
            # tensors with diffs
            diffs = ((lang_ids == src_id) * d).unsqueeze(1).expand_as(features)
            # apply shift
            features = features + diffs

            # logging
            cos = F.cosine_similarity(self.centroids[trg_token],
                                      self.centroids[src_token], dim=0)
            self.xreplace_ema_log[f"ema_cosine"] = cos.item()
            self.xreplace_ema_log[f"ema_euclid"] = torch.norm(d).item()

        return features

    def mask_replace(self, model, tokens, lengths, target):
        masked_tokens = tokens.eq(self.task.mask_idx)

        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if masked_tokens.device == torch.device('cpu'):
            if not masked_tokens.any():
                masked_tokens.fill_(True)
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )

        logits, outputs = model.generator(tokens, lengths,
                                          masked_tokens=masked_tokens)

        # calculate MLM loss
        targets = target[masked_tokens]
        loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )

        # replace the masked tokens in with samples from generator
        with torch.no_grad():

            # generate new logits by shifting the outputs towards the EMA
            # of the target language's outputs
            if self.xreplace_ema:
                shifted_outputs = self._xshift(tokens, lengths, outputs)
                logits = model.generator.output_layer(shifted_outputs,
                                                      masked_tokens=masked_tokens)

            # Masking for cross-lingual token replacement
            if self.xreplace_prob:
                logits = self._xmask(tokens, lengths, logits, masked_tokens)

            logits = logits / self.sampling_temperature
            logits = self.top_k_top_p_filtering(logits,
                                                self.sampling_topk,
                                                self.sampling_topp)
            samples = Categorical(logits=logits).sample()

            replace_tokens = tokens.masked_scatter(masked_tokens, samples)
            correct = target[masked_tokens].eq(samples).sum().item()

        # to debug the batch run:
        # check = numpy.array(list([[self.task.dictionary[x] for x in s]
        #                           for s in zip(*[tokens[0],
        #                                          target[0],
        #                                          replace_tokens[0]])])).T

        return replace_tokens, loss, correct

    def replacement_detection(self, model: MaskReplaceModel, features, targets):

        """
        Replaced Token Detection loss. Given the encoder's outputs,
        we predict whether each of its input tokens is original or fake
        (i.e., sample from generator).
        The model is trained to *predict the fakes*!
        i.e., 1=fake, 0=original


        """
        if self.task.args.rtd_loss_class_weight:
            pos_weight = torch.tensor(
                (1 - self.task.args.mask) / self.task.args.mask,
                device=targets.device,
                dtype=features.dtype)
        else:
            pos_weight = torch.tensor(1.,
                                      device=targets.device,
                                      dtype=features.dtype)

        logits = model.replacement_detection(features).squeeze(2)
        loss = F.binary_cross_entropy_with_logits(logits, targets.float(),
                                                  reduction='sum',
                                                  pos_weight=pos_weight)
        correct = (logits > 0).eq(targets).sum().item()
        del logits
        return loss, correct

    @staticmethod
    def apply_permutation(x, permutation):
        d1, d2 = x.size()
        ret = x[
            torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
            permutation.flatten()
        ].view(d1, d2)
        return ret

    def translator(
            self,
            model,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the forward() of the TransformerModel class, in order
        to get the encoder_out tensor.

        """
        encoder_out = model.encoder(
            src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )

        if (self.task.args.train_sentence_level > 0
                and not self.training
                and self.steps % self.task.args.train_sentence_level == 0):
            # find the timesteps containing the langids
            lsteps = torch.gather(src_tokens, 1, (src_lengths - 1).unsqueeze(1))
            # create a mask the allows only the lsteps
            mask = src_tokens.eq(lsteps).transpose(0, 1).unsqueeze(2)
            # encoder_out.encoder_out is what is used by model.decoder
            encoder_out_sent = encoder_out.encoder_out.clone() * mask
            encoder_out = encoder_out._replace(encoder_out=encoder_out_sent)

        decoder_out = model.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return encoder_out, decoder_out

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # -----------------------------------------------------
        # 1. Replace masked tokens and compute MLM loss
        # -----------------------------------------------------
        if self.task.args.mask_replace:
            enc_tokens, enc_mlm_loss, enc_mlm_correct = self.mask_replace(
                model,
                sample['net_input']['src_tokens'],
                sample['net_input']['src_lengths'],
                sample['target'])
        else:
            enc_tokens = sample['net_input']['src_tokens']
            enc_mlm_loss = None
            enc_mlm_correct = 0
        if self.task.args.mask_replace_decoder:
            assert self.task.args.mask_decoder > 0
            dec_tokens, dec_mlm_loss, dec_mlm_correct = self.mask_replace(
                model,
                sample['net_input']['prev_output_tokens'],
                sample['net_input']['src_lengths'],
                sample['target_self'])
        else:
            dec_tokens = sample['net_input']['prev_output_tokens']
            dec_mlm_loss = None
            dec_mlm_correct = 0

        # -----------------------------------------------------
        # 2. Permute tokens. It has to be done after the replacements
        # -----------------------------------------------------
        if 'permutations' in sample['net_input']:
            enc_tokens = self.apply_permutation(
                enc_tokens, sample['net_input']['permutations'])

        #     todo: remove hack
        if self.task.args.test_multitask:
            enc_tokens = sample['net_input']['src_tokens']

        # -----------------------------------------------------
        # 2. compute the reconstruction loss (cross-entropy)
        # -----------------------------------------------------
        encoder_out, decoder_out = self.translator(
            model,
            enc_tokens,  # after the (optional) addition of noise
            sample['net_input']['src_lengths'],
            dec_tokens)  # after the (optional) addition of noise
        ae_loss, nll_loss = self.seq_criterion.compute_loss(model, decoder_out,
                                                            sample,
                                                            reduce=reduce)

        # -----------------------------------------------------
        # 3. compute the RTD loss, with labels: True=fake, False=original
        # * the inner_states of the encoder do NOT include the embeddings,
        # so the 1st layer has index 0.
        # * the inner_states of the decoder DO include the embeddings,
        # so the 1st layer has index 1, and embeddings have index 0.
        # -----------------------------------------------------
        if self.task.args.replacement_detection_encoder != 0:
            index = self.task.args.replacement_detection_encoder
            if index > 0:
                index -= 1
            enc_features = encoder_out.encoder_states[index]
            enc_rtd_loss, enc_rtd_correct = self.replacement_detection(
                model, enc_features, ~sample['target'].eq(enc_tokens))
            del enc_features
        else:
            enc_rtd_loss = None
            enc_rtd_correct = 0

        if self.task.args.replacement_detection_decoder != 0:
            index = self.task.args.replacement_detection_decoder
            dec_features = decoder_out[1]['inner_states'][index]

            if self.task.args.mask_replace and not self.task.args.mask_replace_decoder:
                rtd_labels = ~sample['target'].eq(enc_tokens)
            else:
                rtd_labels = ~sample['target_self'].eq(dec_tokens)

            dec_rtd_loss, dec_rtd_correct = self.replacement_detection(
                model, dec_features, rtd_labels)
            del dec_features
        else:
            dec_rtd_loss = None
            dec_rtd_correct = 0

        encoder_out.encoder_states.clear()
        decoder_out[1]['inner_states'].clear()
        decoder_out[1]['attn'].clear()
        del decoder_out, encoder_out

        # -----------------------------------------------------
        # Aggregate losses
        # -----------------------------------------------------
        loss = ae_loss * self.ae_loss_coefficient
        mlm_losses = []
        rtd_losses = []

        # MLM losses
        if self.task.args.mask_replace:
            loss = loss + (enc_mlm_loss * self.mlm_loss_coefficient)
            mlm_losses.append(enc_mlm_loss)
        if self.task.args.mask_replace_decoder:
            loss = loss + (dec_mlm_loss * self.mlm_loss_coefficient)
            mlm_losses.append(dec_mlm_loss)

        # RTD losses
        if self.task.args.replacement_detection_encoder:
            loss = loss + (enc_rtd_loss * self.rtd_loss_coefficient)
            rtd_losses.append(enc_rtd_loss)
        if self.task.args.replacement_detection_decoder:
            loss = loss + (dec_rtd_loss * self.rtd_loss_coefficient)
            rtd_losses.append(dec_rtd_loss)

        # compute sample sizes
        sample_size = sample['target'].size(0) if self.sentence_avg else sample[
            'ntokens']

        logging_output = {
            'loss': loss.data,
            'ae_loss': ae_loss.item(),
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }

        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------
        if len(mlm_losses) > 0:
            logging_output['mlm_loss'] = sum([x.item() for x in mlm_losses])
            logging_output['mlm_sample_size'] = 0
            if enc_mlm_loss is not None:
                logging_output['mlm_sample_size'] += \
                    sample['net_input']["src_tokens"].eq(
                        self.task.mask_idx).sum().item()
            if dec_mlm_loss is not None:
                logging_output['mlm_sample_size'] += \
                    sample['net_input']["prev_output_tokens"].eq(
                        self.task.mask_idx).sum().item()
            logging_output['mlm_correct'] = enc_mlm_correct + dec_mlm_correct
            mlm_losses.clear()

        if len(rtd_losses) > 0:
            logging_output['rtd_loss'] = sum([x.item() for x in rtd_losses])
            logging_output['rtd_correct'] = enc_rtd_correct + dec_rtd_correct
            logging_output['rtd_sample_size'] = sample['ntokens'] * len(
                rtd_losses)
            rtd_losses.clear()

        self.log_batch([sample['net_input']['src_tokens'], enc_tokens,
                        sample['net_input']['prev_output_tokens'], dec_tokens,
                        sample['target']])

        if self.xreplace_ema:
            logging_output.update(self.xreplace_ema_log)

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ae_sum = sum(log.get('ae_loss', 0) for log in logging_outputs)

        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        mlm_sample_size = sum(log.get('mlm_sample_size', 0)
                              for log in logging_outputs)
        rtd_sample_size = sum(log.get('rtd_sample_size', 0)
                              for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2),
                           sample_size, round=3)
        metrics.log_scalar('ae_loss', ae_sum / sample_size / math.log(2),
                           sample_size, round=3)

        if sample_size != ntokens:
            metrics.log_scalar('nll_ae_loss', ae_sum / ntokens / math.log(2),
                               ntokens, round=3)
            metrics.log_derived('ae_ppl', lambda meters: utils.get_perplexity(
                meters['nll_ae_loss'].avg))
        else:
            metrics.log_derived('ae_ppl', lambda meters: utils.get_perplexity(
                meters['ae_loss'].avg))

        # MLM
        if mlm_sample_size > 0:
            mlm_sum = sum(log.get('mlm_loss', 0) for log in logging_outputs)

            mlm_corr_sum = sum(log.get('mlm_correct', 0)
                               for log in logging_outputs)

            metrics.log_scalar('mlm_loss',
                               mlm_sum / mlm_sample_size / math.log(2),
                               mlm_sample_size, round=3)
            metrics.log_scalar('mlm_accuracy',
                               100.0 * mlm_corr_sum / mlm_sample_size,
                               mlm_sample_size, round=2)
            metrics.log_derived('mlm_ppl', lambda meters: utils.get_perplexity(
                meters['mlm_loss'].avg))

        # Replacement Detection
        if rtd_sample_size > 0:
            rtd_sum = sum(log.get('rtd_loss', 0) for log in logging_outputs)
            rtd_corr_sum = sum(log.get('rtd_correct', 0)
                               for log in logging_outputs)

            metrics.log_scalar('rtd_loss',
                               rtd_sum / rtd_sample_size / math.log(2),
                               rtd_sample_size, round=3)

            metrics.log_scalar('rtd_accuracy',
                               100.0 * rtd_corr_sum / rtd_sample_size,
                               rtd_sample_size, round=2)

        for k, v in logging_outputs[0].items():
            if k.startswith("ema_"):
                s = sum(log.get(k, 0) for log in logging_outputs)
                metrics.log_scalar(k, s / len(logging_outputs),
                                   1 / len(logging_outputs), round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
