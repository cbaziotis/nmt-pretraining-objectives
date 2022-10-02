import json
import logging
import math
import os
from collections import defaultdict
from typing import Optional

import numpy
import torch
import torch.nn.functional as F
from fairseq import metrics, utils, modules
from fairseq.criterions import FairseqCriterion, register_criterion
from torch import distributions
from torch.distributions import Categorical

logger = logging.getLogger(__name__)


@register_criterion('analysis_entropy')
class AnalysisEntropyCriterion(FairseqCriterion):
    """
    This objective combines replaced token detection and reconstruction.
    """

    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.analysis = defaultdict(list)

        try:
            self.vocab = self.task.src_dict
        except:
            self.vocab = self.task.source_dictionary

    def save_analysis(self, save_dir, log_outputs):

        fname = os.path.join(save_dir, "analysis_entropy.json")
        open(fname, 'w').close()

        # for k, v in self.analysis.items():
        #     self.analysis[k] = list(itertools.chain.from_iterable(v))

        for k, v in self.analysis.items():
            self.analysis[k] = numpy.mean(self.analysis[k])

        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(self.analysis, f, ensure_ascii=False, indent=4)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # parser.add_argument('--analysis-method', type=str, metavar='VAL',
        #                     required=True,
        #                     help='choose between: entropy, states')

    def _detok(self, tensor):
        return [self.task.dictionary.symbols[t] for t in tensor]

    def _debug_batch(self, seqs):
        return [numpy.array(list([[self.task.dictionary[x] for x in s]
                                  for s in zip(*r)])).T
                for r in zip(*seqs)]

    def _debug_sample(self, seqs):
        return numpy.array(list([[self.task.dictionary[x] for x in s]
                                 for s in zip(*seqs)])).T

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

        targets = target[masked_tokens]

        loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )

        # replace the masked tokens in with samples from generator
        samples = Categorical(logits=logits).sample()
        replace_tokens = tokens.masked_scatter(masked_tokens, samples)
        correct = target[masked_tokens].eq(samples).sum().item()

        return replace_tokens, loss, correct

    @staticmethod
    def apply_permutation(x, permutation):
        d1, d2 = x.size()
        ret = x[
            torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
            permutation.flatten()
        ].view(d1, d2)
        return ret

    @staticmethod
    def translator(
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

    def compute_loss(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            # reduction='sum' if reduce else 'none',
        )
        return loss, lprobs

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        b1, b2 = sample['net_input']['src_tokens'].size()
        # -----------------------------------------------------
        # 1. Replace masked tokens and compute MLM loss
        # -----------------------------------------------------
        dec_tokens = sample['net_input']['prev_output_tokens']
        enc_tokens = sample['net_input']['src_tokens']

        if self.task.args.mask_replace:
            enc_tokens, enc_mlm_loss, enc_mlm_correct = self.mask_replace(
                model,
                sample['net_input']['src_tokens'],
                sample['net_input']['src_lengths'],
                sample['target'])

        # -----------------------------------------------------
        # 2. Permute tokens. It has to be done after the replacements
        # -----------------------------------------------------
        if 'permutations' in sample['net_input']:
            enc_tokens = self.apply_permutation(
                enc_tokens, sample['net_input']['permutations'])

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
        loss, lprobs = self.compute_loss(model, decoder_out, sample)

        noisy_toks = ~enc_tokens.eq(sample['target'])
        padded_toks = enc_tokens.eq(self.vocab.pad_index)
        is_noisy = noisy_toks & (~padded_toks)
        is_real = (~noisy_toks) & (~padded_toks)

        # compute sample sizes
        sample_size = sample['target'].size(0) if self.sentence_avg else sample[
            'ntokens']

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }

        # b = self._debug_batch([sample['net_input']['src_tokens'],
        #                        sample['net_input']['prev_output_tokens'],
        #                        sample['target']])

        # self.analysis["encoder"].append([x[0].tolist() for x in b])
        # self.analysis["decoder"].append([x[1].tolist() for x in b])
        # self.analysis["targets"].append([x[2].tolist() for x in b])
        # self.analysis["masked"].append(mask_noise.tolist())

        ents = distributions.Categorical(logits=lprobs).entropy().view(b1, b2)
        ents = ents * (~padded_toks)
        # ent_noisy = (ents * mask_noise).sum(-1) / mask_noise.sum(-1)
        # ent_orig = (ents * mask_orig).sum(-1) / mask_orig.sum(-1)
        ent_noisy = (ents * is_noisy).sum() / is_noisy.sum()
        ent_orig = (ents * is_real).sum() / is_real.sum()

        self.analysis["entropy"].append(ents.mean().item())
        # self.analysis["entropy_per_tok"].append((ents * ~padded_toks).tolist())
        self.analysis["entropy_noisy"].append(ent_noisy.item())
        self.analysis["entropy_original"].append(ent_orig.item())

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2),
                           sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2),
                               ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(
                meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(
                meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
