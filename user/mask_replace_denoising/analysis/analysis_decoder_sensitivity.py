import json
import logging
import math
import os
from collections import defaultdict
from typing import Optional

import numpy
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from torch.distributions import Categorical

logger = logging.getLogger(__name__)


@register_criterion('analysis_decoder_sensitivity')
class AnalysisDecoderSensitivityCriterion(FairseqCriterion):
    """
    This objective combines replaced token detection and reconstruction.
    """

    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.noise_prob = task.args.noise_prob
        self.analysis = defaultdict(list)

        self.sampling_temperature = task.args.mask_replace_sampling_temperature
        self.sampling_topp = task.args.mask_replace_sampling_topp
        self.sampling_topk = task.args.mask_replace_sampling_topk
        try:
            self.vocab = self.task.src_dict
        except:
            self.vocab = self.task.source_dictionary

    def save_analysis(self, save_dir, log_outputs):

        fname = os.path.join(
            save_dir, f"analysis_decoder_sensitivity_p={self.noise_prob}.json")
        open(fname, 'w').close()

        for k, v in self.analysis.items():
            self.analysis[k] = numpy.mean(self.analysis[k])

        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(self.analysis, f, ensure_ascii=False, indent=4)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--noise-prob', type=float, required=True,
                            help='choose between: entropy, states')

    def _detok(self, tensor):
        return [self.vocab.symbols[t] for t in tensor]

    def _debug_batch(self, seqs):
        return [numpy.array(list([[self.vocab[x] for x in s]
                                  for s in zip(*r)])).T
                for r in zip(*seqs)]

    def _debug_sample(self, seqs):
        return numpy.array(list([[self.vocab[x] for x in s]
                                 for s in zip(*seqs)])).T

    def mix_states(self, tokens, outputs, prob, positions=None):

        # 1. sample the position's which will be moved to other sentences
        if positions is None:
            positions = torch.bernoulli(
                tokens.clone().float().fill_(prob)).bool()

            # do not update special lang_id tokens
            special_pos = tokens < (len(self.vocab) - len(self.task.langs))

            # do not update other special tokens (bos, unk, ...)
            special_pos = special_pos & (tokens >= self.vocab.nspecial)

            positions = positions & special_pos

        outs = outputs.transpose(0, 1)

        # encoder_out is (length x batch x features), so we have to transpose
        positions = positions.unsqueeze(2).expand_as(outs)

        out_zeroed = outs * (~positions)

        # 2. select the outputs and roll them across the batch (0th) dim
        out_rolled = torch.roll(outs, -1, 0) * positions

        mixed = out_zeroed + out_rolled

        return mixed.transpose(0, 1)

    def mix_tokens(self, tokens, prob):
        # debug purposes
        # (torch.roll(~index_mask.bool(), -1, 0) * src_tokens + torch.roll(
        #     index_mask * src_tokens, -1, 0)).int().tolist()
        tokens = tokens[:5, :]
        # 1. sample the position's which will be moved to other sentences
        positions = torch.bernoulli(tokens.clone().float().fill_(prob)).bool()

        # 1. select the representations and roll them across the second (batch)
        # dimension: outputs shape is (length x batch x features)
        mix_states = torch.roll(positions * tokens, -1, 0)

        # 2. remove the states that update special tokens
        # do not update special lang_id tokens
        special_mask = tokens < (len(self.vocab) - len(self.task.langs))
        # do not update other special tokens (bos, unk, ...)
        special_mask = special_mask & (tokens >= self.vocab.nspecial)
        # apply filter mask
        mix_states = mix_states * special_mask

        # return the mixed outputs, by zeroing out the target positions
        # and then adding the selected states
        mixed = (mix_states.eq(0) * tokens) + mix_states

        return mixed

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

    def get_masked(self, tokens):
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
        return masked_tokens

    def mask_replace(self, model, tokens, lengths):
        masked_tokens = self.get_masked(tokens)
        logits, _ = model.generator(tokens, lengths,
                                    masked_tokens=masked_tokens)

        # replace the masked tokens in with samples from generator
        logits = logits / self.sampling_temperature
        logits = self.top_k_top_p_filtering(logits,
                                            self.sampling_topk,
                                            self.sampling_topp)
        samples = Categorical(logits=logits).sample()
        replaced_tokens = tokens.masked_scatter(masked_tokens, samples)

        return replaced_tokens

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
            noise,
            noise_method,
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

        if noise is not None:

            if noise_method == "mix-random":
                assert isinstance(noise, (float, int))
                # self.mix_tokens(src_tokens.clone(), noise_prob)
                noisy_outputs = self.mix_states(src_tokens,
                                                encoder_out.encoder_out,
                                                noise)
            elif noise_method == "mask-random":
                assert isinstance(noise, (float, int))
                noisy_outputs = F.dropout2d(encoder_out.encoder_out, noise)

            elif noise_method == "mask-noisy":
                assert isinstance(noise, torch.Tensor)
                mask = noise.transpose(0, 1).unsqueeze(2)
                noisy_outputs = encoder_out.encoder_out.clone() * mask

            elif noise_method == "mix-noisy":
                assert isinstance(noise, torch.Tensor)
                noisy_outputs = self.mix_states(src_tokens,
                                                encoder_out.encoder_out,
                                                0, positions=noise)
            else:
                raise ValueError
            encoder_out = encoder_out._replace(encoder_out=noisy_outputs)

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
        noisy_input = sample['net_input']['src_tokens'].clone()
        clean_input = sample['target'].clone()

        # ----------------------------------------------------------------
        # Clean Input
        # ----------------------------------------------------------------
        # loss over clean input
        encoder_out, decoder_out = self.translator(
            model, None, None,
            clean_input,
            sample['net_input']['src_lengths'],
            sample['net_input']['prev_output_tokens'])
        loss = self.compute_loss(model, decoder_out, sample)[0]

        # mixed loss
        encoder_out, decoder_out = self.translator(
            model, self.noise_prob, "mix-random",
            clean_input,
            sample['net_input']['src_lengths'],
            sample['net_input']['prev_output_tokens'])
        loss_mixed = self.compute_loss(model, decoder_out, sample)[0]

        # masked loss
        encoder_out, decoder_out = self.translator(
            model, self.noise_prob, "mask-random",
            clean_input,
            sample['net_input']['src_lengths'],
            sample['net_input']['prev_output_tokens'])
        loss_masked = self.compute_loss(model, decoder_out, sample)[0]

        # ----------------------------------------------------------------
        # Noisy Input
        # ----------------------------------------------------------------
        # 1. Replace masked tokens and compute MLM loss
        if self.task.args.mask_replace and not self.task.args.test_multitask:
            noisy_input = self.mask_replace(model,
                                            sample['net_input']['src_tokens'],
                                            sample['net_input']['src_lengths'])

        # 2. Permute tokens. It has to be done after the replacements
        if 'permutations' in sample['net_input']:
            noisy_input = self.apply_permutation(
                noisy_input, sample['net_input']['permutations'])

        # 3. Reconstruction
        encoder_out, decoder_out = self.translator(
            model, None, None,
            noisy_input,
            sample['net_input']['src_lengths'],
            sample['net_input']['prev_output_tokens'])
        loss_noisy = self.compute_loss(model, decoder_out, sample)[0]

        encoder_out, decoder_out = self.translator(
            model, clean_input.eq(noisy_input), "mask-noisy",
            noisy_input,
            sample['net_input']['src_lengths'],
            sample['net_input']['prev_output_tokens'])
        loss_noisy_masked = self.compute_loss(model, decoder_out, sample)[0]

        encoder_out, decoder_out = self.translator(
            model, ~clean_input.eq(noisy_input), "mix-noisy",
            noisy_input,
            sample['net_input']['src_lengths'],
            sample['net_input']['prev_output_tokens'])
        loss_noisy_mixed = self.compute_loss(model, decoder_out, sample)[0]

        # compute sample sizes
        sample_size = sample['target'].size(0) if self.sentence_avg else sample[
            'ntokens']

        is_clean = clean_input.eq(noisy_input)
        noise_ratio = (~is_clean).sum().item() / (is_clean >= 0).sum().item()

        logging_output = {
            'loss': loss.data,
            'loss_mixed': loss_mixed.data,
            'loss_masked': loss_masked.data,

            'loss_noisy': loss_noisy.data,
            'loss_noisy_masked': loss_noisy_masked.data,
            'loss_noisy_mixed': loss_noisy_mixed.data,

            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
            'noise_ratio': noise_ratio,
        }

        self.analysis["loss"].append(loss.item())
        self.analysis["loss_mixed"].append(loss_mixed.item())
        self.analysis["loss_masked"].append(loss_masked.item())

        self.analysis["loss_noisy"].append(loss_noisy.item())
        self.analysis["loss_noisy_masked"].append(loss_noisy_masked.item())
        self.analysis["loss_noisy_mixed"].append(loss_noisy_mixed.item())
        self.analysis["noise_ratio"].append(noise_ratio)

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        loss_mixed_sum = sum(
            log.get('loss_mixed', 0) for log in logging_outputs)
        loss_masked_sum = sum(
            log.get('loss_masked', 0) for log in logging_outputs)

        loss_noisy_sum = sum(
            log.get('loss_noisy', 0) for log in logging_outputs)
        loss_noisy_masked_sum = sum(
            log.get('loss_noisy_masked', 0) for log in logging_outputs)

        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2),
                           sample_size, round=3)
        metrics.log_scalar('loss_mixed',
                           loss_mixed_sum / sample_size / math.log(2),
                           sample_size, round=3)
        metrics.log_scalar('loss_masked',
                           loss_masked_sum / sample_size / math.log(2),
                           sample_size, round=3)

        metrics.log_scalar('loss_noisy',
                           loss_noisy_sum / sample_size / math.log(2),
                           sample_size, round=3)
        metrics.log_scalar('loss_noisy_masked',
                           loss_noisy_masked_sum / sample_size / math.log(2),
                           sample_size, round=3)

        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2),
                               ntokens, round=3)
            metrics.log_scalar('nll_loss_mixed',
                               loss_mixed_sum / ntokens / math.log(2), ntokens,
                               round=3)
            metrics.log_scalar('nll_loss_masked',
                               loss_masked_sum / ntokens / math.log(2), ntokens,
                               round=3)

            metrics.log_scalar('nll_loss_noisy',
                               loss_noisy_sum / ntokens / math.log(2),
                               ntokens, round=3)
            metrics.log_scalar('nll_loss_noisy_masked',
                               loss_noisy_masked_sum / ntokens / math.log(2),
                               ntokens,
                               round=3)

            metrics.log_derived('ppl',
                                lambda meters: utils.get_perplexity(
                                    meters['nll_loss'].avg))
            metrics.log_derived('ppl_mixed',
                                lambda meters: utils.get_perplexity(
                                    meters['nll_loss_mixed'].avg))
            metrics.log_derived('ppl_masked',
                                lambda meters: utils.get_perplexity(
                                    meters['nll_loss_masked'].avg))

            metrics.log_derived('ppl_noisy',
                                lambda meters: utils.get_perplexity(
                                    meters['nll_loss_noisy'].avg))
            metrics.log_derived('ppl_noisy_masked',
                                lambda meters: utils.get_perplexity(
                                    meters['nll_loss_noisy_masked'].avg))

        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(
                meters['loss'].avg))
            metrics.log_derived('ppl_mixed',
                                lambda meters: utils.get_perplexity(
                                    meters['loss_mixed'].avg))
            metrics.log_derived('ppl_masked',
                                lambda meters: utils.get_perplexity(
                                    meters['loss_masked'].avg))

            metrics.log_derived('ppl_noisy',
                                lambda meters: utils.get_perplexity(
                                    meters['loss_noisy'].avg))
            metrics.log_derived('ppl_noisy_masked',
                                lambda meters: utils.get_perplexity(
                                    meters['loss_noisy_masked'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
