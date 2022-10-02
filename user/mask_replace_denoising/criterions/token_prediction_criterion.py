import json
import logging
import math
import os
from collections import defaultdict

import numpy
import torch
import torch.nn.functional as F
from fairseq import metrics, utils, modules
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion
from torch.distributions import Categorical

logger = logging.getLogger(__name__)


@register_criterion('token_prediction')
class TokenPredictionCriterion(FairseqCriterion):
    """
    This objective combines replaced token detection and reconstruction.
    """

    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.criterion = CrossEntropyCriterion.build_criterion(task.args, task)
        self.train_noise = task.args.train_with_noise

        self.sampling_temperature = task.args.mask_replace_sampling_temperature
        self.sampling_topp = task.args.mask_replace_sampling_topp
        self.sampling_topk = task.args.mask_replace_sampling_topk

        self.analysis = defaultdict(list)

        try:
            self.vocab = self.task.src_dict
        except:
            self.vocab = self.task.source_dictionary

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--token-prediction-task',
                            type=str, metavar='STR', default='classification',
                            help='choose between regression and classification.')
        parser.add_argument('--train-with-noise',
                            default=False, action='store_true',
                            help='train on both real and fake tokens')

        parser.add_argument('--test-multitask',
                            action='store_true', default=False,
                            help='Hack that does NOT replace the encoder\'s '
                                 'input to test the performance of multitasking.')

    def save_analysis(self, save_dir, log_outputs):

        fname = os.path.join(save_dir, f"token_prediction.json")
        open(fname, 'w').close()

        results = {
            "accuracy": numpy.mean(
                [lo['correct'] / lo['sample_size'] for lo in log_outputs]),
            "accuracy_fake": numpy.mean(
                [lo['correct_fake'] / lo['sample_size_fake']
                 for lo in log_outputs]),
            "accuracy_real": numpy.mean(
                [lo['correct_real'] / lo['sample_size_real']
                 for lo in log_outputs]),

            "loss": numpy.mean(
                [lo['loss'] / lo['sample_size'] for lo in log_outputs]),
            "loss_fake": numpy.mean(
                [lo['loss_fake'] / lo['sample_size_fake']
                 for lo in log_outputs]),
            "loss_real": numpy.mean(
                [lo['loss_real'] / lo['sample_size_real']
                 for lo in log_outputs]),
        }

        results["ppl"] = math.exp(results["loss"])
        results["ppl_real"] = math.exp(results["loss_real"])
        results["ppl_fake"] = math.exp(results["loss_fake"])

        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    def _detok(self, tensor):
        return [self.task.dictionary.symbols[t] for t in tensor]

    def _debug_batch(self, seqs):
        return [numpy.array(list([[self.task.dictionary[x] for x in s]
                                  for s in zip(*r)])).T
                for r in zip(*seqs)]

    def _debug_sample(self, seqs):
        return numpy.array(list([[self.task.dictionary[x] for x in s]
                                 for s in zip(*seqs)])).T

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

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if self.train_noise or not self.training:
            enc_tokens = sample['net_input']['src_tokens']

            if self.task.args.mask_replace:
                enc_tokens = self.mask_replace(
                    model,
                    sample['net_input']['src_tokens'],
                    sample['net_input']['src_lengths'])

            if 'permutations' in sample['net_input']:
                enc_tokens = self.apply_permutation(
                    enc_tokens, sample['net_input']['permutations'])
        else:
            enc_tokens = sample['target']

        #     todo: remove hack
        if self.task.args.test_multitask:
            enc_tokens = sample['net_input']['src_tokens']

        # self._debug_batch([sample['net_input']['src_tokens'], enc_tokens, sample['target']])
        # x = model.classifier.weight[10,:10].tolist()
        # y = model.encoder.embed_tokens.weight[10,:10].tolist()

        encoder_out = model.encoder(
            enc_tokens,
            src_lengths=sample['net_input']['src_lengths'],
            return_all_hiddens=True,
        )
        logits = model.classifier(encoder_out.encoder_out.transpose(0, 1))

        # calculate MLM loss
        # noisy_tokens = self.get_masked(sample['net_input']['src_tokens'])

        # as noisy are considered the tokens that are different
        # from the targets (original input) after the addition of noise
        noisy_tokens = ~enc_tokens.eq(sample['target'])
        logits = logits.view(-1, logits.size(-1))
        targets = sample['target'].view(-1)
        loss = modules.cross_entropy(logits, targets, self.padding_idx, 'none')
        loss_fake = loss[noisy_tokens.view(-1)]  # loss over noisy tokens
        loss_real = loss[~noisy_tokens.view(-1)]  # loss over original tokens
        loss_total = loss_real.sum() + loss_fake.sum()

        # label whether the each prediction is correct
        is_correct = targets.eq(logits.max(-1)[1])
        # number of correct predictions for NOISY tokens
        is_correct_fake = is_correct[noisy_tokens.view(-1)].int().sum().item()
        sample_size = targets.size(0)

        logging_output = {
            'loss': loss_total.item(),
            'loss_fake': loss_fake.sum().item(),
            'loss_real': loss_real.sum().item(),
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
            'sample_size_fake': len(loss_fake),
            'sample_size_real': sample_size - len(loss_fake),
            'correct': is_correct.sum().item(),
            'correct_fake': is_correct_fake,
            'correct_real': is_correct.sum().item() - is_correct_fake,
        }

        return loss_total, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        loss_fake_sum = sum(log.get('loss_fake', 0) for log in logging_outputs)
        loss_real_sum = sum(log.get('loss_real', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        sample_size_real = sum(log.get('sample_size_real', 0)
                               for log in logging_outputs)
        sample_size_fake = sum(log.get('sample_size_fake', 0)
                               for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2),
                           sample_size, round=3)
        metrics.log_scalar('loss_fake',
                           loss_fake_sum / sample_size_fake / math.log(2),
                           sample_size_fake, round=3)
        metrics.log_scalar('loss_real',
                           loss_real_sum / sample_size_real / math.log(2),
                           sample_size_real, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(
            meters['loss'].avg))
        metrics.log_derived('ppl_fake', lambda meters: utils.get_perplexity(
            meters['loss_fake'].avg))
        metrics.log_derived('ppl_real', lambda meters: utils.get_perplexity(
            meters['loss_real'].avg))

        #
        correct_sum = sum(log.get('correct', 0) for log in logging_outputs)
        correct_fake_sum = sum(log.get('correct_fake', 0)
                               for log in logging_outputs)
        correct_real_sum = sum(log.get('correct_real', 0)
                               for log in logging_outputs)

        metrics.log_scalar('accuracy', 100.0 * correct_sum / sample_size,
                           sample_size, round=3)
        metrics.log_scalar('accuracy_fake',
                           100.0 * correct_fake_sum / sample_size_fake,
                           sample_size, round=3)
        metrics.log_scalar('accuracy_real',
                           100.0 * correct_real_sum / sample_size_real,
                           sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
