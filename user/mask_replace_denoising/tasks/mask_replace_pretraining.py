import logging
import os
from collections import defaultdict

import numpy as np
from fairseq.data import (
    data_utils,
    AppendTokenDataset,
    ConcatDataset,
    PrependTokenDataset,
    ResamplingDataset,
    SortDataset,
    TokenBlockDataset,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_denoising import MultilingualDenoisingTask

from user.mask_replace_denoising.datasets.mask_replace_dataset import \
    MaskReplaceDenoisingDataset

logger = logging.getLogger(__name__)


@register_task('marss_pretraining')
class Rtd2SeqTask(MultilingualDenoisingTask):

    @staticmethod
    def add_args(parser):
        MultilingualDenoisingTask.add_args(parser)

        parser.add_argument('--mask-replace',
                            action='store_true',
                            help="replace masked tokens with samples from a "
                                 "generator (MLM).")

        parser.add_argument('--mask-replace-sampling-temperature', type=float,
                            default=1.0,
                            help='temperature for sampling')

        parser.add_argument('--mask-replace-sampling-topk', type=int,
                            default=-1,
                            help='sample from top K likely next words '
                                 'instead of all words. If k < 0 then sample '
                                 'from full distribution.')

        parser.add_argument('--mask-replace-sampling-topp', type=float,
                            default=1.0,
                            help='sample from the smallest set whose cumulative'
                                 ' probability mass exceeds p for next words. '
                                 'If top_p < 1.0, keep the top tokens with '
                                 'cumulative probability >= top_p '
                                 '(nucleus filtering)')

        parser.add_argument('--word-shuffle', type=int, default=0,
                            help="Shuffle words by no more than k positions")

        parser.add_argument('--mask-decoder', default=0.0, type=float,
                            help='fraction of words/subwords in the deocoder\'s '
                                 'input that will be masked')

        parser.add_argument('--mask-decoder-paired',
                            action='store_true', default=False,
                            help='mask the same words in the decoder as in the'
                                 'encoder')

        parser.add_argument('--mask-replace-decoder',
                            action='store_true',
                            help="replace masked tokens in the decoder"
                                 "with samples from a generator (MLM).")

        parser.add_argument('--train-sentence-level',
                            type=int, default=0,
                            help='Every N steps, train with sentence-level '
                                 'reconstruction (used only during training)')

        parser.add_argument('--xreplace-prob',
                            action='store_true', default=False,
                            help='Replace the masked token with a token in '
                                 'another language.')

        parser.add_argument('--xreplace-prob-threshold',
                            default=0.0001, type=float, help='.')

        parser.add_argument('--xreplace-ema',
                            action='store_true', default=False,
                            help='Replace the masked token with a token in '
                                 'another language, by using the '
                                 'exponential-moving-average of the '
                                 'mean language representations.')

        parser.add_argument('--xreplace-ema-rates', default='(0.9, 0.999)',
                            help='smoothing parameter for tracking the '
                                 'exponential moving average of each'
                                 'languages representations')

        parser.add_argument('--xreplace-ema-warmup', type=int, default=8000,
                            help='number of steps of linear warmup'
                                 'for --xreplace-ema-rate')

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

        paths = args.data.split(':')
        languages = args.langs.split(',')

        assert not (args.xreplace_prob and not args.mask_replace)
        assert not (args.xreplace_ema and not args.mask_replace)

        if args.xreplace_prob and args.xreplace_ema:
            logger.warning("You are using both "
                           "--xreplace-ema and --xreplace-prob")

        if args.mask_decoder_paired:
            logger.info("You are using --mask-decoder-paired, which means"
                        "that the number of the value of "
                        "--mask-decoder or --mask-replace-decoder will be "
                        "overridden by the corresponding % of corrupted "
                        "tokens in the encoder!")

        if args.train_sentence_level > 0 and not args.sentence_avg:
            logger.info("Consider combining --train-sentence-level "
                        "with --sentence-avg.")

        if (args.sentence_avg
                and args.mask_replace
                and args.mlm_loss_coefficient <= 1):
            logger.info("You are using --sentence-avg and you should "
                        "consider increasing/adjusting the value of"
                        "--mlm-loss-coefficient")

        # boolean flag
        self.xreplace_ema = args.xreplace_ema
        # tuple with the range for annealing the EMA rates
        self.xreplace_ema_rates = eval(self.args.xreplace_ema_rates)
        # the current EMA rate of the model
        self.xreplace_ema_rate = self.xreplace_ema_rates[0]
        # the number of warmup (annealing) steps for the EMA rate
        self.xreplace_ema_warmup = self.args.xreplace_ema_warmup

        # --------------------------------------------------------------
        # Cross-lingual replacement using token probabilities
        # --------------------------------------------------------------
        self.xreplace_prob = args.xreplace_prob
        if self.xreplace_prob:
            self.token_probs = defaultdict(dict)
            for lang in languages:
                # language token probabilities
                for _lang in languages:
                    if _lang == lang:
                        self.token_probs[lang][f"[{_lang}]"] = 1.0
                    else:
                        self.token_probs[lang][f"[{_lang}]"] = 0.0

                # prevent generation of the <mask> token for all languages
                self.token_probs[lang][dictionary.symbols[self.mask_idx]] = 0.0
                self.token_probs[lang][dictionary.pad_word] = 0.0

                # normal token probabilities
                lines = open(os.path.join(paths[0], lang, 'tok_probs.txt'))
                for i, line in enumerate(lines):
                    token, prob = line.strip().split()
                    if i < dictionary.nspecial - 1:
                        prob = 1.0
                    self.token_probs[lang][token] = float(prob)

    # def build_model(self, args):
    #     model = super().build_model(args)
    #
    #     # we imitate a FairseqMultiModel
    #     model.max_positions = self.max_positions
    #     return model

    def train_step(self, sample, model, criterion, optimizer, update_num,
                   ignore_grad=False):

        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        criterion.steps = update_num

        # lineary anneal the decay factor for the EMA
        if self.xreplace_ema:
            delta = self.xreplace_ema_rates[1] - self.xreplace_ema_rates[0]
            factor = min(1, update_num / self.xreplace_ema_warmup)
            self.xreplace_ema_rate = self.xreplace_ema_rates[0] + delta * factor

            # update the ema rate in the generator
            # model.generator.xreplace_ema_rate = self.xreplace_ema_rate

        loss, sample_size, logging_output = super().train_step(sample, model,
                                                               criterion,
                                                               optimizer,
                                                               update_num,
                                                               ignore_grad)
        return loss, sample_size, logging_output

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """
        It is basically a copy of the load_dataset() from
        `multilingual-denoising`. We just use our own CustomDenoisingDataset
        instead of fairseq's DenoisingDataset.
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        if self.langs is None:
            languages = sorted([
                name for name in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, name))
            ])
        else:
            languages = self.langs.split(',')
            for name in languages:
                assert os.path.exists(os.path.join(data_path,
                                                   name)), "all the languages must exist"

        logger.info("| Training on {0} languages: {1}".format(len(languages),
                                                              languages))
        logger.info("| Language to id mapping: ", {
            lang: id for id, lang in enumerate(languages)
        }
                    )

        mask_whole_words = get_whole_word_mask(self.args, self.dictionary)
        language_without_segmentations = self.args.no_whole_word_mask_langs.split(
            ',')
        lang_datasets = []
        for language in languages:
            split_path = os.path.join(data_path, language, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            if dataset is None:
                raise FileNotFoundError(
                    'Dataset not found: {} ({})'.format(split, split_path))

            end_token = self.source_dictionary.index('[{}]'.format(language)) \
                if self.args.add_lang_token else self.source_dictionary.eos()

            # create continuous blocks of tokens
            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                self.args.tokens_per_sample - 2,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=end_token,
                break_mode=self.args.sample_break_mode,
            )
            logger.info(
                '| loaded {} blocks from: {}'.format(len(dataset), split_path))

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())
            dataset = AppendTokenDataset(dataset, end_token)

            lang_mask_whole_words = mask_whole_words if language not in language_without_segmentations else None
            lang_dataset = MaskReplaceDenoisingDataset(
                dataset,
                dataset.sizes,
                self.dictionary,
                self.mask_idx,
                lang_mask_whole_words,
                shuffle=self.args.shuffle_instance,
                seed=self.seed,
                args=self.args,
                eos=None if not self.args.add_lang_token else self.source_dictionary.index(
                    '[{}]'.format(language)),
            )
            lang_datasets.append(lang_dataset)

        dataset_lengths = np.array(
            [len(d) for d in lang_datasets],
            dtype=float,
        )
        logger.info(
            '| loaded total {} blocks for all languages'.format(
                dataset_lengths.sum(),
            )
        )
        if split == self.args.train_subset:
            # For train subset, additionally up or down sample languages.
            sample_probs = self._get_sample_prob(dataset_lengths)
            logger.info("| Sample probability by language: ", {
                lang: "{0:.4f}".format(sample_probs[id])
                for id, lang in enumerate(languages)
            }
                        )
            size_ratio = (
                                 sample_probs * dataset_lengths.sum()) / dataset_lengths
            logger.info("| Up/Down Sampling ratio by language: ", {
                lang: "{0:.2f}".format(size_ratio[id])
                for id, lang in enumerate(languages)
            }
                        )

            resampled_lang_datasets = [
                ResamplingDataset(
                    lang_datasets[i],
                    size_ratio=size_ratio[i],
                    seed=self.args.seed,
                    epoch=epoch,
                    replace=size_ratio[i] >= 1.0,
                )
                for i, d in enumerate(lang_datasets)
            ]
            dataset = ConcatDataset(
                resampled_lang_datasets,
            )
        else:
            dataset = ConcatDataset(lang_datasets)
            lang_splits = [split]
            for lang_id, lang_dataset in enumerate(lang_datasets):
                split_name = split + '_' + languages[lang_id]
                lang_splits.append(split_name)
                self.datasets[split_name] = lang_dataset

            if split in self.args.valid_subset:
                self.args.valid_subset = self.args.valid_subset.replace(
                    split, ','.join(lang_splits)
                )

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(dataset))

        self.datasets[split] = SortDataset(
            dataset,
            sort_order=[
                shuffle,
                dataset.sizes,
            ],
        )
