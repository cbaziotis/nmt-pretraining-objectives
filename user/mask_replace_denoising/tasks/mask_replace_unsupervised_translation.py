import html
import itertools
import json
import logging
import math
import os
import pprint
from argparse import Namespace
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from fairseq import metrics
from fairseq import options
from fairseq import utils
from fairseq.data import (
    data_utils,
    indexed_dataset, LanguagePairDataset, BacktranslationDataset,
    RoundRobinZipDatasets, Dictionary, PrependTokenDataset,
    AppendTokenDataset, )
from fairseq.data import (
    encoders,
)
from fairseq.sequence_generator import SequenceGenerator, EnsembleModel
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.semisupervised_translation import _get_bt_dataset_key
from fairseq.tasks.translation_from_pretrained_bart import \
    TranslationFromPretrainedBARTTask

logger = logging.getLogger(__name__)
EVAL_BLEU_ORDER = 4


class BannedTokensEnsembleModel(EnsembleModel):
    """
    A wrapper around  EnsembleModel, which will be used for masking the
    log probabilities of certain (low-frequency) tokens

    """

    def __init__(self, models, mask: list):
        super().__init__(models)
        self.mask = mask

    @torch.jit.export
    def forward_decoder(self, *args):
        avg_lprobs, avg_attn = super().forward_decoder(*args)

        if self.mask is not None:
            mask = torch.tensor(self.mask,
                                dtype=torch.bool,
                                device=avg_lprobs.device)
            avg_lprobs.masked_fill_(mask, -math.inf)

        return avg_lprobs, avg_attn


@register_task('unsupervised_translation_from_pretrained')
class UnsupervisedTranslationFromPretrained(FairseqTask):
    """
    Adapted from SemisupervisedTranslationTask

    Use tokenize_for_xlm.sh and preprocess_for_semisupervised.sh

    """

    def _split_exists(self, data_path, split, src, tgt, lang):
        if src is not None:
            filename = os.path.join(data_path,
                                    '{}.{}-{}.{}'.format(split, src, tgt, lang))
        else:
            filename = os.path.join(data_path,
                                    '{}.{}-None.{}'.format(split, src, tgt))
        return indexed_dataset.dataset_exists(filename,
                                              impl=self.args.dataset_impl)

    def _load_indexed_dataset(self, path, dictionary):
        return data_utils.load_indexed_dataset(path, dictionary,
                                               self.args.dataset_impl)

    def _debug_batch(self, sample, unbpe=False):
        return [["".join([self.dictionary.symbols[t] for t in x])
                     .replace('<pad>', '').replace('▁', ' ') for x in d]
                for d in zip(sample['net_input']['src_tokens'],
                             sample['net_input']['prev_output_tokens'],
                             sample['target'])]

    def _log_batch(self, sample):
        path = self.args.save_dir

        for dataset, batch in sample.items():
            pairs = self._debug_batch(batch)

            with open(os.path.join(path, f"{dataset}.html"), "w") as f:
                for src, tgt_in, tgt_out in pairs:
                    f.write("<p>")
                    f.write(f"<span>{html.escape(src)}</span>")
                    f.write(f"</br>")
                    # f.write(f"<span>{html.escape(tgt_in)}</span>")
                    # f.write(f"</br>")
                    f.write(f"<span>{html.escape(tgt_out)}</span>")
                    f.write("</p>")

    def _language_pair_dataset(self, pair, src_datasets, tgt_datasets):
        """
        It creates a parallel dataset, used for validation, as follows:

        src_tokens:
        <s> ▁Wir ▁haben ▁die ▁Diskussion ▁in ▁Deutschland ▁befriedet.</s>[de]

        rev_output_tokens:
        [en]<s> ▁We ▁have ▁dealt ▁with ▁the ▁discussions ▁in ▁Germany.</s>

        target:
        <s> ▁We ▁have ▁dealt ▁with ▁the ▁discussions ▁in ▁Germany.</s>[en]

        Args:
            pair:
            src_datasets:
            tgt_datasets:

        Returns:

        """
        src, tgt = pair.split('-')
        src_dataset, tgt_dataset = src_datasets[pair], tgt_datasets[pair]

        src_dataset = PrependTokenDataset(src_dataset,
                                          self.dictionary.bos())
        src_dataset = AppendTokenDataset(src_dataset,
                                         self.lang_tok_idx[src])

        tgt_dataset = PrependTokenDataset(tgt_dataset,
                                          self.dictionary.bos())
        tgt_dataset = AppendTokenDataset(tgt_dataset,
                                         self.lang_tok_idx[tgt])

        _dataset = LanguagePairDataset(
            src_dataset, src_dataset.sizes, self.dicts[src],
            tgt_dataset, tgt_dataset.sizes, self.dicts[tgt],
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            remove_eos_from_source=False,
            append_eos_to_target=False,
            append_bos=False,
            eos=self.lang_tok_idx[tgt]
        )
        return _dataset
        # return TransformEosLangPairDataset(
        #     ,
        #     src_eos=src_eos,
        #     new_src_eos=new_src_eos,
        #     tgt_bos=tgt_eos,
        #     new_tgt_bos=new_tgt_bos,
        # )

    @staticmethod
    def add_args(parser):
        TranslationFromPretrainedBARTTask.add_args(parser)

        parser.add_argument('--bt-output-filtering-prob',
                            default=.01, type=float, metavar='N',
                            help='during the first steps of on-the-fly'
                                 'backtranslation, we mask the rare tokens of '
                                 'each language, to avoid copying the source.'
                                 'this argument specifies the probability '
                                 'threshold for masking tokens.')
        parser.add_argument('--bt-output-filtering-steps',
                            default=1000, type=int, metavar='N',
                            help='during the first steps of on-the-fly'
                                 'backtranslation, we mask the rare tokens of '
                                 'each language, to avoid copying the source.'
                                 'this argument specifies the number of steps  '
                                 'to use this masking.')
        parser.add_argument('--bt-max-len-a', default=1.1, type=float,
                            metavar='N',
                            help='generate back-translated sequences of '
                                 'maximum length ax + b, where x is the '
                                 'source length')
        parser.add_argument('--bt-max-len-b', default=10.0, type=float,
                            metavar='N',
                            help='generate back-translated sequences of '
                                 'maximum length ax + b, where x is the '
                                 'source length')
        parser.add_argument('--bt-beam-size', default=1, type=int, metavar='N',
                            help='beam size used in beam search of '
                                 'online back-translation')
        parser.add_argument('--bt-no-repeat-ngram-size', default=0, type=int,
                            metavar='N',
                            help='beam size used in beam search of '
                                 'online back-translation')

    def __init__(self, args, dictionary, languages, tok_probs):
        super().__init__(args)
        self.dictionary = dictionary
        self.tok_probs = tok_probs
        self.seed = args.seed
        self.args = args
        self.langs = languages

        # IMPORTANT: we assume a joint dictionary
        self.dicts = {lang: dictionary for lang in languages}

        # add mask token, which was used during pretraining
        self.mask_idx = self.dictionary.add_symbol('<mask>')

        # todo: add logic to change this for inference
        self.training = True

        # comma-separated list of language pairs (training order): en-de,en-fr
        self.lang_pairs = ["-".join(p) for p
                           in itertools.permutations(self.langs, 2)]

        self.lang_tok_idx = {lang: dictionary.index('[{}]'.format(lang))
                             for lang in languages}

        # True=mask, False=permit
        self.lang_masks = {}
        cutoff = self.args.bt_output_filtering_prob
        for lang in languages:
            lang_sym = '[{}]'.format(lang)
            probs = self.tok_probs[lang]
            tmask = [probs.get(s, 0) < cutoff and i > 4 and s != lang_sym
                     for i, s in enumerate(self.dictionary.symbols)]

            self.lang_masks[lang] = tmask
            # [print((s, probs.get(s, 0), m)) for s, m in
            #  zip(self.dictionary.symbols[:100], tmask[:100])]
            nmasked = sum(tmask)
            nallowed = len(self.dictionary.symbols) - nmasked
            logger.info(f"| BT masking [{lang}]: keeping {nallowed} "
                        f"out of {len(self.dictionary.symbols)}")

        self.backtranslate_datasets = {}
        self.backtranslators = {}

    @classmethod
    def setup_task(cls, args, **kwargs):
        """
        Setup the task (e.g., load dictionaries).
        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        paths = args.data.split(':')
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], 'dict.txt'))

        data_path = paths[0]
        if args.langs is None:
            languages = sorted([
                name for name in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, name))
            ])
        else:
            languages = args.langs.split(',')

        # token count per language
        tok_probs = defaultdict(dict)
        for lang in languages:
            dictionary.add_symbol('[{}]'.format(lang))

            for x in open(os.path.join(paths[0], 'tok_probs.{}'.format(lang))):
                _w, _c = x.strip().split()
                tok_probs[lang][_w] = float(_c)

        logger.info("| dictionary: {} types".format(len(dictionary)))

        return cls(args, dictionary, languages, tok_probs)

    def load_dataset(self, split, epoch=1, **kwargs):
        """Load a dataset split."""
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # load parallel datasets - used only for validation
        src_datasets, tgt_datasets = {}, {}
        if not split.startswith("train"):
            for lang_pair in self.lang_pairs:
                src, tgt = lang_pair.split('-')
                if self._split_exists(data_path, split, src, tgt, src):
                    prefix = os.path.join(data_path,
                                          '{}.{}-{}.'.format(split, src, tgt))
                elif self._split_exists(data_path, split, tgt, src, src):
                    prefix = os.path.join(data_path,
                                          '{}.{}-{}.'.format(split, tgt, src))
                else:
                    continue
                src_datasets[lang_pair] = self._load_indexed_dataset(
                    prefix + src, self.dicts[src])
                tgt_datasets[lang_pair] = self._load_indexed_dataset(
                    prefix + tgt, self.dicts[tgt])
                logger.info('parallel-{} {} {} examples'.format(
                    data_path, split, len(src_datasets[lang_pair])))
            if len(src_datasets) == 0:
                raise FileNotFoundError(
                    'Dataset not found: {} ({})'.format(split, data_path))

        # back translation datasets
        backtranslate_datasets = {}
        if split.startswith("train"):
            for lang_pair in self.lang_pairs:

                # eg. en-de
                src, tgt = lang_pair.split('-')
                if not self._split_exists(data_path, split, tgt, None, tgt):
                    raise FileNotFoundError(
                        'Dataset not found: backtranslation {} ({})'.format(
                            split, data_path))

                # eg. train.de-None.de
                filename = os.path.join(data_path,
                                        '{}.{}-None.{}'.format(split, tgt, tgt))

                # monolingual (LM-like) dataset with the target data
                dataset = self._load_indexed_dataset(filename, self.dicts[tgt])

                dataset = PrependTokenDataset(dataset,
                                              self.dictionary.bos())
                dataset = AppendTokenDataset(dataset,
                                             self.lang_tok_idx[tgt])

                # The dataset to be backtranslated.
                # It has src but not trg (will be generated on-the-fly)
                # After backtranslation, the source sentences in this dataset
                # will be returned as the targets.
                pair_dataset_1 = LanguagePairDataset(
                    dataset,
                    dataset.sizes,
                    self.dicts[tgt],
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    eos=self.lang_tok_idx[tgt],
                )

                pair_dataset_1.name = f"dataset_1_{tgt}"
                # dummy parallel dataset, which is used only for obtaining its
                # `collater` function, that is called on the backtranslated
                # samples to create the final batch.
                # Its src will be filled with the backtranslated sentences.
                pair_dataset_2 = LanguagePairDataset(
                    dataset,
                    dataset.sizes,
                    src_dict=self.dicts[src],
                    tgt=dataset,
                    tgt_sizes=dataset.sizes,
                    tgt_dict=self.dicts[tgt],
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    eos=self.lang_tok_idx[tgt]
                )

                # pair_dataset_2 = TransformEosLangPairDataset(
                #     pair_dataset_2,
                #     src_eos=self.lang_tokens[tgt]['id'],
                #     new_src_eos=self.lang_tokens[src]['id'],
                # )

                pair_dataset_2.name = f"dataset_2_{src}'_{tgt}"

                # self.debug_batch(pair_dataset_2.collater([pair_dataset_2[x] for x in range(20)]))

                # construct the backtranslation dataset (en-de), where its
                # src is synthetic, generated by a backward translation model
                backtranslate_datasets[lang_pair] = BacktranslationDataset(
                    # The dataset to be backtranslated
                    tgt_dataset=pair_dataset_1,
                    # this is initialized beforehand in the build_model()
                    backtranslation_fn=self.backtranslators[lang_pair],
                    # the dictionary of backtranslated sentences
                    src_dict=self.dicts[src],
                    # the dictionary of sentences to be backtranslated
                    tgt_dict=self.dicts[tgt],
                    # function to call on the backtranslated samples
                    # to create the final batch
                    output_collater=pair_dataset_2.collater,
                )
                logger.info('backtranslate-{}: {} {} {} examples'.format(
                    tgt, data_path, split,
                    len(backtranslate_datasets[lang_pair]),
                ))
                self.backtranslate_datasets[lang_pair] = backtranslate_datasets[
                    lang_pair]
                self.backtranslate_datasets[lang_pair].name = lang_pair

        # combine all the datasets for all language pairs for a given split
        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict(
                [(lang_pair, self._language_pair_dataset(lang_pair,
                                                         src_datasets,
                                                         tgt_datasets))
                 for lang_pair in src_datasets.keys()]
                +
                [(_get_bt_dataset_key(lang_pair), dataset)
                 for lang_pair, dataset in backtranslate_datasets.items()]
            ),
            eval_key=None if self.training
            else "%s-%s" % (self.args.source_lang, self.args.target_lang),
        )

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)

        # we imitate a FairseqMultiModel
        model.max_positions = self.max_positions

        # set the backtranslation functions for each language, by creating a
        # SequenceGenerator for each language pair
        self.sequence_generators = {}
        if self.training:
            for lang_pair in self.lang_pairs:
                # generator for the src->tgt language pair (e.g., en->de)
                src, tgt = lang_pair.split('-')

                # this means that we want to use the *backward* model
                # to generate the synthetic parallel data: tgt->src (de->en')
                back_key = '{}-{}'.format(tgt, src)

                # therefore, we are going to use for the backtranslate_fn()
                # of a given lang_pair (src, tgt), a SequenceGenerator that
                # generates in the tgt->src direction...

                # replace beginning-of-sentence in target sentence
                # with target language token, therefore the src (i.e., en)
                decoder_lang_idx = self.lang_tok_idx[src]

                # create the sequence_generators for the *backward* model
                self.sequence_generators[back_key] = SequenceGenerator(
                    BannedTokensEnsembleModel([model], self.lang_masks[src]),
                    tgt_dict=self.dicts[src],  # Vocab of synthetic sentences
                    beam_size=args.bt_beam_size,
                    max_len_a=args.bt_max_len_a,
                    max_len_b=args.bt_max_len_b,
                    no_repeat_ngram_size=args.bt_no_repeat_ngram_size,
                    # eos=self.dictionary.eos()  # decoder_lang_idx
                    eos=decoder_lang_idx  # decoder_lang_idx
                )

                def prefix_bos_langid(lang_idx, batch):
                    """
                    Explicitly specify that the prefix should be: [LANG] <s>
                    """
                    src = batch['net_input']['src_tokens']
                    pref = [lang_idx, self.dictionary.bos()]
                    pref = torch.tensor(pref, dtype=src.dtype,
                                        device=src.device)
                    pref = pref.unsqueeze(0).repeat(src.size(0), 1)
                    return pref

                # define the backtranslate_fn() to be attached to the lang_pair
                def backtranslate_fn(sample, model=model,
                                     bos_token=decoder_lang_idx,
                                     sequence_generator=
                                     self.sequence_generators[back_key]):
                    synthetic = sequence_generator.generate(
                        model,
                        sample,
                        bos_token=bos_token,
                        # LangID of synthetic sentences!
                        # prefix_tokens=get_prefix(bos_token, sample)
                    )

                    return synthetic

                # so, the self.backtranslators['en-de'], uses a backtranslate_fn
                # which is the  generate() of model.models['de-en']
                self.backtranslators[lang_pair] = backtranslate_fn

        self.eval_sequence_generators = {}
        # use the parallel data to evaluate using BLEU
        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(
                getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            for lang_pair in self.lang_pairs:
                src, tgt = lang_pair.split('-')
                self.eval_sequence_generators[lang_pair] = SequenceGenerator(
                    [model],
                    self.dictionary,
                    beam_size=getattr(gen_args, 'beam', 5),
                    max_len_a=getattr(gen_args, 'max_len_a', 0),
                    max_len_b=getattr(gen_args, 'max_len_b', 200),
                    min_len=getattr(gen_args, 'min_len', 1),
                    normalize_scores=(
                        not getattr(gen_args, 'unnormalized', False)),
                    len_penalty=getattr(gen_args, 'lenpen', 1),
                    unk_penalty=getattr(gen_args, 'unkpen', 0),
                    temperature=getattr(gen_args, 'temperature', 1.),
                    match_source_len=getattr(gen_args, 'match_source_len',
                                             False),
                    no_repeat_ngram_size=getattr(gen_args,
                                                 'no_repeat_ngram_size', 0),
                    eos=self.dictionary.index('[{}]'.format(tgt))
                )

        return model

    def train_step(self, sample, model, criterion, optimizer, update_num,
                   ignore_grad=False):
        """
        Simplified version of SemisupervisedTranslationTask.train_step
        """
        model.train()

        if update_num == self.args.bt_output_filtering_steps:
            # stop masking the outputs of the backtranslations
            for k, v in self.sequence_generators.items():
                v.model.mask = None
            logger.info(f"| Unblocking on-the-fly backtranslation!")

        agg_loss, agg_sample_size, agg_logging_output = 0., 0., defaultdict(
            float)

        def forward_backward(samples, logging_output_key):
            nonlocal agg_loss, agg_sample_size, agg_logging_output
            if samples is None or len(samples) == 0:
                return

            loss, sample_size, logging_output = criterion(model, samples)
            if ignore_grad:
                loss *= 0

            optimizer.backward(loss)
            agg_loss += loss.detach().item()
            agg_sample_size += sample_size

            for k in logging_output:
                agg_logging_output[k] += logging_output[k]
                agg_logging_output[f"{logging_output_key}:{k}"] += \
                    logging_output[k]

        for lang_pair in self.lang_pairs:
            sample_key = _get_bt_dataset_key(lang_pair)
            forward_backward(sample[sample_key], sample_key)

        if update_num % self.args.log_interval == 0:
            self._log_batch(sample)

        return agg_loss, agg_sample_size, agg_logging_output

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.dictionary.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=(
                    "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
                ),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.dictionary.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
        if self.args.eval_bleu_print_samples:
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])

        for lang in self.langs:
            lang_token = '[{}]'.format(lang)
            hyps = [h.replace(lang_token, "") for h in hyps]
            refs = [r.replace(lang_token, "") for r in refs]

        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    def valid_step(self, sample, model, criterion):

        agg_loss, agg_sample_size, agg_logging_output = 0., 0., defaultdict(
            float)

        for lang_pair in self.lang_pairs:
            src, tgt = lang_pair.split('-')

            if lang_pair not in sample or sample[lang_pair] is None or len(
                    sample[lang_pair]) == 0:
                continue

            if 'nsentences' not in sample[lang_pair]:
                continue

            # loss, sample_size, logging_output = criterion(model,
            #                                               sample[lang_pair])
            loss, sample_size, logging_output = super().valid_step(
                sample[lang_pair], model, criterion)

            agg_loss += loss.detach().data.item()
            agg_sample_size += sample_size

            for k in logging_output:
                agg_logging_output[k] += logging_output[k]
                agg_logging_output[f"{lang_pair}:{k}"] += logging_output[k]

            if self.args.eval_bleu:
                bleu = self._inference_with_bleu(
                    self.eval_sequence_generators[lang_pair],
                    sample[lang_pair], model)
                agg_logging_output[f'_{lang_pair}:bleu_sys_len'] = bleu.sys_len
                agg_logging_output[f'_{lang_pair}:bleu_ref_len'] = bleu.ref_len
                # we split counts into separate entries so that they can be
                # summed efficiently across workers using fast-stat-sync
                assert len(bleu.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    agg_logging_output[f'_{lang_pair}:bleu_counts_' + str(i)] = \
                        bleu.counts[i]
                    agg_logging_output[f'_{lang_pair}:bleu_totals_' + str(i)] = \
                        bleu.totals[i]

        return agg_loss, agg_sample_size, agg_logging_output

    def reduce_metrics(self, logging_outputs, criterion):

        if len(logging_outputs[0]) == 0:
            return None

        # print("---------------------------------")
        # pprint.pprint(len(logging_outputs[0]))
        # pprint.pprint(logging_outputs)

        results = super().reduce_metrics(logging_outputs, criterion)

        bleu_scores = []

        def compute_bleu_for_lang_pair(lang_pair):
            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs(f'_{lang_pair}:bleu_counts_' + str(i)))
                totals.append(sum_logs(f'_{lang_pair}:bleu_totals_' + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar(f'_{lang_pair}:bleu_counts',
                                   np.array(counts))
                metrics.log_scalar(f'_{lang_pair}:bleu_totals',
                                   np.array(totals))
                metrics.log_scalar(f'_{lang_pair}:bleu_sys_len',
                                   sum_logs(f'{lang_pair}:_bleu_sys_len'))
                metrics.log_scalar(f'_{lang_pair}:bleu_ref_len',
                                   sum_logs(f'{lang_pair}:_bleu_ref_len'))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu
                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if 'smooth_method' in fn_sig:
                        smooth = {'smooth_method': 'exp'}
                    else:
                        smooth = {'smooth': 'exp'}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters[f'_{lang_pair}:bleu_counts'].sum,
                        total=meters[f'_{lang_pair}:bleu_totals'].sum,
                        sys_len=meters[f'_{lang_pair}:bleu_sys_len'].sum,
                        ref_len=meters[f'_{lang_pair}:bleu_ref_len'].sum,
                        **smooth
                    )
                    bleu_scores.append(bleu.score)
                    return round(bleu.score, 2)

                metrics.log_derived(f'bleu:{lang_pair}', compute_bleu)

                return True
            else:
                return False

        # this is a very hasty hack. fix this later.
        if self.args.eval_bleu:
            is_eval = False
            for lang_pair in self.lang_pairs:
                is_eval = compute_bleu_for_lang_pair(lang_pair)

            if is_eval:
                def compute_bleu_mean(meters):
                    if len(bleu_scores) > 0:
                        return sum(bleu_scores) / len(bleu_scores)
                    else:
                        return 0.

                metrics.log_derived(f'bleu', compute_bleu_mean)

        return results

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.dictionary

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        if len(self.datasets.values()) == 0:
            return {'%s-%s' % (self.args.source_lang, self.args.target_lang):
                        (self.args.max_source_positions,
                         self.args.max_target_positions)}
        return OrderedDict([
            (key,
             (self.args.max_source_positions, self.args.max_target_positions))
            for split in self.datasets.keys()
            for key in self.datasets[split].datasets.keys()
        ])
