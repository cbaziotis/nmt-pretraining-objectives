import logging
from collections import OrderedDict

from fairseq.data import RoundRobinZipDatasets
from fairseq.tasks import register_task
from fairseq.tasks.translation import load_langpair_dataset
from fairseq.tasks.translation_from_pretrained_bart import \
    TranslationFromPretrainedBARTTask

from .backtranslation_task import BackTranslationTask

logger = logging.getLogger(__name__)


@register_task('backtranslation_from_pretrained_bart')
class BackTranslationFromPretrainedBARTTask(BackTranslationTask,
                                            TranslationFromPretrainedBARTTask):

    @staticmethod
    def add_args(parser):
        TranslationFromPretrainedBARTTask.add_args(parser)
        parser.add_argument('--synthetic-data',
                            help='directory with the backtranslated data.')

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        parallel = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=getattr(self.args, 'max_source_positions',
                                         1024),
            max_target_positions=getattr(self.args, 'max_target_positions',
                                         1024),
            load_alignments=self.args.load_alignments,
            prepend_bos=getattr(self.args, 'prepend_bos', False),
            append_source_id=True
        )

        if split == "train":
            # load the synthetic data
            paths = self.args.synthetic_data.split(':')
            assert len(paths) > 0
            data_path = paths[(epoch - 1) % len(paths)]

            synthetic = load_langpair_dataset(
                data_path, split, src, self.src_dict, tgt, self.tgt_dict,
                combine=combine, dataset_impl=self.args.dataset_impl,
                upsample_primary=self.args.upsample_primary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=getattr(self.args, 'max_source_positions',
                                             1024),
                max_target_positions=getattr(self.args, 'max_target_positions',
                                             1024),
                load_alignments=self.args.load_alignments,
                prepend_bos=getattr(self.args, 'prepend_bos', False),
                append_source_id=True
            )

            self.datasets[split] = RoundRobinZipDatasets(
                OrderedDict([('parallel', parallel), ('synthetic', synthetic)])
            )

        else:
            self.datasets[split] = RoundRobinZipDatasets(
                OrderedDict([('parallel', parallel)])
            )

    def train_step(self, sample, model, criterion, optimizer, update_num,
                   ignore_grad=False):
        return super().train_step(sample, model, criterion, optimizer,
                                  update_num, ignore_grad)

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        # todo: probably I will have to ensure to only use the parallel data...
        raise NotImplementedError
