from collections import OrderedDict, defaultdict

from fairseq.data import RoundRobinZipDatasets
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset


@register_task('backtranslation')
class BackTranslationTask(TranslationTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)

        parser.add_argument('--synthetic-data',
                            help='directory with the backtranslated data.')
        # fmt: on

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
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
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
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                load_alignments=self.args.load_alignments,
                truncate_source=self.args.truncate_source,
            )

            self.datasets[split] = RoundRobinZipDatasets(
                OrderedDict([('parallel', parallel), ('synthetic', synthetic)])
            )

        else:
            # self.datasets[split] = parallel
            self.datasets[split] = RoundRobinZipDatasets(
                OrderedDict([('parallel', parallel)])
            )

    def build_model(self, args):
        model = super().build_model(args)

        # we imitate a FairseqMultiModel
        model.max_positions = self.max_positions
        return model

    def train_step(self, sample, model, criterion, optimizer, update_num,
                   ignore_grad=False):
        model.train()

        agg_loss, agg_sample_size, agg_logging_output = 0., 0., defaultdict(
            float)

        def forward(samples, logging_output_key):
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

        forward(sample['parallel'], 'parallel')
        forward(sample['synthetic'], 'synthetic')

        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(
            sample["parallel"], model, criterion)
        return loss, sample_size, logging_output

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
