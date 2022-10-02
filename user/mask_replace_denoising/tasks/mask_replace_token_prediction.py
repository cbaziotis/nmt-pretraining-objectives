import logging

from fairseq import checkpoint_utils
from fairseq.tasks import register_task

from user.mask_replace_denoising.tasks.mask_replace_pretraining import \
    Rtd2SeqTask

logger = logging.getLogger(__name__)


@register_task('mask_replace_token_prediction')
class MaskReplaceTokenPredictionTask(Rtd2SeqTask):

    def __init__(self, args, dictionary):

        logger.info(f"Loading pretrained model: {args.pretrained_checkpoint}")
        self.pretrained = checkpoint_utils.load_checkpoint_to_cpu(
            args.pretrained_checkpoint)

        logger.info("Updating arguments based on the pretrained model...")
        prefixes = ["rtd", "mask", "generator", "tie", "share", "word",
                    "replace", "rotate", "permute", "mlm", "sample",
                    "test_multitask"]

        for arg_name, arg_val in vars(self.pretrained['args']).items():
            if arg_name.startswith(tuple(prefixes)):
                logger.info(f"Setting arg: {arg_name}={arg_val}")
                setattr(args, arg_name, arg_val)

        super().__init__(args, dictionary)

        print()

    @staticmethod
    def add_args(parser):
        Rtd2SeqTask.add_args(parser)

        parser.add_argument('--pretrained-checkpoint',
                            type=str, metavar='STR', required=True,
                            help='path to pretrained model.')

        parser.add_argument('--init-identifier-from-embeddings',
                            default=False, action='store_true',
                            help='path to pretrained model.')

    def build_model(self, args):
        model = super().build_model(args)

        # todo: fix this hack
        for k, v in model.classifier.state_dict().items():
            self.pretrained["model"]["classifier." + k] = v

        model.load_state_dict(self.pretrained["model"], strict=True)

        if args.init_identifier_from_embeddings:
            model.classifier.weight.data = \
                model.encoder.embed_tokens.weight.data.clone()

        for module in [model.generator, model.encoder, model.decoder]:
            if module is not None:
                for n, p in module.named_parameters():
                    logger.warning(f'freezing layer: ({n})')
                    p.requires_grad = False

        # free memory
        self.pretrained = None

        return model

    # def load_dataset(self, split, epoch=1, combine=False, **kwargs):
    #     super().load_dataset(split, epoch, combine, **kwargs)
    #
    #     self.datasets = {k: v for k, v in self.datasets.items()
    #                      if k in [self.args.train_subset,
    #                               self.args.valid_subset.split(",")[0]]}
