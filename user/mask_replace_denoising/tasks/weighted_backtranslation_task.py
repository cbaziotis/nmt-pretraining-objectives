from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

from user.mask_replace_denoising.tasks.backtranslation_from_pretrained_bart import \
    BackTranslationFromPretrainedBARTTask


@register_task('weighted_backtranslation')
class WeightedBackTranslationTask(BackTranslationFromPretrainedBARTTask):

    def train_step(self, sample, model, criterion, optimizer, update_num,
                   ignore_grad=False):
        return TranslationTask.train_step(self, sample, model,
                                          criterion,
                                          optimizer,
                                          update_num,
                                          ignore_grad=False)

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = TranslationTask.valid_step(
            self, sample["parallel"], model, criterion.seq_criterion)
        return loss, sample_size, logging_output
