import numpy as np
import torch
from fairseq.data import data_utils, DenoisingDataset


def collate(
        samples,
        pad_idx,
        eos_idx,
        vocab,
        left_pad_source=False,
        left_pad_target=False,
        input_feeding=True,
):
    assert input_feeding
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_index=None):
        if pad_index is None:
            pad_index = vocab.pad()

        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx=pad_index,
            eos_idx=None,  # use eos_idx of each sample instead of vocab.eos()
            left_pad=left_pad, move_eos_to_beginning=move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    target_self = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step

            # if we add noise in the decoder's input
            if samples[0].get('decoder', None) is not None:
                prev_output_tokens = merge(  # use noisy input_feeding
                    'decoder',
                    left_pad=left_pad_target,
                    move_eos_to_beginning=True,
                )
                prev_output_tokens = prev_output_tokens.index_select(0,
                                                                     sort_order)
                target_self = merge(  # use original input_feeding for MLM
                    'target',
                    left_pad=left_pad_target,
                    move_eos_to_beginning=True,
                )
                target_self = target_self.index_select(0, sort_order)
            else:
                prev_output_tokens = merge(
                    'target',
                    left_pad=left_pad_target,
                    move_eos_to_beginning=True,
                )
                prev_output_tokens = prev_output_tokens.index_select(0,
                                                                     sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
        'nsentences': samples[0]['source'].size(0),
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    # this is the target for the MLM in the decoder's input
    if target_self is not None:
        batch['target_self'] = target_self

    if samples[0].get('permutations', None) is not None:
        permutations = merge('permutations',
                             left_pad=left_pad_source, pad_index=0)
        permutations = permutations.index_select(0, sort_order)
        r = torch.arange(src_tokens.size(1)).expand_as(src_tokens).contiguous()
        r[:, :permutations.size(1)] = permutations
        batch['net_input']['permutations'] = r.long()

    return batch


class MaskReplaceDenoisingDataset(DenoisingDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = kwargs['args']
        self.mask_decoder = getattr(kwargs['args'], 'mask_decoder')
        self.mask_decoder_paired = getattr(kwargs['args'], 'mask_decoder_paired')
        self.word_shuffle_k = getattr(kwargs['args'], 'word_shuffle')

    def add_encoder_noise(self, tokens):
        if self.permute_sentence_ratio > 0.0:
            tokens = self.permute_sentences(tokens,
                                            self.permute_sentence_ratio)

        # if self.word_shuffle_k > 0:
        #     tokens = self._add_word_shuffle_noise(tokens, self.word_shuffle_k)

        if self.mask_ratio > 0:
            tokens = self.add_whole_word_mask(tokens, self.mask_ratio)

        if self.insert_ratio > 0:
            tokens = self.add_insertion_noise(tokens, self.insert_ratio)

        if self.rotate_ratio > 0.0 and np.random.random() < self.rotate_ratio:
            tokens = self.add_rolling_noise(tokens)

        assert (tokens >= 0).all()
        assert (tokens[1:-1] >= 1).all()
        assert (tokens <= len(self.vocab)).all()
        assert tokens[0] == self.vocab.bos()
        assert tokens[-1] == self.eos

        return tokens

    def add_decoder_noise(self, tokens):
        # if self.word_shuffle_k > 0:
        #     tokens = self.add_word_shuffle_noise(tokens, self.word_shuffle_k)

        if self.mask_decoder > 0:
            tokens = self.add_whole_word_mask(tokens, self.mask_decoder)

        assert (tokens >= 0).all()
        assert (tokens[1:-1] >= 1).all()
        assert (tokens <= len(self.vocab)).all()
        assert tokens[0] == self.vocab.bos()
        assert tokens[-1] == self.eos

        return tokens

    def _add_word_shuffle_noise(self, x,
                                max_shuffle_distance=0,
                                only_noise=False):
        # max_shuffle_distance < 1 will return the same sequence
        if max_shuffle_distance <= 1:
            return x

        # define noise word scores
        noise = np.random.uniform(0, max_shuffle_distance, size=x.size())
        noise[0] = -1  # do not move start sentence symbol
        # be sure to shuffle entire words
        word_idx = self.word_starts(x).cumsum(0).int()
        x2 = x.clone()
        length_no_eos = len(x)
        if x[len(x) - 1] == self.vocab.eos():
            length_no_eos = len(x) - 1

        elif x[len(x) - 2] == self.vocab.eos():
            length_no_eos = len(x) - 2
        # generate a random permutation
        scores = word_idx[:length_no_eos] + noise[word_idx[:length_no_eos]]
        # ensure no reordering inside a word
        scores += 1e-6 * np.arange(length_no_eos)
        permutation = scores.argsort()

        if only_noise:
            return permutation
        else:
            # shuffle words
            x2[:length_no_eos].copy_(x2[:length_no_eos][permutation])
            return x2

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            tokens = self.dataset[index]
            assert tokens[-1] == self.eos
            source, target = tokens, tokens.clone()

            # _ = self.add_word_shuffle_noise(tokens.clone(), 5)
            source = self.add_encoder_noise(source)

        sample = {
            'id': index,
            'source': source,
            'target': target,
        }

        if self.word_shuffle_k > 0:
            sample['permutations'] = self._add_word_shuffle_noise(
                tokens, self.word_shuffle_k, only_noise=True)

        if self.mask_decoder > 0 and not self.mask_decoder_paired:
            sample['decoder'] = self.add_decoder_noise(target.clone())

        elif self.mask_decoder_paired:
            # 1: mask all the tokens that are masked in the encoder
            sample['decoder'] = source.clone()

            # 2: mask all the tokens that are shuffled in the encoder
            if self.word_shuffle_k > 0:  # this could be done with F.pad
                diff = ~sample['permutations'].eq(
                    sample['permutations'].sort()[1])
                mask = torch.zeros_like(tokens).bool()
                mask[:diff.size(0)] = diff
                sample['decoder'] = sample['decoder'].masked_fill(mask,
                                                                  self.mask_idx)

        # if self.mask_decoder == 'independent':
        # elif self.mask_replace_decoder == 'encoder':
        #     sample['decoder'] = source

        return sample

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch of data
        """
        return collate(samples, self.vocab.pad(), self.eos, self.vocab)
