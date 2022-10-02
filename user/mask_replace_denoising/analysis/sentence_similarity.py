#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

import fileinput
import json
import logging
import os
import sys
from collections import namedtuple, defaultdict

import numpy
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders, LanguagePairDataset
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.interactive')

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input],
                         openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = [t.numel() for t in tokens]

    dataset = LanguagePairDataset(tokens, lengths, task.source_dictionary,
                                  shuffle=False)
    itr = task.get_batch_iterator(
        # dataset=task.build_dataset_for_inference(tokens, lengths),
        dataset=dataset,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'],
            src_lengths=batch['net_input']['src_lengths'],
        )


def get_representation(states, lengths, mask,
                       exclude_eos=False,
                       exclude_langid=False):
    if exclude_eos:
        lengths = lengths - 1
        for i, x in enumerate(lengths):
            mask[i, x] = False

    if exclude_langid:
        lengths = lengths - 1
        for i, x in enumerate(lengths):
            mask[i, x] = False

    states = states * mask.unsqueeze(-1)
    rep = states.sum(1).div(mask.sum(1).unsqueeze(1))

    return rep


def encode_sentences(file, model,
                     args, task, max_positions, encode_fn, dictionary,
                     lang=None):
    start_id = 0
    results = {}
    use_cuda = torch.cuda.is_available() and not args.cpu

    for inputs in buffered_read(file, args.buffer_size):
        for batch in make_batches(inputs, args, task, max_positions, encode_fn):
            tokens = batch.src_tokens
            lengths = batch.src_lengths

            # left padding
            # tokens = [utils.strip_pad(x, dictionary.pad()) for x in tokens]

            # if lang is not None:
            #     lang_id = dictionary.index('[{}]'.format(lang))
            #     tokens = [torch.cat([x, x.new(1).fill_(lang_id)])
            #               for x in tokens]
            #     lengths += 1

            # tokens = pad_sequence(tokens, batch_first=True,
            #                       padding_value=dictionary.pad())

            if use_cuda:
                tokens = tokens.cuda()
                lengths = lengths.cuda()

            mask = ~ tokens.eq(dictionary.pad())
            encoder_out = model.encoder(src_tokens=tokens,
                                        src_lengths=lengths,
                                        return_all_hiddens=True)

            states = []

            # add un-contextualized input embeddings
            states.append(encoder_out.encoder_embedding.transpose(0, 1))
            states += encoder_out.encoder_states
            states.append(encoder_out.encoder_out)

            layers = []
            for i, layer_state in enumerate(states):
                layer_state = layer_state.transpose(0, 1)
                reps = get_representation(layer_state, lengths, mask,
                                          exclude_eos=False)
                layers.append(reps)
            # Batch x Layers x Dim
            embeddings = torch.stack(layers).transpose(0, 1).data.cpu().numpy()

            for i, (id, sent) in enumerate(zip(batch.ids.tolist(), embeddings)):
                src_tokens_i = utils.strip_pad(tokens[i], dictionary.pad())
                results[start_id + id] = sent, src_tokens_i

        # update running id counter
        start_id += len(inputs)

    return results


def sentence_nn(x, y, center=None, topk=1, metric="cosine"):
    x = numpy.array(x)
    y = numpy.array(y)

    if center == "separately":
        x -= x.mean(axis=0)
        y -= y.mean(axis=0)
    elif center == "jointly":
        xy_comb = numpy.concatenate([x, y])
        xy_comb -= xy_comb.mean(axis=0)
        x = xy_comb[:x.shape[0], :]
        y = xy_comb[x.shape[0]:, :]
    else:
        pass

    if metric == "euclidean":
        # index = faiss.IndexFlatL2(x.shape[1])
        # index.add(y)
        # dists, ids = index.search(x, topk)
        dists = euclidean_distances(x, y)
        ids = numpy.array([c.argsort()[:topk] for c in dists])

    elif metric == "cosine":
        # faiss.normalize_L2(x)
        # faiss.normalize_L2(y)
        # index = faiss.IndexFlatIP(x.shape[1])
        # index.add(y)
        # dists, ids = index.search(x, topk)
        dists = cosine_similarity(x, y)
        ids = numpy.array([c.argsort()[::-1][:topk] for c in dists])

    else:
        raise ValueError

    y_true = [i for i in range(len(x))]
    y_pred = ids[:, 0].tolist()

    top1 = sum([yt == yp for yt, yp in zip(y_true, y_pred)]) / len(y_pred)
    topk = sum([yt in yp for yt, yp in zip(y_true, ids)]) / len(ids)

    return top1, topk


def fill_results(src, trg):
    results = defaultdict(list)

    for layer in range(src[0][0].shape[0]):
        src_emb = [src[i][0][layer] for i in range(len(src))]
        trg_emb = [trg[i][0][layer] for i in range(len(trg))]

        # euclidean
        results["euclidean"].append(
            sentence_nn(src_emb, trg_emb, center=False,
                        topk=5, metric="euclidean")[0])

        # euclidean-centered
        results["euclidean_centered_separately"].append(
            sentence_nn(src_emb, trg_emb, center="separately",
                        topk=5, metric="euclidean")[0])

        results["euclidean_centered_jointly"].append(
            sentence_nn(src_emb, trg_emb, center="jointly",
                        topk=5, metric="euclidean")[0])

        # cosine
        results["cosine"].append(
            sentence_nn(src_emb, trg_emb, center=False,
                        topk=5, metric="cosine")[0])
        # cosine-centered
        results["cosine_centered_separately"].append(
            sentence_nn(src_emb, trg_emb, center="separately",
                        topk=5, metric="cosine")[0])
        results["cosine_centered_jointly"].append(
            sentence_nn(src_emb, trg_emb, center="jointly",
                        topk=5, metric="cosine")[0])
    return results


def main(args):
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(os.pathsep),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    logger.info('Encoding source sentences...')
    results_src = encode_sentences(args.input, models[0],
                                   args, task, max_positions, encode_fn,
                                   src_dict,
                                   # task.args.source_lang
                                   )

    logger.info('Encoding target sentences...')
    results_trg = encode_sentences(args.target, models[0],
                                   args, task, max_positions, encode_fn,
                                   tgt_dict,
                                   # task.args.source_lang
                                   # task.args.target_lang
                                   )

    # decoded_input = [decode_fn((" ".join([src_dict.symbols[x]
    #                                       for x in results_src[i][1]])))
    #                  for i in range(len(results_src))]

    logger.info('Computing similarity scores...')
    results = fill_results(results_src, results_trg)

    logger.info('Saving results...')
    if not os.path.exists(os.path.split(args.results_path)[0]):
        os.makedirs(os.path.split(args.results_path)[0])

    with open(args.results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    logger.info('Finished successfully!')


def add_custom_args(parser):
    group = parser.add_argument_group("Custom arguments")
    # fmt: off
    group.add_argument('--target', metavar='FILE',
                       help='path(s) to model file(s), colon separated')
    # group.add_argument('--source-lang', metavar='FILE',
    #                    help='path(s) to model file(s), colon separated')
    # group.add_argument('--target-lang', metavar='FILE',
    #                    help='path(s) to model file(s), colon separated')
    # group.add_argument('--sentence-avg', action='store_true',
    #                    help='normalize gradients by the number of sentences in a batch'
    #                         ' (default is to normalize by number of tokens)')

    # fmt: on
    return group


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    add_custom_args(parser)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
