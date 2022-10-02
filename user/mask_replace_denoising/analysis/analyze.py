#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import logging
import os
import random
import sys

import numpy
import numpy as np
import torch
from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    tasks,
    utils,
)
from fairseq.logging import metrics, progress_bar

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('analysis')


def _detok(tensor, vocab):
    return [vocab.symbols[t] for t in tensor]


def _debug_batch(seqs, vocab):
    return [numpy.array(list([[vocab[x] for x in s] for s in zip(*r)])).T
            for r in zip(*seqs)]


def _debug_sample(seqs, vocab):
    return numpy.array(list([[vocab[x] for x in s] for s in zip(*seqs)])).T


def override_from_pretrained(args):
    logger.info(f"Loading pretrained model: {args.path}")
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)

    logger.info("Updating arguments based on the pretrained model...")
    prefixes = ["rtd", "mask", "generator", "tie", "share", "word",
                "replace", "rotate", "permute", "mlm", "sample",
                "test_multitask"]

    for arg_name, arg_val in vars(state['args']).items():
        if arg_name.startswith(tuple(prefixes)):
            logger.info(f"Setting arg: {arg_name}={arg_val}")
            setattr(args, arg_name, arg_val)

    return args


def main(args, init_distributed=False):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'
    metrics.reset()

    # Initialize CUDA and distributed training
    use_cuda = False
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
        use_cuda = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    if args.task == 'marss_pretraining' or args.task == 'multilingual_denoising':
        args = override_from_pretrained(args)

    # Print args
    logger.info(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)
    # task.load_dataset(args.valid_subset.split(',')[0])
    task.load_dataset(args.train_subset)
    criterion = task.build_criterion(args)

    # Build model
    logger.info('loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )
    assert len(models) == 1, 'we assume a single model'
    model = models[0]

    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    logger.info(f"Shuffling is set to '{args.shuffle_samples}'!")

    # Initialize data iterator
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.train_subset),
        # dataset=task.dataset(args.valid_subset.split(',')[0]),
        max_tokens=args.max_tokens_valid,
        max_sentences=args.max_sentences_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions()
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=args.shuffle_samples)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=('tqdm' if not args.no_progress_bar else 'none'),
    )
    model.eval()
    criterion.eval()

    print(model)

    nsamples = 0

    log_outputs = []
    for i, sample in enumerate(progress):
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if 'net_input' not in sample:
            continue

        with torch.no_grad():
            loss, sample_size, log_output = criterion(model, sample)

        progress.log(log_output, step=i)
        log_outputs.append(log_output)

        nsamples += sample['nsentences']
        if 0 < args.max_samples < nsamples:
            logger.info(f"Exceeded max_samples ({args.max_samples}). Stopping!")
            break

        # break
        # batch = criterion._debug_batch(
        #     [sample['net_input']['src_tokens'],
        #      sample['net_input']['prev_output_tokens'],
        #      sample['target']])
    with metrics.aggregate() as agg:
        task.reduce_metrics(log_outputs, criterion)
        log_output = agg.get_smoothed_values()

    progress.print(log_output)

    out_dir = os.path.join(args.save_dir, "analysis")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    criterion.save_analysis(out_dir, log_outputs)

    logger.info("Analysis finished successfully!")
    logger.info("\n")


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def add_analysis_args(parser):
    group = parser.add_argument_group("Analysis arguments")
    # fmt: off
    group.add_argument('--max-samples', type=int, default=0,
                       help='(approx.) how many samples to use for analysis')

    parser.add_argument('--shuffle-samples', default=False, action='store_true',
                        help='.')

    group.add_argument('--path', metavar='FILE',
                       help='path(s) to model file(s), colon separated')
    group.add_argument('--model-overrides', default="{}", type=str,
                       metavar='DICT',
                       help='a dictionary used to override model args at analysis '
                            'that were used during model training')
    group.add_argument('--results-path', metavar='RESDIR', type=str,
                       default=None,
                       help='path to save eval results (optional)"')

    # fmt: on
    return group


def cli_main(modify_parser=None):
    parser = options.get_training_parser()
    add_analysis_args(parser)
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(
            port=port)
        args.distributed_rank = None  # set based on device id
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args,),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()
