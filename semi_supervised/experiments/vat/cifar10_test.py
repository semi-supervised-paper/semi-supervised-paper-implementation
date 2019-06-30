# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import sys
from ...core.utils.params_util import parse_dict_args
from ...core.utils.constant import LOG_TRAIN_PREFIX
from ...core.method.vat import VAT
from ...core import main

import torch


def parameters():
    defaults = {
        # Technical details
        'workers': 2,
        'checkpoint_epochs': 300,

        # Data
        'dataset': 'cifar10',
        'train_subdir': 'train+val',
        'eval_subdir': 'test',

        # Data sampling
        'base_batch_size': 160,
        'base_labeled_batch_size': 32,

        # Architecture
        'arch': 'convlargecifar',

        'base_lr': 0.001,
        'topk': 5,

        'tflog': True,

        **VAT.get_params()
    }

    # 4000 labels:
    for data_seed in range(1000, 1001):
        yield {
            **defaults,
            'title': '4000-label cifar-10',
            'n_labels': 4000,
            'data_seed': data_seed,
            'epochs': 500
        }

    # # 1000 labels:
    # for data_seed in range(1000, 1001):
    #     yield {
    #         **defaults,
    #         'title': '1000-label cifar-10',
    #         'n_labels': 1000,
    #         'data_seed': data_seed,
    #         'epochs': 500
    #     }


def run(title, base_batch_size, base_labeled_batch_size, base_lr, n_labels, data_seed, **kwargs):
    print('run title: {title}, data seed: {seed}'.format(title=title, seed=data_seed))

    ngpu = torch.cuda.device_count()
    assert ngpu > 0, "Expecting at least one GPU, found none."

    adapted_args = {
        'seed': data_seed,
        'batch_size': base_batch_size * ngpu,
        'labeled_batch_size': base_labeled_batch_size * ngpu,
        'lr': base_lr * ngpu,
        'labels': './data_local/labels/cifar10/{}_balanced_labels/{:d}.txt'.format(n_labels, data_seed)
    }
    main.main_args = parse_dict_args(**adapted_args, **kwargs)
    main.main()


if __name__ == "__main__":
    for run_params in parameters():
        run(**run_params)
