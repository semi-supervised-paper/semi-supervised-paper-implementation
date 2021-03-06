# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.`
import re
import argparse
from .fun_util import str2bool
from .constant import METHOD_MEAN_TEACHER

__all__ = ['parse_commandline_args', 'parse_dict_args']


def __create_parser():
    from ..data import datasets
    from ..model import models
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                        choices=datasets.__all__,
                        help='dataset: ' +
                            ' | '.join(datasets.__all__) +
                            ' (default: imagenet)')
    parser.add_argument('--train-subdir', type=str, default='train',
                        help='the subdirectory inside the data directory that contains the training data')
    parser.add_argument('--eval-subdir', type=str, default='val',
                        help='the subdirectory inside the data directory that contains the evaluation data')
    parser.add_argument('--labels', default=None, type=str, metavar='FILE',
                        help='list of image labels (default: based on directory structure)')
    parser.add_argument('--exclude-unlabeled', default=False, type=str2bool, metavar='BOOL',
                        help='exclude unlabeled examples from the training set')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=models.__all__,
                        help='model architecture: ' +
                            ' | '.join(models.__all__))
    parser.add_argument('--method', '-m', metavar='METHOD', default=METHOD_MEAN_TEACHER,
                        help='method to train')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--seed', default=1000, type=int, metavar='N',
                        help='traning seed')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--labeled-batch-size', default=None, type=int,
                        metavar='N', help="labeled examples per minibatch (default: no constrain)")
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='max learning rate')
    parser.add_argument('--initial-lr', default=0.0, type=float,
                        metavar='LR', help='initial learning rate when using linear rampup')
    parser.add_argument('--rampup-epoch', default=80, type=int, metavar='EPOCHS',
                        help='length of learning rate rampup in the beginning')
    parser.add_argument('--rampdown-epoch', default=50, type=int, metavar='EPOCHS',
                        help='length of learning rate cosine rampdown (>= length of training)')
    parser.add_argument('--checkpoint-epochs', default=1, type=int,
                        metavar='EPOCHS',
                        help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
    parser.add_argument('--evaluation-epochs', default=1, type=int,
                        metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--topk', default=5, type=int, help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', type=str2bool,
                        help='evaluate model on evaluation set')
    parser.add_argument('-cu', '--cuda', type=bool, default=True,
                        help='train on cuda or not')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--tflog', default=False, type=bool,
                        help='log to tensorboard')
    parser.add_argument('--consistency', default=None, type=float, metavar='WEIGHT',
                        help='use consistency loss with given weight (default: None)')
    parser.add_argument('--prediction_decay', default=0.6, type=float, metavar='WEIGHT',
                        help='Ensemble prediction decay constant')


    '''
    parameters used in SGD
    '''
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='use nesterov momentum', metavar='BOOL')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    '''
    parameters used in Mean Teacher Method
    '''
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--consistency-type', default="mse", type=str, metavar='TYPE',
                        choices=['mse', 'kl'],
                        help='consistency loss type to use')
    parser.add_argument('--logit-distance-cost', default=-1, type=float, metavar='WEIGHT',
                        help='let the student model have two outputs and use an MSE loss '
                             'between the logits with the given weight (default: only have one output)')
    
    '''
    parameters used in VAT
    '''
    parser.add_argument('--epoch-decay-start', default=80, type=int, help='declare when to change lr')
    parser.add_argument('--beta1-2', default=0.5, type=float, metavar='ALPHA',
                        help='beta2 in Adam (default: 0.5)')
    parser.add_argument('--num-iter-per-epoch', default=400, type=int, help='declare how many iterations in an epoch')
    parser.add_argument('--eps', default=8.0, type=float, help="used in vat")

    return parser


def parse_commandline_args():
    return __create_parser().parse_args()


def parse_dict_args(**kwargs):
    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))

    print("Using these command line args: {}".format(" ".join(cmdline_args)))

    return __create_parser().parse_args(cmdline_args)
