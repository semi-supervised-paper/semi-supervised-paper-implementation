# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
import torchvision.transforms as transforms
from ..utils.fun_util import export
from ..utils.data_util import TransformTwice, RandomTranslateWithReflect, ZCATransformation, pad_resize, hor_flip_tensor


@export
def imagenet():
    '''
    usage: datasets.__dict__['imagenet']
    :return: params of imagenet
    '''
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    train_transformation = TransformTwice(transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/ilsvrc2012/',
        'num_classes': 1000
    }


@export
def cifar10():
    '''
    usage: datasets.__dict__['cifar10']
    :return: params of cifar10
    '''
    import scipy.io as sio
    import torch
    mat_contents = sio.loadmat('./data_local/zca_theano.mat')
    transformation_matrix = torch.from_numpy(mat_contents['zca_matrix']).float()
    transformation_mean = torch.from_numpy(mat_contents['zca_mean'][0]).float()
    train_transformation = TransformTwice(transforms.Compose([
        transforms.ToTensor(),
        ZCATransformation(transformation_matrix, transformation_mean),
        transforms.Lambda(lambda tensor : pad_resize(tensor, 2)),
        transforms.Lambda(lambda tensor : hor_flip_tensor(tensor))
    ]))
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        ZCATransformation(transformation_matrix, transformation_mean)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': './data_local/images/cifar/cifar10/by-image',
        'num_classes': 10
    }


@export
def two_moons():
    train_transformation = TransformTwice(transforms.Compose([
        RandomTranslateWithReflect(1),
        transforms.ToTensor()
    ]))
    return {
        'train_transformation': train_transformation,
        'datadir': r'./data_local/synthetic',
        'num_class': 2
    }


@export
def two_circles():
    train_transformation = TransformTwice(transforms.Compose([
        RandomTranslateWithReflect(1),
        transforms.ToTensor()
    ]))
    return {
        'train_transformation': train_transformation,
        'datadir': r'./data_local/synthetic',
        'num_class': 2
    }


@export
def four_spins():
    train_transformation = TransformTwice(transforms.Compose([
        RandomTranslateWithReflect(1),
        transforms.ToTensor()
    ]))
    return {
        'train_transformation': train_transformation,
        'datadir': r'./data_local/synthetic',
        'num_class': 4
    }
