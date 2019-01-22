# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Functions to load data from folders and augment it"""

import itertools
import torch
import os.path
import numpy as np

from PIL import Image
from torch.utils.data.sampler import Sampler
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from .constant import DATA_NO_LABEL, DATA_FOUR_SPINS, DATA_TWO_CIRCLES, DATA_TWO_MOONS


class RandomTranslateWithReflect:
    """Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image


class ZCATransformation(object):
    def __init__(self, transformation_matrix, transformation_mean):
        if transformation_matrix.size(0) != transformation_matrix.size(1):
            raise ValueError("transformation_matrix should be square. Got " +
                             "[{} x {}] rectangular matrix.".format(*transformation_matrix.size()))
        self.transformation_matrix = transformation_matrix
        self.transformation_mean = transformation_mean

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be whitened.
        Returns:
            Tensor: Transformed image.
        """
        if tensor.size(0) * tensor.size(1) * tensor.size(2) != self.transformation_matrix.size(0):
            raise ValueError("tensor and transformation matrix have incompatible shape." +
                             "[{} x {} x {}] != ".format(*tensor.size()) +
                             "{}".format(self.transformation_matrix.size(0)))
        flat_tensor = tensor.view(1, -1)
        transformed_tensor = torch.mm(flat_tensor - self.transformation_mean, self.transformation_matrix)
        tensor = transformed_tensor.view(tensor.size())
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += (str(self.transformation_matrix.numpy().tolist()) + ')')
        return format_string


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


def relabel_dataset(dataset, labels):
    unlabeled_idxs = []
    for idx in range(len(dataset.imgs)):
        path, _ = dataset.imgs[idx]
        filename = os.path.basename(path)
        if filename in labels:
            label_idx = dataset.class_to_idx[labels[filename]]
            dataset.imgs[idx] = path, label_idx
            del labels[filename]
        else:
            dataset.imgs[idx] = path, DATA_NO_LABEL
            unlabeled_idxs.append(idx)

    if len(labels) != 0:
        message = "List of unlabeled contains {} unknown files: {}, ..."
        some_missing = ', '.join(list(labels.keys())[:5])
        raise LookupError(message.format(len(labels), some_missing))

    labeled_idxs = sorted(set(range(len(dataset.imgs))) - set(unlabeled_idxs))

    return labeled_idxs, unlabeled_idxs


def create_data_loaders_from_data(data_x_train, data_y_train, data_x_test, data_y_test, labeled_mask, train_transformation, args):
    if len(data_x_train.shape) == 4 and data_x_train.shape[-1] == 1 and data_x_train.shape[-2] == 1:
        data_x_train = data_x_train.reshape(data_x_train.shape[0], data_x_train.shape[1])

    if len(data_x_test.shape) == 4 and data_x_test.shape[-1] == 1 and data_x_test.shape[-2] == 1:
        data_x_test = data_x_test.reshape(data_x_test.shape[0], data_x_test.shape[1])

    tensor_x_train = torch.stack([torch.Tensor(i) for i in data_x_train])
    tensor_y_train = torch.from_numpy(data_y_train).long()
    dataset_train = torch.utils.data.TensorDataset(tensor_x_train, tensor_y_train)
    for i in range(len(dataset_train.tensors[0])):
        if labeled_mask[i] == 0.0:
            dataset_train.tensors[1][i] = DATA_NO_LABEL

    num = len(data_x_train)
    labeled_indices = [i for i in range(num) if labeled_mask[i] > 0.0]
    unlabeled_indices = [i for i in range(num) if labeled_mask[i] == 0.0]
    batch_sampler = TwoStreamBatchSampler(unlabeled_indices, labeled_indices, args.batch_size, args.labeled_batch_size)
    train_loader = DataLoader(dataset_train, batch_sampler=batch_sampler, num_workers=args.workers, pin_memory=True,
                              shuffle=False)

    tensor_x_test = torch.stack([torch.Tensor(i) for i in data_x_test])
    tensor_y_test = torch.from_numpy(data_y_test).long()
    dataset_eval = torch.utils.data.TensorDataset(tensor_x_test, tensor_y_test)
    eval_loader = DataLoader(dataset_eval, batch_size=args.batch_size, pin_memory=True, drop_last=False,
                             num_workers=2 * args.workers, shuffle=False)
    return train_loader, eval_loader


def create_data_loaders_from_dir(train_transformation, eval_transformation, datadir, args):
    traindir = os.path.join(datadir, args.train_subdir)
    evaldir = os.path.join(datadir, args.eval_subdir)

    dataset = ImageFolder(traindir, train_transformation)

    if args.labels:
        with open(args.labels) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
        labeled_idxs, unlabeled_idxs = relabel_dataset(dataset, labels)
        if args.labeled_batch_size >= 0:
            batch_sampler = TwoStreamBatchSampler(unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
            train_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.workers, pin_memory=True,
                                      shuffle=False)
        else:
            #labeled numbers are vary in a minibatch
            train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                                      shuffle=True)
    else:
        train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                                  shuffle=True)

    eval_loader = DataLoader(
        ImageFolder(evaldir, eval_transformation),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    return train_loader, eval_loader


def load_synthetic_data(dataset, data_dir):
    if dataset == DATA_FOUR_SPINS:
        return load_four_spins(data_dir)
    elif dataset == DATA_TWO_MOONS:
        return load_two_moons(data_dir)
    elif dataset == DATA_TWO_CIRCLES:
        return load_two_circles(data_dir)
    else:
        raise Exception("unknown synthetic dataset")


def load_four_spins(data_dir):
    import pickle
    path = os.path.join(data_dir, 'four_spins.save')
    f = open(path, 'rb')
    loaded_objects = []
    for i in range(3):
        loaded_objects.append(pickle.load(f, encoding='latin1'))
    f.close()
    num_all = np.shape(loaded_objects[0])[0]

    x_train = np.reshape(loaded_objects[0], [-1, 2, 1, 1])
    x_test = np.reshape(loaded_objects[0], [-1, 2, 1, 1])
    y_train = np.int32(loaded_objects[1])
    y_test = np.int32(loaded_objects[1])
    mask = loaded_objects[2]

    return np.float32(x_train), y_train, np.float32(x_test), y_test, mask


def load_two_moons(data_dir):
    import pickle
    path = os.path.join(data_dir, 'two_moons.save')
    f = open(path, 'rb')
    loaded_objects = []
    for i in range(3):
        loaded_objects.append(pickle.load(f, encoding='latin1'))
    f.close()

    x_train = np.reshape(loaded_objects[0], [-1, 2])
    x_test = np.reshape(loaded_objects[0], [-1, 2])
    y_train = np.int32(loaded_objects[1])
    y_test = np.int32(loaded_objects[1])
    mask = loaded_objects[2]

    return np.float32(x_train), y_train, np.float32(x_test), y_test, mask


def load_two_circles(data_dir):
    import pickle
    path = os.path.join(data_dir, 'two_circles.save')
    f = open(path, 'rb')
    loaded_objects = []
    for i in range(3):
        loaded_objects.append(pickle.load(f))
    f.close()

    x_train = np.reshape(loaded_objects[0], [-1, 2, 1, 1])
    x_test = np.reshape(loaded_objects[0], [-1, 2, 1, 1])
    y_train = np.int32(loaded_objects[1])
    y_test = np.int32(loaded_objects[1])
    mask = loaded_objects[2]

    return np.float32(x_train), y_train, np.float32(x_test), y_test, mask


def pad_resize(tensor, pad):
    assert len(tensor.size()) == 3
    import torch.nn.functional as F
    channel, xsize, ysize = tensor.size(0), tensor.size(1), tensor.size(2)
    tensor = tensor.unsqueeze(0)
    x_pad = F.pad(tensor, (pad, pad, pad, pad), mode='reflect')
    x_pad = x_pad.squeeze(0)
    ofs0 = np.random.randint(-pad, pad + 1) + pad
    ofs1 = np.random.randint(-pad, pad + 1) + pad
    x_pad_resize = x_pad[:, ofs0:ofs0 + xsize, ofs1:ofs1 + ysize]
    return x_pad_resize


def global_contrast_normalize(tensor, scale=55., min_divisor=1e-8):
    X = tensor - tensor.mean(axis=1, keep_dim=True)

    normalizers = torch.sqrt((X ** 2).sum(axis=1)) / scale
    normalizers[normalizers < min_divisor] = 1.

    X /= normalizers.view(-1, 1)

    return X


def hor_flip_tensor(tensor):
    assert tensor.dim() == 3
    if np.random.uniform() > 0.5:
        return tensor.flip([2])
    else:
        return tensor
