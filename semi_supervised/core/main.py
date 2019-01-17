from .data import datasets
from .utils.data_util import create_data_loaders_from_data, create_data_loaders_from_dir, load_synthetic_data
from .utils import constant
from .method import methods

import torch
import numpy as np

main_args = None


def main():
    torch.manual_seed(main_args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(main_args.seed)
    print("seed is {}".format(main_args.seed))

    if main_args.dataset == constant.DATA_CIFAR100 or main_args.dataset == constant.DATA_CIFAR10 \
            or main_args.dataset == constant.DATA_IMAGENET:
        dataset_config = datasets.__dict__[main_args.dataset]()
        train_transformation = dataset_config.pop('train_transformation')
        eval_transformation = dataset_config.pop('eval_transformation')
        datadir = dataset_config.pop('datadir')
        num_classes = dataset_config.pop('num_classes')
        train_loader, eval_loader = create_data_loaders_from_dir(train_transformation, eval_transformation,
                                                                 datadir, main_args)
    elif main_args.dataset == constant.DATA_TWO_CIRCLES \
            or main_args.dataset == constant.DATA_FOUR_SPINS \
            or main_args.dataset == constant.DATA_TWO_MOONS:
        dataset_config = datasets.__dict__[main_args.dataset]()
        train_transformation = dataset_config.pop('train_transformation')
        datadir = dataset_config.pop('datadir')
        num_classes = dataset_config.pop('num_class')
        x_train, y_train, x_test, y_test, labeled_mask = load_synthetic_data(main_args.dataset, datadir)
        train_loader, eval_loader = create_data_loaders_from_data(x_train, y_train, x_test, y_test,
                                                                  labeled_mask, train_transformation, main_args)
    else:
        raise Exception("unknown dataset")

    method_factory = methods.__dict__[main_args.method]
    method_params = dict(train_loader=train_loader, eval_loader=eval_loader, num_classes=num_classes, args=main_args)
    method = method_factory(**method_params)
    method.train_model()
