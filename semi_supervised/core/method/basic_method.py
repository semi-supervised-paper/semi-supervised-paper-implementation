import torch
from abc import ABCMeta, abstractmethod
from ..model import models
from ..utils.log_util import TensorboardLogger


class BasicMethod(metaclass=ABCMeta):
    def __init__(self,
                 train_loader,
                 eval_loader,
                 num_classes,
                 args):
        self.result_folder = './logs/' + args.method
        self.tensorboard_logger = TensorboardLogger(self.result_folder)
        self.num_classes = num_classes
        self.args = args
        self.train_loader, self.eval_loader = train_loader, eval_loader
        self.global_step = 0

        if self.args.resume:
            self._load_checkpoint(self.args.resume)
        else:
            self.best_top1_validate, self.best_top5_validate = None, None
            self.start_epoch = args.start_epoch

    def log_to_tf(self, name, var, step, train):
        name = "Train/" + name if train else "Test/" + name
        self.tensorboard_logger.scalar_summary(name, var, step)

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def _validate(self, *args):
        pass
    
    @abstractmethod
    def _save_checkpoint(self, *args):
        pass
    
    @abstractmethod
    def _load_checkpoint(self, filepath):
        pass

    @staticmethod
    @abstractmethod
    def get_params():
        pass

    def create_model(self, ema, args):
        print("=> creating {pretrained} {ema} model '{arch}'".format(
            pretrained='pre-trained ' if args.pretrained else '',
            ema='ema' if ema else '',
            arch=args.arch))

        model_factory = models.__dict__[args.arch]
        model_params = dict(pretrained=args.pretrained, num_classes=self.num_classes)
        model = model_factory(**model_params)
        if args.cuda:
            model = torch.nn.DataParallel(model).cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

