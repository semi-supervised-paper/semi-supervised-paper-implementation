# This code is modified from https://github.com/kimiyoung/ssl_bad_gan/blob/master/model.py
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter


class GaussianNoise(nn.Module):
    def __init__(self, sigma):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, input):
        if self.training:
            noise = Variable(input.data.new(input.size()).normal_(std=self.sigma))
            return input + noise
        else:
            return input


class WN_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, train_scale=True, init_stdv=1.0, momentum=0.999):
        super(WN_Linear, self).__init__(in_features, out_features, bias=bias)
        if train_scale:
            self.g = Parameter(torch.ones(self.out_features))
        else:
            self.register_buffer('g', torch.ones(out_features))

        self.train_scale = train_scale
        self.init_stdv = init_stdv
        self.has_init = False
        self.register_buffer('avg_mean', torch.zeros(out_features))
        self.momentum = momentum
        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(0, std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.g.data.fill_(1.)
        else:
            self.g.fill_(1.)
        self.has_init = False

    def forward(self, input, moving_average=True, init_mode=False):
        assert self.avg_mean.requires_grad == False
        if self.train_scale:
            g = self.g
        else:
            g = Variable(self.g)

        # normalize weight matrix and linear projection
        norm_weight = self.weight * (g.unsqueeze(1) / torch.sqrt(torch.sum((self.weight ** 2), dim=1, keepdim=True))).expand_as(self.weight)
        activation = F.linear(input, norm_weight)

        if self.training:
            mean_act = activation.mean(0).squeeze(0)
            activation = activation - mean_act.expand_as(activation)
            if init_mode or self.has_init == False:
                inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(0)).squeeze(0)
                activation = activation * inv_stdv.expand_as(activation)

                if self.train_scale:
                    self.g.data = self.g.data * inv_stdv.data
                else:
                    self.g = self.g * inv_stdv.data
                self.has_init = True
            elif moving_average:
                self.avg_mean.mul_(self.momentum).add_(1 - self.momentum, mean_act.data)
        else:
            avg_mean = Variable(self.avg_mean)
            assert avg_mean.requires_grad == False
            activation = activation - avg_mean.expand_as(activation)

        if self.bias is not None:
            activation = activation + self.bias.expand_as(activation)

        return activation


class WN_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 train_scale=False, init_stdv=1.0, momentum=0.999):
        super(WN_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if train_scale:
            self.g = Parameter(torch.ones(out_channels))
        else:
            self.register_buffer('g', torch.ones(out_channels))

        self.train_scale = train_scale
        self.init_stdv = init_stdv

        self.has_init = False

        self.register_buffer('avg_mean', torch.zeros(out_channels))
        self.momentum = momentum

        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.g.data.fill_(1.)
        else:
            self.g.fill_(1.)
        self.has_init = False

    def forward(self, input, moving_average=True, init_mode=False):
        if self.train_scale:
            g = self.g
        else:
            g = Variable(self.g)
        # normalize weight matrix and linear projection [out x in x h x w]
        # for each output dimension, normalize through (in, h, w) = (1, 2, 3) dims
        norm_weight = self.weight * (g[:, None, None, None] / torch.sqrt(
            (self.weight ** 2).sum(3).sum(2).sum(1)).view(-1, 1, 1, 1)).expand_as(self.weight)
        activation = F.conv2d(input, norm_weight, bias=None,
                              stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups)

        if self.training:
            mean_act = activation.mean(3).mean(2).mean(0).squeeze()
            activation = activation - mean_act[None, :, None, None].expand_as(activation)

            if init_mode or self.has_init == False:
                inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(3).mean(2).mean(0)).squeeze()
                activation = activation * inv_stdv[None, :, None, None].expand_as(activation)

                if self.train_scale:
                    self.g.data = self.g.data * inv_stdv.data
                else:
                    self.g = self.g * inv_stdv.data
                
                self.has_init = True
            elif moving_average:
                self.avg_mean.mul_(self.momentum).add_(1. - self.momentum, mean_act.data)
        else:
            avg_mean = Variable(self.avg_mean)
            assert avg_mean.requires_grad == False
            activation = activation - avg_mean[None, :, None, None].expand_as(activation)

        if self.bias is not None:
            activation = activation + self.bias[None, :, None, None].expand_as(activation)

        return activation
