import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np


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
                #self.bias.data = - mean_act.data * inv_stdv.data
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
                #self.bias.data = - mean_act.data * inv_stdv.data
            elif moving_average:
                self.avg_mean.mul_(self.momentum).add_(1. - self.momentum, mean_act.data)
        else:
            avg_mean = Variable(self.avg_mean)
            assert avg_mean.requires_grad == False
            activation = activation - avg_mean[None, :, None, None].expand_as(activation)

        if self.bias is not None:
            activation = activation + self.bias[None, :, None, None].expand_as(activation)

        return activation
        

class WN_ConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, train_scale=False, init_stdv=1.0):
        super(WN_ConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias)
        if train_scale:
            self.weight_scale = Parameter(torch.Tensor(out_channels))
        else:
            self.register_buffer('weight_scale', torch.Tensor(out_channels))
        
        self.train_scale = train_scale 
        self.init_mode = False
        self.init_stdv = init_stdv

        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.weight_scale.data.fill_(1.)
        else:
            self.weight_scale.fill_(1.)

    def forward(self, input, output_size=None):
        if self.train_scale:
            weight_scale = self.weight_scale
        else:
            weight_scale = Variable(self.weight_scale)
        # normalize weight matrix and linear projection [in x out x h x w]
        # for each output dimension, normalize through (in, h, w)  = (0, 2, 3) dims
        norm_weight = self.weight * (weight_scale[None,:,None,None] / torch.sqrt((self.weight ** 2).sum(3).sum(2).sum(0))).expand_as(self.weight)
        output_padding = self._output_padding(input, output_size)
        activation = F.conv_transpose2d(input, norm_weight, bias=None, 
                                        stride=self.stride, padding=self.padding, 
                                        output_padding=output_padding, groups=self.groups)

        if self.init_mode == True:
            mean_act = activation.mean(3).mean(2).mean(0).squeeze()
            activation = activation - mean_act[None,:,None,None].expand_as(activation)

            inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(3).mean(2).mean(0)).squeeze()
            activation = activation * inv_stdv[None,:,None,None].expand_as(activation)

            if self.train_scale:
                self.weight_scale.data = self.weight_scale.data * inv_stdv.data
            else:
                self.weight_scale = self.weight_scale * inv_stdv.data
            self.bias.data = - mean_act.data * inv_stdv.data

        else:
            if self.bias is not None:
                activation = activation + self.bias[None,:,None,None].expand_as(activation)

        return activation


class WN_Linear_V2(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, train_scale=False, init_stdv=1.0):
        super(WN_Linear_V2, self).__init__(in_features, out_features, bias=bias)
        if train_scale:
            self.weight_scale = Parameter(self._norm(self.weight).data)
        else:
            self.register_buffer('weight_scale', self._norm(self.weight).data)

        self.train_scale = train_scale 
        self.init_stdv = init_stdv

        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(0, std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.weight_scale.data.fill_(1.)
        else:
            self.weight_scale.fill_(1.)

    def _norm(self, weight):
        output_size = (self.weight.size(0),) + (1,) * (self.weight.dim() - 1)
        norm = weight.contiguous().view(weight.size(0), -1).norm(dim=1).view(*output_size)
        return norm

    def forward(self, input, init_mode):
        if self.train_scale:
            weight_scale = self.weight_scale
        else:
            weight_scale = Variable(self.weight_scale)

        # print(weight_scale.size())
        # print((weight_scale.unsqueeze(1)).size())
        # print((weight_scale / torch.sqrt((self.weight ** 2).sum(1) + 1e-6)).size())
        # print(((weight_scale / torch.sqrt((self.weight ** 2).sum(1) + 1e-6))[:,None]).size())
        # print(self.weight.size())
        # print((weight_scale / torch.sqrt((self.weight ** 2).sum(1) + 1e-6))[:,None].expand_as(self.weight).size())
        
        # exit()

        # normalize weight matrix and linear projection
        norm_weight = self.weight *  (weight_scale / self._norm(self.weight))
        
        activation = F.linear(input, norm_weight)

        if init_mode == True:
            mean_act = activation.mean(0).squeeze(0)
            activation = activation - mean_act.expand_as(activation)

            inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(0)).squeeze(0)
            activation = activation * inv_stdv.expand_as(activation)

            if self.train_scale:
                self.weight_scale.data = self.weight_scale.data * inv_stdv.view(self.weight_scale.size()).data
            else:
                self.weight_scale = self.weight_scale * inv_stdv.view(self.weight_scale.size()).data
            self.bias.data = - mean_act.data * inv_stdv.data

        else:
            if self.bias is not None:
                activation = activation + self.bias.expand_as(activation)

        return activation

class WN_Conv2d_V2(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, train_scale=False, init_stdv=1.0):
        super(WN_Conv2d_V2, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if train_scale:
            self.weight_scale = Parameter(self._norm(self.weight).data)
        else:
            self.register_buffer('weight_scale', self._norm(self.weight).data)
        
        self.train_scale = train_scale 
        self.init_stdv = init_stdv

        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.weight_scale.data.fill_(1.)
        else:
            self.weight_scale.fill_(1.)
    
    def _norm(self, weight):
        output_size = (weight.size(0),) + (1,) * (weight.dim() - 1)
        norm = weight.contiguous().view(weight.size(0), -1).norm(dim=1).view(*output_size)
        return norm

    def forward(self, input, init_mode=False):
        if self.train_scale:
            weight_scale = self.weight_scale
        else:
            weight_scale = Variable(self.weight_scale)
        # normalize weight matrix and linear projection [out x in x h x w]
        # for each output dimension, normalize through (in, h, w) = (1, 2, 3) dims
        # print(weight_scale.size(), (weight_scale[:,None,None,None]).size())
        # print(((self.weight ** 2).sum(3).sum(2).sum(1)).size())
        # print((self.weight).size())
        # a = (weight_scale / torch.sqrt((self.weight ** 2).sum(3).sum(2).sum(1) + 1e-6))[:,None,None,None]
        # print(a.size(), self.weight.size())
        # b = a.expand_as(self.weight)
        # print(b.size())

        # exit()
        norm_weight = self.weight *  (weight_scale / self._norm(self.weight))
        activation = F.conv2d(input, norm_weight, bias=None, 
                              stride=self.stride, padding=self.padding, 
                              dilation=self.dilation, groups=self.groups)

        if init_mode == True:
            mean_act = activation.mean(3).mean(2).mean(0).squeeze()
            activation = activation - mean_act[None,:,None,None].expand_as(activation)

            inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(3).mean(2).mean(0)).squeeze()
            activation = activation * inv_stdv[None,:,None,None].expand_as(activation)

            if self.train_scale:
                self.weight_scale.data = self.weight_scale.data * inv_stdv.view(self.weight_scale.size()).data
            else:
                self.weight_scale = self.weight_scale * inv_stdv.view(self.weight_scale.size()).data
            self.bias.data = - mean_act.data * inv_stdv.data

        else:
            if self.bias is not None:
                activation = activation + self.bias[None,:,None,None].expand_as(activation)

        return activation

    