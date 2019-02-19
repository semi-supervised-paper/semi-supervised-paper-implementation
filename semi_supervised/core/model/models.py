from ..utils.fun_util import export
from .resnet import ResNet32x32, ResNet224x224
from .resnet_module import ShakeShakeBlock, BottleneckBlock
from .synthetic_net import SyntheticNet
from .simple_model import SimpleModel
from .conv_small import ConvSmallCifar, ConvSmallSVHN
from .conv_large import ConvLargeCifar


@export
def syntheticnet(pretrained=False, **kwargs):
    assert not pretrained
    model = SyntheticNet(**kwargs)
    return model


@export
def simplenet(pretrained=False, **kwargs):
    assert not pretrained
    model = SimpleModel(**kwargs)
    return model


@export
def convsmallcifar(pretrained=False, **kwargs):
    assert not pretrained
    model = ConvSmallCifar(**kwargs)
    return model


@export
def convlargecifar(pretrained=False, **kwargs):
    assert not pretrained
    model = ConvLargeCifar(**kwargs)
    return model


@export
def convsmallsvhn(pretrained=False, **kwargs):
    assert not pretrained
    model = ConvSmallSVHN(**kwargs)
    return model


@export
def cifar_shakeshake26(pretrained=False, **kwargs):
    assert not pretrained
    model = ResNet32x32(ShakeShakeBlock,
                        layers=[4, 4, 4],
                        channels=96,
                        downsample='shift_conv', **kwargs)
    return model


@export
def resnext152(pretrained=False, **kwargs):
    assert not pretrained
    model = ResNet224x224(BottleneckBlock,
                          layers=[3, 8, 36, 3],
                          channels=32 * 4,
                          groups=32,
                          downsample='basic', **kwargs)
    return model
