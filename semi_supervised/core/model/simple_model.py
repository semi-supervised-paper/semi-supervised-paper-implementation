from torch import nn

from ..utils.net_util import GaussianNoise, WN_Conv2d, WN_Linear


class SimpleModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleModel, self).__init__()

        self.gn = GaussianNoise(0.15)
        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = WN_Conv2d(3, 128, 3, padding=1, bias=True, train_scale=True)
        self.conv1b = WN_Conv2d(128, 128, 3, padding=1, bias=True, train_scale=True)
        self.conv1c = WN_Conv2d(128, 128, 3, padding=1, bias=True, train_scale=True)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1 = nn.Dropout(0.5)

        self.conv2a = WN_Conv2d(128, 256, 3, padding=1, bias=True, train_scale=True)
        self.conv2b = WN_Conv2d(256, 256, 3, padding=1, bias=True, train_scale=True)
        self.conv2c = WN_Conv2d(256, 256, 3, padding=1, bias=True, train_scale=True)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2 = nn.Dropout(0.5)

        self.conv3a = WN_Conv2d(256, 512, 3, padding=0, bias=True, train_scale=True)
        self.conv3b = WN_Conv2d(512, 256, 1, padding=0, bias=True, train_scale=True)
        self.conv3c = WN_Conv2d(256, 128, 1, padding=0, bias=True, train_scale=True)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)

        self.fc1 = WN_Linear(128, num_classes, bias=True, train_scale=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, nonlinearity='relu')
                if m.bias is not None:
                   nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=1, nonlinearity='leaky_relu')
                if m.bias is not None:
                   nn.init.constant_(m.bias, 0.)
    
    def forward(self, x, moving_average=True, init_mode=False):
        if self.training:
            x = self.gn(x)
        else:
            moving_average = False
        x = self.activation(self.conv1a(x, moving_average=moving_average, init_mode=init_mode))
        x = self.activation(self.conv1b(x, moving_average=moving_average, init_mode=init_mode))
        x = self.activation(self.conv1c(x, moving_average=moving_average, init_mode=init_mode))
        x = self.mp1(x)
        x = self.drop1(x)

        x = self.activation(self.conv2a(x, moving_average=moving_average, init_mode=init_mode))
        x = self.activation(self.conv2b(x, moving_average=moving_average, init_mode=init_mode))
        x = self.activation(self.conv2c(x, moving_average=moving_average, init_mode=init_mode))
        x = self.mp2(x)
        x = self.drop2(x)

        x = self.activation(self.conv3a(x, moving_average=moving_average, init_mode=init_mode))
        x = self.activation(self.conv3b(x, moving_average=moving_average, init_mode=init_mode))
        x = self.activation(self.conv3c(x, moving_average=moving_average, init_mode=init_mode))
        x = self.ap3(x)

        return self.fc1(x.view(-1, 128), moving_average=moving_average, init_mode=init_mode)
