import torch.nn as nn
from ..utils.net_util import GaussianNoise


class SyntheticNet(nn.Module):
    def __init__(self, num_classes):
        super(SyntheticNet, self).__init__()
        input_size, hidden_size = 2, 100
        self.pre_fc = nn.Sequential(
            GaussianNoise(0.05), nn.Linear(input_size, hidden_size), nn.LeakyReLU(0.1),
            GaussianNoise(0.1), nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.1),
            GaussianNoise(0.1), nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.1),
            GaussianNoise(0.1)
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        pre_fc_output = self.pre_fc(x)
        return self.fc(pre_fc_output)
