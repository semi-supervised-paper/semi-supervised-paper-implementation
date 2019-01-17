import torch.nn as nn
import torch
from ..utils.net_util import GaussianNoise, WN_Linear


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
        for i in range(num_classes):
            self.register_parameter('mean_' + str(i), nn.Parameter(torch.FloatTensor(1, hidden_size)))
            torch.nn.init.xavier_uniform(self._parameters['mean_' + str(i)].data)
        #    , nn.Linear(hidden_size, num_classes), nn.BatchNorm1d(num_classes),nn.LeakyReLU(0.1)
        #)

    def forward(self, x):
        pre_fc_output = self.pre_fc(x)
        return self.fc(pre_fc_output), pre_fc_output
