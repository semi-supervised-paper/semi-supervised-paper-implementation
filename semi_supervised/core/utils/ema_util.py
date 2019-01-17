import torch.nn as nn


class EMA:
    def __init__(self):
        super(EMA, self).__init__()
        self.shadow = {}

    def is_registed(self, name):
        return name in self.shadow

    def register(self, name, val):
        if self.is_registed(name):
            raise Exception("name {n} has been registed".format(n=name))
        self.shadow[name] = val

    def get_data(self, name):
        assert name in self.shadow
        return self.shadow[name].data

    def forward(self, name, x, mu):
        assert name in self.shadow and x.requires_grad == True
        self.shadow[name].data = mu * self.get_data(name) + (1.0 - mu) * x.data
        x.data = self.shadow[name].data
        assert x.requires_grad == True
        return x
