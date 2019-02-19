from torch import nn


class ConvLargeCifar(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvLargeCifar, self).__init__()

        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = nn.Conv2d(3, 128, 3, padding=1)
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = nn.Conv2d(128, 128, 3, padding=1)
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1 = nn.Dropout(0.5)

        self.conv2a = nn.Conv2d(128, 256, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2 = nn.Dropout(0.5)

        self.conv3a = nn.Conv2d(256, 512, 3, padding=0)
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = nn.Conv2d(512, 256, 1, padding=0)
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = nn.Conv2d(256, 128, 1, padding=0)
        self.bn3c = nn.BatchNorm2d(128)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)

        self.fc1 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.activation(self.bn1a(self.conv1a(x)))
        x = self.activation(self.bn1b(self.conv1b(x)))
        x = self.activation(self.bn1c(self.conv1c(x)))
        x = self.mp1(x)
        x = self.drop1(x)

        x = self.activation(self.bn2a(self.conv2a(x)))
        x = self.activation(self.bn2b(self.conv2b(x)))
        x = self.activation(self.bn2c(self.conv2c(x)))
        x = self.mp2(x)
        x = self.drop2(x)

        x = self.activation(self.bn3a(self.conv3a(x)))
        x = self.activation(self.bn3b(self.conv3b(x)))
        x = self.activation(self.bn3c(self.conv3c(x)))
        x = self.ap3(x)

        return self.fc1(x.view(-1, 128))
