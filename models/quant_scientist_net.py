import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 1st conv before any network block
        self.conv0 = torch.nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=2, bias=False)
        )

        block = BasicBlock
        # 1st block
        self.block1 = NetworkBlock(1, 16, 56, block, 1, 0.5)
        # self.block2 = NetworkBlock(1, 64, 128, block, 1, 0.2)
        # self.block3 = NetworkBlock(1, 64, 128, block, 1, 0.2)

        self.features = nn.Sequential(
            self.conv0,
            self.block1,
            # self.block2,
            # self.block3,
        )
        self.classifier = torch.nn.Sequential(
            nn.Linear(18144, 1),
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.sig(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, dropout_prob=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu1 = torch.nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=5, stride=stride,
                               padding=1, bias=False)
        self.maxpool1 = torch.nn.MaxPool2d(2, 2)
        self.avgpool = torch.nn.AvgPool2d(2, 2)
        self.dropout_prob = dropout_prob

    def forward(self, x):
        x = self.conv1(self.relu1(self.bn1(x)))
        x = self.maxpool1(x)
        x = self.avgpool(x)
        if self.dropout_prob > 0:
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
        return x


class NetworkBlock(nn.Module):
    def __init__(self, n_layers, in_channels, out_channels, block, stride, dropout_prob=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_channels, out_channels, n_layers, stride, dropout_prob)

    def _make_layer(self, block, in_channels, out_channels, n_layers, stride, dropout_prob):
        layers = []
        for i in range(int(n_layers)):
            layers.append(block(i == 0 and in_channels or out_channels, out_channels,
                                i == 0 and stride or 1, dropout_prob))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)