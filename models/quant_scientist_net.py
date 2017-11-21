import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# This class is adapted from: https://www.kaggle.com/solomonk/pytorch-gpu-cnn-bceloss-with-predictions
class Net(nn.Module):
    def __init__(self, input_shape=(3, 75, 75)):
        super(Net, self).__init__()
        block = BasicBlock
        base_dropout = 0.5

        self.block0 = NetworkBlock(block, n_layers=1, in_channel=input_shape[0], out_channel=32,
                                   kernel_size=5, stride=1, dropout_prob=base_dropout)
        self.block1 = NetworkBlock(block, n_layers=1, in_channel=32, out_channel=64,
                                   kernel_size=3, stride=1, dropout_prob=base_dropout-0.3)
        self.block2 = NetworkBlock(block, n_layers=1, in_channel=64, out_channel=128,
                                   kernel_size=3, stride=1, dropout_prob=base_dropout-0.3)
        # self.block3 = NetworkBlock(block, n_layers=1, in_channel=512, out_channel=2048,
        #                            kernel_size=3, stride=1, dropout_prob=base_dropout-0.3)

        self.features = nn.Sequential(
            self.block0,
            self.block1,
            self.block2,
            # self.block3,
        )

        # Use a hack to calcluate the number of features during the flatten layer
        # https://discuss.pytorch.org/t/inferring-shape-via-flatten-operator/138/4
        self.n_features = self._get_conv_output(input_shape)

        self.classifier = nn.Sequential(nn.Linear(self.n_features, 1),
                                        # nn.Linear(2048, 512),
                                        # nn.Linear(512, 1)
                                        )
    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_features = self.features(input)
        n_features = output_features.data.view(bs, -1).size(1)
        return n_features


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, dropout_prob=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = torch.nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)
        self.maxpool1 = torch.nn.MaxPool2d(2, 2)
        self.dropout_prob = dropout_prob

    def forward(self, x):
        x = self.bn1(self.relu1(self.conv1(x)))
        x = self.maxpool1(x)
        if self.dropout_prob > 0:
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
        return x


class NetworkBlock(nn.Module):
    def __init__(self, block, n_layers, in_channel, out_channel, kernel_size, stride, dropout_prob=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, n_layers, in_channel, out_channel, kernel_size, stride, dropout_prob)

    def _make_layer(self, block, n_layers, in_channel, out_channel, kernel_size, stride, dropout_prob):
        layers = []
        for i in range(int(n_layers)):
            layers.append(block(in_channel, out_channel, kernel_size, stride, dropout_prob))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)