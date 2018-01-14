import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict

class DenseLayer(nn.Sequential):

    def __init__(self, n_input_features, k, bn_size, drop_prob, bias=False):
        super(DenseLayer, self).__init__()
        self.drop_prob = drop_prob
        self.add_module('bn1', nn.BatchNorm2d(n_input_features))
        self.add_module('relu1', nn.ReLU())
        self.add_module('conv1', nn.Conv2d(n_input_features, k * bn_size,
                                            kernel_size=1, stride=1, bias=bias))
        self.add_module('bn2', nn.BatchNorm2d(k * bn_size))
        self.add_module('relu2', nn.ReLU())
        self.add_module('conv2', nn.Conv2d(k * bn_size, k,
                                           kernel_size=3, stride=1, padding=1, bias=bias))

    def forward(self, inp):
        output = super(DenseLayer, self).forward(inp)
        if self.drop_prob > 0:
            output = F.dropout(output, p=self.drop_prob, training=self.training)
        return torch.cat([inp, output], 1)


class DenseBlock(nn.Sequential):
    def __init__(self, n_layers, n_init_features, k, bn_size, drop_prob, bias=False):
        super(DenseBlock, self).__init__()
        for i in range(n_layers):
            layer = DenseLayer(n_init_features + i * k, k,
                               bn_size, drop_prob, bias)
            self.add_module(f'denselayer{i}', layer)


class TransitionLayer(nn.Sequential):
    def __init__(self, n_input_features, n_output_features, dropout_rate):
        super(TransitionLayer, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(n_input_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('conv', nn.Conv2d(n_input_features, n_output_features,
                                          kernel_size=1, stride=1, padding=(1, 1)))
        self.add_module('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        self.add_module('dropout', nn.Dropout(p=dropout_rate))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=16, block_config=(6, 12, 24, 16), n_init_features=32,
                 bn_size=4, dropout_rates=0, n_classes=1, input_shape=(3, 75, 75)):
        super(DenseNet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, n_init_features, kernel_size=7, stride=2, bias=False)),
            ('norm0', nn.BatchNorm2d(n_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))
        n_features = n_init_features
        for i, n_layers in enumerate(block_config):
            block_drop_prob = dropout_rates[i]
            block = DenseBlock(n_layers, n_features, growth_rate, bn_size, block_drop_prob)
            self.features.add_module(f'denseblock{i+1}', block)
            n_features = n_features + growth_rate * n_layers
            if i != len(block_config) - 1:
                trans = TransitionLayer(n_input_features=n_features,
                                        n_output_features=n_features // 2,
                                        dropout_rate=block_drop_prob)
                self.features.add_module(f'transition{i+1}', trans)
                n_features = n_features // 2

        self.features.add_module('norm', nn.BatchNorm2d(n_features))
        self.bn = nn.BatchNorm2d(n_features)
        # n_fc = self._get_conv_output(input_shape)
        self.classifier = nn.Linear(n_features, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _get_conv_output(self, shape):
        bs = 1
        inp = Variable(torch.rand(bs, *shape))
        output_features = self.features(inp)
        n_features = output_features.data.view(bs, -1).size(1)
        return n_features

    def forward(self, inp):
        features = self.features(inp)
        kernel_size = (features.size()[2], features.size()[3])
        output = F.avg_pool2d(F.relu(self.bn(features)), kernel_size)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output



