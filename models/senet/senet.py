"""
Referenced from https://github.com/QuantScientist/Deep-Learning-Boot-Camp/blob/master/Kaggle-PyTorch/PyTorch-Ensembler/nnmodels/senet.py
"""
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import ResNet

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

REDUCTION=16

class SELayer(nn.Module):
    def __init__(self, channel, reduction=REDUCTION):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduction),
            # nn.ReLU(inplace=True),
            nn.PReLU(),
            nn.Linear(reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_resnet18(num_classes):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet34(num_classes):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(num_classes):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet101(num_classes):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet152(num_classes):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model





class IceSEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, reduction=REDUCTION):
        super(IceSEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1,
                                                  stride=1, bias=False),
                                        nn.BatchNorm2d(planes))
        self.global_features = []
    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out


class IceResNet(nn.Module):
    def __init__(self, block, n_size=1, num_classes=1, num_rgb=2, base=32, dropout_rates=[0, 0, 0, 0]):
        super(IceResNet, self).__init__()
        self.base = base
        self.num_classes = num_classes
        self.inplane = self.base  # 45 epochs
        # self.inplane = 16 # 57 epochs
        self.conv1 = nn.Conv2d(num_rgb, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, self.inplane, blocks=2 * n_size, stride=2)
        self.layer2 = self._make_layer(block, self.inplane * 2, blocks=2 * n_size, stride=2)
        self.layer3 = self._make_layer(block, self.inplane * 4, blocks=2 * n_size, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout_rates = dropout_rates
        self.fc = nn.Linear(int(8 * self.base), num_classes)
        nn.init.kaiming_normal(self.fc.weight)
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride):

        layers = []
        for i in range(1, blocks):
            layers.append(block(self.inplane, planes, stride))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x, features_only=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = F.dropout(self.layer1(x), p=self.dropout_rates[0])
        x = F.dropout(self.layer2(x), p=self.dropout_rates[1])
        x = F.dropout(self.layer3(x), p=self.dropout_rates[2])
        # x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print (x.data.size())
        if features_only:
            return x
        else:
            x = F.dropout(self.fc(x), p=self.dropout_rates[3])
        return x

def iceresnet(n_size=1, num_classes=1, num_rgb=3, base=32, dropout_rates=[0, 0, 0, 0]):
    return IceResNet(IceSEBasicBlock, n_size, num_classes, num_rgb, base, dropout_rates)

class TripleColumnIceResNet(nn.Module):
    def __init__(self, block, n_size, num_classes, num_rgbs, bases, drop_rates, fc_drop_rate, input_shape=[2, 75, 75]):
        super().__init__()
        self.column_ch1 = IceResNet(block, n_size, num_classes, base=bases[0],
                                    num_rgb=num_rgbs[0], drop_rates=drop_rates)
        self.column_ch2 = IceResNet(block, n_size, num_classes, base=bases[1],
                                    num_rgb=num_rgbs[1], drop_rates=drop_rates)
        self.column_combined = IceResNet(block, n_size, num_classes, base=bases[2],
                                         num_rgb=num_rgbs[2], drop_rates=drop_rates)

        n_features = self._get_conv_output(input_shape)
        self.fc = nn.Sequential(
            nn.Dropout(p=fc_drop_rate),
            nn.Linear(n_features, 1)
        )

    def _get_conv_output(self, shape):
        bs = 1
        inp = Variable(torch.rand(bs, *shape))
        ch1, ch2 = torch.split(inp, 1, dim=1)
        fc1 = self.column_ch1(ch1, features_only=True)
        fc2 = self.column_ch2(ch2, features_only=True)
        fc_combined = self.column_combined(inp, features_only=True)
        output_features = torch.cat([fc1, fc2, fc_combined], dim=1)
        n_features = output_features.data.view(bs, -1).size(1)
        return n_features

    def forward(self, inp):
        ch1, ch2 = torch.split(inp, 1, dim=1)
        fc1 = self.column_ch1(ch1, features_only=True)
        fc2 = self.column_ch2(ch2, features_only=True)
        fc_combined = self.column_combined(inp, features_only=True)
        all_features = torch.cat([fc1, fc2, fc_combined], dim=1)
        output = self.fc(all_features)
        return output


def triple_column_iceresnet(num_classes, n_size=1, num_rgbs=[1, 1, 2],
                            bases=[8, 8, 16], drop_rates=[0, 0, 0, 0],
                            fc_drop_rate=0, input_size=[2, 75, 75]):
    return TripleColumnIceResNet(IceSEBasicBlock, n_size, num_classes, num_rgbs, bases, drop_rates, fc_drop_rate,)

