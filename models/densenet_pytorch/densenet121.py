from torchvision import models
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F

class DenseNet121PyTorch(nn.Module):
    def __init__(self, pretrained, n_classes, input_shape):
        super().__init__()
        self.features = models.densenet121(pretrained=pretrained).features
        n_features = self._get_model_features(input_shape)
        self.classifier = nn.Linear(n_features, n_classes)

    def _get_model_features(self, shape):
        bs = 1
        inp = Variable(torch.rand(bs, *shape))
        output_features = self.features(inp)
        n_features = output_features.size()[1]
        return n_features

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        kernel_size = (out.size()[2], out.size()[3])
        out = F.avg_pool2d(out, kernel_size=kernel_size).view(features.size(0), -1)
        out = self.classifier(out)
        return out

