from torchvision import models
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F

class DenseNet121PyTorch(nn.Module):
    def __init__(self, pretrained, n_classes, input_shape, drop_rate):
        super().__init__()
        self.features = models.densenet121(pretrained=pretrained).features
        self.drop_rate = drop_rate
        n_features = self._get_model_features(input_shape)
        self.classifier = nn.Linear(n_features, n_classes)

    def get_parameters(self, optimizer_parameters, fine_tuning_parameters):
        all_optimizer_parameters = dict()
        all_optimizer_parameters["params"] = [
            {"params": self.features.parameters(), **fine_tuning_parameters},
            {"params": self.classifier.parameters()}
        ]
        all_optimizer_parameters.update(**optimizer_parameters)
        return all_optimizer_parameters

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
        out = F.dropout(self.classifier(out), p=self.drop_rate)
        return out

