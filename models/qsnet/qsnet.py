import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from src.Experiment import ExperimentFactory
from src.ExperimentMappings import layers
import random
from copy import deepcopy

class Net(nn.Module):
    def __init__(self, model_config, input_shape=(3, 75, 75)):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            OrderedDict()
        )

        # Use a hack to calculate the number of features during the flatten layer
        # https://discuss.pytorch.org/t/inferring-shape-via-flatten-operator/138/4

        for i in range(int(model_config["n_blocks"])):
            self.features.add_module(f"network_block_{i}", BasicBlock(model_config, i))

        self.n_features = self._get_conv_output(input_shape)

        self.classifier = nn.Sequential(nn.Linear(self.n_features, 1)
                                        # nn.Linear(2048, 512),
                                        # nn.Linear(512, 1)
                                        )

    @staticmethod
    def initialize_random_config():
        config = dict()
        config["n_blocks"] = 4
        prev_out_channels = int(conv[0]["parameters"]["in_channels"])
        for i in range(int(config['n_blocks'])):

            # Pooling
            config[f"block_{i}.pooling"] = random.sample(pooling, 1)[0]

            # Non-linearity
            config[f"block_{i}.non_linearity"] = random.sample(non_linearity, 1)[0]

            # Conv Layer
            conv_layer_parameters = deepcopy(conv[i]["parameters"])
            config[f"block_{i}.conv"] = conv[i]
            n_in_channels = prev_out_channels

            # Sample out_channels
            n_out_channels = int(np.random.normal(**conv_layer_parameters["out_channels"]))
            conv_layer_parameters["in_channels"] = n_in_channels
            conv_layer_parameters["out_channels"] = n_out_channels
            config[f"block_{i}.conv"].update({"parameters": conv_layer_parameters})

            # Batch Norm
            config[f"block_{i}.batch_norm"] = {"name": "BatchNorm2d",
                                                     "parameters": {"num_features": n_in_channels}}

            # Dropout
            dropout_probs = np.linspace(0, 0.5, 11).tolist()
            config[f"block_{i}.dropout"] = {"name": "Dropout",
                                                  "parameters": {"dropout_rate": random.sample(dropout_probs, 1)[0]}
                                                  }
            prev_out_channels = n_out_channels
        return config

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


class BasicBlock(nn.Sequential):
    def __init__(self, config, index):
        super().__init__()
        self.add_module("batch_norm",
                        ExperimentFactory.create_component(layers.batch_norm
                                                           [config[f"block_{index}.batch_norm"]["name"]],
                                                           config[f"block_{index}.batch_norm"]["parameters"]))
        self.add_module("non_linearity",
                        ExperimentFactory.create_component(layers.non_linearity
                                                           [config[f"block_{index}.non_linearity"]["name"]],
                                                           config[f"block_{index}.non_linearity"]["parameters"]))

        self.add_module("conv", ExperimentFactory.create_component(layers.convolutions
                                                                    [config[f"block_{index}.conv"]["name"]],
                                                                     config[f"block_{index}.conv"]["parameters"]))
        self.add_module("pooling", ExperimentFactory.create_component(layers.pooling
                                                                    [config[f"block_{index}.pooling"]["name"]],
                                                           config[f"block_{index}.pooling"]["parameters"]))
        self.dropout_prob = config[f"block_{index}.dropout"]["parameters"]["dropout_rate"]

    def forward(self, x):
        features = super().forward(x)
        if self.dropout_prob > 0:
            features = F.dropout(features, p=self.dropout_prob, training=self.training)
        return features

