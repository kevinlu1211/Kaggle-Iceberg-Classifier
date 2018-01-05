import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict
from src.Experiment import ExperimentFactory
from src.ExperimentMappings import layers

# class DenseLayer(nn.Sequential):
#
#     def __init__(self, n_input_features, k, bn_size, drop_prob, bias=False):
#         super(DenseLayer, self).__init__()
#         self.drop_prob = drop_prob
#         self.add_module('bn1', nn.BatchNorm2d(n_input_features))
#         self.add_module('relu1', nn.ReLU())
#         self.add_module('conv.1', nn.Conv2d(n_input_features, k * bn_size,
#                                             kernel_size=1, stride=1, bias=bias))
#         self.add_module('bn2', nn.BatchNorm2d(k * bn_size))
#         self.add_module('relu2', nn.ReLU())
#         self.add_module('conv2', nn.Conv2d(k * bn_size, k,
#                                            kernel_size=3, stride=1, padding=1, bias=bias))
#
#     def forward(self, inp):
#         output = super(DenseLayer, self).forward(inp)
#         if self.drop_prob > 0:
#             output = F.dropout(output, p=self.drop_prob, training=self.training)
#         return torch.cat([inp, output], 1)
#
#
# class DenseBlock(nn.Sequential):
#     def __init__(self, n_layers, n_init_features, k, bn_size, drop_prob, bias=False):
#         super(DenseBlock, self).__init__()
#         for i in range(n_layers):
#             layer = DenseLayer(n_init_features + i * k, k,
#                                bn_size, drop_prob, bias)
#             self.add_module(f'denselayer{i}', layer)
#
#
# class TransitionLayer(nn.Sequential):
#     def __init__(self, n_input_features, n_output_features):
#         super(TransitionLayer, self).__init__()
#         self.add_module('bn', nn.BatchNorm2d(n_input_features))
#         self.add_module('relu', nn.ReLU())
#         self.add_module('conv', nn.Conv2d(n_input_features, n_output_features,
#                                           kernel_size=1, stride=1, padding=(1, 1)))
#         self.add_module('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
#
#
# class DenseNet(nn.Module):
#     def __init__(self, growth_rate=16, block_config=(6, 12, 24, 16), n_init_features=32,
#                  bn_size=4, dropout_rate=0, n_classes=1, input_shape=(3, 75, 75)):
#         super(DenseNet, self).__init__()
#
#         self.features = nn.Sequential(OrderedDict([
#             ('conv0', nn.Conv2d(3, n_init_features, kernel_size=7, stride=2, bias=False)),
#             ('norm0', nn.BatchNorm2d(n_init_features)),
#             ('relu0', nn.ReLU(inplace=True)),
#         ]))
#         n_features = n_init_features
#         for i, n_layers in enumerate(block_config):
#             block = DenseBlock(n_layers, n_features, growth_rate, bn_size, dropout_rate)
#             self.features.add_module(f'denseblock{i+1}', block)
#             n_features = n_features + growth_rate * n_layers
#             if i != len(block_config) - 1:
#                 trans = TransitionLayer(n_input_features=n_features, n_output_features=n_features // 2)
#                 self.features.add_module(f'transition{i+1}', trans)
#                 n_features = n_features // 2
#
#         self.features.add_module('norm5', nn.BatchNorm2d(n_features))
#         n_fc = self._get_conv_output(input_shape)
#         self.classifier = nn.Linear(n_fc, n_classes)
#
#     def _get_conv_output(self, shape):
#         bs = 1
#         inp = Variable(torch.rand(bs, *shape))
#         output_features = self.features(inp)
#         n_features = output_features.data.view(bs, -1).size(1)
#         return n_features
#
#     def forward(self, inp):
#         features = self.features(inp)
#         output = features.view(features.size(0), -1)
#         output = self.classifier(output)
#         return output


class InitialBlock(nn.Sequential):

    def __init__(self, config):
        super().__init__()
        self.add_module("conv",
                        ExperimentFactory.create_component(
                            layers.convolutions[config["initial_block.conv_0"]["name"]],
                            config["initial_block.conv_0"]["parameters"]))
        self.add_module("batch_norm",
                         ExperimentFactory.create_component(
                            layers.batch_norm[config["initial_block.batch_norm_0"]["name"]],
                            config["initial_block.batch_norm_0"]["parameters"]))
        self.add_module("non_linearity",
                        ExperimentFactory.create_component(
                            layers.non_linearity[config["initial_block.non_linearity_0"]["name"]],
                            config["initial_block.non_linearity_0"]["parameters"]))
        # self.add_module("pooling",
        #                 ExperimentFactory.create_component(
        #                     layers.pooling[config["initial_block.pooling_0"]["name"]],
        #                     config["initial_block.pooling_0"]["parameters"]))


class DenseLayer(nn.Sequential):
    def __init__(self, config, block_i, layer_i):
        super().__init__()
        prefix = f"denseblock_{block_i}.layer_{layer_i}"

        self.add_module("batch_norm_0",
                        ExperimentFactory.create_component(
                            layers.batch_norm[config[f"{prefix}.batch_norm_0"]["name"]],
                            config[f"{prefix}.batch_norm_0"]["parameters"]))
        self.add_module("non_linearity_0",
                        ExperimentFactory.create_component(
                            layers.non_linearity[config[f"{prefix}.non_linearity_0"]["name"]],
                            config[f"{prefix}.non_linearity_0"]["parameters"]))
        self.add_module("conv_0",
                        ExperimentFactory.create_component(
                            layers.convolutions[config[f"{prefix}.conv_0"]["name"]],
                            config[f"{prefix}.conv_0"]["parameters"]))
        # self.add_module("dropout_0",
        #                 ExperimentFactory.create_component(
        #                     layers.dropout[config[f"{prefix}.dropout_0"]["name"]],
        #                     config[f"{prefix}.dropout_0"]["parameters"]))
        self.add_module("batch_norm_1",
                        ExperimentFactory.create_component(
                            layers.batch_norm[config[f"{prefix}.batch_norm_1"]["name"]],
                            config[f"{prefix}.batch_norm_1"]["parameters"]))
        self.add_module("non_linearity_1",
                        ExperimentFactory.create_component(
                            layers.non_linearity[config[f"{prefix}.non_linearity_1"]["name"]],
                            config[f"{prefix}.non_linearity_1"]["parameters"]))
        self.add_module("conv_1",
                        ExperimentFactory.create_component(
                            layers.convolutions[config[f"{prefix}.conv_1"]["name"]],
                            config[f"{prefix}.conv_1"]["parameters"]))
        # self.add_module("dropout_1",
        #                 ExperimentFactory.create_component(
        #                     layers.dropout[config[f"{prefix}.dropout_1"]["name"]],
        #                     config[f"{prefix}.dropout_1"]["parameters"]))

    def forward(self, inp):
        output = super(DenseLayer, self).forward(inp)
        return torch.cat([inp, output], 1)


class DenseBlock(nn.Sequential):
    def __init__(self, config, block_i, n_layers):
        super().__init__()
        for layer_i in range(n_layers):
            layer = DenseLayer(config, block_i, layer_i)
            self.add_module(f"layer_{layer_i}", layer)


class TransitionLayer(nn.Sequential):
    def __init__(self, config, block_i):
        super().__init__()
        prefix = f"transition_{block_i}"
        self.add_module("batch_norm_0",
                        ExperimentFactory.create_component(
                            layers.batch_norm[config[f"{prefix}.batch_norm_0"]["name"]],
                            config[f"{prefix}.batch_norm_0"]["parameters"]))
        self.add_module("non_linearity_0",
                        ExperimentFactory.create_component(
                            layers.non_linearity[config[f"{prefix}.non_linearity_0"]["name"]],
                            config[f"{prefix}.non_linearity_0"]["parameters"]))
        self.add_module("conv_0",
                        ExperimentFactory.create_component(
                            layers.convolutions[config[f"{prefix}.conv_0"]["name"]],
                            config[f"{prefix}.conv_0"]["parameters"]))
        self.add_module("pooling_0",
                        ExperimentFactory.create_component(
                            layers.pooling[config[f"{prefix}.pooling_0"]["name"]],
                            config[f"{prefix}.pooling_0"]["parameters"]))
        self.add_module("dropout_0",
                        ExperimentFactory.create_component(
                            layers.dropout[config[f"{prefix}.dropout_0"]["name"]],
                            config[f"{prefix}.dropout_0"]["parameters"]))

class DenseNet(nn.Module):
    def __init__(self, model_config, input_shape=(3, 75, 75)):
        super().__init__()
        self.features = nn.Sequential(
            OrderedDict()
        )
        # self.features.add_module("conv",
        #                 ExperimentFactory.create_component(
        #                     layers.convolutions[model_config["initial_block.conv_0"]["name"]],
        #                     model_config["initial_block.conv_0"]["parameters"]))
        # self.features.add_module("batch_norm",
        #                 ExperimentFactory.create_component(
        #                     layers.batch_norm[model_config["initial_block.batch_norm_0"]["name"]],
        #                     model_config["initial_block.batch_norm_0"]["parameters"]))
        # self.features.add_module("non_linearity",
        #                 ExperimentFactory.create_component(
        #                     layers.non_linearity[model_config["initial_block.non_linearity_0"]["name"]],
        #                     model_config["initial_block.non_linearity_0"]["parameters"]))
        self.features.add_module("initial_block", InitialBlock(model_config))
        n_features = model_config['n_initial_features']
        for block_i, n_layers in enumerate(model_config['layers_per_block']):
            block = DenseBlock(model_config, block_i, n_layers)
            self.features.add_module(f"denseblock_{block_i}", block)
            n_features += model_config['growth_rate'] * n_layers
            if block_i != len(model_config['layers_per_block']) - 1:
                trans = TransitionLayer(model_config, block_i)
                self.features.add_module(f"transition{block_i}", trans)
                n_features = n_features // 2
        self.final_bn = nn.BatchNorm2d(n_features)
        self.classifier = nn.Linear(n_features, 1)

    def _get_conv_output(self, shape):
        bs = 1
        inp = Variable(torch.rand(bs, *shape))
        output_features = self.features(inp)
        n_features = output_features.data.view(bs, -1).size(1)
        return n_features

    def forward(self, inp):
        features = self.features(inp)
        kernel_size = (features.size()[2], features.size()[3])
        output = F.avg_pool2d(F.relu(self.final_bn(features)), kernel_size=kernel_size)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output



