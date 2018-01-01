import torch
non_linearity = {
    "LeakyReLU": torch.nn.LeakyReLU,
    'ReLU': torch.nn.modules.activation.ReLU
}

pooling = {
    "MaxPool2d": torch.nn.MaxPool2d,
    "AvgPool2d": torch.nn.AvgPool2d
}

convolutions = {
    "Conv2d": torch.nn.Conv2d
}

batch_norm = {
    "BatchNorm2d": torch.nn.BatchNorm2d
}

dropout = {
    "Dropout": torch.nn.Dropout
}

