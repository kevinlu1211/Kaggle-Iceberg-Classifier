from torchvision import models
from src.models.densenet import DenseNet
from src.models.densenet_pytorch import DenseNet121PyTorch

models = {
    "DenseNet": DenseNet,
    "DenseNet121PyTorch": DenseNet121PyTorch,
    "DenseNet161PyTorch": models.densenet161,
    "DenseNet169PyTorch": models.densenet169,
    "DenseNet201PyTorch": models.densenet201
}


