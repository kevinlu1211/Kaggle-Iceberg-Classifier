from torchvision import models
from src.models.densenet import DenseNet
from src.models.densenet_pytorch import DenseNet121PyTorch
from src.models.senet import iceresnet
from src.models.senet import triple_column_iceresnet
from src.models.my_net import MyNet

models = {
    "MyNet": MyNet,
    "DenseNet": DenseNet,
    "DenseNet121PyTorch": DenseNet121PyTorch,
    "DenseNet161PyTorch": models.densenet161,
    "DenseNet169PyTorch": models.densenet169,
    "DenseNet201PyTorch": models.densenet201,
    "IceResNet": iceresnet,
    "TripleColumnIceResNet": triple_column_iceresnet
}


