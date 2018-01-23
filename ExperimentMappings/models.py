from torchvision import models
from src.models.densenet import DenseNet
from src.models.densenet_pytorch import DenseNet121PyTorch
from src.models.senet import iceresnet
from src.models.senet import iceresnet_fcn
from src.models.senet import triple_column_iceresnet
from src.models.senet import iceresnet_image_statistics
from src.models.my_net import MyNet
from src.models.my_net import MyNet2
from src.models.fully_convolutional_networks import MyFCN


models = {
    "MyNet": MyNet,
    "MyNet2": MyNet2,
    "MyFCN": MyFCN,
    "DenseNet": DenseNet,
    "DenseNet121PyTorch": DenseNet121PyTorch,
    "DenseNet161PyTorch": models.densenet161,
    "DenseNet169PyTorch": models.densenet169,
    "DenseNet201PyTorch": models.densenet201,
    "IceResNet": iceresnet,
    "IceResNetFCN": iceresnet_fcn,
    "TripleColumnIceResNet": triple_column_iceresnet,
    "IceResNetWithImageStatistics": iceresnet_image_statistics
}


