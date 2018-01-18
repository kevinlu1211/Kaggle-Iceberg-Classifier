import torch
import torch.nn as nn
from torch.autograd import Variable

class MyNet(nn.Module):
    def __init__(self, dropout_rates=[0, 0], input_shape=(2, 75, 75)):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        n_features = self._get_conv_output(input_shape)
        self.fc1 = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(dropout_rates[0]),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(512, 1),
            nn.Dropout(dropout_rates[1]),
        )

    def _get_conv_output(self, input_shape):
        bs = 1
        inp = Variable(torch.rand(bs, *input_shape))
        output = self.block1(inp)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        n_features = output.view(1, -1).size()[-1]
        return n_features

    def forward(self,  inp):
        bs = inp.size()[0]
        output = self.block1(inp)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = output.view(bs, -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output

