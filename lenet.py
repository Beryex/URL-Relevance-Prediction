from torch.nn import Module
from torch import nn


class Lenet5(Module):
    def __init__(self, in_channels, num_class):
        super(Lenet5, self).__init__()
        self.fc1 = nn.Linear(in_channels, 6)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(6, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 256)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(256, 120)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(120, 84)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(84, num_class)
        self.relu6 = nn.ReLU()

    def forward(self, x):
        y = self.fc1(x)
        y = self.relu1(y)
        y = self.fc2(y)
        y = self.relu2(y)
        y = self.fc3(y)
        y = self.relu3(y)
        y = self.fc4(y)
        y = self.relu4(y)
        y = self.fc5(y)
        y = self.relu5(y)
        y = self.fc6(y)
        y = self.relu6(y)
        return y