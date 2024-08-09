"""
Implementation of the ResNet model, inspired by the architecture proposed in:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Deep Residual Learning for Image Recognition."
Paper: https://arxiv.org/abs/1512.03385v1

This implementation is based on the PyTorch replication available at:
https://github.com/weiaicunzai/pytorch-cifar100/
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels * BasicBlock.expansion),
            nn.BatchNorm1d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels * BasicBlock.expansion),
                nn.BatchNorm1d(out_channels * BasicBlock.expansion)
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels * BottleNeck.expansion),
            nn.BatchNorm1d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels * BottleNeck.expansion),
                nn.BatchNorm1d(out_channels * BottleNeck.expansion)
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, in_channels=3, num_class=100):
        super().__init__()

        self.block_in_channels = 64

        self.linear1 = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.linear2_x = self._make_layer(block, 64, num_block[0], 1)
        self.linear3_x = self._make_layer(block, 128, num_block[1], 2)
        self.linear4_x = self._make_layer(block, 256, num_block[2], 2)
        self.linear5_x = self._make_layer(block, 512, num_block[3], 2)
        self.fc = nn.Linear(512 * block.expansion, num_class)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.block_in_channels, out_channels, stride))
            self.block_in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.linear1(x)
        output = self.linear2_x(output)
        output = self.linear3_x(output)
        output = self.linear4_x(output)
        output = self.linear5_x(output)
        output = self.fc(output)

        return output

def resnet18(in_channels: int, num_class: int):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels, num_class)

def resnet34(in_channels: int, num_class: int):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], in_channels, num_class)

def resnet50(in_channels: int, num_class: int):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], in_channels, num_class)

def resnet101(in_channels: int, num_class: int):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3], in_channels, num_class)

def resnet152(in_channels: int, num_class: int):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3], in_channels, num_class)