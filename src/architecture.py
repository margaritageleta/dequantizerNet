from calendar import c
import warnings
import torch
from torch import nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

FUNCS = {"tanh":F.tanh, "relu":F.relu, "leakyRelu":F.leaky_relu, "gelu":F.gelu}

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, params):                                                                                                           
        super().__init__()
        self.convTransp1 = nn.ConvTranspose2d(input_channels, input_channels*2, 5, stride=2, padding=2,output_padding=1)
        self.conv1 = nn.Conv2d(input_channels*2,input_channels*2,3,stride=1, padding=1)
        self.conv2 = nn.Conv2d(input_channels*2,output_channels,3,stride=2, padding=1)
        self.F = FUNCS[params["conv_TF"]]
        
    def forward(self, x):
        x = self.convTransp1(x)
        x = self.F(x)
        x = self.conv1(x)
        x = self.F(x)
        x = self.conv2(x)
        return x

class Residual(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return torch.cat([self.module(inputs),inputs], dim=1)

class DequantizerNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.F = FUNCS[params["block_TF"]]

        self.aug_block1 = Residual(ConvBlock(3,3,params))
        self.aug_block2 = Residual(ConvBlock(6,6,params))

        self.red_block1 = ConvBlock(12,9,params)
        self.red_block2 = ConvBlock(9,6,params)
        self.red_block3 = ConvBlock(6,3,params)

    def forward(self, x):
        x = self.aug_block1(x)
        x = self.F(x)
        x = self.aug_block2(x)
        x = self.F(x)
        x = self.red_block1(x)
        x = self.F(x)
        x = self.red_block2(x)
        x = self.F(x)
        x = self.red_block3(x)
        return x
        