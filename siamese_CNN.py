import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int



class SiameseCNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count



        # convolutional layers:
        # ((conv, batch norm) x 2 , pool) x 4
        # named as conv_(layer number)_(conv number in layer)
        # and batch_norm_(layer number)_(conv number in layer)

        # first conv layer pair:
        
        self.conv_1_0 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.batch_norm_1_0 = nn.BatchNorm2d(64)
        self.initialise_layer(self.conv_1_0)

        self.conv_1_1 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.batch_norm_1_1 = nn.BatchNorm2d(64)
        self.initialise_layer(self.conv_1_1)

        # self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # second conv layer pair:


        self.conv2_0 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.batch_norm2_0 = nn.BatchNorm2d(128)
        self.initialise_layer(self.conv2_0)

        self.conv2_1 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.batch_norm2_1 = nn.BatchNorm2d(128)
        self.initialise_layer(self.conv2_1)

        # self.pool2 = nn.AdaptiveAvgPool2d(56)
        
        # third conv layer pair:

        self.conv3_0 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.batch_norm3_0 = nn.BatchNorm2d(256)
        self.initialise_layer(self.conv3_0)

        self.conv3_1 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.batch_norm3_1 = nn.BatchNorm2d(256)
        self.initialise_layer(self.conv3_1)
        # self.pool3_1 = nn.AdaptiveAvgPool2d(28)

        # fourth conv layer pair:

        self.conv4_0 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.batch_norm4_0 = nn.BatchNorm2d(512)
        self.initialise_layer(self.conv4_0)

        self.conv4_1 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.batch_norm4_1 = nn.BatchNorm2d(512)
        self.initialise_layer(self.conv4_1)
        # self.pool4_1 = nn.AdaptiveAvgPool2d(14)













        # fully connected layers:




    def convForward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)

        x = F.relu(self.batch_norm_1_0(self.conv_1_0(x)))
        x = F.relu(self.batch_norm_1_1(self.conv_1_1(x)))

        print(x.shape)        


        return 


    def forward(self, images: torch.Tensor) -> torch.Tensor:


        # do conv for each image in the pair
        conv_outputs = []
        for image in images:
            x = self.convForward(image)
            conv_outputs.append(x)
        # 
        # 
        # 
        # 
        # concatenate the outputs
        # 
        # 
        # 
        # 
        # do fully connected layers
        


    
        return x
    

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
