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
        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(5, 5),
            padding=(2, 2),
        )
        self.initialise_layer(self.conv1)









        # fully connected layers:




    def convForward(self, x: torch.Tensor) -> torch.Tensor:
        


        return x


    def forward(self, images: torch.Tensor) -> torch.Tensor:


        # do conv for each image in the pair
        # 
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
