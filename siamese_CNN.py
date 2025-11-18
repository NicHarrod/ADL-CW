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
              
        # global avg pooling to reduce spatial dimension from 14 x14 to 1 x 1
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        # first FC layer
        self.fc1 = nn.Sequential (
            nn.Linear (1024, 512),
            nn.ReLU()
        )

        # second FC Layer: 512 -> 3 with dropout (p = 0.5)
        self.fc2 = nn.Sequential (
            nn.Dropout (p = 0.5),
            nn.Linear (512,3)
        )



    def convForward(self, x: torch.Tensor) -> torch.Tensor:
        
  



        return x


    def forward(self, images: torch.Tensor) -> torch.Tensor:


        # do conv for each image in the pair
        # 
        # 
        # 
        # 
        # 

        # concatenate the outputs to give 1x1x1024 - here x and y are results of convolution at layer 7 and 8 
        feat_map_1 = self.gap (x)
        feat_map_2 = self.gap (y)

        # flatten the feature maps to turn (B,512,1,1) into (B,512)
        flat_feat_map_1 = feat_map_1.flatten(start_dim=1)
        flat_feat_map_2 = feat_map_2.flatten(start_dim=1)

        concatenated_features = torch.cat ((flat_feat_map_1, flat_feat_map_2), dim=1)
        print (f"concatenated shape: {concatenated_features.shape}")

        # FC layer 1 
        fc_output = self.fc1 (concatenated_features)

        # FC layer 2
        final_output = self.fc2 (fc_output)

        

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

