import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple
from dataloader import ProgressionDataset

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

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

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

        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
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
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

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
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))













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
        
        x = F.relu(self.batch_norm_1_0(self.conv_1_0(x)))
        x = F.relu(self.batch_norm_1_1(self.conv_1_1(x))) 
        x= self.pool1(x)  
        

        x= F.relu(self.batch_norm2_0(self.conv2_0(x)))
        x= F.relu(self.batch_norm2_1(self.conv2_1(x)))
        x= self.pool2(x)   
        

        x= F.relu(self.batch_norm3_0(self.conv3_0(x)))
        x= F.relu(self.batch_norm3_1(self.conv3_1(x)))
        x= self.pool3(x)
        

        x= F.relu(self.batch_norm4_0(self.conv4_0(x)))
        x= F.relu(self.batch_norm4_1(self.conv4_1(x)))
        x= self.pool4(x)
        

        return x



    def forward(self, images: torch.Tensor) -> torch.Tensor:


        # do conv for each image in the pair
        conv_outputs = []
        # [anchor,comparator]
        for image in images:
            x = self.convForward(image)
            conv_outputs.append(x)
        
        

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
        


    
        return 
    

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


def test_CNN():
    model = SiameseCNN(height=512, width=512, channels=3, class_count=3)
    # generate 2 batches of 1 image each
    sample_inputs = [torch.randn(1, 3, 512, 512), torch.randn(1, 3, 512, 512)]
    # resize inputs to 224x224 using transforms
    resize = transforms.Resize((224, 224))
    sample_inputs = [resize(img) for img in sample_inputs]
    output = model(sample_inputs)


def main():
    transform = transforms.ToTensor()
    """
        Initialize the ProgressionDataset.

        Parameters
        ----------
        root_dir : str
            Root directory containing recipe folders or image pairs.
        transform : callable, optional
            Optional torchvision transform for image preprocessing.
        mode : str, default='train'
            Operation mode: 'train', 'val', or 'test'.
        recipe_ids_list : list of str, optional
            List of recipe folder names (required for 'train' mode).
        epoch_size : int, optional
            Number of samples per epoch (required for 'train' mode).
        label_file : str, optional
            Path to text file containing image pair indices and labels 
            (required for 'val'/'test' mode).
        """
    data_loader = ProgressionDataset("dataset", transform=transform, mode="train")

if __name__ == "__main__":
    test_CNN()