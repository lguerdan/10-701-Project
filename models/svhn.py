from typing import Tuple, Optional, List

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch import Tensor

"""Adapted from Exploring-svhn-using-deep-neural-network, Medium article"""

class Net(nn.Module):
    """Feedfoward neural network with 6 hidden layer"""
    def __init__(self):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(32 * 32, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 128)
        self.linear5 = nn.Linear(128, 64)
        self.linear6 = nn.Linear(64, 32)
        # output layer
        self.linear7 = nn.Linear(32, 10)
        
    def forward(self, xb):
        # Flatten the image tensors
        # xb = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        out = F.relu(out)
        out = self.linear5(out)
        out = F.relu(out)
        out = self.linear6(out)
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear7(out)
        return out