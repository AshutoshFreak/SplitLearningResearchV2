from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class front(nn.Module):
    def __init__(self):
        super(front, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        return x
    

class center(nn.Module):
    def __init__(self):
        super(center, self).__init__()
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x


class back(nn.Module):
    def __init__(self):
        super(back, self).__init__()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
