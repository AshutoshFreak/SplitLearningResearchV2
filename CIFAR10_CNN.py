from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


# model taken from https://shonit2096.medium.com/cnn-on-cifar10-data-set-using-pytorch-34be87e09844
class front(nn.Module):
    def __init__(self):
        super(front, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x
    

class center(nn.Module):
    def __init__(self):
        super(center, self).__init__()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(p=0.05)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)


    def forward(self, x):
        x = self.conv3(x)
        x = self.batch_norm2(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.dropout1(x)
        x = self.conv5(x)
        x = self.batch_norm3(x)
        x = F.relu(x, inplace=True)
        x = self.conv6(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x


class back(nn.Module):
    def __init__(self):
        super(back, self).__init__()
        self.dropout2 = nn.Dropout(p=0.1)
        self.linear1 = nn.Linear(2304, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.dropout3 = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(512, 10)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout2(x)
        # print(f'tensor: {x.size()}')
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        x = self.linear2(x)
        x = F.relu(x, inplace=True)
        x = self.dropout3(x)
        x = self.linear3(x)
        # output = F.log_softmax(x, dim=1)
        return x
