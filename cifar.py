# Importing Dependencies

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# Defining model
arch = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGGNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(arch)
        self.fcs = nn.Sequential(
            nn.Linear(in_features=512*1*1, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # print(x.shape)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, arch):
        layers = []
        in_channels = self.in_channels
        
        for x in arch:            
            
            if type(x) == int:

                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                        nn.BatchNorm2d(x), 
                        nn.ReLU(),
                        ]

                in_channels = x
            
            elif x =='M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)

# Hyperparameters and settings

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 16
EPOCHS = 10

train_data = CIFAR10(root=".", train=True, 
                    transform=transforms.Compose([transforms.ToTensor()]), download=True)

# print(len(train_data))
val_data = CIFAR10(root=".", train=False,
                    transform=transforms.Compose([transforms.ToTensor()]), download=True)
# print(len(val_data))


train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=8)
val_loader = DataLoader(val_data, batch_size=VAL_BATCH_SIZE, shuffle=True, num_workers=8)
# print(len(train_loader))
# print(len(val_loader))


num_train_batches = int(len(train_data)/TRAIN_BATCH_SIZE) 
num_val_batches = int(len(val_data)/VAL_BATCH_SIZE)

# Training and Val Loop

model = VGGNet(3, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
# optim = torch.optim.Adam(model.parameters(), lr=0.01)

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

# save_path = os.path.join(r"trained_models", f'{datetime.now().strftime("%m%d_%H%M%S")}.pth')

def train_val():
    
    for epoch in range(1, EPOCHS+1):
        print(f"Epoch: {epoch}/{EPOCHS}", end='\t')
        model.train()
        
        running_loss = 0
        total = 0
        correct = 0
        for data in train_loader:
            image, target = data[0], data[1]
            image, target = image.to(device), target.to(device) 
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, target) 
            running_loss += loss.item()
            
            _, pred = torch.max(output, dim=1)
            total += target.size(0)
            correct += torch.sum(pred == target).item()
            
            loss.backward()
            optimizer.step()
        print(f"Training Loss: {running_loss/len(train_loader):.3f}\tTraining Acc: {correct/total}", end='\t')
        
        save_path = os.path.join(r"trained_models", f'{datetime.now().strftime("%m%d_%H%M%S")}_{epoch}.pth')

        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_path)
            with torch.no_grad():
                
                model.eval()
                running_val_loss = 0
                total = 0
                correct = 0
                for data in val_loader:
                    image, target = data[0], data[1]
                    image, target = image.to(device), target.to(device) 
                    output = model(image)
                    val_loss = criterion(output, target)
                    running_val_loss += val_loss
                    _, pred = torch.max(output, dim=1)
                    correct += torch.sum(pred == target).item()
                    total += target.size(0)
                running_val_loss = running_val_loss/len(val_loader)
                print(f"Val Loss: {running_val_loss:.3f}\tVal Acc: {correct/total}")
                
                # scheduler.step(running_val_loss)


train_val()