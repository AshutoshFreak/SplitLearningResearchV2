import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm.notebook import tqdm
import glob
import os
from PIL import Image

# source: https://www.kaggle.com/saranga7/1000fundus-pytorch-transferlearning
class custom_1000Fundus(Dataset):
    def __init__(self,root_dir,transform=None):
        self.data = []
        self.transform = transform

        for img_path in tqdm(glob.glob(root_dir+"/*/**")):
            class_name = img_path.split("/")[-2]
            self.data.append([img_path,class_name])
 
        self.class_map = {}
        for index,item in enumerate(os.listdir(root_dir)):
             self.class_map[item] = index
        print(f"Total Classes:{len(self.class_map)}")


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        img_path,class_name = self.data[idx]
        img = Image.open(img_path)
        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)

        if self.transform:
            img = self.transform(img)

        return img, class_id


def create_transforms(normalize=False,mean=[0,0,0],std=[1,1,1]):
    if normalize:
        my_transforms=transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.ColorJitter(brightness=0.3,saturation=0.5,contrast=0.7,),
            # transforms.RandomRotation(degrees=33),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
        ])
       
    else:
         my_transforms=transforms.Compose([
            transforms.Resize((512,512)),
            # transforms.ColorJitter(brightness=0.3,saturation=0.5,contrast=0.7,p=0.57),
            # transforms.RandomRotation(degrees=33),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    return my_transforms


def get_mean_std(loader):
    #var=E[x^2]-(E[x])^2
    channels_sum, channels_squared_sum,num_batches=0,0,0
    for data,_ in tqdm(loader):
        channels_sum+=torch.mean(data,dim=[0,2,3]) # we dont want to a singuar mean for al 3 channels (in case of RGB)
        channels_squared_sum+=torch.mean(data**2,dim=[0,2,3])
        num_batches+=1
    mean=channels_sum/num_batches
    std=(channels_squared_sum/num_batches-mean**2)**0.5
    
    return mean, std


def Fundus1000(root_dir, split=0.8):
    my_transforms = create_transforms(normalize=False)
    dataset = custom_1000Fundus(root_dir, my_transforms)
    train_set, val_set = torch.utils.data.random_split(dataset, [800, 200], generator=torch.Generator().manual_seed(7))
    # train_set, val_set = torch.utils.data.random_split(dataset, [split*len(dataset), (1-split)*len(dataset)])
    BS = 8
    train_loader = DataLoader(train_set, batch_size=BS, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BS, shuffle=True)
    mean, std = get_mean_std(train_loader)

    print(mean, std)

    my_transforms = create_transforms(normalize=True, mean=mean,std = std)
    dataset = custom_1000Fundus(root_dir, my_transforms)
    print(len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [800, 200], generator=torch.Generator().manual_seed(7))
    # train_set, val_set = torch.utils.data.random_split(dataset, [split*len(dataset), (1-split)*len(dataset)])
    return train_set, val_set


def MNIST(path):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_dataset = torchvision.datasets.MNIST(root=path,
                                            train=True,
                                            transform=transform,download=True)

    test_dataset = torchvision.datasets.MNIST(root=path,
                                            train=False,
                                            transform=transforms.ToTensor())
    return train_dataset, test_dataset


def CIFAR10(path):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(root=path,
                                            train=True,
                                            transform=transform,download=True)

    test_dataset = torchvision.datasets.CIFAR10(root=path,
                                            train=False,
                                            transform=transform)
    return train_dataset, test_dataset


def load_full_dataset(dataset, dataset_path):
    train_dataset, test_dataset = torch.tensor([0]), torch.tensor([0])
    if dataset == 'MNIST':
        return MNIST(dataset_path)

    if dataset == 'CIFAR10':
        return CIFAR10(dataset_path)
    
    if dataset == '1000Fundus':
        dataset_path = os.path.join(dataset_path, '1000Fundus/1000images/1000images')
        return Fundus1000(dataset_path)
    return train_dataset, test_dataset


# if __name__ == "__main__":
#     train_dataset, test_dataset = MNIST('../data')
#     print(type(train_dataset))