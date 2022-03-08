import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models

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
    if dataset == 'mnist':
        train_dataset, test_dataset = MNIST(dataset_path)

    if dataset == 'cifar10':
        train_dataset, test_dataset = CIFAR10(dataset_path)
    return train_dataset, test_dataset
    

# if __name__ == "__main__":
#     train_dataset, test_dataset = MNIST('../data')
#     print(type(train_dataset))