import os
import torch
import sys
from utils import datasets
import pickle
from torch.utils.data import Dataset, random_split

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def split_dataset(dataset: str, client_ids: list, output_dir='data'):
    train_dataset, test_dataset = datasets.load_full_dataset(dataset, output_dir)
    per_client_trainset_size = len(train_dataset)//len(client_ids)
    train_split = [per_client_trainset_size]*len(client_ids)
    # train_split.append(len(train_dataset)-per_client_trainset_size*(len(client_ids)-1))

    per_client_testset_size = len(test_dataset)//len(client_ids)
    test_split = [per_client_testset_size]*len(client_ids)
    # test_split.append(len(test_dataset)-test_batch_size*(num_clients-1))

    train_datasets = list(torch.utils.data.random_split(train_dataset, train_split))
    test_datasets = list(torch.utils.data.random_split(test_dataset, test_split))
    # print(type(train_datasets[0]))
    for i in range(len(client_ids)):
        out_dir = f'{output_dir}/{dataset}/{client_ids[i]}'
        os.makedirs(out_dir + '/train', exist_ok=True)
        os.makedirs(out_dir + '/test', exist_ok=True)
        torch.save(train_datasets[i], out_dir + f'/train/{client_ids[i]}.pt')
        torch.save(test_datasets[i], out_dir + f'/test/{client_ids[i]}.pt')
