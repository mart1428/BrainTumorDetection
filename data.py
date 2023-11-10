import torch
import torch.nn as nn

import torch.optim

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import sys

def get_data_loader_ResNet(batch_size = 128):
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Resize((300,300)), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])     #ResNet input requirements
    ])
    data = ImageFolder(root = './data', transform = transform)
    
    classes = data.classes
    data_index = {}

    for i, d in enumerate(data):
        if data_index.get(d[1]) != None:
            data_index[d[1]].append(i)

        else:
            data_index[d[1]] = [i]

    
    train_index = []
    val_index = []
    test_index = []

    for k, v in data_index.items():
        split1 = int(len(v)* 0.7)
        split2 = int(len(v) * 0.9)

        train_index += v[:split1]
        val_index += v[split1:split2]
        test_index += v[split2:]    

    train_sampler = SubsetRandomSampler(train_index)
    val_sampler = SubsetRandomSampler(val_index)
    test_sampler = SubsetRandomSampler(test_index)

    train_loader = DataLoader(data, batch_size=batch_size, sampler = train_sampler)
    val_loader = DataLoader(data, batch_size=batch_size, sampler = val_sampler)
    test_loader = DataLoader(data, batch_size=batch_size, sampler = test_sampler)


    return classes, train_loader, val_loader, test_loader

def get_data_loader(batch_size = 128):
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Resize((100,100)), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    data = ImageFolder(root = './data', transform = transform)
    
    classes = data.classes
    data_index = {}

    for i, d in enumerate(data):
        if data_index.get(d[1]) != None:
            data_index[d[1]].append(i)

        else:
            data_index[d[1]] = [i]

    
    train_index = []
    val_index = []
    test_index = []

    for k, v in data_index.items():
        split1 = int(len(v)* 0.7)
        split2 = int(len(v) * 0.9)

        train_index += v[:split1]
        val_index += v[split1:split2]
        test_index += v[split2:]    

    train_sampler = SubsetRandomSampler(train_index)
    val_sampler = SubsetRandomSampler(val_index)
    test_sampler = SubsetRandomSampler(test_index)

    train_loader = DataLoader(data, batch_size=batch_size, sampler = train_sampler)
    val_loader = DataLoader(data, batch_size=batch_size, sampler = val_sampler)
    test_loader = DataLoader(data, batch_size=batch_size, sampler = test_sampler)


    return classes, train_loader, val_loader, test_loader