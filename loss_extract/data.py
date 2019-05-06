import torch
import torchvision as tv
import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from tensorflow.python.keras.datasets.cifar10 import load_data
import os

def data_set(batch_size,save_path,data):
    #The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors of normalized range [-1, 1].
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]

    train_transforms = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])
    if data == 'cifar100':
        train_set = datasets.CIFAR100(root='./cifar100_data/', train=True, transform=train_transforms, download=True)
        valid_set = datasets.CIFAR100(root='./cifar100_data/', train=True, transform=test_transforms, download=False)
        test_set = datasets.CIFAR100(root='./cifar100_data/', train=False, transform=test_transforms, download=False)
    elif data == 'cifar10':
        train_set = datasets.CIFAR10(root='./cifar10_data/', train=True, transform=train_transforms, download=True)
        valid_set = datasets.CIFAR10(root='./cifar10_data/', train=True, transform=test_transforms, download=False)
        test_set = datasets.CIFAR10(root='./cifar10_data/', train=False, transform=test_transforms, download=False)

    indices = torch.randperm(len(train_set))
    train_indices = indices

    train = torch.utils.data.Subset(train_set, train_indices)
    valid = torch.utils.data.Subset(valid_set, train_indices[45000:])
    train_feature = torch.utils.data.Subset(valid_set, train_indices)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,shuffle=False, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size,shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, num_workers=4)
    train_feature_loader = torch.utils.data.DataLoader(train_feature, batch_size=128,shuffle=False, num_workers=4)

    torch.save(train_indices, save_path + 'train_indices.pth')
    torch.save(train_loader, save_path + 'train_loader.pth')
    torch.save(train_feature_loader, save_path + 'train_feature_loader.pth')

    print(len(train_loader.dataset),len(valid_loader.dataset),len(test_loader.dataset))
    print(len(train_loader),len(valid_loader),len(test_loader))
    return train_loader, valid_loader, test_loader, train_feature_loader
