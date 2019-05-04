import torch
import torchvision as tv
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import utlis
import random
from PIL import Image

def make_data(label, x_data, y_data, data_count, name, label_count):
    y_data = np.array(y_data)
    idx = np.where(y_data == label)[0]

    y = []
    if name == 'train':
        x = idx[:data_count]
        # y = y_data[idx[:data_count]]
    elif name == 'valid':
        x = idx[data_count:data_count + 500]
    else:
        x = idx[:1000]
    print(len(x))
    return np.asarray(x)

def data_loader_make():
    train_transforms = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = datasets.CIFAR10(root='./cifar10_data/', train=True, transform=train_transforms, download=True)
    valid_set = datasets.CIFAR10(root='./cifar10_data/', train=True, transform=test_transforms, download=False)
    test_set = datasets.CIFAR10(root='./cifar10_data/', train=False, transform=test_transforms, download=False)

    label = [i for i in range(0, 10)]
    data_count = [450, 450, 450, 450, 450, 4500, 4500, 4500, 4500, 4500]
    # data_count = [450, 0, 0, 0, 0, 0, 0, 0, 0, 4500]
    #data_count = [4500, 4500, 4500, 4500, 4500, 4500, 4500, 4500, 4500, 4500]

    train_data_x = []
    validation_data_x = []
    test_data_x = []

    label_count = 0
    for idx, i in enumerate(label):
        print(i, ' data Create start')
        if (data_count[idx] != 0):
            train_x_li = make_data(i, train_set.data, train_set.targets, data_count[i], 'train', label_count)
            validation_x_li = make_data(i, valid_set.data, valid_set.targets, data_count[i],'valid', label_count)
            test_x_li = make_data(i, test_set.data, test_set.targets, data_count[i], 'test', label_count)

            train_data_x.extend(train_x_li)
            validation_data_x.extend(validation_x_li)
            test_data_x.extend(test_x_li)
            print(i, ' data Create finish')
            label_count += 1

    random.shuffle(train_data_x)

    # creat imbalance index data.set
    train_data = torch.utils.data.Subset(train_set, train_data_x)
    validation_data = torch.utils.data.Subset(valid_set, validation_data_x)
    test_data = torch.utils.data.Subset(test_set, test_data_x)
    feature_data = torch.utils.data.Subset(valid_set, train_data_x)
        # create data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=128, shuffle=False, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(dataset=validation_data, batch_size=128, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=100, shuffle=False, num_workers=4)
    feature_train_loader = torch.utils.data.DataLoader(dataset=feature_data, batch_size=128, shuffle=False,num_workers=4)
    print(len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset),
          len(feature_train_loader.dataset))

    return train_loader, valid_loader, test_loader, feature_train_loader