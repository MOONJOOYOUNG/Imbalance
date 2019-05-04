import torch
import torchvision as tv
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class CIFAR100_Dataset(Dataset):
  def __init__(self,x,y):
    self.x = x
    self.y = y
  def __len__(self):
    return len(self.x)
  def __getitem__(self,idx):
    x = self.x[idx]
    y = self.y[idx]
    return torch.tensor(x,dtype=torch.float),torch.tensor(y,dtype=torch.long)

def next_batch(start_index, finish_index, data_x):
    x = data_x[start_index:finish_index]
    return x

def make_data(suffle_idx, x_data, y_data,data_count, loss_idx,name,label_count):
    x_data = np.array(x_data)
    loss_idx = np.asarray(loss_idx)
    x_data = x_data[suffle_idx]

    #idx = np.where(y_data == label)[0]
    y = []
    if name == 'train':
        x = x_data[loss_idx[:4000]]
        for i in range(4000):
            y.append(label_count)
        print('train data : {0}'.format(len(loss_idx[:4000])))
    elif name == 'valid':
        x = x_data[loss_idx[4000:4500]]
        for i in range(500):
            y.append(label_count)
        print('valid data : {0}'.format(len(loss_idx[4000:4500])))
    else:
        x = x_data[loss_idx[4500:5000]]
        for i in range(500):
            y.append(label_count)
        print('test data : {0}'.format(len(loss_idx[4500:5000])))

    return np.asarray(x), np.asarray(y)

def data_loader_make(data_sets):
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

    train_set = datasets.CIFAR100(root='./cifar100_data/', train=True, transform=train_transforms, download=True)

    label = [i for i in range(0,10)]
    data_count = [5000,5000,5000,5000,5000,5000,5000,5000,5000,5000]
    #data_count = [450, 450, 450, 450, 450, 4500, 4500, 4500, 4500, 4500]
    shuffle = True

    train_data_x = []
    train_data_y = []
    validation_data_x = []
    validation_data_y = []
    test_data_x = []
    test_data_y = []
    if data_sets == 1:
        loss_csv = pd.read_csv('./data_index_0.1.csv', index_col=0)
    elif data_sets == 2:
        loss_csv = pd.read_csv('./data_index_0.01.csv', index_col=0)
    elif data_sets == 3:
        loss_csv = pd.read_csv('./data_index_0.05.csv', index_col=0)
    elif data_sets == 4:
        loss_csv = pd.read_csv('./data_index_0.005.csv', index_col=0)
    elif data_sets == 5:
        loss_csv = pd.read_csv('./train_data_index_0.05.csv', index_col=0)
    elif data_sets == 6:
        loss_csv = pd.read_csv('./train_data_index_0.005.csv', index_col=0)
    elif data_sets == 7:
        loss_csv = pd.read_csv('./train_data_index_0.003.csv', index_col=0)
    elif data_sets == 8:
        loss_csv = pd.read_csv('./train_data_index_0.001.csv', index_col=0)

    suffle_idx = torch.load('./train_indices.pth')
    suffle_idx = np.asarray(suffle_idx)

    start_index = 0
    finish_index = 5000
    label_count = 0
    for idx, i in enumerate(label):
        print(i, ' data Create start')
        loss_idx = next_batch(start_index, finish_index, loss_csv)

        train_x_li, train_y_li = make_data(suffle_idx, train_set.data, train_set.targets,data_count[i] , loss_idx, 'train',label_count)
        validation_x_li, validation_y_li = make_data(suffle_idx, train_set.data, train_set.targets, data_count[i], loss_idx, 'valid',label_count)
        test_x_li, test_y_li = make_data(suffle_idx, train_set.data, train_set.targets, data_count[i], loss_idx, 'test',label_count)

        train_data_x.extend(train_x_li)
        train_data_y.extend(train_y_li)
        validation_data_x.extend(validation_x_li)
        validation_data_y.extend(validation_y_li)
        test_data_x.extend(test_x_li)
        test_data_y.extend(test_y_li)

        start_index += 5000
        finish_index += 5000
        label_count += 1
        print(i,' data Create finish')

    train_x = [a for i in train_data_x for a in i]
    train_y = y_data_preprocessing(train_data_y)

    validation_x = [a for i in validation_data_x for a in i]
    validation_y = y_data_preprocessing(validation_data_y)

    test_x = [a for i in test_data_x for a in i]
    test_y = y_data_preprocessing(test_data_y)

    print(len(train_x),len(train_y),len(validation_x),len(validation_y),len(test_x),len(test_y))

    train_data = CIFAR100_Dataset(train_x,train_y)
    validation_data = CIFAR100_Dataset(validation_x,validation_y)
    test_data = CIFAR100_Dataset(test_x,test_y)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=128,shuffle=True,num_workers=4)
    valid_loader = torch.utils.data.DataLoader(dataset=validation_data, batch_size=128,shuffle=True,num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=100,shuffle=True,num_workers=4)

    print(len(train_loader.dataset),len(valid_loader.dataset),len(test_loader.dataset))

    return train_loader, valid_loader, test_loader

def y_data_preprocessing(y_test):
    idx = 0

    one_hot_y = []

    for i in y_test:
        one_hot_y.append(np.asscalar(y_test[idx]))
        idx += 1

    one_hot = []
    one_hot.extend(one_hot_y)

    return one_hot

def data_augmentation(batch,mode):
    if mode == 'train':
        batch = np.asarray(batch)
        batch = _random_crop(batch, [32, 32], 4)
        batch = _random_flip_updown(batch)
        batch = _data_scaling(batch)
        batch = _data_normalization(batch)
    elif mode == 'test':
        batch = np.asarray(batch)
        batch = _data_scaling(batch)
        batch = _data_normalization(batch)

    return  torch.tensor(batch,dtype=torch.float)

def _data_scaling(x_data):
    x_data = x_data.astype('float32')
    x_data[:, :, :, 0] = (x_data[:, :, :, 0] / 255)
    x_data[:, :, :, 1] = (x_data[:, :, :, 1] / 255)
    x_data[:, :, :, 2] = (x_data[:, :, :, 2] / 255)

    return x_data

def _data_normalization(x_data):
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    x_data = x_data.astype('float32')
    x_data[:, :, :, 0] = (x_data[:, :, :, 0] - mean[0]) / std[0]
    x_data[:, :, :, 1] = (x_data[:, :, :, 1] - mean[1]) / std[1]
    x_data[:, :, :, 2] = (x_data[:, :, :, 2] - mean[2]) / std[2]

    return x_data

def _random_crop(batch, crop_shape, padding=None):

    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                                    nw:nw + crop_shape[1]]

    return new_batch

def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

def _random_flip_updown(batch):
    # for i in range(len(batch)):
    #     if bool(random.getrandbits(1)):
    batch = np.flipud(batch)
    return batch