import  torch
import os
import sys
import time
import torch.nn as nn
import torch.nn.init as init
import matplotlib
matplotlib.use('agg')
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

def save_loss_acc(name, loss, acc, save_path):
    df_loss = pd.DataFrame(loss)
    df_loss.to_csv(save_path + '{0}_loss.csv'.format(name), encoding='ms949')
    df_acc = pd.DataFrame(acc)
    df_acc.to_csv(save_path + '{0}_acc.csv'.format(name), encoding='ms949')

def save_data(name, data, save_path):
    df_data = pd.DataFrame(data)
    df_data.to_csv(save_path + '{0}_data.csv'.format(name), encoding='ms949')

def draw_test_curve(test_acc_path,test_loss_path,save_path):
    f = plt.figure(figsize=(8, 8))
    test_acc = pd.read_csv(save_path + test_acc_path, names=['idx', 'acc'])
    test_loss = pd.read_csv(save_path + test_loss_path, names=['idx', 'loss'])

    test_acc['idx'] += 1

    visual1_x = test_acc['idx']
    visual1_y = test_acc['acc']

    visual2_x = test_loss['idx']
    visual2_y = test_loss['loss']

    plt.plot(visual1_x, visual1_y, label='test_acc')
    plt.plot(visual2_x, visual2_y, label='test_loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss * test')

    plt.legend()
    plt.title('test acc loss curve')
    plt.savefig(save_path + 'test acc loss curve.png')
    plt.close(f)
    print("Save Loss curve")


def draw_test_curve(test_acc_path,test_loss_path,save_path):
    f = plt.figure(figsize=(8, 8))
    test_acc = pd.read_csv(save_path + test_acc_path, names=['idx', 'acc'])
    test_loss = pd.read_csv(save_path + test_loss_path, names=['idx', 'loss'])

    test_acc['idx'] += 1

    visual1_x = test_acc['idx']
    visual1_y = test_acc['acc']

    visual2_x = test_loss['idx']
    visual2_y = test_loss['loss']

    plt.plot(visual1_x, visual1_y, label='test_acc')
    plt.plot(visual2_x, visual2_y, label='test_loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss * test')

    plt.legend()
    plt.title('test acc loss curve')
    plt.savefig(save_path + 'test acc loss curve.png')
    plt.close(f)
    print("Save Loss curve")

def draw_curve(train_path, vali_path, check,save_path):
    if (check == 'acc'):
        f = plt.figure(figsize=(8, 8))
        train_acc = pd.read_csv(save_path + train_path, names=['idx', 'acc'])
        vail_acc = pd.read_csv(save_path + vali_path, names=['idx', 'acc'])

        train_acc['idx'] += 1

        visual1_x = train_acc['idx']
        visual1_y = train_acc['acc']

        visual2_x = vail_acc['idx']
        visual2_y = vail_acc['acc']

        plt.plot(visual1_x, visual1_y, label='train_acc')
        plt.plot(visual2_x, visual2_y, label='vali_acc')
        plt.xlabel('epoch')
        plt.ylabel('Acc')

        plt.legend()
        plt.title('Accuracy Curve')
        f.savefig(save_path + 'Accuracy curve.png')
        plt.close(f)
        print("Save Accuracy curve")
    elif (check == 'loss'):
        f = plt.figure(figsize=(8, 8))
        train_acc = pd.read_csv(save_path + train_path, names=['idx', 'loss'])
        vail_acc = pd.read_csv(save_path + vali_path, names=['idx', 'loss'])

        train_acc['idx'] += 1

        visual1_x = train_acc['idx']
        visual1_y = train_acc['loss']

        visual2_x = vail_acc['idx']
        visual2_y = vail_acc['loss']

        plt.plot(visual1_x, visual1_y, label='train_loss')
        plt.plot(visual2_x, visual2_y, label='vali_loss')
        plt.xlabel('epoch')
        plt.ylabel('Loss')

        plt.legend()
        plt.title('Loss curve')
        plt.savefig(save_path + 'Loss curve.png')
        plt.close(f)
        print("Save Loss curve")

