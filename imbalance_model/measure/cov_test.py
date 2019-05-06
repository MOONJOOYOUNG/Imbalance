import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import *
from sklearn.covariance import MinCovDet
from sklearn.manifold import TSNE


def load_data(mode):
    if mode == 'train':
        x_feature = pd.read_csv('./train_feature_data.csv',index_col=0)
        y_data = pd.read_csv('./train_y_data.csv',index_col=0)
    else:
        x_feature = pd.read_csv('./test_feature_data.csv', index_col=0)
        y_data = pd.read_csv('./test_y_data.csv', index_col=0)
    x_data = []
    for idx, i in enumerate(x_feature.iterrows()):
        x_data.append(i[1])

    print('x , y len')
    print(len(x_data), len(y_data))
    # np.asarray(x_data), np.asarray(y_data)
    return np.asarray(x_data), np.asarray(y_data)

def class_feature(x, y):
    x_data = []
    for i in range(0,10):
        idx = np.where(y==i)[0]
        x_data.append(x[idx])
    print('class feature')
    print(len([a for i in x_data for a in i]))
    return np.asarray(x_data)

def cov_volume(data):
    determinant = np.linalg.det(data)
    determinant = np.sqrt(determinant)

    print('determinant {:.6f}'.format(determinant))

    return round(determinant,4)

def draw(x_data,label,mode):
    if (len(x_data) == 0):
        return

    x = [i for i in range(1, 65)]

    for idx, i in enumerate(x_data[:100]):
        y = []
        y.append(i)
        y = [a for i in y for a in i]

        f = plt.figure(figsize=(8, 8))
        plt.plot(x, y)
        plt.xlabel('nodes')
        plt.ylabel('activation value')
        plt.title('activation value plot')
        plt.savefig('./{0}_feature_{1}_{2}.png'.format(mode,label,idx))
        plt.close(f)

def high_dim_cov(high_data):
    if (len(high_data) == 0):
        return

    cov_val = np.cov(high_data, rowvar=False)
    sum_variance = 0

    for i in range(0, 64):
        sum_variance += cov_val[i][i]

    print(len(high_data), ' {:.6f}'.format(sum_variance / 64))

    return len(high_data) #,determinant

def mean(x_data):
    if (len(x_data) == 0):
        return
    mean = np.mean(x_data,axis=0)
    # 1 = 1000 , 0 = 64
    #mean = np.mean(x_data, axis=1)
    #print(sum(mean)/len(mean))
    print(len(x_data), round(sum(mean)/len(mean),6))

def cov_mean(x_data):
    if (len(x_data) == 0):
        return
    # 0 -> 64 / 1 -> len(data)

    i = x_data.mean(axis=0)
    cov_val = float(np.cov(i, rowvar=False))
    #cov_val = np.var(i)
    print(round(cov_val,6))

# train_x,train_y =load_data('train')
# #print(np.mean(train_x[:2],axis=0))
#
# #train_x = class_feature(train_x,train_y)
# test_x,test_y =load_data('test')
#test_x = class_feature(test_x,test_y)

# print(x_feature.mean(axis=1))

train_x = pd.read_csv('./train_feature_data.csv',index_col=0)
train_y = pd.read_csv('./train_y_data.csv',index_col=0)
test_x = pd.read_csv('./test_feature_data.csv', index_col=0)
test_y = pd.read_csv('./test_y_data.csv', index_col=0)

train_x = np.array(train_x)
train_y = np.array(train_y)
print(train_x[0])
train_y = train_y.reshape(24750)

test_x = np.array(test_x)
test_y = np.array(test_y)
test_y = test_y.reshape(10000)

print(train_x.shape,train_y.shape)

print('train')
for i in range(10):
    print(train_x[(np.array(train_y)==i)].mean(0).var())
print('test')
for i in range(10):
    print(test_x[(np.array(test_y)==i)].mean(0).var())

# print('train data')
# for idx,i in enumerate(train_x):
#     cov_mean(i)
#     #determinant = high_dim_cov(i)
#     #mean(i)
#     #draw(i,idx,'train')
# print('test data')
# for idx,i in enumerate(test_x):
#     cov_mean(i)
#     #determinant = high_dim_cov(i)
#     #mean(i)
#     #draw(i,idx,'test')
# #
