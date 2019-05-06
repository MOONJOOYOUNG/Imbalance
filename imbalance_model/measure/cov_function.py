import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import *
from sklearn.covariance import MinCovDet
from sklearn.manifold import TSNE

def det2(mat):
    matrix = [[a%2 for a in row] for row in mat]
    n = len(matrix)
    if n == 1: return matrix[0][0]
    #first find a nonzero element in first column
    i = 0
    while i < n and matrix[i][0] == 0: i += 1
    if i == n:
        return 0 #since a column of zeros
    else:
        if 0 < i: matrix[0],matrix[i] = matrix[i],matrix[0]
    #clear rest of column:
    for i in range(1,n):
        if matrix[i][0] == 1:
            matrix[i] = [a+b for a,b in zip(matrix[i],matrix[0])]
    rest = [row[1:] for row in matrix[1:]]
    return np.sqrt(det2(rest))


def cov_volume(data):
    determinant = np.linalg.det(data)
    determinant = np.sqrt(determinant)

    print('determinant {:.4f}'.format(determinant))

    return round(determinant,4)

def high_dim_cov(high_data):
    # 데이터 없을시예욏러ㅣ
    if (len(high_data) == 0):
        return
    cov_val = np.array(high_data)
    cov_val = np.cov(cov_val, rowvar=False)
    determinant = cov_volume(cov_val)

    sum_variance = 0
    #print(cov_val)
    for i in range(0, 512):
        sum_variance += cov_val[i][i]

    print(len(high_data), ' {:.4f}'.format(sum_variance / 512))

    return determinant,len(high_data)

def low_dim_cov(low_data):
    if (len(low_data) == 0):
        return

    cov_val = np.array(low_data)
    cov_val = np.cov(cov_val, rowvar=False)
    determinant = cov_volume(cov_val)

    sum_variance = 0
    for i in range(0,2):
        sum_variance+=cov_val[i][i]

    print(len(low_data),' {:.4f}'.format(sum_variance/ 2))
    return determinant, len(low_data)

def class_feature(x, y):
    x_data = []
    for i in range(0,10):
        idx = np.where(np.array(y)==i)[0]
        x = np.array(x)
        x_data.append(x[idx])

    return x_data

def class_feature_low(x, y):
    x_data = []
    for i in range(0,10):
        idx = np.where(np.array(y)==i)[0]
        x = np.array(x)
        x_data.append(x[idx[0:250]])

    return x_data

def tsne_feature(feature_layer, Y_lists):
    print("extract_layer process")

    feature_layer = np.array(feature_layer)
    # print(len(tsne_x_l4),len(y_label))
    data_Y = np.array(Y_lists[0:3000])
    data_TSNE = TSNE().fit_transform(feature_layer[0:3000])

    return data_TSNE, data_Y

def x_data_To_float(path):
    tsne = pd.read_csv(path)
    tsne_x = []
    tsne_x.append(tsne.values)
    tsne_x = [a for i in tsne_x for a in i]
    tsne_x = [a for i in tsne_x for a in i]
    tsne_x.remove(0)

    new_li_x1 = []
    for i in tsne_x:
        i = i.replace("'", '')
        i = i.replace("[", '')
        i = i.replace("]", '')
        i = i.replace("\n", '')
        new_li_x1.append((i))

    float_x = []

    for i in new_li_x1:
        i = i.strip()
        st = i.split(' ')
        split_li = []
        for j in st:
            a = ''
            if(j!=''):
                a = float(j)
            if(type(a)==float):
                split_li.append(a)
        float_x.append(split_li)
    return float_x

def y_data_To_list(path):
    tsne = pd.read_csv(path)
    tsne_y = []

    tsne_y.append(tsne.values)

    tsne_y = [a for i in tsne_y for a in i]
    tsne_y = [a for i in tsne_y for a in i]
    del tsne_y[0]

    from collections import Counter
    result = Counter(tsne_y)
    print(result)
    return tsne_y

def extract_layer_test_merge(feature_layer_train, feature_layer_test, Y_lists_train, Y_lists_test):
    print("Merger ectract_layer process")
    print(len(feature_layer_train), len(feature_layer_test), len(Y_lists_train), len(Y_lists_test))
    y_test = []
    for i in Y_lists_test:
        i += 10
        y_test.append(i)
    data_x = []
    data_x.extend(feature_layer_train[0:2500])
    data_x.extend(feature_layer_test[0:2500])
    data_Y = []
    data_Y.extend(Y_lists_train[0:2500])
    data_Y.extend(y_test[0:2500])
    print(len(data_x), len(data_Y))
    data_Y = np.array(data_Y)
    data_TSNE = TSNE().fit_transform(data_x)

    data_TSNE.shape[0]
    print(data_TSNE.shape[0])
    return data_TSNE, data_Y


def scatter_merge(x, label, idx):
    print("Merger scatter process")
    new_x = []
    new_y = []
    i = 0
    for i in range(0, 20):
        index = 0
        new_x_2 = []
        new_y_2 = []
        for j in label:
            if (i == j):
                new_x_2.append(x[index, 0])
                new_y_2.append(x[index, 1])
            index += 1
        new_x.append(new_x_2)
        new_y.append(new_y_2)

    palette = np.array(sns.color_palette("hls", 20))
    col = palette[label.astype(np.int)]
    # label2 = list(itertools.chain.from_iterable(palette[label.astype(np.int)].tolist()))
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    lables = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
              '19']

    idx = 0
    for color in palette:
        c, d = new_x[idx], new_y[idx]
        ax.scatter(c, d, c=color, s=20, label=lables[idx])
        idx += 1
    #    ax.legend()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # We add the labels for each digit.
    txts = []
    for i in range(20):
        # Position of each label.
        xtext, ytext = np.median(x[label == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txts.append(txt)
    f.savefig(r'.\mnist_%d.png' % 100)
    print("Save & epoch finish")


# Visiualizing feature_map
def scatter(x, label, idx):
    print("scatter process")
    palette = np.array(sns.color_palette("hls", 10))
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[label.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[label == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txts.append(txt)

    f.savefig(r'.\mnist_%d.png' % idx)
    print("Save & epoch finish")

# test_extract
def extract_layer_test(feature_layer_train, feature_layer_test, Y_lists_train, Y_lists_test):
    print("TEST_dataset_extract_layer process")
    print(len(feature_layer_train), len(feature_layer_test), len(Y_lists_train), len(Y_lists_test))

    data_x = []
    data_Y = []

    data_x.extend(feature_layer_train[0:2500])
    data_x.extend(feature_layer_test[0:2500])

    data_Y.extend(Y_lists_train[0:2500])
    data_Y.extend(Y_lists_test[0:2500])

    print(len(data_x), len(data_Y))
    data_Y = np.array(data_Y)
    data_TSNE = TSNE().fit_transform(data_x)

    return data_TSNE, data_Y
