import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def draw_loss_plot(loss,idx_number,threshold):
    epoch = [i for i in range(1, 161)]
    f = plt.figure(figsize=(8, 8))
    plt.plot(epoch, loss, c='blue', alpha=0.5)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss of data #{}'.format(idx_number))
    f.savefig('./loss_plt/resnet18-cifar100-data_{0}_{1}-loss.png'.format(idx_number,threshold))
    plt.close(f)

def draw_idx(idx,threshold):
    #idx_draw = [6432,34073,181,326, 663, 936,1057,204, 986, 987,1751,1772,47899,49486,49397,49192,49086,48973]
    idx_draw = [i for i in range(0,50000)]
    idx_loss = []
    for i in range(1,161):
        #loss_csv = pd.read_csv('./train_csv/train_loss_idv_data_{0}.csv'.format(i), index_col=0)
        loss_csv = pd.read_csv('./loss_csv/feature_loss_idv_data_{0}.csv'.format(i), index_col=0)
        loss = np.asarray(loss_csv)
        idx_loss.append(loss[idx])
    draw_loss_plot(idx_loss,idx,threshold)

def draw_all(threshold):
    #idx_draw = [6432,34073,181,326, 663, 936,1057,204, 986, 987,1751,1772,47899,49486,49397,49192,49086,48973]
    idx_draw = [i for i in range(0,50000)]
    for j in idx_draw:
        idx = j
        idx_loss = []
        for i in range(1,161):
            #loss_csv = pd.read_csv('./train_csv/train_loss_idv_data_{0}.csv'.format(i), index_col=0)
            loss_csv = pd.read_csv('./loss_csv/feature_loss_idv_data_{0}.csv'.format(i), index_col=0)
            loss = np.asarray(loss_csv)
            idx_loss.append(loss[idx])
        draw_loss_plot(idx_loss,idx,threshold)

def save_data(name, data, save_path):
    df_data = pd.DataFrame(data)
    df_data.to_csv(save_path + '{0}_data.csv'.format(name), encoding='ms949')
# loss_csv = pd.read_csv('./train_csv/train_loss_idv_data_{0}.csv'.format(i), index_col=0)

def save_data(name, data, save_path):
    df_data = pd.DataFrame(data)
    df_data.to_csv(save_path + '{0}_data.csv'.format(name), encoding='ms949')

def check_idx_list(idx_list, idx):
    if idx_list == []:
        return idx

    if len(idx_list) +len(idx) > 49999:
        id = abs(49999 - (len(idx_list) +len(idx)))
        return idx[:id]

    return idx

def load_data(idx):
    loss_csv = pd.read_csv('./loss_csv/feature_loss_idv_data_{0}.csv'.format(idx), index_col=0)
    loss = np.asarray(loss_csv)

    return np.array(loss)

def extract_loss_index(threshold=0.01):
    new_index = []
    delete_index = []  #

    num_count = 0
    sum_count = 0
    for i in range(1,161):
        if num_count == 49999:
            break

        loss = load_data(i)
        # 이미 list.append 된 데이터 예외처리.
        loss[delete_index] = 100

        idx = np.where(loss < threshold)[0]
        sum_count+=len(idx)
        print(i, len(idx), sum_count,idx[:10])

        idx = check_idx_list(new_index,idx)
        new_index.extend(idx)
        delete_index.extend(idx)
        num_count += len(idx)

    print(len(new_index))
    #나머지 데이터 처리.
    loss[delete_index] = 100
    idx = np.where(loss < 99)[0]
    new_index.extend(idx)
    print(len(new_index))

    save_data('loss_index', new_index, './')