import matplotlib
matplotlib.use('agg')
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def preprocssing_softmax(name):
    df_softmax = pd.read_csv('./{0}_cali_prob.csv'.format(name))
    df_acc = pd.read_csv('./{0}_cali_corr.csv'.format(name))

    confidence = np.array( df_softmax['0'])
    acc = np.array(df_acc['0'])

    return confidence, acc

def binging(confidence , acc):
    # 구간별 값 나누기
    confidence_li = []
    acc_li = []
    gap_li = []
    gap_count = []
    start_li = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    end_li = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    for i in range(10):
        if(i == 9):
            idx = np.where(start_li[i] <= confidence)
        elif(i ==0):
            idx = np.where(end_li[i] > confidence)
        else:
            idx = np.where((start_li[i] <= confidence) & (end_li[i] > confidence))
        print(start_li[i],end_li[i])
        print('data count:',len(idx[0]))
        # nan 조건
        if(len(idx[0]) == 0):
            confidence_li.append(0)
            acc_li.append(0)
            gap_count.append(0)
            gap_li.append(0)
            print(0,0,0)
            print("real : {0}".format(0))
            print("model : {0}".format(0))
            print("gap : {0}".format(0))
            print('------------------------------')
        else:
            avg_confidence = np.mean(confidence[idx])

            correct = np.where(acc[idx] == 1)
            incorrect = np.where(acc[idx] == 0)
            avg_acc = len(acc[correct]) / len(acc[idx])

            gap = abs(avg_confidence-avg_acc)
            confidence_li.append(avg_confidence)
            acc_li.append(avg_acc)
            gap_li.append(gap)
            gap_count.append(len(idx[0]))
            print(len(correct[0]), len(incorrect[0]), len(acc[idx]))
            print("model : {0}".format(round(avg_confidence,4)))
            print("real : {0}".format(round(avg_acc,4)))
            print("gap : {0}".format(round(gap,4)))
            print('------------------------------')

    return acc_li, gap_li, gap_count,confidence_li
# ECE 구하기.
def ECE(confidence, gap_li, gap_count):
    print('gap_list')
    print(gap_li)
    ece = 0
    for idx,i in enumerate(gap_li):
        ece += i * gap_count[idx]

    ece = ece / len(confidence)
    print("Expected Calibration Error : ",round(ece,4))
    return ece

# MCE
def MCE(gap_li):
    mce = np.max(gap_li)
    print("Maximum Calibration Error : ", round(mce,4))
    return mce

# reliability diagrams 그리기.
def draw_reliability_diagrams(acc_li, ece,gap_count,name):
    x_label = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    line = [0,1,2,3,4,5,6,7,8,9]
    x = [i for i in range(10)]

    # 차트에 그릴 gap 구하기.
    bar_gap = []
    for i in range(0,10):
        bar_gap.append(abs(x_label[i]-acc_li[i]))

    bin_size = 1/10
    positions = np.arange(0+bin_size/2, 1+bin_size/2, bin_size)
    bbox_props = dict(boxstyle="Square", fc="w", ec="0.5", alpha=0.9)

    f = plt.figure()
    plt.text( 0.62, 0.05, 'ECE={:.4f}'.format(ece), fontsize=20, bbox=bbox_props)

    acc_bar =plt.bar(positions, acc_li,width=bin_size,linewidth=1,edgecolor = 'black', color = 'b',label='Outputs',alpha=1)
    plt.bar(positions, bar_gap,width=bin_size,linewidth=3,edgecolor = 'r', color = 'r',label='Gap',alpha=0.3,bottom=acc_li, hatch="/")
    plt.plot(line, ls="--", color='gray',linewidth=5)

    idx = 0
    for rect in acc_bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 0.99 * height,
                '%d' % int(gap_count[idx]), ha='center', va='bottom')
        idx+=1

    plt.ylabel('Accuracy')
    plt.xlabel('Confidence')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.legend()
    plt.title('ResNet-110 CIFAR-100')
    f.savefig('{0}_Reliability_diagrams.png'.format(name))
    plt.close(f)

if __name__ == "__main__":
    print('------------------------before calibration-----------------------------------')
    confidence, acc = preprocssing_softmax('before')
    acc_li, gap_li, gap_count, confidence_li = binging(confidence, acc)
    ece = ECE(confidence, gap_li, gap_count)
    MCE(gap_li)
    draw_reliability_diagrams(acc_li, ece, gap_count,'before')

    print('------------------------After calibration-----------------------------------')
    confidence, acc = preprocssing_softmax('after')
    acc_li, gap_li, gap_count, confidence_li = binging(confidence, acc)
    ece = ECE(confidence, gap_li, gap_count)
    MCE(gap_li)
    draw_reliability_diagrams(acc_li, ece, gap_count,'after')

