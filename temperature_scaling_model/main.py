import data
import resnet
import utlis
import torch
import os
import argparse
import torch.nn as nn
import torch.optim as optim
import densenet
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn as cudnn
import itertools
import torchvision as tv
from torch.utils.data.sampler import SubsetRandomSampler
from temperature_scaling import ModelWithTemperature

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--epoch',default=110, type=int)
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--gpu_id', default='2', type=str, help='devices')

parser.add_argument('--data',default='cifar100', type=str)
parser.add_argument('--valid_size', default=5000, type=int, help='valid size')

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--decay',default=160,type=int)
parser.add_argument('--scheduler',default=True,type=bool)

parser.add_argument('--model',default='res', type=str)
parser.add_argument('--layer',default=110,type=int)
parser.add_argument('--opt',default='sgd',type=str)
parser.add_argument('--gamma',default=0.1,type=float)
parser.add_argument('--momentum',default=0.1,type=float)
parser.add_argument('--nesterov',default=True,type=bool)
parser.add_argument('--depth',default=40, type=int)
parser.add_argument('--save_path',default='./cali_cifar100_1/', type=str)
args = parser.parse_args()

def main():
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print("valid size : {0} batch size : {1}".format(args.valid_size, args.batch_size))
    train_loader, valid_loader, test_loader, train_feature_loader = data.data_set(args.valid_size, args.batch_size,save_path,args.data)

    if args.data == 'cifar100':
        num_class = 100
    elif args.data == 'cifar10':
        num_class = 10

    if args.model == 'dense':
        if (args.depth - 4) % 3:
            raise Exception('Invalid depth')
        block_config = [(args.depth - 4) // 6 for _ in range(3)]
        growth_rate = 12

    # layer , numclass    #[18, 34, 50, 101, 110, 152]
    if args.model == 'res':
        network = resnet.ResNet(args.layer, num_class).cuda()
        #network = resnet.resnet18().cuda()
    elif args.model == 'dense':
        #print(num_class)
        #network = densenet.DenseNet(growth_rate=growth_rate,block_config=block_config, num_classes=num_class).cuda()
        # bottleneck = False , reduction = 0.0 /  reduction=0.5, bottleneck=True,
        network = densenet.densenet_cifar().cuda()

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(reduce=False)

    if args.opt == 'sgd':                                                                                 # 0.001,5e-4
        optimizer = optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True,weight_decay=0.001)
    elif args.opt == 'rms':
        optimizer = optim.RMSprop(network.parameters(), lr=args.lr)
    elif args.opt == 'adam':
        optimizer = optim.Adam(network.parameters(), lr=args.lr)

    if args.scheduler == True:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[160, 180], gamma=0.1)
        #scheduler = lr_scheduler.StepLR(optimizer,step_size=args.decay, gamma=args.gamma)
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * args.epoch, 0.75 * args.epoch], gamma=0.1)

    print("Model's state_dict:")
    for param_tensor in network.state_dict():
        print(param_tensor, "\t", network.state_dict()[param_tensor].size())

    train_loss_li = []
    train_acc_li = []
    valid_loss_li = []
    valid_acc_li = []
    test_loss_li = []
    test_acc_li = []

    for epoch in range(0, args.epoch):
        if args.scheduler == True:
            scheduler.step()
        train_loss, train_acc, train_loss_idv = train(train_loader,network,criterion,optimizer,epoch+1)
        valid_loss, valid_acc, valid_softmax, valid_correct , valid_y,valid_loss_idv = test(valid_loader,network,criterion,epoch+1,'valid')
        test_loss, test_acc, test_softmax, test_correct , test_y,test_loss_idv = test(test_loader,network,criterion,epoch+1,'test')
        feature_loss, feature_acc, feature_softmax, feature_correct , feature_y,feature_loss_idv = test(train_feature_loader,network,criterion,epoch+1,'feature')
        # save files
        train_loss_li.append(train_loss)
        valid_loss_li.append(valid_loss)
        test_loss_li.append(test_loss)
        train_acc_li.append(train_acc)
        valid_acc_li.append(valid_acc)
        test_acc_li.append(test_acc)

        if epoch+1 == args.epoch:
            torch.save(network.state_dict(), save_path+'{0}_{1}.pth'.format('resnet110', epoch+1))
            utlis.save_data('train_loss_idv', train_loss_idv, save_path)
            utlis.save_data('feature_loss_idv', feature_loss_idv, save_path)

    for i in range(len(test_softmax)):
        test_softmax[i] = test_softmax[i].item()

    print(train_loss,train_acc,valid_loss,valid_acc,test_loss,test_acc)

    print('save logs')
    utlis.save_data('test_correct', test_correct, save_path)
    utlis.save_data('test_softmax', test_softmax, save_path)
    utlis.save_loss_acc('train', train_loss_li, train_acc_li, save_path)
    utlis.save_loss_acc('valid', valid_loss_li, valid_acc_li, save_path)
    utlis.save_loss_acc('test', test_loss_li, test_acc_li, save_path)

    print('draw curve')
    utlis.draw_curve('train_acc.csv', 'valid_acc.csv', 'acc', save_path)
    utlis.draw_curve('train_loss.csv', 'valid_loss.csv', 'loss', save_path)
    utlis.draw_test_curve('test_acc.csv', 'test_loss.csv', save_path)

def train(loader,network,criterion,optimizer,epoch):
    print('\nEpoch: %d' % epoch)
    network.train()
    li_softmax = []
    li_correct = []
    li_loss = []
    total_loss = 0
    total_acc = 0
    accuracy = 0

    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        output = network(input)
        loss = criterion(output, target).cuda()

        for i in loss:
            li_loss.append(i.cpu().data.numpy())

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        total_loss += loss.mean().item()
        pred = output.data.max(1, keepdim=True)[1]
        total_acc += pred.eq(target.data.view_as(pred)).sum()

        max_prob, max_label = torch.topk(F.softmax(output), 1, dim=1)
        li_softmax.extend(max_prob)
        for i in range(len(pred)):
            if pred[i] == target[i]:
                accuracy += 1
                cor = 1
            else:
                cor = 0

            li_correct.append(cor)

    total_loss /= len(loader)
    total_acc = 100. * total_acc / len(loader.dataset)
    print('Train Epoch: {} loss: {:.4f} Accuracy : {:.4f}%)'.format(epoch, total_loss,total_acc))

    return total_loss, float(total_acc), li_loss

def test(loader,network,criterion,epoch,mode):
    network.eval()
    with torch.no_grad():
        li_softmax = []
        li_correct = []
        li_y = []
        li_loss = []
        total_loss = 0
        total_acc = 0
        accuracy = 0

        for i, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()

            output = network(input)
            loss = criterion(output, target).cuda()
            if mode == 'feature':
                for i in loss:
                    li_loss.append(i.cpu().data.numpy())

            for i in target:
                li_y.append(i.cpu().data.numpy())

            total_loss += loss.mean().item()
            pred = output.data.max(1, keepdim=True)[1]
            total_acc += pred.eq(target.data.view_as(pred)).sum()

            max_prob, max_label = torch.topk(F.softmax(output), 1, dim=1)

            li_softmax.extend(max_prob)
            for i in range(len(pred)):
                if pred[i] == target[i]:
                    accuracy += 1
                    cor = 1
                else:
                    cor = 0

                li_correct.append(cor)

        total_loss /= len(loader)
        total_acc = 100. * total_acc / len(loader.dataset)
        print('{} Epoch: {} loss: {:.4f} Accuracy : {:.4f}%)'.format(mode, epoch, total_loss, total_acc))

    return total_loss, float(total_acc), li_softmax, li_correct, li_y, li_loss

def temperature_scaling():
    print("Calibration")
    state_dict = torch.load(args.save_path + 'resnet110_{0}.pth'.format(args.epoch))
    valid_indices = torch.load(args.save_path + 'valid_indices.pth')

    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    test_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])

    valid_set = tv.datasets.CIFAR100(root='./cifar100_data/', train=True, transform=test_transforms, download=False)
    test_set = tv.datasets.CIFAR100(root='./cifar100_data/', train=False, transform=test_transforms, download=False)

    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, sampler=SubsetRandomSampler(valid_indices))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, num_workers=4)
    #valid_loader = valid_loader_pth
    criterion = nn.CrossEntropyLoss()
    network = resnet.ResNet(args.layer, 100).cuda()

    network.load_state_dict(state_dict)

    test_loss, test_acc, test_softmax, test_correct, test_y, test_loss_idv = test(test_loader, network, criterion, args.epoch + 1,'test')

    for i in range(len(test_softmax)):
        test_softmax[i] = test_softmax[i].item()
    utlis.save_data('before_cali_correct', test_correct, args.save_path)
    utlis.save_data('before_cali_softmax', test_softmax, args.save_path)

    model = ModelWithTemperature(network)
    model.set_temperature(valid_loader)

    test_loss, test_acc, test_softmax, test_correct,  test_y, test_loss_idv = test(test_loader, model, criterion, args.epoch + 1,'test')

    for i in range(len(test_softmax)):
        test_softmax[i] = test_softmax[i].item()

    utlis.save_data('after_cali_correct', test_correct, args.save_path)
    utlis.save_data('after_cali_softmax', test_softmax, args.save_path)

if __name__ == "__main__":
    main()
    temperature_scaling()

