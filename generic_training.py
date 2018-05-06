'''
Author: Ruihang Du
Description:
Functions for training and evaluating any neural network models
Inspired by the PyTorch ImageNet example https://github.com/pytorch/examples/blob/master/imagenet/main.py
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import time
from datetime import timedelta


def train(model, train_loader, val_loader, batch_size, criterion, optimizer, \
        target_accr=None, err_margin=(0.01, 0.01), best_accr=(0, 0), topk=(1, 5), lr_decay=0.1, \
        saved_epoch=0, log='train.csv', pname='model.pth'):

    meters = {}
    for i in topk:
        meters[i] = AverageMeter()

    # log activity in the log file
    with open(log, 'a') as f:
        f.write(time.strftime('%b/%d/%Y %H:%M:%S', time.localtime()) + '\n')
        f.write('epoches, ' + ','.join(['top{}'.format(i) for i in topk]) + '\n')
    
    # resume epoch
    num_epoch = saved_epoch 
    
    # interval of evaluating performance
    epoch = 5

    # total number of data points in the dataset
    num_data = len(train_loader) * batch_size

    # if does not have a target accuracy, train to convergence
    if target_accr is None:
        # the accuracy obtained in the last round of training
        old_accr = best_accr

    while True:
        model.eval()

        result = tuple(validate(model, batch_size, val_loader, topk, True))

        # if the current accuracy is better than the best accuracy
        if len(list(filter(lambda t: t[0] > t[1], zip(best_accr, result)))) == 0:
            torch.save({
            'params':model.state_dict(), \
            'optim':optimizer.state_dict(), \
            'epoch':num_epoch}, pname)

        
        with open(log, 'a') as f:
            f.write(str(num_epoch) + \
            ',' + ','.join([str(r) for r in result]) + '\n')

        for i, r in enumerate(result):
            if target_accr is None:
                # if not converge, continue training
                if r - old_accr[i] > err_margin[i]: break
            elif target_accr[i] - r > err_margin[i]: break
        else: 
            with open(log, 'a') as f:
                f.write(time.strftime('%b/%d/%Y %H:%M:%S', time.localtime()) + '\n')
            break

        # update the old accuracy to current accuracy
        if target_accr is None:
            old_accr = result

        model.train()

        for e in range(epoch):
            for i in topk:
                meters[i].reset()

            # print('Validating ', end='', flush=True)
            print('Training on {} data'.format(num_data))

            for i, data in enumerate(train_loader, 0):
                index = 0

                inputs, labels = data

                # wrap inputs and labels in variables
                inputs, labels = Variable(inputs).cuda(), \
                Variable(labels).cuda()

                # zero the param gradient
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)

                loss = criterion(outputs, labels)

                result = accuracy(outputs.data, labels.data, topk)
                for j, k in enumerate(meters.keys()):
                    meters[k].update(result[j][0], inputs.size(0))

                loss.backward()

                optimizer.step()

                if i % 20 == 0:
                    print("Progress {:2.1%}".format(i * batch_size / num_data), end="\r")

            print('Epoch: [{0}]\t'.format(e))
            for k in meters.keys():
                print(' * Prec@{i} {topi.avg:.3f}%'
                  .format(i=k, topi=meters[k]))

            num_epoch += 1
            
            # decrement learning rate
            if num_epoch % 10 == 0:
                for p in optimizer.param_groups:
                    if p['lr'] > 1e-7:
                        p['lr'] *= (lr_decay ** (num_epoch/10))
                    print('change lr to {}'.format(p['lr']))


def validate(model, batch_size, val_loader, topk=(1, 5), cuda=True):
    # torch.cuda.set_device(0)
    meters = {}
    for i in topk:
        meters[i] = AverageMeter()

    # switch to evaluate mode
    model.eval()

    start = time.time()

    num_data = len(val_loader) * batch_size
    # print('Validating ', end='', flush=True)
    print('Validating on {} data'.format(num_data))

    for i, (input, target) in enumerate(val_loader):
        if cuda:
            input = input.cuda()
            target = target.cuda(non_blocking=True)
        input_var = Variable(input)
        target_var = Variable(target)

        output = model(input_var)

        # measure accuracy
        result = accuracy(output.data, target, topk)
        for j, k in enumerate(meters.keys()):
            meters[k].update(result[j][0], input.size(0))

        if i % 20 == 0:
            # print('.', end='', flush=True)
            print("Progress {:2.1%}".format(i * batch_size / num_data), end="\r")

    time_elapse = time.time() - start
    print('\ninference time:', str(timedelta(seconds=time_elapse)))
    for k in meters.keys():
        print(' * Prec@{i} {topi.avg:.3f}%'
          .format(i=k, topi=meters[k]))

    return (meters[k].avg for k in meters.keys())

    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

