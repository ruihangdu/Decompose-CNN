import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from datetime import timedelta
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import tensorly as tl
import numpy as np
import argparse

# added on march 27
from collections import OrderedDict

from TVBMF import EVBMF
from generic_training import train, validate
from torch_cp_decomp import torch_cp_decomp
from torch_tucker import tucker_decomp

from decomp_resnet50 import decomp_resnet
from decomp_alexnet import decomp_alexnet

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--decomp', type=str, default='cp', \
    help='specify which decomposition to use (cp or tucker)')
parser.add_argument('-m', '--model', type=str, default=None, \
    help='using an already decomposed or modified model') 
parser.add_argument('-r', '--resume', type=str, default=None, \
    help='resume a previous retraining state') 
parser.add_argument('-s', '--state', type=str, default=None, \
    help='using a previous retrained (completed) state') 
parser.add_argument('-p', '--path', type=str, default=None, \
    help='path to dataset')
parser.add_argument('-v', '--val', action='store_true', \
    help='training or validation mode')
parser.add_argument('-a', '--arch', type=str, default='resnet50',\
    help='network architecture to decompose')


def gen_loaders(path, BATCH_SIZE, NUM_WORKERS):
    # Data loading code
    traindir = os.path.join(path, 'train')
    valdir = os.path.join(path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True)

    return (train_loader, val_loader)


def main():
    global args
    args = parser.parse_args()
    use_cp = (args.decomp == 'cp')
    use_model = (args.model is not None)
    use_param = (args.resume is not None)
    use_state = (args.state is not None)
    eval_mode = (args.val)

    decomp_func = torch_cp_decomp if use_cp else tucker_decomp

    if args.arch == 'resnet50':
        decomp_arch = decomp_resnet
    elif args.arch == 'alexnet':
        decomp_arch = decomp_alexnet
    else:
        sys.exit('architecture not supported')

    rank_func = est_rank if use_cp else tucker_rank

    # here the batch size and the number of threads are preset
    # change if needed
    BATCH_SIZE = 128
    NUM_WORKERS = 8

    DATA_PATH = args.path
    if DATA_PATH is None:
        sys.exit('Path to dataset cannot be empty')

    tl.set_backend('pytorch')

    train_loader, val_loader = gen_loaders(DATA_PATH, BATCH_SIZE, NUM_WORKERS)

    # here use the PyTorch ResNet50 architecture
    net = models.__dict__[args.arch](pretrained=True)

    if use_model:
        checkpoint = torch.load(args.model)
        arch = checkpoint['arch']
        params = checkpoint['params']

        for n, m in net.named_children():
            setattr(net, n, arch[n])
        net.load_state_dict(params)
    else:
        net = decomp_arch(net, rank_func, decomp_func)

        
        torch.save({'arch':dict(net.named_children()),\
        'params': net.state_dict()},\
        'cp_round_model.pth' if use_cp else 'tucker_round_model.pth')

    
    lr = 1e-6 if use_cp else 1e-3
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    epoch = 0

    if use_param:
        checkpoint = torch.load(args.resume)
        old_state = checkpoint['params']
        opt = checkpoint['optim']
        epoch = checkpoint['epoch']
        
        new_state = OrderedDict()
        for k, v in old_state.items():
            name = k[7:]
            new_state[name] = v

        net.load_state_dict(new_state)
        optimizer.load_state_dict(opt)
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    if use_state:
        old_state = torch.load(args.state)
        new_state = OrderedDict()
        for k, v in old_state.items():
            name = k[7:]
            new_state[name] = v

        net.load_state_dict(new_state)


    net = nn.DataParallel(net).cuda()
    target = (76.15, 92.87)   # define the retrain target
        
    criterion = nn.CrossEntropyLoss().cuda()

    train_args = OrderedDict()

    if not eval_mode:
        train_args['model'] = net
        train_args['trainloader'] = train_loader
        train_args['testloader'] = val_loader
        train_args['batch_size'] = BATCH_SIZE
        train_args['criterion'] = criterion
        train_args['optimizer'] = optimizer
        train_args['target_accr'] = target
        train_args['err_margin'] = (1.5,1.5)
        train_args['best_acc'] = (0,0)
        train_args['topk'] = (1,5)
        train_args['lr_decay'] = 0.8
        train_args['saved_epoch'] = epoch
        train_args['log'] = 'cp_acc.csv' if use_cp else 'tucker_acc.csv'
        train_args['pname'] = 'cp_best.pth' if use_cp else 'tucker_best.pth'

        train(*train_args.values())

        torch.save(net.state_dict(), 'cp_state.pth' if use_cp\
        else 'tucker_state.pth') 

    else:
        train_args['model'] = net
        train_args['batch_size'] = BATCH_SIZE
        train_args['testloader'] = val_loader

        validate(*train_args.values())


def tucker_rank(layer):
    W = layer.weight.data
    mode3 = tl.base.unfold(W, 0)
    mode4 = tl.base.unfold(W, 1)
    diag_0 = EVBMF(mode3)
    diag_1 = EVBMF(mode4)
    d1 = diag_0.shape[0]
    d2 = diag_1.shape[1]

    del mode3
    del mode4
    del diag_0
    del diag_1

    # round to multiples of 16
    return [int(np.ceil(d1 / 16) * 16) \
            , int(np.ceil(d2 / 16) * 16)]


def est_rank(layer):
    W = layer.weight.data
    mode3 = tl.base.unfold(W, 0)
    mode4 = tl.base.unfold(W, 1)
    diag_0 = EVBMF(mode3)
    diag_1 = EVBMF(mode4)

    # round to multiples of 16
    return int(np.ceil(max([diag_0.shape[0], diag_1.shape[0]]) / 16) * 16)


if __name__ == '__main__':
    main()
