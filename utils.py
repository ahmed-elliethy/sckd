import os
import argparse
import time

import shutil
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

from models import model_dict
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np


###############################################################################
#General Functions
###############################################################################
def load_model_evaluate(model_t, n_cls,model_path):
    print('==> loading model')
    model_t = model_t
    # model = torch.nn.DataParallel(model_dict[model_t](num_classes=n_cls))

    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model


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
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def choose_dataset(dataset, batch_size, num_workers):
    if dataset == 'cifar10':
        print(dataset)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    
        
    
        test_set = datasets.CIFAR10(root='./data/Cifar10',
                                     download=True,
                                     train=False,
                                     transform=test_transform)
        val_loader = DataLoader(test_set,
                                 batch_size=int(batch_size/2),
                                 shuffle=False,
                                 num_workers=int(num_workers/2))
        

        train_set = datasets.CIFAR10(root='./data/Cifar10',
                                      download=True,
                                      train=True,
                                      transform=train_transform)
        print(len(train_set))

        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
            
        n_cls = 10
    else:
        print(dataset)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    
        
        
       
        test_set = datasets.CIFAR100(root='./data/Cifar100',
                                     download=True,
                                     train=False,
                                     transform=test_transform)
        val_loader = DataLoader(test_set,
                                 batch_size=int(batch_size/2),
                                 shuffle=False,
                                 num_workers=int(num_workers/2)) 
        
 

        train_set = datasets.CIFAR100(root='./data/Cifar100',
                                      download=True,
                                      train=True,
                                      transform=train_transform)
        print(len(train_set))

        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    
            
        n_cls = 100

    return train_loader,val_loader,n_cls

###############################################################################
#Validation
###############################################################################

def validate(val_loader, model, criterion,print_freq):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output, _ ,_ = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

###############################################################################
