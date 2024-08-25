import os
import argparse
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

from models import model_dict
from utils import *


from torch.utils.data import DataLoader

import torch.nn.functional as F




def parse_option():


    parser = argparse.ArgumentParser('argument for evaluating')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=1, help='num of workers to use')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10','cifar100'], help='dataset')

    # model
    parser.add_argument('--model_t', type=str, default='resnet20',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--model_path', type=str, default=r"C:\Users\ashaz\OneDrive\Desktop\supervised-contrastive-kd-main - Copy\supervised-contrastive-kd-main\models\Evalute results\save\student_model\cifar10\few shot_0.25\resnet20\Sresnet20_Tresnet56_cifar10_FW0.25_SCKD\resnet20_0.25.pth", help='weights of the model you want to evaluate')
    # parser.add_argument('--model_path', type=str, default="./save/models/cifar100/resnet20_cifar100/ckpt_epoch_200.pth", help='weights of the model you want to evaluate')

    opt = parser.parse_args()
    print(opt)

    return opt

def main():

    opt = parse_option()
    #Loading data:
##########################################################################################
            
    _, val_loader,n_cls = choose_dataset(dataset=opt.dataset, batch_size=opt.batch_size, num_workers=opt.num_workers)

##########################################################################################

    
    # model
    model = load_model_evaluate(opt.model_t, n_cls,opt.model_path)
    

    
    ce_criterion = nn.CrossEntropyLoss().cuda()

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    printfreq=opt.print_freq
    
    model_acc= validate(val_loader, model, ce_criterion, printfreq)
    print('model accuracy: ', model_acc)



if __name__ == '__main__':
    main()
    


