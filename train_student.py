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

from feature_projection import *



def parse_option():


    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=1, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100,150,180', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10','cifar100'], help='dataset')

    # model`
    parser.add_argument('--model_s', type=str, default='resnet20',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default="./save/models/cifar10/resnet56_cifar10/resnet56_best.pth", help='best weights of the teacher model')
    parser.add_argument('--Projector_path', type=str, default="./save/projector/cifar10/projector_resnet56_cifar10_SCKD/projector_model.pth", help='best weights of the projector model')

    #beta 

    parser.add_argument('-b', '--beta', type=float, default=80 , help='weight balance for other losses')
   
    parser.add_argument('-few_shot', '--few_shot', type=float, default=1 , help='weight balance for other losses')# 0.75 ,beta=120 # 0.5 beta= #0.25 beta=
    parser.add_argument('-n', '--num_runs', type=float, default=1 , help='weight balance for other losses')# number of runs

    
    opt = parser.parse_args()
    print(opt)

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']: 
        opt.learning_rate = 0.01

    # set the path according to the environment

    opt.model_path = r'./save/student_model/{}/few shot_{}/{}'.format(opt.dataset,opt.few_shot,opt.model_s)
    
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_model_name(opt.path_t)

    opt.model_name = 'S{}_T{}_{}_FW{}_SCKD'.format(opt.model_s, opt.model_t, opt.dataset,opt.few_shot)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    print(opt.save_folder)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
        


    return opt



    






    
def main():

    best_prec1 = 0

    for i in range(1,10) :    

        opt = parse_option()
        #Loading data:
        
            
        # print(opt.save_folder)
            
        with open(opt.save_folder+ ".txt", 'a') as f:
            print('####################################################################################################################################\n'
                  'Beta: [{0}]:\t' 'teacher:{T}\t' 'student:{S}\t' 'Dataset:{D}\t' 'current run:{x}\t' 'few shot:{f}%\t'
                  
                  .format(opt.beta, T=opt.model_t,S=opt.model_s,D=opt.dataset,x=i,f=opt.few_shot*100), file=f)     
            
    ##########################################################################################
        if opt.few_shot == 1:
                
            train_loader, val_loader,n_cls = choose_dataset(dataset=opt.dataset, batch_size=opt.batch_size, num_workers=opt.num_workers)
        else :   
            train_loader, val_loader,n_cls = choose_dataset_fewshots(dataset=opt.dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, a=opt.few_shot)
    
    ##########################################################################################
        # student model
        student_model = model_dict[opt.model_s](num_classes=n_cls) 
        
        # teacher model
        teacher_model = load_model(opt.path_t, n_cls)
        
        # optimizer        
        
        optimizer = optim.SGD(student_model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
        
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=opt.lr_decay_epochs)
        # print(opt.lr_decay_epochs)
        
        ce_criterion = nn.CrossEntropyLoss().cuda()
        mse_criterion = nn.MSELoss().cuda()
        l1_criterion = nn.L1Loss().cuda()
    
        if torch.cuda.is_available():
            student_model = student_model.cuda()
            teacher_model = teacher_model.cuda()
    
            cudnn.benchmark = True
    
        printfreq=opt.print_freq
        
        teacher_acc= validate(val_loader, teacher_model, ce_criterion, printfreq)
        print('teacher accuracy: ', teacher_acc)
    
    ###############################################################################
        T_num_feature_maps = get_num_feature_maps(teacher_model)
        S_num_feature_maps = get_num_feature_maps(student_model)
        # print(f"Number of output feature maps from the last convolutional layer: {num_feature_maps}")
        projector_t = ProjectionNet(T_num_feature_maps)
        interpreter = ProjectionNet(T_num_feature_maps)
        projector_t.cuda()
        interpreter.cuda()
                    
        checkpoint = torch.load(opt.Projector_path)
        print(checkpoint.keys())
        
        # load Projector and interpreter weights
        projector_t.load_state_dict(torch.load(opt.Projector_path)['projector'])
        interpreter.load_state_dict(torch.load(opt.Projector_path)['projector'])
        
        
        #in case of different families#########################################
        if  T_num_feature_maps != S_num_feature_maps :
            regressor   = Regressornet(S_num_feature_maps,T_num_feature_maps)
            regressor.cuda()
            regressor_optimizer = torch.optim.SGD(regressor.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            regressor_scheduler = torch.optim.lr_scheduler.MultiStepLR(regressor_optimizer, milestones=[80, 120],
                                                                gamma=0.1)
        #######################################################################    
            
            
        time_begin = time.time()
        for epoch in range(0, opt.epochs):
    
            # train for one epoch
            print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
            print('current best {}'.format(best_prec1))
            print('current run {}'.format(i))
            # train_loader, teacher_model, student_model, paraphraser, interpreter,beta, epoch
            # if epoch <= 140 :
            #     beta=40
            # else:
            #     beta=80
            time1 = time.time()
      
            
            #check if different families#######################################
            if  T_num_feature_maps == S_num_feature_maps :
                train_SCKD(train_loader, teacher_model, student_model, projector_t, interpreter, optimizer, opt.beta,mse_criterion, l1_criterion, ce_criterion, printfreq, epoch, opt.dataset, opt.save_folder)
            else:
                train_SCKD_diffarch(train_loader, teacher_model, student_model, projector_t, interpreter, optimizer, opt.beta, mse_criterion , l1_criterion, ce_criterion, printfreq, epoch, opt.dataset, opt.save_folder,regressor, regressor_optimizer)
            ###################################################################

            
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
                    
            lr_scheduler.step()
            
            
            #in case of different families#####################################
            if  T_num_feature_maps != S_num_feature_maps :
                regressor_scheduler.step()
            ###################################################################
            
            
            # evaluate on validation set
            prec1 = validate(val_loader, student_model, ce_criterion, printfreq)
    
            # save the best model
            if prec1 > best_prec1:
                best_prec1 = prec1
                state = {
                    'epoch': epoch,
                    'model': student_model.state_dict(),
                    'best_acc': best_prec1,
                    'optimizer': optimizer.state_dict(),
                }
                save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
                print('saving the best model!')
                torch.save(state, save_file)
                with open(opt.save_folder+ ".txt", 'a') as f:
                    print('Epoch: [{0}]:\t'
                          'Accuraccy: {best_prec1:.2f}'
                          .format(epoch, best_prec1=best_prec1,), file=f)
    
            # regular saving
            if epoch % opt.save_freq == 0:
                print('==> Saving...')
                state = {
                    'epoch': epoch,
                    'model': student_model.state_dict(),
                    'accuracy': prec1,
                    'optimizer': optimizer.state_dict(),
                }
                save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    print('best accuracy:', best_prec1)
    time_end = time.time()
    print('total training time is {:.2f}'.format(time_end - time_begin))

    # save model
    state = {
        'opt': opt,
        'model': student_model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)     
    


if __name__ == '__main__':
    main()
    