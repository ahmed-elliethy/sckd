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
from utils import *


from torch.utils.data import DataLoader

import torch.nn.functional as F

from feature_projection import ProjectionNet
from sup_con_loss import SupConLoss



def parse_option():


    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=1, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,40', help='where to decay lr, can be a list')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10','cifar100'], help='dataset')

    # model
    parser.add_argument('--path_t', type=str, default="./save/models/cifar100/wrn_40_2_cifar100/ckpt_epoch_240.pth", help='best weights of the teacher model')

    # parser.add_argument('-few_shot', '--few_shot', type=float, default=1 , help='weight balance for other losses')
    
    
    
    opt = parser.parse_args()
    print(opt)


    # set the path according to the environment

    opt.projector_path = r'./save/projector/{}'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_model_name(opt.path_t)

    opt.model_name = 'projector_{}_{}_SCKD'.format(opt.model_t, opt.dataset)


    opt.save_folder = os.path.join(opt.projector_path, opt.model_name)
    print(opt.save_folder)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt





    
def main():

    opt = parse_option()
    #Loading data:
##########################################################################################

            
    train_loader, val_loader,n_cls = choose_dataset_projector(dataset=opt.dataset, batch_size=opt.batch_size, num_workers=opt.num_workers)

##########################################################################################




    # teacher model
    teacher_model = load_model(opt.path_t, n_cls)
    #Projector
    T_num_feature_maps = get_num_feature_maps(teacher_model)

    projector_t = ProjectionNet(T_num_feature_maps)
    
    if torch.cuda.is_available():
        teacher_model = teacher_model.cuda()
        projector_t   = projector_t.cuda()
        cudnn.benchmark = True
        
    # optimizer
    decay_epoch = opt.lr_decay_epochs

    projector_optimizer = torch.optim.Adam(projector_t.parameters(), lr=opt.learning_rate)#0.00001

    
    projector_scheduler = torch.optim.lr_scheduler.MultiStepLR(projector_optimizer, milestones=decay_epoch, gamma=0.1)
    
    sup_con_criterion = SupConLoss(temperature=0.07).cuda()
    ce_criterion = nn.CrossEntropyLoss().cuda()




    printfreq=opt.print_freq
    
    teacher_acc= validate(val_loader, teacher_model, ce_criterion, printfreq)
    print('teacher accuracy: ', teacher_acc)

###############################################################################

    teacher_model.eval()
    projector_t.train()
    sim=0
    es=ed=ek=0
    dis=km=100000
    for epoch in range(0, opt.epochs):
        print('current lr {:.5e}'.format(projector_optimizer.param_groups[0]['lr']))
        train_projector(train_loader, teacher_model, projector_t, projector_optimizer, epoch, sup_con_criterion, printfreq,val_loader)
        projector_scheduler.step()
        # loss=evaluate_loss(val_loader, teacher_model, projector_t, sup_con_criterion)
        similarity=(evaluate_similarity(val_loader,teacher_model,projector_t))*100
        distances=evaluate_distances(val_loader,teacher_model,projector_t)
        clustering=evaluate_clustering(val_loader, teacher_model, projector_t, 100)
        print("Report---------------------------------------------------------")
        # print(f"Loss: {loss:.2f}%")
        print(f"Similarity: {similarity:.2f}%")
        print(f"distances: {distances:f}")
        print(f"clustering: {clustering:f}")
        if epoch >= 40:
            # save model
            print(f"saving epoch number : {epoch:.1f}")
            state = {
                'opt': opt,
                'projector': projector_t.state_dict(),
                'optimizer': projector_optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, f'projector_model_epoch_{epoch:03d}.pth')
            torch.save(state, save_file) 
        print("---------------------------------------------------------------")

###############################################################################       
        if epoch >= 10:
            if similarity > sim:
                es=epoch
                print(f"---saving the best similarity at epoch:{es:f}")
                sim = similarity
                # save best model
                state = {
                    'opt': opt,
                    'projector': projector_t.state_dict(),
                    'optimizer': projector_optimizer.state_dict(),
                }
                save_file = os.path.join(opt.save_folder, 'sim_best_projector_model.pth')
                torch.save(state, save_file)  
            print(f"---current best similarity is:{sim:f}")   
            print(f"at:{es:f}")
            
            if distances < dis:
                ed=epoch
                print(f"--saving the best distances at epoch:{ed:f}")
                dis = distances
                # save best model
                state = {
                    'opt': opt,
                    'projector': projector_t.state_dict(),
                    'optimizer': projector_optimizer.state_dict(),
                }
                save_file = os.path.join(opt.save_folder, 'dis_best_projector_model.pth')
                torch.save(state, save_file)  
            print(f"current best distances is:{dis:f}")  
            print(f"at:{ed:f}")
    
            if clustering < km:
                ek=epoch
                print(f"saving the best kmeans at epoch:{ek:f}")
                km = clustering
                # save best model
                state = {
                    'opt': opt,
                    'projector': projector_t.state_dict(),
                    'optimizer': projector_optimizer.state_dict(),
                }
                save_file = os.path.join(opt.save_folder, 'Kmeans_best_projector_model.pth')
                torch.save(state, save_file)  
            print(f"current best kmeans is:{km:f}")  
            print(f"at:{ek:f}")
    
            print("---------------------------------------------------------------")

    # save model
    print('saving the last projector!')
    state = {
        'opt': opt,
        'projector': projector_t.state_dict(),
        'optimizer': projector_optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, 'projector_model.pth')
    torch.save(state, save_file)        


if __name__ == '__main__':
    main()
    