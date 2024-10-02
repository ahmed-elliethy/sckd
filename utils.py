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

from feature_projection import  ProjectionNet
from sup_con_loss import SupConLoss

###############################################################################
#General Functions
###############################################################################
def get_model_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_model(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_model_name(model_path)
    # model = torch.nn.DataParallel(model_dict[model_t](num_classes=n_cls))

    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model

class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def ft_normalize(x):
    return F.normalize(x, dim=1)

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
#gets the number of the output feature maps of the last conv layer from both the teacher and the student
#its output is used as an input for the projector, interpreter and the regressor.
###############################################################################
def get_num_feature_maps(model):

    num_feature_maps = None
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            num_feature_maps = layer.out_channels
    return num_feature_maps



###############################################################################
#Choosing dataset
###############################################################################
def choose_dataset_fewshots(dataset, batch_size, num_workers, a):
    
    
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
        
        from sklearn.model_selection import StratifiedShuffleSplit
        labels = [train_set[i][1] for i in range(len(train_set))]
        ss = StratifiedShuffleSplit(n_splits=1, test_size=1-a, random_state=0)
        train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]
        train_set = torch.utils.data.Subset(train_set, train_indices)
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
        
        from sklearn.model_selection import StratifiedShuffleSplit
        labels = [train_set[i][1] for i in range(len(train_set))]
        ss = StratifiedShuffleSplit(n_splits=1, test_size=1-a, random_state=0)
        train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]
        train_set = torch.utils.data.Subset(train_set, train_indices)
        print(len(train_set))
        
        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    
            
        n_cls = 100

    return train_loader,val_loader,n_cls

#choosing projector's dataset#############
def choose_dataset_projector_fewshots(dataset, batch_size, num_workers, a):
    
    
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
                                      transform=TwoCropTransform(train_transform))
        
        from sklearn.model_selection import StratifiedShuffleSplit
        labels = [train_set[i][1] for i in range(len(train_set))]
        ss = StratifiedShuffleSplit(n_splits=1, test_size=1-a, random_state=0)
        train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]
        train_set = torch.utils.data.Subset(train_set, train_indices)
        print(len(train_set))
        
        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  shuffle=True,pin_memory=True,
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
                                      transform=TwoCropTransform(train_transform))
        
        from sklearn.model_selection import StratifiedShuffleSplit
        labels = [train_set[i][1] for i in range(len(train_set))]
        ss = StratifiedShuffleSplit(n_splits=1, test_size=1-a, random_state=0)
        train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]
        train_set = torch.utils.data.Subset(train_set, train_indices)
        print(len(train_set))
        
        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  shuffle=True,pin_memory=True,
                                  num_workers=num_workers)
    
            
        n_cls = 100

    return train_loader,val_loader,n_cls

def choose_dataset_projector(dataset, batch_size, num_workers):
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
        
       
        train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(root='./data', train=True, transform=TwoCropTransform(train_transform), download=True),
                batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

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
        
        train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(root='./data', train=True, transform=TwoCropTransform(train_transform), download=True),
                batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
 
            
        n_cls = 100

    return train_loader,val_loader,n_cls


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

###############################################################################
#Normal training
###############################################################################
def train(train_loader, model, criterion, optimizer, epoch,print_freq):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target

        # compute output
        output, _ , _= model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

def train_W_regressor(train_loader, model, criterion, optimizer, epoch,print_freq):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target

        # compute output
        output, _ , _= model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))
###############################################################################
#Projector training
###############################################################################
def train_projector(trainloader, teacher_model, projector, optimizer, epoch, sup_con_criterion, printfreq,val_loader):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    # teacher_model = torch.nn.DataParallel(teacher_model)  # Wrap the model with DataParallel for parallel execution

    for batch_idx, (images, labels) in enumerate(trainloader):
        
        # optimizer.zero_grad()
        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        outputs, pre_out, pre_pool = teacher_model(images)
        features,x = projector(pre_pool)

        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        loss = sup_con_criterion(features, labels)
        #print(loss.item())

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # print info
        if (batch_idx + 1) % printfreq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, batch_idx + 1, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()


    return losses.avg    

###############################################################################
#Projector testing
###############################################################################
def evaluate_loss(val_loader, teacher_model, projector, criterion):
    projector.eval()
    teacher_model.eval()

    losses = AverageMeter()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.cuda(non_blocking=True) if torch.cuda.is_available() else images
            bsz = labels.shape[0]

            # Forward pass through the teacher model
            outputs, pre_out, pre_pool = teacher_model(images)
            # Get features from the projector
            features, _ = projector(pre_pool)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            # Compute loss
            loss = criterion(features, labels)

            # Update loss
            losses.update(loss.item(), images.size(0))

    return losses.avg

###############################################################################
def evaluate_similarity(val_loader, teacher_model, projector, metric_fn=torch.cosine_similarity):
  """
  Evaluates the projector indirectly by measuring similarity between features 
  from the same class.

  Args:
      val_loader: DataLoader for the validation set.
      teacher_model: The teacher model used for generating pre-bottleneck features.
      projector: The projector network trained with SupCon loss.
      metric_fn: Function to calculate similarity between features (e.g., cosine_similarity).

  Returns:
      Average similarity between features from the same class.
  """
  projector.eval()
  teacher_model.eval()

  similarities = AverageMeter()

  with torch.no_grad():
    for images, labels in val_loader:
      images = images.cuda(non_blocking=True) if torch.cuda.is_available() else images
      bsz = labels.shape[0]

      # Forward pass through the teacher model
      outputs, pre_out, pre_pool = teacher_model(images)

      # Get features from the projector
      features, _ = projector(pre_pool)

      # Calculate pairwise similarities within each mini-batch
      for i in range(bsz):
        target_label = labels[i]
        # Find indices of images with the same label
        same_label_indices = (labels == target_label).nonzero(as_tuple=True)[0]
        if len(same_label_indices) > 1:  # Avoid empty batches or single-element batches
          anchor_feature = features[i]
          other_features = features[same_label_indices]
          # Calculate similarity between anchor and other features with the same label
          similarities_batch = metric_fn(anchor_feature.unsqueeze(0), other_features)
          similarities.update(similarities_batch.mean().item(), 1)  # Average over other features

  return similarities.avg
###############################################################################
def evaluate_distances(val_loader, teacher_model, projector):
  """
  Evaluates the projector indirectly by measuring Euclidean distance between features 
  from the same class.

  Args:
      val_loader: DataLoader for the validation set.
      teacher_model: The teacher model used for generating pre-bottleneck features.
      projector: The projector network trained with SupCon loss.

  Returns:
      Average Euclidean distance between features from the same class.
  """
  projector.eval()
  teacher_model.eval()

  distances = AverageMeter()

  with torch.no_grad():
    for images, labels in val_loader:
      images = images.cuda(non_blocking=True) if torch.cuda.is_available() else images
      bsz = labels.shape[0]

      # Forward pass through the teacher model
      outputs, pre_out, pre_pool = teacher_model(images)

      # Get features from the projector
      features, _ = projector(pre_pool)

      # Calculate pairwise distances within each mini-batch
      for i in range(bsz):
        target_label = labels[i]
        # Find indices of images with the same label
        same_label_indices = (labels == target_label).nonzero(as_tuple=True)[0]
        if len(same_label_indices) > 1:  # Avoid empty batches or single-element batches
          anchor_feature = features[i]
          other_features = features[same_label_indices]
          # Calculate Euclidean distance between anchor and other features with the same label
          distances_batch = torch.norm(anchor_feature.unsqueeze(0) - other_features, dim=1)
          distances.update(distances_batch.mean().item(), 1)  # Average over other features

  return distances.avg

###############################################################################
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

def evaluate_clustering(val_loader, teacher_model, projector, num_clusters):
  """
  Evaluates the projector indirectly using Davies-Bouldin Index on projected features.

  Args:
      val_loader: DataLoader for the validation set.
      teacher_model: The teacher model used for generating pre-bottleneck features.
      projector: The projector network trained with SupCon loss.
      num_clusters: Number of clusters for k-means clustering.

  Returns:
      Davies-Bouldin Index score (lower is better).
      
      
  Choosing num_clusters:
      The optimal number of clusters depends on your dataset and the inherent structure of the features 
      learned by the projector. Experiment with different values of num_clusters and evaluate the Davies-Bouldin
      Index score (lower is better) to find a reasonable balance between capturing class separation and avoiding
      overfitting.        
  """
  projector.eval()
  teacher_model.eval()

  all_features = []
  all_labels = []

  with torch.no_grad():
    for images, labels in val_loader:
      images = images.cuda(non_blocking=True) if torch.cuda.is_available() else images
      bsz = labels.shape[0]

      # Forward pass through the teacher model
      outputs, pre_out, pre_pool = teacher_model(images)

      # Get features from the projector
      features, _ = projector(pre_pool)

      all_features.append(features.cpu().detach().numpy())
      all_labels.append(labels.cpu().detach().numpy())

  # Concatenate features and labels from all mini-batches
  all_features = np.concatenate(all_features, axis=0)
  all_labels = np.concatenate(all_labels, axis=0)

  # Perform k-means clustering on projected features
  kmeans = KMeans(n_clusters=num_clusters, random_state=0)
  kmeans.fit(all_features)
  cluster_labels = kmeans.labels_

  # Calculate Davies-Bouldin Index score
  db_score = davies_bouldin_score(all_features, cluster_labels)

  return db_score

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
#measure accuracy
###############################################################################
def evaluate_resnet(model,val_loader):
   #print("model_name")
    model.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        #print("model_name")

        inputs, targets = inputs.cuda(), targets.cuda()
        outputs, _, _ = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
    print('Test Accuracy: %.3f%% (%d/%d)' % (100. * correct / total, correct, total))
###############################################################################
#Our training function
###############################################################################
def train_SCKD(train_loader, teacher_model, student_model, projector_t, interpreter, optimizer ,beta,mse_criterion, l1_criterion, ce_criterion ,printfreq, epoch, dataset ,st_path):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    teacher_model.eval()
    projector_t.eval()
    interpreter.eval()

    # switch to train mode
    student_model.train()

    end = time.time()

    for i, (images, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = images.cuda()

        optimizer.zero_grad()
        
        # compute output
        teacher_output, teacher_pre_out, teacher_pre_pool = teacher_model(input_var)
        student_output, student_pre_out, student_pre_pool = student_model(input_var)
        # print("student_pre_pool")
        # print(student_pre_pool.shape)
        
        # print('teacher_pre_pool')
        # print(teacher_pre_pool.shape)
        _, factor_t = projector_t(teacher_pre_pool)
        _, factor_s = interpreter(student_pre_pool)

        #print(factor_t.detach(), factor_s)
        # mse_loss = mse_criterion(ft_normalize(factor_s), ft_normalize(factor_t.detach()))

        # mse_loss = mse_criterion(factor_s, factor_t.detach())
        # l1_loss = l1_criterion(factor_s, factor_t.detach())
        l1_loss = l1_criterion(ft_normalize(factor_s), ft_normalize(factor_t.detach()))

        ce_loss = ce_criterion(student_output, target)

        loss = beta * l1_loss + ce_loss
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        output = student_output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % printfreq == 0:
            mse_loss_val = l1_loss.item()
            ce_loss_val = ce_loss.item()
            
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

            # with open(st_path+ ".txt", 'a') as f:
            #     print('Epoch: [{0}][{1}/{2}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #         epoch, i, len(train_loader), batch_time=batch_time,
            #         data_time=data_time, loss=losses, top1=top1), file=f)

##########################################################################################

def train_SCKD_diffarch(train_loader, teacher_model, student_model, projector_t, interpreter, optimizer ,beta, mse_criterion , l1_criterion, ce_criterion ,printfreq, epoch, dataset ,st_path,regressor,regressor_optimizer):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    teacher_model.eval()
    projector_t.eval()
    interpreter.eval()
    # switch to train mode
    student_model.train()
 
    regressor.train()

    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        
        optimizer.zero_grad()
        regressor_optimizer.zero_grad()
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = images.cuda()


        
        # compute output
        teacher_output, teacher_pre_out, teacher_pre_pool = teacher_model(input_var)
        student_output, student_pre_out, student_pre_pool = student_model(input_var)
        # print("student_pre_pool")
        # print(student_pre_pool.shape)
        out=regressor(student_pre_pool)
        # print('regressor_out')
        # print(out.shape)
        # print('teacher_pre_pool')
        # print(teacher_pre_pool.shape)

        _, factor_t = projector_t(teacher_pre_pool)
        # print('projector_out')
        # print(factor_t.shape)
        _, factor_s = interpreter(out)

        #print(factor_t.detach(), factor_s)
        # mse_loss = mse_criterion(ft_normalize(factor_s), ft_normalize(factor_t.detach()))
        # l1_loss = l1_criterion(factor_s, factor_t.detach())
        l1_loss = l1_criterion(ft_normalize(factor_s), ft_normalize(factor_t.detach()))

        ce_loss = ce_criterion(student_output, target)

        loss = beta * l1_loss + ce_loss
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        regressor_optimizer.step()

        output = student_output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % printfreq == 0:
            l1_loss_val = l1_loss.item()
            ce_loss_val = ce_loss.item()
            
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

            # with open(st_path+ ".txt", 'a') as f:
            #     print('Epoch: [{0}][{1}/{2}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #         epoch, i, len(train_loader), batch_time=batch_time,
            #         data_time=data_time, loss=losses, top1=top1), file=f)


