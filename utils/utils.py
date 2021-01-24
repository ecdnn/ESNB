import argparse
import os
import random
import shutil
import time
import warnings
import pickle
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from models import resnet

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Dataset
from PIL import Image

from abc import ABCMeta, abstractmethod
from math import *


class Pruner(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass
        
    @abstractmethod
    def prune(self):
        pass
    

cudnn.benchmark = True

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

# ImageData Loader (Memory version)
class SampledImageNetDataset(Dataset):
    def __init__(self, samples, transform):
        self.data_tuples = [] 
        self.transform = transform
        for (path, label) in samples:
            img = self.pil_loader(path)
            if self.transform:
                img = self.transform(img)
            self.data_tuples.append((img, label))
            
    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        
    def __getitem__(self, index):
        return self.data_tuples[index][0], self.data_tuples[index][1]
 
    def __len__(self):
        return len(self.data_tuples)
    
    
class NetworkPruning(object):
    def __init__(self, args):
        print("Initializing NetworkPruning...")
        self.args = args
        self.model = self.init_model(args)
        self.data_loader = self.init_data(args)
        self.pruner = args.pruner(self.model, args.num_blocks)
        
    def test(self, solution):
        model_pruned, block_nums_pruned = self.pruner.prune(solution)
        top1_avg, top5_avg = self.validate(self.data_loader, model_pruned.cuda(), self.args)
        del model_pruned
        return top1_avg.item(), top5_avg.item(), block_nums_pruned
        
    def init_model(self, args):
        print("=> creating model '{}'".format(args.arch))
        model = resnet.__dict__[args.arch]() # resnet.resnet50().cuda() # model = models.__dict__[args.arch]() 
        # Removing the DataParallel Wrapper using the following code
        # torch.save(model.module.state_dict(), "/root/workspace/home/zy/code/Imagenet/resnet50/model_best_no_module.pth.tar")
        # model = torch.nn.DataParallel(model).cuda()
        
        if args.load_model:
            print("=> loading checkpoint '{}'".format(args.model_path))
            state_dict = torch.load(args.model_path)
            model.load_state_dict(state_dict)
        
        return model
    
    
    def init_data(self, args):
        # Data Sampling
        print("Sampling images from each {}...".format(args.sampled_imgs))
        with open(args.sampled_imgs, 'rb') as f:
            sampled_imgs = pickle.load(f)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # Saving data in memory
        test_dataset = SampledImageNetDataset(sampled_imgs, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
        ## Sampling End ##

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)
        
        return test_loader
    
    
    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
    
    def validate(self, val_loader, model, args):
        batch_time = AverageMeter('Time', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, top5],
            prefix='Test: ')

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                if args.gpu is not None:
                    images = images.cuda()
                    target = target.cuda()

                # compute output
                output = model(images)
                
                # measure accuracy and record loss
                acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # if i % args.print_freq == 0:
                #     progress.display(i)

            # TODO: this should also be done with the ProgressMeter
            # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            #       .format(top1=top1, top5=top5))
        return top1.avg, top5.avg
    
def compute_pk(nn_pruner): 
    skip_list = []

    num_blocks = nn_pruner.args.num_blocks
    for i, x in enumerate(num_blocks):
        skip_list.append(sum(num_blocks[:i]))

    blocks = nn_pruner.pruner.get_blocks_from_model(nn_pruner.model)
    
    pk_list = []
    for block in blocks:
        conv_layer_idx_list = [1,3,5,-2]
        pk_block = 0
        for layer_idx in conv_layer_idx_list:
            weight_conv = list(block.modules())[layer_idx].weight.data.cpu().numpy()
            mean_indicator = np.sum(np.abs(weight_conv)) / np.prod(weight_conv.shape)
            pk_block += mean_indicator
        pk_block /= len(conv_layer_idx_list)
        pk_list.append(pk_block)

    pk_npy = np.array(pk_list)
    pk_npy_normalized = pk_npy/np.max(pk_npy)
    
    pk_npy_normalized_list = list(pk_npy_normalized)
    pk_npy_normalized_list_final = []
    for i, x in enumerate(pk_npy_normalized_list):
        if i not in skip_list:
            pk_npy_normalized_list_final.append(x)

    return pk_npy_normalized_list_final