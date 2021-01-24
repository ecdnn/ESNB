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
from models import densenet
import re

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
from utils.pruner import DenseNetPruner

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
    
    
class DenseNetworkPruning(object):
    def __init__(self, args):
        self.args = args
        self.model = self.init_model(args)
        self.data_loader = self.init_data(args)
        self.pruner = DenseNetPruner(self.model, args.dense_layer_nums)
        
    def test(self, solution):
        model_pruned, dense_layer_nums_pruned = self.pruner.prune(solution)
        top1_avg, top5_avg = self.validate(self.data_loader, model_pruned.cuda(), self.args)
        # top1_avg, top5_avg = self.validate(self.data_loader, self.model.cuda(), self.args)
        return top1_avg.item(), top5_avg.item(), dense_layer_nums_pruned
        
    def init_model(self, args):
        print("=> creating model '{}'".format(args.arch))
        model = densenet.__dict__[args.arch]() # resnet.resnet50().cuda() # model = models.__dict__[args.arch]() 
        # Removing the DataParallel Wrapper using the following code
        # torch.save(model.module.state_dict(), "/root/workspace/home/zy/code/Imagenet/resnet50/model_best_no_module.pth.tar")
        # model = torch.nn.DataParallel(model).cuda()
        
        if args.load_model:
            print("=> loading checkpoint '{}'".format(args.model_path))
            state_dict = torch.load(args.model_path)
            # Pre-process checkpoint
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
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
            num_workers=args.workers, pin_memory=True)
        
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
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
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
                    images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

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