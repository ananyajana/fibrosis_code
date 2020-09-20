#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 22:14:42 2020

@author: aj611
"""

import shutil
import time
import os
import math
import json
import torchvision

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
from models import BaselineNet, BaselineNet2
import numpy as np
from sklearn import metrics
from torch.autograd import Variable
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from dataset import my_collate_fn, PretrainLiverDataset3, PretrainLiverDataset4, PretrainLiverDataset5
from torch.utils.data import DataLoader
from models import ResNet_extractor2, FeaExtractor2, FeaExtractor3
import matplotlib.pyplot as plt
from collections import OrderedDict
import argparse
import logging

chkpt_str='checkpoint_fea_ex2_rotate_run2'
#os.environ["CUDA_VISIBLE_DEVICES"]="5, 6, 7"
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--num_class", type=int, default=2, help="number of rotations")
parser.add_argument("--data_path", type=str, default='', help="path to data folder")
parser.add_argument("--save_dir", type=str, default=''.format(chkpt_str), help="saves checkpoint in this dir")
parser.add_argument("--lr", type=int, default=0.01, help="learning rate")
parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
parser.add_argument("--num_feats", type=int, default=128, help="number of features")
parser.add_argument("--sup_model", type=str, default='fea_ex2', help="batch_size")
parser.add_argument("--log_file", type=str, default='log.txt', help="batch_size")
parser.add_argument("--method", type=str, default='', help="batch_size")
opt = parser.parse_args()
print(opt)
os.makedirs(opt.save_dir, exist_ok=True)
#full_log_path = save_dir + '/' + opt.log_file
#logging.basicConfig(filename=opt.full_log_path, level=logging.INFO)

logger = logging.getLogger('train_logger')
logger.setLevel(logging.INFO)
# create console handler and file handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
file_handler = logging.FileHandler('{:s}/train_log.txt'.format(opt.save_dir), mode='w')
file_handler.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter('%(asctime)s\t%(message)s', datefmt='%m-%d %I:%M')
# add formatter to handlers
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
# add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


if opt.sup_model == 'fea_ex2':
    data_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485],
                                         std=[0.229])
                ])

    #model = FeaExtractor2(in_features=opt.channels, num_features=opt.num_feats, num_classes=opt.num_class, pre_train=True)
    model = FeaExtractor3(in_features=opt.channels, num_features=opt.num_feats, num_classes=opt.num_class, pre_train=True)
    is_resnet = False
elif opt.sup_model =='resnet':
    data_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
    model = ResNet_extractor2(num_classes=opt.num_class, pre_train=True)
    is_resnet = True

model = model.cuda()
logger.info(model)

#model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).cuda()
#model = torch.nn.DataParallel(model, device_ids=[1,2,3,4,5,6,7]).cuda()
model = torch.nn.DataParallel(model).cuda()

if opt.method in 'rotate':
    train_set = PretrainLiverDataset3('{:s}/train/'.format(opt.data_path), data_transform, num_class=opt.num_class, is_resnet=is_resnet)
    test_set = PretrainLiverDataset3('{:s}/test/'.format(opt.data_path), data_transform, num_class=opt.num_class, is_resnet=is_resnet)
elif opt.method in 'downsize':
    train_set = PretrainLiverDataset4('{:s}/train/'.format(opt.data_path), data_transform, num_class=opt.num_class, is_resnet=is_resnet)
    test_set = PretrainLiverDataset4('{:s}/test/'.format(opt.data_path), data_transform, num_class=opt.num_class, is_resnet=is_resnet)
elif opt.method in 'smooth':
    train_set = PretrainLiverDataset5('{:s}/train/'.format(opt.data_path), data_transform, num_class=opt.num_class, is_resnet=is_resnet)
    test_set = PretrainLiverDataset5('{:s}/test/'.format(opt.data_path), data_transform, num_class=opt.num_class, is_resnet=is_resnet)


trainloader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=8, collate_fn=my_collate_fn)
testloader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=True, num_workers=8, collate_fn=my_collate_fn)

'''
model_path='./checkpoint_fea_ex2_rotate/checkpoint_best.pth.tar'
print("=> loading pre-trained model")
chkpt = torch.load(model_path)
new_chkpt=OrderedDict()
for k, v in chkpt.items():
    name = k[7:] # remove module
    new_chkpt[name] = v
model.load_state_dict(chkpt)
'''
#img, label = next(iter(trainloader))

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

best_epoch_loss = 100000
for epoch in range(opt.n_epochs):
    model = model.train()
    epoch_loss = 0
    test_epoch_loss = 0
    len_train = len(trainloader)
    for i, data in enumerate(trainloader):
    
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
    
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        
        outputs = model(inputs)
        ops = outputs.squeeze()
        
        optimizer.zero_grad()
        if opt.sup_model == 'fea_ex2':
            loss = criterion(outputs, labels.unsqueeze(1).unsqueeze(1))
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        pred = torch.argmax(outputs, dim=1).cpu()
        p = pred.squeeze().cpu()
        l = labels.squeeze().cpu()
        accuracy = (p == l).sum().detach().numpy()
        logger.info('Train: [{}][{}] epoch [{}] accuracy: {}'.format(i, len_train, epoch, accuracy))
        epoch_loss += loss
    epoch_loss = epoch_loss/len(trainloader)
    logger.info('Train: epoch {} loss: {} '.format(epoch, epoch_loss))
            
    model = model.eval()
    len_test = len(testloader)
    for i, data in enumerate(testloader):
    
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        
        outputs = model(inputs)
        ops = outputs.squeeze()
        
        if opt.sup_model == 'fea_ex2':
            loss = criterion(outputs, labels.unsqueeze(1).unsqueeze(1))
        else:
            loss = criterion(outputs, labels)
        pred = torch.argmax(outputs, dim=1).cpu()
        p = pred.squeeze().cpu()
        l = labels.squeeze().cpu()
        accuracy = (p == l).sum().detach().numpy()
        logger.info('Test: [{}][{}] epoch [{}] accuracy: {}'.format(i, len_test, epoch, accuracy))
        test_epoch_loss += loss
    test_epoch_loss = epoch_loss/len(testloader)
    logger.info('Test: epoch {} loss: {} '.format(epoch, test_epoch_loss))

    logger.info('going to save the model')
    if test_epoch_loss < best_epoch_loss:
        logger.info('saving the model')
        best_epoch_loss = test_epoch_loss
        torch.save(model.state_dict(), '{:s}/checkpoint_best.pth.tar'.format(opt.save_dir))
logger.info('training completed')
