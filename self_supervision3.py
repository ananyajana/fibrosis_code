#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:54:33 2020

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

from dataset import PretrainLiverDataset2
from torch.utils.data import DataLoader
from models import  FeaExtractorContextRestore2
import matplotlib.pyplot as plt
#from pre_processing import *
#os.environ["CUDA_VISIBLE_DEVICES"]="6,7"

import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=5000, help="number of epochs of training")
parser.add_argument("--n_cpu", type=int, default=30, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--num_class", type=int, default=2, help="number of rotations")
parser.add_argument("--data_path", type=str, default='./data/lits_131_pat_all_train/', help="path to data folder")
parser.add_argument("--save_dir", type=str, default='./chkpts/checkpoint_contextrestore2_13oct', help="saves checkpoint in this dir")
parser.add_argument("--lr", type=int, default=0.0001, help="learning rate")
parser.add_argument("--batch_size", type=int, default=30, help="batch_size")
parser.add_argument("--num_feats", type=int, default=256, help="batch_size")
opt = parser.parse_args()
print(opt)

train_data_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485],
                                     std=[0.229])
            ])
data_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485],
                                     std=[0.229])
            ])

#os.environ["CUDA_VISIBLE_DEVICES"]="6, 7"
os.makedirs(opt.save_dir, exist_ok=True)

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


train_set = PretrainLiverDataset2('{:s}/train/'.format(opt.data_path), 40, 10, data_transform, random_mirror=False)
test_set = PretrainLiverDataset2('{:s}/test/'.format(opt.data_path), 40, 10, data_transform)

trainloader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=8)
testloader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=True, num_workers=8)

model = FeaExtractorContextRestore2(in_features=1, num_features=opt.num_feats, pre_train=True)
model = model.cuda()
print(model)

# run from the previous checkpointed model
#model_path='{:s}/checkpoint_best.pth.tar'.format(opt.save_dir)
'''
model_path='{:s}/checkpoint_best.pth.tar'.format('./chkpts/checkpoint_contextrestore2_10oct')
print("=> loading pre-trained model from {:s}".format(model_path))
chkpt = torch.load(model_path)
from collections import OrderedDict
new_chkpt=OrderedDict()
for k, v in chkpt.items():
    name = k[7:] # remove module
    new_chkpt[name] = v
model.load_state_dict(new_chkpt)
'''
model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).cuda()


'''
model_path='{:s}/checkpoint_best.pth.tar'.format(opt.save_dir)
print("=> loading pre-trained model")
chkpt = torch.load(model_path)
from collections import OrderedDict
new_chkpt=OrderedDict()
for k, v in chkpt.items():
    name = k[7:] # remove module
    new_chkpt[name] = v
model.load_state_dict(new_chkpt)

#model.load_state_dict(chkpt)
img, tampered = next(iter(trainloader))
plt.imshow(img.squeeze().detach().cpu().numpy()[0])
plt.imshow(tampered.squeeze().detach().cpu().numpy()[0])
outputs = model(tampered.cuda())
plt.imshow(outputs.squeeze().detach().cpu().numpy()[0])
plt.imshow(img.squeeze().detach().cpu().numpy()[0])
plt.imshow(tampered.squeeze().detach().cpu().numpy()[0])
'''


criterion = nn.MSELoss()
criterion = criterion.cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

best_epoch_loss = 100000
for epoch in range(opt.n_epochs):
    model = model.train()
    epoch_loss = 0
    test_epoch_loss = 0
    for i, data in enumerate(trainloader):
    
        # get the inputs; data is a list of [inputs, labels]
        img, tampered = data
        
        img = Variable(img.cuda())
        tampered = Variable(tampered.cuda())
        
        outputs = model(tampered)
        optimizer.zero_grad()
        #loss = criterion(outputs, img)
        loss_mse = criterion(outputs, img)
        loss = torch.sqrt(loss_mse)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss = epoch_loss/len(trainloader)
    logger.info('Train: epoch {} loss: {} '.format(epoch, epoch_loss))

    for i, data in enumerate(testloader):
    
        # get the inputs; data is a list of [inputs, labels]
        img, tampered = data
        
        img = Variable(img.cuda())
        tampered = Variable(tampered.cuda())
        
        outputs = model(tampered)
        optimizer.zero_grad()
        #loss = criterion(outputs, img)
        loss_mse = criterion(outputs, img)
        loss = torch.sqrt(loss_mse)

        test_epoch_loss += loss.item()
    test_epoch_loss = test_epoch_loss/len(testloader)
    logger.info('Test: epoch {} loss: {} '.format(epoch, test_epoch_loss))
    if test_epoch_loss < best_epoch_loss:
        best_epoch_loss = test_epoch_loss
        logger.info('Saving model at Test epoch {} loss: {} '.format(epoch, test_epoch_loss))
        torch.save(model.state_dict(), '{:s}/checkpoint_best.pth.tar'.format(opt.save_dir))

