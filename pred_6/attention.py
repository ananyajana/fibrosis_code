#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:19:42 2020

@author: aj611
"""

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules import conv
from torch.nn.modules.utils import _pair

import pdb

os.makedirs("images_cifar10", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
#parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--n_dis", type = int, default = 2, help="discriminator critic iters")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

def _l2normalize(v, eps=1e-12):
    return v / (torch.norm(v) + eps)

def max_singular_value(W, u = None, Ip = 1):
    """
    power iteration for weight parameter
    """
    #xp  = W.data
    if not Ip >= 1:
        raise ValueError("Power iteration should be a positive integer")
    
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
    _u = u
    
    for _ in range(Ip):
        _v = _l2normalize(torch.matmul(_u, W.data))
        _u = _l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
    sigma = torch.sum(F.linear(_u, torch.transpose(W.data, 0, 1))* _v)
    return sigma, _u

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class SNConv2d(conv._ConvNd):
    r"""Applies a 2D convolution over an input signal composed of several input planes."""
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                False, _pair(0), groups, bias)
        self.register_buffer('u', torch.Tensor(1, out_channels).normal_())
        
    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma
    
    def forward(self, input):
        return F.conv2d(input, self.W_, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Self_Attention(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attention, self).__init__()
        self.in_dim = in_dim
        '''
        self.query_conv = SNConv2d(in_channels = in_dim, out_channels = in_dim//8, kernel_size = 1)
        self.key_conv = SNConv2d(in_channels = in_dim, out_channels = in_dim // 8, kernel_size= 1)
        self.val_conv = SNConv2d(in_channels = in_dim, out_channels = in_dim//2, kernel_size = 1)
        self.val_conv2 = SNConv2d(in_channels = in_dim //2, out_channels = in_dim, kernel_size = 1)
        '''
        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim//8, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // 8, kernel_size= 1)
        self.val_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim//2, kernel_size = 1)
        self.val_conv2 = nn.Conv2d(in_channels = in_dim //2, out_channels = in_dim, kernel_size = 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, x):
        batch_size, nc, h, w = list(x.size())
        location_num = h * w
        downsampled_num = location_num // 4
        
        query = self.query_conv(x)
        query = query.view(query.shape[0], -1, nc // 8) # B x W*H x NC/8
        
        key = F.max_pool2d(self.key_conv(x), 2)
        key = key.view(key.shape[0], downsampled_num, nc // 8).permute(0, 2, 1) # B x W*H/4 x NC/8
        
        val = F.max_pool2d(self.val_conv(x), 2)
        val = val.view(val.shape[0], downsampled_num, nc // 2) # B x W*H/4 x NC/2
        
        attn = torch.bmm(query, key) # B x W*H x W*H/4
        attn_weights = self.softmax(attn) # B x W*H x W*H/4
        
        attn_applied = torch.bmm(attn_weights, val) # B x W*H x NC/2
        out = self.val_conv2(attn_applied.view(val.shape[0], nc // 2, w, h))
        
        out = self.gamma * out + x
        return out
        
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 8
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 256 * self.init_size ** 2))
        #pdb.set_trace()
        self.l2 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2),
            SNConv2d(256, 256, 3, stride=1, padding=1))
        self.l3 = nn.Sequential(
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            SNConv2d(256, 128, 3, stride=1, padding=1))
        self.l4 = nn.Sequential(
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            SNConv2d(128, 64, 3, stride=1, padding=1))
        self.l5 = nn.Sequential(
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(64, opt.channels, 3, stride=1, padding=1))
        self.l6 = nn.Sequential(nn.Tanh())
        
        self.attn = Self_Attention(64)

    def forward(self, z):
        #pdb.set_trace()
        out = self.l1(z)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.attn(out)
        out = self.l5(out)
        out = self.l6(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

        self.l1 = nn.Sequential(
            #nn.Conv2d(opt.channels, 16, 3, 2, 1))
            SNConv2d(opt.channels, 16, 3, 2, 1))
        self.l2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Dropout2d(0.25),
            SNConv2d(16, 32, 3, 2, 1))
        self.l3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(32, 0.8),
            SNConv2d(32, 64, 3, 2, 1))
        self.l4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(64, 0.8),
            SNConv2d(64, 128, 3, 2, 1))
        self.l5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(128, 0.8)
            )
        self.attn = Self_Attention(64)
    def forward(self, img):
        #pdb.set_trace()
        out = self.l1(img)
        out = self.l2(out)
        out = self.l3(out)
        out = self.attn(out)
        out = self.l4(out)
        out = self.l5(out)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
'''
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)
'''
os.makedirs("../../../data/cifar10", exist_ok=True)        
dataloader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        "../../../data/cifar10",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        #pdb.set_trace()
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator twice
        # -----------------


        for _ in range(opt.n_dis):
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            #pdb.set_trace()
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            #pdb.set_trace()
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        real_loss.backward()

        
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        fake_loss.backward()
        
        d_loss = (real_loss + fake_loss) / 2
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images_cifar10/%d.png" % batches_done, nrow=5, normalize=True)
