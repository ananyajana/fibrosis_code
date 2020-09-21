#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:19:42 2020

@author: aj611
"""


import torch.nn as nn
import torch.nn.functional as F
import torch

class Self_Attention(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attention, self).__init__()
        self.in_dim = in_dim

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
        