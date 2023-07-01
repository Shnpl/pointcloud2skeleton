import os
import sys

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.blocks import(
    TNetBlock,
    EdgeConvBlock
)

class Heads(nn.Module):
    def __init__(self,in_ch =1024,num_joints = 21) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.num_joints = num_joints
        self.mlp1_1 = nn.Sequential(
            nn.Linear(self.in_ch,4096),
            nn.ReLU()
        )
        self.mlp1_2 = nn.Sequential(
            nn.Linear(4096,1024),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(nn.Linear(1024,512),nn.ReLU())
        self.mlp4 = nn.Linear(512,3*self.num_joints)
        logging.info(self)
    def forward(self,x):
        x = self.mlp1_1(x)
        x = self.mlp1_2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        batchsize = x.shape[0]
        x = x.reshape(batchsize,self.num_joints,3)
        return x
    
class SeparatedHeads(nn.Module):
    def __init__(self,in_ch =1024,num_joints = 21) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.num_joints = num_joints
        # self.mlp1_1 = nn.Sequential(
        #     nn.Linear(self.in_ch,4096),
        #     nn.ReLU()
        # )
        # self.mlp1_2 = nn.Sequential(
        #     nn.Linear(4096,1024),
        #     nn.ReLU()
        # )
        self.mlp3 = nn.Sequential(nn.Linear(self.in_ch,512),nn.ReLU())
        self.mlp4 = nn.Sequential(nn.Linear(512,128*self.num_joints),nn.ReLU())
        self.mlp_x = nn.Linear(128,1)
        self.mlp_y = nn.Linear(128,1)
        self.mlp_z = nn.Linear(128,1)
        logging.info(self)
    def forward(self,x):
        # x = self.mlp1_1(x)
        # x = self.mlp1_2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        if len(x.shape) == 1:
            batchsize = 1
        else:
            batchsize = x.shape[0]
        x = x.reshape(batchsize,self.num_joints,-1)
        pred_x = self.mlp_x(x)
        pred_y = self.mlp_y(x)
        pred_z = self.mlp_z(x)
        return torch.concat((pred_x,pred_y,pred_z),dim=2)

class SeparatedClassifierHeads(nn.Module):
    def __init__(self,in_ch =1024,num_joints = 21,spatial_quality = 250) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.num_joints = num_joints
        self.spatial_quality = spatial_quality
        # self.mlp1_1 = nn.Sequential(
        #     nn.Linear(self.in_ch,4096),
        #     nn.ReLU()
        # )
        # self.mlp1_2 = nn.Sequential(
        #     nn.Linear(4096,1024),
        #     nn.ReLU()
        # )
        self.mlp3 = nn.Sequential(nn.Linear(in_ch,256),nn.LeakyReLU())
        self.mlp4 = nn.Sequential(nn.Linear(256,128*self.num_joints),nn.LeakyReLU())
        self.mlp_x = nn.Linear(128,spatial_quality)
        self.mlp_y = nn.Linear(128,spatial_quality)
        self.mlp_z = nn.Linear(128,spatial_quality)
        logging.info(self)
    def forward(self,x):
        # x = self.mlp1_1(x)
        # x = self.mlp1_2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        if len(x.shape) == 1:
            batchsize = 1
        else:
            batchsize = x.shape[0]
        x = x.reshape(batchsize,self.num_joints,-1)
        pred_x = self.mlp_x(x)
        pred_y = self.mlp_y(x)
        pred_z = self.mlp_z(x)
        return pred_x,pred_y,pred_z