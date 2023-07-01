import os
import sys
import copy
import math
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.blocks import(
    TNetBlock,
    EdgeConvBlock
)

class DGCNNBackbone(nn.Module):
    def __init__(self,k=16) -> None:
        super().__init__()
        self.tnet = TNetBlock()
        self.edgeconvblock1 = EdgeConvBlock(
            k=k,
            in_ch=3,
            out_ch=128
        )
        self.edgeconvblock2 = EdgeConvBlock(
            k=k,
            in_ch=128,
            out_ch=128
        )
        #TODO: SHALL IT BE MLP OR CONV1D?
        #self.mlp1 = nn.Linear(256,1024)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels=256,out_channels=1024,kernel_size=1),
            nn.ReLU()
        )

        self.edgeconvblock3 = EdgeConvBlock(
            k=k,
            in_ch=1024,
            out_ch=1024
        )
        self.edgeconvblock4 = EdgeConvBlock(
            k=k,
            in_ch=1024,
            out_ch=1024
        )
        #TODO: SHALL IT BE MLP OR CONV1D?
        #self.mlp1 = nn.Linear(256,1024)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(in_channels=2048,out_channels=1024,kernel_size=1),
            nn.ReLU()
        )
        logging.info(self)
    def forward(self,x):
        #T-Net
        # b,c,N
        # x:b,3,N
        # trans_mat = self.tnet(x)
        # x = torch.bmm(x.permute(0,2,1),trans_mat).permute(0,2,1) #->x:b,3,N
        #EdgeConv Unit 1
        x1 = self.edgeconvblock1(x)
        x2 = self.edgeconvblock2(x1)
        x = torch.concat((x1,x2),dim=1)#->b,256,N
        x = self.mlp1(x)
        #EdgeConv Unit 2
        x3 = self.edgeconvblock3(x)
        x4 = self.edgeconvblock4(x3)
        x = torch.concat((x3,x4),dim=1)#->b,512,N
        x = self.mlp2(x)#->b,1024,N
        x = x.max(dim=-1, keepdim=False)[0]#->b,1024
        return x
