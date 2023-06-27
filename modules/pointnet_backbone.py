import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(2048, 1024, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)#->64
        x2 = self.conv2(x1)#->64
        x3 = self.conv3(x2)#->128
        x4 = self.conv4(x3)#->256
        x = torch.concat((x1,x2,x3,x4),dim=1)#->512
        x = self.conv5(x)#->1024
        x_max = self.max_pool(x).squeeze()
        x_mean = self.avg_pool(x).squeeze()
        x = torch.concat((x_max,x_mean),dim = 1)#->2048
        x = self.linear1(x)
        return x
