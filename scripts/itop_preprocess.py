import os
#os.environ['CUDA_VISIBLE_DEVICES']='2'
import sys
sys.path.append(".")
from torch.utils.tensorboard import SummaryWriter   
import logging
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import time
import uuid
import types
import numpy as np
from modules.dgcnn_backbone import DGCNNBackbone
from modules.pointnet_backbone import PointNetBackbone
from modules.itop_dataset import _ITOP_Dataset as ITOP_Dataset
from modules.losses import MPJPELoss,JointLengthLoss,SymmetryLoss,KLDiscretLoss
from modules.heads import Heads,SeparatedHeads,SeparatedClassifierHeads
dataset:Dataset = ITOP_Dataset(point_number=2048,train=False)
skeletons = np.zeros((4863,15,3))
pointclouds = np.zeros((4863,2048,3))
for it_count,data in enumerate(dataset):
    skeletons[it_count] = np.asarray(data['skeletons'])
    pointclouds[it_count] = np.asarray(data['pointclouds'])
    print(it_count)
np.save('skeletons',skeletons)
np.save('pointclouds',pointclouds)