import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import sys
sys.path.append(".")
sys.path.append(".")
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import numpy as np

from modules.ntu_rgbd_dataset import _NTU_RGBD_Dataset as NTU_RGBD_Dataset
from modules.losses import MPJPELoss,JointLengthLoss
from modules.utils import collate_fn_alt

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import cv2
import numpy as np
import time
from modules.pku_mmd_dataset import _PKU_MMD_Dataset as PKU_MMD_Dataset
from modules.ntu_rgbd_dataset import _NTU_RGBD_Dataset as NTU_RGBD_Dataset

if __name__ == "__main__":
    for epoch in range(17):
        # net = Estimate3DPoseModule(n=2048)
        net:nn.Module = torch.load(f"logs/0622A/{epoch}_net.pt").to(torch.device('cuda'))
        dataset:Dataset = NTU_RGBD_Dataset(select_length =2,point_number=2048,train=False,pre_downsample=True)
        dataloader = DataLoader(dataset,batch_size=1,shuffle=True,collate_fn=collate_fn_alt)
        loss_func1 = MPJPELoss()
        loss_func2 = JointLengthLoss(use_hand=False)
        net = net.eval()
        total_len = len(dataset)
        losses = []
        for it_count,data in enumerate(dataloader):
            # Read Data
            if data['pointclouds'] == []:
                continue
            pointcloud_data = data['pointclouds'] # BxFxNx3
            skeletons = data['skeletons']
            skeletons = skeletons[:,:,0:3]# (BxF)xNx3
            #TODO:顺序是否一致?
            x = pointcloud_data.permute(0,2,1).float().to(torch.device('cuda'))# b,3,N
            skeletons = skeletons.float().to(torch.device('cuda'))

            # Net
            x:Tensor = net(x)
            # Move the result and ground truth skeleton to zero point
            # x: bx25x3
            bs = x.shape[0]
            for i in range(bs):
                sk_tmp = x[i]
                sk_tmp[:,0] = sk_tmp[:,0] - sk_tmp[0,0]
                sk_tmp[:,1] = sk_tmp[:,1] - sk_tmp[0,1]
                sk_tmp[:,2] = sk_tmp[:,2] - sk_tmp[0,2]
            for i in range(bs):
                sk_tmp = skeletons[i]
                sk_tmp[:,0] = sk_tmp[:,0] - sk_tmp[0,0]
                sk_tmp[:,1] = sk_tmp[:,1] - sk_tmp[0,1]
                sk_tmp[:,2] = sk_tmp[:,2] - sk_tmp[0,2]
            loss1 = loss_func1(x,skeletons)
            loss2 = loss_func2(x,skeletons)
            loss = 0.7*loss1+0.3*loss2
            print(f'\r{it_count}/{total_len}',end="")
            losses.append(loss.detach().cpu())
        print(f"epoch:{epoch},loss:{torch.mean(torch.tensor(losses))}")
   