import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import sys
sys.path.append(".")
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import cv2
import numpy as np
import time
from modules.dgcnn_backbone import Estimate3DPoseModule
from modules.pku_mmd_dataset import _PKU_MMD_Dataset as PKU_MMD_Dataset
from modules.ntu_rgbd_dataset import _NTU_RGBD_Dataset as NTU_RGBD_Dataset

if __name__ == "__main__":
    # net = Estimate3DPoseModule(n=2048)
    net = torch.load("logs/0617/14_net.pt").to(torch.device('cuda'))
    dataset:Dataset = NTU_RGBD_Dataset(select_length =8,point_number=2048)
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True)
    for it_count,data in enumerate(dataloader):
        # Read Data
        pointcloud_data = data['pointcloud'] # BxFxNx3
        pointcloud_data = torch.reshape(pointcloud_data,(-1,pointcloud_data.shape[2],3))# (BxF)xNx3
        skeletons = data['skeletons']['data']
        skeletons = torch.reshape(skeletons,(-1,skeletons.shape[2],12))[:,:,0:3]# (BxF)xNx3
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
        break
    for i in range(bs):
        x_out = x[0]
        skeletons_out = skeletons[0]
        sk_predict = x_out.detach().to('cpu').numpy()
        sk_gt = skeletons_out.detach().to('cpu').numpy()
        np.save(f'out/pred_{i}',sk_predict)
        np.save(f'out/gt_{i}',sk_gt)
        #     loss = loss_func(x,skeletons)
        #     loss.backward()
        #     optimizer.step()
        #     print(f"{time.asctime()},Epoch:{epoch},batch:{it_count*1}/{dataset.__len__()},train_loss:{loss.detach()}")
        # torch.save(net,f"{epoch}_net.pt")
        # torch.save(optimizer,f"{epoch}_opt.pt")