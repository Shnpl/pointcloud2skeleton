import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys
sys.path.append(".")
import torch
from torch import Tensor
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import numpy as np
import time
from modules.ntu_rgbd_dataset import _NTU_RGBD_Dataset as NTU_RGBD_Dataset
import open3d as o3d
from visualize import visualize
import logging


# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 18:54
# @Author  : Jiaan Chen, Hao Shi
# Modified from "https://github.com/qq456cvb/Point-Transformers/tree/master/models/Hengshuang"

import torch.nn as nn

from modules.utils import skeleton_centralize,decode_batch_sa_simdr,GaussianBlur,skeleton_rasterize,accuracy
if __name__ == "__main__":
    # net = Estimate3DPoseModule(n=2048)
    net = torch.load("logs/Jun_29-70d6c0d2/29_net.pt")
    net = net.to(torch.device('cuda'))
    net = net.eval()
    dataset:Dataset = NTU_RGBD_Dataset(select_length =1,point_number=2048,train=True,small_size_dataset=True)
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True)
    for data in dataloader:
        # Read Data
        print(data['name'])
        pointcloud_data = data['pointcloud'] # BxFxNx3
        pointcloud_data = torch.reshape(pointcloud_data,(-1,pointcloud_data.shape[2],3))# (BxF)xNx3
        skeletons = data['skeletons']['data']
        pointcloud_data[:,:,2] -=3.5
        skeletons = torch.reshape(skeletons,(-1,skeletons.shape[2],12))[:,:,0:3]# (BxF)xNx3
        x = pointcloud_data.permute(0,2,1).float().to(torch.device('cuda'))# b,3,N
        gt_skeletons = skeletons.float()
        
        gt_skeletons[:,:,2] -= 3.5
        # Net
        pred = net(x)
        
        # Move the result and ground truth skeleton to zero point
        # x: bx25x3
      
        visualize(pred[0].cpu().detach().numpy(),gt_skeletons[0],pointcloud_data[0])
        # for i in range(bs):
        #     sk_tmp = x[i]
        #     sk_tmp[:,0] = sk_tmp[:,0] - sk_tmp[0,0]
        #     sk_tmp[:,1] = sk_tmp[:,1] - sk_tmp[0,1]
        #     sk_tmp[:,2] = sk_tmp[:,2] - sk_tmp[0,2]
        # bs = x.shape[0]
        # for i in range(bs):
        #     sk_tmp = skeletons[i]
        #     sk_tmp[:,0] = sk_tmp[:,0] - sk_tmp[0,0]
        #     sk_tmp[:,1] = sk_tmp[:,1] - sk_tmp[0,1]
        #     sk_tmp[:,2] = sk_tmp[:,2] - sk_tmp[0,2]
        break
    ############
    # for i in range(bs):
    #     x_out = x[0]
    #     skeletons_out = skeletons[0]
    #     sk_predict = x_out.detach().to('cpu').numpy()
    #     sk_gt = skeletons_out.detach().to('cpu').numpy()
        
        # np.save(f'out/pred_{data["name"][i]}_{i}',sk_predict)
        # np.save(f'out/gt_{data["name"][i]}_{i}',sk_gt)

        # cloud_im = sk_predict

        # # xx= cloud_im[0,:]
        # # yy= cloud_im[1,:]
        # # zz= cloud_im[2,:]
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(cloud_im) 
        # o3d.visualization.draw_geometries([pcd], window_name='Open3D downSample', width=1920, height=1080, left=50, top=50,
        #                                 point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)