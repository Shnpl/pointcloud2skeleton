import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
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
from modules.utils import accuracy,collate_fn_alt

# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 18:54
# @Author  : Jiaan Chen, Hao Shi
# Modified from "https://github.com/qq456cvb/Point-Transformers/tree/master/models/Hengshuang"

import torch.nn as nn

from modules.utils import skeleton_centralize,decode_batch_sa_simdr,GaussianBlur,skeleton_rasterize,accuracy
if __name__ == "__main__":
    for epoch in range(4,5):
        # net = Estimate3DPoseModule(n=2048)
        net = torch.load(f"logs/Jun_29-f7f0de16/{epoch}_net.pt")
        net = net.to(torch.device('cuda'))
        net = net.eval()
        dataset:Dataset = NTU_RGBD_Dataset(select_length =2,point_number=2048,train=False,small_size_dataset=False)
        dataloader = DataLoader(dataset,batch_size=1,shuffle=True,collate_fn=collate_fn_alt)
        cnt = 0
        acc_total = 0
        for data in dataloader:
            # Read Data
            if data['pointclouds'] == []:
                continue
            
            if data['names'][0].find('S002') != -1:
                continue
            if data['names'][0].find('C002') != -1:
                continue
            if data['names'][0].find('C003') != -1:
                continue
            pointcloud_data = data['pointclouds'].float() # (BxF)xNx3
            gt_skeletons = data['skeletons'].float()[:,:,:3] # (BxF)xNx3
            pointcloud_data[:,:,2] -=3.5
            gt_skeletons[:,:,2] -= 3.5

            
            
            
            # Net
            x = pointcloud_data.permute(0,2,1).to(torch.device('cuda'))# b,3,N
            pred = net(x)
            
            # Move the result and ground truth skeleton to zero point
            # x: bx25x3
            acc, avg_acc = accuracy(pred,gt_skeletons,thr=0.1)
            acc_total += sum(acc)
            print(data['names'],acc)
            cnt += 2
        acc_avg = acc_total/cnt
        print(f"Epoch:{epoch},Acc:{acc_avg}")
        #visualize(pred[0].cpu().detach().numpy(),gt_skeletons[0],pointcloud_data[0])
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
        #break
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