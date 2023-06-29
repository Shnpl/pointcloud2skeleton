import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
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
from modules.point_transformer_backbone import PointTransformerBackbone
from modules.ntu_rgbd_dataset import _NTU_RGBD_Dataset as NTU_RGBD_Dataset
from modules.itop_dataset import _ITOP_Dataset as ITOP_Dataset
from modules.losses import MPJPELoss,JointLengthLoss,SymmetryLoss,KLDiscretLoss
from modules.heads import Heads,SeparatedHeads,SeparatedClassifierHeads
from modules.utils import (
    collate_fn_alt,
    skeleton_centralize,
    skeleton_rasterize,
    GaussianBlur,
    log_grad_norm,
    decode_batch_sa_simdr,
    accuracy
    )


def main():
    exp_date = "_".join(time.asctime().split(' ')[1:3])
    exp_id = uuid.uuid1()
    exp_name = exp_date+"-"+str(exp_id)[0:8]
    print(exp_name)
    log_path = f"logs/{exp_name}"
    writer = SummaryWriter(log_path)
    logging.basicConfig(
                     format = '[%(asctime)s][%(name)s][%(levelname)s][%(funcName)s]:%(message)s',
                     level=logging.DEBUG,
                     filename=f"{log_path}/log.txt",
                     filemode='w')
    logger = logging.getLogger("train_logger")
    
    logger.info(f"start time:{time.asctime()}")
    logger.info(f"exp name:{exp_name}")
    #
    backbone = 'pointtransformer'
    logger.info(f"backbone:{backbone}")
    if backbone == 'pointnet':
        batchsize =2
        select_length = 1e8
        net = torch.nn.Sequential(
            PointNetBackbone(),
            SeparatedClassifierHeads(num_joints=21)
        ).to(torch.device('cuda'))
    elif backbone == 'dgcnn':
        batchsize =2
        select_length = 1e8
        net = torch.nn.Sequential(
            DGCNNBackbone(k=16),
            SeparatedHeads(num_joints=20)
        ).to(torch.device('cuda'))
    elif backbone == 'pointtransformer':
        batchsize =5
        logger.info(f"batchsize:{batchsize}")
        select_length = 1e8
        logger.info(f"select length:{select_length}")
        net = torch.nn.Sequential(
            PointTransformerBackbone(num_points=2048),
            SeparatedHeads(num_joints=21,in_ch=512)
        ).to(torch.device('cuda'))
    else:
        raise"No Backbone"
    net.train()
    init_lr = 1e-4
    lr_gamma = 0.996
    optimizer = torch.optim.Adam(params=net.parameters(),lr =init_lr,eps=1e-8,betas=(0.9,0.999))

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer = optimizer,
        gamma = lr_gamma
        )
    logger.info(f"init lr: {init_lr}, lr gamma:{lr_gamma}")

    loss_func1 = MPJPELoss()
    #loss_func1 = KLDiscretLoss()
    loss_func2 = JointLengthLoss(use_hand=False)
    loss_func3 = SymmetryLoss()

    dataset:Dataset = NTU_RGBD_Dataset(select_length =select_length,
                                       point_number=2048,
                                       small_size_dataset=False,
                                       pre_downsample=True)
    dataloader = DataLoader(dataset,batch_size=batchsize,shuffle=True,collate_fn=collate_fn_alt)

    # dataset:Dataset = NTU_RGBD_Dataset(point_number=2048,train=True)
    # dataloader = DataLoader(dataset,batch_size=batchsize,shuffle=True)
    for epoch in range(140):
        torch.save(net,      f"{log_path}/{epoch}_net.pt")
        torch.save(optimizer,f"{log_path}/{epoch}_opt.pt")
        torch.save(lr_scheduler,f"{log_path}/{epoch}_lr.pt")
        logger.info(f"save checkpoint:{log_path}..")
        logger.info(f"lr:{lr_scheduler.get_last_lr()}")

        for it_count,data in enumerate(dataloader):
            # Read Data
            try:

                pointcloud_data_batch = data['pointclouds']
                skeletons_batch = data['skeletons'][:,:,0:3]# (BxF)xNx3

                index_order = torch.randperm(pointcloud_data_batch.shape[0])
                pointcloud_data_batch = torch.index_select(pointcloud_data_batch,dim=0,index=index_order)
                skeletons_batch = torch.index_select(skeletons_batch,dim=0,index=index_order)

                if  len(pointcloud_data_batch) > 0:                    
                    mini_batchsize = 8
                    for mini_i in range(0,len(pointcloud_data_batch),mini_batchsize):
                        net.zero_grad()
                        if mini_i+mini_batchsize > len(pointcloud_data_batch):
                            break
                        pointcloud_data = pointcloud_data_batch[mini_i:mini_i+mini_batchsize,:,:]
                        pointcloud_data = pointcloud_data.float()
                        skeletons = skeletons_batch[mini_i:mini_i+mini_batchsize,:,:]
                        pointcloud_data[:,:,2] -= 3.5
                        skeletons[:,:,2] -= 3.5
                        for i in range(batchsize):
                            rand_rot1 = torch.rand(1)*torch.pi*2
                            rand_rot_m_1= torch.tensor([[torch.cos(rand_rot1),-torch.sin(rand_rot1),0],
                                                        [torch.sin(rand_rot1), torch.cos(rand_rot1),0],
                                                        [0,                    0,                   1]]).T
                            rand_rot2 = torch.rand(1)*torch.pi*2
                            rand_rot_m_2= torch.tensor([[1,0,                   0],
                                                        [0,torch.cos(rand_rot2),-torch.sin(rand_rot2)],
                                                        [0,torch.sin(rand_rot2), torch.cos(rand_rot2)]]).T
                            rand_rot3 = torch.rand(1)*torch.pi*2
                            rand_rot_m_3= torch.tensor([[torch.cos(rand_rot3),0,torch.sin(rand_rot3)],
                                                        [0,                   1,  0],
                                                        [-torch.sin(rand_rot3),0, torch.cos(rand_rot3)]]).T
                            rand_move = torch.rand((1,3))
                            pointcloud_data[i] = torch.mm(pointcloud_data[i],rand_rot_m_1) #N,3
                            pointcloud_data[i] = torch.mm(pointcloud_data[i],rand_rot_m_2) #N,3
                            pointcloud_data[i] = torch.mm(pointcloud_data[i],rand_rot_m_3) #N,3
                            pointcloud_data[i] += rand_move
                            skeletons[i]  = torch.mm(skeletons[i],rand_rot_m_1)#k,3
                            skeletons[i]  = torch.mm(skeletons[i],rand_rot_m_2)#k,3
                            skeletons[i]  = torch.mm(skeletons[i],rand_rot_m_3)#k,3
                            skeletons[i] += rand_move
                        x = pointcloud_data.permute(0,2,1).to(torch.device('cuda'))# b,3,N

                        
                # for i in range(batchsize):
                #     base_coord = skeletons[i][8] - torch.tensor([0.1,0.1,0.1])
                #     x[i,0] = x[i,0] - base_coord[0]
                #     x[i,1] = x[i,1] - base_coord[1]
                #     x[i,2] = x[i,2] - base_coord[2]
                #     skeletons[i,:,0] -= base_coord[0]
                #     skeletons[i,:,1] -= base_coord[1]
                #     skeletons[i,:,2] -= base_coord[2]
                
                        skeletons = skeletons.float().to(torch.device('cuda'))

                # Net
                # 
                #     
                #     x_tmp[0] = x_tmp[0] - torch.mean(x_tmp[0])
                #     x_tmp[1] = x_tmp[1] - torch.mean(x_tmp[1])
                #     x_tmp[2] = x_tmp[2] - torch.mean(x_tmp[2])
                #pred_x,pred_y,pred_z = net(x)
                        pred = net(x)
                # pred_z = pred_z-2        
                # # gt_skeleton_x,gt_skeleton_y,gt_skeleton_z = skeleton_centralize(skeletons)
                # gt_skeleton_x = skeletons[:,0:21,0]
                # gt_skeleton_y = skeletons[:,0:21,1]
                # gt_skeleton_z = skeletons[:,0:21,2] 

                # gt_skeleton_x = GaussianBlur(skeleton_rasterize(gt_skeleton_x),sigma=5).to(torch.device('cuda'))
                # gt_skeleton_y = GaussianBlur(skeleton_rasterize(gt_skeleton_y),sigma=5).to(torch.device('cuda'))
                # gt_skeleton_z = GaussianBlur(skeleton_rasterize(gt_skeleton_z),sigma=5).to(torch.device('cuda'))

                #target_weight = torch.ones(21).to(torch.device('cuda'))
                        loss_mpjpe3d = loss_func1.forward(pred,skeletons)
                # gt_skeleton_x = gt_skeleton_x[:,1:,:]
                # gt_skeleton_y = gt_skeleton_y[:,1:,:]
                # gt_skeleton_z = gt_skeleton_z[:,1:,:]
                # loss = loss_func1.forward(pred_x,pred_y,pred_z,gt_skeleton_x,gt_skeleton_y,gt_skeleton_z,target_weight)
                # #(x,skeletons)
                        loss_length = loss_func2(pred,skeletons)
                        loss_symmetry = loss_func3(pred)
                        loss = 0.7*loss_mpjpe3d+0.2*loss_length+0.1*loss_symmetry
                # decode_batch_pred = decode_batch_sa_simdr(pred_x, pred_y,pred_z)
                # decode_batch_label = decode_batch_sa_simdr(gt_skeleton_x, gt_skeleton_y,gt_skeleton_z)

                # acc, avg_acc, cnt, pred = accuracy(decode_batch_pred, decode_batch_label, hm_type='sa-simdr', thr=10)
                        loss.backward()
                        grad_norm = log_grad_norm(net)
                        optimizer.step()
                        #print(loss.detach())
                        log_info = f"\
                            Epoch:{epoch},\n\
                            batch:{(it_count+1)*batchsize}/{dataset.__len__()},\n\
                            loss:{loss.detach()},\n\
                            loss_mpjpe:{loss_mpjpe3d.detach()}\n\
                            loss_length:{loss_length.detach()}\n\
                            loss_symmetry:{loss_symmetry.detach()}\n\
                            grad_norm:{grad_norm},\n"
                            #acc:{[f'{acci:.3f}' for acci in acc[1:]]},\n"
                            #avrage_acc:{avg_acc}\n"
                        logger.info(log_info)
                        print(log_info)
                    
                    # tmp_count += 1
                    # writer.add_scalar('train loss',loss.detach(),tmp_count)
            except Exception as e:
                print(e)
                continue
        lr_scheduler.step()
        print(lr_scheduler.get_lr())
if __name__ == "__main__":
    main()