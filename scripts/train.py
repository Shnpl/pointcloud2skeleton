import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys
sys.path.append(".")
import logging
import time
import uuid

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter   
from torch.utils.data import Dataset,DataLoader

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
    accuracy,
    arg_parse
    )


def main():
    args = arg_parse()
    exp_date = "_".join(time.asctime().split(' ')[1:3])
    if args.exp_name == '':
        exp_name = str(uuid.uuid1())[0:8]
    else:
        exp_name = args.exp_name
    exp_name = exp_date+"-"+exp_name

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
    backbone = args.backbone_type
    logger.info(f"backbone:{backbone}")
    batchsize = 16
    if backbone == 'pointnet':
        mini_batchsize = 192
        select_length = 1e8
        net = torch.nn.Sequential(
            PointNetBackbone(),
            SeparatedHeads(num_joints=21,in_ch =1024)
        ).to(torch.device('cuda'))
    elif backbone == 'dgcnn':
        select_length = 1e8
        net = torch.nn.Sequential(
            DGCNNBackbone(k=16),
            SeparatedHeads(num_joints=20)
        ).to(torch.device('cuda'))
    elif backbone == 'pointtransformer':
        mini_batchsize = 8
        logger.info(f"batchsize:{batchsize}")
        select_length = 1e8
        logger.info(f"select length:{select_length}")
        net = torch.nn.Sequential(
            PointTransformerBackbone(num_points=2048),
            SeparatedHeads(num_joints=21,in_ch=512)
        ).to(torch.device('cuda'))
    else:
        raise"No Backbone"
    init_lr = 1e-4
    lr_gamma = 0.996
    optimizer = torch.optim.Adam(params=net.parameters(),lr =init_lr,eps=1e-8,betas=(0.9,0.999))

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer = optimizer,
        gamma = lr_gamma
        )
    logger.info(f"init lr: {init_lr}, lr gamma:{lr_gamma}")
    acc_thr = 0.1
    logger.info(f"acc_thr: {acc_thr}")
    loss_func1 = MPJPELoss()
    #loss_func1 = KLDiscretLoss()
    loss_func2 = JointLengthLoss(use_hand=False)
    loss_func3 = SymmetryLoss()

    train_dataset:Dataset = NTU_RGBD_Dataset(select_length =select_length,
                                       point_number=2048,
                                       pre_downsample=True,
                                       train=True,
                                       specific_setting='S001',
                                       specific_camera_id='C001'
                                       )
    train_dataloader = DataLoader(train_dataset,batch_size=batchsize,shuffle=True,collate_fn=collate_fn_alt,num_workers=1,pin_memory=True)
    val_dataset:Dataset = NTU_RGBD_Dataset(select_length =4,
                                           point_number=2048,
                                           train=False,
                                           specific_setting='S001',
                                           specific_camera_id='C001'
                                           )
    val_dataloader = DataLoader(val_dataset,batch_size=1,shuffle=False,collate_fn=collate_fn_alt,num_workers=1,pin_memory=True)

    # dataset:Dataset = NTU_RGBD_Dataset(point_number=2048,train=True)
    # dataloader = DataLoader(dataset,batch_size=batchsize,shuffle=True)
    batch_count = 0
    for epoch in range(140):
        torch.save(net,      f"{log_path}/{epoch}_net.pt")
        torch.save(optimizer,f"{log_path}/{epoch}_opt.pt")
        torch.save(lr_scheduler,f"{log_path}/{epoch}_lr.pt")
        logger.info(f"save checkpoint:{log_path}..")
        net.train()
        
        epoch_grad_norm = []
        epoch_loss = []
        epoch_loss_mpjpe3d = []
        epoch_loss_length = []
        epoch_loss_symmetry = []
        epoch_acc = []
        for it_count,data in enumerate(train_dataloader):
            
            try:
                pointcloud_data_batch = data['pointclouds']
                skeletons_batch = data['skeletons'][:,:,0:3]# (BxF)xNx3

                index_order = torch.randperm(pointcloud_data_batch.shape[0])
                pointcloud_data_batch = torch.index_select(pointcloud_data_batch,dim=0,index=index_order)
                skeletons_batch = torch.index_select(skeletons_batch,dim=0,index=index_order)
                actual_batchsize = len(pointcloud_data_batch)

                batch_grad_norm = []
                batch_loss = []
                batch_loss_mpjpe3d = []
                batch_loss_length = []
                batch_loss_symmetry = []
                batch_acc = []
                for mini_i in range(0,actual_batchsize,mini_batchsize):
                    net.zero_grad()
                    if mini_i+mini_batchsize > len(pointcloud_data_batch):
                        break
                    pointcloud_data = pointcloud_data_batch[mini_i:mini_i+mini_batchsize,:,:].float()
                    skeletons = skeletons_batch[mini_i:mini_i+mini_batchsize,:,:].float()
                    pointcloud_data[:,:,2] -= 3.5
                    skeletons[:,:,2] -= 3.5
                    for i in range(mini_batchsize):
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
                    
                    skeletons = skeletons.to(torch.device('cuda'))

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
                        
                    # gt_skeleton_x = gt_skeleton_x[:,1:,:]
                    # gt_skeleton_y = gt_skeleton_y[:,1:,:]
                    # gt_skeleton_z = gt_skeleton_z[:,1:,:]
                    # loss = loss_func1.forward(pred_x,pred_y,pred_z,gt_skeleton_x,gt_skeleton_y,gt_skeleton_z,target_weight)
                    # #(x,skeletons)
                        
                    # decode_batch_pred = decode_batch_sa_simdr(pred_x, pred_y,pred_z)
                    # decode_batch_label = decode_batch_sa_simdr(gt_skeleton_x, gt_skeleton_y,gt_skeleton_z)

                    # acc, avg_acc, cnt, pred = accuracy(decode_batch_pred, decode_batch_label, hm_type='sa-simdr', thr=10)
                    # Loss
                    loss_mpjpe3d = loss_func1.forward(pred,skeletons)
                    loss_length = loss_func2(pred,skeletons)
                    loss_symmetry = loss_func3(pred)
                    loss = 0.7*loss_mpjpe3d+0.2*loss_length+0.1*loss_symmetry
                    loss.backward()
                    optimizer.step()
                    # Log
                    grad_norm = log_grad_norm(net)
                    acc,avg_acc = accuracy(pred,skeletons,thr=acc_thr)
                    
                    batch_grad_norm.append(grad_norm)
                    batch_loss.append(loss.detach().cpu())
                    batch_loss_mpjpe3d.append(loss_mpjpe3d.detach().cpu())
                    batch_loss_length.append(loss_length.detach().cpu())
                    batch_loss_symmetry.append(loss_symmetry.detach().cpu())
                    batch_acc.append(avg_acc)

                    
                batch_count += 1
                logger.info(f'Epoch:{epoch},Current epoch status{(it_count+1)*batchsize}/{train_dataset.__len__()}')
                print(f'Epoch:{epoch},Current epoch status{(it_count+1)*batchsize}/{train_dataset.__len__()}')

                write_log(logger,writer,batch_count,'batch',
                          batch_grad_norm,
                          batch_loss,
                          batch_loss_mpjpe3d,
                          batch_loss_length,
                          batch_loss_symmetry,
                          batch_acc
                          )
                epoch_grad_norm.append(user_mean(batch_grad_norm))
                epoch_loss.append(user_mean(batch_loss))
                epoch_loss_mpjpe3d.append(user_mean(batch_loss_mpjpe3d))
                epoch_loss_length.append(user_mean(batch_loss_length))
                epoch_loss_symmetry.append(user_mean(batch_loss_symmetry))
                epoch_acc.append(user_mean(batch_acc))

                                
            except Exception as e:
                print(e)
                continue
        # Epoch end
        lr_scheduler.step()
        # Write info about this epoch:
        last_lr = lr_scheduler.get_last_lr()[0]
        logger.info(last_lr)
        writer.add_scalar('train lr',last_lr,epoch)
        write_log(logger,writer,epoch,'epoch',
                    epoch_grad_norm,
                    epoch_loss,
                    epoch_loss_mpjpe3d,
                    epoch_loss_length,
                    epoch_loss_symmetry,
                    epoch_acc
                          )

        with torch.no_grad():
            net.eval()
            eval_acc = []
            for data in val_dataloader:
                if  data['pointclouds'] == []:
                    continue
                pointcloud_data = data['pointclouds'].float() # (BxF)xNx3
                gt_skeletons = data['skeletons'].float()[:,:,:3] # (BxF)xNx3
                pointcloud_data[:,:,2] -=3.5
                gt_skeletons[:,:,2] -= 3.5
                x = pointcloud_data.permute(0,2,1).to(torch.device('cuda'))# b,3,N
                pred = net(x)
                acc, avg_acc = accuracy(pred,gt_skeletons,thr=0.1)
                eval_acc.append(avg_acc)
            acc_avg = user_mean(eval_acc)
            logger.info(f'Epoch:{epoch},Eval Acc:{acc_avg}')
            print(f'Epoch:{epoch},Eval Acc:{acc_avg}')
            writer.add_scalar(f'Eval Acc',acc_avg,epoch)
            
def user_mean(data:list):
    return sum(data)/len(data)

def write_log(logger:logging.Logger,writer:SummaryWriter,count,type:str,
                grad_norm,
                loss,
                loss_mpjpe3d,
                loss_length,
                loss_symmetry,
                acc,
):
    logger.info(f'train grad_norm per {type}:{user_mean(grad_norm)}')
    print(f'train grad_norm per {type}:{user_mean(grad_norm)}')
    writer.add_scalar(f'train grad_norm per {type}',user_mean(grad_norm),count)

    logger.info(f'train loss per {type}:{user_mean(loss)}')
    print(f'train loss per {type}:{user_mean(loss)}')
    writer.add_scalar(f'train loss per {type}',user_mean(loss),count)

    logger.info(f'train loss_mpjpe per {type}:{user_mean(loss_mpjpe3d)}')
    print(f'train loss_mpjpe per {type}:{user_mean(loss_mpjpe3d)}')
    writer.add_scalar(f'train loss_mpjpe per {type}',user_mean(loss_mpjpe3d),count)
    
    logger.info(f'train loss_length per {type}:{user_mean(loss_length)}')
    print(f'train loss_length per {type}:{user_mean(loss_length)}')
    writer.add_scalar(f'train loss_length per {type}',user_mean(loss_length),count)

    logger.info(f'train loss_symmetry per {type}:{user_mean(loss_symmetry)}')
    print(f'train loss_symmetry per {type}:{user_mean(loss_symmetry)}')
    writer.add_scalar(f'train loss_symmetry per {type}',user_mean(loss_symmetry),count)

    logger.info(f'train acc per {type}:{user_mean(acc)}')
    print(f'train acc per {type}:{user_mean(acc)}')
    writer.add_scalar(f'train acc per {type}',user_mean(acc),count)

if __name__ == "__main__":
    main()