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

from modules.dgcnn_backbone import DGCNNBackbone
from modules.pointnet_backbone import PointNetBackbone
from modules.point_transformer_backbone import PointTransformerBackbone
from modules.ntu_rgbd_dataset import _NTU_RGBD_Dataset as NTU_RGBD_Dataset
from modules.losses import MPJPELoss,JointLengthLoss,SymmetryLoss
from modules.heads import Heads,SeparatedHeads
from modules.utils import collate_fn_alt

def main():
    exp_date = "_".join(time.asctime().split(' ')[1:3])
    exp_id = uuid.uuid1()
    exp_name = exp_date+"-"+str(exp_id)[0:8]
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
        net = torch.nn.Sequential(
            PointNetBackbone(),
            SeparatedHeads(num_joints=21)
        ).to(torch.device('cuda'))
    elif backbone == 'dgcnn':
        net = torch.nn.Sequential(
            DGCNNBackbone(k=16),
            SeparatedHeads(num_joints=21)
        ).to(torch.device('cuda'))
    elif backbone == 'pointtransformer':
        batchsize =2
        logger.info(f"batchsize:{batchsize}")
        select_length = 1e8
        logger.info(f"select length:{select_length}")
        net = torch.nn.Sequential(
            PointTransformerBackbone(),
            SeparatedHeads(num_joints=21)
        ).to(torch.device('cuda'))
    else:
        raise"No Backbone"
    
    init_lr = 1e-4
    lr_gamma = 0.999
    optimizer = torch.optim.Adam(params=net.parameters(),lr =init_lr,eps=1e-8,betas=(0.9,0.999))

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer = optimizer,
        gamma = lr_gamma
        )
    logger.info(f"init lr: {init_lr}, lr gamma:{lr_gamma}")

    loss_func1 = MPJPELoss()
    loss_func2 = JointLengthLoss(use_hand=False)
    loss_func3 = SymmetryLoss()
    dataset:Dataset = NTU_RGBD_Dataset(select_length =select_length,
                                       point_number=2048,
                                       small_size_dataset=True,
                                       pre_downsample=True)
    dataloader = DataLoader(dataset,batch_size=batchsize,shuffle=True,collate_fn=collate_fn_alt)
    tmp_count = 0
    for epoch in range(50):
        torch.save(net,      f"{log_path}/{epoch}_net.pt")
        torch.save(optimizer,f"{log_path}/{epoch}_opt.pt")
        torch.save(lr_scheduler,f"{log_path}/{epoch}_lr.pt")
        logger.info(f"save checkpoint:{log_path}..")
        logger.info(f"lr:{lr_scheduler.get_last_lr()}")
        discarded = 0
        for it_count,data in enumerate(dataloader):
            # Read Data
            try:
                pointcloud_data_batch = data['pointclouds']

                skeletons_batch = data['skeletons'][:,:,0:3]# (BxF)xNx3
                # for i in range(batchsize):                
                #     if data['skeletons']["skeleton_num"][i] > 1:
                        
                #         logger.debug(f"dropped:{data['name'][i]}")
                #         discarded += 1
                #     else:
                #         logger.debug(f"accepted:{data['name'][i]}")
                #         pointcloud_data.append(data['pointcloud'][i])
                #         skeletons.append(data['skeletons']['data'][i])
                if  len(pointcloud_data_batch) > 0:
                #     pointcloud_data = torch.stack(pointcloud_data) # BxFxNx3

                #     pointcloud_data = torch.reshape(pointcloud_data,(-1,pointcloud_data.shape[2],3))# (BxF)xNx3
                #     for pointcloud in pointcloud_data:
                #         pointcloud[:,0] = pointcloud[:,0] -torch.mean(pointcloud[:,0])
                #         pointcloud[:,1] = pointcloud[:,1] -torch.mean(pointcloud[:,1])
                #         pointcloud[:,2] = pointcloud[:,2] -torch.mean(pointcloud[:,2])
                #     skeletons = torch.stack(skeletons)
                #     skeletons = torch.reshape(skeletons,(-1,skeletons.shape[2],12))
                    net.zero_grad()
                    mini_batchsize = 8
                    for mini_i in range(0,len(pointcloud_data_batch),mini_batchsize):
                        if mini_i+mini_batchsize > len(pointcloud_data_batch):
                            break
                        pointcloud_data = pointcloud_data_batch[mini_i:mini_i+mini_batchsize,:,:]
                        skeletons = skeletons_batch[mini_i:mini_i+mini_batchsize,:,:]
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
                        #x = x.reshape(bs,-1)
                        for i in range(bs):
                            sk_tmp = skeletons[i]
                            sk_tmp[:,0] = sk_tmp[:,0] - sk_tmp[0,0]
                            sk_tmp[:,1] = sk_tmp[:,1] - sk_tmp[0,1]
                            sk_tmp[:,2] = sk_tmp[:,2] - sk_tmp[0,2]
                        loss_mjpe3d = loss_func1(x,skeletons)
                        loss_length = loss_func2(x,skeletons)
                        loss_symmetry = loss_func3(x)
                        loss = 0.7*loss_mjpe3d+0.2*loss_length+0.1*loss_symmetry
                        loss.backward()
                    optimizer.step()
                    logger.info(f"Epoch:{epoch},batch:{(it_count+1)*batchsize}/{dataset.__len__()}train_loss:{loss.detach()},loss_mpjpe:{loss_mjpe3d.detach()},loss_len:{loss_length.detach()},loss_symm:{loss_symmetry.detach()}")
                    tmp_count += 1
                    writer.add_scalar('train loss',loss.detach(),tmp_count)
            except Exception as e:
                print(e)
                continue
        lr_scheduler.step()
if __name__ == "__main__":
    main()