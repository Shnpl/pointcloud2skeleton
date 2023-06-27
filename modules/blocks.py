import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeConvBlock(nn.Module):
    #TODO: There are several more method in the paper, shall we implement it?
    def __init__(self,k=20,in_ch = 3,out_ch = 64) -> None:
        super().__init__()
        self.k = k
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.bn = nn.BatchNorm2d(out_ch)
        self.conv = nn.Sequential(nn.Conv2d(in_ch*2, out_ch, kernel_size=1, bias=False),
                                   self.bn,
                                   nn.LeakyReLU(negative_slope=0.2))
    def forward(self,x):
        x = self._get_graph_feature(x, k=self.k)
        x = self.conv(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        return x1
    def _knn(self,x, k):
        inner = -2*torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
        idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
        return idx
    def _get_graph_feature(self,x, k=20, idx=None):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            idx = self._knn(x, k=k)   # (batch_size, num_points, k)
        device = torch.device('cuda')

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

        idx = idx + idx_base

        idx = idx.view(-1)
    
        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims) 
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
        return feature
        # batch_size = x.size(0)
        # num_points = x.size(2)
        # x = x.view(batch_size, -1, num_points)
        # if idx is None:
        #     idx = self._knn(x, k=k)   # (batch_size, num_points, k)
        # device = torch.device('cuda')

        # idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

        # idx = idx + idx_base

        # idx = idx.view(-1)
    
        # _, num_dims, _ = x.size()

        # x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        # feature = x.view(batch_size*num_points, -1)[idx, :]
        # feature = feature.view(batch_size, num_points, k, num_dims) 
        # x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        
        # feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
        # return feature

class TNetBlock(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Sequential(
            torch.nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            torch.nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            torch.nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 3),
        )



    def forward(self, x):
        batchsize = x.size()[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        trans_mat =torch.diag_embed(x)
        #repeat(batchsize,1).to(x.get_device())
        # iden = torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32)).view(1,9).repeat(batchsize,1)        
        # iden = iden.to(x.get_device())
        
        # x = x + iden
        # trans_mat = x.view(-1, 3, 3)
        return trans_mat






# class PointNet(nn.Module):
#     def __init__(self, args, output_channels=40):
#         super(PointNet, self).__init__()
#         self.args = args
#         self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
#         self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
#         self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
#         self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.bn3 = nn.BatchNorm1d(64)
#         self.bn4 = nn.BatchNorm1d(128)
#         self.bn5 = nn.BatchNorm1d(args.emb_dims)
#         self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
#         self.bn6 = nn.BatchNorm1d(512)
#         self.dp1 = nn.Dropout()
#         self.linear2 = nn.Linear(512, output_channels)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.bn4(self.conv4(x)))
#         x = F.relu(self.bn5(self.conv5(x)))
#         x = F.adaptive_max_pool1d(x, 1).squeeze()
#         x = F.relu(self.bn6(self.linear1(x)))
#         x = self.dp1(x)
#         x = self.linear2(x)
#         return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        #
        class args_C():
            def __init__(self) -> None:
                self.k = 20
                self.emb_dims = 1024
                self.dropout = 0.1
        args = args_C()
        #
        
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

if __name__ == "__main__":
    net = EdgeConvBlock().to(torch.device('cuda'))
    tnet = TNetBlock().to(torch.device('cuda'))
    
    x = torch.zeros((2,3,100)).to(torch.device('cuda'))
    x = tnet(x)
    x = net(x)
    pass