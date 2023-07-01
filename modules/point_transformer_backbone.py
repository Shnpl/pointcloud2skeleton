import os
import sys
import logging

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 18:54
# @Author  : Jiaan Chen, Hao Shi
# Modified from "https://github.com/qq456cvb/Point-Transformers/tree/master/models/Hengshuang"

# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 18:56
# @Author  : Jiaan Chen
# Modified from "https://github.com/qq456cvb/Point-Transformers/tree/master/models/Hengshuang"

# reference https://github.com/yanx27/Pointnet_Pointnet2_pytorch, modified by Yang You
sys.path.append(".")
from modules.utils import square_distance

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False, knn=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint

    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    if knn:
        dists = square_distance(new_xyz, xyz)  # B x npoint x N
        idx = dists.argsort()[:, :, :nsample]  # B x npoint x K
    else:
        idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, knn=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, knn=self.knn)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0].transpose(1, 2)
        return new_xyz, new_points


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k,MHA_enable =False) -> None:
        super().__init__()
        self.MHA_enable = MHA_enable
        if self.MHA_enable:
            self.head_num = 8
        else:
            self.head_num = 1
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model//self.head_num),
            nn.ReLU(),
            nn.Linear(d_model//self.head_num, d_model//self.head_num)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model//self.head_num, d_model//self.head_num),
            nn.ReLU(),
            nn.Linear(d_model//self.head_num, d_model//self.head_num)
        )
        
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)

        pre = features
        x = self.fc1(features)
        if self.MHA_enable:
            q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
            pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

            q= rearrange(q, 'b n (h d) -> b h n d', h = self.head_num)
            k= rearrange(k, 'b n k (h d) -> b h n k d', h = self.head_num)
            v= rearrange(v, 'b n k (h d) -> b h n k d', h = self.head_num)
            reses = []
            for i in range(self.head_num):
                qi = q[:,i,:,:]
                ki = k[:,i,:,:,:]
                vi = v[:,i,:,:,:]
                attn = self.fc_gamma(qi[:, :, None] - ki + pos_enc)
                attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)
                res = torch.einsum('bmnf,bmnf->bmf', attn, vi + pos_enc)
                reses.append(res)
            reses = torch.concat(reses,dim=-1)
            reses = self.fc2(reses) + pre
            return reses,None
        else:   
            q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

            pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

            attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
            attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

            res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
            res = self.fc2(res) + pre
            return res, attn

class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        # k is the number of center points in furthest point sample
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

    def forward(self, xyz, points):
        return self.sa(xyz, points)


class PointTransformerBackbone(nn.Module):
    def __init__(self, num_points = 2048):
        super().__init__()
        npoints = num_points
        nblocks = 4
        nneighbor = 20   # 16 for default
        d_points = 3  # dim for input points
        transformer_dim = 768
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))
            #self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))
        self.nblocks = nblocks
        logging.info(self)
    def forward(self, x):
        x = x.permute(0,2,1)
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            # points = self.transformers[2*i+1](xyz, points)[0]
        points = points.mean(1)
        return points


# class PointTransformerBackbone(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.backbone = Backbone(args)
#         self.args = args
#         self.sizeH = args.sensor_sizeH
#         self.sizeW = args.sensor_sizeW
#         self.num_joints = args.num_joints
#         nblocks = 4

#         self.fc2 = nn.Sequential(
#             nn.Linear(32 * 2 ** nblocks, 256),
#             nn.ReLU(),
#             nn.Linear(256, self.num_joints * 128),
#             nn.ReLU(),
#         )

#         self.mlp_head_x = nn.Linear(128, self.sizeW)
#         self.mlp_head_y = nn.Linear(128, self.sizeH)

#     def forward(self, x):
#         batch_size = x.size(0)

#         points = self.backbone(x)

#         res = self.fc2(points.mean(1))
#         res = res.view(batch_size, 13, -1)
#         pred_x = self.mlp_head_x(res)
#         pred_y = self.mlp_head_y(res)

#         return pred_x, pred_y

