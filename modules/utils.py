import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import logging
import random
from torch import Tensor
from torch.nn import Module
def read_and_preprocess_pointcloud_data(pcd,visualize = False,downsample_number = 4096,remove_abnormals = False):
    def farthest_point_sample(point, npoint):
        N, D = point.shape
        xyz = point[:,:3]
        centroids = np.zeros((npoint,))
        distance = np.ones((N,)) * 1e10
        farthest = np.random.randint(0, N)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, -1)
        point = point[centroids.astype(np.int32)]
        return point
    def display_inlier_outlier(cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name='Open3D Removal Outlier', width=1920,
                                        height=1080, left=50, top=50, point_show_normal=False, mesh_show_wireframe=False,
                                        mesh_show_back_face=False)
    # # Voxel DownSampling
    # pcd = pcd.voxel_down_sample(voxel_size=0.001)
    # if visualize:
    #     o3d.visualization.draw_geometries([pcd], window_name='Open3D downSample', width=1920, height=1080, left=50, top=50,
    #                                     point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)
    # #Remove abnormals
    # if remove_abnormals:
    #     cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    #     pcd = pcd.select_by_index(ind)
    #     if visualize:
    #         o3d.visualization.draw_geometries([pcd], window_name='Open3D downSample', width=1920, height=1080, left=50, top=50,
    #                                         point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)
       
    # FPS
    pcd_numpy = np.asarray(pcd.points)
    pcd_numpy = farthest_point_sample(pcd_numpy,downsample_number)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd_numpy)) 
    # if visualize:
    #     o3d.visualization.draw_geometries([pcd], window_name='Open3D downSample', width=1920, height=1080, left=50, top=50,
    #                                     point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)
    
    # pcd_numpy = np.asarray(pcd.points)
    # Invert Z
    # pcd_numpy[:,2] = - pcd_numpy[:,2]
    return pcd_numpy

def read_skeleton_data(skeleton_data_path,select_length = 1e8):
    logger = logging.getLogger('dataread_logger')
    try:
        with open(skeleton_data_path,'r') as f:
            frame_number = int(f.readline())
            data = []
            for frame in range(min(frame_number,select_length)):
                skeleton = {}
                frame_data = []
                skeleton["skeleton_num"] = int(f.readline())
                misc = f.readline().split(' ')
                skeleton["skeleton_id"] = misc[0]
                skeleton['clipedEdges'] = misc[1]
                skeleton['handLeftConfidence'] = misc[2]
                skeleton['handLeftState'] = misc[3]
                skeleton['handRightConfidence'] = misc[4]
                skeleton['handRightState'] = misc[5]
                skeleton['isResticted'] = misc[6]
                skeleton['xOffset'] = misc[7]
                skeleton['yOffset'] = misc[8]
                skeleton['trackingState'] = misc[9]
                skeleton['joint_num'] = int(f.readline())
                for i in range(skeleton['joint_num']):
                    joint_data=f.readline().replace("\n","").split(' ')
                    joint_data = torch.tensor([float(data) for data in joint_data])
                    frame_data.append(joint_data)
                frame_data = torch.stack(frame_data)
                skeleton['data'] = frame_data
                data.append(skeleton)
                if skeleton["skeleton_num"] > 1:
                    for extra_person in range(skeleton["skeleton_num"]-1):
                        for i in range(27):
                            f.readline()
        final_skeleton_data = {}
        final_skeleton_data['skeleton_num'] = int(data[0]['skeleton_num'])
        final_skeleton_data['skeleton_id'] = str(data[0]['skeleton_id'])
        final_skeleton_data['joint_num'] = int(data[0]['joint_num'])
        final_skeleton_data['data'] = torch.stack([datum['data'] for datum in data])
    except:
        logger.critical(f"Fail to read skeleton{skeleton_data_path}")
        raise
    return final_skeleton_data

def collate_fn_alt(data):
    out_data = {}
    names = []
    skeletons = []
    pointclouds = []
    for datum in data:
        if datum == None:
            continue
        names.append(datum['name'])
        frame_num,_,_=datum['skeletons']['data'].shape
        start_rand_idx = random.randint(0,frame_num//2-1)
        end_rand_idx = random.randint(frame_num//2+1,frame_num)

        if 'skeletons' in datum.keys():
            skeletons.append(datum['skeletons']['data'][start_rand_idx:end_rand_idx,:,:])
        if 'pointcloud' in datum.keys():
            pointclouds.append(torch.tensor(datum['pointcloud'])[start_rand_idx:end_rand_idx,:,:])
    if skeletons != []:
        skeletons = torch.concat(skeletons,dim=0)
        pointclouds = torch.concat(pointclouds,dim=0)
    out_data['names'] = names
    out_data['skeletons'] = skeletons
    out_data['pointclouds'] = pointclouds
    return out_data

def GaussianBlur(x:Tensor,sigma:int):
    # from one-hot to gaussian
    # x:b,joint_num,spatial_quality
    bs,joint_num,spatial_quality = x.shape
    out = torch.zeros((bs,joint_num,spatial_quality))#b,joint_num,spatial_quality 
    for b in range(bs):
        for joint in range(joint_num):
            x_tmp = x[b,joint,:]
            one_hot_index = x_tmp.argmax(dim=-1)
            out[b,joint,:] = torch.linspace(0,spatial_quality,spatial_quality,dtype=torch.int32)
            out[b,joint,:] = torch.abs(out[b,joint,:]-one_hot_index)
            out[b,joint,:] = torch.exp(-(out[b,joint,:]**2)/(2*sigma**2))        
    return out                  

def skeleton_centralize(skeletons:Tensor):
    bs = skeletons.shape[0]
    for i in range(bs):
        sk_tmp = skeletons[i]
        sk_tmp[:,0] = sk_tmp[:,0] - sk_tmp[0,0]
        sk_tmp[:,1] = sk_tmp[:,1] - sk_tmp[0,1]
        sk_tmp[:,2] = sk_tmp[:,2] - sk_tmp[0,2]
    gt_skeleton_x = skeletons[:,0:21,0]
    gt_skeleton_y = skeletons[:,0:21,1]
    gt_skeleton_z = skeletons[:,0:21,2]
    return gt_skeleton_x,gt_skeleton_y,gt_skeleton_z

def skeleton_rasterize(x:Tensor,spatial_quality = 250):
    #x:b,joint_num
    bs,joint_num = x.shape
    x1 = torch.round(x*100) + spatial_quality/2
    out = torch.zeros((bs,joint_num,spatial_quality))#b,joint_num,spatial_quality
    for b in range(bs):
        for joint in range(joint_num):
            idx = int(x1[b,joint])
            out[b,joint,idx]=1
    return out

def log_grad_norm(net:Module):
    sqsum = 0.0
    for p in net.parameters():
        if p.grad is None:
            continue
        sqsum += (p.grad ** 2).sum().item()
    return np.sqrt(sqsum)
    logging.info(f)
    print(f"grad_norm{np.sqrt(sqsum)}")
def decode_batch_sa_simdr(output_x, output_y,output_z):

    max_val_x, preds_x = output_x.max(2, keepdim=True)
    max_val_y, preds_y = output_y.max(2, keepdim=True)
    max_val_z, preds_z = output_z.max(2, keepdim=True)

    output = torch.ones([output_x.size(0), preds_x.size(1), 3])
    output[:, :, 0] = torch.squeeze(preds_x)
    output[:, :, 1] = torch.squeeze(preds_y)
    output[:, :, 2] = torch.squeeze(preds_z)
    output = output.cpu().numpy()
    preds = output.copy()

    return preds


def accuracy(output, target, hm_type='sa-simdr', thr=5, resolution=250):
    """
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    "https://github.com/leeyegy/SimDR"
    """
    idx = list(range(output.shape[0]))
    norm = 1.0
    if hm_type == 'sa-simdr':
        pred = output
        target = target
        #norm = np.ones((pred.shape[0], 3)) * np.array([resolution, resolution,resolution]) / 10
    # dists => [batch, 13]
    dists = calc_dists(pred, target)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]], thr)
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred

def calc_dists(preds, target):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[0], preds.shape[1]))
    for b in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            # if target[b, j, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[b, j, :]
                normed_targets = target[b, j, :]
                dists[b, j] = np.linalg.norm(normed_preds - normed_targets)
            #else:
                #dists[c, n] = -1
    return dists
def dist_acc(dists, thr=0.5):
    """ Return percentage below threshold while ignoring values with a -1 """
    # dist => [batch,]
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1
def square_distance(src:Tensor, dst:Tensor):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    # batchsize,n,_ = src.shape

    # distances = torch.zeros(batchsize, n, n)
    sum_sq_src = torch.sum(src**2,dim=-1).unsqueeze(2)
    sum_sq_dst = torch.sum(dst**2,dim=-1).unsqueeze(1)
    dists = torch.sqrt(sum_sq_src+sum_sq_dst-2*torch.bmm(src,dst.permute(0,2,1)))
    return dists
    # for i in range(n):
    #     diff = src[:, i, None] - dst[:, None]  # 广播操作
    # distance_sq = torch.sum(diff ** 2, dim=-1)
    # distances[:, i] = distance_sq
    # return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)
