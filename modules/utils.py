import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import logging
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
    # Voxel DownSampling
    pcd = pcd.voxel_down_sample(voxel_size=0.001)
    if visualize:
        o3d.visualization.draw_geometries([pcd], window_name='Open3D downSample', width=1920, height=1080, left=50, top=50,
                                        point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)
    #Remove abnormals
    if remove_abnormals:
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        pcd = pcd.select_by_index(ind)
        if visualize:
            o3d.visualization.draw_geometries([pcd], window_name='Open3D downSample', width=1920, height=1080, left=50, top=50,
                                            point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)
       
    # FPS
    pcd_numpy = np.asarray(pcd.points)
    pcd_numpy = farthest_point_sample(pcd_numpy,downsample_number)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd_numpy)) 
    if visualize:
        o3d.visualization.draw_geometries([pcd], window_name='Open3D downSample', width=1920, height=1080, left=50, top=50,
                                        point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)
    
    pcd_numpy = np.asarray(pcd.points)
    # Invert Z
    pcd_numpy[:,2] = - pcd_numpy[:,2]
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
        if 'skeletons' in datum.keys():
            skeletons.append(datum['skeletons']['data'])
        if 'pointcloud' in datum.keys():
            pointclouds.append(torch.tensor(datum['pointcloud']))
    if skeletons != []:
        skeletons = torch.concat(skeletons,dim=0)
        pointclouds = torch.concat(pointclouds,dim=0)
    out_data['names'] = names
    out_data['skeletons'] = skeletons
    out_data['pointclouds'] = pointclouds
    return out_data