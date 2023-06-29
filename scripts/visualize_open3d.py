import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os

def read_and_preprocess_pointcloud_data(pcd,visualize = False,downsample_number = 4096):
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
    # if visualize:
    #     o3d.visualization.draw_geometries([pcd], window_name='Open3D downSample', width=1920, height=1080, left=50, top=50,
    #                                     point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)
    # Remove abnormals
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    # if visualize:
    #     o3d.visualization.draw_geometries([pcd], window_name='Open3D downSample', width=1920, height=1080, left=50, top=50,
    #                                     point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)
    pcd_numpy = np.asarray(pcd.points)   
    # FPS
    pcd_numpy = np.asarray(pcd.points)
    pcd_numpy = farthest_point_sample(pcd_numpy,downsample_number)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd_numpy)) 
    if visualize:
        o3d.visualization.draw_geometries([pcd], window_name='Open3D downSample', width=1920, height=1080, left=50, top=50,
                                        point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)
    
    return pcd_numpy

pcd = o3d.io.read_point_cloud("datasets/NTU-RGB+D/pointcloud2048/train/S001C001P001R001A001/MDepth-00000060.pcd")
read_and_preprocess_pointcloud_data(pcd,downsample_number=2048,visualize=True)