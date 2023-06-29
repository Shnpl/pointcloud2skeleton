import torch
from torch.utils.data import DataLoader,Dataset
import open3d as o3d
import os
import numpy as np
import numpy.ma as ma
import sys
import h5py
sys.path.append(".")
from modules.utils import read_and_preprocess_pointcloud_data,read_skeleton_data
def find_and_remove_neighbors(original_list,center,thres):
    out = []
    remove_idxs = []
    for i,point in enumerate(original_list):
        if np.linalg.norm(point-center) <thres:
            out.append(point)
            remove_idxs.append(i)
    for idx in remove_idxs:
        original_list.pop(idx)
    return out
class _ITOP_Dataset(Dataset):
    def __init__(self,
                 data_dir='datasets/ITOP',
                 train = True,
                 datatype = ['pointcloud','skeleton'],
                 point_number = 2048,
                 small_size_dataset = False,
                 pre_downsample = True
                 ) -> None:
        super().__init__()
        self.datadir = data_dir
        self.train = train
        self.point_number = point_number
        self.datatype = datatype
        if self.train:
            pass
        else:
            f_label = h5py.File('datasets/ITOP/ITOP_side_test_labels.h5','r')
            is_valid = np.asarray(f_label.get('is_valid'))
            self.is_valid_idx = np.where(is_valid == 1)
            #label_segmentation = f_label.get('segmentation')
            self.side_skeleton_real_world_coordinates = np.asarray(f_label.get('real_world_coordinates'))[self.is_valid_idx]
            self.visible_side_joints = np.asarray(f_label.get('visible_joints'))[self.is_valid_idx]
            
            
            #label_ids_np = np.asarray(f_label.get('id'))
            f_side_pointcloud = h5py.File(os.path.join(self.datadir,'ITOP_side_test_point_cloud.h5'), 'r')
            self.side_pointcloud_data = np.asarray(f_side_pointcloud.get('data'))[self.is_valid_idx]
            self.side_pointcloud_ids = np.asarray(f_side_pointcloud.get('id'))[self.is_valid_idx]
            #
            self.len = self.side_pointcloud_data.shape[0]
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        data_dict = {}
        skeleton = self.side_skeleton_real_world_coordinates[index]
        pointcloud = self.side_pointcloud_data[index]
        torso = skeleton[8]
        filtered_1_pointcloud = []
        for point in pointcloud:
            flag = False
            for skeleton_point in skeleton:
                if np.linalg.norm(point-skeleton_point) < 0.3:
                    flag = True
                    break
            if flag == True:
                filtered_1_pointcloud.append(point)
                
        # filtered_1_pointcloud = np.stack(filtered_1_pointcloud)
        
        # open_list = [torso]
        # filtered_2_pointcloud = []
        # while open_list != []:
        #     p = open_list.pop(0)
        #     neighbors = find_and_remove_neighbors(filtered_1_pointcloud,p,thres = 0.1)
        #     filtered_2_pointcloud.extend(neighbors)
        #     open_list.extend(neighbors)
        # filtered_2_pointcloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_1_pointcloud)
        
        pcd_numpy = read_and_preprocess_pointcloud_data(pcd,downsample_number=self.point_number)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(filtered_1_pointcloud)
        # o3d.visualization.draw_geometries([pcd], window_name='Open3D downSample', width=1920, height=1080, left=50, top=50,
        #                                 point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)
        data_dict['skeletons']  = skeleton
        data_dict['pointclouds'] = pcd_numpy
        return data_dict
    
if __name__ == "__main__":
    

    # print(data.shape, ids.shape)
    # (10501, 240, 320) (10501,)
    ds = _ITOP_Dataset(train=False)
    i = 0
    for data in ds:
        i += 1
        print(i)
    print(ds.__len__())
    pass
