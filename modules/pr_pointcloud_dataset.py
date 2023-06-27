import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader,Dataset
import torchvision
import open3d as o3d
import os
import numpy as np
from PIL import Image
import sys
sys.path.append(".")
from modules.utils import read_and_preprocess_pointcloud_data,read_skeleton_data

class _PR_Pointcloud_Dataset(Dataset):
    def __init__(self,
                 data_dir='datasets/pr_dataset_pointcloud',
                 train = True,
                 datatype = ['pointcloud'],
                 select_length = 1e8,
                 point_number = 2048,
                 small_size_dataset = False,
                 pre_downsample = False
                 ):
        super().__init__()
        self.datadir = data_dir
        self.train = train
        self.select_length = select_length
        self.point_number = point_number
        raw_type_dir = os.listdir(self.datadir)
        type_dir = [os.path.join(self.datadir,dirname) for dirname in raw_type_dir if os.path.isdir(os.path.join(self.datadir,dirname))]
        self.data = []
        for dir_path in type_dir:
            actions = os.listdir(dir_path)
            filtered_actions = [action for action in actions if int(action.split('A')[-1]) < 50]

            actions = [os.path.join(dir_path,action) for action in filtered_actions]
            self.data.extend(actions)


        self.len = len(self.data)
        self.datatype = datatype
        self.pre_downsample = pre_downsample
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        
        data_path = self.data[index]
        data_name = data_path.split('/')[-1]
        data_dict = dict({"name":data_name})
        try:
            if 'pointcloud' in self.datatype:
                if self.train:
                    pointcloud_data_dir = data_path
                else:
                    pointcloud_data_dir = data_path
                # else:
                    # pointcloud_data_dir = os.path.join(self.datadir,"pointcloud",data_name)
                frames = os.listdir(pointcloud_data_dir)
                frames = sorted(frames)
                if self.select_length < len(frames):
                    frames = frames[:self.select_length]
                pointclouds = []
                for name in frames:
                    path = os.path.join(pointcloud_data_dir,name)
                    cloud_im = np.load(path)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cloud_im)
                    if self.pre_downsample:
                        pcd_numpy = np.asarray(pcd.points)
                        N,_ = pcd_numpy.shape
                        if N < self.point_number:
                            fill_num = self.point_number-N
                            sample_list = [i for i in range(pcd_numpy.shape[0])]
                            fill_list = np.random.choice(sample_list,fill_num)
                            fill_pt = pcd_numpy[fill_list,:]
                            pcd_numpy = np.concatenate((pcd_numpy,fill_pt),axis = 0)
                    else:
                        pcd_numpy = read_and_preprocess_pointcloud_data(pcd,downsample_number=self.point_number)
                    pointclouds.append(pcd_numpy)
                pointclouds = np.stack(pointclouds)
                data_dict.update({"pointcloud":pointclouds})
        except:
            data_dict = None
        return data_dict
    
if __name__ == "__main__":
    ds = _PR_Pointcloud_Dataset()
    i = 0
    for data in ds:
        i += 1
        print(i)
    print(ds.__len__())
    pass
