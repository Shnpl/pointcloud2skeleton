import torch
import pytorch_lightning as pl
import sys
from torch.utils.data import DataLoader,Dataset
import torchvision
import open3d as o3d
import os
import numpy as np
from PIL import Image
sys.path.append('.')
from modules.utils import read_and_preprocess_pointcloud_data,read_skeleton_data

def calibrate(pcd_numpy):
    fx_o = 525   # 相机水平方向的焦距
    fy_o = 525   # 相机垂直方向的焦距
    cx_o = 319.5   # 相机水平方向的光心（principal point）x坐标
    cy_o = 239.5  # 相机垂直方向的光心（principal point）y坐标
    fx = -371.35623250663923
    fy = 370.4995539322389
    cx = 256.32786847500023
    cy = 208.58669761762442
    scale = 971
    scale_o = 5000
    N,_ =pcd_numpy.shape
    x_from_depth_o = pcd_numpy[:,0]
    y_from_depth_o = pcd_numpy[:,1]
    z_from_depth_o = -pcd_numpy[:,2]
    z_from_depth_rec = z_from_depth_o*scale_o/scale

    x_from_depth_pix_rec = -(x_from_depth_o * fx_o / z_from_depth_o)+cx_o
    x_from_depth_rec = -(x_from_depth_pix_rec - cx) * z_from_depth_rec / fx

    y_from_depth_pix_rec = -(y_from_depth_o * fy_o / z_from_depth_o)+cy_o
    y_from_depth_rec = -(y_from_depth_pix_rec - cy) * z_from_depth_rec / fy

    out = np.zeros((N,3))
    out[:,0] = x_from_depth_rec
    out[:,1] = y_from_depth_rec
    out[:,2] = z_from_depth_rec
    return out


class _NTU_RGBD_Dataset(Dataset):
    def __init__(self,
                 data_dir='datasets/NTU-RGB+D',
                 train = True,
                 datatype = ['pointcloud','skeleton'],
                 select_length = 1e8,
                 point_number = 2048,
                 small_size_dataset = False,
                 pre_downsample = True
                 ) -> None:
        super().__init__()
        self.datadir = data_dir
        self.train = train
        self.select_length = select_length
        self.point_number = point_number
        if self.train == True:
            if small_size_dataset:
                dataset_desc_path = os.path.join(self.datadir,"train_data_small.txt")
            else:
                dataset_desc_path = os.path.join(self.datadir,"train_data.txt")
            with open(dataset_desc_path,'r') as f:
                self.data = f.readlines()
        else:
            if small_size_dataset:
                dataset_desc_path = os.path.join(self.datadir,"train_data_small.txt")
            else:
                test_dir = os.path.join(self.datadir,'pointcloud2048','test')

                raw_type_dir = os.listdir(test_dir)
                type_dir = [os.path.join(test_dir,dirname) for dirname in raw_type_dir if os.path.isdir(os.path.join(test_dir,dirname))]
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
        
        data_name = self.data[index]
        if '/' in data_name:
            data_dir = data_name
            data_name = data_dir.split('/')[-1]

        data_name = data_name.replace("\n","")
        data_dict = dict({"name":data_name})
        
        try:
            if 'skeleton' in self.datatype:
                skeleton_data_path = os.path.join(self.datadir,"skeleton",f"{data_name}.skeleton")
                data = read_skeleton_data(skeleton_data_path,self.select_length)                
                if data['skeleton_num'] > 1:
                    raise
                data_dict.update({"skeletons":data})
            if 'pointcloud' in self.datatype:
                if self.train:
                    pointcloud_data_dir = os.path.join(self.datadir,"pointcloud2048",'train',data_name)
                else:
                    pointcloud_data_dir = data_dir
                # else:
                    # pointcloud_data_dir = os.path.join(self.datadir,"pointcloud",data_name)
                frames = os.listdir(pointcloud_data_dir)
                frames = sorted(frames)
                if self.select_length < len(frames):
                    frames = frames[:self.select_length]
                pointclouds = []
                for name in frames:
                    path = os.path.join(pointcloud_data_dir,name)
                    pcd = o3d.io.read_point_cloud(path)
                    if self.pre_downsample:
                        pcd_numpy = np.asarray(pcd.points)
                        pcd_numpy = pcd_numpy[[not np.all(pcd_numpy[i] == 0) for i in range(pcd_numpy.shape[0])], :]
                        N,_ = pcd_numpy.shape
                        if N < self.point_number:
                            fill_num = self.point_number-N
                            sample_list = [i for i in range(pcd_numpy.shape[0])]
                            fill_list = np.random.choice(sample_list,fill_num)
                            fill_pt = pcd_numpy[fill_list,:]
                            pcd_numpy = np.concatenate((pcd_numpy,fill_pt),axis = 0)
                    else:
                        pcd_numpy = read_and_preprocess_pointcloud_data(pcd,downsample_number=self.point_number)
                    
                    pcd_numpy= calibrate(pcd_numpy)
                    
                    pointclouds.append(pcd_numpy)
                pointclouds = np.stack(pointclouds)

                data_dict.update({"pointcloud":pointclouds})
        except:
            data_dict = None
        return data_dict
    
if __name__ == "__main__":
    ds = _NTU_RGBD_Dataset(select_length =2)
    i = 0
    for data in ds:
        i += 1
        print(i)
    print(ds.__len__())
    pass
