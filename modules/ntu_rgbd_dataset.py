import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader,Dataset
import torchvision
import open3d as o3d
import os
import numpy as np
from PIL import Image

from modules.utils import read_and_preprocess_pointcloud_data,read_skeleton_data

# class NTU_RGBD_DataModule(pl.LightningDataModule):
#     def __init__(self,
#                  data_dir: str = 'datasets/NTU-RGB+D',
#                  batch_size: int = 64,
#                  num_workers: int = 8):
#         super().__init__()
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.num_workers = num_workers
        
#         # self.transform = transforms.Compose([
#         #     transforms.ToTensor(),
#         #     transforms.Normalize((0.1307,), (0.3081,))
#         # ])

#         # self.dims is returned when you call dm.size()
#         # Setting default dims here because we know them.
#         # Could optionally be assigned dynamically in dm.setup()
#         # self.dims = (1, 28, 28)
#         # self.num_classes = 13

#     def prepare_data(self):
#         # download or anything like that
#         pass

#     def setup(self, stage=None):
#         # Assign train/val datasets for use in dataloaders
#         if stage == 'fit' or stage is None:
#             self.PKU_Dataset_train = _PKU_MMD_Dataset(
#                 data_dir=self.data_dir,
#                 train=True)
#             self.PKU_Dataset_val = _PKU_MMD_Dataset(
#             data_dir=self.data_dir,
#             train=False)
#         # Assign test dataset for use in dataloader(s)
#         # if stage == 'test' or stage is None:
#         #     self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
#         pass
#     def train_dataloader(self):
#         return DataLoader(self.PKU_Dataset_train, batch_size=self.batch_size, num_workers=self.num_workers)

#     def val_dataloader(self):
#         return DataLoader(self.PKU_Dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)
#     # def test_dataloader(self):
#     #     return DataLoader(self.PKU_Dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

class _NTU_RGBD_Dataset(Dataset):
    def __init__(self,
                 data_dir='datasets/NTU-RGB+D',
                 train = True,
                 datatype = ['pointcloud','skeleton'],
                 select_length = 128,
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
    ds = _NTU_RGBD_Dataset(select_length =2)
    i = 0
    for data in ds:
        i += 1
        print(i)
    print(ds.__len__())
    pass
