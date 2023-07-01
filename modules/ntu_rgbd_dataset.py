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
missing = ['S001C002P005R002A008', 'S001C002P006R001A008', 'S001C003P002R001A055', 'S001C003P002R002A012',
               'S001C003P005R002A004', 'S001C003P005R002A005', 'S001C003P005R002A006', 'S001C003P006R002A008',
               'S002C002P011R002A030', 'S002C003P008R001A020', 'S002C003P010R002A010', 'S002C003P011R002A007',
               'S002C003P011R002A011', 'S002C003P014R002A007', 'S003C001P019R001A055', 'S003C002P002R002A055',
               'S003C002P018R002A055', 'S003C003P002R001A055', 'S003C003P016R001A055', 'S003C003P018R002A024',
               'S004C002P003R001A013', 'S004C002P008R001A009', 'S004C002P020R001A003', 'S004C002P020R001A004',
               'S004C002P020R001A012', 'S004C002P020R001A020', 'S004C002P020R001A021', 'S004C002P020R001A036',
               'S005C002P004R001A001', 'S005C002P004R001A003', 'S005C002P010R001A016', 'S005C002P010R001A017',
               'S005C002P010R001A048', 'S005C002P010R001A049', 'S005C002P016R001A009', 'S005C002P016R001A010',
               'S005C002P018R001A003', 'S005C002P018R001A028', 'S005C002P018R001A029', 'S005C003P016R002A009',
               'S005C003P018R002A013', 'S005C003P021R002A057', 'S006C001P001R002A055', 'S006C002P007R001A005',
               'S006C002P007R001A006', 'S006C002P016R001A043', 'S006C002P016R001A051', 'S006C002P016R001A052',
               'S006C002P022R001A012', 'S006C002P023R001A020', 'S006C002P023R001A021', 'S006C002P023R001A022',
               'S006C002P023R001A023', 'S006C002P024R001A018', 'S006C002P024R001A019', 'S006C003P001R002A013',
               'S006C003P007R002A009', 'S006C003P007R002A010', 'S006C003P007R002A025', 'S006C003P016R001A060',
               'S006C003P017R001A055', 'S006C003P017R002A013', 'S006C003P017R002A014', 'S006C003P017R002A015',
               'S006C003P022R002A013', 'S007C001P018R002A050', 'S007C001P025R002A051', 'S007C001P028R001A050',
               'S007C001P028R001A051', 'S007C001P028R001A052', 'S007C002P008R002A008', 'S007C002P015R002A055',
               'S007C002P026R001A008', 'S007C002P026R001A009', 'S007C002P026R001A010', 'S007C002P026R001A011',
               'S007C002P026R001A012', 'S007C002P026R001A050', 'S007C002P027R001A011', 'S007C002P027R001A013',
               'S007C002P028R002A055', 'S007C003P007R001A002', 'S007C003P007R001A004', 'S007C003P019R001A060',
               'S007C003P027R002A001', 'S007C003P027R002A002', 'S007C003P027R002A003', 'S007C003P027R002A004',
               'S007C003P027R002A005', 'S007C003P027R002A006', 'S007C003P027R002A007', 'S007C003P027R002A008',
               'S007C003P027R002A009', 'S007C003P027R002A010', 'S007C003P027R002A011', 'S007C003P027R002A012',
               'S007C003P027R002A013', 'S008C002P001R001A009', 'S008C002P001R001A010', 'S008C002P001R001A014',
               'S008C002P001R001A015', 'S008C002P001R001A016', 'S008C002P001R001A018', 'S008C002P001R001A019',
               'S008C002P008R002A059', 'S008C002P025R001A060', 'S008C002P029R001A004', 'S008C002P031R001A005',
               'S008C002P031R001A006', 'S008C002P032R001A018', 'S008C002P034R001A018', 'S008C002P034R001A019',
               'S008C002P035R001A059', 'S008C002P035R002A002', 'S008C002P035R002A005', 'S008C003P007R001A009',
               'S008C003P007R001A016', 'S008C003P007R001A017', 'S008C003P007R001A018', 'S008C003P007R001A019',
               'S008C003P007R001A020', 'S008C003P007R001A021', 'S008C003P007R001A022', 'S008C003P007R001A023',
               'S008C003P007R001A025', 'S008C003P007R001A026', 'S008C003P007R001A028', 'S008C003P007R001A029',
               'S008C003P007R002A003', 'S008C003P008R002A050', 'S008C003P025R002A002', 'S008C003P025R002A011',
               'S008C003P025R002A012', 'S008C003P025R002A016', 'S008C003P025R002A020', 'S008C003P025R002A022',
               'S008C003P025R002A023', 'S008C003P025R002A030', 'S008C003P025R002A031', 'S008C003P025R002A032',
               'S008C003P025R002A033', 'S008C003P025R002A049', 'S008C003P025R002A060', 'S008C003P031R001A001',
               'S008C003P031R002A004', 'S008C003P031R002A014', 'S008C003P031R002A015', 'S008C003P031R002A016',
               'S008C003P031R002A017', 'S008C003P032R002A013', 'S008C003P033R002A001', 'S008C003P033R002A011',
               'S008C003P033R002A012', 'S008C003P034R002A001', 'S008C003P034R002A012', 'S008C003P034R002A022',
               'S008C003P034R002A023', 'S008C003P034R002A024', 'S008C003P034R002A044', 'S008C003P034R002A045',
               'S008C003P035R002A016', 'S008C003P035R002A017', 'S008C003P035R002A018', 'S008C003P035R002A019',
               'S008C003P035R002A020', 'S008C003P035R002A021', 'S009C002P007R001A001', 'S009C002P007R001A003',
               'S009C002P007R001A014', 'S009C002P008R001A014', 'S009C002P015R002A050', 'S009C002P016R001A002',
               'S009C002P017R001A028', 'S009C002P017R001A029', 'S009C003P017R002A030', 'S009C003P025R002A054',
               'S010C001P007R002A020', 'S010C002P016R002A055', 'S010C002P017R001A005', 'S010C002P017R001A018',
               'S010C002P017R001A019', 'S010C002P019R001A001', 'S010C002P025R001A012', 'S010C003P007R002A043',
               'S010C003P008R002A003', 'S010C003P016R001A055', 'S010C003P017R002A055', 'S011C001P002R001A008',
               'S011C001P018R002A050', 'S011C002P008R002A059', 'S011C002P016R002A055', 'S011C002P017R001A020',
               'S011C002P017R001A021', 'S011C002P018R002A055', 'S011C002P027R001A009', 'S011C002P027R001A010',
               'S011C002P027R001A037', 'S011C003P001R001A055', 'S011C003P002R001A055', 'S011C003P008R002A012',
               'S011C003P015R001A055', 'S011C003P016R001A055', 'S011C003P019R001A055', 'S011C003P025R001A055',
               'S011C003P028R002A055', 'S012C001P019R001A060', 'S012C001P019R002A060', 'S012C002P015R001A055',
               'S012C002P017R002A012', 'S012C002P025R001A060', 'S012C003P008R001A057', 'S012C003P015R001A055',
               'S012C003P015R002A055', 'S012C003P016R001A055', 'S012C003P017R002A055', 'S012C003P018R001A055',
               'S012C003P018R001A057', 'S012C003P019R002A011', 'S012C003P019R002A012', 'S012C003P025R001A055',
               'S012C003P027R001A055', 'S012C003P027R002A009', 'S012C003P028R001A035', 'S012C003P028R002A055',
               'S013C001P015R001A054', 'S013C001P017R002A054', 'S013C001P018R001A016', 'S013C001P028R001A040',
               'S013C002P015R001A054', 'S013C002P017R002A054', 'S013C002P028R001A040', 'S013C003P008R002A059',
               'S013C003P015R001A054', 'S013C003P017R002A054', 'S013C003P025R002A022', 'S013C003P027R001A055',
               'S013C003P028R001A040', 'S014C001P027R002A040', 'S014C002P015R001A003', 'S014C002P019R001A029',
               'S014C002P025R002A059', 'S014C002P027R002A040', 'S014C002P039R001A050', 'S014C003P007R002A059',
               'S014C003P015R002A055', 'S014C003P019R002A055', 'S014C003P025R001A048', 'S014C003P027R002A040',
               'S015C001P008R002A040', 'S015C001P016R001A055', 'S015C001P017R001A055', 'S015C001P017R002A055',
               'S015C002P007R001A059', 'S015C002P008R001A003', 'S015C002P008R001A004', 'S015C002P008R002A040',
               'S015C002P015R001A002', 'S015C002P016R001A001', 'S015C002P016R002A055', 'S015C003P008R002A007',
               'S015C003P008R002A011', 'S015C003P008R002A012', 'S015C003P008R002A028', 'S015C003P008R002A040',
               'S015C003P025R002A012', 'S015C003P025R002A017', 'S015C003P025R002A020', 'S015C003P025R002A021',
               'S015C003P025R002A030', 'S015C003P025R002A033', 'S015C003P025R002A034', 'S015C003P025R002A036',
               'S015C003P025R002A037', 'S015C003P025R002A044', 'S016C001P019R002A040', 'S016C001P025R001A011',
               'S016C001P025R001A012', 'S016C001P025R001A060', 'S016C001P040R001A055', 'S016C001P040R002A055',
               'S016C002P008R001A011', 'S016C002P019R002A040', 'S016C002P025R002A012', 'S016C003P008R001A011',
               'S016C003P008R002A002', 'S016C003P008R002A003', 'S016C003P008R002A004', 'S016C003P008R002A006',
               'S016C003P008R002A009', 'S016C003P019R002A040', 'S016C003P039R002A016', 'S017C001P016R002A031',
               'S017C002P007R001A013', 'S017C002P008R001A009', 'S017C002P015R001A042', 'S017C002P016R002A031',
               'S017C002P016R002A055', 'S017C003P007R002A013', 'S017C003P008R001A059', 'S017C003P016R002A031',
               'S017C003P017R001A055', 'S017C003P020R001A059', 'S001C002P006R001A008']

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
    '''
    small_size_dataset : now deprecated
    '''
    def __init__(self,
                 data_dir='datasets/NTU-RGB+D',
                 train = True,
                 datatype = ['pointcloud','skeleton'],
                 select_length = 1e8,
                 point_number = 2048,
                 small_size_dataset = False,
                 pre_downsample = True,
                 specific_setting:str = None,
                 specific_camera_id:str = None
                 ) -> None:
        super().__init__()
        self.datadir = data_dir
        self.train = train
        self.select_length = select_length
        self.point_number = point_number
        self.specific_setting = specific_setting
        self.specific_camera_id = specific_camera_id
        if self.train == True:
            train_dir = os.path.join(self.datadir,'pointcloud2048','train')
            sample_list = os.listdir(train_dir)
            sample_list = [sample for sample in sample_list if int(sample.split('A')[-1]) < 50]
            if self.specific_setting:
                sample_list = [sample for sample in sample_list if sample.find(self.specific_setting) != -1]
            if self.specific_camera_id:
                sample_list = [sample for sample in sample_list if sample.find(self.specific_camera_id) != -1]
            sample_list = [sample for sample in sample_list if sample not in missing]
            sample_list = [os.path.join(train_dir,sample) for sample in sample_list]
            self.data = sample_list
        else:
            test_dir = os.path.join(self.datadir,'pointcloud2048','test')
            raw_type_dir = os.listdir(test_dir)
            type_dir = [os.path.join(test_dir,dirname) for dirname in raw_type_dir if os.path.isdir(os.path.join(test_dir,dirname))]
            self.data = []
            for dir_path in type_dir:
                samples = os.listdir(dir_path)
                samples = [sample for sample in samples if int(sample.split('A')[-1]) < 50]
                if self.specific_setting:
                    samples = [sample for sample in samples if sample.find(self.specific_setting) != -1]
                if self.specific_camera_id:
                    samples = [sample for sample in samples if sample.find(self.specific_camera_id) != -1]
                samples = [sample for sample in samples if sample not in missing]
                samples = [os.path.join(dir_path,sample) for sample in samples]
                self.data.extend(samples)
        self.len = len(self.data)
        self.datatype = datatype
        self.pre_downsample = pre_downsample
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        
        data_dir = self.data[index]
        data_name = data_dir.split('/')[-1]
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
