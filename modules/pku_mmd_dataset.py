import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader,Dataset
import torchvision

import os

from PIL import Image
import imageio

class PKU_MMD_DataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = './datasets/PKU-MMD',
                 batch_size: int = 64,
                 num_workers: int = 8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        # self.dims = (1, 28, 28)
        # self.num_classes = 13

    def prepare_data(self):
        # download or anything like that
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.PKU_Dataset_train = _PKU_MMD_Dataset(
                data_dir=self.data_dir,
                train=True)
            self.PKU_Dataset_val = _PKU_MMD_Dataset(
            data_dir=self.data_dir,
            train=False)
        # Assign test dataset for use in dataloader(s)
        # if stage == 'test' or stage is None:
        #     self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
        pass
    def train_dataloader(self):
        return DataLoader(self.PKU_Dataset_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.PKU_Dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)
    # def test_dataloader(self):
    #     return DataLoader(self.PKU_Dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)
class _PKU_MMD_Dataset(Dataset):
    def __init__(self,
                 data_dir='./datasets/PKU-MMDv2',
                 train = True,
                 datatype = ['depth','skeleton'],
                 select_length = 128
                 ) -> None:
        super().__init__()
        self.datadir = data_dir
        self.train = train
        self.select_length = select_length
        if self.train == True:
            with open(os.path.join(self.datadir,"Split","cross_subject_v2.txt"),'r') as f:
                f.readline()
                self.data = f.readline()
        else:
            with open(os.path.join(self.datadir,"Split","cross_subject_v2.txt"),'r') as f:
                f.readline()
                f.readline()
                f.readline()
                self.data = f.readline()
        self.data = self.data.split(",")
        self.action_per_epoch = 5
        self.len = len(self.data)*self.action_per_epoch
        self.datatype = datatype
    def __len__(self):
        return self.len
    def __getitem__(self, index):

        data_name = self.data[int(index/self.action_per_epoch)]
        while data_name[0] == ' ':
            data_name = data_name[1:]
        while data_name[-1] == ' ':
            data_name = data_name[:-1]
        data_name = data_name.replace("\n","")
        action_index = int(index%self.action_per_epoch)

        depth_data_dir = os.path.join(self.datadir,"Data","DEPTH_v2",data_name,"depth")
        infrared_data_dir = os.path.join(self.datadir,"Data","INFRARED_v2",data_name,"infrared")
        rgb_data_path = os.path.join(self.datadir,"Data","RGB_VIDEO_v2",f"{data_name}_color.avi")
        skeleton_data_path = os.path.join(self.datadir,"Data","skeleton",f"{data_name}.txt")

        label_path =  os.path.join(self.datadir,"Label",f"{data_name}.txt")
        labels = []
        with open(label_path,'r') as f:
            for line in f:
                labels.append(line)
        label = labels[action_index]
        label = label.split(",")
        action_type = int(label[0])
        start_frame = int(label[1])
        end_frame = int(label[2])
        #confidence = int(label[3])
        
        #depth
        if 'depth' in self.datatype:
            depth_imgs = []
            transform = torchvision.transforms.ToTensor()
            for i in range(start_frame,min(end_frame,start_frame+self.select_length)):
                depth_img_path = os.path.join(depth_data_dir,f"{i}.png")
                depth_img = Image.open(depth_img_path)
                depth_img = transform(depth_img)
                depth_imgs.append(depth_img)
            depth_imgs = torch.stack(depth_imgs)#t,c,h,w
        #infra
        if 'ir' in self.datatype:
            infra_imgs = []
            transform = torchvision.transforms.ToTensor()
            for i in range(start_frame,min(end_frame,start_frame+self.select_length)):
                infra_img_path = os.path.join(infrared_data_dir,f"{i}.png")
                infra_img = Image.open(infra_img_path)
                infra_img = transform(infra_img)
                infra_imgs.append(infra_img)
            infra_imgs = torch.stack(infra_imgs)#t,c,h,w
        #rgb
        if 'rgb' in self.datatype:
            vid = imageio.get_reader(rgb_data_path,  'ffmpeg')
            rgb_imgs = []
            transform = torchvision.transforms.ToTensor()
            for i in range(start_frame,min(end_frame,start_frame+self.select_length)):
                rgb_image = vid.get_data(i)
                rgb_image = Image.fromarray(rgb_image)
                rgb_image = rgb_image.resize((640,360))
                rgb_image = transform(rgb_image)
                rgb_imgs.append(rgb_image)
            rgb_imgs = torch.stack(rgb_imgs)#t,c,h,w
        #skeleton
        if 'skeleton' in self.datatype:
            skeletons_0 = []
            skeletons_1 = []
            with open(skeleton_data_path,'r') as f:
                if start_frame == 0:
                    pass
                elif start_frame == 1:
                    f.readline(1)
                else:
                    f.readlines(start_frame-1)
                for i in range(start_frame,min(end_frame,start_frame+self.select_length)):
                    line = f.readline()
                    line = line.split(" ")
                    if line[0] == '':
                        line = line[1:]
                    line[-1] = line[-1].replace("\n","")
                    line = [float(i) for i in line]
                    skeleton_0 = line[0:75]
                    skeleton_0 = torch.tensor(skeleton_0)
                    skeleton_0 = skeleton_0.reshape(-1,3)#25,3
                    skeletons_0.append(skeleton_0)

                    skeleton_1 = line[75:150]
                    skeleton_1 = torch.tensor(skeleton_1)
                    skeleton_1 = skeleton_1.reshape(-1,3)#25,3
                    skeletons_1.append(skeleton_1)
            skeletons_0 = torch.stack(skeletons_0)#t,25,3
            skeletons_1 = torch.stack(skeletons_1)#t,25,3
        # Collect
        data_dict = dict({"name":data_name,
                "action_type":action_type,
                #"confidence":confidence
                })
        if 'depth' in self.datatype:
            data_dict.update({"depth_imgs":depth_imgs})
        if 'ir' in self.datatype:
            data_dict.update({"infra_imgs":infra_imgs})
        if 'rgb' in self.datatype:
            data_dict.update({"rgb_imgs":rgb_imgs})
        if 'skeleton' in self.datatype:
            data_dict.update({"skeletons_0":skeletons_0,
                              "skeletons_1":skeletons_1})
        
        return data_dict
    
if __name__ == "__main__":
    ds = _PKU_MMD_Dataset(select_length =2)
    i = 0
    for data in ds:
        i += 1
        print(i)
    print(ds.__len__())
    pass
