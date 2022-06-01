import os
import random
import numpy as np
import time
import PIL.Image as pil
import glob

import torch
import torch.utils.data as data

import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from . import joint_transforms as joint_T
from . import joint_transforms_depth as joint_T_depth
import os


class cp_dataset(data.Dataset):

    def __init__(self, config):

        super(cp_dataset, self).__init__()
        self.config = config
        self.width, self.height = self.config.input_size[1], self.config.input_size[0]
        self.mix_num = self.config.mix_num
        self.instance_num = 2+self.config.cp_times[0]+self.config.cp_times[1]

        self.to_tensor = T.ToTensor()
        self.to_pil = T.ToPILImage()
        self.instance_spatial_trans = joint_T_depth.Compose([
            joint_T_depth.RandomDistance(p=1, center=160, scale = (0.2, 2), min_size = 8, max_size = (2*self.height, 2*self.width), img_size = (self.height,self.width)),
            joint_T_depth.RandomHorizontalFlip(),
            ])

        self.instance_color_trans = T.Compose([
            T.RandomApply([T.ColorJitter(0.5, 0.5, 0.5, 0.2)], p=0.8),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2))], p=0.2),
            T.RandomGrayscale(p=0.1),
            ])
        self.train_trans = joint_T.Compose([
            joint_T.RandomResize(scale=(1.0, 2.0), antialias=True),
            joint_T.RandomCrop(size=self.config.input_size),
            joint_T.RandomApply([joint_T.ColorJitter(0.5, 0.5, 0.5, 0.2)], p=0.8),
            joint_T.RandomApply([joint_T.GaussianBlur(kernel_size=5, sigma=(0.1, 2))], p=0.2),
            joint_T.RandomGrayscale(p=0.1),
            joint_T.RandomHorizontalFlip(p=0.5),
            joint_T.ToTensor(),
            joint_T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])

        cs_path = "rgb/leftImg8bit/train/*/*.png"
        self.img_path = sorted(glob.glob(self.config.data_path+cs_path))
        self.dataset_size = len(self.img_path)//self.config.mix_num


    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, index):

        instance_list = []
        img_list = []

        for i in range(self.config.mix_num):
            path = self.img_path[i*self.dataset_size+index]
            inst = self.get_inst(path.replace("rgb","inst"))
            disp = self.get_disp(path.replace("rgb","disp"))
            rgb = self.get_rgb(path)

            instance_binary_mask = torch.zeros((inst.max()+1, self.height, self.width), dtype=torch.bool)
            instance_binary_mask.scatter_(dim = 0, index = inst, src = torch.ones_like(instance_binary_mask,dtype=torch.bool))

            num_instances = instance_binary_mask.shape[0]
            bbox_h = instance_binary_mask.any(2)
            bbox_w = instance_binary_mask.any(1)
            top = bbox_h.long().argmax(dim=1)
            left = bbox_w.long().argmax(dim=1)
            bottom = self.height-torch.flip(bbox_h.long(),[1]).argmax(dim=1)
            right = self.width-torch.flip(bbox_w.long(),[1]).argmax(dim=1)
            transforms = torch.eye(3).repeat(num_instances,1,1)
            transforms[:,0,2] = left
            transforms[:,1,2] = top
            
            start_id = len(instance_list)
            for i in range(0, num_instances):
                instance = {}
                instance["id"] = i+start_id
                instance["img"] = rgb[:,top[i]:bottom[i],left[i]:right[i]]
                instance["disp"] = disp[:,top[i]:bottom[i],left[i]:right[i]]
                instance["mask"] = instance_binary_mask[i:i+1,top[i]:bottom[i],left[i]:right[i]]
                instance["top"] = top[i]
                instance["left"] = left[i]
                instance_list.append(instance)

            img = {}

            img["num_instances"] = num_instances
            img["rgb"] = rgb
            img["disp"] = disp
            img["inst"] = inst
            img["start_id"] = start_id
            img["transforms"] = transforms
            img_list.append(img)
            

        mix_img_tensor, mix_label_tensor, transforms = self.mix(img_list, instance_list)
        mix_label_tensor = mix_label_tensor.int()
        if self.config.only_generate_data == True:
            inputs = {}
            mix_img_tensor = mix_img_tensor.permute(1,0,2,3).flatten(1,2)
            mix_label_tensor = mix_label_tensor.permute(1,0,2,3).flatten(1,2)
          
            inputs["img"] = self.to_pil(mix_img_tensor)
            inputs["label"] = self.to_pil(mix_label_tensor)
            inputs["t"] = transforms
            return inputs

        mix_img = [self.to_pil(mix_img_tensor[i]) for i in range(mix_img_tensor.shape[0])]
        mix_label = [self.to_pil(mix_label_tensor[i]) for i in range(mix_label_tensor.shape[0])]


        t_img = [torch.eye(3) for i in range(len(mix_img))]
        for i in range(len(mix_img)):
          mix_img[i], mix_label[i], t_img[i] = self.train_trans(mix_img[i], mix_label[i], t_img[i])

        t = torch.zeros(16384,transforms.shape[1],10)
        t[:transforms.shape[0]] = transforms.float()
        t_img = torch.stack(t_img,dim=0)
        mix_img = torch.stack(mix_img,dim=0)
        mix_label = torch.stack(mix_label,dim=0)
        
        inputs = {}
        inputs["img"] = mix_img
        inputs["label"] = mix_label.long()
        inputs["t"] = transforms
        inputs["t_img"] = t_img

        return inputs

    

    def mix(self, img_list, instance_list):
        
        transforms = torch.eye(3).repeat(len(instance_list),self.instance_num,1,1)
        img_id = -torch.ones((len(instance_list),self.instance_num))
        count = torch.full((len(instance_list),),2,dtype=torch.long)
        
        mix_img = []
        mix_label = []
        mix_disp = []

        for i in range(2):
            for j in range(self.mix_num):
                mix_disp.append(img_list[j]["disp"].clone())
                mix_label.append((img_list[j]["inst"].clone()+img_list[j]["start_id"])*self.instance_num+i)
                mix_img.append(img_list[j]["rgb"].clone())

                img_id[img_list[j]["start_id"]:img_list[j]["start_id"]+img_list[j]["num_instances"],i] = i*self.mix_num+j
                transforms[img_list[j]["start_id"]:img_list[j]["start_id"]+img_list[j]["num_instances"],i] = img_list[j]["transforms"]

        for i in range(2):
            instance_list_i = []
            for j in range(self.config.cp_times[i]):
                instance_list_i = instance_list_i+instance_list
            random.shuffle(instance_list_i)
            for j in range(self.mix_num):
                index = i*self.mix_num+j
                self.copy_paste(mix_img[index], mix_disp[index], mix_label[index], instance_list_i[j::self.mix_num], transforms, count, img_id, index)
        
        mix_img = torch.stack(mix_img,dim=0)
        mix_label = torch.stack(mix_label,dim=0)
        t1 = torch.Tensor([[1,0,-1],[0,1,-1],[0,0,1]])
        t2 = torch.Tensor([[1/(self.width/2),0,0],[0,1/(self.height/2),0],[0,0,1]])
        t = torch.mm(t1,t2)
        transforms = torch.matmul(t.unsqueeze(0),transforms)
        transforms = torch.cat([transforms.flatten(2,3),img_id.unsqueeze(-1)],dim=-1)

        return mix_img, mix_label, transforms

    def copy_paste(self, mix_img, mix_disp, mix_label, instance_list, transforms, count, img_id, img_i):

        for instance in instance_list:
            
            h = instance["img"].shape[1]
            w = instance["img"].shape[2]
            t = torch.Tensor([[ 1,0,0],
                      [ 0,1,0],
                      [ 0,0,1]])

            if h<16 or w<6 or (h<32 and w<32) or (h>self.height-16 and w > self.width-16):
              
              img_id[instance["id"],count[instance["id"]]] = img_i
              transforms[instance["id"],count[instance["id"]]] = t
              count[instance["id"]] = count[instance["id"]]+1
              continue
            
            aug_img, aug_mask, aug_disp, aug_top, t = self.instance_spatial_trans(instance["img"].clone(), 
                                  instance["mask"], 
                                  instance["disp"], 
                                  instance["top"],
                                  t)

            aug_img = self.instance_color_trans(aug_img)
                
            h, w = aug_img.shape[1], aug_img.shape[2]
            aug_left = random.randint(-w+1,self.width-1)
            aug_top = aug_top+random.randint(-16,16)
            
            t_cp = torch.Tensor([[ 1,0,aug_left],
                        [ 0,1,aug_top],
                        [ 0,0,1]])
            
            aug_top = min(self.height-1,aug_top)
            aug_top = max(aug_top,-h+1)

            h1 = max(0,0-aug_top)
            w1 = max(0,0-aug_left)
            h2 = max(0,aug_top+h-self.height)
            w2 = max(0,aug_left+w-self.width)
            aug_img = aug_img[:,h1:h-h2,w1:w-w2]
            aug_mask = aug_mask[:,h1:h-h2,w1:w-w2]
            aug_disp = aug_disp[:,h1:h-h2,w1:w-w2]
            aug_left = max(0,aug_left)
            aug_top = max(0,aug_top)
            
            mix_img_bbox = mix_img[:,aug_top:aug_top+aug_img.shape[1],aug_left:aug_left+aug_img.shape[2]]
            mix_disp_bbox = mix_disp[:,aug_top:aug_top+aug_img.shape[1],aug_left:aug_left+aug_img.shape[2]]
            mix_label_bbox = mix_label[:,aug_top:aug_top+aug_img.shape[1],aug_left:aug_left+aug_img.shape[2]]
            
            mask = (aug_mask & (mix_disp_bbox<aug_disp))
            if mask.shape[1]>=3 and mask.shape[2]>=3:
                soft_mask = TF.gaussian_blur(mask.float(), 3, (1,1))
            else:
                soft_mask = mask.float()

            mix_img_bbox = mix_img_bbox*(1-soft_mask) + aug_img*soft_mask
            mix_disp_bbox = mix_disp_bbox*(1-soft_mask) + aug_disp*soft_mask
            mix_label_bbox[:,mask.squeeze(0)] = instance["id"]*self.instance_num+count[instance["id"]]
            
            mix_img[:,aug_top:aug_top+aug_img.shape[1],aug_left:aug_left+aug_img.shape[2]] = mix_img_bbox
            mix_disp[:,aug_top:aug_top+aug_img.shape[1],aug_left:aug_left+aug_img.shape[2]] = mix_disp_bbox
            mix_label[:,aug_top:aug_top+aug_img.shape[1],aug_left:aug_left+aug_img.shape[2]] = mix_label_bbox
            
            img_id[instance["id"],count[instance["id"]]] = img_i
            transforms[instance["id"],count[instance["id"]]] = torch.mm(t_cp,t)
            count[instance["id"]] = count[instance["id"]]+1


    def get_rgb(self, path):
        rgb = pil.open(path).resize((self.width,self.height),pil.ANTIALIAS)
        rgb = self.to_tensor(rgb)
        return rgb
      
    def get_inst(self, path):
        instance_label = pil.open(path).resize((self.width,self.height),pil.NEAREST)
        instance_label = self.to_tensor(instance_label).long()
        return instance_label

    def get_disp(self, path):
        
        disp = pil.open(path).resize((self.width,self.height),pil.BILINEAR)
        disp = torch.from_numpy(np.array(disp)).unsqueeze(0)/65535.0
        disp[disp==0] = 1e-6

        return disp


class prepared_dataset(data.Dataset):

    def __init__(self, config):
        super(prepared_dataset, self).__init__()
        self.config = config
        
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.trans = joint_T.Compose([
            joint_T.RandomResize(scale=(1.0, 2.0), antialias=True),
            joint_T.RandomCrop(size=self.config.input_size),
            joint_T.RandomApply([joint_T.ColorJitter(0.5, 0.5, 0.5, 0.2)], p=0.8),
            joint_T.RandomApply([joint_T.GaussianBlur(kernel_size=5, sigma=(0.1, 2))], p=0.2),
            joint_T.RandomGrayscale(p=0.1),
            joint_T.RandomHorizontalFlip(p=0.5),
            joint_T.ToTensor(),
            joint_T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ])
        self.data_path = self.config.data_path
        self.dataset_config = torch.load(self.config.data_path+"/config.pth")
        self.input_size = self.dataset_config["input_size"]
        self.instance_num = self.dataset_config["instance_num"]
        self.mix_num = self.dataset_config["mix_num"]
        self.num_epoch = self.dataset_config["epoch_num"]
        self.set_epoch(0)
        

    def set_epoch(self, epoch):
        self.epoch = epoch%self.num_epoch
        self.num_epoch = len(glob.glob(self.config.data_path+"/epoch_{:d}".format(self.epoch)))
        self.p = sorted(glob.glob(self.config.data_path+"/epoch_{:d}/*img.png".format(self.epoch)))
        self.dataset_size = len(self.p)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):

        inputs = {} 
        
        img = pil.open(os.path.join(self.p[index]))
        mix_img = [img.crop((0, i*self.input_size[0], self.input_size[1], i*self.input_size[0]+self.input_size[0])) for i in range(2*self.mix_num)]
        label = pil.open(os.path.join(self.p[index].replace("img","label")))
        mix_label = [label.crop((0, i*self.input_size[0], self.input_size[1], i*self.input_size[0]+self.input_size[0])) for i in range(2*self.mix_num)]
        transforms = torch.load(os.path.join(self.p[index].replace("img.png","transforms.pt")))

        t = torch.zeros(16384,self.instance_num,10)
        t[:transforms.shape[0]] = transforms.float()

        t_img = [torch.eye(3) for i in range(2*self.mix_num)]

        for i in range(len(mix_img)):
          mix_img[i], mix_label[i], t_img[i] = self.trans(mix_img[i], mix_label[i], t_img[i])

        mix_img = torch.stack(mix_img,dim=0)
        mix_label = torch.stack(mix_label,dim=0)
        t_img = torch.stack(t_img,dim=0)
        
        inputs["img"] = mix_img
        inputs["label"] = mix_label.long()
        inputs["t"] = t
        inputs["t_img"] = t_img

        return inputs



