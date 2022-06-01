import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader, sampler

import os
import time
import PIL.Image as pil
import math
import random
import matplotlib.pyplot as plt
import glob
#import wandb

import datasets
from utils import *
from model import *
#from apex import amp
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

torch.manual_seed(0)
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class Trainer():
    def __init__(self, config):
        #super(Trainer_depth2seg, self).__init__(config, models)
        if config.resume_model_path is not None:
            path = config.resume_model_path
            config = torch.load(os.path.join(self.config.resume_model_path, "config.pth"))
            self.config = config["config"]
            self.config.resume_model_path = path
            self.start_epoch = config["epoch"]
        else:
            self.config = config

        self.device = torch.device(self.config.device)
        self.loss_avg = 0
        self.scaler = GradScaler()
        self.start_epoch = 0

    def init_training(self):
        print("Training model named:\n  ", self.config.model_name)
        print("Models and logs are saved to:\n  ", self.config.save_path)
        print("Training is using:\n  ", self.device)

        if self.config.resume_model_path is not None:
            self.load_model()
            
        self.step = self.start_epoch*len(self.train_loader)
        self.start_step = self.start_epoch*len(self.train_loader)
        self.epoch = self.start_epoch
        self.start_time = time.time()
        
        num_train_samples = len(self.train_loader)
        self.num_total_steps = num_train_samples * (self.config.num_epochs-self.start_epoch)

        print("There are {:d} training items\n".format(len(self.train_dataset)))
        
        
    def train(self):
        self.init_training()
        for self.epoch in range(self.start_epoch, self.config.num_epochs):

            self.train_dataset.set_epoch(self.epoch)
            self.run_epoch()

            self.epoch = self.epoch+1
            if self.epoch % self.config.save_frequency_epoch == 0:
                self.save_model()
        
        self.save_model()
                
        return self.model
                
    def run_epoch(self):
        #for param_group in self.model_optimizer.param_groups:
            #print(param_group['lr'])
        
        for batch_idx, inputs in enumerate(self.train_loader):

            for key in inputs.keys():
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].to(self.device)
            
            with autocast():
                outputs = self.forward(inputs)
                loss = self.compute_loss(inputs, outputs)

            self.model_optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.model_optimizer)
            self.scaler.update()
            self.model_lr_scheduler.step()

            self.step += 1
            self.loss_avg = self.loss_avg+loss
            
            if self.step % self.config.log_frequency == 0:
                self.log_time(batch_idx)
             
    def log_time(self, batch_idx):

        loss_avg = self.loss_avg/self.config.log_frequency
        self.loss_avg = 0
        time_sofar = time.time() - self.start_time
        samples_per_sec = (self.step-self.start_step)/time_sofar
        training_time_left = (self.num_total_steps / (self.step-self.start_step) - 1.0) * time_sofar if self.step > 0 else 0
        print_string_time = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f} | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string_time.format(self.epoch, batch_idx, samples_per_sec, loss_avg, sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))


    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.config.save_path, "weights_{}_{}".format(self.epoch, self.step))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        torch.save(self.model.state_dict(), os.path.join(save_folder, "model.pth"))
        torch.save(self.model_optimizer.state_dict(), os.path.join(save_folder, "optimizer.pth"))
        torch.save(self.model_lr_scheduler.state_dict(), os.path.join(save_folder, "scheduler.pth"))
        config = {"config":self.config, "epoch": self.epoch}
        torch.save(config, os.path.join(save_folder, "config.pth"))

    
    def load_model(self):
        """Load model(s) from disk
        """

        assert os.path.isdir(self.config.resume_model_path), "Cannot find folder {}".format(self.config.resume_model_path)
        print("loading model from folder {}".format(self.config.resume_model_path))
        
        self.model.load_state_dict(torch.load(os.path.join(self.config.resume_model_path, "model.pth")))
        self.model_optimizer.load_state_dict(torch.load(os.path.join(self.config.resume_model_path, "optimizer.pth")))
        self.model_lr_scheduler.load_state_dict(torch.load(os.path.join(self.config.resume_model_path, "scheduler.pth")))

class Trainer_FYA(Trainer):
    def __init__(self, config):
        super(Trainer_FYA, self).__init__(config)

        self.model = build_SwAV(self.config)
        self.model.to(self.device)
        
        params = [{"params":self.model.backbone.parameters(),"lr":self.config.backbone_lr},
              {"params":self.model.sem_seg_head.parameters(), "lr":self.config.sem_seg_head_lr},
              {"params":self.model.projector.parameters(), "lr":self.config.projector_lr},
              {"params":self.model.prototypes.parameters(), "lr":self.config.prototypes_lr}]
        
        self.model_optimizer = optim.SGD(params,momentum=0.9,weight_decay=1e-4)
        #self.model_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.model_optimizer, self.config.num_epochs, eta_min=1e-6)

        if self.config.use_prepared_data == True:
            self.train_dataset = datasets.prepared_dataset(self.config)
            self.train_loader = DataLoader(
                self.train_dataset, batch_size = self.config.batch_size, shuffle = True,
                num_workers=self.config.num_workers, pin_memory=True, drop_last=True, persistent_workers=True)
        else:
            self.train_dataset = datasets.cp_dataset(self.config)
            self.train_loader = DataLoader(
                self.train_dataset, batch_size = self.config.batch_size, shuffle = True,
                num_workers=self.config.num_workers, pin_memory=True, drop_last=True, persistent_workers=True)

        warmup_iter = len(self.train_loader) * self.config.warmup_epochs
        cos_iter = len(self.train_loader) * (self.config.num_epochs - self.config.warmup_epochs)
        all_iter = len(self.train_loader) * self.config.num_epochs

        
        backbone_lambda = lambda cur_iter: cur_iter/warmup_iter if cur_iter<warmup_iter else \
          (self.config.final_lr/self.config.backbone_lr+0.5*(1 - self.config.final_lr/self.config.backbone_lr) * \
          (1.0+math.cos(math.pi*(cur_iter-warmup_iter)/cos_iter)))

        sem_seg_head_lambda = lambda cur_iter: cur_iter/warmup_iter if cur_iter<warmup_iter else \
          (self.config.final_lr/self.config.sem_seg_head_lr+0.5*(1 - self.config.final_lr/self.config.sem_seg_head_lr) * \
          (1.0+math.cos(math.pi*(cur_iter-warmup_iter)/cos_iter)))

        prototypes_lambda = lambda cur_iter: cur_iter/warmup_iter if cur_iter<warmup_iter else \
          (self.config.final_lr/self.config.prototypes_lr+0.5*(1 - self.config.final_lr/self.config.prototypes_lr) * \
          (1.0+math.cos(math.pi*(cur_iter-warmup_iter)/cos_iter)))
        
        self.model_lr_scheduler = optim.lr_scheduler.LambdaLR(self.model_optimizer,lr_lambda = [backbone_lambda,sem_seg_head_lambda,sem_seg_head_lambda,prototypes_lambda])


    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        Q = torch.exp(out / self.config.epsilon).t() # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] # number of samples to assign
        K = Q.shape[0] # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.config.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            #Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment
        return Q.t()


    def forward(self,inputs):
        #outputs = {}
        with torch.no_grad():
            w = self.model.prototypes.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.model.prototypes.weight.copy_(w)

        outputs = self.model(inputs["img"].flatten(0,1),False)
        
        return outputs

    def compute_loss(self, inputs, outputs):
        
        loss = 0
        if self.config.loss_pix_weight>0:
            projection, sample_id, num_id = self.sampling_pixel(inputs, outputs)
            loss_pix = self.loss_Swapped_Prediction(projection, sample_id, num_id)
            loss = loss + self.config.loss_pix_weight*loss_pix
        if self.config.loss_region_weight>0:
            projection, sample_id, num_id = self.sampling_region(inputs, outputs)
            loss_reg = self.loss_Swapped_Prediction(projection, sample_id, num_id)
            loss = loss + self.config.loss_region_weight*loss_reg
        
        return loss

    def loss_Swapped_Prediction(self, projection, sample_id, num_id):
        projection = F.normalize(projection, dim=1, p=2)
        prototypes = self.model.prototypes.weight.squeeze(-1).squeeze(-1)
        similarity = torch.mm(projection,prototypes.t())

        with torch.no_grad():
            q = self.distributed_sinkhorn(similarity.detach())#[-num_instances:]
            sum_q = torch.zeros((num_id, self.config.num_prototypes),device=self.device)
            sum_q.index_add_(dim=0,index=sample_id,source=q)
            count_ = torch.bincount(sample_id,minlength=num_id)
            count_[count_==0] = 1
            avg_q = sum_q/count_.unsqueeze(-1)
            aligned_q = avg_q[sample_id]
        
        loss = -(aligned_q*F.log_softmax(similarity/self.config.temperature,dim=1)).sum(-1)
        return loss.mean()

    def sampling_region(self, inputs, outputs):
        
        label = inputs["label"]//self.config.mix_num
        num_id = torch.amax(label,dim=(1,2,3,4))+1
        num_id = num_id[-1]
        projection = outputs[1]
        N = self.config.region_sample_per_img

        grid = torch.empty((projection.shape[0], N, 1 ,2),device=self.device).uniform_(-1, 1)
        projection = F.grid_sample(projection,grid,mode="bilinear",align_corners=False,padding_mode="border")
        projection = projection.permute(0,2,3,1).flatten(0,2)
        sample_id = F.grid_sample(label.flatten(0,1).float(),grid,mode="nearest",align_corners=False,padding_mode="border")
        sample_id = sample_id.flatten().long()

        return projection, sample_id, num_id
        #return self.loss_Swapped_Prediction(projection, sample_id, num_id)


    def sampling_pixel(self, inputs, outputs):

        t = inputs["t"]
        t_img = inputs["t_img"]
        label = inputs["label"]
        projection = outputs[1]

        B,M = label.shape[0],label.shape[1]
        H,W = projection.shape[2],projection.shape[3]
        N = self.config.pix_sample_per_img
        T = t.shape[2]
        #random sample
        grid = torch.empty((B, M, N ,2),device=self.device).uniform_(-1, 1)
        pixel_id_s = F.grid_sample(label.flatten(0,1).float(),grid.flatten(0,1).unsqueeze(2),mode="nearest",align_corners=False,padding_mode="border")
        pixel_id_s = pixel_id_s.view(B,-1).long()
        
        #get transforms to original image
        t_s = t.flatten(1,2)[:,:,0:9].view(B,-1,3,3)
        t_s = torch.gather(t_s,1,pixel_id_s.view(B,-1,1,1).expand(-1,-1,3,3))
        #get homo coord
        grid = torch.cat([grid,torch.ones((B,M,N,1),device=self.device)],dim=3).unsqueeze(-1)
        #reverse image trans
        #grid_o = torch.linalg.solve(t_img.unsqueeze(2),grid)
        grid_o,_ = torch.solve(grid,t_img.unsqueeze(2))
        #reverse instance trans
        #grid_o = torch.linalg.solve(t_s,grid_o.flatten(1,2))
        grid_o,_ = torch.solve(grid_o.flatten(1,2),t_s)
        
        #get same object trans and id
        object_id = (pixel_id_s/T).long()
        t_id = torch.arange(0,t.shape[1]*T,dtype=torch.long,device=self.device).view(1,-1,T).expand(B,-1,-1)
        object_t = torch.gather(t,1,object_id.view(B,-1,1,1).expand(-1,-1,T,10))#.flatten(0,1)
        t_id = torch.gather(t_id,1,object_id.view(B,-1,1).expand(-1,-1,T))#.flatten(0,1)
        img_id = object_t[:,:,:,9].long()
        object_t = object_t[:,:,:,0:9].view(B,-1,T,3,3)
        
        #do trans
        grid_t = torch.matmul(object_t, grid_o.unsqueeze(2))
        t_img_t = torch.gather(t_img,1,img_id.view(B,-1,1,1).expand(-1,-1,3,3))
        grid_t = torch.matmul(t_img_t.view(B,-1,T,3,3),grid_t)
        grid_t = grid_t[:,:,:,0:2,0]/grid_t[:,:,:,2:3,0]
        
        #validate trans
        grid_t_id = grid_t.clone()
        grid_t_id[:,:,:,1] = (grid_t[:,:,:,1]/M-(1-1/M))+img_id*2/M
        pixel_id_t = F.grid_sample((label+1).flatten(1,3).unsqueeze(1).float(),grid_t_id,mode="nearest",align_corners=False,padding_mode="zeros")
        pixel_id_t = pixel_id_t.squeeze(1).long()-1
        valid = t_id == pixel_id_t
        valid_1 = valid.sum(dim=2)>1
        valid = valid[valid_1]
        sample_id = torch.arange(0,valid.shape[0],dtype=torch.long,device=self.device)
        num_id = sample_id.shape[0]
        sample_id = sample_id.unsqueeze(-1).expand(-1,T)#.reshape(B,-1)
        sample_id = sample_id[valid]

        #get projection
        grid_t_p = grid_t.clone()
        img_id = img_id + torch.arange(0,B*M,step=M,device=self.device).view(-1,1,1)
        grid_t_p[:,:,:,1] = (grid_t[:,:,:,1]*(projection.shape[2]/(projection.shape[2]+2))/(B*M)-(1-1/(B*M)))+img_id*2/(B*M)
        grid_t_p = grid_t_p[valid_1][valid]
        projection = F.pad(projection, (0,0,1,1), "replicate")
        projection = projection.unsqueeze(0).transpose(1,2).flatten(2,3)
        projection = F.grid_sample(projection,grid_t_p.view(1,1,-1,2),mode="bilinear",align_corners=False,padding_mode="border")
        projection = projection.permute(0,2,3,1).flatten(0,2)

        return projection, sample_id, num_id
        #return self.loss_Swapped_Prediction(projection, sample_id, num_id), projection.shape[0]
        