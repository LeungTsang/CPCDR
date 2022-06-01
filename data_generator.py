import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch
import os
import numpy as np
from utils import sec_to_hm_str
import time

class Config():

    def __init__(self):
        self.input_size = (384,768)

        self.data_path = "/content/"
        self.mix_num = 4
        self.num_workers = 6
        self.epoch = 1
        self.use_prepared_data = False
        self.generate_data = True
        self.expectation = [2,4]
        self.output_path = "/content/prepared_data/"
    
config = Config()

train_dataset = datasets.cp_dataset(config)            
train_loader = DataLoader(train_dataset, None, True,
            num_workers=config.num_workers, pin_memory=True, drop_last=False)
            
to_pil = T.ToPILImage()

if __name__ == '__main__':
    num_step = config.epoch*len(train_loader)//4
    start_time = time.time()
    step = 0
    for epoch in range(config.epoch):
        out = os.path.join(config.output_path,"epoch_{}".format(epoch))
        if not os.path.exists(out):
            os.makedirs(out)
        num_instances = torch.zeros(train_dataset.__len__(), dtype=torch.long)
        num_id = torch.zeros(train_dataset.__len__(), dtype=torch.long)
        for batch_idx, inputs in enumerate(train_loader):
            step = step+1
            save_path_img = os.path.join(out,"{:010d}_img.png".format(step))
            save_path_label = os.path.join(out,"{:010d}_label.png".format(step))
            save_path_transforms = os.path.join(out,"{:010d}_transforms.pt".format(step))

            print(epoch, step)
            print("time left: "+ sec_to_hm_str((num_step-step)*(time.time()-start_time)/step))
            inputs["img"].save(save_path_img)
            inputs["label"].save(save_path_label)
            torch.save(inputs["t"], save_path_transforms)