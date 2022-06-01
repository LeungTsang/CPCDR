import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

import glob
import os
import time
import PIL.Image as pil
import json
import random
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import sklearn.cluster as sc

import datasets
#from networks import *
from utils import *
from detectron2_model.modeling.backbone import build_backbone
from detectron2_model.modeling.meta_arch.semantic_seg import build_sem_seg_head
from detectron2_model.config import get_cfg

datasets = "kitti"

if datasets == "cs":
  paths = "leftImg8bit/val/*/*.png"
  size = (768,384)
else:
  paths = "data_semantics/training/image_2/*.png"
  size = (1280,384)


img_paths = sorted(glob.glob(paths))

t = T.Compose([T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
cfg = get_cfg()
cfg.merge_from_file("depth2seg/detectron2_model/Base-Panoptic-FPN.yaml")
backbone = build_backbone(cfg)
sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
projector = MLP(in_channels = 128, out_channels=128, layer_num = 1)
prototypes = nn.Conv2d(in_channels=projector.out_channels, out_channels=1000, kernel_size=1, padding=0, bias = False)
model = SwAVModel(backbone,sem_seg_head,projector,prototypes).to("cuda")
model.eval()


pretrained_dict = torch.load("drive/MyDrive/depth2seg_models_ab/swav_kitti_depth_cp_6_mix_full_x/weights_14_203364/model.pth")
model_dict = model.state_dict()
pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
print(pretrained_dict.keys())
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

prototypes = model.prototypes.weight.squeeze(-1).squeeze(-1).detach()
prototypes = F.normalize(prototypes, dim=1, p=2)
prototypes_np = prototypes.cpu().numpy()
clustering = sc.AgglomerativeClustering(n_clusters=19,affinity="cosine",linkage="average")
assignments = clustering.fit_predict(prototypes_np)

num_seg = 19
num_gt = 19
intersection = np.zeros((num_gt, num_seg))
with torch.no_grad():
    for i,img_path in enumerate(img_paths):
        #print(img_path)
        img = pil.open(img_path).resize(size, pil.ANTIALIAS)
        if datasets == "cs":
          gt = pil.open(img_path.replace("leftImg8bit","gtFine").replace(".png","_labelIds.png"))
        else:
          gt = pil.open(img_path.replace("image_2","semantic"))
        gt = np.array(gt.resize(size, pil.NEAREST)).astype(np.uint16)

        img = t(img).unsqueeze(0).to("cuda")
        _,projection = model(img)
        projection = F.interpolate(projection, gt.shape, mode="bilinear", align_corners=False)
        projection = F.normalize(projection, dim=1, p=2)
        projection = projection.detach().permute(0,2,3,1).squeeze(0)
        
        seg = torch.matmul(projection,prototypes.t())
        seg = assignments[torch.argmax(seg,dim=2).cpu().numpy()].astype(np.uint16)
        
        mask = (gt >= 0) & (gt < num_gt)
        intersection_ = np.bincount((num_seg*gt[mask] + seg[mask]).flatten(), minlength=num_seg*num_gt).reshape(num_gt, num_seg)
        intersection += intersection_

union = -intersection + intersection.sum(axis=1,keepdims=True) + intersection.sum(axis=0,keepdims=True)
cost = -intersection/union
cost[np.isnan(cost)] = -1
i, j = linear_sum_assignment(cost)

hist = intersection
acc = hist[i,j].sum() / hist.sum()
acc_cls = hist[i,j] / hist.sum(axis=0)[j]
acc_cls = np.nanmean(acc_cls)
iu = hist[i,j] / (hist.sum(axis=1) + hist.sum(axis=0)[j] - hist[i,j])
mean_iu = np.nanmean(iu)
freq = hist.sum(axis=0)[j] / hist.sum()
fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
labels = ['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain',
                      'sky','person','rider','car','truck','bus','train','motorcycle','bicycle']
cls_iu = dict(zip(labels, iu))
metrics = {
    "Overall Acc": acc,
    "Mean Acc": acc_cls,
    "FreqW Acc": fwavacc,
        "Mean IoU": mean_iu
    }
metrics.update(cls_iu)

iu = hist[i,j] / (hist.sum(axis=1) + hist.sum(axis=0)[j] - hist[i,j])
mean_iu = np.nanmean(iu)

print("---------")
print(metrics)



