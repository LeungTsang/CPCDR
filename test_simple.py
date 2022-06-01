import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import time
import wandb
import PIL.Image as pil
import json
import random
import torchvision.transforms as T

import datasets
from networks import *
from utils import *

config_encoder = "resnet18"
config_decoder = "FPN"
img_path = "000000_10_c.png"

return_layers_dict = {"FPN": {'layer4': 'layer4', 'layer3': 'layer3','layer2': 'layer2','layer1': 'layer1'},
            "DeepLabV3":{'layer4': 'layer4'},
            "DeepLabV3Plus":{'layer4': 'layer4','layer1': 'layer1'}}
                          
encoder_dict = {"resnet18":resnet18, "resnet34":resnet34, "resnet50":resnet50, "resnet101":resnet101}
decoder_dict = {"FPN": FPNDecoder, "DeepLabV3":DeepLabV3, "DeepLabV3Plus": DeepLabV3Plus}


trans = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])

img = pil.open(img_path).resize((640,192), pil.ANTIALIAS)
img = trans(img).unsqueeze(0)

encoder = encoder_dict[config_encoder]()
return_layers = return_layers_dict[config_decoder]
encoder = IntermediateLayerGetter(encoder, return_layers=return_layers)
decoder = decoder_dict[config_decoder](config_encoder)

model = BaseModel(encoder,decoder)


pretrained_dict = torch.load("model.pth", map_location=torch.device('cpu'))

model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

model.eval()
with torch.no_grad():
    representation = decoder(encoder(img))

rep = representation.detach().cpu().numpy()

np.save(img_path[:-4]+"_rep.npy",rep)
#print(representation.shape)






