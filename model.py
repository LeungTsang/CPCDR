import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import copy

class SwAVModel(nn.Module):
    def __init__(self, backbone, sem_seg_head, projector, prototypes):
        super(SwAVModel, self).__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.projector = projector
        self.prototypes = prototypes

        
    def forward(self, x, predict = False):
        features = self.backbone(x)
        representation = self.sem_seg_head(features)
        projection = self.projector(representation)
        
        if predict:
            projection = F.normalize(projection, dim=1, p=2)
            similarity = self.prototypes(projection)
            return representation, projection, similarity
        else:
            return representation, projection


class MLP(nn.Module):

    def __init__(self, in_channels, hidden_channels = None, out_channels = None, layer_num = 2, get_intermediate = False):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        if hidden_channels is None:
            self.hidden_channels = self.in_channels
        else:
            self.hidden_channels = hidden_channels

        if out_channels is None:
            self.out_channels = self.hidden_channels
        else:
            self.out_channels = out_channels
            
        self.layer_num = layer_num
        self.get_intermediate = get_intermediate
        layers_list = []
        in_dim = self.in_channels

        for i in range(self.layer_num):
            layers_list.append(("layer"+str(i), MLP._block(in_dim, self.hidden_channels)))
            in_dim = self.hidden_channels
        
        self.layers = nn.Sequential(OrderedDict(layers_list))
        if not self.get_intermediate:
            self.convs = nn.Conv2d(in_channels=self.hidden_channels,
                        out_channels=self.out_channels,
                        kernel_size=1,
                        padding=0,
                        bias=False
                        )


    def forward(self, representation):
        if self.get_intermediate:
            return self.layers(representation)
        else:
            return self.convs(self.layers(representation))

    @staticmethod
    def _block(in_channels, features):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=1,
                            padding=0,
                            bias=True,
                        ),
                    ),
                    ("norm", nn.BatchNorm2d(num_features=features)),
                    ("relu", nn.ReLU(inplace=True))
                ]
            )
        )