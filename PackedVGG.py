#coding:utf8
import torch
import torch.nn as nn
from torchvision.models import vgg19
from collections import namedtuple

class Vgg19(torch.nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        features = list(vgg19(pretrained = True).features)[:23]
        # features的第3，8，15，22层分别是: relu1_2,relu2_2,relu3_3,relu4_2
        self.features = nn.ModuleList(features).eval() 
        
    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {3,8,13,22}:
                results.append(x)
        
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_2', 'relu4_2'])
        return vgg_outputs(*results)
