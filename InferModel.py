import math
import torch.nn as nn
import torch.nn.functional as nnF
from typing import Tuple, List, Union
import os
import numpy as np
import torch
import torchvision
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from PIL import Image
from torch.nn import Parameter
from spectral_clustering import spectral_clustering, pairwise_cosine_similarity, KMeans
import time
import torchvision.transforms.functional as F
import sys
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
import copy

###########################
###set initial prameters###
###########################
proj_size = [2048,2048,2048,2048]
pred_size = [2048,512,2048]
CLD = True
###########################
####Pretrained ResNet50####
###########################

def timer(func):
    def wrapper(*args, **kw):
        time_start=time.time()  
        result = func(*args, **kw)
        time_end=time.time()
        print('time cost',time_end-time_start,'s')
        return result
    return wrapper

class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = nnF.normalize(x, dim=1).mm(nnF.normalize(self.weight, dim=0))
        return out

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out

class AddProjector(nn.Module):

    def __init__(self,backbone,proj_size=[512,512,512]):
        super(AddProjector, self).__init__()  
        self.backbone = backbone
        self.backbone.fc = nn.Identity()   
        proj_layer=[]
        for i in range(len(proj_size) - 2):
            proj_layer.append(nn.Linear(proj_size[i], proj_size[i + 1]))
            proj_layer.append(nn.BatchNorm1d(proj_size[i + 1]))
            proj_layer.append(nn.ReLU(inplace=True))
        proj_layer.append(nn.Linear(proj_size[-2], proj_size[-1]))
        proj_layer.append(nn.BatchNorm1d(proj_size[-1]))
        self.projector = nn.Sequential(*proj_layer)

    def forward(self, x):
        out = self.backbone(x)
        out = self.projector(out)
        return out

class PredAndCLD(torch.nn.Module):
    def __init__(self,pred_size=[512,128,512]):
        super(PredAndCLD, self).__init__()  
        #branches for BYOL
        self.predictor = nn.Sequential(
            nn.Linear(pred_size[0], pred_size[1]),
            nn.BatchNorm1d(pred_size[1]),
            nn.ReLU(inplace=True),
            nn.Linear(pred_size[1], pred_size[2]))

        #branches for CLD
        self.groupDis = nn.Sequential(
            NormedLinear(pred_size[0], pred_size[1]*2),
            Normalize(2))
        
    def forward(self, x, online = True):
          
        CLD = self.groupDis(x)

        if online:  #for online network only
            pred = self.predictor(x)
            return pred, CLD

        return CLD

# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val
        
def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

class CLD_Byol(nn.Module):

    def __init__(self, backbone,predictor,CLD,moving_average_decay = 0.99):
        super().__init__()
        self.online_encoder = backbone
        self.target_encoder = self._get_target_encoder()
        self.CLD = CLD
        self.predictor = predictor
        self.target_ema_updater =  EMA(moving_average_decay)

    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def update_moving_average(self):
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def loss_fn_SS(self,z1,z2): #z1 = (emb1,pred1) z2 = (emb2,pred2)
        t1, p1 = z1
        t2, p2 = z2
        return 2 - nnF.cosine_similarity(p1, t2.detach(), dim=-1).mean() - nnF.cosine_similarity(p2, t1.detach(), dim=-1).mean()

    def loss_fn_cld(self,z1,z2,cld_t=0.07): # z1 = (img1_online_cld,img1_target_cld) z2 = (img2_online_cld,img2_target_cld)
        cluster_label1_1, centroids1_1 = KMeans(z1[0], K=clusters, Niters=num_iters)
        cluster_label2_1, centroids2_1 = KMeans(z2[1], K=clusters, Niters=num_iters)       
        affnity1_1 = torch.mm(z1[0], centroids2_1.t()) 
        affnity2_1 = torch.mm(z2[1], centroids1_1.t())
        loss_CLD_1 = 0.5*(nnF.cross_entropy(affnity1_1.div_(cld_t), cluster_label2_1)+nnF.cross_entropy(affnity2_1.div_(cld_t), cluster_label1_1))
        cluster_label1_2, centroids1_2 = KMeans(z1[1], K=clusters, Niters=num_iters)
        cluster_label2_2, centroids2_2 = KMeans(z2[0], K=clusters, Niters=num_iters)       
        affnity1_2 = torch.mm(z1[1], centroids2_2.t()) 
        affnity2_2 = torch.mm(z2[0], centroids1_2.t())
        loss_CLD_2 = 0.5*(nnF.cross_entropy(affnity1_2.div_(cld_t), cluster_label2_2)+nnF.cross_entropy(affnity2_2.div_(cld_t), cluster_label1_2))
        return 0.5*(loss_CLD_1+loss_CLD_2)

    def forward(self, img1, img2,Lambda):
        y1 = self.online_encoder(img1)
        y2 = self.online_encoder(img2)
        pred1, online_cld1 = self.predictor(y1)
        pred2, online_cld2 = self.predictor(y2)

        with torch.no_grad():
            emb1 = self.target_encoder(img1)
            emb2 = self.target_encoder(img2)
            target_cld1 = self.predictor(emb1,online = False)
            target_cld2 = self.predictor(emb2,online = False)
            #target_cld1.detach_()
            #target_cld2.detach_()

        loss_SS = self.loss_fn_SS((emb1,pred1),(emb2,pred2))
        if self.CLD:       
            loss_CLD = self.loss_fn_cld((online_cld1,target_cld1),(online_cld2,target_cld2))
        else:
            loss_CLD=torch.tensor(0)

        loss= loss_SS+Lambda*loss_CLD
        return loss, loss_SS, loss_CLD, y1, online_cld1
        
    @torch.no_grad()
    def infer(self,x):
        return self.online_encoder(x)

class CLD_SimSiam(nn.Module):
    def __init__(self, backbone,predictor,CLD):
        super().__init__()
        self.backbone = backbone
        self.loss_fn_cld = nn.CrossEntropyLoss()
        self.CLD = CLD
        self.predictor = predictor

    def loss_fn_SS(self,z1,z2):
        t1, p1 = z1
        t2, p2 = z2
        return 2 - nnF.cosine_similarity(p1, t2.detach(), dim=-1).mean() - nnF.cosine_similarity(p2, t1.detach(), dim=-1).mean()

    def forward(self, y1, y2,Lambda,cld_t=0.07):
        y1 = self.backbone(y1)
        y2 = self.backbone(y2)
        z1_H,z1_L = self.predictor(y1)
        z2_H,z2_L = self.predictor(y2)
        loss_SS = self.loss_fn_SS((y1,z1_H),(y2,z2_H))
        if self.CLD:
            cluster_label1, centroids1 = KMeans(z1_L, K=clusters, Niters=num_iters)
            cluster_label2, centroids2 = KMeans(z2_L, K=clusters, Niters=num_iters)       
            affnity1 = torch.mm(z1_L, centroids2.t()) 
            affnity2 = torch.mm(z2_L, centroids1.t())
            loss_CLD = 0.5*(self.loss_fn_cld(affnity1.div_(cld_t), cluster_label2)+self.loss_fn_cld(affnity2.div_(cld_t), cluster_label1))
        else:
            loss_CLD=torch.tensor(0)

        loss= loss_SS+Lambda*loss_CLD
        return loss, loss_SS, loss_CLD, z1_H, z1_L
        
    @torch.no_grad()
    def infer(self,x):
        return self.backbone(x)

#################
###Infer loops###
#################

class ConvModelInfer():
    def __init__(self,CheckpointDir=None,windowSize=16,BYOL = False):
        ResNet = torchvision.models.resnet50(pretrained = False)
        ResNetAndProj = AddProjector(ResNet,proj_size)
        predictor = PredAndCLD(pred_size)
        if BYOL:
            self.model = CLD_Byol(ResNetAndProj,predictor,CLD)
        else:
            self.model = CLD_SimSiam(ResNetAndProj,predictor,CLD)

        self.model.eval() ##always save and load under eval()
        self.model.load_state_dict(torch.load(CheckpointDir))
        device =  torch.device('cuda')
        self.model.to(device)
        self.pooling = nn.AvgPool1d(windowSize,stride=windowSize)
    
    def __call__(self,x,pool):
        with torch.no_grad():
            out = self.model.infer(x)
            if pool != 0:
                out = out.t_().unsqueeze_(0)
                out = self.pooling(out)[0].t_()
        return out
        
