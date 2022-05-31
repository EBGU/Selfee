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
import cv2
###########################
###set initial prameters###
###########################
home = os.path.dirname(__file__)
initializing = False # if initializing, save a .pkl files, else read saved .pkl files
AMP= True 
CLD = True # use cld loss
maxLambda = 2.0
increaseLambda = False

BYOL = False
if BYOL:
    FrameWork = 'BYOL'
else:
    FrameWork = 'SimSiam'

RGB_3F = True # use RGB channels for past future current

if RGB_3F:
    temporal = '_RGB3F'
    p_RGB_3F = 1.0
else:
    temporal = ''
    p_RGB_3F = 0.0

innerShuffle = True #only use this for data with dramatica batch effect, like mice data. Don't use this with fly!
if innerShuffle:
    Ishuffle = '_innerShuffle_'
else:
    Ishuffle = '_randomShuffle_'


num_workers = 16
batch_size = 128
frame_interval = 2 #2 for mice and 1 for flies
if AMP:
    batch_size=batch_size*2
proj_size = [2048,2048,2048,2048]
pred_size = [2048,512,2048]

steps = 20000  
input_size = [256,192] #for flies, use 224,224 
k_eigen=10
clusters=10
num_iters=10

base_lr=0.05 #per batchsize256
videoSets = '' #fill with the dir name of your dataset
modelName = FrameWork+'_ResNet50_maxLambda'+str(maxLambda)+temporal+str(p_RGB_3F)+Ishuffle+videoSets+"_"

ValidDir = videoSets+'/For_Emb'
TrainDir = videoSets+'/Train_Set'
TestDir = videoSets+'/Test_Set'

SavedDir=home+"/Saved_Models/"+FrameWork+"_CLD_ResNet50_initial_FlyCourtship.pkl"
CheckpointDir=home+"/Saved_Models/"+modelName
log=home+"/Saved_Models/"+modelName
embeddingName="/embedded_by_"+modelName+"steps"
ValidEmbedDir = home+"/Embed"

f = open(log+"train.log",'a')
localtime = time.asctime( time.localtime(time.time()) )
f.write(str(localtime)+'\n')
f.write('frame_interval = '+str(frame_interval)+'\n')
f.write("Learning Rate per 256 images = "+str(base_lr)+'\n')
f.write("Initialte weights with "+SavedDir.split('/')[-1]+'\n')
f.write("max Lambda = "+str(maxLambda)+'\n')
f.write("Lambda increase in cosine style = "+str(increaseLambda)+'\n')
f.close()
f = open(log+"test.log",'a')
localtime = time.asctime( time.localtime(time.time()) )
f.write(str(localtime)+'\n')
f.close()

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

################
###Dataloader###
################

TurboColor =  np.array([[ 48,  18,  59],
                        [ 50,  21,  67],
                        [ 51,  24,  74],
                        [ 52,  27,  81],
                        [ 53,  30,  88],
                        [ 54,  33,  95],
                        [ 55,  36, 102],
                        [ 56,  39, 109],
                        [ 57,  42, 115],
                        [ 58,  45, 121],
                        [ 59,  47, 128],
                        [ 60,  50, 134],
                        [ 61,  53, 139],
                        [ 62,  56, 145],
                        [ 63,  59, 151],
                        [ 63,  62, 156],
                        [ 64,  64, 162],
                        [ 65,  67, 167],
                        [ 65,  70, 172],
                        [ 66,  73, 177],
                        [ 66,  75, 181],
                        [ 67,  78, 186],
                        [ 68,  81, 191],
                        [ 68,  84, 195],
                        [ 68,  86, 199],
                        [ 69,  89, 203],
                        [ 69,  92, 207],
                        [ 69,  94, 211],
                        [ 70,  97, 214],
                        [ 70, 100, 218],
                        [ 70, 102, 221],
                        [ 70, 105, 224],
                        [ 70, 107, 227],
                        [ 71, 110, 230],
                        [ 71, 113, 233],
                        [ 71, 115, 235],
                        [ 71, 118, 238],
                        [ 71, 120, 240],
                        [ 71, 123, 242],
                        [ 70, 125, 244],
                        [ 70, 128, 246],
                        [ 70, 130, 248],
                        [ 70, 133, 250],
                        [ 70, 135, 251],
                        [ 69, 138, 252],
                        [ 69, 140, 253],
                        [ 68, 143, 254],
                        [ 67, 145, 254],
                        [ 66, 148, 255],
                        [ 65, 150, 255],
                        [ 64, 153, 255],
                        [ 62, 155, 254],
                        [ 61, 158, 254],
                        [ 59, 160, 253],
                        [ 58, 163, 252],
                        [ 56, 165, 251],
                        [ 55, 168, 250],
                        [ 53, 171, 248],
                        [ 51, 173, 247],
                        [ 49, 175, 245],
                        [ 47, 178, 244],
                        [ 46, 180, 242],
                        [ 44, 183, 240],
                        [ 42, 185, 238],
                        [ 40, 188, 235],
                        [ 39, 190, 233],
                        [ 37, 192, 231],
                        [ 35, 195, 228],
                        [ 34, 197, 226],
                        [ 32, 199, 223],
                        [ 31, 201, 221],
                        [ 30, 203, 218],
                        [ 28, 205, 216],
                        [ 27, 208, 213],
                        [ 26, 210, 210],
                        [ 26, 212, 208],
                        [ 25, 213, 205],
                        [ 24, 215, 202],
                        [ 24, 217, 200],
                        [ 24, 219, 197],
                        [ 24, 221, 194],
                        [ 24, 222, 192],
                        [ 24, 224, 189],
                        [ 25, 226, 187],
                        [ 25, 227, 185],
                        [ 26, 228, 182],
                        [ 28, 230, 180],
                        [ 29, 231, 178],
                        [ 31, 233, 175],
                        [ 32, 234, 172],
                        [ 34, 235, 170],
                        [ 37, 236, 167],
                        [ 39, 238, 164],
                        [ 42, 239, 161],
                        [ 44, 240, 158],
                        [ 47, 241, 155],
                        [ 50, 242, 152],
                        [ 53, 243, 148],
                        [ 56, 244, 145],
                        [ 60, 245, 142],
                        [ 63, 246, 138],
                        [ 67, 247, 135],
                        [ 70, 248, 132],
                        [ 74, 248, 128],
                        [ 78, 249, 125],
                        [ 82, 250, 122],
                        [ 85, 250, 118],
                        [ 89, 251, 115],
                        [ 93, 252, 111],
                        [ 97, 252, 108],
                        [101, 253, 105],
                        [105, 253, 102],
                        [109, 254,  98],
                        [113, 254,  95],
                        [117, 254,  92],
                        [121, 254,  89],
                        [125, 255,  86],
                        [128, 255,  83],
                        [132, 255,  81],
                        [136, 255,  78],
                        [139, 255,  75],
                        [143, 255,  73],
                        [146, 255,  71],
                        [150, 254,  68],
                        [153, 254,  66],
                        [156, 254,  64],
                        [159, 253,  63],
                        [161, 253,  61],
                        [164, 252,  60],
                        [167, 252,  58],
                        [169, 251,  57],
                        [172, 251,  56],
                        [175, 250,  55],
                        [177, 249,  54],
                        [180, 248,  54],
                        [183, 247,  53],
                        [185, 246,  53],
                        [188, 245,  52],
                        [190, 244,  52],
                        [193, 243,  52],
                        [195, 241,  52],
                        [198, 240,  52],
                        [200, 239,  52],
                        [203, 237,  52],
                        [205, 236,  52],
                        [208, 234,  52],
                        [210, 233,  53],
                        [212, 231,  53],
                        [215, 229,  53],
                        [217, 228,  54],
                        [219, 226,  54],
                        [221, 224,  55],
                        [223, 223,  55],
                        [225, 221,  55],
                        [227, 219,  56],
                        [229, 217,  56],
                        [231, 215,  57],
                        [233, 213,  57],
                        [235, 211,  57],
                        [236, 209,  58],
                        [238, 207,  58],
                        [239, 205,  58],
                        [241, 203,  58],
                        [242, 201,  58],
                        [244, 199,  58],
                        [245, 197,  58],
                        [246, 195,  58],
                        [247, 193,  58],
                        [248, 190,  57],
                        [249, 188,  57],
                        [250, 186,  57],
                        [251, 184,  56],
                        [251, 182,  55],
                        [252, 179,  54],
                        [252, 177,  54],
                        [253, 174,  53],
                        [253, 172,  52],
                        [254, 169,  51],
                        [254, 167,  50],
                        [254, 164,  49],
                        [254, 161,  48],
                        [254, 158,  47],
                        [254, 155,  45],
                        [254, 153,  44],
                        [254, 150,  43],
                        [254, 147,  42],
                        [254, 144,  41],
                        [253, 141,  39],
                        [253, 138,  38],
                        [252, 135,  37],
                        [252, 132,  35],
                        [251, 129,  34],
                        [251, 126,  33],
                        [250, 123,  31],
                        [249, 120,  30],
                        [249, 117,  29],
                        [248, 114,  28],
                        [247, 111,  26],
                        [246, 108,  25],
                        [245, 105,  24],
                        [244, 102,  23],
                        [243,  99,  21],
                        [242,  96,  20],
                        [241,  93,  19],
                        [240,  91,  18],
                        [239,  88,  17],
                        [237,  85,  16],
                        [236,  83,  15],
                        [235,  80,  14],
                        [234,  78,  13],
                        [232,  75,  12],
                        [231,  73,  12],
                        [229,  71,  11],
                        [228,  69,  10],
                        [226,  67,  10],
                        [225,  65,   9],
                        [223,  63,   8],
                        [221,  61,   8],
                        [220,  59,   7],
                        [218,  57,   7],
                        [216,  55,   6],
                        [214,  53,   6],
                        [212,  51,   5],
                        [210,  49,   5],
                        [208,  47,   5],
                        [206,  45,   4],
                        [204,  43,   4],
                        [202,  42,   4],
                        [200,  40,   3],
                        [197,  38,   3],
                        [195,  37,   3],
                        [193,  35,   2],
                        [190,  33,   2],
                        [188,  32,   2],
                        [185,  30,   2],
                        [183,  29,   2],
                        [180,  27,   1],
                        [178,  26,   1],
                        [175,  24,   1],
                        [172,  23,   1],
                        [169,  22,   1],
                        [167,  20,   1],
                        [164,  19,   1],
                        [161,  18,   1],
                        [158,  16,   1],
                        [155,  15,   1],
                        [152,  14,   1],
                        [149,  13,   1],
                        [146,  11,   1],
                        [142,  10,   1],
                        [139,   9,   2],
                        [136,   8,   2],
                        [133,   7,   2],
                        [129,   6,   2],
                        [126,   5,   2],
                        [122,   4,   3]], dtype=np.uint8)

def Turbo(img):
    np_img=np.array(img)
    b, g ,r =cv2.split(np_img) 
    b_color=np.array([TurboColor[xi,0] for xi in b])
    g_color=np.array([TurboColor[xi,1] for xi in g])
    r_color=np.array([TurboColor[xi,2] for xi in r])
    np_color = cv2.merge([b_color,g_color,r_color])
    return Image.fromarray(np_color)

def RandomTransformed(img,resize=input_size,min_scale=0.7,degrees=37,kernel=2,p_hflip=0.5,p_vflip=0.5,p_turbo=0.5,brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, is_train=True):
    #37 degree for XXH 45 for others
    if is_train:
        width, height = F._get_image_size(img)
        w = torch.randint(int(width*min_scale), width, size=(1,)).item()
        h = torch.randint(int(height*min_scale), height, size=(1,)).item()
        i = torch.randint(0, height - h + 1, size=(1,)).item()
        j = torch.randint(0, width - w + 1, size=(1,)).item()
        angle = float(torch.empty(1).uniform_(float(-degrees), float(degrees)).item())
        r1 = torch.rand(1)
        r2 = torch.rand(1)
        r3 = torch.rand(1)
        #blur_factor = torch.randint(0, kernel, size=(1,)).item() *2 +1
        brightness_factor = float(torch.empty(1).uniform_(1-brightness, 1+brightness))
        contrast_factor = float(torch.empty(1).uniform_(1-contrast, 1+contrast))
        saturation_factor = float(torch.empty(1).uniform_(1-saturation, 1+saturation))
        hue_factor = float(torch.empty(1).uniform_(-hue, hue))
        fn_idx = torch.randperm(4)
        img = F.resized_crop(img, i, j, h, w, resize)
        img = F.rotate(img, angle)
        if r1 < p_hflip:
            img = F.hflip(img)
        if r2 < p_vflip:
            img = F.vflip(img)
        if r3 < p_turbo:
            img = Turbo(img)
        #img = F.gaussian_blur(img,(blur_factor,blur_factor))
        for fn_id in fn_idx:
            if fn_id == 0:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3:
                img = F.adjust_hue(img, hue_factor)
    else:
        img = F.resize(img,resize)
    img = F.to_tensor(img)
    return img

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class preShuffleImageDataset(Dataset): 
    '''
    your folder should have following structrues:
    
    root
        -folder1
            --1.jpg
            --2.jpg
            --...
        -folder2
            --1.jpg
            --2.jpg
            --...
        -...
    This dataset should be sampled sequentially. It return batches within same videos respectively. 
    That means, for each batch, we only sampled within one video to avoid batch effect.
    '''
    def __init__(self,
                 root: str,
                 transform = None,
                 p_RGB_3F = p_RGB_3F,
                 frame_interval = frame_interval, # interval = 2 equals to 30fps -> 6fps
                 batchsize = batch_size,
                 innerShuffle = innerShuffle,
                 shuffle = True):
        self.p_RGB_3F = p_RGB_3F
        self.frame_interval = frame_interval
        self.batchsize = batchsize
        self.transform = transform
        self.videoList, self.videoLen = self.videoList(root)
        self.innerShuffle = innerShuffle
        if shuffle:
            self.list = self.preShuffleSampleList()
        else:
            self.list = self.SampleList()

    def videoList(self,root):
        dirlist=os.listdir(root)
        videoList = []
        videoLen = []
        for dirs in dirlist:
            if os.path.isdir(os.path.join(root,dirs)):
                filenum=len(os.listdir(os.path.join(root,dirs)))
                videoList.append(os.path.join(root,dirs))
                videoLen.append(filenum)
        return videoList, videoLen

    def preShuffleSampleList(self):  
        if self.innerShuffle: #force to use innerShuffle for mice data due to batch effects
            for i,lens in enumerate(self.videoLen):
                droplastLen = lens // self.batchsize * self.batchsize
                videoIndex = i * torch.ones(droplastLen).unsqueeze(1)
                imgIndex = torch.randperm(droplastLen).unsqueeze(1)
                try:
                    indexMat = torch.cat((indexMat,torch.cat((videoIndex,imgIndex),1).reshape(-1,self.batchsize,2)),0)
                except:
                    indexMat = torch.cat((videoIndex,imgIndex),1).reshape(-1,self.batchsize,2)
            indexMat=indexMat[torch.randperm(len(indexMat))]
        else:
            for i,lens in enumerate(self.videoLen):
                droplastLen = lens // self.batchsize * self.batchsize
                videoIndex = i * torch.ones(droplastLen).unsqueeze(1)
                imgIndex = torch.arange(droplastLen).unsqueeze(1)
                try:
                    indexMat = torch.cat((indexMat,torch.cat((videoIndex,imgIndex),1)),0)
                except:
                    indexMat = torch.cat((videoIndex,imgIndex),1)
            indexMat=indexMat[torch.randperm(len(indexMat))]
            indexMat=indexMat.reshape(-1,self.batchsize,2)

        for _ in range(len(indexMat)):
            p_rgb =  torch.rand(1)*torch.ones(self.batchsize).unsqueeze(1).unsqueeze(0)
            try:
                probMat = torch.cat((probMat,p_rgb),0)
            except:
                probMat = p_rgb
        indexMat = torch.cat((indexMat,probMat),2)
        indexMat=indexMat.reshape(-1,3)
        return indexMat.tolist()
    
    def SampleList(self):
        for i,lens in enumerate(self.videoLen):
            droplastLen = lens
            videoIndex = i * torch.ones(droplastLen).unsqueeze(1)
            imgIndex = torch.arange(droplastLen).unsqueeze(1)
            try:
                indexMat = torch.cat((indexMat,torch.cat((videoIndex,imgIndex),1)),0)
            except:
                indexMat = torch.cat((videoIndex,imgIndex),1)
        probMat = torch.zeros(len(indexMat)).unsqueeze(1)
        indexMat = torch.cat((indexMat,probMat),1)
        return indexMat.tolist()

    def __getitem__(self, index):
        path ,num ,p_rgb = self.list[index]
        path = int(path)
        num = int (num)
        #print(path ,num ,p_rgb)
        if p_rgb < self.p_RGB_3F:
            current = cv2.imread(self.videoList[path]+'/'+str(num+1)+'.jpg', cv2.IMREAD_GRAYSCALE)
            past = cv2.imread(self.videoList[path]+'/'+str(num-self.frame_interval+1)+'.jpg', cv2.IMREAD_GRAYSCALE)
            if past is None:
                past = current
            future = cv2.imread(self.videoList[path]+'/'+str(num+self.frame_interval+1)+'.jpg', cv2.IMREAD_GRAYSCALE)
            if future is None:
                future = current
            img = cv2.merge([past,current,future])
            img = Image.fromarray(img)
        else:
            img = pil_loader(self.videoList[path]+'/'+str(num+1)+'.jpg')
        if self.transform != None:
            img = self.transform(img)
        return img,self.videoList[path]+'/'+str(num+1)+'.jpg'

    def __len__(self):
        return len(self.list)

class CollateFunction(nn.Module):

    def __init__(self, is_train: bool = True):

        super(CollateFunction, self).__init__()
        self.is_train = is_train

    def forward(self, batch: List[tuple]):
 
        batch_size = len(batch)
        # list of labels
        labels = [item[1] for item in batch]
        if self.is_train:

            # list of transformed images
            transformed = [RandomTransformed(batch[i % batch_size][0]).unsqueeze_(0)
                        for i in range(2 * batch_size)]
            # tuple of transforms
            transformed = (
                torch.cat(transformed[:batch_size], 0),
                torch.cat(transformed[batch_size:], 0)
            )

        else:
            transformed = torch.cat([RandomTransformed(batch[i % batch_size][0],is_train=False).unsqueeze_(0)
                        for i in range(batch_size)])
            # tuple of transforms            
        return transformed, labels

train_set=preShuffleImageDataset(TrainDir,innerShuffle=innerShuffle)
print("TrainSet:"+str(train_set.__len__()))
test_set=preShuffleImageDataset(TestDir,innerShuffle=innerShuffle)
print("TestSet:"+str(test_set.__len__()))
valid_set=preShuffleImageDataset(ValidDir,shuffle=False)
print("ValidSet:"+str(valid_set.__len__()))

TrainLoader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle= False,
    drop_last=True,
    num_workers=16,
    collate_fn= CollateFunction(is_train=True)
)
  
TestLoader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle= False,
    drop_last=True,
    num_workers=16,
    collate_fn= CollateFunction(is_train=True)
)

ValidLoader = torch.utils.data.DataLoader(
    valid_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=16,
    collate_fn= CollateFunction(is_train=False)
)

#################
###Train loops###
#################

ResNet = torchvision.models.resnet50(pretrained = True)
ResNetAndProj = AddProjector(ResNet,proj_size)
predictor = PredAndCLD(pred_size)

if BYOL:
    model = CLD_Byol(ResNetAndProj,predictor,CLD)
else:
    model = CLD_SimSiam(ResNetAndProj,predictor,CLD)

model.eval() ##always save and load under eval()
if initializing:
    torch.save(model.state_dict(),SavedDir)

model.load_state_dict(torch.load(SavedDir),strict=False)
model.train()    #This behaviour can cause problems when storing the state_dict() of a model while in a mode and lately loading it in a model with a different mode, as the attributes of this class change. To avoid this issue, we recommend converting the model to eval mode before storing or loading the state dictionary.
device =  torch.device('cuda')
model.to(device)

#SDG optimizer and adjust lr
optimizer = torch.optim.SGD(model.parameters(),
                            lr=base_lr*batch_size/256,
                            momentum=0.9,
                            weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=batch_size*base_lr/256, total_steps=steps,pct_start=0.025)

avg_loss = np.array([0.,0.,0.])
avg_collapse_level = np.array([0.,0.])

# main training loop
TrainGenerator = iter(TrainLoader)
TestGenerator = iter(TestLoader)

scaler=torch.cuda.amp.GradScaler(enabled=AMP)
time_start=time.time()

for i in range(steps):

    if increaseLambda:
        Lambda = maxLambda*0.5*(1-np.cos(i*np.pi/steps))
    else:
        Lambda = maxLambda

    optimizer.zero_grad()

    try:
        # Samples the batch
        (x0, x1), _= next(TrainGenerator)

    except StopIteration:
        # restart the generator if the previous generator is exhausted.
        train_set=preShuffleImageDataset(TrainDir,innerShuffle=innerShuffle)
        TrainLoader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle= False,
            drop_last=True,
            num_workers=16,
            collate_fn= CollateFunction(is_train=True)
        )
        TrainGenerator = iter(TrainLoader)
        (x0, x1), _ = next(TrainGenerator)

    # move images to the gpu
    x0 = x0.to(device)
    x1 = x1.to(device)

    with torch.cuda.amp.autocast(enabled=AMP):
        loss, loss_SS, loss_CLD,out_H,output_L=model(x0,x1,Lambda)

    #loss.backward()
    scaler.scale(loss).backward()
    #optimizer.step()
    scaler.step(optimizer)
    scaler.update()
    if BYOL:
        model.update_moving_average()
    #use one cycle scheduler
    scheduler.step()
    
    # calculate the per-dimension standard deviation of the outputs
    # we can use this later to check whether the embeddings are collapsing
    output_H = out_H.detach()
    output_H = torch.nn.functional.normalize(output_H, dim=1)
    output_H_std = torch.std(output_H, 0)
    output_H_std = output_H_std.mean()

    output_L = output_L.detach()
    output_L = torch.nn.functional.normalize(output_L, dim=1)
    output_L_std = torch.std(output_L, 0)
    output_L_std = output_L_std.mean()
    loss_array = np.array([loss.item(),loss_SS.item(),loss_CLD.item()])
    collapse_level = np.array([1-math.sqrt(proj_size[-1]) * output_H_std.item(),1-math.sqrt(pred_size[1]*2) * output_L_std.item()])
    # use moving averages to track the loss and standard deviation
    w = 0.9
    avg_loss = avg_loss * w + loss_array * (1 - w)
    avg_collapse_level = avg_collapse_level * w + collapse_level * (1 - w)
    #Lambda = k * loss_array[1] / loss_array[2]

    if (i+1)  % 5000 == 0 or i == 0:
        ###validation
        model.eval()
        torch.save(model.state_dict(),CheckpointDir+str(i+1)+".pkl")
        output=[]
        for img, fname in ValidLoader:
            img = img.to(device)
            with torch.cuda.amp.autocast(enabled=AMP):
                embeddings=model.infer(img)
            for j in range(len(embeddings)):
                output.append((embeddings[j].cpu().numpy(),fname[j]))     
        output=np.array(output,dtype=object)    
        np.save(ValidEmbedDir+embeddingName+str(i+1)+".npy",output)   
        model.train()
        del img,embeddings
        
    if (i+1)  % 100 == 0:
        
        ###evaluation
        model.eval()
        try:
            (t0, t1), _= next(TestGenerator)
        except StopIteration:
            test_set=preShuffleImageDataset(TestDir,innerShuffle=innerShuffle)
            TestLoader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle= False,
                drop_last=True,
                num_workers=16,
                collate_fn= CollateFunction(is_train=True)
            )
            TestGenerator = iter(TestLoader)
            (t0, t1), _= next(TestGenerator)
        t0 = t0.to(device)
        t1 = t1.to(device)
        with torch.no_grad():  
            with torch.cuda.amp.autocast(enabled=AMP):
                T_loss, T_loss_SS, T_loss_CLD,NR_1,NR_2 =model(t0,t1,Lambda)            
        print(f'[Test {i+1:8d}] '
            f'Loss = {T_loss.item():.5f} | '
            f'Loss_SS = {T_loss_SS.item():.5f} | '
            f'Loss_CLD = {T_loss_CLD.item():.5f} ')
        model.train()
        # the level of collapse is large if the standard deviation of the l2
        # normalized output is much smaller than 1 / sqrt(dim)
        # print intermediate results
        print(f'[Steps {i+1:8d}] '
            f'Loss = {avg_loss[0]:.5f} | '
            f'Loss_SS = {avg_loss[1]:.5f} | '
            f'Loss_CLD = {avg_loss[2]:.5f} | '
            f'Collapse Level Low Dim: {collapse_level[1]:.5f} | '
            f'Collapse Level High Dim: {collapse_level[0]:.5f}')
        f = open(log+"train.log",'a')
        f.write(f'[Steps {i+1:8d}] '
            f'Loss = {avg_loss[0]:.5f} | '
            f'Loss_SS = {avg_loss[1]:.5f} | '
            f'Loss_CLD = {avg_loss[2]:.5f} | '
            f'Collapse Level Low Dim: {collapse_level[1]:.5f} | '
            f'Collapse Level High Dim: {collapse_level[0]:.5f}\n')
        f.close()
        f = open(log+"test.log",'a')
        f.write(f'[Test {i+1:8d}] '
            f'Loss = {T_loss.item():.5f} | '
            f'Loss_SS = {T_loss_SS.item():.5f} | '
            f'Loss_CLD = {T_loss_CLD.item():.5f} \n')            
        f.close()
        time_end=time.time()
        print('time cost',time_end-time_start,'s')
        time_start=time.time()
        del t0,t1,T_loss, T_loss_SS, T_loss_CLD,NR_1,NR_2   

     
