from InferModel import ConvModelInfer
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
from torchvision import datasets, transforms, models
from PIL import Image, ImageOps
from torch.nn import Parameter
from spectral_clustering import spectral_clustering, pairwise_cosine_similarity, KMeans
import time
import torchvision.transforms.functional as F
import sys
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
import cv2
###########################
###set initial prameters###
###########################

AMP= True 
RGB_3F = True # use RGB channels for past future current
frame_interval = 2

if RGB_3F:
    temporal = '_RGB3F'
    p_RGB_3F = 1.0

else:
    temporal = ''
    p_RGB_3F = 0.0
reverse = False

num_workers = 16
batch_size = 512
if AMP:
    batch_size=batch_size*2

windowSize = 0

#input_size = [224,224] #for fly
input_size = [256,192] #for mice
inferDir = ' ' #fill this with where your images are

embeddingName=os.listdir(inferDir)
embeddingDir = " /" # fill this with your output dir, must contain / at last!!!!!!

CheckpointDir=" " #fill this with your chpt file address

################
###Dataloader###
################
def RandomTransformed(img,resize=input_size,reverse=reverse,is_train=True):
    if is_train:
        pass
    else:
        img = F.resize(img,resize)
    if reverse:
        img = ImageOps.invert(img)
    img = F.to_tensor(img)
    return img

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class screeningDataset(Dataset): 
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
    This dataset access one folder each time and root SHOULD NOT provided to it!!!
    '''
    def __init__(self,
                 folder: str,
                 transform = None,
                 RGB_3F = RGB_3F,
                 frame_interval = 2 # interval = 2 equals to 30fps -> 6fps
                 ):

        self.RGB_3F = RGB_3F
        self.frame_interval = frame_interval
        self.transform = transform
        self.list = self.SampleList(folder)
    def SampleList(self,folder):
        sample = []
        if os.path.isdir(folder):
            filenum=len(os.listdir(folder))
            i = 1
            while(i <= filenum):
                sample.append((folder,i))
                i += 1
        return sample

    def __getitem__(self, index):
        path ,num = self.list[index]
        if self.RGB_3F:
            current = cv2.imread(path+'/'+str(num)+'.jpg', cv2.IMREAD_GRAYSCALE)
            past = cv2.imread(path+'/'+str(num-self.frame_interval)+'.jpg', cv2.IMREAD_GRAYSCALE)
            if past is None:
                past = current
            future = cv2.imread(path+'/'+str(num+self.frame_interval)+'.jpg', cv2.IMREAD_GRAYSCALE)
            if future is None:
                future = current
            img = cv2.merge([past,current,future])
            img = Image.fromarray(img)
        else:
            img = pil_loader(path+'/'+str(num)+'.jpg')
        if self.transform != None:
            img = self.transform(img)
        return img,path+'/'+str(num)+'.jpg'

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
            pass
        else:
            transformed = torch.cat([RandomTransformed(batch[i % batch_size][0],is_train=False).unsqueeze_(0)
                        for i in range(batch_size)])
            # tuple of transforms            
        return transformed, labels

#################
###Infer loops###
#################
model = ConvModelInfer(CheckpointDir = CheckpointDir,windowSize=windowSize)

device =  torch.device('cuda')


for folder in embeddingName:
    infer_set = screeningDataset(os.path.join(inferDir,folder),frame_interval=frame_interval)
    inferLoader = torch.utils.data.DataLoader(
        infer_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=16,
        collate_fn= CollateFunction(is_train=False)
    )
    output=[]
    for img, fname in inferLoader:
        img = img.to(device)
        with torch.cuda.amp.autocast(enabled=AMP):
            embeddings = model(img,windowSize)
        for j in range(len(embeddings)):
            if windowSize != 0:
                output.append((embeddings[j].cpu().numpy(),fname[j*windowSize+windowSize//2]))     
            else:
                output.append((embeddings[j].cpu().numpy(),fname[j])) 
    output=np.array(output,dtype=object)    
    np.save(embeddingDir+folder+".npy",output)   
    print(len(output))


  