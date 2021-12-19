import numpy as np
import os
from scipy import stats
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics.pairwise import cosine_distances
import torch
from pytorch_metric_learning.distances import CosineSimilarity
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score as AP
import time
from scipy import signal
import pickle
import lightgbm as lgb
from lightgbm import LGBMClassifier
########################
# defined behaviors
# 0 = Nothing
# 1 = oriantation
# 2 = chasing
# 3 = winging
# 4 = pre-copulation 
# 5 = copulation
# 6 = unlabeled
########################
Features = {
    0: 'raw',
    1: 'movingAver',
    2: 'movingStd',
    3: 'averAmp',
    4: 'mainFreq',
    5: 'mainAmp',
    }

labelDic = {
    "others":"others",
    "social_interest":"social_interest",
    "mount":"mount",
    "intromission": "intromission",
    "ejaculation":"ejaculation",
}


home = os.path.dirname(__file__)+'/'
Labels = home+'Annotations'
Embeds = home+'XXH_0903_2'
saved = Embeds+'_SavedData_LGBM'
raster = home+'RasterPlot/esr1_mm41_20191104_MPOA_fDIO_EYFP_VTA_retro_DIO_Flpo_OVX_2th_new'
#raster = home+'RasterPlot/行为八_通道2_main_20210310164554_173201_crop'
try:
    os.mkdir(saved)
except:
    pass
RasterFile = 6
rasterStart = 0
rasterEnd = -1
tempAgg = 'Mode'
assert(tempAgg in ['Mode','Aver','None'])
usedFeature = [0,1,2,3,4,5]
windowSize=81
glance = 300
'''
param = {
    'metric': '',
    'is_unbalance': True,
    'objective':'multiclassova',
    'num_class': 5, 
    'num_leaves':31,
    }
'''
########################
####prepare datasets####
########################
class slideAver():

    def __init__(self,dataArray,windowSize,moreFeature=True):
        assert(windowSize % 2 == 1) #window size has to be odd
        self.dataArray = dataArray
        self.windowSize = windowSize
        self.moreFeature = moreFeature

    def __call__(self,index):
        minIndex = max(0,index-self.windowSize//2)
        maxIndex = min(index+self.windowSize//2,len(self.dataArray)-1)
        seq = self.dataArray[minIndex:maxIndex+1]
        avg = np.mean(seq,axis=0)
        if self.moreFeature:
            std = np.std(seq,axis=0)
            sAmp = np.sqrt(np.mean((seq-avg)**2,axis=0))
            return [self.dataArray[index],avg,std,sAmp] 
        else:
            return avg

class slideMode():

    def __init__(self,dataArray,windowSize):
        assert(windowSize % 2 == 1) #window size has to be odd
        self.dataArray = dataArray
        self.windowSize = windowSize

    def __call__(self,index):
        minIndex = max(0,index-self.windowSize//2)
        maxIndex = min(index+self.windowSize//2,len(self.dataArray)-1)
        avg = stats.mode(self.dataArray[minIndex:maxIndex+1])[0][0]
        return avg 

class MVMP_STFT():
    def __init__(self,dataArray,windowSize):
        assert(windowSize % 2 == 1) #window size has to be odd
        self.windowSize = windowSize
        self.dataArray = dataArray.transpose()
    def __call__(self,index):
        f,_,Z = signal.stft(self.dataArray[index],nperseg=self.windowSize,noverlap=self.windowSize-1)
        Z = np.abs(Z)
        mainF = f[np.argmax(Z,axis=0)]
        maxZ = np.max(Z,axis=0)
        return [mainF,maxZ]

labelset = np.load(saved+'/labels.npy',allow_pickle=True)
uniqueDefinedLabels = np.unique(np.concatenate(labelset).flatten())

try:
    rasterData = np.load(raster+'_processed.npy')
except:
    npy = raster+'.npy'
    npy = np.load(os.path.join(Embeds,npy),allow_pickle=True)
    npy = np.stack(npy[:,0])
    aver = slideAver(npy,windowSize)
    with Pool(8) as p:
        averaged=np.array(p.map(aver,range(len(npy))))
    averaged = np.array(averaged)
    mySTFT = MVMP_STFT(npy,windowSize)
    with Pool(8) as p:
        FandZ =np.array(p.map(mySTFT,range(npy.shape[-1])))
    FandZ = np.transpose(FandZ,(2,1,0))
    rasterData = np.concatenate([averaged,FandZ],axis=1).astype(np.float16)
    print(rasterData.shape)
    np.save(raster+'_processed.npy',rasterData)

rasterData = rasterData[:,usedFeature,:].transpose(1,0,2)
rasterData = np.concatenate(rasterData,axis=1)
rasterLabel =  np.array(['others']*len(rasterData))

with open(raster+'.txt') as f:
    for i, lines in enumerate(f.readlines()):
        x = lines.split(" ")
        while ('' in x):
            x.remove('')
        if x[4] == 'start\n':
            start = int(x[0])-1
        if x[4] == 'stop\n' or x[4] == 'stop':
            stop = int(x[0])-1
            rasterLabel[start:stop+1] = x[2]

print(len(rasterLabel))

########################
###Test k-NN accuracy###
########################

labelmapper = lambda x: np.where(uniqueDefinedLabels==x)[0][0]
inverseMapClass =  np.vectorize(labelmapper)
mapClass = np.vectorize(lambda j: uniqueDefinedLabels[j])

bagOfExperts = []
i = 0
while(1):
    try:
        with open(saved+'/'+str(i)+'model.pkl','rb') as f:
            sorter = pickle.load(f)
        bagOfExperts.append(sorter)
        i += 1
    except:
        break

try:
    allInfer = np.load(raster+'_inferedLabels.npy')
except:
    allInfer = []
    for sorter in bagOfExperts:
        scores = sorter.predict_proba(rasterData)
        if tempAgg == 'Aver':
            aver = slideAver(scores,windowSize,moreFeature=False)
            with Pool(8) as p:
                averaged=np.array(p.map(aver,range(len(scores))))
            scores = averaged
            inferedLabel = np.argmax(scores,1)
            inferedLabel = mapClass(inferedLabel)
        elif tempAgg == 'Mode':
            inferedLabel = np.argmax(scores,1)
            Mode = slideMode(inferedLabel,windowSize)
            with Pool(8) as p:
                averaged=np.array(p.map(Mode,range(len(inferedLabel))))
            inferedLabel = mapClass(averaged)
        elif tempAgg == 'None':
            inferedLabel = np.argmax(scores,1)
            inferedLabel = mapClass(inferedLabel)
        allInfer.append(inferedLabel)
    allInfer = np.array(allInfer)
    np.save(raster+'_inferedLabels.npy',allInfer)

inferedLabel = stats.mode(allInfer,axis=0)[0][0]
f1 = f1_score(rasterLabel,inferedLabel, average='macro')
print('f1 score is '+str(f1))

frameDic = {}
inferDic = {}
for key in uniqueDefinedLabels:
    frameDic.update({key:[]})
    inferDic.update({key:[]})

for i,(l,k) in enumerate(zip(rasterLabel,inferedLabel)):
    frameDic[l].append(i)
    inferDic[k].append(i)

colorList = ["crimson", "gold", "forestgreen","aqua","slateblue",]

for d,c in zip(frameDic.keys(),colorList):
    plt.eventplot(np.asarray(frameDic[d]),lineoffsets=0,linewidths=1,colors=c)
    plt.eventplot(np.asarray(inferDic[d]),lineoffsets=1.5,linewidths=1,colors=c)
plt.savefig(saved+'/raster.pdf',format='pdf')
plt.show()
