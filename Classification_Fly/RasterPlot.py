import numpy as np
import os
from scipy import stats
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_distances
import torch
from pytorch_metric_learning.distances import CosineSimilarity
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score as AP
import time
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

home = os.path.dirname(__file__)+'/'
Labels = home+'Annotations'
Embeds = home+'Final_1'
saved = Embeds+'_SavedData'
raster = home+'RasterPlot/B_CS-M_w1118-F_M-F_1__2019-05-29_at_16-38-40__2'
try:
    os.mkdir(saved)
except:
    pass
rasterStart = 1150
rasterEnd = -1
windowSize = 21
tempAgg = 'Mode'
########################
####prepare datasets####
########################

BehavDic ={
    0 : 'nothing',
    2 : 'chasing',
    3 : 'winging',
    4 : 'precopu',
    5 : 'copulat',
    6 : 'unlabel',
}

try: 
    fileList = np.load(saved+'/FileList.npy',allow_pickle=True)
    dataset = np.load(saved+'/embeddings.npy',allow_pickle=True)
    labelset = np.load(saved+'/labels.npy',allow_pickle=True)

except:
    dataset = []
    labelset = []
    fileList = os.listdir(Labels)
    for txt in fileList:
        npy = txt.split('.')[0]+'.npy'
        npy = np.load(os.path.join(Embeds,npy),allow_pickle=True)
        npy = np.stack(npy[:,0])
        dataset.append(npy)
        LabelList = []
        with open(os.path.join(Labels,txt)) as f:
            for lines in f.readlines():
                    if lines[0] != '#':
                        start = int(lines.split(' ')[0])-1
                        end = int(lines.split(' ')[1])
                        behav = BehavDic[int(lines.split(' ')[2])]
                        LabelList += [behav]*(end-start)
        labelset.append(LabelList)
        print(len(LabelList))
        del npy, LabelList
    dataset = np.array(dataset)
    labelset = np.array(labelset)
    fileList = np.array(fileList)
    np.save(saved+'/FileList.npy',fileList)
    np.save(saved+'/embeddings.npy',dataset)
    np.save(saved+'/labels.npy',labelset)


########################
###define k-NN sorter###
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

class MedianBlur():

    def __init__(self,dataArray,windowSize):
        assert(windowSize % 2 == 1) #window size has to be odd
        self.dataArray = dataArray
        self.windowSize = windowSize

    def __call__(self,index):
        minIndex = max(0,index-self.windowSize//2)
        maxIndex = min(index+self.windowSize//2,len(self.dataArray)-1)
        avg = np.median(self.dataArray[minIndex:maxIndex+1],axis=0)
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
    
class knnSorter():

    def __init__(self,refdata=None,labels=None,cpu=True,balance=True):
        self.cpu = cpu
        if cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')
        self.dist = CosineSimilarity()
        if not(refdata is None):
            classlist , counts = np.unique(labels,return_counts=True)
            self.labels = labels
            self.mapLabels = np.vectorize(lambda j: labels[j])
            self.mapClass = np.vectorize(lambda j: classlist[j]) # five kind of behaviors
            self.ref = torch.tensor(refdata).half().to(self.device)
            self.classlist = classlist
            if balance:
                self.balanceWeigh = np.sum(counts)/counts
                #self.balanceWeigh = 1 - counts/np.sum(counts)
            else:
                self.balanceWeigh=np.ones(len(classlist))

    def cosdist(self,query,ref=None): # this return cosine similarity for topk but NOT distance!!!
        if self.cpu:
            query = torch.tensor(query).float().to(self.device)
            ref = ref.float().to(self.device)
        else:
            query = torch.tensor(query).half().to(self.device)
        if ref is None:
            correlation = self.dist(query,query)
        else:
            correlation = self.dist(query,ref)
        return correlation #return GPU tensor

    def weighedKNN(self,distMat,k,tau,windowSize,temp): #adopted from Unsupervised feature learning via non-parametric instance discrimination.
        topkDist, topkIndice = torch.topk(distMat,k,dim=1)
        topkScore = torch.exp(topkDist/tau).cpu().numpy()
        topkIndice = topkIndice.cpu().numpy()
        topkClass = self.mapLabels(topkIndice)
        classification = np.zeros((len(distMat),len(self.classlist)))
        for i,classes in enumerate(self.classlist):
            classification[:,i] = np.sum(np.where(topkClass==classes,topkScore,0),1)
        score= classification * self.balanceWeigh
        if temp == 'Aver':
            aver = slideAver(score,windowSize,moreFeature=False)
            with Pool(8) as p:
                averaged=np.array(p.map(aver,range(len(score))))
            score = averaged
            classification = np.argmax(score,1)
            classification = self.mapClass(classification)
            return classification,score
        elif temp == 'Median':
            aver = MedianBlur(score,windowSize)
            with Pool(8) as p:
                averaged=np.array(p.map(aver,range(len(score))))
            score = averaged
            classification = np.argmax(score,1)
            classification = self.mapClass(classification)
            return classification,score
        elif temp == 'Mode':
            classification = np.argmax(score,1)
            Mode = slideMode(classification,windowSize)
            with Pool(8) as p:
                averaged=np.array(p.map(Mode,range(len(classification))))
            classification = self.mapClass(averaged)
            return classification,score
        elif temp == 'None':
            classification = np.argmax(score,1)
            classification = self.mapClass(classification)
            return classification,score

    def __call__(self,queryData,k=200,tau=0.07,windowSize=21,temp = 'Mode'):  #querydata should be tensors on GPU!!!
        distMat = self.cosdist(queryData,self.ref) 
        return self.weighedKNN(distMat,k,tau,windowSize,temp)

class logger(object):
    def __init__(self,logFile):
        self.logFile = logFile
        localtime = time.asctime( time.localtime(time.time()) )
        self.__call__(localtime)
    def __call__(self,x):
        print(x)
        f = open(self.logFile,'a')
        f.write(str(x)+'\n')
        f.close()
#######################
######Raster plot######
#######################

labelDic = {
    'nothing' : 0,
    'chasing' : 1,
    'winging' : 2,
    'precopu' : 3,
    'copulat' : 4,
}

rasterData = np.load(raster+'.npy',allow_pickle=True)
rasterData = np.stack(rasterData[rasterStart:rasterEnd,0])

rasterLabel = []
with open(raster+'.txt') as f:
    for lines in f.readlines():
            if lines[0] != '#':
                start = int(lines.split(' ')[0])-1
                end = int(lines.split(' ')[1])
                behav = BehavDic[int(lines.split(' ')[2])]
                rasterLabel += [behav]*(end-start)
print(len(rasterLabel))
rasterLabel = rasterLabel[rasterStart:rasterEnd]

finalData =  np.concatenate(dataset).astype(np.float64)
finalLabel =  np.concatenate(labelset)
FinalSorter = knnSorter(finalData,finalLabel)
inferRaster,_ = FinalSorter(rasterData,windowSize=windowSize,temp=tempAgg)
frameDic = {
    0 : [],
    1 : [],
    2 : [],
    3 : [],
    4 : []
}
inferDic = {
    0 : [],
    1 : [],
    2 : [],
    3 : [],
    4 : []
}

for i,(l,k) in enumerate(zip(rasterLabel,inferRaster)):
#for i,k in enumerate(inferRaster):   
    frameDic[labelDic[l]].append(i)
    inferDic[labelDic[k]].append(i)


colorList = ["crimson", "gold", "forestgreen","aqua","slateblue",]

for d,c in zip(frameDic.keys(),colorList):
    plt.eventplot(np.asarray(frameDic[d]),lineoffsets=0,linewidths=1,colors=c)
    plt.eventplot(np.asarray(inferDic[d]),lineoffsets=1.5,linewidths=1,colors=c)
plt.savefig(saved+'/raster.pdf',format='pdf')
plt.show()
