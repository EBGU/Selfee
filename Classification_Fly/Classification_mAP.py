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
try:
    os.mkdir(saved)
except:
    pass
RasterFile = 4
rasterStart = 2401
rasterEnd = 4000
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


########################
###Choose window size###
########################
logee = logger(saved+'/result.txt')
AllData = np.concatenate(dataset)
AllLabel = np.concatenate(labelset)
nullSorter  = knnSorter()
AutoCorr = []
for i in range(0,len(AllData),100):
    query = AllData[i:i+100]
    ref = torch.tensor(np.expand_dims(AllData[i+50],0))
    AutoCorr.append(1-nullSorter.cosdist(query,ref).cpu().numpy().astype('float64').flatten())
AutoCorr = np.asarray(AutoCorr)
t = np.arange(100)
mean = 1 - np.mean(AutoCorr,axis=0)
std = stats.sem(AutoCorr)
plt.plot(t, mean,color="#FF0000",alpha=1.0)
plt.fill_between(t,mean-std,mean+std, color="#FF9999",alpha=0.5) 
plt.savefig(saved+'/AutoCorr.pdf',format='pdf')
plt.show()

########################
###Test k-NN accuracy###
########################
labelDic = {
    'nothing' : 0,
    'chasing' : 1,
    'winging' : 2,
    'precopu' : 3,
    'copulat' : 4,
}
logee('Temporal Aggeration with:'+tempAgg)
if tempAgg == 'Mode':
    logee('mAP is not reliable during Mode mode!')
try:
    allInfer = np.load(saved+'/Inferedlabels.npy',allow_pickle=True)
    allTruth = np.load(saved+'/groundTruth.npy',allow_pickle=True)
    allScore = np.load(saved+'/InferedScores.npy',allow_pickle=True)
except:
    X = list(range(len(dataset)))
    loo = LeaveOneOut()
    allInfer = []
    allTruth = []
    allScore = []
    for train_index, test_index in loo.split(X):
        trainSet = np.concatenate([dataset[i] for i in train_index]).astype(np.float64)
        trainLabel = np.concatenate([labelset[i] for i in train_index])
        testSet = dataset[test_index[0]].astype(np.float64)
        testLabel = labelset[test_index[0]]
        #knn_sorter = KNeighborsClassifier(n_neighbors=5,weights='distance',algorithm='brute',metric=cosine_distances,n_jobs=16)
        #knn_sorter.fit(trainSet,trainLabel)
        #inferedLabel = knn_sorter.predict(testSet)
        sorter = knnSorter(trainSet,trainLabel)
        inferedLabel,scores = sorter(testSet,windowSize=windowSize,temp=tempAgg)
        testOneHot = np.array([np.where(sorter.classlist == i,1,0) for i in testLabel])
        f1 = f1_score(testLabel,inferedLabel, average='macro')
        mAP = AP(testOneHot,scores,average='macro')
        allInfer.append(inferedLabel)
        allTruth.append(testLabel)
        allScore.append(scores)
        logee(fileList[test_index])
        logee('f1 score is '+str(f1))
        logee('mAP score is '+str(mAP))

    np.save(saved+'/Inferedlabels.npy',np.array(allInfer))
    np.save(saved+'/groundTruth.npy',np.array(allTruth))
    np.save(saved+'/InferedScores.npy',np.array(allScore))

logee('\nFinal!')
allInfer = np.concatenate(allInfer)
allTruth = np.concatenate(allTruth)
allScore = np.concatenate(allScore)
classList = np.unique(allTruth)
OneHot = np.array([np.where( classList== i,1,0) for i in allTruth])
f1 = f1_score(allTruth, allInfer, average='macro')
logee('f1 score is '+str(f1))
mAP = AP(OneHot,allScore, average='macro')
logee('mAP score is '+str(mAP))

logee(np.unique(allTruth,return_counts=True))
finalAccuracy = np.zeros((5,5))
#recall
for i,j in zip(allInfer,allTruth):
    finalAccuracy[labelDic[i],labelDic[j]] += 1
total = np.sum(finalAccuracy,axis = 0)
finalAccuracy =finalAccuracy/total
averageAcc = np.mean([finalAccuracy[i,i] for i in range(len(finalAccuracy))])
logee('Final! mean recall is '+str(averageAcc))
logee(finalAccuracy)
plt.imshow(finalAccuracy,cmap='plasma',vmin=0,vmax=1)
plt.colorbar()
plt.savefig(saved+'/recall.pdf',format='pdf')
plt.show()

#precision
for i,j in zip(allTruth,allInfer):
    finalAccuracy[labelDic[i],labelDic[j]] += 1
total = np.sum(finalAccuracy,axis = 0)
finalAccuracy =finalAccuracy/total
averageAcc = np.mean([finalAccuracy[i,i] for i in range(len(finalAccuracy))])
logee('Final! mean precision is '+str(averageAcc))
logee(finalAccuracy)
plt.imshow(finalAccuracy,cmap='plasma',vmin=0,vmax=1)
plt.colorbar()
plt.savefig(saved+'/precision.pdf',format='pdf')
plt.show()



'''
#######################
######Raster plot######
#######################
rasterData = dataset[RasterFile]
rasterLabel = labelset[RasterFile]
rasterData = rasterData[rasterStart:rasterEnd]
rasterLabel = rasterLabel[rasterStart:rasterEnd]
finalData =  np.concatenate([dataset[i] for i in np.delete(range(len(dataset)),RasterFile)]).astype(np.float64)
finalLabel =  np.concatenate([labelset[i] for i in np.delete(range(len(dataset)),RasterFile)])
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
    frameDic[labelDic[l]].append(i)
    inferDic[labelDic[k]].append(i)

colorList = ["crimson", "gold", "forestgreen","aqua","slateblue",]

for d,c in zip(frameDic.keys(),colorList):
    plt.eventplot(np.asarray(frameDic[d]),lineoffsets=0,linewidths=1,colors=c)
    plt.eventplot(np.asarray(inferDic[d]),lineoffsets=1.5,linewidths=1,colors=c)
plt.savefig(saved+'/raster.pdf',format='pdf')
plt.show()
'''