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

try: 
    fileList = np.load(saved+'/FileList.npy',allow_pickle=True)
    rawdata = np.load(saved+'/embeddings.npy',allow_pickle=True)
    labelset = np.load(saved+'/labels.npy',allow_pickle=True)

except:
    rawdata = []
    labelset = []
    fileList = os.listdir(Labels)
    for txt in fileList:
        npy = txt.split('.')[0]+'.npy'
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
        averaged = np.concatenate([averaged,FandZ],axis=1).astype(np.float16)

        print(averaged.shape)
        rawdata.append(averaged)
        LabelList =  np.array(['others']*len(npy))
        with open(os.path.join(Labels,txt)) as f:
            for i, lines in enumerate(f.readlines()):
                x = lines.split(" ")
                while ('' in x):
                    x.remove('')
                if x[4] == 'start\n':
                    start = int(x[0])-1
                if x[4] == 'stop\n' or x[4] == 'stop':
                    stop = int(x[0])-1
                    LabelList[start:stop+1] = x[2]
        labelset.append(LabelList)
        print(len(LabelList))
        del npy,averaged,LabelList
    rawdata = np.array(rawdata)    
    labelset = np.array(labelset)
    fileList = np.array(fileList)
    np.save(saved+'/FileList.npy',fileList)
    np.save(saved+'/labels.npy',labelset)
    np.save(saved+'/embeddings.npy',rawdata)
dataset = []
for subset in rawdata:
    x = subset[:,usedFeature,:].transpose(1,0,2)
    dataset.append(np.concatenate(x,axis=1))

uniqueDefinedLabels = np.unique(np.concatenate(labelset).flatten())
########################
###define k-NN sorter###
########################
    
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
        assert(temp in ['Mode','Aver','None'])
        topkDist, topkIndice = torch.topk(distMat,k,dim=1)
        topkScore = torch.exp(topkDist/tau).cpu().numpy()
        topkIndice = topkIndice.cpu().numpy()
        topkClass = self.mapLabels(topkIndice)
        classification = np.zeros((len(distMat),len(self.classlist)))
        for i,classes in enumerate(self.classlist):
            classification[:,i] = np.sum(np.where(topkClass==classes,topkScore,0),1)
        score= classification * self.balanceWeigh
        if temp == 'Aver':
            aver = slideAver(classification,windowSize,moreFeature=False)
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
            classification = np.argmax(classification,1)
            classification = self.mapClass(averaged)
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

def f1_eval(testLabel,scores):
    scores = scores.reshape(len(uniqueDefinedLabels),-1).transpose()
    inferedLabel = np.argmax(scores,1)
    f1 = f1_score(testLabel,inferedLabel, average='macro')
    return 'f1 score',f1,True

########################
###Choose window size###
########################
logee = logger(saved+'/result.txt')
AllData = np.concatenate(dataset)[:,0:2048]
AllLabel = np.concatenate(labelset)
nullSorter  = knnSorter()
AutoCorr = []
for i in range(0,len(AllData)-glance,glance):
    query = AllData[i:i+glance]
    ref = torch.tensor(np.expand_dims(AllData[i+glance//2],0))
    AutoCorr.append(1-nullSorter.cosdist(query,ref).cpu().numpy().astype('float64').flatten())
AutoCorr = np.asarray(AutoCorr)
t = np.arange(glance)
mean = 1 - np.mean(AutoCorr,axis=0)
std = stats.sem(AutoCorr)
plt.plot(t, mean,color="#FF0000",alpha=1.0)
plt.fill_between(t,mean-std,mean+std, color="#FF9999",alpha=0.5) 
plt.savefig(saved+'/AutoCorr.pdf',format='pdf')
plt.show()

########################
###Test k-NN accuracy###
########################

labelmapper = lambda x: np.where(uniqueDefinedLabels==x)[0][0]
inverseMapClass =  np.vectorize(labelmapper)
mapClass = np.vectorize(lambda j: uniqueDefinedLabels[j])

logee('Features used:')
for f in usedFeature:
    logee(Features[f])
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
        #trainOneHot = np.array([np.where(sorter.classlist == i,1,0) for i in trainLabel])
        #train_data = lgb.Dataset(trainSet, label=inverseMapClass(trainLabel))

        testSet = dataset[test_index[0]].astype(np.float64)
        testLabel = labelset[test_index[0]]
        testOneHot = np.array([np.where(uniqueDefinedLabels == i,1,0) for i in testLabel])
        #test_data = lgb.Dataset(testSet, label=inverseMapClass(testLabel))

        sorter = LGBMClassifier(
            num_leaves = 31,
            objective = 'multiclass',
            class_weight = 'balanced'
            )
        sorter.fit(
            trainSet,
            trainLabel,
            eval_set = (testSet,testLabel),
            #eval_metric = f1_eval,
            early_stopping_rounds = 5
        )
        with open(saved+'/'+str(test_index[0])+'model.pkl','wb') as f:
            pickle.dump(sorter,f)
        #bst = lgb.train(param, train_data, valid_sets=test_data, early_stopping_rounds=5)
        #bst.save_model(saved+'/'+str(test_index[0])+'model.txt', num_iteration=bst.best_iteration)
        #scores = bst.predict(testSet, num_iteration=bst.best_iteration)
        scores = sorter.predict_proba(testSet)
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
        
        f1 = f1_score(testLabel,inferedLabel, average='macro')
        mAP = AP(testOneHot,scores,average='macro')
        allInfer.append(inferedLabel)
        allTruth.append(testLabel)
        allScore.append(scores)
        logee(fileList[test_index[0]])
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
    finalAccuracy[labelmapper(i),labelmapper(j)] += 1
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
    finalAccuracy[labelmapper(i),labelmapper(j)] += 1
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
with open(saved+'/'+str(RasterFile)+'model.pkl','rb') as f:
    sorter = pickle.load(f)

#bst = lgb.Booster(model_file=saved+'/'+str(RasterFile)+'model.txt') 
#scores = bst.predict(rasterData, num_iteration=bst.best_iteration)
scores = sorter.predict_proba(rasterData)

if tempAgg == 'Aver':
    aver = slideAver(scores,windowSize,moreFeature=False)
    with Pool(8) as p:
        averaged=np.array(p.map(aver,range(len(scores))))
    scores = averaged
    inferedLabel = np.argmax(scores,1)
    inferRaster = mapClass(inferedLabel)
elif tempAgg == 'Mode':
    inferedLabel = np.argmax(scores,1)
    Mode = slideMode(inferedLabel,windowSize)
    with Pool(8) as p:
        averaged=np.array(p.map(Mode,range(len(inferedLabel))))
    inferRaster = mapClass(averaged)
elif tempAgg == 'None':
    inferedLabel = np.argmax(scores,1)
    inferRaster = mapClass(inferedLabel)

frameDic = {}
inferDic = {}
for key in uniqueDefinedLabels:
    frameDic.update({key:[]})
    inferDic.update({key:[]})

for i,(l,k) in enumerate(zip(rasterLabel,inferRaster)):
    frameDic[l].append(i)
    inferDic[k].append(i)

colorList = ["crimson", "gold", "forestgreen","aqua","slateblue",]

for d,c in zip(frameDic.keys(),colorList):
    plt.eventplot(np.asarray(frameDic[d]),lineoffsets=0,linewidths=1,colors=c)
    plt.eventplot(np.asarray(inferDic[d]),lineoffsets=1.5,linewidths=1,colors=c)
plt.savefig(saved+'/raster.pdf',format='pdf')
plt.show()
'''