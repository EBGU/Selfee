import numpy as np
import os
from scipy import stats
from sklearn.model_selection import KFold
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
from scipy import stats
from random import shuffle
from Robust_correlation_analysis_framework import RCAF
home = os.path.dirname(__file__)+'/'
Labels = home+'label'
Embeds = home+'JAABA_feat'
saved = Embeds+'_SavedData'
try:
    os.mkdir(saved)
except:
    pass


########################
####prepare datasets####
########################


try: 
    fileList = np.load(saved+'/FileList.npy',allow_pickle=True)
    dataset = np.load(saved+'/embeddings.npy',allow_pickle=True)
    labelset = np.load(saved+'/labels.npy',allow_pickle=True)

except:
    dataset = []
    labelset = []
    fileList = os.listdir(Labels)
    for f in fileList:
        npy = np.load(os.path.join(Embeds,f),allow_pickle=True)
        #npy = np.stack(npy[:,0]) #only for neural network embs
        npy = np.nan_to_num(npy)
        dataset.append(npy)
        LabelList = np.load(os.path.join(Labels,f))[0:len(npy)]
        labelset.append(LabelList)
        print(npy.shape,LabelList.shape)
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

    def weighedKNN(self,distMat,k,tau): #adopted from Unsupervised feature learning via non-parametric instance discrimination.
        topkDist, topkIndice = torch.topk(distMat,k,dim=1)
        topkScore = torch.exp(topkDist/tau).cpu().numpy()
        topkIndice = topkIndice.cpu().numpy()
        topkClass = self.mapLabels(topkIndice)
        classification = np.zeros((len(distMat),len(self.classlist)))
        for i,classes in enumerate(self.classlist):
            classification[:,i] = np.sum(np.where(topkClass==classes,topkScore,0),1)
        score= classification * self.balanceWeigh
        classification = np.argmax(score,1)
        classification = self.mapClass(classification)
        return classification,score

    def __call__(self,queryData,k=200,tau=0.07):  #querydata should be tensors on GPU!!!
        distMat = self.cosdist(queryData,self.ref) 
        return self.weighedKNN(distMat,k,tau)

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
###Test k-NN accuracy###
########################
logee = logger(saved+'/result.txt')
AllData = np.concatenate(dataset)
AllLabel = np.concatenate(labelset)

try:
    allInfer = np.load(saved+'/Inferedlabels.npy',allow_pickle=True)
    allTruth = np.load(saved+'/groundTruth.npy',allow_pickle=True)
    allScore = np.load(saved+'/InferedScores.npy',allow_pickle=True)
except:
    X = list(range(len(AllLabel)))
    loo = KFold(n_splits=6,shuffle=False)#,random_state=42)
    allInfer = []
    allTruth = []
    allScore = []
    for train_index, test_index in loo.split(X):
        trainSet = AllData[train_index]
        trainLabel = AllLabel[train_index]
        testSet = AllData[test_index]
        testLabel = AllLabel[test_index]
        sorter = knnSorter(trainSet,trainLabel)
        inferedLabel,scores = sorter(testSet)
        testOneHot = np.array([np.where(sorter.classlist == i,1,0) for i in testLabel])
        f1 = f1_score(testLabel,inferedLabel, average='macro')
        mAP = AP(testOneHot,scores,average='macro')
        allInfer.append(inferedLabel)
        allTruth.append(testLabel)
        allScore.append(scores)
        logee('f1 score is '+str(f1))
        logee('mAP score is '+str(mAP))

    np.save(saved+'/Inferedlabels.npy',np.array(allInfer))
    np.save(saved+'/groundTruth.npy',np.array(allTruth))
    np.save(saved+'/InferedScores.npy',np.array(allScore))

logee('\nFinal!')
allInfer = np.concatenate(allInfer)
allTruth = np.concatenate(allTruth)
allScore = np.concatenate(allScore)

regression = allScore / np.sum(allScore,axis=-1,keepdims=True)
regression = regression * np.arange(7).reshape(1,7)
regression = regression.sum(axis=-1)
logee('Pearson correlation coefficient,Two-tailed p-value')
logee(str(RCAF(allTruth,regression)))
ind = list(range(len(allTruth)))
shuffle(ind)
ind = ind[0:500]
t = np.arange(7)
mean = np.array([(regression[allTruth==i]).mean() for i in t])
std = np.array([(regression[allTruth==i]).std() for i in t])
plt.plot(t, mean,color="#FF0000",alpha=1.0)
plt.fill_between(t,mean-std,mean+std, color="#FF9999",alpha=0.5) 
plt.savefig(saved+'/Corr.pdf',format='pdf')
plt.show()

classList = np.unique(allTruth)
OneHot = np.array([np.where( classList== i,1,0) for i in allTruth])
f1 = f1_score(allTruth, allInfer,average='micro')# average='macro')
logee('f1 score (micro-averaged) is '+str(f1))
mAP = AP(OneHot,allScore,average='micro')#, average='macro')
logee('mAP score (micro-averaged) is '+str(mAP))

logee(np.unique(allTruth,return_counts=True))
finalAccuracy = np.zeros((7,7))
#recall
for i,j in zip(allInfer,allTruth):
    finalAccuracy[i,j] += 1
total = np.sum(finalAccuracy,axis = 0)
finalAccuracy =finalAccuracy/total
averageAcc = np.mean([finalAccuracy[i,i] for i in range(len(finalAccuracy))])
logee('Final! mean recall is '+str(averageAcc))
logee(finalAccuracy)
plt.imshow(finalAccuracy,cmap='plasma',vmin=0,vmax=1)
plt.colorbar()
plt.savefig(saved+'/recall.pdf',format='pdf')
plt.show()

finalAccuracy = np.zeros((7,7))
#precision
for i,j in zip(allTruth,allInfer):
    finalAccuracy[i,j] += 1
total = np.sum(finalAccuracy,axis = 0)
finalAccuracy =finalAccuracy/total
averageAcc = np.mean([finalAccuracy[i,i] for i in range(len(finalAccuracy))])
logee('Final! mean precision is '+str(averageAcc))
logee(finalAccuracy)
plt.imshow(finalAccuracy,cmap='plasma',vmin=0,vmax=1)
plt.colorbar()
plt.savefig(saved+'/precision.pdf',format='pdf')
plt.show()

