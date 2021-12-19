import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import manifold
from matplotlib.colors import ListedColormap
from multiprocessing import Pool
import dtw
from dtw_pytorch import pdtw
from scipy.spatial.distance import cosine
from scipy import stats
from scipy.spatial.distance import cdist
import torch
import time
from pytorch_metric_learning.distances import CosineSimilarity
import os
from scipy import stats



trainID = 'W1118'
inferID = 'NorpA'
home = os.path.dirname(__file__)+'/'
TrainDir = home+trainID
InferDir = home+inferID
SavedDir = home+'SavedModels/'
video_len =10000

def extractNpy(TrainDir):
    embs = []
    fileList = os.listdir(TrainDir)
    for f in fileList:
        npy = np.load(os.path.join(TrainDir,f),allow_pickle=True)
        embs.append(np.stack(npy[:,0]))
    return embs

refEmb = extractNpy(TrainDir)
queryEmb = extractNpy(InferDir)

class dynamic_time_warping():
    def __init__(self,query,ref):
        self.query = query
        self.ref = ref
        self.ref_index = []
        for i in range(len(query)):
            for j in range(len(ref)):
                self.ref_index.append([i,j])
        self.device = torch.device('cuda')
        self.dist = CosineSimilarity()
    def distance(self,query,ref):
        query = torch.tensor(query).to(self.device)
        ref = torch.tensor(ref).to(self.device)
        correlation = self.dist(query,ref)
        correlation = correlation.cpu().numpy().astype('float64')
        correlation = 1 - correlation
        return correlation
    def __call__(self,index):
        pairs = self.ref_index[index]
        distance_metric = self.distance(self.query[pairs[0]],self.ref[pairs[1]])
        alignment = pdtw(distance_metric,keep_internals=False,step_pattern=dtw.asymmetric,open_end=False,open_begin=False)
        #print(distance)
        return alignment.index2


try:
    QeuryWarped = np.load(SavedDir+inferID+'_vs_'+trainID+'.npy')
    RefWarped = np.load(SavedDir+trainID+'_vs_'+trainID+'.npy')
except:
    #buid ref curve
    selfwarping = dynamic_time_warping(refEmb,refEmb)
    with Pool(8) as p:
        RefWarped = p.map(selfwarping,range(len(refEmb)**2))
    RefWarped = np.asarray(RefWarped)
    print(RefWarped.shape)
    np.save(SavedDir+trainID+'_vs_'+trainID+'.npy',RefWarped)

    #build exp curve
    selfwarping = dynamic_time_warping(queryEmb,refEmb)
    with Pool(8) as p:
        QeuryWarped = p.map(selfwarping,range(len(refEmb)*len(queryEmb)))
    QeuryWarped = np.asarray(QeuryWarped)
    print(QeuryWarped.shape)
    np.save(SavedDir+inferID+'_vs_'+trainID+'.npy',QeuryWarped)



mean = np.mean(RefWarped,axis=0)
std = stats.sem(RefWarped,axis=0) # use 3 times sem 
t = np.array(range(video_len))
plt.plot(t, mean,color="#0000FF",alpha=0.5)
plt.fill_between(t,mean-std,mean+std, color="#9999FF",alpha=0.5,label="WT baseline") 
#plt.show()


mean = np.mean(QeuryWarped,axis=0)
std = stats.sem(QeuryWarped,axis=0)  # use 3 times sem 
t = np.array(range(video_len))
plt.plot(t, mean,color="#FF0000",alpha=0.5)
plt.fill_between(t,mean-std,mean+std, color="#FF9999",alpha=0.5,label="NorpA mutant") 

plt.legend()
plt.show()

pValue =[]
for i in range(video_len):
    try:
        sig = stats.mannwhitneyu(RefWarped[:,i],QeuryWarped[:,i],alternative='two-sided')
        pValue.append(sig[1])
    except:
        pValue.append(1.0)
pValue = np.asarray(pValue)
pRank = stats.rankdata(pValue, method='min')
qValue = pValue * len(pValue)/pRank  # do Benjamini and Hochberg correction to p value

try:
    MaxP = max(pRank[np.where(qValue<0.05)])
    SigIndex = np.where(pRank<=MaxP)[0]
    print(SigIndex)
except:
    print('no significance!')

plt.plot(t,0.05-qValue,color="#FF0000",alpha=0.5)
plt.show()
plt.plot(t,0.05-pValue,color="#FF0000",alpha=0.5)
plt.show()