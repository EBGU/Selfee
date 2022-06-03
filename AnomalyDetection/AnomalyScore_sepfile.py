import numpy as np
from multiprocessing import Pool
from harmony import harmonize
import torch
from pytorch_metric_learning.distances import CosineSimilarity
import matplotlib.pyplot as plt
import time
from PIL import Image
import pandas as pd
import cv2
from scipy import stats
import os
#torch.multiprocessing.set_start_method('spawn')
home = "/home/harold/Documents/Harold_D206PC_Data/ZW_Lab/Selfee_figures/AnomalyDetection_New/"
ref_npy ='Ctrl_ref'# "ppk23"
query_npy ='CIS' # "Trh"
neg_control ='Ctrl_neg'# "CS"
SavedDir = home+'SavedModels/'
log = SavedDir+"log.log"
video_len =10000
plotWindow =100
try:
    os.makedirs(SavedDir)
except:
    pass
def extractNpy(TrainDir):
    embs = []
    dirs = []
    fileList = os.listdir(TrainDir)
    fileList.sort()
    for f in fileList:
        npy = np.load(os.path.join(TrainDir,f),allow_pickle=True)
        embs.append(np.stack(npy[:video_len,0]))
        dirs.append(np.stack(npy[:video_len,1]))
    return embs,dirs

refEmb,ref_dir = extractNpy(home+ref_npy)
queryEmb,query_dir = extractNpy(home+query_npy)
negEmb,neg_dir = extractNpy(home+neg_control)
query_dir = np.concatenate(query_dir)
class blockwiseAnomaly():
    def __init__(self,query,ref):
        self.query = query
        self.ref = ref
        self.ref_index = []
        for i in range(len(query)):
            for j in range(len(ref)):
                self.ref_index.append([i,j])
        self.device = torch.device('cpu')
        self.dist = CosineSimilarity()
    def distance(self,query,ref):
        query = torch.tensor(query).float().to(self.device)
        ref = torch.tensor(ref).float().to(self.device)
        correlation = self.dist(query,ref)
        #correlation = correlation.cpu().numpy().astype('float64')
        correlation = 1 - correlation
        return correlation
    def maskMetric(self,size,width = 101): # this return a POSITIVE diagl matrix!!!
        assert (width+1) % 2 == 0, "Width has to be odd"
        x = np.ones((size,size))
        x = x-np.triu(x,k=(width+1)//2)-np.tril(x,k=-(width+1)//2)
        return torch.tensor(x).float().to(self.device)

    def basal(self,index):
        pairs = self.ref_index[index]
        distance_metric = self.distance(self.query[pairs[0]],self.ref[pairs[1]])
        if pairs[0] == pairs[1]:
            distance_metric += 2 * self.maskMetric(len(self.query[pairs[0]]))
        return torch.amin(distance_metric,dim=1)
    def anomaly(self,index):
        pairs = self.ref_index[index]
        distance_metric = self.distance(self.query[pairs[0]],self.ref[pairs[1]])
        # if pairs[0] == pairs[1]:
        #     distance_metric += 2 * self.maskMetric(len(self.query[pairs[0]]))
        return torch.amin(distance_metric,dim=1)

# ref = np.load(ref_npy+steps+".npy",allow_pickle=True)
# refer_data = np.stack(ref[0:30000,0])
# query = np.load(query_npy+steps+".npy",allow_pickle=True)
# query_data = np.stack(query[0:60000,0])
# query_dir = list(query[0:60000,1])
# negC = np.load(neg_control+steps+".npy",allow_pickle=True)
# negC_data = np.stack(negC[30000:,0])
# plotWindow = 100


# class CosDist():

#     def __init__(self):

#         self.device = torch.device('cuda')
#         self.dist = CosineSimilarity()

#     def __call__(self,query,ref):

#         query = torch.tensor(query).half().to(self.device)
#         ref = torch.tensor(ref).half().to(self.device)
#         correlation = self.dist(query,ref)
#         #correlation = correlation.cpu().numpy().astype('float64')
#         correlation = 1 - correlation
#         return correlation


try:
    query_corr = np.load(SavedDir+query_npy+'_AnomalyScore.npy',allow_pickle=True)
    negC_corr = np.load(SavedDir+neg_control+'_AnomalyScore.npy',allow_pickle=True)

except:
    anomaly = blockwiseAnomaly(queryEmb,queryEmb)
    with Pool(8) as p:
        querybasal = p.map(anomaly.basal,range(len(queryEmb)**2))
    querybasal = torch.stack(querybasal)
    querybasal = querybasal.view(len(queryEmb),len(queryEmb),-1)
    print(querybasal.shape)
    querybasal = torch.amin(querybasal,dim=1).flatten()
    #np.save(SavedDir+trainID+'_vs_'+trainID+'.npy',RefWarped)

    anomaly = blockwiseAnomaly(negEmb,negEmb)
    with Pool(8) as p:
        negbasal = p.map(anomaly.basal,range(len(negEmb)**2))
    negbasal = torch.stack(negbasal)
    negbasal = negbasal.view(len(negEmb),len(negEmb),-1)
    print(negbasal.shape)
    negbasal = torch.amin(negbasal,dim=1).flatten()

    anomaly = blockwiseAnomaly(queryEmb,refEmb)
    with Pool(8) as p:
        queryscore = p.map(anomaly.basal,range(len(queryEmb)*len(refEmb)))
    queryscore = torch.stack(queryscore)
    queryscore = queryscore.view(len(queryEmb),len(refEmb),-1)
    print(queryscore.shape)
    queryscore = torch.amin(queryscore,dim=1).flatten()

    anomaly = blockwiseAnomaly(negEmb,refEmb)
    with Pool(8) as p:
        negscore = p.map(anomaly.basal,range(len(negEmb)*len(refEmb)))
    negscore = torch.stack(negscore)
    negscore = negscore.view(len(negEmb),len(refEmb),-1)
    print(negscore.shape)
    negscore = torch.amin(negscore,dim=1).flatten()

    # with torch.no_grad():

    #     query_self = torch.amin(cosdist(query_data,query_data) + 2 * maskMetric(size=len(query_data)),axis = 0)
    #     negC_self = torch.amin(cosdist(negC_data,negC_data) + 2 * maskMetric(size=len(negC_data)),axis = 0)
    #     #baseline = np.min(cosdist(refer_data,refer_data) + 2 * maskMetric(size=video_len*refer_num),axis = 0)
    #     query_corr = torch.amin(cosdist(refer_data,query_data),axis=0) - query_self
    #     negC_corr = torch.amin(cosdist(refer_data,negC_data),axis=0) - negC_self

    query_corr = (queryscore-querybasal).cpu().numpy().astype('float64')
    negC_corr = (negscore-negbasal).cpu().numpy().astype('float64')
    np.save(SavedDir+query_npy+'_AnomalyScore.npy',query_corr)
    np.save(SavedDir+neg_control+'_AnomalyScore.npy',negC_corr)

query_corr[np.where(query_corr<0)] = 0

t1=[]
t2=[]
plot1=[]
plot2=[]
standard = np.max(negC_corr)
for i,m in enumerate(query_corr):
    if m > standard:
        t1.append(i)
        plot1.append(m)
    else:
        t2.append(i)
        plot2.append(m)

fig, ax = plt.subplots()
ax.scatter(t1,plot1,s=1,color = "#FF0000")
#ax.fill_between(t1,np.min(query_corr),np.max(query_corr), color="#FF9999",alpha=0.1,label="CNMaR-attp anomalous range") 
ax.scatter(t2,plot2,s=1,color = "#0000FF")
ax.fill_between(np.arange(len(query_corr)),0,standard, color="#9999FF",alpha=0.3) 
plt.savefig(SavedDir+'anomaly.pdf',format='pdf')
plt.show()

f = open(log,'a')
localtime = time.asctime( time.localtime(time.time()) )
f.write(str(localtime)+'\n')
f.write("Query="+query_npy+'\n')
f.write("Ref="+ref_npy+'\n')
f.write("NegControl="+neg_control+'\n')
f.close()


fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
output_movie = cv2.VideoWriter(SavedDir+query_npy+'.mp4', fourcc, 30, (500, 500))
f = open(log,'a')
peak = []
AnomalyScore = []
start = -1

for i, item in enumerate(query_corr):
    if item > standard and start == -1:
        start = i
    if item <= standard and start != -1:
        p = (i+start-1)//2
        peak.append(p)
        if p >= plotWindow//2:
            AnomalyScore.append(query_corr[p-plotWindow//2:p+plotWindow//2])
        start = -1
    if item > standard:
        f.write(str(query_dir[i])+'\n')
        img = cv2.imread(query_dir[i])
        img = cv2.resize(img,(500,500))
        output_movie.write(img)
f.close()
#cv2.destroyAllWindows()  

t = np.arange(plotWindow)/30
mean = np.mean(AnomalyScore,axis=0)
std = stats.sem(AnomalyScore)
plt.plot(t, mean,color="#FF0000",alpha=1.0)
plt.fill_between(t,mean-std,mean+std, color="#FF9999",alpha=0.5) 
plt.show()
