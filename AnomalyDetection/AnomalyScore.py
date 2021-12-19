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
home = os.path.dirname(__file__)+'/'
ref_npy = home+"ppk23"
query_npy = home+"TRH-KI"
neg_control = home+"CS"
steps="" 
log = home+"log.log"

ref = np.load(ref_npy+steps+".npy",allow_pickle=True)
refer_data = np.stack(ref[0:60000,0])
query = np.load(query_npy+steps+".npy",allow_pickle=True)
query_data = np.stack(query[0:60000,0])
query_dir = list(query[0:60000,1])
negC = np.load(neg_control+steps+".npy",allow_pickle=True)
negC_data = np.stack(negC[:60000,0])
plotWindow = 100


class CosDist():

    def __init__(self):

        self.device = torch.device('cuda')
        self.dist = CosineSimilarity()

    def __call__(self,query,ref):

        query = torch.tensor(query).half().to(self.device)
        ref = torch.tensor(ref).half().to(self.device)
        correlation = self.dist(query,ref)
        #correlation = correlation.cpu().numpy().astype('float64')
        correlation = 1 - correlation
        return correlation

def maskMetric(width = 101,size=30000): # this return a POSITIVE diagl matrix!!!
    device = torch.device('cuda')
    assert (width+1) % 2 == 0, "Width has to be odd"
    x = np.ones((size,size))
    x = x-np.triu(x,k=(width+1)//2)-np.tril(x,k=-(width+1)//2)
    return torch.tensor(x).half().to(device)

try:
    query_corr = np.load(query_npy+'_AnomalyScore.npy',allow_pickle=True)
    negC_corr = np.load(neg_control+'_AnomalyScore.npy',allow_pickle=True)

except:
    cosdist = CosDist()
    with torch.no_grad():

        query_self = torch.amin(cosdist(query_data,query_data) + 2 * maskMetric(size=len(query_data)),axis = 0)
        negC_self = torch.amin(cosdist(negC_data,negC_data) + 2 * maskMetric(size=len(negC_data)),axis = 0)
        #baseline = np.min(cosdist(refer_data,refer_data) + 2 * maskMetric(size=video_len*refer_num),axis = 0)
        query_corr = torch.amin(cosdist(refer_data,query_data),axis=0) - query_self
        negC_corr = torch.amin(cosdist(refer_data,negC_data),axis=0) - negC_self

    query_corr = query_corr.cpu().numpy().astype('float64')
    negC_corr = negC_corr.cpu().numpy().astype('float64')
    np.save(query_npy+'_AnomalyScore.npy',query_corr)
    np.save(neg_control+'_AnomalyScore.npy',negC_corr)

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
plt.show()

f = open(log,'a')
localtime = time.asctime( time.localtime(time.time()) )
f.write(str(localtime)+'\n')
f.write("Query="+query_npy+'\n')
f.write("Ref="+ref_npy+'\n')
f.write("NegControl="+neg_control+'\n')
f.close()


fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
output_movie = cv2.VideoWriter(query_npy+'.mp4', fourcc, 6, (500, 500))
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
print('x')
