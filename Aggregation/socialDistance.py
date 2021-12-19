import numpy as np 
import pandas as pd 
from scipy.spatial.distance import cdist 
import os
from scipy import stats
import matplotlib.pyplot as plt

refLen = 16.72*2 #one-side length of the chamber； ref points are selected from left corner to the top to the right corner
bin_num = 20
src =  os.path.dirname(__file__)+'/Cordiantes'
tgt =  os.path.dirname(__file__)+'/saveFiles'
fileList = os.listdir(src)
ctrlNearBin = []
ctrlMedianBin = []
ctrlAverBin = []
ctrlNear = []
ctrlMedian = []
ctrlAver = []
expNearBin = []
expMedianBin = []
expAverBin = []
expNear = []
expMedian = []
expAver = []
for f in fileList:
    pdFrame =pd.read_csv(src+'/'+f,header=0,index_col=0)
    cods = pdFrame.values[3:]
    distMat = cdist(cods,cods)
    ref = cdist(pdFrame.values[0:3],pdFrame.values[0:3])[0,1]+cdist(pdFrame.values[0:3],pdFrame.values[0:3])[2,1]
    nearDist = np.min(distMat+np.eye(len(distMat))*1e7,axis=1)/ref*refLen
    medianDist = np.median(distMat+np.eye(len(distMat))*1e7,axis=1)/ref*refLen
    averageDist = np.mean(distMat,axis=1)/ref*refLen
    distArr = np.stack([nearDist,medianDist,averageDist])
    np.savetxt(tgt+'/'+f+'.txt',distArr)
    np.save(tgt+'/'+f+'.npy',distArr)
    if 'W1118' in f:
        ctrlNearBin.append(np.histogram(nearDist,bins=bin_num,range=(0,np.ceil(refLen/2)),density=False)[0]/len(nearDist))
        ctrlMedianBin.append(np.histogram(medianDist,bins=bin_num,range=(0,np.ceil(refLen/2)),density=False)[0]/len(medianDist))
        ctrlAverBin.append(np.histogram(averageDist,bins=bin_num,range=(0,np.ceil(refLen/2)),density=False)[0]/len(averageDist))
        ctrlNear.append(np.median(nearDist))
        ctrlMedian.append(np.median(medianDist))
        ctrlAver.append(np.median(averageDist))
    else:
        expNearBin.append(np.histogram(nearDist,bins=bin_num,range=(0,np.ceil(refLen/2)),density=False)[0]/len(nearDist))
        expMedianBin.append(np.histogram(medianDist,bins=bin_num,range=(0,np.ceil(refLen/2)),density=False)[0]/len(medianDist))
        expAverBin.append(np.histogram(averageDist,bins=bin_num,range=(0,np.ceil(refLen/2)),density=False)[0]/len(averageDist))
        expNear.append(np.median(nearDist))
        expMedian.append(np.median(medianDist))
        expAver.append(np.median(averageDist))

def plotBin(ctrlBin,expBin):
    ctrlBin=np.array(ctrlBin)
    expBin = np.array(expBin)
    t = np.linspace(0.0, np.ceil(refLen/2), num=bin_num, endpoint=False)+np.ceil(refLen/2)/2/bin_num
    mean = np.mean(ctrlBin,axis=0)
    std = stats.sem(ctrlBin,axis=0) 
    plt.plot(t, mean,color="#0000FF")
    plt.fill_between(t,mean-std,mean+std, color="#9999FF",alpha=0.5) 
    #plt.show()


    mean = np.mean(expBin,axis=0)
    std = stats.sem(expBin,axis=0) 
    plt.plot(t, mean,color="#FF0000")
    plt.fill_between(t,mean-std,mean+std, color="#FF9999",alpha=0.5) 

    plt.show()

    for i in range(bin_num):
        try:
            p = stats.mannwhitneyu(ctrlBin[:,i],expBin[:,i])[1]
        except:
            p = 1
        if p<0.05:
            print(i,p)


print(ctrlNear)
print(expNear)
print(stats.mannwhitneyu(ctrlNear,expNear)[1])
print(ctrlMedian)
print(expMedian)
print(stats.mannwhitneyu(ctrlMedian,expMedian)[1])
print(ctrlAver)
print(expAver)
print(stats.mannwhitneyu(ctrlAver,expAver)[1])
plotBin(ctrlNearBin,expNearBin)
plotBin(ctrlMedianBin,expMedianBin)
plotBin(ctrlAverBin,expAverBin)
