import numpy as np
from sklearn.decomposition import PCA
import os
import time
import pickle as pickle
import pyhsmm
from pyhsmm.util.text import progprint_xrange
from pyhsmm.util.stats import whiten, cov
import autoregressive.models as ARmodel
import autoregressive.distributions as ARdist
import matplotlib.pyplot as plt
from sklearn import manifold
from matplotlib.colors import ListedColormap
from multiprocessing import Pool
import cv2
from PIL import Image
from matplotlib import cm
from scipy import stats
import time
current_path = os.path.dirname(__file__).split('/')[:-1]+['rawData']
lalamove = np.vectorize(lambda x: '/'+os.path.join(*(current_path+x.split('/')[-2:])))



trainID = 'w1118'
inferID = 'Ir76b'
home = os.path.dirname(__file__)+'/'
TrainDir = home+trainID
InferDir = home+inferID
SavedDir = 'SavedModels'
pcaMatrix = '/pcaWTMat.pkl'



#set for HDP prior, the last four don't matter
epochs = 1000
Nmax = 40 #modes for HMM
kappa = 1e7
modelName = '/ARHMM_'+trainID+inferID+'_epochs'+str(epochs)+'_Nmax'+str(Nmax)+'_kappa'+str(kappa)+'.pkl'
alpha_a_0=1
alpha_b_0=1/100
gamma_a_0=1
gamma_b_0=1/100


#################
###perform PCA###
#################

try:
    pca = pickle.load(open(SavedDir+pcaMatrix,'rb'))
    pcaTrain = np.load(SavedDir+'/pcaTrain.npy',allow_pickle=True)
    addrTrain = np.load(SavedDir+'/filelistTrain.npy',allow_pickle=True)
    dim_pca = pcaTrain.shape[-1]
    print(pcaTrain.shape)
except:
    pca = PCA(n_components=0.95, svd_solver = 'full')
    fileList = os.listdir(TrainDir)
    embs = []
    for f in fileList:
        embs.append(np.load(os.path.join(TrainDir,f),allow_pickle=True))
    embs = np.concatenate(embs)
    embs = np.stack(embs[:,0])
    addrTrain = []
    pcaTrain = [] 
    pca.fit(embs)
    for f in fileList:
        data = np.load(os.path.join(TrainDir,f),allow_pickle=True)
        addrTrain.append(np.stack(data[:,1]))
        pcaTrain.append(pca.transform(np.stack(data[:,0])))
    pickle.dump(pca, open(SavedDir+pcaMatrix,"wb"))
    pcaTrain = np.asarray(pcaTrain)
    np.save(SavedDir+'/pcaTrain.npy',pcaTrain)
    addrTrain = np.asarray(addrTrain)
    np.save(SavedDir+'/filelistTrain.npy',addrTrain)
    dim_pca = pcaTrain.shape[-1]
    print(pcaTrain.shape)

try:
    pcaInfer = np.load(SavedDir+'/pcaInfer.npy',allow_pickle=True)
    addrInfer = np.load(SavedDir+'/filelistInfer.npy',allow_pickle=True)
    print(pcaInfer.shape)
except:
    fileList = os.listdir(InferDir)
    pcaInfer = []
    addrInfer = []
    for f in fileList:
        data = np.load(os.path.join(InferDir,f),allow_pickle=True)
        addrInfer.append(np.stack(data[:,1]))
        pcaInfer.append(pca.transform(np.stack(data[:,0])))
    pcaInfer = np.asarray(pcaInfer)
    np.save(SavedDir+'/pcaInfer.npy',pcaInfer)
    addrInfer = np.asarray(addrInfer)
    np.save(SavedDir+'/filelistInfer.npy',addrInfer)
    print(pcaInfer.shape)

addrTrain = lalamove(addrTrain)
addrInfer = lalamove(addrInfer)
#################
### fit ARHMM ###
#################

#set for autoregresion initialize, don't matter
affine = False
nlags = 3
nu_0 = dim_pca + 1 #initialize with dim_pca + 1
S_0 = np.eye(dim_pca)
M_0 = np.hstack((np.eye(dim_pca), np.zeros((dim_pca, dim_pca*(nlags-1)+affine))))
a = 25
b = 25

# construct AR-HMM model with HDP prior and ARD prior
model = ARmodel.ARWeakLimitStickyHDPHMMSeparateTrans(
        alpha_a_0=alpha_a_0, alpha_b_0=alpha_b_0, gamma_a_0=gamma_a_0, gamma_b_0=gamma_b_0, kappa=kappa, 
        init_state_distn='uniform',
        obs_distns=[
            ARdist.ARDAutoRegression( 
                nu_0=nu_0, 
                S_0=S_0,
                M_0=M_0,
                a = a,
                b = b,
                affine=affine)
            for state in range(Nmax)],
        )
try:
    model = pickle.load(open(SavedDir+modelName,'rb'))
except:
    for trainData in pcaTrain:
        model.add_data(trainData, group_id = trainID)
    for inferData in pcaInfer:
        model.add_data(inferData,group_id = inferID)
    for itr in progprint_xrange(epochs):
        model.resample_model()
    pickle.dump(model, open(SavedDir+modelName,"wb"))
    print('model fit!')

#################
###count usage###
#################

trainLabel = []
usageTrain = []
for data in pcaTrain:
    l = model.predict(data,1,group_id = trainID)[1]
    trainLabel.append(l)
    usageTrain.append(np.bincount(l,minlength=Nmax)/len(l))
trainLabel = np.asarray(trainLabel).flatten()
usageTrain = np.stack(usageTrain)

inferLabel = []
usageInfer = []
for data in pcaInfer:
    l = model.predict(data,1,group_id = inferID)[1]
    inferLabel.append(l)
    usageInfer.append(np.bincount(l,minlength=Nmax)/len(l))
inferLabel = np.asarray(inferLabel).flatten()
usageInfer = np.stack(usageInfer)


total_usage = np.concatenate([usageTrain,usageInfer])
pca_usage = PCA(n_components=3, svd_solver = 'full')
pca_usage.fit(total_usage)
trainPcaPlot = pca_usage.transform(usageTrain)
inferPcaPlot = pca_usage.transform(usageInfer)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(trainPcaPlot[:,0],trainPcaPlot[:,1],trainPcaPlot[:,2],c='b')
ax.scatter3D(inferPcaPlot[:,0],inferPcaPlot[:,1],inferPcaPlot[:,2],c='r')
plt.show()

x =list(range(Nmax))
total_width, n = 0.8, 2
width = total_width / n
 
plt.bar(x, np.mean(usageTrain,axis=0), width=width, yerr = stats.sem(usageTrain), label=trainID,fc = 'b')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, np.mean(usageInfer,axis=0), width=width,yerr = stats.sem(usageInfer),label=inferID,tick_label=range(Nmax),fc = 'r')
plt.legend()
plt.show()

pValue =[]
for i in range(Nmax):
    try:
        sig = stats.mannwhitneyu(usageTrain[:,i],usageInfer[:,i],alternative='two-sided')
        pValue.append(sig[1])
    except:
        pValue.append(1.0)
pValue = np.asarray(pValue)
pRank = stats.rankdata(pValue, method='min')
qValue = pValue * len(pValue)/pRank  # do Benjamini and Hochberg correction to p value

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
try:
    MaxP = max(pRank[np.where(qValue<0.05)])
    SigIndex = np.where(pRank<=MaxP)[0]
    print(SigIndex)
    for i in SigIndex:
        print(i,np.mean(usageTrain[:,i]),np.mean(usageInfer[:,i]),qValue[i])
        print(usageTrain[:,i])
        print(usageInfer[:,i])
        f = open(SavedDir+"/Log_"+str(i)+".log",'a')
        f1 = open(SavedDir+"/uasgeLog.log",'a')
        localtime = time.asctime( time.localtime(time.time()) )
        f1.write(str(i)+'\t')
        f1.write(str(usageTrain[:,i])+'\t')
        f1.write(str(usageInfer[:,i])+'\n')
        list1 = list(addrTrain.flatten()[np.where(trainLabel==i)])
        list2 = list(addrInfer.flatten()[np.where(inferLabel==i)])
        output_movie = cv2.VideoWriter(SavedDir+"/Log_"+str(i)+'.mp4', fourcc, 30, (224, 224))
        for row in list1+list2:
            f.write(str(row)+'\n')
        f.close()
        f1.close()
        
        for row in list1+list2:    
            img = cv2.imread(row)
            img = cv2.resize(img,(224,224))
            output_movie.write(img)
            #cv2.imshow("image", img)
            #if cv2.waitKey(10) & 0xFF == ord('q'):
            #    break
        cv2.destroyAllWindows()    

        '''
        for row in list1+list2:    
            img = cv2.imread(row)
            cv2.imshow("image", img)
            if cv2.waitKey(40) & 0xFF == ord('q'):
                break
        '''
    cv2.destroyAllWindows()    
    print(pValue)
    print(qValue)
except:
    print('no significance!')
    print(pValue)
    print(qValue)
   
