import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from multiprocessing import Pool
import cv2
from PIL import Image
from scipy import stats
import os
current_path = os.path.dirname(__file__).split('/')[:-1]+['rawData']
lalamove = np.vectorize(lambda x: '/'+os.path.join(*(current_path+x.split('/')[-2:])))

labelDic = {
    "others" : 0,
    "social_interest" : 1,
    "mount" : 2,
    "intromission" : 3,
    "ejaculation" : 4,
}


class slideAverage():

    def __init__(self,dataArray,windowSize):
        assert(windowSize % 2 == 1) #window size has to be odd
        self.dataArray = dataArray
        self.windowSize = windowSize

    def __call__(self,index):
        minIndex = max(0,index-self.windowSize//2)
        maxIndex = min(index+self.windowSize//2,len(self.dataArray)-1)
        avg = np.sum(self.dataArray[minIndex:maxIndex+1],axis=0)/(maxIndex-minIndex+1)
        return avg       


class frameAverage():
    def __init__(self,dataArray,windowSize):        
        self.dataArray = dataArray
        self.windowSize = windowSize
    def __call__(self,index):
        maxIndex = min(index+self.windowSize,len(self.dataArray))
        avg = np.mean(self.dataArray[index:maxIndex],axis=0)
        return avg       
    def labelMode(self,index):
        maxIndex = min(index+self.windowSize,len(self.dataArray))
        avg = stats.mode(self.dataArray[index:maxIndex])[0][0]
        return avg

npy = os.path.dirname(__file__)+'/'
steps = "WT_C57DBAF1m6_20210310_OVX" 


slideAver = False
discreteAver = True
windowSize = 3
start = 0
end = -1
dataO = np.load(npy+steps+".npy",allow_pickle=True)

labels =  np.zeros(len(dataO))
with open(npy+steps+'.txt') as f:
    for i, lines in enumerate(f.readlines()):
        x = lines.split(" ")
        while ('' in x):
            x.remove('')
        if x[4] == 'start\n':
            begin = int(x[0])-1
        if x[4] == 'stop\n' or x[4] == 'stop':
            stop = int(x[0])-1
            labels[begin:stop+1] = labelDic[x[2]]


data = np.stack(dataO[start:end,0])
fileList = np.stack(dataO[start:end,1])
fileList = lalamove(fileList)
labels = labels[start:end]
if discreteAver:
    NfileList = fileList[0:len(fileList):windowSize]
else:
    NfileList = fileList


tsne = manifold.TSNE(n_components=2, init='pca',metric='cosine')


if slideAver:

    aver=slideAverage(data,windowSize)
    with Pool(16) as p:
        averaged=np.array(p.map(aver,range(len(data))))
    embedding=tsne.fit_transform(averaged)

elif discreteAver:

    aver=frameAverage(data,windowSize)
    with Pool(16) as p:
        averaged=np.array(p.map(aver,range(0,len(data),windowSize)))
    embedding=tsne.fit_transform(averaged)
    aver=frameAverage(labels,windowSize)
    with Pool(16) as p:
        labels=np.array(p.map(aver.labelMode,range(0,len(labels),windowSize)))
        
else:
    embedding=tsne.fit_transform(data)



map1 = ListedColormap(["crimson", "gold", "forestgreen", "lightseagreen","slateblue"])

fig = plt.figure()
ax = plt.subplot(111)
scatter = ax.scatter(embedding[:, 0], embedding[:, 1],s=1,c=labels,cmap=map1,picker=True, pickradius=3)
cscatter = plt.colorbar(scatter)

def onpick(event):
    N = len(event.ind)
    if not N:
        return True

    dataind = event.ind[0]
    k = np.where(fileList == NfileList[dataind])[0][0]
    for i in range(windowSize):
        current = cv2.imread(fileList[k-i], cv2.IMREAD_GRAYSCALE)
        past = cv2.imread(fileList[k-i-1], cv2.IMREAD_GRAYSCALE)
        if past is None:
            past = current
        future = cv2.imread(fileList[k-i+1], cv2.IMREAD_GRAYSCALE)
        if future is None:
            future = current
    img = cv2.merge([past,current,future])
    img = Image.fromarray(img)
    img.resize((224,224)).show()
    return True
fig.canvas.mpl_connect('pick_event', onpick)
plt.show()
