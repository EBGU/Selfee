import scipy.io as sio
import os
import cv2
import numpy as np
well2fly = {
    'well1': 0,
    'well5': 8,
    'well7': 12
}
def flytracker_feat(mat,ids):
    return mat['feat'][0][0][2][ids]

def JAABA_feat(src,ids):
    feat = []
    for mat in os.listdir(src):
        mat = sio.loadmat(os.path.join(src,mat))
        feat.append(mat['data'][0][ids])
    feat = np.concatenate(feat,axis=0)
    feat = feat.transpose(1,0)
    return feat

def dist_feat(mat,ids):
    track = mat['trk'][0][0][1][ids:ids+2] #track*frame*pos
    center = track[:,:,0:2]
    cos_theta = np.cos(track[:,:,2])*track[:,:,3]/2
    sin_theta = np.sin(track[:,:,2])*track[:,:,3]/2
    vec = np.stack([cos_theta,-sin_theta],axis=-1)
    head = center+vec
    tail = center-vec
    l_wing = track[:,:,9:11]
    r_wing = track[:,:,11:13]
    coord = np.stack([head,tail,center,l_wing,r_wing],axis=2)
    coord = coord.transpose(1,0,2,3)
    F,T,N,D = coord.shape
    #coord = coord.reshape(F,T*N,D)
    dist = coord.reshape(F,T*N,1,D) - coord.reshape(F,1,T*N,D)
    dist = np.linalg.norm(dist,axis=-1)
    msk = np.tri(T*N) == 0
    dist = dist[:,msk]    
    #.reshape(F,-1)
    return dist

def dist_feat_legs(mat,ids):
    track = mat['trk'][0][0][1][ids:ids+2] #track*frame*pos
    center = track[:,:,0:2]
    cos_theta = np.cos(track[:,:,2])*track[:,:,3]/2
    sin_theta = np.sin(track[:,:,2])*track[:,:,3]/2
    vec = np.stack([cos_theta,-sin_theta],axis=-1)
    head = center+vec
    tail = center-vec
    l_wing = track[:,:,9:11]
    r_wing = track[:,:,11:13]
    leg1 = track[:,:,17:19]
    leg2 = track[:,:,19:21]
    leg3 = track[:,:,21:23]
    leg4 = track[:,:,23:25]
    leg5 = track[:,:,25:27]
    leg6 = track[:,:,27:29]
    coord = np.stack([head,tail,center,l_wing,r_wing,leg1,leg2,leg3,leg4,leg5,leg6],axis=2)
    coord = coord.transpose(1,0,2,3)
    F,T,N,D = coord.shape
    #coord = coord.reshape(F,T*N,D)
    dist = coord.reshape(F,T*N,1,D) - coord.reshape(F,1,T*N,D)
    dist = np.linalg.norm(dist,axis=-1)
    msk = np.tri(T*N) == 0
    dist = dist[:,msk]    
    #.reshape(F,-1)
    return dist

src = './tracking_result/042815_assay4-track.mat'
tgt = './dist_feat_legs/'
feat = sio.loadmat(src)
for x in well2fly.keys():
    f = dist_feat_legs(feat,well2fly[x])
    np.save(tgt+x+'.npy',f)

src = './tracking_result/042815_assay4-track.mat'
tgt = './dist_feat/'
feat = sio.loadmat(src)
for x in well2fly.keys():
    f = dist_feat(feat,well2fly[x])
    np.save(tgt+x+'.npy',f)


src = './tracking_result/JAABA'
tgt = './JAABA_feat/'

for x in well2fly.keys():
    f = JAABA_feat(src,well2fly[x])
    np.save(tgt+x+'.npy',f)

src = './tracking_result/042815_assay4-feat.mat'
tgt = './flytracker_feat/'
feat = sio.loadmat(src)
for x in well2fly.keys():
    f = flytracker_feat(feat,well2fly[x])
    np.save(tgt+x+'.npy',f)
