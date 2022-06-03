 
import h5py
import numpy as np
def readh5py(file):
    with h5py.File(file, 'r') as f:
        occupancy_matrix = f['track_occupancy'][:]
        tracks_matrix = f['tracks'][:]#track*2*node*frames
        track_names = f['track_names'][:]
        node_names = f['node_names'][:]
    tracks_matrix = tracks_matrix.transpose(3,0,2,1) #frames*tracks*node*2
    print(occupancy_matrix.shape)
    print(tracks_matrix.shape)
    print(track_names)
    print(node_names)
    return occupancy_matrix,tracks_matrix

def coord2dist(coord):
    F,T,N,D = coord.shape
    #coord = coord.reshape(F,T*N,D)
    dist = coord.reshape(F,T*N,1,D) - coord.reshape(F,1,T*N,D)
    dist = np.linalg.norm(dist,axis=-1) #.reshape(F,-1)
    msk = np.tri(T*N) == 0
    dist = dist[:,msk]
    return dist

def coord2feat(coord):
    axis = coord[:,:,0,:]-coord[:,:,4,:]
    axis = axis/np.linalg.norm(axis,axis=-1,keepdims=True)
    axis_angle = np.arccos((axis[:,0,:]*axis[:,1,:]).sum(axis=-1,keepdims=True))*180/np.pi
    l_wing = coord[:,:,2,:]-coord[:,:,1,:]
    l_wing = l_wing/np.linalg.norm(l_wing,axis=-1,keepdims=True)
    r_wing = coord[:,:,3,:]-coord[:,:,1,:]
    r_wing = r_wing/np.linalg.norm(r_wing,axis=-1,keepdims=True)
    wing_angle = np.arccos((l_wing*r_wing).sum(axis=-1))*180/np.pi
    mh_ft = coord[:,0,4,:]-coord[:,1,0,:] #female tail to male head
    mh_ft = np.linalg.norm(mh_ft,axis=-1,keepdims=True)
    dist = coord[:,0,1,:]-coord[:,1,1,:]
    dist = np.linalg.norm(dist,axis=-1,keepdims=True)
    tocenter = coord[:,:,1,:] - np.array([[[250.,250.]]])
    tocenter = np.linalg.norm(tocenter,axis=-1)
    feat = np.concatenate([axis_angle,wing_angle,mh_ft,dist,tocenter],axis=-1)
    return np.nan_to_num(feat)