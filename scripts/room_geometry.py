import numpy as np
from plyfile import PlyData
def unit(x):return x/np.linalg.norm(x)
def room_axes(xyz, faces, poses):
    # Fit the room axes to planar patches, with the trajectory disambiguating vertical.
    sample=xyz[faces[::120]]
    normal=np.cross(sample[:,1]-sample[:,0],sample[:,2]-sample[:,0])
    area=np.linalg.norm(normal,axis=1)
    normal=normal/np.maximum(area[:,None],1e-10)
    down=unit(poses[:,:3,1].mean(axis=0))
    def dominant(candidates):
        indices=np.flatnonzero(candidates & (area>1e-6))
        rng=np.random.default_rng(119)
        choices=rng.choice(indices,160,replace=False)
        best=max(choices,key=lambda i:area[candidates & (np.abs(normal@normal[i])>.996)].sum())
        mask=candidates & (np.abs(normal@normal[best])>.985)
        return unit(np.sum(normal[mask]*np.sign(normal[mask]@normal[best])[:,None]*area[mask,None],axis=0))
    up=dominant(abs(normal@down)>.91)
    if up@down>0:up=-up
    side=dominant(abs(normal@up)<.13)
    side=unit(side-up*(side@up))
    return np.stack((side,up,np.cross(side,up)),axis=1)

def read_poses(root):
    lines=(root/'bedroom.log').read_text().splitlines()
    return np.array([np.fromstring(l,sep=' ') for i,l in enumerate(lines) if i%5]).reshape(-1,4,4)

def geometry(root, step=.033):
    ply=PlyData.read(root/'bedroom.ply',known_list_len={'face':{'vertex_indices':3}})
    xyz=np.column_stack([ply['vertex'][k] for k in 'xyz']).astype(np.float32)
    faces=ply['face']['vertex_indices']
    poses=read_poses(root)
    axes=room_axes(xyz,faces,poses)
    # Cluster very dense reconstruction vertices before assigning photograph UVs.
    vox=np.floor((xyz-xyz.min(0))/step).astype(np.int32)
    dims=vox.max(0)+1
    keys=(vox[:,0]*dims[1]+vox[:,1])*dims[2]+vox[:,2]
    keys,remap,counts=np.unique(keys,return_inverse=True,return_counts=True)
    pos=np.stack([np.bincount(remap,weights=xyz[:,i])/counts for i in range(3)],axis=1)
    tris=remap[faces].astype(np.int32)
    tris=tris[(tris[:,0]!=tris[:,1])&(tris[:,1]!=tris[:,2])&(tris[:,0]!=tris[:,2])]
    # Merging indices can create coincident duplicate triangles.
    _,first=np.unique(np.sort(tris,axis=1),return_index=True,axis=0)
    tris=tris[np.sort(first)]
    return pos,tris,axes,poses
