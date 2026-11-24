"""Texture the public-domain Redwood bedroom reconstruction with its registered RGB-D frames."""
import argparse
import json
import struct
from pathlib import Path

import cv2
import fast_simplification
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from room_geometry import geometry

FOCAL = 525
TILE_W, TILE_H, PAD, SIDE = 384, 288, 2, 2048
COLS, ROWS = SIDE//(TILE_W+2*PAD), SIDE//(TILE_H+2*PAD)
PER_PAGE = COLS*ROWS
CAMERAS = [(9637,'床邊'), (877,'書桌'), (14017,'鏡台'), (17521,'床尾地毯')]


def projection(points, pose):
    p = (points-pose[:3,3]) @ pose[:3,:3]
    z = np.maximum(p[:,2],0.001)
    return p[:,0]*FOCAL/z+319.5, p[:,1]*FOCAL/z+239.5, p[:,2]


def matches(p, t, frames, poses, root):
    tris = p[t]
    centers = tris.mean(1)
    n = np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0])
    n /= np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9)
    scores = np.zeros((len(frames),len(t)),np.float16)
    for row,i in enumerate(frames):
        pose=poses[i-1]
        u,v,z=projection(centers,pose)
        inside=(u>4)&(u<635)&(v>4)&(v<475)&(z>.2)
        ids=np.flatnonzero(inside)
        depth=cv2.imread(str(root/'depth'/f'{i:06d}.png'),cv2.IMREAD_UNCHANGED)
        observed=depth[v[ids].round().astype(int),u[ids].round().astype(int)]*.001
        error=np.abs(observed-z[ids])
        keep=error<.055+.015*z[ids]
        ids=ids[keep]
        ray=pose[:3,3]-centers[ids]
        cosine=np.sum(ray*n[ids],axis=1)/np.maximum(np.linalg.norm(ray,axis=1),1e-6)
        quality=np.sqrt(np.maximum(0,cosine)) / (.6+z[ids])
        mid=1-.35*((u[ids]-319.5)/320)**2-.22*((v[ids]-239.5)/240)**2
        scores[row,ids]=quality*mid/(1+12*error[keep])
    # Choose representative images by improvement in surface coverage and projected detail.
    sample=scores[:,::24].astype('f4')
    best=np.zeros(sample.shape[1],np.float32)
    chosen=[]
    for _ in range(min(95,len(frames))):
        gain=np.maximum(0,sample-best).sum(axis=1)
        if gain.max()<=0:break
        row=int(np.argmax(gain))
        chosen.append(row)
        best=np.maximum(best,sample[row])
    selected=scores[chosen]
    assigned=selected.argmax(axis=0).astype('i4')
    # Prefer coherent photo patches across adjacent triangles when a second
    # camera sees the same surface almost as well as the best camera.
    face_ids=np.arange(len(t))
    edges=np.sort(np.stack((t[:,[0,1]],t[:,[1,2]],t[:,[2,0]]),axis=1),axis=2)
    keys=edges[:,:,0].astype('i8')*(int(t.max())+1)+edges[:,:,1]
    order=keys.ravel().argsort()
    shared=np.flatnonzero(np.diff(keys.ravel()[order])==0)
    neighbors=np.repeat(face_ids,3)
    left,right=order[shared],order[shared+1]
    neighbors[left]=right//3;neighbors[right]=left//3
    neighbors=neighbors.reshape(-1,3)
    for _ in range(5):
        for parity in (0,1):
            labels=assigned[neighbors]
            candidates=np.column_stack((assigned,labels))
            agreement=(labels[:,None,:]==candidates[:,:,None]).sum(axis=2)
            quality=selected[candidates,face_ids[:,None]].astype('f4')
            rating=np.where(quality>0,np.log(np.maximum(quality,1e-8))+.43*agreement,-100)
            updated=candidates[face_ids,rating.argmax(axis=1)]
            assigned[parity::2]=updated[parity::2]
    assigned[selected.max(axis=0)<=0]=-1
    # Robust exposure offsets from corresponding surface samples across cameras.
    logs=np.full((len(chosen),len(centers[::11])),np.nan)
    for k,row in enumerate(chosen):
        i=frames[row];ids=np.flatnonzero(scores[row,::11]>.15)
        u,v,_=projection(centers[::11][ids],poses[i-1])
        image=cv2.imread(str(root/'image'/f'{i:06d}.jpg'),cv2.IMREAD_GRAYSCALE)
        values=image[v.round().astype(int),u.round().astype(int)].astype(float)
        available=(values>9)&(values<242)
        logs[k,ids[available]]=np.log(values[available])
    available=np.sum(np.isfinite(logs),axis=0)>1
    logs=logs[:,available];gains=np.zeros(len(chosen))
    for _ in range(5):
        shared=np.nanmedian(logs+gains[:,None],axis=0)
        for k in range(len(gains)):
            overlap=np.isfinite(logs[k])
            if overlap.sum()>8:gains[k]=np.median(shared[overlap]-logs[k,overlap])
        gains-=np.median(gains)
    return [frames[i] for i in chosen],assigned,np.exp(np.clip(gains,-.48,.48))


class Glb:
    def __init__(self):
        self.data=bytearray()
        self.doc=dict(asset=dict(version='2.0',generator='teslarvpoc / Redwood RGB-D photo projection',
                          copyright='Public-domain Redwood Indoor LiDAR-RGBD Scan Dataset, Park et al., ICCV 2017'),
            extensionsUsed=['KHR_materials_unlit'],scene=0,scenes=[dict(nodes=[0])],
            nodes=[dict(mesh=0,name='Redwood Bedroom')],meshes=[dict(primitives=[])],
            accessors=[],bufferViews=[],images=[],textures=[],materials=[])

    def view(self,data,target=None):
        self.data.extend(bytes(-len(self.data)%4))
        v=dict(buffer=0,byteOffset=len(self.data),byteLength=len(data))
        if target:v['target']=target
        self.doc['bufferViews'].append(v)
        self.data.extend(data)
        return len(self.doc['bufferViews'])-1

    def acc(self,arr,pos=False):
        components={'float32':5126,'uint32':5125,'uint8':5121}
        scalar=arr.ndim==1
        view=self.view(arr.tobytes(),34963 if scalar else 34962)
        a=dict(bufferView=view,componentType=components[str(arr.dtype)],count=len(arr),type='SCALAR' if scalar else 'VEC'+str(arr.shape[1]))
        if pos:a.update(min=arr.min(0).tolist(),max=arr.max(0).tolist())
        self.doc['accessors'].append(a)
        return len(self.doc['accessors'])-1

    def primitive(self,name,positions,faces,uv=None,atlas=None,colors=None,*,jpeg_quality=89):
        attr={'POSITION':self.acc(positions.astype('<f4'),True)}
        pbr=dict(metallicFactor=0,roughnessFactor=1)
        if uv is not None:
            attr['TEXCOORD_0']=self.acc(uv.astype('<f4'))
            ok,jpg=cv2.imencode('.jpg',atlas,[cv2.IMWRITE_JPEG_QUALITY,jpeg_quality])
            if not ok:raise RuntimeError('JPEG encoding failed')
            image=len(self.doc['images'])
            self.doc['images'].append(dict(mimeType='image/jpeg',bufferView=self.view(jpg.tobytes())))
            self.doc['textures'].append(dict(source=image))
            pbr['baseColorTexture']=dict(index=image)
        if colors is not None:
            # glTF vertex colors are linear. Source pixels are sRGB.
            val=colors[:,::-1].astype('f4')/255
            attr['COLOR_0']=self.acc(np.where(val<=.04045,val/12.92,((val+.055)/1.055)**2.4).astype('<f4'))
        mat=len(self.doc['materials'])
        self.doc['materials'].append(dict(name=name,pbrMetallicRoughness=pbr,extensions={'KHR_materials_unlit':{}}))
        self.doc['meshes'][0]['primitives'].append(dict(attributes=attr,indices=self.acc(faces.ravel().astype('<u4')),material=mat))

    def write(self,path):
        self.data.extend(bytes(-len(self.data)%4))
        self.doc['buffers']=[dict(byteLength=len(self.data))]
        info=json.dumps(self.doc,separators=(',',':')).encode()
        info+=b' '*(-len(info)%4)
        chunks=struct.pack('<I4s',len(info),b'JSON')+info+struct.pack('<I4s',len(self.data),b'BIN\0')+self.data
        path.write_bytes(struct.pack('<4sII',b'glTF',2,12+len(chunks))+chunks)


def build(root,output,cache=None):
    if cache and cache.exists():
        pack=np.load(cache)
        p,t,axes,poses=[pack[k] for k in ('p','t','a','poses')]
    else:
        p,t,axes,poses=geometry(root)
    # Fit the coordinate origin on the dense scan before simplification changes
    # the sampling density near walls and furniture.
    aligned=p@axes
    hist,edges=np.histogram(aligned[:,1],bins=360)
    center=np.median(aligned,axis=0)
    center[1]=(edges[hist.argmax()]+edges[hist.argmax()+1])/2
    # Retain detail at furniture edges while simplifying large planar patches.
    p,t=fast_simplification.simplify(p,t,target_count=min(87000,len(t)),agg=4)
    frames=[i for i in range(1,len(poses)+1,73) if (root/'image'/f'{i:06d}.jpg').exists() and (root/'depth'/f'{i:06d}.png').exists()]
    if not frames:raise ValueError('Extract matching image and depth frames first')
    included,assignment,gains=matches(p,t,frames,poses,root)
    print('source views',len(frames),'atlas views',len(included),'photo coverage',np.mean(assignment>=0),flush=True)
    xyz=p@axes-center
    output.mkdir(parents=True,exist_ok=True)
    glb=Glb()
    seen_pos=[]; seen_col=[]
    for start in range(0,len(included),PER_PAGE):
        atlas=np.zeros((SIDE,SIDE,3),np.uint8)
        points=[]; indices=[]; uvs=[]; offset=0
        for slot,num in enumerate(included[start:start+PER_PAGE]):
            source=cv2.imread(str(root/'image'/f'{num:06d}.jpg'))
            # Lift the dark original exposure using one consistent curve.
            source=cv2.LUT(source,np.rint(np.minimum(1,np.arange(256)*gains[start+slot]/255)**.77*255).astype('u1'))
            left=(slot%COLS)*(TILE_W+2*PAD); top=(slot//COLS)*(TILE_H+2*PAD)
            tile=cv2.resize(source,(TILE_W,TILE_H),interpolation=cv2.INTER_AREA)
            padded=cv2.copyMakeBorder(tile,PAD,PAD,PAD,PAD,cv2.BORDER_REPLICATE)
            h,w=padded.shape[:2];atlas[top:top+h,left:left+w]=padded
            faces=t[assignment==start+slot]
            if not len(faces):continue
            ids,local=np.unique(faces,return_inverse=True)
            u,v,_=projection(p[ids],poses[num-1])
            u=np.clip(u,0,639);v=np.clip(v,0,479)
            uv=np.stack((left+PAD+(u+.5)*TILE_W/640,top+PAD+(v+.5)*TILE_H/480),axis=1)/SIDE
            points.append(xyz[ids]);uvs.append(uv);indices.append(local+offset);offset+=len(ids)
            seen_pos.append(xyz[ids]);seen_col.append(source[v.round().astype(int),u.round().astype(int)])
        glb.primitive('Registered photographs',np.concatenate(points),np.concatenate(indices),np.concatenate(uvs),atlas)
    missing=t[assignment<0]
    if len(missing):
        ids,local=np.unique(missing,return_inverse=True)
        dist,nearest=cKDTree(np.concatenate(seen_pos)).query(xyz[ids])
        fallback=np.concatenate(seen_col)[nearest]
        fallback[dist>.28]=[130,139,144]
        glb.primitive('Estimated vertex colors',xyz[ids],local,colors=fallback)
    # Fit a separate rectangular envelope just behind the observed dominant
    # planes. Fill only its broad color using measurements nearest each plane.
    # Keep that geometry independently selectable in a glTF editor.
    glb.doc['nodes'].append(dict(mesh=1,name='Inferred wall, floor and ceiling envelope'))
    glb.doc['scenes'][0]['nodes'].append(1)
    walls=[]
    grid=22
    units=np.linspace(0,1,grid)
    su,sv=np.meshgrid(units,units)
    x0,x1,y0,y1,z0,z1=-2.35,3.38,-.065,3.78,-2.44,2.33
    plane_defs=[
        ([x0,y0,z0],[0,0,z1-z0],[x1-x0,0,0],1,0,[132,145,155]),
        ([x0,y1,z0],[x1-x0,0,0],[0,0,z1-z0],1,y1,[199,205,212]),
        ([x0,y0,z0],[x1-x0,0,0],[0,y1-y0,0],2,z0,[58,64,88]),
        ([x1,y0,z1],[x0-x1,0,0],[0,y1-y0,0],2,z1,[58,64,88]),
        ([x0,y0,z1],[0,0,z0-z1],[0,y1-y0,0],0,x0,[58,64,88]),
        ([x1,y0,z0],[0,0,z1-z0],[0,y1-y0,0],0,x1,[58,64,88])]
    ids=np.arange(grid*grid).reshape(grid,grid)
    aa,bb,cc,dd=ids[:-1,:-1],ids[:-1,1:],ids[1:,:-1],ids[1:,1:]
    plane_faces=np.stack((aa,bb,cc,bb,dd,cc),axis=-1).reshape(-1,3)
    measured=np.concatenate(seen_pos)
    colors=np.concatenate(seen_col)
    for origin,along,vertical,axis,coord,tint in plane_defs:
        surface=(np.array(origin)+su[:,:,None]*along+sv[:,:,None]*vertical).reshape(-1,3)
        inliers=np.abs(measured[:,axis]-coord)<.16
        # A matte, untextured ceiling identifies the area between the wall tops.
        fill=np.broadcast_to(tint,(len(surface),3)).astype('f4').copy()
        if inliers.sum()>16 and not (axis==1 and coord>0):
            dist,near=cKDTree(measured[inliers]).query(surface,k=16)
            sampled=np.median(colors[inliers][near],axis=1)
            weight=np.exp(-dist[:,0:1]*1.9)
            fill=(1-weight)*fill+weight*sampled
            fill=cv2.GaussianBlur(fill.reshape(grid,grid,3),(5,5),.9).reshape(-1,3)
        glb.primitive('Inferred room boundary',surface,plane_faces,colors=fill.astype('u1'))
        walls.append(glb.doc['meshes'][0]['primitives'].pop())
    glb.doc['meshes'].append(dict(name='Inferred room envelope',primitives=walls))
    glb.write(output/'bedroom.glb')
    stations=[]
    for num,name in CAMERAS:
        pose=poses[num-1]; pos=pose[:3,3]@axes-center; direction=pose[:3,2]@axes
        img=Image.open(root/'image'/f'{num:06d}.jpg')
        img.save(output/f'frame-{num}.jpg',quality=90)
        stations.append(dict(id=num,name=name,position=pos.tolist(),direction=direction.tolist(),photo=f'frame-{num}.jpg'))
    stats=dict(vertices=len(p),triangles=len(t),frames=len(frames),selected=len(included),photoFraction=float(np.mean(assignment>=0)),stations=stations)
    (output/'scene.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
    print('aligned space range',np.percentile(xyz,[1,5,50,95,99],axis=0),'stations',stations,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('dataset',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--geometry-cache',type=Path)
    args=ap.parse_args();build(args.dataset,args.output,args.geometry_cache)
