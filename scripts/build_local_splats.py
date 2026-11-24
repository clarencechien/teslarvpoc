"""Build local RGB-D Gaussian clouds. ±6 frames are reserved for evaluation."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from room_geometry import read_poses

STATIONS = (9637, 877, 14017, 17521)
OFFSETS = (-24, -12, 0, 12, 24)
FOCAL = 525.0
STRIDE = 2

def capture(root, i):
    color = cv2.imread(str(root/'image'/f'{i:06d}.jpg'))
    depth = cv2.imread(str(root/'depth'/f'{i:06d}.png'), cv2.IMREAD_UNCHANGED)
    if color is None or depth is None:
        raise FileNotFoundError(f'Missing RGB-D frame {i}')
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB), depth.astype('f4')*.001

def unproject(depth, stride):
    y,x = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[::stride,::stride]
    return np.stack(((x-319.5)*z/FOCAL,(y-239.5)*z/FOCAL,z),axis=-1).reshape(-1,3)

def transform(xyz, matrix):
    return xyz @ matrix[:3,:3].T + matrix[:3,3]

def project(xyz):
    z = xyz[:,2].clip(.001)
    x = np.rint(319.5+FOCAL*xyz[:,0]/z).astype(int)
    y = np.rint(239.5+FOCAL*xyz[:,1]/z).astype(int)
    valid = (xyz[:,2]>.15)&(x>=0)&(x<640)&(y>=0)&(y<480)
    return x.clip(0,639),y.clip(0,479),valid

def build(root, out, poses, anchor):
    frame_ids = [anchor+d for d in OFFSETS]
    inputs = [capture(root,i) for i in frame_ids]
    transforms = [np.linalg.inv(poses[anchor-1]) @ poses[i-1] for i in frame_ids]
    rgb,measured = inputs[OFFSETS.index(0)]
    # Complete missing depths using other captures before borrowing a nearby
    # range. Keep their radiance from the anchor so windows and mirrors stay sharp.
    depth = measured.copy()
    neighboring = []
    for offset,(_,d),matrix in zip(OFFSETS,inputs,transforms):
        if offset==0: continue
        xyz = transform(unproject(d,1)[d.ravel()>.15],matrix)
        x,y,visible = project(xyz)
        raster = np.full(depth.size,np.inf,'f4')
        np.minimum.at(raster, y[visible]*640+x[visible],xyz[visible,2])
        neighboring.append(raster.reshape(depth.shape))
    # Opposite sides of an occlusion can project to the same unknown pixel.
    # The nearest available observed sample accounts for foreground coverage.
    registered = np.min(neighboring, axis=0)
    use = (depth==0)&np.isfinite(registered)
    depth[use] = registered[use]
    nearest = distance_transform_edt(depth==0,return_distances=False,return_indices=True)
    depth = depth[tuple(nearest)]
    xyz = unproject(depth,STRIDE)
    anchor_colors = rgb[::STRIDE,::STRIDE].reshape(-1,3).astype('f4')
    # Robust weighted least-squares fit of the constant (diffuse) color.
    # Downweight a registration error or large view-dependent radiance change.
    numerator = 3*anchor_colors; denominator = np.full((len(xyz),1),3.,'f4')
    for offset,(other,d),matrix in zip(OFFSETS,inputs,transforms):
        if offset==0: continue
        sample = transform(xyz,np.linalg.inv(matrix))
        x,y,inside = project(sample)
        diff = np.abs(d[y,x]-sample[:,2])
        colors = other[y,x].astype('f4')
        color_error = np.mean(np.abs(colors-anchor_colors),axis=1)
        weight = (inside*np.exp(-(diff/.025)**2-(color_error/18)**2))[:,None]
        numerator += weight*colors; denominator += weight
    points = [xyz]; colors = [numerator/denominator]
    # Use neighboring photographs behind silhouettes and at the edge of the
    # anchor frustum. Deduplicate those surfels in a 4 mm spatial grid.
    for offset,(other,d),matrix in zip(OFFSETS,inputs,transforms):
        if offset==0: continue
        stride=STRIDE
        local = unproject(d,stride)
        sample = transform(local,matrix)
        x,y,inside = project(sample)
        missing = (~inside)|(sample[:,2]>depth[y,x]+.05)
        keep = (local[:,2]>.2)&(sample[:,2]>.2)&missing
        points.append(sample[keep]);colors.append(other[::stride,::stride].reshape(-1,3)[keep])
    extra=np.concatenate(points[1:]);extra_colors=np.concatenate(colors[1:])
    _,indices=np.unique(np.floor(extra/.004).astype(int),axis=0,return_index=True)
    centers=np.concatenate((xyz,extra[indices]))
    color=np.vstack((colors[0],extra_colors[indices]))
    # Monotone radius gives each splat a roughly two-pixel footprint at its
    # source distance. A thinner third axis follows the capture image plane.
    order=np.argsort(-centers[:,2],kind='stable')
    centers=centers[order]
    packed=np.zeros(len(centers),dtype=[('xyz','<i2',(3,)),('sigma','<u2'),('rgb','u1',(3,)),('alpha','u1')])
    packed['xyz']=np.rint(centers*[1000,-1000,-1000]).clip(-32767,32767).astype('<i2')
    packed['sigma']=np.rint(1e6*centers[:,2]*STRIDE*.82/FOCAL).clip(1,65535)
    packed['rgb']=np.rint(color[order]).clip(0,255)
    packed['alpha']=250
    (out/f'station-{anchor}.splatbin').write_bytes(packed.tobytes())
    print(anchor,'splats',len(packed),'depth completed',round(float((measured==0).mean()),3))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    poses=read_poses(args.dataset)
    for anchor in STATIONS:
        build(args.dataset,args.output,poses,anchor)
