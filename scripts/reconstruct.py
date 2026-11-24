"""Build an offline WebGL viewer and an embedded-buffer glTF from a rectified photo pair."""
import base64
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/"
FOCAL, CX, CY, DOFFS, BASELINE_M = 994.978, 311.193, 254.877, 31.086, 0.193001


CHECKSUMS = {'left': 'db18e9c4157617403c3537a6ba355dfeafe9a7eabb6b9b94cb33f6525dd49179', 'right': '5fc913ae870e42a4b662314bc904d1786bcad8e2f0b9b67dba5a229406357797'}


def photo(side):
    name = f"motorcycle_{side}.png"
    with urlopen(SOURCE + name, timeout=45) as response:
        raw = response.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != CHECKSUMS[side]:
        raise ValueError(f"Unexpected contents for {name}: {digest}")
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def estimate(left, right):
    # SGBM returns disparity in sixteenths of a pixel. Pad the search margin
    # on the left so its full 96-pixel width is not discarded at every row.
    margin = 96
    def padded(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.copyMakeBorder(gray, 0, 0, margin, 0, cv2.BORDER_REPLICATE)
    matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=margin, blockSize=5,
        P1=8*25, P2=32*25, uniquenessRatio=10, disp12MaxDiff=1,
        speckleWindowSize=80, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = matcher.compute(padded(left), padded(right))[:, margin:].astype(np.float32) / 16
    # The corresponding right pixel must exist, even when the matcher sees padding.
    valid = (disp >= 0) & (np.arange(left.shape[1]) - disp >= 0)
    return disp, valid


def triangulate(disp, valid, stride=4):
    y, x = np.mgrid[0:disp.shape[0]:stride, 0:disp.shape[1]:stride]
    z = FOCAL * BASELINE_M / (disp[y, x] + DOFFS)
    positions = np.stack(((x-CX)*z/FOCAL, (CY-y)*z/FOCAL, -z), axis=-1)
    uv = np.stack(((x+0.5)/disp.shape[1], (y+0.5)/disp.shape[0]), axis=-1)
    grid = np.arange(x.size).reshape(x.shape)
    a, b, c, d = grid[:-1,:-1], grid[:-1,1:], grid[1:,:-1], grid[1:,1:]
    triangles = np.concatenate((np.stack((a,c,b), -1), np.stack((b,c,d), -1))).reshape(-1,3)
    depths = z.ravel()[triangles]
    # Reject occlusion boundaries instead of stretching foreground texture across gaps.
    keep = valid[y,x].ravel()[triangles].all(axis=1) & (np.ptp(depths,axis=1) < 0.045*depths.min(axis=1))
    used, remap = np.unique(triangles[keep], return_inverse=True)
    return positions.reshape(-1,3)[used], uv.reshape(-1,2)[used], remap.reshape(-1,3)


def data_uri(mime, data):
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def gltf(positions, uv, triangles, left):
    blob = bytearray()
    views, accessors = [], []
    def append(array, dtype, component, shape, target):
        blob.extend(bytes((-len(blob)) % 4))
        raw = np.asarray(array, dtype=dtype).tobytes()
        views.append(dict(buffer=0, byteOffset=len(blob), byteLength=len(raw), target=target))
        blob.extend(raw)
        accessors.append(dict(bufferView=len(views)-1, componentType=component, count=len(array), type=shape))
        return len(accessors)-1
    pos = append(positions, '<f4', 5126, 'VEC3', 34962)
    accessors[pos].update(min=positions.min(axis=0).astype('float32').tolist(), max=positions.max(axis=0).astype('float32').tolist())
    tex = append(uv, '<f4', 5126, 'VEC2', 34962)
    indices = append(triangles.ravel(), '<u2', 5123, 'SCALAR', 34963)
    ok, jpg = cv2.imencode('.jpg', left, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("Failed to encode photo texture")
    return dict(asset=dict(version='2.0', generator='teslarvpoc / OpenCV StereoSGBM',
                          copyright='Source photos: Middlebury Stereo 2014, Scharstein et al.'),
        scene=0, scenes=[dict(nodes=[0])], nodes=[dict(mesh=0, name='Motorcycle')],
        meshes=[dict(primitives=[dict(attributes={'POSITION':pos,'TEXCOORD_0':tex}, indices=indices, material=0)])],
        buffers=[dict(byteLength=len(blob), uri=data_uri('application/octet-stream',blob))],
        bufferViews=views, accessors=accessors, images=[dict(uri=data_uri('image/jpeg',jpg))],
        textures=[dict(source=0)], extensionsUsed=['KHR_materials_unlit'],
        materials=[dict(doubleSided=True, extensions={'KHR_materials_unlit':{}},
          pbrMetallicRoughness=dict(baseColorTexture=dict(index=0), metallicFactor=0, roughnessFactor=1))])


if __name__ == "__main__":
    left, right = photo('left'), photo('right')
    disp, valid = estimate(left, right)
    positions, uv, triangles = triangulate(disp, valid)
    model = gltf(positions, uv, triangles, left)
    template = (ROOT/'src/viewer.html').read_text()
    app = (ROOT/'src/viewer.js').read_text()
    html = template.replace('/* APP */', app).replace('<!-- MODEL -->', json.dumps(model, separators=(',',':')))
    (ROOT/'viewer.html').write_text(html)
    print(f"{len(positions):,} vertices / {len(triangles):,} triangles / {valid.mean():.1%} valid pixels")
