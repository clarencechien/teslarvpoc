"""Roll back individual proposed mesh edits that worsen separate photographic observations."""

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from build_christ_church import UP, read_photo
from build_room import Glb
from church_steps import read_primitives
from evaluate_christ_church import glb_scene, render, sample


def control_views(cameras, site):
    used = {v["id"] for v in json.loads((site / "scene.json").read_text())["views"]}
    chosen = []
    for cam, frames in enumerate(cameras):
        # Exclude the original one-in-four atlas photos, every viewer source,
        # the 12 final audit poses and the extra front-camera benchmark.
        reserved = {
            next(i for i in range(a, a + 12) if i % 4 and cam * 1000 + i not in used)
            for a in (29, 87, 175, 203)
        }
        eligible = [
            f
            for i, f in enumerate(frames)
            if i % 4
            and i not in reserved
            and i != 198
            and cam * 1000 + i not in used
            and 1710755131 < float(Path(f["file_path"]).stem) < 1710755444
        ]
        chosen.extend(eligible[::3])
    return chosen


def face_keys(groups):
    # Positions are shipped as float32. Repacking UVs keeps winding and positions.
    keys = []
    for _, _, p, t, _ in groups:
        q = np.ascontiguousarray(p[t].astype("<f4").reshape(-1, 9))
        q[q == 0] = 0  # canonical positive zero for bytewise comparison
        keys.append(q.view("V36").ravel())
    return np.concatenate(keys), np.r_[0, np.cumsum([len(g[3]) for g in groups])]


def surface_ids(rendered, offsets, keys):
    valid, geom, tri = rendered[2], rendered[-2], rendered[-1]
    out = np.full(valid.shape, -1, "i4")
    out[valid] = keys[offsets[geom[valid]] + tri[valid]]
    return out


def photo_error(img, ref):
    # Evaluate local structure as well as colour. A slightly better mean colour
    # can still draw a dark triangular seam across an otherwise smooth wall.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("f4")
    target = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype("f4")

    def average(a):
        return cv2.GaussianBlur(a, (7, 7), 1.25)

    mu, other = average(gray), average(target)
    cov = average(gray * target) - mu * other
    var = np.maximum(0, average(gray * gray) - mu * mu) + np.maximum(
        0, average(target * target) - other * other
    )
    sim = ((2 * mu * other + 6.5025) * (2 * cov + 58.5225)) / (
        (mu * mu + other * other + 6.5025) * (var + 58.5225)
    )
    color = abs(average(img.astype("f4") - ref.astype("f4"))).mean(2)
    return 100 * (1 - sim) + 0.12 * color


def assess(data, views, paths, offsets, keys, changed):
    scenes = [glb_scene(p) for p in paths]
    score = np.zeros(len(changed))
    positive = np.zeros(len(changed), int)
    negative = positive.copy()
    total = np.zeros(len(changed), int)
    with zipfile.ZipFile(data / "images.zip") as photos:
        for frame in views:
            old, new = [
                render(scene, mats, frame, 448, 304, 88, with_hits=True) for scene, mats in scenes
            ]
            ref = sample(read_photo(photos, frame), old[3], old[4])
            gain = np.clip(photo_error(old[0], ref) - photo_error(new[0], ref), -55, 55)
            a, b = [surface_ids(r, o, k) for r, o, k in zip((old, new), offsets, keys)]
            ca = (a >= 0) & changed[a]
            cb = (b >= 0) & changed[b]
            # When one proposal reveals a different proposal, share that pixel;
            # a repainted triangle has the same physical ID in both renders.
            ids = np.r_[a[ca], b[cb & (b != a)]]
            values = np.r_[gain[ca], gain[cb & (b != a)]]
            if not len(ids):
                continue
            ids, ind = np.unique(ids, return_inverse=True)
            count = np.bincount(ind)
            delta = np.bincount(ind, weights=values) / count
            evidence = np.minimum(count, 50)
            score[ids] += delta * evidence
            total[ids] += evidence
            positive[ids] += (delta > 0.7) & (count >= 2)
            negative[ids] += (delta < -0.7) & (count >= 2)
    mean = score / np.maximum(total, 1)
    return (total >= 8) & (mean > 1.1) & (positive >= 2) & (positive >= 3 * negative)


def suppress_new_seams(old, new, doc, images, ak, bk, keep):
    """Reassign an existing photograph only when it repairs a local colour outlier."""
    from scipy.spatial import cKDTree

    def face_colors(groups):
        rgb = []
        pos = []
        norm = []
        textured = []
        for mat, key, p, t, attr in groups:
            xyz = p[t]
            pos.append(xyz.mean(1))
            n = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
            norm.append(n / np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-9))
            vals = attr[t].mean(1)
            if key == "COLOR_0":
                color = 255 * np.where(
                    vals < 0.0031308, 12.92 * vals, 1.055 * np.maximum(vals, 0) ** (1 / 2.4) - 0.055
                )
                rgb.append(color[:, ::-1])
            else:
                texture = doc["materials"][mat]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
                img = cv2.imdecode(np.frombuffer(images[doc["textures"][texture]["source"]], "u1"), 1)
                rgb.append(sample(img, vals[:, 0] * img.shape[1] - 0.5, vals[:, 1] * img.shape[0] - 0.5))
            textured.append(np.full(len(t), key == "TEXCOORD_0"))
        return tuple(np.concatenate(x) for x in (rgb, pos, norm, textured))

    color, xyz, normal, textured = face_colors(old)
    new_color, *_ = face_colors(new)
    lookup = np.full(len(keep), -1, int)
    lookup[bk] = np.arange(len(bk))
    idx = np.flatnonzero(keep[ak] & (lookup[ak] >= 0) & textured)
    if not len(idx):
        return
    distance, neighbors = cKDTree(xyz).query(xyz[idx], k=18, workers=3)
    coplanar = abs(np.einsum("ij,ikj->ik", normal[idx], normal[neighbors])) > 0.9
    usable = (distance > 0.0005) & (distance < 0.45) & coplanar
    for i, mask, near in zip(idx, usable, neighbors):
        if mask.sum() >= 4:
            target = np.median(color[near[mask]], axis=0)
            before = abs(color[i] - target).mean()
            after = abs(new_color[lookup[ak[i]]] - target).mean()
            if before > 12 and after + 5 < before:
                continue
        keep[ak[i]] = False


def verify(data, site, proposal, target, controls):
    old_path = site / proposal.name
    original, _, old = read_primitives(old_path, trim_flights=False)
    doc, images, new = read_primitives(proposal, trim_flights=False)
    mats = len(original["materials"])
    a, ao = face_keys(old)
    b, bo = face_keys(new)
    unique, ids = np.unique(np.concatenate([a, b]), return_inverse=True)
    ak, bk = np.split(ids, [len(a)])
    n = len(unique)
    oldcount = np.bincount(ak, minlength=n)
    newcount = np.bincount(bk, minlength=n)
    # Keep duplicate/coplanar keys untouched. These lack an unambiguous pairing.
    unchanged = np.zeros(n, bool)
    for i, g in enumerate(new):
        if g[0] < mats:
            unchanged[bk[bo[i] : bo[i + 1]]] = True
    changed = ~unchanged & (oldcount <= 1) & (newcount <= 1)
    keep = assess(data, controls, [old_path, proposal], [ao, bo], [ak, bk], changed) & changed
    suppress_new_seams(old, new, doc, images, ak, bk, keep)
    glb = Glb()
    glb.doc = doc
    doc.update(accessors=[], bufferViews=[], buffers=[])
    doc["meshes"][0]["primitives"] = []
    for img, blob in zip(doc["images"], images):
        img["bufferView"] = glb.view(blob)
    for take_new, groups, keys, off in [(False, old, ak, ao), (True, new, bk, bo)]:
        for i, (mat, key, p, t, attr) in enumerate(groups):
            if take_new and mat < mats:
                continue
            accept = keep[keys[off[i] : off[i + 1]]]
            t = t[accept if take_new else ~accept]
            if not len(t):
                continue
            idx, local = np.unique(t, return_inverse=True)
            attrs = {
                "POSITION": glb.acc((p[idx] @ UP.T).astype("f4"), True),
                key: glb.acc(attr[idx].astype("f4")),
            }
            doc["meshes"][0]["primitives"].append(
                dict(attributes=attrs, material=mat, indices=glb.acc(local.ravel().astype("u4")))
            )
    proposed = doc["asset"]["extras"]["surveyCleanup"]
    applied = dict(
        removed=int((keep & (newcount == 0)).sum()),
        added=int((keep & (oldcount == 0)).sum()),
        retextured=int((keep & (oldcount == 1) & (newcount == 1)).sum()),
    )
    report = dict(proposed=proposed, applied=applied, reviewCameras=len(controls))
    doc["asset"]["extras"]["surveyCleanup"] = report
    glb.write(target)
    print(target.name, report, flush=True)
    return report
