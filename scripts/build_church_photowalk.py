"""Build multi-view visibility maps and a depth-discontinuity-aware patch from fixed laser returns."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from build_christ_church import project
from church_geometry import survey_samples

W, H = 360, 270
SCALE = 1440 / W


def measured_depth(points, frame):
    u, v, z = project(points, frame)
    inside = (z > 0.25) & (u >= 0) & (u < 1440) & (v >= 0) & (v < 1080)
    distance = np.linalg.norm(points[inside] - np.asarray(frame["transform_matrix"])[:3, 3], axis=1)
    pixel = (v[inside] / SCALE).astype(int) * W + (u[inside] / SCALE).astype(int)
    flat = np.full(W * H, np.inf, "f4")
    np.minimum.at(flat, pixel, distance.astype("f4"))
    raw = flat.reshape(H, W)
    missing = ~np.isfinite(raw)
    gap, nearest = distance_transform_edt(missing, return_indices=True)
    # Local samples support narrow interpolation gaps. Do not bridge across a railing
    # to the distant wall, even when both sides of the hole have neighbours.
    nearest_depth = raw[tuple(nearest)]
    smooth = cv2.medianBlur(nearest_depth, 3)
    isolated = abs(nearest_depth - smooth) > 0.12 + 0.022 * smooth
    depth = np.where(isolated & missing, smooth, nearest_depth)
    depth[gap > 4.5] = 0
    return depth


def people_visibility(image, net, shape):
    # OpenCV Zoo YOLOX uses top-left letterboxing, RGB values in [0,255],
    # COCO class 0 for people, and one output grid for each detection stride.
    rgb = cv2.cvtColor(cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    padded = np.full((640, 640, 3), 114, "u1")
    padded[:480] = rgb
    net.setInput(cv2.dnn.blobFromImage(padded))
    det = net.forward()[0]
    visible = np.ones(shape, "f4")
    seen = 0
    for stride in (8, 16, 32):
        size = 640 // stride
        block = det[seen : seen + size**2]
        seen += size**2
        scores = block[:, 4] * block[:, 5]
        for i in np.flatnonzero((scores > 0.48) & (block[:, 5:].argmax(axis=1) == 0)):
            x, y = (block[i, :2] + [i % size, i // size]) * stride
            w, h = np.exp(block[i, 2:4]) * stride
            x0, x1 = np.clip(
                np.rint((x + np.array([-0.54, 0.54]) * w) * shape[1] / 640), 0, shape[1]
            ).astype(int)
            y0, y1 = np.clip(
                np.rint((y + np.array([-0.53, 0.53]) * h) * shape[0] / 480), 0, shape[0]
            ).astype(int)
            visible[y0:y1, x0:x1] = 0
    return cv2.GaussianBlur(visible, (5, 5), 0.85)


def build(data, site, person_model):
    cv2.setNumThreads(3)
    net = cv2.dnn.readNet(str(person_model))
    frames = sorted(
        json.loads((data / "transforms_colmap_scaled.json").read_text())["frames"],
        key=lambda f: f["file_path"],
    )
    cams = [[f for f in frames if f"cam{i}_" in f["file_path"]] for i in range(3)]
    xyz, _, _ = survey_samples(data, cams[0], 2)
    low = np.array([-8, 3, -8])
    high = np.array([13, 27, 9])
    xyz = xyz[((xyz >= low) & (xyz <= high)).all(1)]
    dest = site / "photo-walk"
    dest.mkdir(exist_ok=True)
    views = json.loads((site / "scene.json").read_text())["views"]
    source_ids = []
    for view in views:
        id = view["id"]
        frame = cams[id // 1000][id % 1000]
        center = np.asarray(frame["transform_matrix"])[:3, 3]
        if not view["detail"] or not 8 < center[1] < 22:
            continue
        source_ids.append(id)
        depth = measured_depth(xyz, frame)
        # The exposure-time mobile LiDAR can see occlusions absent from the static survey.
        # A missing depth in the original visibility map can also mean a hole in the old mesh;
        # keep it available with reduced confidence instead of cutting out that geometry.
        raw = cv2.imread(str(site / f"depth-{id}.png"), cv2.IMREAD_COLOR).astype("f4")
        mobile = cv2.resize(
            (raw[:, :, 2] * 256 + raw[:, :, 1]) * 0.001, (W, H), interpolation=cv2.INTER_NEAREST
        )
        confidence = np.ones((H, W), "f4")
        confidence[mobile == 0] = 0.38
        confidence[(mobile > 0) & (mobile + 0.17 + 0.016 * depth < depth)] = 0.08
        confidence[depth == 0] = 0
        confidence *= people_visibility(cv2.imread(str(site / f"photo-{id}.webp")), net, depth.shape)
        mm = np.rint(np.minimum(depth, 65) * 1000).astype("u2")
        encoded = np.stack([confidence * 255, mm % 256, mm // 256], -1).astype("u1")
        cv2.imwrite(str(dest / f"{id}.png"), encoded)
    (dest / "sources.json").write_text(json.dumps(source_ids))
    print("Photo sources:", len(source_ids), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["data", "site", "person_model"]:
        parser.add_argument(key, type=Path)
    build(**vars(parser.parse_args()))
