"""Recover the photographed stone beneath the shoes at landing stop 2 (front camera frame 199)."""

import argparse
import json
from pathlib import Path
import tempfile
import zipfile

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from build_christ_church import camera_rays, project, read_photo
from build_church_photowalk import people_visibility
from church_rail_repair import camera_list

# Crop coordinates in the full-resolution registered photograph. Keep a border of untouched stone.
FRAME = 199
LEFT, RIGHT, TOP, BOTTOM = 580, 1008, 462, 730
WITNESS = 1183


def floor_plane(pose, rays, distance, visible):
    world = pose[:3, 3] + rays * distance[:, None]
    A = np.column_stack((world[:, :2], np.ones(len(rays))))
    height = world[:, 2]
    if visible.sum() < 500:
        raise ValueError("Insufficient visible floor samples")
    center = np.median(height[visible])
    fit = visible & (abs(height - center) < 0.10)
    for _ in range(3):
        if fit.sum() < 500:
            raise ValueError("Insufficient visible landing surface")
        plane = np.linalg.lstsq(A[fit], height[fit], rcond=None)[0]
        fit &= abs(A @ plane - height) < 0.035
    normal = np.array([-plane[0], -plane[1], 1.0])
    distances = (plane[2] - pose[:3, 3] @ normal) / (rays @ normal)
    return distances, plane, float(np.median(abs(A[fit] @ plane - height[fit])))


def donors(data, cameras, origin, world, raw, ring, target, net, shape):
    found = []
    yy, xx = np.indices(shape)
    validation = ((xx // 16 + yy // 16) % 3) == 0
    with zipfile.ZipFile(data / "images.zip") as photos, zipfile.ZipFile(
        data / "depths_euc.zip"
    ) as depths:
        available = set(depths.namelist())
        for cam, frames in enumerate(cameras):
            for i, frame in enumerate(frames):
                ident = 1000 * cam + i
                center = np.asarray(frame["transform_matrix"])[:3, 3]
                baseline = np.linalg.norm(center - origin)
                path = f"depths_euc/cam{cam}/" + Path(frame["file_path"]).stem + ".png"
                if ident == FRAME or baseline > 9 or path not in available:
                    continue
                u, v, z = project(world, frame)
                u, v = u.reshape(shape).astype("f4"), v.reshape(shape).astype("f4")
                inside = (z.reshape(shape) > 0.3) & (u > 3) & (v > 3) & (u < 1435) & (v < 1075)
                if inside[target].mean() < 0.55:
                    continue
                im = read_photo(photos, frame)
                visibility = people_visibility(im, net, (270, 360))
                clear = cv2.remap(visibility, u / 4, v / 4, cv2.INTER_NEAREST) >= 0.92
                measured = (
                    cv2.imdecode(np.frombuffer(depths.read(path), "u1"), cv2.IMREAD_UNCHANGED).astype(
                        "f4"
                    )
                    / 256
                )
                if not np.any(measured):
                    continue
                gap, near = distance_transform_edt(measured == 0, return_indices=True)
                distance = cv2.remap(measured[near[0], near[1]], u, v, cv2.INTER_NEAREST)
                pixel_gap = cv2.remap(gap.astype("f4"), u, v, cv2.INTER_NEAREST)
                predicted = np.linalg.norm(world - center, axis=1).reshape(shape)
                supported = (
                    inside
                    & clear
                    & (pixel_gap < 13)
                    & (abs(distance - predicted) < 0.12 + 0.015 * predicted)
                )
                train, test = supported & ring & ~validation, supported & ring & validation
                coverage = supported[target].mean()
                if coverage < 0.08 or supported[ring].mean() < 0.10 or min(train.sum(), test.sum()) < 35:
                    continue
                rgb = cv2.remap(im, u, v, cv2.INTER_LINEAR).astype("f4")
                rgb += np.median(raw[train] - rgb[train], axis=0)
                rgb = rgb.clip(0, 255)
                score = float(np.median(abs(raw[test] - rgb[test]).mean(1)) + 10 * (1 - coverage))
                found.append(
                    dict(id=ident, pixels=rgb, support=supported, baseline=float(baseline), score=score)
                )
    return sorted(found, key=lambda item: item["score"])


def row_median(im, valid):
    # Both sides of the occlusion see the same tread under the target camera's light.
    h = len(im)
    trend = np.full((h, 3), np.nan)
    for row in range(h):
        band = slice(max(0, row - 5), row + 6)
        values = im[band][valid[band]]
        if len(values) > 30:
            trend[row] = np.median(values, axis=0)
    for c in range(3):
        use = np.isfinite(trend[:, c])
        if not use.any():
            raise ValueError("No visible reference for tread illumination")
        trend[:, c] = np.interp(np.arange(h), np.flatnonzero(use), trend[use, c])
    return cv2.GaussianBlur(trend[:, None].astype("f4"), (1, 0), sigmaX=2, sigmaY=4)


def repair_landing(data, site, detector, dest):
    cv2.setNumThreads(2)
    cameras = camera_list(data)
    net = cv2.dnn.readNet(str(detector))
    frame = cameras[0][FRAME]
    source = cv2.imread(str(site / "photo-199.webp"))
    previous = cv2.imread(str(site / "walk-light/199.webp"))
    packed = cv2.imread(str(site / "walk-light/199.png"))
    old_depth = (packed[:, :, 2].astype("f4") * 256 + packed[:, :, 1]) * 0.001
    moving = people_visibility(source, net, old_depth.shape) < 0.92
    stable = ~cv2.resize(
        moving.astype("u1"), source.shape[1::-1], interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    sample = stable[::8, ::8] & (source[::8, ::8].min(2) > 18) & (source[::8, ::8].max(2) < 225)
    gain = np.median((previous[::8, ::8][sample] + 0.5) / (source[::8, ::8][sample] + 0.5), axis=0)
    yy, xx = np.mgrid[TOP:BOTTOM, LEFT:RIGHT]
    raw = np.clip(source[yy, xx] * gain, 0, 255).astype("f4")
    old = previous[yy, xx].astype("f4")
    box = moving[yy // 4, xx // 4].astype("u1")
    mask = cv2.dilate(box, np.ones((43, 43), "u1")) != 0
    ring = (cv2.dilate(box, np.ones((51, 51), "u1")) == 0) & (yy < 704)
    target = (xx > 690) & (xx < 891) & (yy > 508) & (yy < 700)
    pose = np.asarray(frame["transform_matrix"])
    rays = camera_rays(frame, xx, yy) @ pose[:3, :3].T
    measured = old_depth[yy // 4, xx // 4].ravel()
    depth, plane, residual = floor_plane(pose, rays, measured, (ring.ravel() & (measured > 0.4)))
    world = pose[:3, 3] + rays * depth[:, None]
    options = donors(data, cameras, pose[:3, 3], world, raw, ring, target, net, yy.shape)
    witness = next((item for item in options if item["id"] == WITNESS), None)
    if witness is None:
        raise ValueError("Reserved camera does not observe this repair")
    options = [item for item in options if item["id"] != WITNESS]
    if len(options) < 2:
        raise ValueError("Two background observations are required")
    near = min((c for c in options if c["score"] < options[0]["score"] + 3), key=lambda c: c["baseline"])
    other = next(item for item in options if item is not near)
    A, B = near["pixels"], other["pixels"].copy()
    both = (near["support"] & other["support"]).astype("f4")
    support = near["support"] | other["support"]
    border = ~mask
    illumination = row_median(raw, border)
    B += row_median(raw - B, border & other["support"])
    B = illumination + np.clip(B - illumination, -42, 37)
    # Match low spatial frequencies in overlap before blending observed tile detail.
    num = cv2.GaussianBlur((A - B) * both[:, :, None], (0, 0), 19)
    den = cv2.GaussianBlur(both, (0, 0), 19)
    B += np.clip(num / np.maximum(0.03, den[:, :, None]), -30, 30)
    gap = distance_transform_edt(near["support"])
    weight = cv2.GaussianBlur(np.clip(gap / 23, 0, 1).astype("f4"), (0, 0), 4)
    weight = np.minimum(weight, gap / 5).clip(0, 1)[:, :, None]
    photo = A * weight + B * (1 - weight)
    low = cv2.GaussianBlur(photo, (0, 0), 13)
    texture = illumination + photo - low + 0.18 * (low - illumination)
    alpha = cv2.GaussianBlur(mask.astype("f4"), (0, 0), 7)
    alpha *= np.clip((yy - TOP) / 40, 0, 1) * np.clip((712 - yy) / 20, 0, 1)
    # Smooth short visibility gaps. Never extend the blend far into an unobserved area.
    alpha *= np.clip(1 - distance_transform_edt(~support) / 12, 0, 1)
    corrected = np.clip(old + alpha[:, :, None] * (texture - old), 0, 255).astype("u1")
    previous[yy, xx] = corrected
    # Coarse grid centers are the same rays used by the offline range exporter.
    centers = (xx % 4 == 2) & (yy % 4 == 2) & (alpha > 0.65) & (both > 0)
    rows, cols = yy[centers] // 4, xx[centers] // 4
    mm = np.rint(depth.reshape(yy.shape)[centers] * 1000).astype("u2")
    original_range = old_depth[rows, cols]
    replace = (original_range == 0) | (abs(original_range - mm * 0.001) > 0.10)
    packed[rows[replace], cols[replace], 1] = mm[replace] % 256
    packed[rows[replace], cols[replace], 2] = mm[replace] // 256
    packed[rows, cols, 0] = np.maximum(packed[rows, cols, 0], 235)
    # Evaluate tile texture independently of the other camera's exposure and reflection.
    check = witness["support"] & (alpha > 0.8) & (yy >= 550) & (yy < 696)

    def detail(im):
        gray = cv2.cvtColor(im.astype("f4"), cv2.COLOR_BGR2GRAY)
        return gray - cv2.GaussianBlur(gray, (0, 0), 13)

    if check.sum() < 200:
        raise ValueError("Insufficient withheld floor observations")
    ref = detail(witness["pixels"])

    def error(im):
        return float(np.mean(abs(detail(im)[check] - ref[check])))

    before, after = error(old), error(corrected)
    if after >= before:
        raise ValueError("Repaired texture did not improve on the reserved floor image")
    report = dict(
        view=FRAME,
        donors=[near["id"], other["id"]],
        witness=WITNESS,
        plane=plane.tolist(),
        visibleHeightMedianErrorMetres=residual,
        witnessPixels=int(check.sum()),
        beforeDetailMAE=before,
        afterDetailMAE=after,
        depthCellsChanged=int(replace.sum()),
    )
    if not cv2.imwrite(str(dest / "199.webp"), previous, [cv2.IMWRITE_WEBP_QUALITY, 101]):
        raise OSError("Cannot write the repaired photograph")
    if not cv2.imwrite(str(dest / "199.png"), packed):
        raise OSError("Cannot write the corrected range")
    (dest / "landing-step.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "site", "detector", "out"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output directory")
    with tempfile.TemporaryDirectory(dir=args.out.parent) as folder:
        staged = Path(folder) / "floor"
        staged.mkdir()
        print(repair_landing(args.data, args.site, args.detector, staged))
        staged.rename(args.out)
