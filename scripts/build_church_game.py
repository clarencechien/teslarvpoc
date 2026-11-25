"""Build continuous architectural relief panels for the game staircase from Oxford Spires."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, median_filter

from build_christ_church import UP, project
from build_room import Glb
from church_geometry import survey_samples
from church_steps import FLIGHTS, fit_flight

# Local image axes, range of each image axis, and range along the relief axis (survey metres).
PANELS = {
    "window wall": ((0, 2, 1), (-4.8, 8.8), (-7.2, 6.4), (21.45, 23.58)),
    "west wall": ((1, 2, 0), (7, 23.5), (-7.2, 6.4), (-5.8, -4.30)),
    "east wall": ((1, 2, 0), (7, 23.5), (-7.2, 6.4), (8.38, 10.18)),
    "vault": ((0, 1, 2), (-4.8, 8.8), (7, 22.1), (2.8, 6.4)),
}


def survey_panel(points, colors, axes, ub, vb, db):
    u, v, d = points[:, axes].T
    mask = (u >= ub[0]) & (u <= ub[1]) & (v >= vb[0]) & (v <= vb[1]) & (d >= db[0]) & (d <= db[1])
    u, v, d = u[mask], v[mask], d[mask]
    width, height = [int(np.ceil((b[1] - b[0]) * 44)) for b in (ub, vb)]
    col = np.rint((u - ub[0]) / (ub[1] - ub[0]) * (width - 1)).astype(int)
    row = np.rint((vb[1] - v) / (vb[1] - vb[0]) * (height - 1)).astype(int)
    flat = row * width + col
    count = np.bincount(flat, minlength=width * height).reshape(height, width)
    nearest = distance_transform_edt(count == 0, return_distances=False, return_indices=True)

    def raster(values):
        total = np.bincount(flat, weights=values, minlength=width * height).reshape(height, width)
        return (total / np.maximum(count, 1))[tuple(nearest)].astype("f4")

    depth = cv2.medianBlur(median_filter(raster(d), size=3), 5)
    base = np.stack([raster(ch) for ch in colors[mask, ::-1].T], axis=-1)
    # Isolated scanner colours and unobserved cells would look like white pinholes on stone.
    # Keep the broad lighting of the survey, then add photographic detail where visible.
    filtered = cv2.GaussianBlur(cv2.medianBlur(base.astype("u1"), 7), (0, 0), 2.2)
    luminance = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY).astype("f4")
    base = np.clip(44 + 0.65 * luminance[:, :, None] + np.array([0, 8, 16]), 0, 255)
    return depth, base.astype("f4")


def world_grid(depth, axes, ub, vb):
    h, w = depth.shape
    u, v = np.meshgrid(np.linspace(*ub, w, dtype="f4"), np.linspace(*vb[::-1], h, dtype="f4"))
    local = np.stack((u, v, depth), -1)
    return local[:, :, np.argsort(axes)].copy()


def photo_texture(world, base, sources):
    h, w = world.shape[:2]
    out = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    for start in range(0, h, 50):
        stop = min(start + 50, h)
        points = world[start:stop].reshape(-1, 3)
        rgb = np.zeros((len(points), 3), "f4")
        weights = np.zeros(len(points), "f4")
        for frame, image, depth in sources:
            x, y, z = project(points, frame)
            edge = np.minimum.reduce((x, y, 1439 - x, 1079 - y))
            dist = np.linalg.norm(points - np.array(frame["transform_matrix"])[:3, 3], axis=1)
            observed = depth[np.clip((y / 6).astype(int), 0, 179), np.clip((x / 6).astype(int), 0, 239)]
            # Prefer central, close photographs. Reject people and intervening railings.
            wgt = (
                np.clip(edge / 120, 0, 1)
                * (
                    np.exp(-(((dist - observed) / (0.18 + 0.035 * dist)) ** 2))
                    + 0.16 * ((observed == 0) | (observed >= dist - 0.40))
                )
                * (z > 0.25)
                / (1 + 0.12 * dist) ** 3
            )
            wgt *= np.clip(z / np.maximum(dist, 0.1), 0, 1) ** 4
            if wgt.max() < 0.0005:
                continue
            shape = (stop - start, w)
            sample = cv2.remap(
                image,
                (x.reshape(shape) * 0.5).astype("f4"),
                (y.reshape(shape) * 0.5).astype("f4"),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            rgb += sample.reshape(-1, 3) * wgt[:, None]
            weights += wgt
        prior = 0.003
        out[start:stop] = (rgb + prior * out[start:stop].reshape(-1, 3)).reshape(stop - start, w, 3) / (
            weights.reshape(stop - start, w, 1) + prior
        )
    return np.clip(out, 0, 255).astype("u1")


def build(data, site):
    frames = sorted(
        json.loads((data / "transforms_colmap_scaled.json").read_text())["frames"],
        key=lambda f: f["file_path"],
    )
    cams = [[f for f in frames if f"cam{i}_" in f["file_path"]] for i in range(3)]
    points, colors, _ = survey_samples(data, cams[0], 3)
    mask = (
        (points[:, 0] > -6)
        & (points[:, 0] < 11)
        & (points[:, 1] > 6)
        & (points[:, 1] < 24)
        & (points[:, 2] > -7.4)
        & (points[:, 2] < 6.6)
    )
    points, colors = points[mask], colors[mask]
    # Use the same fitted surfaces as the scan viewer (same sampling and solver).
    fit_points, _, _ = survey_samples(data, cams[0], 11)
    flights = [np.round(fit_flight(fit_points, b).astype("f8"), 4).tolist() for b in FLIGHTS]
    (site / "game-flights.json").write_text(json.dumps(flights, separators=(",", ":")))
    sources = []
    for view in json.loads((site / "scene.json").read_text())["views"]:
        idx = view["id"]
        frame = cams[idx // 1000][idx % 1000]
        y = frame["transform_matrix"][1][3]
        if not view["detail"] or not 8 < y < 22 or idx % 2:
            continue
        color = cv2.imread(str(site / f"photo-{idx}.webp"), cv2.IMREAD_REDUCED_COLOR_2)
        color = cv2.resize(color, (720, 540), interpolation=cv2.INTER_AREA)
        encoded = cv2.imread(str(site / f"depth-{idx}.png")).astype("f4")
        sources.append((frame, color, (encoded[:, :, 2] * 256 + encoded[:, :, 1]) * 0.001))
    glb = Glb()
    glb.doc["asset"].update(
        generator="teslarvpoc / architectural relief",
        copyright="Oxford Spires Dataset, Yifu Tao et al., CC BY-NC-SA 4.0",
    )
    glb.doc["nodes"][0]["name"] = "Christ Church staircase enclosure"
    for name, (axes, ub, vb, db) in PANELS.items():
        print(name, flush=True)
        relief, base = survey_panel(points, colors, axes, ub, vb, db)
        tex_depth = cv2.resize(relief, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        photo = photo_texture(world_grid(tex_depth, axes, ub, vb), base, sources)
        mesh_depth = cv2.resize(relief, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        h, w = mesh_depth.shape
        coords = world_grid(mesh_depth, axes, ub, vb)
        uv = np.stack(np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h)), axis=-1)
        idx = np.arange(h * w).reshape(h, w)[:-1, :-1].ravel()
        tri = np.vstack(
            [
                np.stack((idx, idx + w, idx + 1), axis=1),
                np.stack((idx + 1, idx + w, idx + w + 1), axis=1),
            ]
        )
        glb.primitive(
            name, coords.reshape(-1, 3) @ UP.T, tri, uv.reshape(-1, 2), photo, jpeg_quality=91
        )
        glb.doc["materials"][-1]["doubleSided"] = True
    glb.write(site / "game-enclosure.glb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["data", "site"]:
        parser.add_argument(key, type=Path)
    build(**vars(parser.parse_args()))
