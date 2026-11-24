"""Reconstruct a continuous interior surface and bake all calibrated RGB-D views into one atlas."""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.spatial import cKDTree
import xatlas

from build_room import Glb
from room_geometry import geometry

LOW = np.array([-2.29, -0.025, -2.41])
HIGH = np.array([3.35, 3.72, 2.31])
EVALUATION = {i + d for i in (9637, 877, 14017, 17521) for d in (-6, 6)}


def write_raw(file, *arrays):
    with open(file, "wb") as fp:
        for array in arrays:
            np.ascontiguousarray(array).tofile(fp)


def reconstruct(dataset, work, cache=None):
    if cache:
        src = np.load(cache)
        p, t, axes, poses = (src[k] for k in ("p", "t", "a", "poses"))
    else:
        p, t, axes, poses = geometry(dataset)
    xyz = p @ axes
    hist, edges = np.histogram(xyz[:, 1], bins=360)
    origin = np.median(xyz, axis=0)
    origin[1] = (edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2
    xyz -= origin
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(xyz), o3d.utility.Vector3iVector(t)
    )
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    good = np.all((xyz >= LOW - 0.04) & (xyz <= HIGH + 0.04), axis=1) & (
        np.linalg.norm(normals, axis=1) > 0.9
    )
    measured = xyz[good]
    nmeasured = normals[good]
    pts = [measured]
    ns = [nmeasured]
    tree = cKDTree(measured)
    # Supply inward-oriented boundaries only in gaps. Most unobserved material
    # is on the ceiling and above the vertical extent of the hand-held capture.
    for axis in range(3):
        u, v = [i for i in range(3) if i != axis]
        uv = np.meshgrid(
            np.arange(LOW[u], HIGH[u] + 0.012, 0.033),
            np.arange(LOW[v], HIGH[v] + 0.012, 0.033),
        )
        for side, height in [(1, LOW[axis]), (-1, HIGH[axis])]:
            patch = np.empty((uv[0].size, 3))
            patch[:, u] = uv[0].ravel()
            patch[:, v] = uv[1].ravel()
            patch[:, axis] = height
            gap = tree.query(patch)[0] > 0.075
            patch = patch[gap]
            n = np.zeros_like(patch)
            n[:, axis] = side
            pts.append(patch)
            ns.append(n)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.vstack(pts))
    cloud.normals = o3d.utility.Vector3dVector(np.vstack(ns))
    result, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=9, scale=1.045, linear_fit=True, n_threads=4
    )
    labels, _, areas = result.cluster_connected_triangles()
    result.remove_triangles_by_mask(np.asarray(areas)[np.asarray(labels)] < 0.006)
    result.remove_unreferenced_vertices()
    result = result.simplify_quadric_decimation(115000)
    result.remove_degenerate_triangles()
    result.remove_unreferenced_vertices()
    p = np.asarray(result.vertices).astype("f4")
    t = np.asarray(result.triangles).astype("u4")
    print("Meshed", len(p), len(t), "bounds", p.min(0), p.max(0), flush=True)
    np.savez(work / "surface.npz", p=p, t=t, axes=axes, origin=origin, poses=poses)
    return p, t, axes, origin, poses


def unwrap(work, p, t):
    atlas = xatlas.Atlas()
    atlas.add_mesh(p, t)
    chart = xatlas.ChartOptions()
    chart.max_iterations = 1
    chart.max_chart_area = 1.5
    pack = xatlas.PackOptions()
    pack.resolution = 3072
    pack.padding = 4
    atlas.generate(chart, pack)
    if atlas.atlas_count != 1:
        raise RuntimeError("Expected one packed atlas")
    ids, tri, uv = atlas[0]
    print(
        "Texture charts",
        atlas.chart_count,
        "occupancy",
        atlas.get_utilization(0),
        flush=True,
    )
    np.savez(work / "uv.npz", ids=ids, tri=tri, uv=uv)
    return ids, tri, uv


def walk_grid(scene):
    step = 0.065
    x = np.arange(LOW[0], HIGH[0], step)
    z = np.arange(LOW[2], HIGH[2], step)
    xx, zz = np.meshgrid(x, z)

    # A ray must reach the floor. Check body clearance in four horizontal
    # directions as well, so a chair back or tabletop doesn't become a shortcut.
    def cast(height, direction):
        start = np.stack((xx, np.full_like(xx, height), zz), axis=-1)
        vec = np.broadcast_to(direction, start.shape)
        return scene.cast_rays(
            o3d.core.Tensor(np.concatenate((start, vec), axis=-1).astype("f4"))
        )["t_hit"].numpy()

    floor = 1.65 - cast(1.65, [0, -1, 0])
    ok = (floor > -0.18) & (floor < 0.24)
    for height in (0.58, 1.16):
        for direction in ([1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]):
            ok &= cast(height, direction) > 0.13
    ok = ndimage.binary_erosion(ok, iterations=1)
    groups, total = ndimage.label(ok)
    if not total:
        raise ValueError("No walkable cells")
    ok = groups == np.bincount(groups[groups > 0]).argmax()
    return dict(
        step=step,
        origin=[float(x[0]), float(z[0])],
        rows=["".join("1" if v else "0" for v in row) for row in ok],
    )


def camera_data(dataset, work, poses, axes, origin, scene):
    frames = [
        i
        for i in range(1, len(poses) + 1)
        if (i % 73 in (1, 37)) and i not in EVALUATION
    ]
    h, w = 240, 320
    v, u = np.mgrid[:h, :w]
    dirs = np.stack(
        ((2 * u + 0.5 - 319.5) / 525, (2 * v + 0.5 - 239.5) / 525, np.ones_like(v)),
        axis=2,
    ).astype("f4")
    temporary = work / "views.partial"
    with open(temporary, "wb") as fp:
        np.array([len(frames)], "<i4").tofile(fp)
        for i in frames:
            pose = poses[i - 1]
            C = pose[:3, 3] @ axes - origin
            R = axes.T @ pose[:3, :3]
            rays = np.empty((h, w, 6), "f4")
            rays[:, :, :3] = C
            rays[:, :, 3:] = dirs @ R.T
            rendered = scene.cast_rays(o3d.core.Tensor(rays), nthreads=4)[
                "t_hit"
            ].numpy()
            rgb = cv2.imread(str(dataset / "image" / f"{i:06}.jpg"))
            d = cv2.imread(str(dataset / "depth" / f"{i:06}.png"), -1)
            if (
                rgb is None
                or rgb.shape != (480, 640, 3)
                or d is None
                or d.shape != (480, 640)
            ):
                raise ValueError(f"Missing or invalid RGB-D frame {i}")
            for arr in (
                np.r_[C, R.ravel()].astype("<f4"),
                rgb,
                d.astype("<u2"),
                rendered.astype("<f4"),
            ):
                arr.tofile(fp)
    temporary.replace(work / "views.bin")
    return len(frames)


def upper_wall_finish(xyz, known, colors):
    # Estimate the plaster from photographed upper walls. At the bed wall,
    # trace the bright curtain columns at a measured height so missing pixels
    # continue vertically instead of stretching toward the nearest bright patch.
    points = xyz[known]
    samples = colors[known]
    near_wall = (
        np.minimum(abs(points - LOW), abs(points - HIGH))[:, [0, 2]].min(axis=1) < 0.17
    )
    b, g, r = samples.T.astype(float)
    plaster = (
        near_wall
        & (points[:, 1] > 1.9)
        & (r > 1.22 * g)
        & (r > 1.25 * b)
        & (r > 37)
        & (r < 150)
    )
    tone = np.median(samples[plaster], axis=0)
    if not np.isfinite(tone).all():
        raise ValueError("No upper-wall colour samples")
    width = 512

    def column(x):
        return np.clip(
            ((x - LOW[0]) / (HIGH[0] - LOW[0]) * (width - 1)).astype(int), 0, width - 1
        )

    curtains = (
        (abs(points[:, 2] - HIGH[2]) < 0.42)
        & (points[:, 1] > 1.37)
        & (points[:, 1] < 1.94)
    )
    cols = column(points[curtains, 0])
    counts = np.bincount(cols, minlength=width)
    den = ndimage.gaussian_filter1d(counts.astype(float), 1.8)
    curtain = np.column_stack(
        [
            ndimage.gaussian_filter1d(
                np.bincount(cols, samples[curtains, k], minlength=width), 1.8
            )
            / np.maximum(den, 0.01)
            for k in range(3)
        ]
    )

    def estimate(points, dist, nearest):
        h = points[:, 1]
        wall = np.minimum(abs(points - LOW), abs(points - HIGH))[:, [0, 2]].min(axis=1)
        # Blend in only where photographs become distant. The measured atlas
        # itself is never overwritten.
        weight = (
            np.clip((dist - 0.025) / 0.24, 0, 1)
            * np.clip((h - 1.78) / 0.32, 0, 1)
            * np.clip((0.52 - wall) / 0.18, 0, 1)
        )
        material = np.broadcast_to(tone, nearest.shape).copy()
        c = curtain[column(points[:, 0])]
        bright = np.clip((c.mean(axis=1) - 100) / 65, 0, 1)
        drape = (
            bright
            * np.clip((2.83 - h) / 0.065, 0, 1)
            * np.clip((0.52 - abs(points[:, 2] - HIGH[2])) / 0.10, 0, 1)
        )
        material += drape[:, None] * (c - material)
        return nearest + (material - nearest) * weight[:, None]

    return estimate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("site", type=Path)
    ap.add_argument("work", type=Path)
    ap.add_argument("--geometry-cache", type=Path)
    ap.add_argument(
        "--reuse-bake",
        action="store_true",
        help="reuse a completed bake with identical input geometry, UVs and frames",
    )
    args = ap.parse_args()
    work = args.work.resolve()
    args.site.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    if (work / "surface.npz").exists():
        s = np.load(work / "surface.npz")
        p, t, axes, origin, poses = (
            s[k] for k in ("p", "t", "axes", "origin", "poses")
        )
    else:
        p, t, axes, origin, poses = reconstruct(args.dataset, work, args.geometry_cache)
    if (work / "uv.npz").exists():
        s = np.load(work / "uv.npz")
        ids, tri, uv = (s[k] for k in ("ids", "tri", "uv"))
    else:
        ids, tri, uv = unwrap(work, p, t)
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(p), o3d.core.Tensor(t))
    nav = walk_grid(scene)
    if not (work / "views.bin").exists():
        camera_data(args.dataset, work, poses, axes, origin, scene)
    write_raw(
        work / "geometry.bin",
        np.array([len(ids), len(tri)], "<u4"),
        p[ids].astype("<f4"),
        uv.astype("<f4"),
        tri.astype("<u4"),
    )
    binary = work / "bake"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O3",
            "-fopenmp",
            Path(__file__).with_name("bake_full_room.cpp"),
            "-o",
            binary,
        ],
        check=True,
    )
    if not args.reuse_bake:
        subprocess.run([binary, work], check=True)
    side = 3072
    colors = np.fromfile(work / "atlas.bgr", "u1").reshape(side, side, 3)
    valid = np.fromfile(work / "mask.bin", "u1").reshape(side, side)
    # Complete unobserved texels in world space, keeping adjacent atlas
    # islands independent. Above the captures the ceiling receives a neutral tint.
    xyz = np.memmap(work / "world.bin", dtype="f4", mode="r", shape=(side * side, 3))
    known = np.flatnonzero(valid.ravel() == 2)[::5]
    tree = cKDTree(xyz[known])
    missing = np.flatnonzero(valid.ravel() == 1)
    flat = colors.reshape(-1, 3)
    wall_finish = upper_wall_finish(xyz, known, flat)
    for start in range(0, len(missing), 50000):
        pixels = missing[start : start + 50000]
        dist, neighbors = tree.query(xyz[pixels], k=8, eps=0.15, workers=3)
        weights = 1 / np.maximum(dist, 0.01)
        sample = np.sum(
            flat[known[neighbors]] * weights[:, :, None], axis=1
        ) / weights.sum(axis=1, keepdims=True)
        sample = wall_finish(xyz[pixels], dist[:, 0], sample)
        neutral = np.broadcast_to([182, 190, 197], sample.shape)
        ceiling = (xyz[pixels, 1] > 3.59)[:, None]
        fade = (np.clip((dist[:, 0] - 0.15) / 0.85, 0, 1)[:, None]) * ceiling
        flat[pixels] = np.rint((1 - fade) * sample + fade * neutral)
    # Four-texel gutters avoid dark chart edges under texture filtering.
    nearest = ndimage.distance_transform_edt(
        valid == 0, return_distances=False, return_indices=True
    )
    colors = colors[tuple(nearest)]
    model = Glb()
    model.doc["asset"][
        "generator"
    ] = "teslarvpoc / Poisson surface with registered multi-view texture"
    model.primitive("Bedroom photographic atlas", p[ids], tri, uv, colors)
    model.write(args.site / "bedroom-full.glb")
    (args.site / "walk-grid.json").write_text(json.dumps(nav, separators=(",", ":")))
    print("Ready", flush=True)


if __name__ == "__main__":
    main()
