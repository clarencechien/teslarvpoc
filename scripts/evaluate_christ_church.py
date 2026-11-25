"""Compare baked textures and nearby photo projection at twelve unused camera views."""

import argparse
import csv
import json
import struct
import zipfile
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from skimage.metrics import structural_similarity
from build_christ_church import UP, project, read_photo


def glb_scene(path):
    raw = path.read_bytes()
    length = struct.unpack_from("<I", raw, 12)[0]
    doc = json.loads(raw[20 : 20 + length])
    binary = memoryview(raw)[28 + length :]

    def view(i):
        item = doc["bufferViews"][i]
        start = item.get("byteOffset", 0)
        return binary[start : start + item["byteLength"]]

    def attribute(i, dtype, width):
        return np.frombuffer(view(doc["accessors"][i]["bufferView"]), dtype).reshape(-1, width)

    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    materials = {}
    for primitive in doc["meshes"][0]["primitives"]:
        attributes = primitive["attributes"]
        pos = attribute(attributes["POSITION"], "<f4", 3)
        # Work in original Z-up coordinates for camera calibration.
        xyz = np.ascontiguousarray(pos @ UP, dtype="f4")
        faces = attribute(primitive["indices"], "<u4", 3)
        idx = scene.add_triangles(o3d.core.Tensor(xyz), o3d.core.Tensor(faces))
        if "TEXCOORD_0" in attributes:
            uv = attribute(attributes["TEXCOORD_0"], "<f4", 2)
            mat = doc["materials"][primitive["material"]]["pbrMetallicRoughness"]
            image = doc["textures"][mat["baseColorTexture"]["index"]]["source"]
            tex = cv2.imdecode(np.frombuffer(view(doc["images"][image]["bufferView"]), "u1"), 1)
            materials[idx] = (faces, uv, tex)
        else:
            colors = attribute(attributes["COLOR_0"], "<f4", 3)
            materials[idx] = (faces, colors, None)
    return scene, materials


def sample(img, x, y):
    # cv2.remap imposes a 32767-pixel limit per dimension.
    shape = x.shape
    xi, yi = x.ravel(), y.ravel()
    rows = []
    for start in range(0, len(xi), 20000):
        xx = xi[start : start + 20000, None].astype("f4")
        yy = yi[start : start + 20000, None].astype("f4")
        rows.append(cv2.remap(img, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[:, 0])
    return np.concatenate(rows).reshape(shape + (img.shape[2],))


def render(scene, materials, frame):
    width, height = 400, 300
    u, v = np.meshgrid(np.arange(width) + 0.5 - width / 2, np.arange(height) + 0.5 - height / 2)
    focal = height / (2 * np.tan(np.deg2rad(33)))
    local = np.stack([u, -v, -np.full_like(u, focal)], axis=-1)
    local /= np.linalg.norm(local, axis=2, keepdims=True)
    pose = np.array(frame["transform_matrix"], dtype="f4")
    rays = local @ pose[:3, :3].T
    origin = np.broadcast_to(pose[:3, 3], rays.shape)
    cast = scene.cast_rays(
        o3d.core.Tensor(np.concatenate([origin, rays], axis=2).astype("f4")), nthreads=4
    )
    ids, tri, bary, dist = [
        cast[k].numpy() for k in ["geometry_ids", "primitive_ids", "primitive_uvs", "t_hit"]
    ]
    valid = np.isfinite(dist)
    xyz = origin + rays * np.where(valid, dist, 0)[:, :, None]
    base = np.full((height, width, 3), [40, 46, 48], "u1")
    for idx, (faces, attrs, tex) in materials.items():
        mask = ids == idx
        if not mask.any():
            continue
        weights = bary[mask]
        weights = np.column_stack([1 - weights.sum(1), weights])
        coords = (attrs[faces[tri[mask]]] * weights[:, :, None]).sum(1)
        if tex is not None:
            base[mask] = sample(
                tex, coords[:, 0] * tex.shape[1] - 0.5, coords[:, 1] * tex.shape[0] - 0.5
            )
        else:
            srgb = np.where(
                coords < 0.0031308, 12.92 * coords, 1.055 * np.maximum(coords, 0) ** (1 / 2.4) - 0.055
            )
            base[mask] = np.clip(srgb[:, ::-1] * 255, 0, 255).astype("u1")
    # Source pixels for the same undistorted perspective rays.
    u, v, _ = project((origin + rays).reshape(-1, 3), frame)
    return base, xyz, valid, u.reshape(height, width), v.reshape(height, width)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def evaluate(data, site, output):
    scene, materials = glb_scene(site / "christ-church.glb")
    views = json.loads((site / "scene.json").read_text())["views"]
    frames = sorted(
        json.loads((data / "transforms_colmap_scaled.json").read_text())["frames"],
        key=lambda f: f["file_path"],
    )
    cameras = [[f for f in frames if f"cam{i}_" in f["file_path"]] for i in range(3)]
    sources = []
    for view in views:
        cam, i = divmod(view["id"], 1000)
        frame = cameras[cam][i]
        sources.append((view["id"], frame, np.array(frame["transform_matrix"])))
    archive = zipfile.ZipFile(data / "images.zip")
    rows = []
    for cam, camera in enumerate(cameras):
        for anchor in [29, 87, 175, 203]:
            # Exclude the 1-in-4 bake candidates and every on-demand source.
            i = next(
                i
                for i in range(anchor, anchor + 12)
                if i % 4 and all(id != cam * 1000 + i for id, _, _ in sources)
            )
            frame = camera[i]
            pose = np.array(frame["transform_matrix"])
            base, xyz, valid, u, v = render(scene, materials, frame)
            ref = sample(read_photo(archive, frame), u, v)

            def score(source):
                other = source[2]
                return np.sum((pose[:3, 3] - other[:3, 3]) ** 2) + 4.2 * (
                    1 - pose[:3, 2] @ other[:3, 2]
                )

            id, other, other_pose = min(sources, key=score)
            distance = np.linalg.norm(pose[:3, 3] - other_pose[:3, 3])
            uvx, uvy, z = project(xyz.reshape(-1, 3), other)
            x, y, z = [a.reshape(valid.shape) for a in (uvx, uvy, z)]
            photo = cv2.imread(str(site / f"photo-{id}.webp"))
            depth = cv2.imread(str(site / f"depth-{id}.png")).astype("f4")
            depth = (depth[:, :, 2] * 256 + depth[:, :, 1]) * 0.001
            observed = depth[
                np.clip(((y + 0.5) / 6).astype(int), 0, 179),
                np.clip(((x + 0.5) / 6).astype(int), 0, 239),
            ]
            diff = np.abs(observed - np.linalg.norm(xyz - other_pose[:3, 3], axis=2))
            visible = 1 - smoothstep(0.13 + 0.012 * observed, 0.30 + 0.023 * observed, diff)
            edge = np.clip(
                35
                * np.minimum.reduce(
                    [(x + 0.5) / 1440, (1439.5 - x) / 1440, (y + 0.5) / 1080, (1079.5 - y) / 1080]
                ),
                0,
                1,
            )
            weight = visible * edge * valid * (z > 0) * (1 - smoothstep(0.2, 2.8, distance))
            if score((id, other, other_pose)) > 9:
                weight[:] = 0

            # Blend in linear light, matching Three.js sRGB texture decoding.
            def linear(img):
                val = img.astype("f4") / 255
                return np.where(val <= 0.04045, val / 12.92, ((val + 0.055) / 1.055) ** 2.4)

            mix = (
                linear(base) * (1 - weight[:, :, None])
                + linear(sample(photo, x, y)) * weight[:, :, None]
            )
            detail = np.rint(
                np.clip(
                    np.where(mix <= 0.0031308, 12.92 * mix, 1.055 * mix ** (1 / 2.4) - 0.055) * 255,
                    0,
                    255,
                )
            ).astype("u1")
            gray = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            measure = lambda im: round(structural_similarity(gray(ref), gray(im), data_range=255), 4)
            rows.append(
                [
                    cam,
                    i,
                    Path(frame["file_path"]).stem,
                    id,
                    round(distance, 3),
                    round(valid.mean(), 4),
                    measure(base),
                    measure(detail),
                ]
            )
    with output.open("w") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "camera",
                "frame",
                "timestamp",
                "detail_view",
                "distance_m",
                "mesh_coverage",
                "baked_ssim",
                "projected_ssim",
            ]
        )
        writer.writerows(rows)
    print("mean SSIM (baked, detail)", np.mean([r[-2:] for r in rows], axis=0))
    print("improved", sum(r[-1] > r[-2] for r in rows), "/", len(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["data", "site", "csv"]:
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    evaluate(args.data, args.site, args.csv)
