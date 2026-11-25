"""Build a photo-textured Oxford Spires Christ Church 03 scene from the published archives."""

import argparse
import json
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from church_geometry import reconstruct_survey, survey_colors, base_to_front, walk_clearance
from church_geometry import survey_cloud, static_infill, fill_remaining
from church_mobile import supplement
from build_room import Glb

UP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
CREDIT = "Oxford Spires Dataset, Yifu Tao et al. | CC BY-NC-SA 4.0"
LANDMARKS = {21: "大廳長桌", 50: "高窗與肖像", 175: "樓梯平台", 192: "石階入口", 198: "石階仰視"}


def project(points, frame):
    pose = np.asarray(frame["transform_matrix"], dtype="f4")
    cam = (points - pose[:3, 3]) @ pose[:3, :3]
    x, y, z = cam[:, 0], -cam[:, 1], -cam[:, 2]
    radius = np.hypot(x, y)
    theta = np.arctan2(radius, z)
    t = theta * theta
    k = [frame["k" + str(i)] for i in range(1, 5)]
    scale = theta * (1 + t * (k[0] + t * (k[1] + t * (k[2] + t * k[3])))) / np.maximum(radius, 1e-7)
    return x * scale * frame["fl_x"] + frame["cx"], y * scale * frame["fl_y"] + frame["cy"], z


def camera_rays(frame, u, v):
    K = np.array([[frame["fl_x"], 0, frame["cx"]], [0, frame["fl_y"], frame["cy"]], [0, 0, 1.0]])
    k = np.array([frame["k" + str(i)] for i in range(1, 5)])
    grid = np.stack([u, v], axis=-1).astype("f4")
    xy = cv2.fisheye.undistortPoints(grid.reshape(-1, 1, 2), K, k).reshape(-1, 2)
    rays = np.column_stack([xy[:, 0], -xy[:, 1], -np.ones(len(xy))])
    return (rays / np.linalg.norm(rays, axis=1)[:, None]).astype("f4")


def visibility(scene, frame, step=6):
    u, v = np.meshgrid(np.arange(step / 2, 1440, step), np.arange(step / 2, 1080, step))
    pose = np.asarray(frame["transform_matrix"], dtype="f4")
    ray = camera_rays(frame, u, v) @ pose[:3, :3].T
    rays = np.column_stack([np.broadcast_to(pose[:3, 3], ray.shape), ray]).astype("f4")
    return scene.cast_rays(o3d.core.Tensor(rays), nthreads=4)["t_hit"].numpy().reshape(u.shape)


class MeasuredVisibility:
    """Test line of sight in the mesh and veto contradictions from the moving scanner."""

    def __init__(self, scene, data, front):
        self.scene = scene
        self.front = front
        self.front_times = np.array([float(Path(f["file_path"]).stem) for f in front])
        self.archive = zipfile.ZipFile(data / "undist-clouds.zip")
        self.names = sorted(n for n in self.archive.namelist() if n.endswith(".pcd"))
        self.times = np.array(
            [float(re.search(r"cloud_(\d+)_(\d+)\.pcd", n).expand(r"\1.\2")) for n in self.names]
        )
        self.maps = {}

    def depth(self, frame):
        name = frame["file_path"]
        if name in self.maps:
            return self.maps[name]
        depth = visibility(self.scene, frame)
        time = float(Path(name).stem)
        i = np.abs(self.times - time).argmin()
        j = np.abs(self.front_times - self.times[i]).argmin()
        if abs(self.times[i] - time) <= 0.025 and abs(self.front_times[j] - self.times[i]) <= 0.025:
            raw = self.archive.read(self.names[i]).split(b"DATA binary\n", 1)[1]
            pts = np.frombuffer(raw, "<f4").reshape(-1, 8)[:, :3]
            pose = np.array(self.front[j]["transform_matrix"]) @ base_to_front()
            world = pts @ pose[:3, :3].T + pose[:3, 3]
            x, y, z = project(world, frame)
            valid = (x >= 0) & (x < 1440) & (y >= 0) & (y < 1080) & (z > 0.2)
            measured = np.linalg.norm(world[valid] - np.array(frame["transform_matrix"])[:3, 3], axis=1)
            pix = (y[valid] / 6).astype(int) * 240 + (x[valid] / 6).astype(int)
            mobile = np.full(depth.size, np.inf, "f4")
            np.minimum.at(mobile, pix, measured)
            mobile = mobile.reshape(depth.shape)
            # Require neighbouring beams that see a closer foreground. Expand only
            # the veto by one pixel to cover small timing and calibration differences.
            blocked = np.isfinite(mobile) & (mobile + 0.20 + 0.016 * mobile < depth)
            votes = cv2.boxFilter(blocked.astype("f4"), -1, (3, 3), normalize=False)
            depth[cv2.dilate(((votes >= 2) & blocked).astype("u1"), np.ones((3, 3), "u1")) != 0] = (
                np.inf
            )
        self.maps[name] = depth
        return depth


def read_photo(archive, frame):
    raw = archive.read(frame["file_path"].removeprefix("images/"))
    return cv2.imdecode(np.frombuffer(raw, "u1"), cv2.IMREAD_COLOR)


def assignments(points, faces, frames, visibility_maps):
    tri = points[faces]
    centers = tri.mean(1)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(1e-8, np.linalg.norm(normals, axis=1)[:, None])
    scores = np.zeros((len(frames), len(faces)), "f2")
    for i, frame in enumerate(frames):
        u, v, z = project(centers, frame)
        ids = np.flatnonzero((u > 5) & (u < 1434) & (v > 5) & (v < 1074) & (z > 0.2))
        ray = np.asarray(frame["transform_matrix"])[:3, 3] - centers[ids]
        dist = np.linalg.norm(ray, axis=1)
        depth = visibility_maps.depth(frame)
        observed = depth[(v[ids] / 6).astype(int), (u[ids] / 6).astype(int)]
        valid = np.abs(observed - dist) < (0.10 + 0.016 * dist)
        # A visible centre can belong to a triangle spanning an occlusion or
        # the fisheye image border. Inspect its vertices before assigning UVs.
        vx, vy, vz = project(points, frame)
        in_frame = (vx >= 1) & (vx < 1439) & (vy >= 1) & (vy < 1079) & (vz > 0.12)
        vertex_depth = depth[
            np.clip((vy / 6).astype(int), 0, 179), np.clip((vx / 6).astype(int), 0, 239)
        ]
        vertex_range = np.linalg.norm(points - np.asarray(frame["transform_matrix"])[:3, 3], axis=1)
        consistent = abs(vertex_depth - vertex_range) < (0.16 + 0.023 * vertex_range)
        valid &= in_frame[faces[ids]].all(1) & (consistent[faces[ids]].sum(1) >= 2)
        cosine = (ray * normals[ids]).sum(1) / dist
        border = np.minimum(np.minimum(u[ids], 1439 - u[ids]), np.minimum(v[ids], 1079 - v[ids]))
        score = np.abs(cosine) * np.clip(border / 90, 0.12, 1) / (1 + 0.14 * dist)
        scores[i, ids] = np.where(valid, score, 0)
    # Coherent image patches retain straight woodwork and window edges.
    best = scores.argmax(0)
    index = np.arange(len(faces))
    maximum = scores[best, index].astype("f4")
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=2).reshape(-1, 2)
    keys = edges[:, 0].astype("i8") * len(points) + edges[:, 1]
    order = np.argsort(keys)
    pairs = np.flatnonzero(keys[order[:-1]] == keys[order[1:]])
    a, b = order[pairs], order[pairs + 1]
    neighbors = np.repeat(index, 3)
    neighbors[a], neighbors[b] = b // 3, a // 3
    neighbors = neighbors.reshape(-1, 3)
    labels = best.copy()
    for _ in range(5):
        nearby = labels[neighbors]
        choices = np.column_stack([best, labels, nearby])
        likelihood = scores[choices, index[:, None]] / np.maximum(maximum[:, None], 1e-5)
        support = (choices[:, :, None] == nearby[:, None, :]).sum(2)
        labels = choices[
            index, np.argmax(np.where(likelihood > 0.18, likelihood + 0.32 * support, -1), axis=1)
        ]
    labels[maximum < 0.004] = -1
    return labels


def fallback_colors(points, archive, frames, maps, survey):
    # Use up to five independently visible photo samples. A per-channel median
    # limits contamination from changing highlights and small moving objects.
    best = np.full((len(points), 5), np.inf, "f4")
    colors = np.zeros((len(points), 5, 3), "u1")
    for frame in frames:
        u, v, z = project(points, frame)
        visible = (z > 0.15) & (u >= 1) & (u < 1439) & (v >= 1) & (v < 1079)
        ids = np.flatnonzero(visible)
        if not len(ids):
            continue
        dist = np.linalg.norm(points[ids] - np.asarray(frame["transform_matrix"])[:3, 3], axis=1)
        depth = maps.depth(frame)[(v[ids] / 6).astype(int), (u[ids] / 6).astype(int)]
        where = abs(dist - depth) < (0.10 + 0.012 * dist)
        ids, dist = ids[where], dist[where]
        slot = best[ids].argmax(1)
        closer = dist < best[ids, slot]
        ids, slot, dist = ids[closer], slot[closer], dist[closer]
        if not len(ids):
            continue
        best[ids, slot] = dist
        img = read_photo(archive, frame)
        colors[ids, slot] = img[v[ids].round().astype(int), u[ids].round().astype(int)]
    tree, rgb = survey
    _, near = tree.query(points, workers=4)
    result = rgb[near, ::-1].copy()
    valid = np.isfinite(best)
    seen = valid.any(1)
    result[seen] = np.nanmedian(np.where(valid[seen, :, None], colors[seen], np.nan), axis=1)
    return result


def bake(path, archive, points, faces, frames, labels, tile_width, survey, maps):
    glb = Glb()
    glb.doc["asset"].update(
        generator="Oxford Spires LiDAR and fisheye photo projection", copyright=CREDIT
    )
    glb.doc["nodes"][0]["name"] = "Christ Church, Oxford"
    world = points @ UP.T
    photo_ids = np.unique(labels[labels >= 0])
    w, h, pad = tile_width, tile_width * 3 // 4, 2
    for start in range(0, len(photo_ids), 16):
        atlas = np.zeros((4 * (h + 2 * pad), 4 * (w + 2 * pad), 3), "u1")
        positions, faces_out, uvs, offset = [], [], [], 0
        for slot, i in enumerate(photo_ids[start : start + 16]):
            img = read_photo(archive, frames[i])
            left, top = (slot % 4) * (w + 2 * pad), (slot // 4) * (h + 2 * pad)
            tile = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            atlas[top : top + h + 2 * pad, left : left + w + 2 * pad] = cv2.copyMakeBorder(
                tile, pad, pad, pad, pad, cv2.BORDER_REPLICATE
            )
            ids, local = np.unique(faces[labels == i], return_inverse=True)
            u, v, _ = project(points[ids], frames[i])
            u, v = np.clip(u, 0, 1439), np.clip(v, 0, 1079)
            uv = np.stack(
                [
                    (left + pad + (u + 0.5) * w / 1440) / atlas.shape[1],
                    (top + pad + (v + 0.5) * h / 1080) / atlas.shape[0],
                ],
                axis=1,
            )
            positions.append(world[ids])
            faces_out.append(local.reshape(-1, 3) + offset)
            uvs.append(uv)
            offset += len(ids)
        glb.primitive(
            "Registered photographs",
            np.concatenate(positions),
            np.concatenate(faces_out),
            np.concatenate(uvs),
            atlas,
            jpeg_quality=86,
        )
    missing = faces[labels < 0]
    if len(missing):
        ids, local = np.unique(missing, return_inverse=True)
        color = fallback_colors(points[ids], archive, frames, maps, survey)
        glb.primitive("Observed vertex colours", world[ids], local, colors=color)
    for mat in glb.doc["materials"]:
        mat["doubleSided"] = True
    glb.write(path)


def bake_mobile(out, archive, points, faces, frames, visibility_maps, survey):
    # Keep visibility testing against the full mesh when texturing the lighter version.
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(points), o3d.utility.Vector3iVector(faces)
    )
    mesh = mesh.simplify_quadric_decimation(180000)
    p = np.asarray(mesh.vertices).astype("f4")
    t = np.asarray(mesh.triangles)
    labels = assignments(p, t, frames, visibility_maps)
    bake(out / "christ-church-mobile.glb", archive, p, t, frames, labels, 448, survey, visibility_maps)


def run(data, out, cache):
    from church_steps import refine_stairs

    build_base(data, out, cache)
    refine_stairs(data, out)


def build_base(data, out, cache):
    cache.mkdir(exist_ok=True, parents=True)
    out.mkdir(exist_ok=True, parents=True)
    frames = sorted(
        json.loads((data / "transforms_colmap_scaled.json").read_text())["frames"],
        key=lambda f: f["file_path"],
    )
    cameras = [[f for f in frames if f"cam{i}_" in f["file_path"]] for i in range(3)]
    mesh_path = cache / "survey-tsdf-v2.npz"
    if mesh_path.exists():
        m = np.load(mesh_path)
        points, faces = m["p"], m["t"]
    else:
        points, faces = reconstruct_survey(data, cameras[0])
        np.savez(mesh_path, p=points, t=faces)
    if not len(faces):
        raise ValueError("Empty survey mesh")
    archive = zipfile.ZipFile(data / "images.zip")
    # Omit frames after the camera exposure/white balance changes to deep red.
    selected = [
        f
        for camera in cameras
        for f in camera[::4]
        if 1710755131 < float(Path(f["file_path"]).stem) < 1710755444
    ]
    survey = survey_colors(data, cameras[0])
    infill_file = cache / "static-repaired.npz"
    if infill_file.exists():
        infill = np.load(infill_file)
        sp, sf = infill["p"], infill["t"]
    else:
        samples, _ = survey_cloud(data, cameras[0])
        sp, sf = static_infill(samples, points, faces)
        np.savez(infill_file, p=sp, t=sf)
    points, faces = supplement(data, cameras[0], cache, points, faces, survey[0])
    points, faces = fill_remaining(points, faces, sp, sf)
    mesh = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(points), o3d.core.Tensor(faces.astype("i8")))
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(mesh)
    maps = MeasuredVisibility(scene, data, cameras[0])
    labels = assignments(points, faces, selected, maps)
    bake(out / "christ-church.glb", archive, points, faces, selected, labels, 896, survey, maps)
    bake_mobile(out, archive, points, faces, selected, maps, survey)
    write_views(out, archive, maps, cameras)
    maps.archive.close()
    archive.close()
    stats = dict(
        keyframes=len(frames),
        textureViews=len(set(labels[labels >= 0])),
        triangles=len(faces),
        photoCoverage=round(float((labels >= 0).mean()), 4),
    )
    (out / "stats.json").write_text(json.dumps(stats))
    print(stats, flush=True)


def write_views(out, archive, maps, cameras):
    front = cameras[0]
    # Each landmark includes the original image and a measured-mesh visibility map.
    stations = []
    # Keep the original twelve evaluation frames out of every appearance source.
    indices = sorted(set(range(14, 247, 5)) | (set(range(174, 212)) - {177, 203}) | LANDMARKS.keys())
    sources = [(i, front[i], LANDMARKS.get(i)) for i in indices]
    # Side cameras retain photographic detail when looking across the path.
    for cam_id, camera in enumerate(cameras[1:], 1):
        sources.extend(
            (cam_id * 1000 + i, frame, None)
            for i, frame in enumerate(camera)
            if (
                i % 8 == 0
                or (i % 3 == 0 and 1710755318 < float(Path(frame["file_path"]).stem) < 1710755416)
            )
            and i not in {29, 87, 175, 203}
            and 1710755131 < float(Path(frame["file_path"]).stem) < 1710755444
        )
    for i, frame, name in sources:
        photo = read_photo(archive, frame)
        cv2.imwrite(str(out / f"photo-{i}.webp"), photo, [cv2.IMWRITE_WEBP_QUALITY, 93])
        depth = maps.depth(frame)
        encoded = np.rint(np.minimum(depth, 65) * 1000).astype(np.uint16)
        # Two PNG channels encode Euclidean millimetres, zero marks a miss.
        encoded[~np.isfinite(depth)] = 0
        rgb = np.stack([np.zeros_like(encoded), encoded % 256, encoded // 256], axis=-1).astype("u1")
        cv2.imwrite(str(out / f"depth-{i}.png"), rgb)
        pose = np.array(frame["transform_matrix"])
        pose[:3, :] = UP @ pose[:3, :]
        stations.append(
            dict(
                name=name,
                id=i,
                detail=i != 198,
                pose=pose.tolist(),
                lens=[frame[k] for k in ["fl_x", "fl_y", "cx", "cy"]],
                distortion=[frame["k" + str(k)] for k in range(1, 5)],
            )
        )
    route = np.array([np.array(f["transform_matrix"])[:3, 3] for f in front[12:248]])
    (out / "scene.json").write_text(
        json.dumps(dict(views=stations, walkGrid=walk_clearance(maps.scene, route)), ensure_ascii=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["data", "output", "cache"]:
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    run(args.data, args.output, args.cache)
