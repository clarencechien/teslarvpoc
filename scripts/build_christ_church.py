"""Build a photo-textured Oxford Spires Christ Church 03 scene from the published archives."""

import argparse
import json
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from build_room import Glb

UP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
CREDIT = "Oxford Spires Dataset, Yifu Tao et al. | CC BY-NC-SA 4.0"
LANDMARKS = {21: "大廳長桌", 50: "高窗與肖像", 175: "樓梯平台", 192: "石階入口"}


def rigid(translation, quaternion):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def fuse(data, front):
    # Oxford Spires configs/sensor.yaml. The PCDs are in the robot base frame.
    cam_lidar = rigid(
        [0.000350609, -0.07957405, -0.05382606], [0.50347256, 0.50033372, -0.49475393, 0.50139815]
    )
    base_lidar = rigid([0, 0, 0.124], [0, 0, 1, 0])
    cam_base = np.diag([1, -1, -1, 1]) @ cam_lidar @ np.linalg.inv(base_lidar)
    times = np.array([float(Path(f["file_path"]).stem) for f in front])
    clouds = []
    used_scans = []
    with zipfile.ZipFile(data / "undist-clouds.zip") as archive:
        for name in sorted(n for n in archive.namelist() if n.endswith(".pcd")):
            stamp = float(re.search(r"cloud_(\d+)_(\d+)\.pcd", name).expand(r"\1.\2"))
            i = abs(times - stamp).argmin()
            if abs(times[i] - stamp) > 0.025:
                continue
            used_scans.append(name)
            transform = np.array(front[i]["transform_matrix"]) @ cam_base
            raw = archive.read(name).split(b"DATA binary\n", 1)[1]
            scan = np.frombuffer(raw, "<f4").reshape(-1, 8)
            xyz, normal = scan[:, :3], scan[:, 4:7]
            length = np.linalg.norm(xyz, axis=1)
            keep = (length > 0.5) & (length < 45) & (np.linalg.norm(normal, axis=1) > 0.5)
            xyz, normal = xyz[keep], normal[keep]
            normal[(normal * xyz).sum(1) > 0] *= -1
            cloud = o3d.geometry.PointCloud()
            vector = o3d.utility.Vector3dVector
            cloud.points = vector(xyz @ transform[:3, :3].T + transform[:3, 3])
            cloud.normals = vector(normal @ transform[:3, :3].T)
            clouds.append(cloud.voxel_down_sample(0.075))
    cloud = o3d.geometry.PointCloud()
    for block in clouds:
        cloud += block
    del clouds
    cloud = cloud.voxel_down_sample(0.065)
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=12, std_ratio=2.3)
    cloud.normalize_normals()
    print("registered scans, fused points", len(used_scans), len(cloud.points), flush=True)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=10, n_threads=4)
    tree = cKDTree(np.asarray(cloud.points))
    dist, _ = tree.query(np.asarray(mesh.vertices), workers=4)
    mesh.remove_vertices_by_mask(dist > 0.38)
    clusters, counts, _ = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(counts)[np.asarray(clusters)] < 40)
    mesh.remove_unreferenced_vertices()
    mesh = mesh.simplify_quadric_decimation(235000)
    print("surface triangles", len(mesh.triangles), flush=True)
    return np.asarray(mesh.vertices).copy().astype("f4"), np.asarray(mesh.triangles).copy()


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


def read_photo(archive, frame):
    raw = archive.read(frame["file_path"].removeprefix("images/"))
    return cv2.imdecode(np.frombuffer(raw, "u1"), cv2.IMREAD_COLOR)


def assignments(points, faces, frames, scene):
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
        depth = visibility(scene, frame)
        observed = depth[(v[ids] / 6).astype(int), (u[ids] / 6).astype(int)]
        valid = np.abs(observed - dist) < (0.10 + 0.016 * dist)
        cosine = (ray * normals[ids]).sum(1) / dist
        border = np.minimum(np.minimum(u[ids], 1439 - u[ids]), np.minimum(v[ids], 1079 - v[ids]))
        score = np.maximum(0, cosine) * np.clip(border / 90, 0.12, 1) / (1 + 0.14 * dist)
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


def bake(path, archive, points, faces, frames, labels, tile_width):
    glb = Glb()
    glb.doc["asset"].update(
        generator="Oxford Spires LiDAR and fisheye photo projection", copyright=CREDIT
    )
    glb.doc["nodes"][0]["name"] = "Christ Church, Oxford"
    world = points @ UP.T
    photo_ids = np.unique(labels[labels >= 0])
    colored, colors = [], []
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
            colored.append(points[ids])
            colors.append(img[v.round().astype(int), u.round().astype(int)])
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
        dist, near = cKDTree(np.concatenate(colored)).query(points[ids])
        col = np.concatenate(colors)[near]
        col[dist > 0.8] = [79, 84, 87]
        glb.primitive("Untextured scan, nearby colour", world[ids], local, colors=col)
    for mat in glb.doc["materials"]:
        mat["doubleSided"] = True
    glb.write(path)


def bake_mobile(out, archive, points, faces, frames, visibility_scene):
    # Keep visibility testing against the full mesh when texturing the lighter version.
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(points), o3d.utility.Vector3iVector(faces)
    )
    mesh = mesh.simplify_quadric_decimation(72000)
    p = np.asarray(mesh.vertices).astype("f4")
    t = np.asarray(mesh.triangles)
    labels = assignments(p, t, frames, visibility_scene)
    bake(out / "christ-church-mobile.glb", archive, p, t, frames, labels, 448)


def run(data, out, cache):
    cache.mkdir(exist_ok=True, parents=True)
    out.mkdir(exist_ok=True, parents=True)
    frames = sorted(
        json.loads((data / "transforms_colmap_scaled.json").read_text())["frames"],
        key=lambda f: f["file_path"],
    )
    cameras = [[f for f in frames if f"cam{i}_" in f["file_path"]] for i in range(3)]
    mesh_path = cache / "mesh.npz"
    if mesh_path.exists():
        m = np.load(mesh_path)
        points, faces = m["p"], m["t"]
    else:
        points, faces = fuse(data, cameras[0])
        np.savez(mesh_path, p=points, t=faces)
    archive = zipfile.ZipFile(data / "images.zip")
    # Omit frames after the camera exposure/white balance changes to deep red.
    selected = [
        f
        for camera in cameras
        for f in camera[::4]
        if 1710755131 < float(Path(f["file_path"]).stem) < 1710755444
    ]
    mesh = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(points), o3d.core.Tensor(faces.astype("i8")))
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(mesh)
    labels = assignments(points, faces, selected, scene)
    bake(out / "christ-church.glb", archive, points, faces, selected, labels, 896)
    bake_mobile(out, archive, points, faces, selected, scene)
    write_views(out, archive, scene, cameras)
    stats = dict(
        keyframes=len(frames),
        textureViews=len(set(labels[labels >= 0])),
        triangles=len(faces),
        photoCoverage=round(float((labels >= 0).mean()), 4),
    )
    (out / "stats.json").write_text(json.dumps(stats))
    print(stats, flush=True)


def write_views(out, archive, scene, cameras):
    front = cameras[0]
    # Each landmark includes the original image and a measured-mesh visibility map.
    stations = []
    indices = sorted(set(range(14, 247, 5)) | LANDMARKS.keys())
    sources = [(i, front[i], LANDMARKS.get(i)) for i in indices]
    # Side cameras retain photographic detail when looking across the path.
    for cam_id, camera in enumerate(cameras[1:], 1):
        sources.extend(
            (cam_id * 1000 + i, frame, None)
            for i, frame in enumerate(camera)
            if i % 8 == 0 and 1710755131 < float(Path(frame["file_path"]).stem) < 1710755444
        )
    for i, frame, name in sources:
        photo = read_photo(archive, frame)
        cv2.imwrite(str(out / f"photo-{i}.webp"), photo, [cv2.IMWRITE_WEBP_QUALITY, 93])
        depth = visibility(scene, frame)
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
                pose=pose.tolist(),
                lens=[frame[k] for k in ["fl_x", "fl_y", "cx", "cy"]],
                distortion=[frame["k" + str(k)] for k in range(1, 5)],
            )
        )
    route = [(UP @ np.array(f["transform_matrix"])[:3, 3]).round(3).tolist() for f in front[12:248]]
    (out / "scene.json").write_text(json.dumps(dict(views=stations, route=route), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["data", "output", "cache"]:
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    run(args.data, args.output, args.cache)
