"""Retain repeatedly measured mobile surfaces where the terrestrial survey has gaps."""

import re
import zipfile
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from church_geometry import base_to_front


def scans(data, front):
    times = np.array([float(Path(f["file_path"]).stem) for f in front])
    calibration = base_to_front()
    with zipfile.ZipFile(data / "undist-clouds.zip") as archive:
        for name in sorted(n for n in archive.namelist() if n.endswith(".pcd")):
            stamp = float(re.search(r"cloud_(\d+)_(\d+)\.pcd", name).expand(r"\1.\2"))
            idx = abs(times - stamp).argmin()
            if abs(times[idx] - stamp) > 0.025:
                continue
            pose = np.array(front[idx]["transform_matrix"]) @ calibration
            raw = archive.read(name).split(b"DATA binary\n", 1)[1]
            scan = np.frombuffer(raw, "<f4").reshape(-1, 8)
            yield pose, scan


def mobile_envelope(data, front):
    blocks = []
    for pose, scan in scans(data, front):
        xyz, normal = scan[:, :3], scan[:, 4:7]
        length = np.linalg.norm(xyz, axis=1)
        keep = (length > 0.5) & (length < 45) & (np.linalg.norm(normal, axis=1) > 0.5)
        xyz, normal = xyz[keep], normal[keep]
        normal[(normal * xyz).sum(1) > 0] *= -1
        block = o3d.geometry.PointCloud()
        block.points = o3d.utility.Vector3dVector(xyz @ pose[:3, :3].T + pose[:3, 3])
        block.normals = o3d.utility.Vector3dVector(normal @ pose[:3, :3].T)
        blocks.append(block.voxel_down_sample(0.075))
    cloud = o3d.geometry.PointCloud()
    for block in blocks:
        cloud += block
    cloud = cloud.voxel_down_sample(0.065)
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=12, std_ratio=2.3)
    cloud.normalize_normals()
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=10, n_threads=4)
    distance = cKDTree(np.asarray(cloud.points)).query(np.asarray(mesh.vertices), workers=4)[0]
    mesh.remove_vertices_by_mask(distance > 0.38)
    labels, counts, _ = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(counts)[np.asarray(labels)] < 40)
    mesh.remove_unreferenced_vertices()
    mesh = mesh.simplify_quadric_decimation(235000)
    return np.asarray(mesh.vertices).astype("f4"), np.asarray(mesh.triangles).copy()


def bins(ray):
    horizontal = np.hypot(ray[:, 0], ray[:, 1])
    col = (np.arctan2(ray[:, 1], ray[:, 0]) * (720 / (2 * np.pi)) + 360).astype("i4") % 720
    row = np.clip((np.arctan2(ray[:, 2], horizontal) * (360 / np.pi) + 180).astype("i4"), 0, 359)
    return row, col, np.hypot(horizontal, ray[:, 2])


def supplement(data, front, cache, p, t, survey_tree):
    # Candidate geometry is regenerated on a fresh build and always filtered.
    file = cache / "mobile-candidates.npz"
    if file.exists():
        raw = np.load(file)
        mp, mt = raw["p"], raw["t"]
    else:
        mp, mt = mobile_envelope(data, front)
        np.savez(file, p=mp, t=mt)
    center = mp[mt].mean(1)
    votes = cache / "mobile-repeat-v1.npz"
    if votes.exists():
        measured = np.load(votes)
        count, free, baseline = (measured[k] for k in ("count", "free", "baseline"))
    else:
        count = np.zeros(len(mt), "i4")
        free = count.copy()
        first = np.zeros((len(mt), 3), "f4")
        baseline = np.zeros(len(mt), bool)
        for pose, scan in scans(data, front):
            origin = pose[:3, 3].astype("f4")
            # The PCD is in the base frame and the lidar origin is 12.4 cm above it.
            offset = pose[:3, :3] @ np.array([0, 0, 0.124])
            world_ray = scan[:, :3] @ pose[:3, :3].T - offset
            r, c, d = bins(world_ray)
            ok = (d > 0.45) & (d < 40)
            depths = np.full((360, 720), np.inf, "f4")
            np.minimum.at(depths, (r[ok], c[ok]), d[ok])
            origin = origin + offset
            r, c, d = bins(center - origin)
            measured = depths[r, c]
            close = abs(measured - d) < (0.085 + 0.006 * d)
            # A beam across a depth boundary cannot certify free space: use the
            # nearest local return and require a return in the actual angular cell.
            near = cv2.erode(depths, np.ones((3, 3), "u1"))[r, c]
            free += np.isfinite(measured) & (near > d + 0.22 + 0.012 * d)
            first[close & (count == 0)] = origin
            baseline |= close & (count > 0) & (((first - origin) ** 2).sum(1) > 0.40**2)
            count += close
        np.savez(votes, count=count, free=free, baseline=baseline)
    repeated = (count >= 3) & baseline & (free <= np.maximum(1, count * 0.45))
    # The mobile scanner sees a narrow vertical band. For treads and ceilings
    # it often has few observations; use the separate terrestrial survey there.
    vertex_support = survey_tree.query(mp, workers=4)[0]
    center_support = survey_tree.query(center, workers=4)[0]
    surveyed = (
        (center_support < 0.15)
        & (vertex_support[mt].max(1) < 0.20)
        & (free <= np.maximum(2, count * 0.8))
    )
    accepted = repeated | surveyed
    print("Mobile faces supported by repeated ranges or the terrestrial survey", accepted.sum(), "/", len(mt), flush=True)
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(p), o3d.core.Tensor(t.astype("u4")))
    # Leave the survey surface in place when both meshes agree. Add supported
    # mobile faces across holes and where the survey saw another surface.
    ids = np.flatnonzero(accepted)
    dist = scene.compute_distance(o3d.core.Tensor(center[ids]), nthreads=4).numpy()
    vertex_dist = scene.compute_distance(o3d.core.Tensor(mp), nthreads=4).numpy()
    use = ids[(dist > 0.027) | (vertex_dist[mt[ids]].max(1) > 0.065)]
    combined = np.vstack([p, mp])
    faces = np.vstack([t, mt[use] + len(p)])
    print("Added supported mobile faces", len(use), flush=True)
    return combined, faces
