"""Reconstruct static surfaces in the photographic coordinate system from Oxford's terrestrial survey."""

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def base_to_front():
    cam_lidar = np.eye(4)
    cam_lidar[:3, :3] = Rotation.from_quat(
        [0.50347256, 0.50033372, -0.49475393, 0.50139815]
    ).as_matrix()
    cam_lidar[:3, 3] = [0.000350609, -0.07957405, -0.05382606]
    base_lidar = np.diag([-1.0, -1.0, 1.0, 1.0])
    base_lidar[2, 3] = 0.124
    return np.diag([1, -1, -1, 1]) @ cam_lidar @ np.linalg.inv(base_lidar)


def calibrated_centers(data, front):
    # The published trajectory describes the base. Translate its origin to cam0.
    gt = np.loadtxt(data / "gt-tum.txt")
    times = np.array([float(Path(frame["file_path"]).stem) for frame in front])
    indices = abs(times[:, None] - gt[:, 0]).argmin(axis=1)
    paired = abs(times - gt[indices, 0]) < 0.025
    poses = np.array([frame["transform_matrix"] for frame in front])[paired]
    rows = gt[indices[paired]]
    base_to_cam_center = np.linalg.inv(base_to_front())[:3, 3].copy()
    source = rows[:, 1:4] + Rotation.from_quat(rows[:, 4:]).apply(base_to_cam_center)
    target = poses[:, :3, 3]
    u, _, v = np.linalg.svd((source - source.mean(0)).T @ (target - target.mean(0)))
    rot = v.T @ np.diag([1, 1, np.linalg.det(v.T @ u.T)]) @ u.T
    trans = target.mean(0) - rot @ source.mean(0)
    error = np.linalg.norm(source @ rot.T + trans - target, axis=1)
    print(
        "Survey alignment: paired poses, median and P95 metres",
        len(rows),
        np.percentile(error, [50, 95]),
        flush=True,
    )
    return rot.astype("f4"), trans.astype("f4"), target.astype("f4")


def survey_samples(data, front, stride):
    rotation, translation, route = calibrated_centers(data, front)
    path = data / "merged-cloud-1cm.pcd"
    with path.open("rb") as handle:
        header = {}
        for line in handle:
            key, _, value = line.decode("ascii").strip().partition(" ")
            header[key] = value
            if key == "DATA":
                break
        if (
            header.get("FIELDS") != "Intensity rgb x y z _"
            or header["DATA"] != "binary"
            or header["COUNT"] != "1 1 1 1 1 4"
        ):
            raise ValueError("Expected the published Christ Church 1 cm binary PCD")
        offset = handle.tell()
    raw = np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=(int(header["POINTS"]), 6))
    clouds = []
    colors = []
    # Preserve height above the gallery and below the lowest stair landing.
    low = route.min(0) - [9, 7, 4]
    high = route.max(0) + [9, 7, 18]
    raw = raw[::stride]
    for i in range(0, len(raw), 1000000):
        block = raw[i : i + 1000000]
        xyz = block[:, 2:5] @ rotation.T + translation
        mask = ((xyz >= low) & (xyz <= high)).all(1)
        if not mask.any():
            continue
        clouds.append(xyz[mask])
        if stride > 1:
            packed = block[mask, 1].copy().view("u4")
            colors.append(np.column_stack([packed >> 16, packed >> 8, packed]).astype("u1"))
    return np.concatenate(clouds), np.concatenate(colors) if colors else None, route


def survey_colors(data, front):
    xyz, rgb, _ = survey_samples(data, front, 11)
    return cKDTree(xyz), rgb


def survey_cloud(data, front):
    xyz, _, route = survey_samples(data, front, 1)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pts = np.asarray(cloud.voxel_down_sample(0.028).points).astype("f4")
    print("Survey points used for surface fusion", len(pts), flush=True)
    return pts, route


def virtual_depth(points, origin, axes, size, focal):
    # Convert a dense set of measured surface samples into a first-return depth image.
    xyz = (points - origin) @ axes.T
    z = xyz[:, 2]
    use = (z > 0.30) & (z < 32) & (abs(xyz[:, 0]) < z) & (abs(xyz[:, 1]) < z)
    xyz = xyz[use]
    z = xyz[:, 2]
    px = np.clip(np.floor(xyz[:, :2] * (focal / z[:, None]) + size / 2).astype("i4"), 0, size - 1)
    depth = np.full(size * size, np.inf, "f4")
    np.minimum.at(depth, px[:, 1] * size + px[:, 0], z)
    depth = depth.reshape(size, size)
    # Fill a one-pixel sampling gap only where surrounding measured ranges agree.
    kernel = np.ones((3, 3), "u1")
    near = cv2.erode(depth, kernel)
    far = cv2.dilate(np.where(np.isfinite(depth), depth, 0), kernel)
    fill = ~np.isfinite(depth) & (near < 32) & (far >= near) & (far - near < 0.06 + 0.012 * near)
    depth[fill] = near[fill]
    depth[~np.isfinite(depth)] = 0
    return np.ascontiguousarray(depth)


def reconstruct_survey(data, front):
    pts, route = survey_cloud(data, front)
    return fuse_survey_points(pts, route)


def fuse_survey_points(pts, route):
    size = 384
    tensor = o3d.core.Tensor
    intrinsic = tensor(
        [[size / 2, 0, (size - 1) / 2], [0, size / 2, (size - 1) / 2], [0, 0, 1]],
        dtype=o3d.core.float64,
    )
    volume = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight"),
        attr_dtypes=(o3d.core.float32, o3d.core.float32),
        attr_channels=(1, 1),
        voxel_size=0.046,
        block_resolution=8,
        block_count=64000,
    )
    # Select observed camera positions spread along all aisles and both stair flights.
    centers = []
    for pos in route:
        if not centers or np.linalg.norm(np.asarray(centers) - pos, axis=1).min() > 1.65:
            centers.append(pos)
    unit = np.eye(3, dtype="f4")
    for n, center in enumerate(centers):
        for direction in [*unit, *-unit]:
            down = unit[1] if abs(direction[2]) > 0.5 else -unit[2]
            right = np.cross(down, direction)
            axes = np.stack([right, down, direction])
            depth = virtual_depth(pts, center, axes, size, size / 2)
            image = o3d.t.geometry.Image(tensor(np.rint(depth * 1000).astype("u2")))
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = axes
            extrinsic[:3, 3] = -axes @ center
            pose = tensor(extrinsic)
            options = dict(depth_scale=1000.0, depth_max=32.0, trunc_voxel_multiplier=3.0)
            blocks = volume.compute_unique_block_coordinates(image, intrinsic, pose, **options)
            if len(blocks):
                volume.integrate(blocks, image, intrinsic, pose, **options)
        if n % 8 == 0:
            print("Fused survey viewpoints", n + 1, "/", len(centers), flush=True)
    mesh = volume.extract_triangle_mesh(weight_threshold=2.0).to_legacy()
    if not len(mesh.triangles):
        raise RuntimeError("The terrestrial survey produced no observed surface")
    mesh.remove_unreferenced_vertices()
    tree = cKDTree(pts)

    # Trim unsupported triangles both before and after edge-collapse simplification.
    def trim(mesh):
        p = np.asarray(mesh.vertices)
        t = np.asarray(mesh.triangles)
        distance = tree.query(p, workers=4)[0]
        center_dist = tree.query(p[t].mean(1), workers=4)[0]
        mesh.remove_triangles_by_mask((distance[t].max(1) > 0.135) | (center_dist > 0.135))
        mesh.remove_unreferenced_vertices()
        return mesh

    mesh = trim(mesh)
    labels, counts, _ = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(counts)[np.asarray(labels)] < 60)
    mesh.remove_unreferenced_vertices()
    mesh = trim(mesh.simplify_quadric_decimation(235000))
    print("Survey mesh", len(mesh.triangles), flush=True)
    return np.asarray(mesh.vertices).astype("f4"), np.asarray(mesh.triangles).copy()


def walk_clearance(scene, route):
    """Test floor and body clearance on a grid around the recorded walking corridors (Z-up)."""
    cell = 0.16
    lower = np.floor((route[:, :2].min(0) - 1.6) / cell) * cell
    upper = route[:, :2].max(0) + 1.6
    cols, rows = np.ceil((upper - lower) / cell).astype(int) + 1
    x, y = np.meshgrid(lower[0] + np.arange(cols) * cell, lower[1] + np.arange(rows) * cell)
    xy = np.column_stack([x.ravel(), y.ravel()])
    dist, idx = cKDTree(route[:, :2]).query(xy)
    inside = dist < 1.55
    xy = xy[inside]
    nearest = idx[inside]
    sq = ((xy[:, None, :] - route[None, :, :2]) ** 2).sum(2)
    weight = np.where(
        (sq < dist[inside, None] ** 2 + 2) & (abs(route[None, :, 2] - route[nearest, None, 2]) < 1),
        1 / (0.06 + sq) ** 1.6,
        0,
    )
    height = (weight @ route[:, 2]) / weight.sum(1)
    eye = np.column_stack([xy, height]).astype("f4")

    def cast(origins, directions):
        rays = np.column_stack([origins, np.broadcast_to(directions, origins.shape)]).astype("f4")
        return scene.cast_rays(o3d.core.Tensor(rays), nthreads=4)["t_hit"].numpy()

    # Near a stair nosing a single vertical ray can pass through a small scan gap.
    # A neighbouring tread within a shoe-width also provides ground support.
    floors = np.stack(
        [
            cast(eye + [dx, dy, 0], [0, 0, -1])
            for dx, dy in [(0, 0), (0.12, 0), (-0.12, 0), (0, 0.12), (0, -0.12)]
        ]
    )
    floor = ((floors > 0.42) & (floors < 2.5)).sum(0) >= 2
    clear = floor & (cast(eye, [0, 0, 1]) > 0.15)
    for z in [0, -0.45]:
        for angle in np.arange(8) * np.pi / 4:
            clear &= cast(eye + [0, 0, z], [np.cos(angle), np.sin(angle), 0]) > 0.24
    values = np.full(rows * cols, -32768, "i2")
    values[np.flatnonzero(inside)[clear]] = np.rint(height[clear] * 100).astype("i2")
    import base64

    return dict(
        origin=lower.tolist(),
        cols=int(cols),
        rows=int(rows),
        cell=cell,
        heights=base64.b64encode(values.astype("<i2").tobytes()).decode(),
    )


def static_infill(points, reference_points, reference_faces):
    # Infer normals from the measured point neighbourhood, with the survey
    # TSDF fixing their sign on opposite sides of walls and railings.
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(
        o3d.core.Tensor(reference_points), o3d.core.Tensor(reference_faces.astype("u4"))
    )
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(0.066)
    xyz = np.asarray(cloud.points).astype("f4")
    near = scene.compute_closest_points(o3d.core.Tensor(xyz), nthreads=4)
    keep = np.linalg.norm(xyz - near["points"].numpy(), axis=1) < 0.75
    xyz = xyz[keep]
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    cloud.normals = o3d.utility.Vector3dVector(near["primitive_normals"].numpy()[keep])
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.22, max_nn=34))
    cloud.normalize_normals()
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=10, n_threads=3)
    tree = cKDTree(xyz)
    mesh.remove_vertices_by_mask(tree.query(np.asarray(mesh.vertices), workers=4)[0] > 0.16)
    labels, counts, _ = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(counts)[np.asarray(labels)] < 55)
    mesh.remove_unreferenced_vertices()
    mesh = mesh.simplify_quadric_decimation(215000)
    p = np.asarray(mesh.vertices).astype("f4")
    t = np.asarray(mesh.triangles)
    return p, t[tree.query(p[t].mean(1), workers=4)[0] < 0.17]


def fill_remaining(points, faces, sp, sf):
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(points), o3d.core.Tensor(faces.astype("u4")))
    # Only fill local gaps. The terrestrial-only fit must not be superimposed
    # on existing treads, where a second nearly coplanar layer flickers.
    centers = sp[sf].mean(1)
    distance = scene.compute_distance(o3d.core.Tensor(centers), nthreads=4).numpy()
    support = scene.compute_distance(o3d.core.Tensor(sp), nthreads=4).numpy()
    keep = (distance > 0.075) & (support[sf].min(1) > 0.015)
    return np.vstack([points, sp]), np.vstack([faces, sf[keep] + len(points)])
