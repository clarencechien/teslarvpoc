"""Reconstruct the three stair balustrades from the dense terrestrial survey."""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from build_christ_church import UP, project
from build_room import Glb
from church_geometry import survey_samples
from church_steps import FLIGHTS, read_primitives, subtract_prism, update_stats

# Survey-space work windows. The shape and cut-outs inside each window come from points.
BOXES = [
    ((-1.78, 14.25, -6.31), (-0.67, 19.80, -0.83)),
    ((1.71, 11.63, -6.22), (2.79, 18.02, 0.80)),
    ((5.88, 11.64, -4.65), (6.84, 18.02, 0.80)),
]


def planes_for_box(low, high):
    return np.column_stack([np.vstack([np.eye(3), -np.eye(3)]), np.r_[-low, high]])


# Traced lower edges of the pierced panels in the survey elevation (y, z).
# Keeping the masonry below these lines avoids cutting holes in solid plinths
# where the railing scan alone cannot see the near face.
SILLS = [
    [(14.25, -5.55), (17.37, -4.48), (19.80, -4.48)],
    [(11.63, -1.99), (18.02, -4.42)],
    [(11.64, -2.00), (18.02, -4.45)],
]


def trim_panel(points, triangles, attributes, index, *, patch):
    low, high = map(np.asarray, BOXES[index])
    knots = SILLS[index]
    for (y0, z0), (y1, z1) in zip(knots, knots[1:]):
        slope = (z1 - z0) / (y1 - y0)
        above = np.array([0.0, -slope, 1.0, slope * y0 - z0])
        if patch:
            # The patch is already clipped at the outside of its work window.
            planes = np.array([[0, 1, 0, -y0], [0, -1, 0, y1], -above])
        else:
            a, b = low.copy(), high.copy()
            a[1], b[1] = y0, y1
            planes = np.vstack([planes_for_box(a, b), above])
        points, triangles, attributes = subtract_prism(points, triangles, attributes, planes)
    return points, triangles, attributes


def reconstruct(points, bounds, spacing=0.016):
    low, high = map(np.asarray, bounds)
    # Extend the work volume beyond the replacement boundary to provide neighbourhoods
    # for normal estimation and to avoid capping solid masonry at the cut plane.
    pad = 0.09
    selected = points[((points >= low - pad) & (points <= high + pad)).all(1)]
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(selected))
    cloud = cloud.voxel_down_sample(0.013)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.055, max_nn=32))
    p = np.asarray(cloud.points)
    n = np.asarray(cloud.normals).copy()
    votes = np.zeros(len(p))
    for direction in [
        (1, 0, 0),
        (0.7, 0.7, 0.18),
        (0.7, -0.7, 0.18),
        (0, 1, 0),
        (0, 0, 1),
        (0.7, 0, 0.7),
        (0, 0.7, 0.7),
    ]:
        direction = np.array(direction, dtype="f8")
        direction /= np.linalg.norm(direction)
        axis = np.eye(3)[np.argmin(abs(direction))]
        u = np.cross(direction, axis)
        u /= np.linalg.norm(u)
        v = np.cross(direction, u)
        xy = np.rint(np.column_stack([p @ u, p @ v]) / 0.024).astype("i4")
        xy -= xy.min(0)
        ids = xy[:, 1] * (1 + xy[:, 0].max()) + xy[:, 0]
        d = p @ direction
        front = np.full(ids.max() + 1, -np.inf)
        back = np.full_like(front, np.inf)
        np.maximum.at(front, ids, d)
        np.minimum.at(back, ids, d)
        facing = n @ direction
        votes += (
            facing * abs(facing) * ((front[ids] - d < 0.028).astype(float) - (d - back[ids] < 0.028))
        )
    n[votes < 0] *= -1
    # A bounded, oriented point-to-plane distance field. Ignore voxels more than
    # a few centimetres from real returns; an open arch must not become a filled panel.
    origin = low - 0.04
    dims = np.ceil((high + 0.04 - origin) / spacing).astype(int) + 1
    field = np.zeros(dims, "f4")
    near = np.zeros(dims, "f4")
    tree = cKDTree(p)
    yz = np.indices(dims[1:]).reshape(2, -1).T
    for i in range(dims[0]):
        grid = origin + np.column_stack([np.full(len(yz), i), yz]) * spacing
        delta, idx = tree.query(grid, k=6, distance_upper_bound=0.075, workers=3)
        good = np.isfinite(delta)
        idx = np.minimum(idx, len(p) - 1)
        w = np.where(good, np.exp(-np.minimum(delta / 0.024, 5) ** 2), 0.0)
        signed = np.einsum("nij,nij->ni", grid[:, None, :] - p[idx], n[idx])
        field[i] = ((w * signed).sum(1) / np.maximum(w.sum(1), 1e-9)).reshape(dims[1:])
        near[i] = delta[:, 0].reshape(dims[1:])
    verts, faces, _, _ = marching_cubes(
        field, level=0, spacing=(spacing,) * 3, mask=near < 0.040, allow_degenerate=False
    )
    verts = verts + origin
    # Check vertices and centroids as the marching-cubes mask samples cell corners.
    dist = tree.query(verts, workers=3)[0]
    valid = dist[faces].max(1) < 0.041
    centers = verts[faces].mean(1)
    valid &= tree.query(centers, workers=3)[0] < 0.035
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces[valid])
    )
    labels, sizes, _ = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(sizes)[np.asarray(labels)] < 15)
    mesh.remove_unreferenced_vertices()
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    for axis in range(3):
        normal = np.eye(3)[axis]
        for loc, sign in [(low, 1), (high, -1)]:
            tmesh = tmesh.clip_plane(loc.astype("f4"), (normal * sign).astype("f4"))
    mesh = tmesh.to_legacy()
    # Exact coplanar fitted treads already have continuous high-resolution photographs.
    # Remove the flat part of the duplicate scan where it meets one of these treads.
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    tri = verts[faces]
    centers = tri.mean(1)
    norm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm /= np.maximum(np.linalg.norm(norm, axis=1)[:, None], 1e-10)
    remove = np.zeros(len(faces), bool)
    for left, right, start, end, z0, z1 in FLIGHTS:
        predicted = np.interp(centers[:, 1], [start + 0.36, end - 0.36], [z0, z1])
        remove |= (
            (centers[:, 0] > left)
            & (centers[:, 0] < right)
            & (centers[:, 1] > start)
            & (centers[:, 1] < end)
            & (abs(centers[:, 2] - predicted) < 0.24)
            & (abs(norm[:, 2]) > 0.7)
        )
    mesh.remove_triangles_by_mask(remove)
    mesh.remove_unreferenced_vertices()
    return mesh


def camera_list(data):
    rows = sorted(
        json.loads((data / "transforms_colmap_scaled.json").read_text())["frames"],
        key=lambda f: f["file_path"],
    )
    return [[f for f in rows if f"cam{i}_" in f["file_path"]] for i in range(3)]


def write_model(source, target, patches):
    doc, photos, groups = read_primitives(source, trim_flights=False)
    if any(m.get("name", "").startswith("Survey repair:") for m in doc["materials"]):
        raise ValueError("Input already has repaired rails; start from the fitted stairs build")
    glb = Glb()
    glb.doc = doc
    doc.update(accessors=[], bufferViews=[], buffers=[])
    doc["meshes"][0]["primitives"] = []
    for image, blob in zip(doc["images"], photos):
        image["bufferView"] = glb.view(blob)
    for material, key, p, t, attr in groups:
        if not doc["materials"][material].get("name", "").startswith("Measured stair flight"):
            for i in range(len(BOXES)):
                p, t, attr = trim_panel(p, t, attr, i, patch=False)
        if not len(t):
            continue
        attrs = {"POSITION": glb.acc((p @ UP.T).astype("f4"), True), key: glb.acc(attr.astype("f4"))}
        doc["meshes"][0]["primitives"].append(
            dict(attributes=attrs, material=material, indices=glb.acc(t.ravel().astype("u4")))
        )
    for i, (p, t, uv, image) in enumerate(patches):
        p, t, uv = trim_panel(p, t, uv, i, patch=True)
        glb.primitive(f"Survey repair: balustrade {i+1}", p @ UP.T, t, uv, image, jpeg_quality=94)
        doc["materials"][-1]["doubleSided"] = True
    glb.write(target)


def photo_sources(site, cameras, parts):
    from build_christ_church import visibility

    depth_scene = o3d.t.geometry.RaycastingScene(nthreads=3)
    for mesh in parts:
        depth_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    result = []
    for ident in json.loads((site / "photo-walk/sources.json").read_text()):
        frame = cameras[ident // 1000][ident % 1000]
        image = cv2.imread(str(site / f"photo-{ident}.webp"))
        data = cv2.imread(str(site / "photo-walk" / f"{ident}.png")).astype("f4")
        depth = (data[:, :, 2] * 256 + data[:, :, 1]) * 0.001
        own = visibility(depth_scene, frame, step=4)
        result.append((frame, image, depth, data[:, :, 0] / 255, own))
    return result


def bake_samples(points, normals, sources):
    """Fuse the three best visible photographic samples, confidence weighted in each texel."""
    count = len(points)
    weights = np.zeros((count, 3), "f4")
    rgb = np.zeros((count, 3, 3), "f4")
    row = np.arange(count)
    for frame, image, depth, confidence, own in sources:
        u, v, z = project(points, frame)
        edge = np.minimum.reduce([u, v, 1439 - u, 1079 - v])
        inside = (z > 0.22) & (edge > 0)
        if not inside.any():
            continue
        x = np.clip(((u + 0.5) / 4).astype(int), 0, 359)
        y = np.clip(((v + 0.5) / 4).astype(int), 0, 269)
        observed = depth[y, x]
        nearest = own[y, x]
        ray = np.asarray(frame["transform_matrix"], dtype="f4")[:3, 3] - points
        distance = np.linalg.norm(ray, axis=1)
        facing = abs((normals * ray).sum(1)) / np.maximum(0.01, distance)
        # A second visibility test on the new mesh prevents colour from the
        # opposite side of a thin baluster from being projected onto this side.
        disagreement = abs(distance - observed) / (0.075 + 0.012 * distance)
        weight = (
            inside
            * np.clip(edge / 75, 0, 1)
            * confidence[y, x]
            * np.exp(-np.minimum(disagreement**2, 20))
            * (nearest + 0.045 + 0.008 * distance >= distance)
            * (0.07 + 0.93 * facing**2)
            / (1 + 0.20 * distance) ** 2
        )
        worst = np.argmin(weights, axis=1)
        change = (weight > weights[row, worst]) & (weight > 0.0015)
        if not change.any():
            continue
        uu = u[change]
        vv = v[change]
        ix = uu.astype("i4")
        iy = vv.astype("i4")
        fx = (uu - ix)[:, None]
        fy = (vv - iy)[:, None]
        samples = (1 - fy) * ((1 - fx) * image[iy, ix] + fx * image[iy, ix + 1]) + fy * (
            (1 - fx) * image[iy + 1, ix] + fx * image[iy + 1, ix + 1]
        )
        rgb[row[change], worst[change]] = samples
        weights[row[change], worst[change]] = weight[change]
    # A robust reference reduces exposure seams and residual foreground contamination.
    # A single observation stays sharp when it is the only valid source.
    best = np.argmax(weights, axis=1)
    deviation = np.mean(abs(rgb - rgb[row, best][:, None, :]), axis=2)
    weights = weights**1.5 / (1 + (deviation / 26) ** 2)
    total = weights.sum(1)
    return (rgb * weights[:, :, None]).sum(1) / np.maximum(total[:, None], 1e-10), total > 0


def atlas_patch(mesh, side, sources):
    import xatlas

    p = np.asarray(mesh.vertices, dtype="f4")
    faces = np.asarray(mesh.triangles, dtype="u4")
    atlas = xatlas.Atlas()
    atlas.add_mesh(p, faces)
    chart = xatlas.ChartOptions()
    chart.max_iterations = 1
    pack = xatlas.PackOptions()
    pack.resolution = side
    pack.padding = 4
    atlas.generate(chart, pack)
    if atlas.atlas_count != 1:
        raise ValueError("Repair UVs must occupy a single texture atlas")
    ids, tri, uv = atlas[0]
    vertices = p[ids]
    # Rasterize the packed UV charts so every texel has an actual 3D sample position.
    scene = o3d.t.geometry.RaycastingScene(nthreads=3)
    scene.add_triangles(
        o3d.core.Tensor(np.column_stack([uv, np.zeros(len(uv), "f4")]).astype("f4")),
        o3d.core.Tensor(tri.astype("u4")),
    )
    pixel = (np.arange(side, dtype="f4") + 0.5) / side
    xx, yy = np.meshgrid(pixel, pixel)
    rays = np.stack(
        [xx, yy, np.ones_like(xx), np.zeros_like(xx), np.zeros_like(xx), -np.ones_like(xx)], -1
    )
    hit = scene.cast_rays(o3d.core.Tensor(rays), nthreads=3)
    covered = np.isfinite(hit["t_hit"].numpy())
    polys = tri[hit["primitive_ids"].numpy()[covered]]
    bary = hit["primitive_uvs"].numpy()[covered]
    pts = (vertices[polys] * np.column_stack([1 - bary.sum(1), bary])[:, :, None]).sum(1)
    mesh.compute_vertex_normals()
    smooth = np.asarray(mesh.vertex_normals, dtype="f4")[ids]
    normal = (smooth[polys] * np.column_stack([1 - bary.sum(1), bary])[:, :, None]).sum(1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-8)
    result = np.zeros_like(pts)
    valid = np.zeros(len(pts), bool)
    for start in range(0, len(pts), 32000):
        sub = slice(start, start + 32000)
        result[sub], valid[sub] = bake_samples(pts[sub], normal[sub], sources)
    if not valid.any():
        raise ValueError("No visible photographs in repair window")
    if (~valid).any():
        # Fill unobserved texture in surface coordinates; include orientation to avoid
        # colour bleeding between neighbouring faces on opposite sides of the stone.
        feature = np.column_stack([pts, normal * 0.075])
        tree = cKDTree(feature[valid])
        dist, near = tree.query(feature[~valid], k=5, workers=3)
        w = 1 / np.maximum(dist, 0.005) ** 2
        result[~valid] = (result[valid][near] * w[:, :, None]).sum(1) / w.sum(1)[:, None]
    image = np.zeros((side, side, 3), "u1")
    image[covered] = np.clip(result, 0, 255).astype("u1")
    padding = distance_transform_edt(~covered, return_distances=False, return_indices=True)
    image = image[tuple(padding)]
    print("atlas texels with photograph", int(valid.sum()), "/", len(valid), flush=True)
    return vertices, tri, uv, image


def build(data, site, cache):
    import tempfile
    from build_christ_church import write_views, MeasuredVisibility
    from evaluate_christ_church import glb_scene
    import zipfile

    cameras = camera_list(data)
    source = site / "christ-church.glb"
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    points = None
    for i, bounds in enumerate(BOXES):
        path = cache / f"rail-repair-16mm-v1-{i}.npz"
        if not path.exists():
            if points is None:
                points, _, _ = survey_samples(data, cameras[0], 1)
            mesh = reconstruct(points, bounds)
            mesh = mesh.simplify_quadric_decimation(min(len(mesh.triangles), 41000))
            np.savez(path, p=np.asarray(mesh.vertices), t=np.asarray(mesh.triangles))
        values = np.load(path)
        parts.append(
            o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(values["p"]), o3d.utility.Vector3iVector(values["t"])
            )
        )
    sources = photo_sources(site, cameras, parts)
    with tempfile.TemporaryDirectory(dir=site) as tmp:
        out = Path(tmp)
        for compact in [False, True]:
            output = "christ-church-mobile.glb" if compact else source.name
            patches = []
            for i, mesh in enumerate(parts):
                print("baking", output, i, flush=True)
                if compact:
                    mesh = mesh.simplify_quadric_decimation(12000)
                patches.append(atlas_patch(mesh, 768 if compact else 1280, sources))
            write_model(site / output, out / output, patches)
        scene, _ = glb_scene(out / source.name, textures=False)
        maps = MeasuredVisibility(scene, data, cameras[0])
        try:
            with zipfile.ZipFile(data / "images.zip") as archive:
                write_views(out, archive, maps, cameras)
        finally:
            maps.archive.close()
        for file in out.iterdir():
            file.replace(site / file.name)
    stats = json.loads((site / "stats.json").read_text())
    update_stats(site, stats["stairRisers"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ["data", "site", "cache"]:
        parser.add_argument(arg, type=Path)
    build(**vars(parser.parse_args()))
