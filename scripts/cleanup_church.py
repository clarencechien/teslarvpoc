"""Clean reconstructed surfaces with calibrated photographs and a separate terrestrial survey."""

import argparse
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from build_christ_church import UP, MeasuredVisibility, camera_rays, project, write_views
from build_church_photowalk import H, W, measured_depth, people_visibility
from build_room import Glb
from church_geometry import survey_samples
from church_rail_repair import camera_list
from church_steps import read_primitives, update_stats
from evaluate_christ_church import glb_scene
from review_church_cleanup import DEFAULT_FONT, review
from verify_church_edits import control_views, verify


@dataclass
class Observation:
    ident: int
    frame: dict
    origin: np.ndarray
    rays: np.ndarray
    depth: np.ndarray
    confidence: np.ndarray
    photo: np.ndarray
    nearest: np.ndarray


def prepare(data, site, detector):
    cameras = camera_list(data)
    points, _, _ = survey_samples(data, cameras[0], 5)
    tree = cKDTree(points)
    net = cv2.dnn.readNet(str(detector))
    cv2.setNumThreads(2)
    u, v = np.meshgrid(np.arange(W) * 4 + 2, np.arange(H) * 4 + 2)
    chosen = []
    views = json.loads((site / "scene.json").read_text())["views"]
    for view in views:
        if not view.get("detail", True):
            continue
        ident = view["id"]
        frame = cameras[ident // 1000][ident % 1000]
        pose = np.asarray(frame["transform_matrix"], dtype="f4")
        origin = pose[:3, 3]
        if any(
            np.linalg.norm(origin - o.origin) < 0.95
            and pose[:3, 2] @ np.asarray(o.frame["transform_matrix"])[:3, 2] > 0.88
            for o in chosen
        ):
            continue
        photo = cv2.imread(str(site / f"photo-{ident}.webp"))
        existing = site / "photo-walk" / f"{ident}.png"
        if existing.exists():
            rgb = cv2.imread(str(existing)).astype("f4")
            depth = (rgb[:, :, 2] * 256 + rgb[:, :, 1]) * 0.001
            confidence = rgb[:, :, 0] / 255
        else:
            near = tree.query_ball_point(origin, 11)
            if not near:
                continue
            depth = measured_depth(points[near], frame)
            confidence = people_visibility(photo, net, depth.shape)
        rays = (camera_rays(frame, u, v) @ pose[:3, :3].T).reshape(H, W, 3)
        chosen.append(
            Observation(
                ident,
                frame,
                origin,
                rays,
                depth,
                confidence,
                photo,
                cv2.erode(depth, np.ones((3, 3), "u1")),
            )
        )
    return cameras, chosen, tree


def ray_scene(parts):
    scene = o3d.t.geometry.RaycastingScene(nthreads=3)
    for _, _, points, faces, _ in parts:
        scene.add_triangles(o3d.core.Tensor(points.astype("f4")), o3d.core.Tensor(faces.astype("u4")))
    return scene


def cast(scene, ob):
    rays = np.concatenate([np.broadcast_to(ob.origin, ob.rays.shape), ob.rays], axis=2)
    return scene.cast_rays(o3d.core.Tensor(rays), nthreads=3)


def clean(parts, observations, tree):
    scene = ray_scene(parts)
    counts = np.array([len(group[3]) for group in parts])
    offsets = np.r_[0, counts.cumsum()]
    length = offsets[-1]
    seen = np.zeros(length, bool)
    separate = np.zeros(length, bool)
    first = np.zeros((length, 3), "f4")
    supported = np.zeros(length, bool)
    for ob in observations:
        hit = cast(scene, ob)
        pred = hit["t_hit"].numpy()
        measured = ob.depth
        near = cv2.erode(measured, np.ones((3, 3), "u1"))
        valid = np.isfinite(pred) & (measured > 0.45) & (measured < 10) & (ob.confidence > 0.65)
        idx = offsets[hit["geometry_ids"].numpy()[valid]] + hit["primitive_ids"].numpy()[valid]
        same = abs(pred[valid] - measured[valid]) < 0.10 + 0.009 * measured[valid]
        supported[idx[same]] = True
        ids = np.unique(idx[near[valid] > pred[valid] + 0.26 + 0.012 * pred[valid]])
        separate[ids] |= seen[ids] & (np.linalg.norm(first[ids] - ob.origin, axis=1) > 0.35)
        first[ids[~seen[ids]]] = ob.origin
        seen[ids] = True
    out = []
    for i, (mat, key, p, t, attr) in enumerate(parts):
        section = slice(offsets[i], offsets[i + 1])
        ids = np.flatnonzero(separate[section] & ~supported[section])
        remove = np.zeros(len(t), bool)
        if len(ids):
            tri = p[t[ids]]
            # Protect thin surveyed features even if a raster missed them in two views.
            d = tree.query(tri.mean(1), workers=3)[0]
            corners = tree.query(tri.reshape(-1, 3), workers=3)[0].reshape(-1, 3)
            remove[ids] = (d > 0.075) & (corners.max(1) > 0.09)
        if (~remove).any():
            out.append((mat, key, p, t[~remove], attr))
    return out, int(length - sum(len(g[3]) for g in out))


def corroborated(points, ob, others):
    yes = np.zeros(len(points), bool)
    no = np.zeros(len(points), bool)
    for other in others:
        baseline = np.linalg.norm(other.origin - ob.origin)
        if baseline < 0.35 or baseline > 7:
            continue
        u, v, z = project(points, other.frame)
        inside = (z > 0.3) & (u > 6) & (u < 1434) & (v > 6) & (v < 1074)
        ids = np.flatnonzero(inside)
        if not len(ids):
            continue
        x = (u[ids] / 4).astype(int)
        y = (v[ids] / 4).astype(int)
        distance = np.linalg.norm(points[ids] - other.origin, axis=1)
        measured = other.depth[y, x]
        usable = (measured > 0) & (other.confidence[y, x] > 0.7)
        yes[ids] |= usable & (abs(measured - distance) < 0.065 + 0.006 * distance)
        no[ids] |= usable & (other.nearest[y, x] > distance + 0.18 + 0.015 * distance)
    return yes & ~no


def patches(parts, observations, tree, compact):
    scene = ray_scene(parts)
    # Adjacent source views can see the same gap. Cast against accepted patches
    # too, so a later photograph cannot paint a second overlapping surface.
    stride = 5 if compact else 3
    yy, xx = np.mgrid[1 : H - 1 : stride, 1 : W - 1 : stride]
    grid = np.arange(xx.size).reshape(xx.shape)
    a, b, c, d = [g.ravel() for g in (grid[:-1, :-1], grid[1:, :-1], grid[:-1, 1:], grid[1:, 1:])]
    triangles = np.vstack([np.column_stack([a, b, c]), np.column_stack([b, d, c])])
    uv = np.column_stack([(xx.ravel() + 0.5) / W, (yy.ravel() + 0.5) / H])
    result = []
    for ob in observations:
        depth = ob.depth[yy, xx].ravel()
        pts = ob.origin + ob.rays[yy, xx].reshape(-1, 3) * depth[:, None]
        predicted = cast(scene, ob)["t_hit"].numpy()[yy, xx].ravel()
        valid = (depth > 0.55) & (depth < 9.5) & (ob.confidence[yy, xx].ravel() > 0.72)
        # Inspect the samples between grid vertices too: three stone corners
        # alone can straddle a true opening or a foreground silhouette.
        kernel = np.ones((stride, stride), "u1")
        near = cv2.erode(ob.depth, kernel)[yy, xx].ravel()
        far = cv2.dilate(ob.depth, kernel)[yy, xx].ravel()
        valid &= (far - near < 0.095 + 0.030 * depth) & (near > 0)
        gap = predicted > depth + 0.13 + 0.013 * depth
        faces = triangles[valid[triangles].all(1) & gap[triangles].any(1)]
        if not len(faces):
            continue
        span = np.ptp(depth[faces], axis=1)
        faces = faces[span < 0.065 + 0.021 * depth[faces].min(1)]
        if not len(faces):
            continue
        centers = pts[faces].mean(1)
        dist = scene.compute_distance(o3d.core.Tensor(centers), nthreads=3).numpy()
        keep = (dist > 0.018) & (dist < 0.85)
        faces, centers = faces[keep], centers[keep]
        if not len(faces):
            continue
        near = tree.query(centers, workers=3)[0]
        verts = tree.query(pts, workers=3)[0]
        supported = (near < 0.048) & (verts[faces].max(1) < 0.065)
        faces, centers = faces[supported], centers[supported]
        faces = faces[corroborated(centers, ob, observations)]
        if not len(faces):
            continue
        # Isolated triangles make visible speckles. Keep connected repairs
        # with several faces and at least 25 square centimetres of surface.
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(pts), o3d.utility.Vector3iVector(faces)
        )
        labels, counts, areas = mesh.cluster_connected_triangles()
        keep = (np.asarray(counts) >= (3 if compact else 5)) & (np.asarray(areas) >= 0.0025)
        faces = faces[keep[np.asarray(labels)]]
        if not len(faces):
            continue
        ids, local = np.unique(faces, return_inverse=True)
        fs = local.reshape(-1, 3).astype("u4")
        scene.add_triangles(o3d.core.Tensor(pts[ids]), o3d.core.Tensor(fs))
        result.append((ob, pts[ids], fs, uv[ids]))
    return result


def repair_colour(parts, doc, images, observations):
    """Find baked colours contradicted by two other real views of the same surface."""
    from evaluate_christ_church import sample

    sizes = [len(t) for _, _, _, t, _ in parts]
    offsets = np.r_[0, np.cumsum(sizes)]
    centers, normals, colors, plain = [], [], [], []
    for mat, key, p, tri, attr in parts:
        corners = p[tri]
        centers.append(corners.mean(1))
        n = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        normals.append(n / np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-9))
        mean = attr[tri].mean(1)
        if key == "TEXCOORD_0":
            ref = doc["materials"][mat]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
            blob = images[doc["textures"][ref]["source"]]
            photo = cv2.imdecode(np.frombuffer(blob, "u1"), 1)
            colors.append(
                sample(photo, mean[:, 0] * photo.shape[1] - 0.5, mean[:, 1] * photo.shape[0] - 0.5)
            )
        else:
            rgb = np.where(
                mean < 0.0031308, 12.92 * mean, 1.055 * np.maximum(mean, 0) ** (1 / 2.4) - 0.055
            )
            colors.append(np.clip(rgb[:, ::-1] * 255, 0, 255))
        plain.append(np.full(len(tri), key == "COLOR_0"))
    center, norm, old, plain = map(np.concatenate, (centers, normals, colors, plain))
    tree = cKDTree(center)
    scene = ray_scene(parts)
    num = len(center)
    scores = np.zeros((num, 4), "f4")
    sources = np.full((num, 4), -1, "i4")
    rgb = np.zeros((num, 4, 3), "f4")
    for j, ob in enumerate(observations):
        idx = np.asarray(tree.query_ball_point(ob.origin, 8), dtype=int)
        if not len(idx):
            continue
        xyz = center[idx]
        u, v, z = project(xyz, ob.frame)
        inside = (u > 10) & (u < 1429) & (v > 10) & (v < 1069) & (z > 0.4)
        idx, u, v = idx[inside], u[inside], v[inside]
        x, y = (u / 4).astype(int), (v / 4).astype(int)
        dist = np.linalg.norm(center[idx] - ob.origin, axis=1)
        angle = abs(np.sum(norm[idx] * (center[idx] - ob.origin), axis=1)) / dist
        seen = cast(scene, ob)["t_hit"].numpy()
        margin = 0.045 + 0.008 * dist
        ok = (
            (abs(ob.depth[y, x] - dist) < margin)
            & (abs(seen[y, x] - dist) < margin)
            & (ob.confidence[y, x] > 0.8)
            & (ob.nearest[y, x] > dist - margin - 0.04)
            & (angle > 0.38)
        )
        idx, u, v, angle, dist = idx[ok], u[ok], v[ok], angle[ok], dist[ok]
        rating = angle**1.5 / (0.7 + dist)
        slot = scores[idx].argmin(1)
        take = rating > scores[idx, slot]
        idx, slot = idx[take], slot[take]
        if len(idx):
            scores[idx, slot] = rating[take]
            sources[idx, slot] = j
            rgb[idx, slot] = sample(ob.photo, u[take], v[take])
    origins = np.array([o.origin for o in observations])
    choice = np.full(num, -1, int)
    best = np.zeros(num)
    for k in range(4):
        others = np.arange(4) != k
        agreement = np.mean(abs(rgb[:, k, None, :] - rgb[:, others]), axis=2)
        delta_old = np.mean(abs(old[:, None, :] - rgb[:, others]), axis=2)
        distinct = (
            np.linalg.norm(origins[sources[:, k, None]] - origins[sources[:, others]], axis=2) > 0.35
        )
        good = (
            (scores[:, others] > 0)
            & distinct
            & (agreement < 15)
            & ((delta_old > agreement + 14) | plain[:, None])
        )
        rating = scores[:, k] * np.sum(good, axis=1)
        use = (rating > best) & (good.sum(1) >= 2)
        choice[use] = sources[use, k]
        best[use] = rating[use]
    retained = []
    charts = []
    for i, (mat, key, p, t, attr) in enumerate(parts):
        labels = choice[offsets[i] : offsets[i + 1]].copy()
        for j in np.unique(labels[labels >= 0]):
            ob = observations[j]
            selected = np.flatnonzero(labels == j)
            ids, local = np.unique(t[selected], return_inverse=True)
            xyz = p[ids]
            u, v, z = project(xyz, ob.frame)
            x, y = np.clip((u / 4).astype(int), 0, W - 1), np.clip((v / 4).astype(int), 0, H - 1)
            dist = np.linalg.norm(xyz - ob.origin, axis=1)
            ok = (
                (z > 0.3)
                & (u > 4)
                & (v > 4)
                & (u < 1435)
                & (v < 1075)
                & (abs(ob.depth[y, x] - dist) < 0.07 + 0.009 * dist)
                & (ob.confidence[y, x] > 0.7)
            )
            local = local.reshape(-1, 3)
            good = ok[local].all(1)
            labels[selected[~good]] = -1
            if good.any():
                uv = np.column_stack([(u + 0.5) / 1440, (v + 0.5) / 1080])
                charts.append((ob, xyz, local[good], uv))
        if (labels < 0).any():
            retained.append((mat, key, p, t[labels < 0], attr))
    # One photograph may contribute to many original primitives. Collect it once
    # before atlas packing, so neighbouring repaired faces use the same image tile.
    combined = []
    for ob in observations:
        batches = [(p, t, uv) for o, p, t, uv in charts if o is ob]
        if batches:
            offsets = np.r_[0, np.cumsum([len(p) for p, _, _ in batches])]
            combined.append(
                (
                    ob,
                    np.vstack([p for p, _, _ in batches]),
                    np.vstack([t + n for (_, t, _), n in zip(batches, offsets)]),
                    np.vstack([uv for _, _, uv in batches]),
                )
            )
    return retained, combined


def pack_photographs(glb, additions, compact):
    """Pack just the visible patches; many photographs contribute only a small corner."""
    side, width, pad = 2048, (720 if compact else 1440), 4
    height = width * 3 // 4
    atlas = np.zeros((side, side, 3), "u1")
    vertices, faces, coords = [], [], []
    offset = x = y = row_height = page = 0

    def flush():
        used_height = y + row_height
        if not vertices:
            return
        uv = np.vstack(coords) / [side, used_height]
        glb.primitive(
            f"Automatic photo repairs {page+1}",
            np.vstack(vertices) @ UP.T,
            np.vstack(faces),
            uv,
            atlas[:used_height],
            jpeg_quality=92,
        )
        glb.doc["materials"][-1]["doubleSided"] = True

    for ob, pts, tris, uv in additions:
        photo = cv2.resize(ob.photo, (width, height), interpolation=cv2.INTER_AREA)
        pixel = uv * [width, height]
        center = pixel[tris].mean(1)
        cell = np.floor(center / 110).astype(int)
        keys = cell[:, 1] * (width // 110 + 1) + cell[:, 0]
        for key in np.unique(keys):
            ids, local = np.unique(tris[keys == key], return_inverse=True)
            low = np.maximum(0, np.floor(pixel[ids].min(0)).astype(int) - 1)
            high = np.minimum([width, height], np.ceil(pixel[ids].max(0)).astype(int) + 2)
            tile = cv2.copyMakeBorder(
                photo[low[1] : high[1], low[0] : high[0]], pad, pad, pad, pad, cv2.BORDER_REPLICATE
            )
            h, w = tile.shape[:2]
            if x + w > side:
                y += row_height
                x = row_height = 0
            if y + h > side:
                # A wrapped row has height zero, so y still records the used height.
                flush()
                vertices, faces, coords = [], [], []
                x = y = row_height = offset = 0
                page += 1
                atlas.fill(0)
            atlas[y : y + h, x : x + w] = tile
            vertices.append(pts[ids])
            faces.append(local.reshape(-1, 3) + offset)
            coords.append(pixel[ids] - low + [x + pad, y + pad])
            offset += len(ids)
            x += w
            row_height = max(row_height, h)
    flush()


def write_cleaned(source, target, observations, tree, compact):
    doc, images, parts = read_primitives(source, trim_flights=False)
    parts, removed = clean(parts, observations, tree)
    additions = patches(parts, observations, tree, compact)
    parts, repainted = repair_colour(parts, doc, images, observations)
    glb = Glb()
    glb.doc = doc
    doc.update(accessors=[], bufferViews=[], buffers=[])
    doc["meshes"][0]["primitives"] = []
    for img, blob in zip(doc["images"], images):
        img["bufferView"] = glb.view(blob)
    for material, key, p, t, attr in parts:
        ids, local = np.unique(t, return_inverse=True)
        attrs = {
            "POSITION": glb.acc((p[ids] @ UP.T).astype("f4"), True),
            key: glb.acc(attr[ids].astype("f4")),
        }
        doc["meshes"][0]["primitives"].append(
            dict(attributes=attrs, material=material, indices=glb.acc(local.ravel().astype("u4")))
        )
    pack_photographs(glb, additions + repainted, compact)
    report = dict(
        views=len(observations),
        removed=removed,
        added=sum(len(x[2]) for x in additions),
        retextured=sum(len(x[2]) for x in repainted),
    )
    doc["asset"].setdefault("extras", {})["surveyCleanup"] = report
    glb.write(target)
    print(target.name, report, flush=True)
    return report


def finish(data, source, proposal, site, font):
    cameras = camera_list(data)
    controls = control_views(cameras, source)
    results = {}
    for name in ("christ-church.glb", "christ-church-mobile.glb"):
        results[name] = verify(data, source, proposal / name, site / name, controls)
    scene, _ = glb_scene(site / "christ-church.glb", textures=False)
    maps = MeasuredVisibility(scene, data, cameras[0])
    try:
        with zipfile.ZipFile(data / "images.zip") as archive:
            write_views(site, archive, maps, cameras)
    finally:
        maps.archive.close()
    update_stats(site, json.loads((source / "stats.json").read_text())["stairRisers"])
    (site / "cleanup.json").write_text(json.dumps(results, indent=2) + "\n")
    viewer = Path(__file__).resolve().parents[1] / "room/site/christ-church"
    for name in ("index.html", "cleanup.html", "viewer.js", "viewer.js.LEGAL.txt"):
        shutil.copy2(viewer / name, site / name)
    return review(data, source, site, font)


def run(data, source, output, detector, font=DEFAULT_FONT):
    if output.exists() or source.resolve() in output.resolve().parents:
        raise ValueError("Use a new output directory outside the input site")
    _, observations, tree = prepare(data, source, detector)
    print("Prepared calibrated views", len(observations), flush=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        site = Path(tmp) / "site"
        proposal = Path(tmp) / "proposals"
        proposal.mkdir()
        shutil.copytree(source, site)
        for compact in (False, True):
            name = "christ-church-mobile.glb" if compact else "christ-church.glb"
            write_cleaned(source / name, proposal / name, observations, tree, compact)
        del observations, tree
        report = finish(data, source, proposal, site, font)
        if report["failures"]:
            rejected = output.with_name(output.name + ".rejected.json")
            shutil.copyfile(site / "cleanup-review.json", rejected)
            raise ValueError(f"Cleanup review failed. See {rejected}")
        site.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "source", "output", "detector"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    run(**vars(parser.parse_args()))
