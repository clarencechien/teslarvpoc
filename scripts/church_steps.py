"""Fit the two measured stair flights and bake a continuous photograph texture on each face."""

import argparse
import json
import struct
import zipfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

# Footprints bounded by the inside edges of the stone balustrades, in survey metres.
# The end elevations bound the search; individual nosings and tread planes are fitted to points.
FLIGHTS = [(-1.10, 2.10, 13.0, 17.65, -6.09, -4.44), (2.50, 6.12, 11.95, 17.67, -2.28, -4.44)]


def fit_flight(points, bounds):
    left, right, start, end, z0, z1 = bounds
    x, y, z = points.T
    height = np.interp(y, [start + 0.36, end - 0.36], [z0, z1])
    mask = (x > left + 0.09) & (x < right - 0.09) & (y > start) & (y < end) & (abs(z - height) < 0.36)
    x, y, z = points[mask].T
    bins = np.arange(start, end + 0.02, 0.02)
    heights = np.arange(min(z0, z1) - 0.22, max(z0, z1) + 0.24, 0.012)
    hist = np.histogram2d(y, z, bins=[bins, heights])[0]
    center = (bins[:-1] + bins[1:]) * 0.5
    h = heights[hist.argmax(1)] + 0.006
    supported = hist.max(1) > 2
    h = median_filter(np.interp(center, center[supported], h[supported]), size=3)
    peaks = find_peaks(abs(h[4:] - h[:-4]), height=0.08, distance=12)[0] + 2
    edges = np.r_[start, center[peaks], end]
    quads = []
    planes = []
    residuals = []
    for a, b in zip(edges, edges[1:]):
        level = np.median(h[(center > a + 0.06) & (center < b - 0.06)])
        select = (y > a + 0.06) & (y < b - 0.06) & (abs(z - level) < 0.048)
        design = np.column_stack([x[select], y[select], np.ones(select.sum())])
        target = z[select]
        if len(target) < 20:
            raise ValueError("Insufficient survey returns on a tread")
        plane = np.linalg.lstsq(design, target, rcond=None)[0]
        inliers = abs(design @ plane - target) < 0.024
        plane = np.linalg.lstsq(design[inliers], target[inliers], rcond=None)[0]
        planes.append(plane)
        residuals.extend(abs(design @ plane - target))
    ends = [np.array([start, start])]
    # Locate each vertical face in width-wise strips. The fit can follow a nosing
    # that is slightly oblique in the survey coordinate system.
    for i, e in enumerate(edges[1:-1]):
        xs = []
        ys = []
        for a, b in zip(np.linspace(left, right, 11)[:-1], np.linspace(left, right, 11)[1:]):
            xm = (a + b) / 2
            za, zb = [np.dot(plane, [xm, e, 1]) for plane in planes[i : i + 2]]
            lane = (x >= a) & (x < b) & (abs(y - e) < 0.12)
            face = lane & (abs(z - (za + zb) / 2) < 0.045)
            if face.sum() >= 4:
                xs.append(xm)
                ys.append(np.median(y[face]))
            else:
                lower = lane & (abs(z - za) < 0.028)
                upper = lane & (abs(z - zb) < 0.028)
                if lower.sum() > 3 and upper.sum() > 3:
                    xs.append(xm)
                    ys.append((np.percentile(y[lower], 95) + np.percentile(y[upper], 5)) / 2)
        if len(xs) < 3:
            raise ValueError("Insufficient survey returns at a stair edge")
        slope, offset = np.polyfit(xs, ys, 1)
        ends.append(offset + slope * np.array([left, right]))
    ends.append(np.array([end, end]))

    def line(loc, plane):
        xy = np.column_stack([[left, right], loc])
        return np.column_stack([xy, xy @ plane[:2] + plane[2]])

    for i, plane in enumerate(planes):
        a, b = line(ends[i], plane), line(ends[i + 1], plane)
        quads.append(np.array([a[0], a[1], b[1], b[0]]))
        if i + 1 < len(planes):
            c = line(ends[i + 1], planes[i + 1])
            quads.append(np.array([b[0], b[1], c[1], c[0]]))
    print(
        "Measured risers, tread fit P95 cm:",
        len(peaks),
        round(np.percentile(residuals, 95) * 100, 2),
        flush=True,
    )
    return np.asarray(quads, dtype="f4")


def clip_box(points, triangles, attributes, bounds):
    """Subtract a sloped prism, interpolating texture coordinates on cut edges."""
    left, right, start, end, z0, z1 = bounds
    slope = (z1 - z0) / (end - start - 0.72)
    intercept = z0 - slope * (start + 0.36)
    planes = np.array(
        [
            [1, 0, 0, -left],
            [-1, 0, 0, right],
            [0, 1, 0, -start],
            [0, -1, 0, end],
            [0, -slope, 1, 0.47 - intercept],
            [0, slope, -1, 0.42 + intercept],
        ]
    )
    return subtract_prism(points, triangles, attributes, planes)


def subtract_prism(points, triangles, attributes, planes):
    margin = points @ planes[:, :3].T + planes[:, 3]
    distances = margin[triangles]
    outside = (distances.max(1) < 0).any(1)
    boundary = np.flatnonzero(~outside & (distances.min((1, 2)) < 0))
    extra = []
    new_faces = []
    offset = len(points)
    for face in triangles[boundary]:
        poly = np.column_stack([points[face], attributes[face]])
        for plane in planes:
            d = poly[:, :3] @ plane[:3] + plane[3]
            if d.min() >= 0:
                continue
            if d.max() <= 0:
                keep = poly
                poly = []
            else:
                keep = []
                inside = []
                for j, corner in enumerate(poly):
                    k = (j + 1) % len(poly)
                    (inside if d[j] >= 0 else keep).append(corner)
                    if (d[j] > 0 and d[k] < 0) or (d[j] < 0 and d[k] > 0):
                        cut = corner + (poly[k] - corner) * (d[j] / (d[j] - d[k]))
                        keep.append(cut)
                        inside.append(cut)
                    elif d[j] == 0:
                        keep.append(corner)
                poly = np.asarray(inside)
            if len(keep) >= 3:
                extra.extend(keep)
                new_faces.extend([offset, offset + j, offset + j + 1] for j in range(1, len(keep) - 1))
                offset += len(keep)
            if len(poly) < 3:
                break
    faces = triangles[outside]
    if extra:
        new = np.array(extra)
        points = np.vstack([points, new[:, :3]])
        attributes = np.vstack([attributes, new[:, 3:]])
        faces = np.vstack([faces, np.asarray(new_faces)])
    used, local = np.unique(faces, return_inverse=True)
    return points[used], local.reshape(-1, 3), attributes[used]


def read_primitives(path, *, trim_flights=True):
    from build_christ_church import UP

    raw = path.read_bytes()
    length = struct.unpack_from("<I", raw, 12)[0]
    doc = json.loads(raw[20 : 20 + length])
    binary = memoryview(raw)[28 + length :]

    def view(i):
        item = doc["bufferViews"][i]
        offset = item.get("byteOffset", 0)
        return binary[offset : offset + item["byteLength"]]

    def attr(i, dtype, width):
        return np.frombuffer(view(doc["accessors"][i]["bufferView"]), dtype).reshape(-1, width)

    groups = []
    for group in doc["meshes"][0]["primitives"]:
        attrs = group["attributes"]
        points = attr(attrs["POSITION"], "<f4", 3) @ UP
        tri = attr(group["indices"], "<u4", 3)
        key = "TEXCOORD_0" if "TEXCOORD_0" in attrs else "COLOR_0"
        data = attr(attrs[key], "<f4", 2 if key == "TEXCOORD_0" else 3)
        if trim_flights:
            for bounds in FLIGHTS:
                points, tri, data = clip_box(points, tri, data, bounds)
        if len(tri):
            groups.append((group["material"], key, points, tri, data))
    photos = [bytes(view(img["bufferView"])) for img in doc["images"]]
    return doc, photos, groups


def quad_mesh(quads):
    positions = quads.reshape(-1, 3)
    faces = (np.arange(len(quads))[:, None, None] * 4 + np.array([[0, 1, 2], [0, 2, 3]])).reshape(-1, 3)
    return positions, faces


def bake_tiles(quads, frames, images, maps, survey):
    from build_christ_church import project

    width = 768
    pad = 4
    inner = width - 2 * pad
    tiles = []
    coordinates = []
    y_offset = 0
    for surface, quad in enumerate(quads):
        dx = quad[1] - quad[0]
        dy = quad[3] - quad[0]
        height = max(12, round(inner * np.linalg.norm(dy) / np.linalg.norm(dx)))
        u = (np.arange(inner, dtype="f4") + 0.5) / inner
        v = (np.arange(height, dtype="f4") + 0.5) / height
        near = quad[0] + u[:, None] * dx
        far = quad[3] + u[:, None] * (quad[2] - quad[3])
        points = (near[None, :, :] + v[:, None, None] * (far - near)[None, :, :]).reshape(-1, 3)
        normal = np.cross(dx, dy)
        normal /= np.linalg.norm(normal)
        scores = []
        for i, frame in enumerate(frames):
            probes = np.vstack([quad, quad.mean(0)])
            x, y, z = project(probes, frame)
            inside = (z > 0.18) & (x >= 0) & (x < 1440) & (y >= 0) & (y < 1080)
            if not inside.any():
                continue
            rays = np.asarray(frame["transform_matrix"])[:3, 3] - probes[inside]
            dist = np.linalg.norm(rays, axis=1)
            score = np.max((rays @ normal / dist) / (1 + 0.18 * dist) ** 2)
            if score > 0.001:
                scores.append((score, i))
        observations = []
        weights = []
        for score, i in sorted(scores, reverse=True)[:16]:
            frame = frames[i]
            x, y, z = project(points, frame)
            border = np.minimum.reduce([x, y, 1439 - x, 1079 - y])
            depth = maps.depth(frame)[
                np.clip((y / 6).astype(int), 0, 179), np.clip((x / 6).astype(int), 0, 239)
            ]
            distance = np.linalg.norm(points - np.asarray(frame["transform_matrix"])[:3, 3], axis=1)
            error = abs(depth - distance)
            tol = 0.095 + 0.012 * distance
            weight = (
                score * np.clip(border / 65, 0, 1) * np.clip(1 - error / tol, 0, 1) * (z > 0.15)
            ).astype("f4")
            if not weight.any():
                continue
            img = images(i)
            sample = cv2.remap(
                img,
                x.reshape(height, inner).astype("f4"),
                y.reshape(height, inner).astype("f4"),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ).reshape(-1, 3)
            observations.append(sample)
            weights.append(weight)
        if weights:
            obs = np.asarray(observations, dtype="f4")
            w = np.asarray(weights)
            seen = w.max(0) > 0
            # Suppress pedestrians / specular outliers before feathering overlapping images.
            median = np.zeros_like(obs[0])
            median[seen] = np.nanmedian(np.where(w[:, seen, None] > 0, obs[:, seen], np.nan), axis=0)
            deviation = abs(obs - median).mean(2)
            w *= 1 / (1 + (deviation / 19) ** 4)
            result = (obs * w[:, :, None]).sum(0) / np.maximum(w.sum(0)[:, None], 1e-10)
        else:
            seen = np.zeros(len(points), bool)
            result = np.zeros((len(points), 3), "f4")
        print("Texture coverage", surface, round(seen.mean(), 3), flush=True)
        if (~seen).any():
            tree, rgb = survey
            _, index = tree.query(points[~seen], workers=4)
            result[~seen] = rgb[index, ::-1]
        tile = np.clip(result, 0, 255).astype("u1").reshape(height, inner, 3)
        tiles.append(cv2.copyMakeBorder(tile, pad, pad, pad, pad, cv2.BORDER_REPLICATE))
        coordinates.extend(
            [
                [pad, y_offset + pad],
                [width - pad, y_offset + pad],
                [width - pad, y_offset + pad + height],
                [pad, y_offset + pad + height],
            ]
        )
        y_offset += height + 2 * pad
    atlas = np.vstack(tiles)
    return np.asarray(coordinates) / [width, y_offset], atlas


def update_stats(site, riser_counts):
    path = site / "stats.json"
    stats = json.loads(path.read_text())
    for filename, field in [
        ("christ-church.glb", "triangles"),
        ("christ-church-mobile.glb", "mobileTriangles"),
    ]:
        with (site / filename).open("rb") as stream:
            header = stream.read(20)
            doc = json.loads(stream.read(struct.unpack_from("<I", header, 12)[0]))
        primitives = doc["meshes"][0]["primitives"]
        counts = [doc["accessors"][p["indices"]]["count"] // 3 for p in primitives]
        stats[field] = sum(counts)
        if field == "triangles":
            textured = sum(n for n, p in zip(counts, primitives) if "TEXCOORD_0" in p["attributes"])
            stats["photoCoverage"] = round(textured / stats[field], 4)
    stats["stairRisers"] = riser_counts
    path.write_text(json.dumps(stats))


def refine_stairs(data, site):
    import open3d as o3d
    from build_room import Glb
    from build_christ_church import UP, MeasuredVisibility, read_photo, write_views
    from church_geometry import survey_colors

    frames = sorted(
        json.loads((data / "transforms_colmap_scaled.json").read_text())["frames"],
        key=lambda f: f["file_path"],
    )
    cameras = [[f for f in frames if f"cam{i}_" in f["file_path"]] for i in range(3)]
    survey = survey_colors(data, cameras[0])
    quads = [fit_flight(survey[0].data, b) for b in FLIGHTS]
    sources = json.loads((site / "scene.json").read_text())["views"]
    sources = [cameras[v["id"] // 1000][v["id"] % 1000] for v in sources if v.get("detail", True)]
    sources = [f for f in sources if 1710755315 < float(Path(f["file_path"]).stem) < 1710755420]
    with zipfile.ZipFile(data / "images.zip") as archive:

        @lru_cache(maxsize=24)
        def images(i):
            return read_photo(archive, sources[i])

        for compact in [False, True]:
            path = site / ("christ-church-mobile.glb" if compact else "christ-church.glb")
            doc, photos, groups = read_primitives(path)
            if any(m["name"].startswith("Measured stair flight") for m in doc["materials"]):
                raise ValueError("Input already contains fitted flights; use an unrefined build")
            if not compact:
                scene = o3d.t.geometry.RaycastingScene(nthreads=4)
                for _, _, p, t, _ in groups:
                    scene.add_triangles(
                        o3d.core.Tensor(p.astype("f4")), o3d.core.Tensor(t.astype("u4"))
                    )
                for flight in quads:
                    p, t = quad_mesh(flight)
                    scene.add_triangles(o3d.core.Tensor(p), o3d.core.Tensor(t.astype("u4")))
                maps = MeasuredVisibility(scene, data, cameras[0])
                tiles = [bake_tiles(f, sources, images, maps, survey) for f in quads]
            glb = Glb()
            glb.doc = doc
            glb.doc.update(accessors=[], bufferViews=[], buffers=[])
            glb.doc["meshes"][0]["primitives"] = []
            for img, blob in zip(doc["images"], photos):
                img["bufferView"] = glb.view(blob)
            for mat, key, p, t, attribute in groups:
                attrs = {
                    "POSITION": glb.acc((p @ UP.T).astype("f4"), True),
                    key: glb.acc(attribute.astype("f4")),
                }
                glb.doc["meshes"][0]["primitives"].append(
                    dict(attributes=attrs, material=mat, indices=glb.acc(t.ravel().astype("u4")))
                )
            for i, (flight, (uv, atlas)) in enumerate(zip(quads, tiles)):
                p, t = quad_mesh(flight)
                texture = (
                    cv2.resize(atlas, None, fx=0.65, fy=0.65, interpolation=cv2.INTER_AREA)
                    if compact
                    else atlas
                )
                glb.primitive(f"Measured stair flight {i+1}", p @ UP.T, t, uv, texture, jpeg_quality=94)
                glb.doc["materials"][-1]["doubleSided"] = True
            glb.write(path)
        write_views(site, archive, maps, cameras)
        maps.archive.close()
    update_stats(site, [len(flight) // 2 for flight in quads])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["data", "site"]:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    refine_stairs(args.data, args.site)
