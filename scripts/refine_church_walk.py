"""Build a calibrated walk layer, recovering transient foregrounds and matching exposure."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import hashlib
import shutil
import tempfile
import zipfile

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import lsq_linear

from build_christ_church import UP, camera_rays, project, read_photo
from build_church_photowalk import W, H, measured_depth, people_visibility
from church_geometry import survey_samples
from church_rail_repair import camera_list


@dataclass
class View:
    ident: int
    frame: dict
    original: np.ndarray
    depth: np.ndarray
    people: np.ndarray
    origin: np.ndarray
    direction: np.ndarray
    rays: np.ndarray


def prepare(data, site, detector, cache):
    cameras = camera_list(data)
    scene = json.loads((site / "scene.json").read_text())
    original = {v["id"] for v in scene["views"] if v.get("detail", True)}
    atlas = {
        c * 1000 + i
        for c, cam in enumerate(cameras)
        for i, f in enumerate(cam)
        if i % 4 == 0 and 1710755131 < float(Path(f["file_path"]).stem) < 1710755444
    }
    ids = sorted(original | atlas)
    net = cv2.dnn.readNet(str(detector))
    cv2.setNumThreads(2)
    tree = None
    views = []
    yy, xx = np.mgrid[:H, :W]
    with zipfile.ZipFile(data / "images.zip") as archive:
        for ident in ids:
            frame = cameras[ident // 1000][ident % 1000]
            pose = np.asarray(frame["transform_matrix"], "f4")
            path = cache / f"{ident}.npz"
            photo_path = site / f"photo-{ident}.webp"
            photo = cv2.imread(str(photo_path)) if photo_path.exists() else read_photo(archive, frame)
            if path.exists():
                with np.load(path) as raw:
                    depth = raw["depth"]
                    mask = raw["mask"]
            else:
                old = site / "photo-walk" / f"{ident}.png"
                if old.exists():
                    packed = cv2.imread(str(old)).astype("f4")
                    depth = (packed[:, :, 2] * 256 + packed[:, :, 1]) * 0.001
                else:
                    if tree is None:
                        pts, _, _ = survey_samples(data, cameras[0], 6)
                        tree = cKDTree(pts)
                    inds = tree.query_ball_point(pose[:3, 3], 16)
                    depth = measured_depth(pts[inds], frame)
                mask = (people_visibility(photo, net, (H, W)) < 0.92).astype("u1")
                with tempfile.NamedTemporaryFile(dir=cache, suffix=".npz", delete=False) as file:
                    np.savez_compressed(file, depth=depth, mask=mask)
                Path(file.name).replace(path)
            rays = (camera_rays(frame, xx * 4 + 2, yy * 4 + 2) @ pose[:3, :3].T).reshape(H, W, 3)
            views.append(View(ident, frame, photo, depth, mask, pose[:3, 3], -pose[:3, 2], rays))
    return views, original, atlas


def neighbours(views, view, limit):
    def score(b):
        return np.linalg.norm(view.origin - b.origin) + 2.6 * (1 - view.direction @ b.direction)

    sources = [
        b
        for b in views
        if b is not view
        and np.linalg.norm(b.origin - view.origin) < 10
        and abs(float(Path(b.frame["file_path"]).stem) - float(Path(view.frame["file_path"]).stem))
        > 0.32
    ]
    return sorted(sources, key=score)[:limit]


def observation(b, world):
    u, v, z = project(world, b.frame)
    x = np.clip((u / 4).astype(int), 0, W - 1)
    y = np.clip((v / 4).astype(int), 0, H - 1)
    dist = np.linalg.norm(world - b.origin, axis=1)
    usable = (z > 0.15) & (u > 5) & (v > 5) & (u < 1434) & (v < 1074)
    delta = abs(b.depth[y, x] - dist)
    return u, v, usable, delta, dist, b.people[y, x]


def lookup(img, u, v):
    # Bilinear sampling in blocks also works for longer flattened arrays than cv2.remap.
    out = []
    for i in range(0, len(u), 24000):
        out.append(
            cv2.remap(
                img,
                u[i : i + 24000, None].astype("f4"),
                v[i : i + 24000, None].astype("f4"),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )[:, 0]
        )
    return np.concatenate(out) if out else np.empty((0, 3), "u1")


def match_light(views):
    """Fit bounded per-channel gains using many views of the same diffuse surfaces."""
    ids = {v.ident: i for i, v in enumerate(views)}
    yy, xx = np.mgrid[6 : H - 6 : 7, 6 : W - 6 : 7]
    edges = []
    observed = []
    pairs = set()
    for i, a in enumerate(views):
        take = (a.people[yy, xx] == 0) & (a.depth[yy, xx] > 0.5)
        y, x = yy[take], xx[take]
        world = a.origin + a.rays[y, x] * a.depth[y, x, None]
        rgb = a.original[4 * y + 2, 4 * x + 2].astype(float)
        for b in neighbours(views, a, 9):
            j = ids[b.ident]
            key = tuple(sorted((i, j)))
            if key in pairs:
                continue
            pairs.add(key)
            u, v, valid, delta, dist, mask = observation(b, world)
            other = lookup(b.original, u, v).astype(float)
            valid &= (
                (delta < 0.055 + 0.009 * dist)
                & (mask == 0)
                & (rgb.min(1) > 18)
                & (rgb.max(1) < 232)
                & (other.min(1) > 18)
                & (other.max(1) < 232)
            )
            if valid.sum() < 50:
                continue
            # Work in display-referred log RGB. Reject inconsistent correspondences.
            ratio = np.log(rgb[valid] + 1) - np.log(other[valid] + 1)
            med = np.median(ratio, axis=0)
            good = abs(ratio - med).max(1) < 0.13
            if good.sum() < 35:
                continue
            edges.append((i, j))
            observed.append(np.median(ratio[good], axis=0))
    if not edges:
        raise ValueError("No overlapping surfaces for exposure calibration")
    rows = np.arange(len(edges))
    i, j = np.asarray(edges).T
    A = np.zeros((len(edges), len(views)))
    A[rows, i] = -1
    A[rows, j] = 1
    d = np.array(observed)
    weight = np.ones(len(d))
    for _ in range(3):
        lhs = np.vstack([A * weight[:, None], 0.3 * np.eye(len(views))])
        rhs = np.vstack([d * weight[:, None], np.zeros((len(views), 3))])
        g = np.column_stack([lsq_linear(lhs, rhs[:, c], bounds=(-0.24, 0.24)).x for c in range(3)])
        residual = A @ g - d
        weight = 1 / np.sqrt(1 + (abs(residual).max(1) / 0.05) ** 2)
    return np.exp(g), dict(
        pairs=len(edges), before=float(np.median(abs(d))), after=float(np.median(abs(A @ g - d)))
    )


def candidates(views, gains, a, x, y, depth, donors, *, spatial=False):
    """Score observed backgrounds after compensating range and exposure differences."""
    world = a.origin + a.rays[y, x] * depth[:, None]
    n = len(x)
    colors = np.empty((len(donors), n, 3), "f4")
    valid = np.empty((len(donors), n), bool)
    reliable = np.zeros((len(donors), n), bool)
    static_votes = np.zeros(n, "u1")
    target = a.original[y * 4 + 2, x * 4 + 2].astype("f4")
    for i, b in enumerate(donors):
        u, v, inside, delta, dist, mask = observation(b, world)
        color = lookup(b.original, u, v).astype("f4") * (gains[b.ident] / gains[a.ident])
        colors[i] = color
        far = (a.depth[y, x] == 0) & (depth >= 24)
        observed = delta < 0.13 + 0.017 * dist
        good = inside & (observed | far)
        reliable[i] = inside & observed & (mask == 0)
        valid[i] = good & (mask == 0)
        if np.linalg.norm(b.origin - a.origin) > 0.4:
            static_votes += good & observed & (abs(color - target).max(1) < 12)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(np.where(valid[:, :, None], colors, np.nan), axis=0)
    err = abs(colors - np.nan_to_num(median)).mean(2)
    agree = valid & (err < 19)
    # Prefer a nearby real photograph amongst the views supporting the median.
    cost = np.where(valid, err + np.arange(len(donors))[:, None] * 0.35, 1e5)
    if spatial:
        region = np.zeros((H, W), "f4")
        region[y, x] = 1
        count = np.maximum(0.001, cv2.boxFilter(region, -1, (7, 7))[y, x])
        base = cost.min(0)
        for i in range(len(donors)):
            region[y, x] = np.clip(cost[i] - base, 0, 44)
            local = cv2.boxFilter(region, -1, (7, 7))[y, x] / count
            cost[i] = np.where(valid[i], 0.42 * cost[i] + 0.58 * (base + local), 1e5)
    best = cost.argmin(0)
    support = agree.sum(0)
    energy = (
        np.where(agree, err, 0).sum(0) / np.maximum(support, 1)
        + 8 / np.maximum(support, 1)
        + 30 * (support < 2)
        + 40 * (support == 0)
    )
    return best.astype("i2"), energy, support, static_votes, reliable.any(0)


def model_samples(model, a, px, py):
    import open3d as o3d
    from evaluate_christ_church import sample

    scene, materials = model
    rays = camera_rays(a.frame, px, py) @ np.asarray(a.frame["transform_matrix"], "f4")[:3, :3].T
    hit = scene.cast_rays(
        o3d.core.Tensor(np.column_stack([np.broadcast_to(a.origin, rays.shape), rays]).astype("f4")),
        nthreads=3,
    )
    distance = hit["t_hit"].numpy()
    groups = hit["geometry_ids"].numpy()
    faces = hit["primitive_ids"].numpy()
    bary = hit["primitive_uvs"].numpy()
    rgb = np.zeros((len(px), 3), "f4")
    for group in np.unique(groups[np.isfinite(distance)]):
        selected = groups == group
        t, attr, texture = materials[group]
        uv = bary[selected]
        coords = (attr[t[faces[selected]]] * np.column_stack([1 - uv.sum(1), uv])[:, :, None]).sum(1)
        if texture is not None:
            rgb[selected] = sample(
                texture, coords[:, 0] * texture.shape[1] - 0.5, coords[:, 1] * texture.shape[0] - 0.5
            )
        else:
            rgb[selected] = (
                255
                * np.where(
                    coords < 0.0031308,
                    12.92 * coords,
                    1.055 * np.maximum(coords, 0) ** (1 / 2.4) - 0.055,
                )[:, ::-1]
            )
    return rgb, distance


def transient_shadow(a, donors, gains):
    """Extend a person mask only onto adjacent shadows missing from other real views."""
    mask = a.people
    spread = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (43, 39)))
    y, x = np.nonzero((spread != 0) & (mask == 0) & (a.depth > 0.45))
    result = np.zeros_like(mask)
    if not len(x):
        return result
    xyz = a.origin + a.rays[y, x] * a.depth[y, x, None]
    source = a.original[4 * y + 2, 4 * x + 2].astype("f4")
    observations = []
    for b in donors[:14]:
        u, v, inside, delta, dist, person = observation(b, xyz)
        rgb = lookup(b.original, u, v) * gains[b.ident] / gains[a.ident]
        good = inside & (delta < 0.065 + 0.012 * dist) & (person == 0)
        observations.append(np.where(good[:, None], rgb, np.nan))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = np.nanmedian(observations, axis=0)
    matched = (abs(observations - ref).mean(2) < 15).sum(0)
    brightness = np.array([0.114, 0.587, 0.299])
    src = source @ brightness
    dst = ref @ brightness
    ratio = src / np.maximum(dst, 1)
    chroma = abs(source - ratio[:, None] * ref).max(1)
    dark = (matched >= 3) & (ratio < 0.79) & (ratio > 0.18) & (dst - src > 11) & (chroma < 12)
    result[y[dark], x[dark]] = 1
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, np.ones((3, 3), "u1"))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(result, 8)
    touches = np.unique(labels[cv2.dilate(mask, np.ones((7, 7), "u1")) != 0])
    keep = np.isin(np.arange(count), touches) & (stats[:, cv2.CC_STAT_AREA] >= 10)
    keep[0] = False
    return keep[labels].astype("u1")


def correct(a, views, gains, net, model=None):
    """Refine proposed background range in stages. Always resample original donors."""
    mask = a.people.copy()
    donors = neighbours(views, a, 24)
    light = gains[a.ident]
    output = np.clip(a.original * light, 0, 255).astype("u1")
    confidence = np.ones((H, W), "f4")
    steps = []
    evidence = unknown = 0
    geometry = a.depth.copy()
    if not mask.any() or len(donors) < 2:
        confidence[mask != 0] = 0
        return (
            output,
            geometry,
            confidence,
            dict(maskPixels=int(mask.sum()), rounds=[], supported=0, unknown=int(mask.sum())),
        )
    shadow = transient_shadow(a, donors, gains)
    mask |= shadow
    for sweep in range(2):
        y, x = np.nonzero(mask)
        if not len(x):
            break
        pool = donors[:14] if sweep == 0 else donors
        z = np.where(a.depth[y, x] > 0.35, a.depth[y, x], 30.0).astype("f4")
        winner, err, support, stable, measured = candidates(views, gains, a, x, y, z, pool)
        protect = stable >= 3
        for increment in (0.16, 0.055):
            start = z.copy()
            for sign in (-1, 1):
                trial = start + increment * sign
                if np.all(protect):
                    break
                k, e, v, _, obs = candidates(views, gains, a, x, y, trial, pool)
                take = (e + 1.25 < err) & (v >= 2) & (trial > 0.4) & (a.depth[y, x] > 0.35) & ~protect
                z[take] = trial[take]
                winner[take] = k[take]
                err[take] = e[take]
                support[take] = v[take]
                measured[take] = obs[take]
        winner, _, support, _, measured = candidates(views, gains, a, x, y, z, pool, spatial=True)
        consistent = np.zeros((H, W), "u1")
        consistent[y, x] = protect
        consistent = cv2.morphologyEx(consistent, cv2.MORPH_OPEN, np.ones((3, 3), "u1"))
        protect = consistent[y, x] > 0
        verified = (support >= 2) & ~protect
        usable = (support > 0) & ~protect
        ids = np.full((H, W), -1, "i2")
        ids[y[usable], x[usable]] = winner[usable]
        zmap = geometry.copy()
        zmap[y, x] = z
        repaired = output.copy()
        assigned = cv2.resize(ids, (1440, 1080), interpolation=cv2.INTER_NEAREST)
        full_depth = cv2.resize(zmap, (1440, 1080), interpolation=cv2.INTER_NEAREST)
        for i, b in enumerate(pool):
            py, px = np.nonzero(assigned == i)
            if not len(px):
                continue
            rays = camera_rays(a.frame, px, py) @ np.asarray(a.frame["transform_matrix"], "f4")[:3, :3].T
            pos = a.origin + rays * full_depth[py, px, None]
            u, v, inside, _, _, m = observation(b, pos)
            good = inside & (m == 0)
            repaired[py[good], px[good]] = np.clip(
                lookup(b.original, u[good], v[good]) * gains[b.ident], 0, 255
            )
            assigned[py[~good], px[~good]] = -2
        holes = np.zeros((H, W), "u1")
        holes[y[~(usable | protect)], x[~(usable | protect)]] = 1
        holes = cv2.resize(holes, (1440, 1080), interpolation=cv2.INTER_NEAREST)
        holes[assigned == -2] = 1
        if holes.any():
            # Only these unresolved pixels use a local appearance estimate; they
            # receive no measured depth or projection weight from inpainting.
            small = cv2.resize(repaired, (720, 540), interpolation=cv2.INTER_AREA)
            bad = cv2.dilate(
                cv2.resize(holes, (720, 540), interpolation=cv2.INTER_AREA), np.ones((3, 3), "u1")
            )
            painted = cv2.resize(cv2.inpaint(small, bad, 4, cv2.INPAINT_TELEA), (1440, 1080))
            repaired[holes != 0] = painted[holes != 0]
        if model is not None:
            # A continuous photographed tread or wall supplies a stable reference
            # where individual donors disagree. Keep rejected geometry out of it.
            update = np.zeros((H, W), "u1")
            update[y[~protect], x[~protect]] = 1
            fy, fx = np.nonzero(cv2.resize(update, (1440, 1080), interpolation=cv2.INTER_NEAREST))
            if len(fx):
                base, md = model_samples(model, a, fx, fy)
                sy, sx = fy // 4, fx // 4
                valid = (abs(md - geometry[sy, sx]) < 0.12 + 0.015 * geometry[sy, sx]) & (
                    geometry[sy, sx] > 0.3
                )
                ref = repaired[fy, fx].astype("f4")
                # Use the mesh only when the original transient foreground is
                # photometrically distinct, or a donor corroborates its colour.
                base = np.clip(base * light, 0, 255)
                olddiff = abs(base - output[fy, fx]).mean(1)
                disagreement = abs(base - ref).mean(1)
                valid &= (olddiff > 10) | (disagreement < 12)
                weight = valid * np.clip(0.45 + disagreement / 48, 0, 1)
                repaired[fy, fx] = np.clip(ref + weight[:, None] * (base - ref), 0, 255)
        strength = np.zeros((H, W), "f4")
        strength[y[~protect], x[~protect]] = 1
        blend = cv2.GaussianBlur(cv2.resize(strength, (1440, 1080)), (5, 5), 0.95)[:, :, None]
        proposal = np.clip(repaired * blend + output * (1 - blend), 0, 255).astype("u1")
        remaining = (people_visibility(proposal, net, (H, W)) < 0.92).astype("u1")
        # Persistent surveyed colour (e.g. a painted portrait) is exempt from
        # the repeat pass, while newly revealed detections get fresh proposals.
        remaining[y[protect], x[protect]] = 0
        before = int(mask.sum())
        after = int(remaining.sum())
        accept = sweep == 0 or after < before
        steps.append(dict(before=before, after=after, accepted=accept))
        if not accept:
            break
        output = proposal
        confidence[y[~protect], x[~protect]] = np.where(
            verified[~protect], 0.96, np.where(usable[~protect], 0.42, 0.06)
        )
        geo = verified & measured & (a.depth[y, x] > 0.35)
        geometry[y[geo], x[geo]] = z[geo]
        evidence += int(verified.sum())
        unknown += int((~(usable | protect)).sum())
        mask = remaining
    return (
        output,
        geometry,
        confidence,
        dict(
            maskPixels=int(a.people.sum()),
            rounds=steps,
            supported=evidence,
            unknown=unknown,
            shadowPixels=int(shadow.sum()),
        ),
    )


def inspect_people(image, net):
    """Repeat detection at a closer scale to catch small figures missed in the wide view."""
    mask = (people_visibility(image, net, (H, W)) < 0.92).astype("u1")
    for left in (0, 120):
        crop = image[180:900, 4 * left : 4 * left + 960]
        region = (people_visibility(crop, net, (180, 240)) < 0.92).astype("u1")
        mask[45:225, left : left + 240] |= region
    return mask


def review_people(view, views, gains, net, model, photo, depth, confidence, stats):
    residual = inspect_people(photo, net)
    original = inspect_people(view.original, net)
    stats["reviewBefore"] = int(original.sum())
    stats["reviewAfter"] = int(residual.sum())
    if not residual.any():
        return photo, depth, confidence, stats
    # Keep static painted/decorative figures if several different positions
    # see the same colour on the measured wall.
    y, x = np.nonzero(residual)
    sources = neighbours(views, view, 14)
    if len(sources) < 3:
        return photo, depth, confidence, stats
    z = np.maximum(view.depth[y, x], 0.4)
    _, _, _, stable, _ = candidates(views, gains, view, x, y, z, sources)
    if (stable >= 3).mean() > 0.70:
        return photo, depth, confidence, stats
    from dataclasses import replace

    expanded = replace(view, people=(view.people | residual))
    candidate, d, c, detail = correct(expanded, views, gains, net, model)
    final = int(inspect_people(candidate, net).sum())
    if final < 0.8 * int(residual.sum()) and detail["supported"] >= 0.25 * residual.sum():
        detail.update(
            maskPixels=int(view.people.sum()),
            reviewBefore=int(original.sum()),
            reviewAfter=final,
            closerReview=dict(proposed=int(residual.sum()), accepted=True),
        )
        return candidate, d, c, detail
    stats["closerReview"] = dict(proposed=int(residual.sum()), accepted=False)
    return photo, depth, confidence, stats


def select_sources(views, original, budget=40):
    """Add the worst-covered registered poses until the view budget is exhausted."""
    active = [v for v in views if v.ident in original]
    left = [v for v in views if v.ident not in original]

    def error(v, b):
        return np.sum((v.origin - b.origin) ** 2) + 3 * (1 - v.direction @ b.direction)

    gaps = np.array([min(error(v, b) for b in active) for v in left])
    for _ in range(min(budget, len(left))):
        i = gaps.argmax()
        if gaps[i] < 0.25:
            break
        pick = left[i]
        active.append(pick)
        gaps = np.minimum(gaps, [error(v, pick) for v in left])
    return {v.ident for v in active}


def audit_light(views, gains, example=None, font=None):
    # Check different pixels in camera pairs excluded from the exposure fit.
    fitted = {tuple(sorted((a.ident, b.ident))) for a in views for b in neighbours(views, a, 9)}
    checked = set()
    pairs = []
    errors = []
    yy, xx = np.mgrid[3 : H - 5 : 11, 4 : W - 5 : 11]

    def luma(color):
        s = np.clip(color / 255, 0, 1)
        linear = np.where(s < 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
        return np.maximum(0.002, linear @ np.array([0.0722, 0.7152, 0.2126]))

    for a in views:
        take = (a.people[yy, xx] == 0) & (a.depth[yy, xx] > 0.5)
        y, x = yy[take], xx[take]
        xyz = a.origin + a.rays[y, x] * a.depth[y, x, None]
        left = a.original[4 * y + 2, 4 * x + 2].astype("f4")
        for b in neighbours(views, a, 20)[9:]:
            key = tuple(sorted((a.ident, b.ident)))
            if key in fitted or key in checked:
                continue
            u, v, ok, delta, dist, mask = observation(b, xyz)
            right = lookup(b.original, u, v).astype("f4")
            ok &= (
                (delta < 0.055 + 0.009 * dist)
                & (mask == 0)
                & (left.min(1) > 18)
                & (left.max(1) < 232)
                & (right.min(1) > 18)
                & (right.max(1) < 232)
            )
            if ok.sum() < 35:
                continue
            before = np.log2(luma(left[ok]) / luma(right[ok]))
            after = np.log2(luma(left[ok] * gains[a.ident]) / luma(right[ok] * gains[b.ident]))
            errors.append(
                [
                    abs(np.median(before)),
                    abs(np.median(after)),
                    np.median(abs(before)),
                    np.median(abs(after)),
                ]
            )
            checked.add(key)
            pairs.append((a, b))
            break
    if not errors:
        raise ValueError("No independent exposure comparisons")
    before, after, pixel_before, pixel_after = np.median(errors, axis=0)
    result = dict(
        pairs=len(errors),
        beforeStops=float(before),
        afterStops=float(after),
        pixelMedianBeforeStops=float(pixel_before),
        pixelMedianAfterStops=float(pixel_after),
    )
    if example is not None:
        from review_church_walk import exposure_example

        # Select by original discrepancy and overlap alone. The improved result
        # is not a selection criterion.
        initial = np.asarray(errors)[:, 0]
        for index in np.argsort(abs(initial - np.percentile(initial, 75))):
            a, b = pairs[index]
            if not (174 <= a.ident % 1000 <= 210):
                continue
            crop = exposure_example(a, b, gains, example, font)
            if crop is not None:
                result["example"] = dict(views=[a.ident, b.ident], crop=crop)
                break
        else:
            raise ValueError("No independent pair has enough visible overlap for the comparison")
    return result


def build(data, site, detector, cache, dest, tiles, font):
    cache.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    views, original, atlas = prepare(data, site, detector, cache)
    gains, lighting = match_light(views)
    gains = {v.ident: g for v, g in zip(views, gains)}
    lighting["independent"] = audit_light(views, gains, dest / "lighting-comparison.webp", font)
    active = select_sources(views, original)
    net = cv2.dnn.readNet(str(detector))
    metadata = []
    report = {}
    from evaluate_christ_church import glb_scene
    from church_walk_seams import refine_boundary

    model = glb_scene(site / "christ-church.glb")
    for n, v in enumerate(views):
        photo, depth, conf, stats = correct(v, views, gains, net, model)
        # Inspect the compressed pixels that will actually be delivered.
        quality = 93 if v.ident in active else 94
        ok, blob = cv2.imencode(".webp", photo, [cv2.IMWRITE_WEBP_QUALITY, quality])
        if not ok:
            raise OSError("Cannot encode source photo")
        preview = cv2.imdecode(blob, 1)
        reviewed, depth, conf, stats = review_people(
            v, views, gains, net, model, preview, depth, conf, stats
        )
        if stats.get("closerReview", {}).get("accepted"):
            photo = reviewed
        photo = refine_boundary(v, views, gains, model, net, photo, stats)
        report[v.ident] = stats
        if v.ident in atlas:
            cv2.imwrite(str(tiles / f"clean-{v.ident}.webp"), photo, [cv2.IMWRITE_WEBP_QUALITY, 94])
        if v.ident not in active:
            continue
        cv2.imwrite(str(dest / f"{v.ident}.webp"), photo, [cv2.IMWRITE_WEBP_QUALITY, 93])
        mm = np.rint(np.clip(depth, 0, 65) * 1000).astype("u2")
        packed = np.stack([np.rint(conf * 255), mm % 256, mm // 256], axis=-1).astype("u1")
        cv2.imwrite(str(dest / f"{v.ident}.png"), packed)
        pose = np.asarray(v.frame["transform_matrix"]).copy()
        pose[:3] = UP @ pose[:3]
        metadata.append(
            dict(
                id=v.ident,
                pose=pose.tolist(),
                lens=[v.frame[k] for k in ("fl_x", "fl_y", "cx", "cy")],
                distortion=[v.frame["k" + str(k)] for k in range(1, 5)],
            )
        )
        if n % 30 == 0:
            print("prepared", n, "/", len(views), flush=True)
    (dest / "views.json").write_text(json.dumps(metadata, separators=(",", ":")))
    (dest / "audit.json").write_text(
        json.dumps(dict(lighting=lighting, views=len(metadata), sources=report), separators=(",", ":"))
    )
    print(
        lighting,
        "views",
        len(metadata),
        "detected",
        sum(bool(v["maskPixels"]) for v in report.values()),
        flush=True,
    )


def publishable_walk(data, source, detector, cache, output, chromium, font):
    from church_walk_atlas import repair_atlases
    from review_church_walk import review_directories

    if output.exists() or source.resolve() in output.resolve().parents:
        raise ValueError("Choose a new output directory outside the input site")
    fingerprint = hashlib.sha256(b"oxford-walk-range-v1")
    for path in [
        detector,
        data / "transforms_colmap_scaled.json",
        source / "scene.json",
        *sorted(source.glob("photo-*.webp")),
        *sorted((source / "photo-walk").glob("*.png")),
    ]:
        fingerprint.update(path.read_bytes())
    for name in ("merged-cloud-1cm.pcd", "images.zip", "gt-tum.txt"):
        stat = (data / name).stat()
        fingerprint.update(str((name, stat.st_size, stat.st_mtime_ns)).encode())
    ranges = cache / fingerprint.hexdigest()[:20]
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        staged = Path(tmp) / "site"
        tiles = Path(tmp) / "tiles"
        tiles.mkdir()
        shutil.copytree(source, staged, ignore=shutil.ignore_patterns("walk-light"))
        build(data, source, detector, ranges, staged / "walk-light", tiles, font)
        repair_atlases(data, source, tiles, staged)
        from church_landing_floor import repair_landing

        repair_landing(data, staged, detector, staged / "walk-light")
        templates = Path(__file__).resolve().parents[1] / "room/site/christ-church"
        for name in ("index.html", "viewer.js", "viewer.js.LEGAL.txt"):
            shutil.copy2(templates / name, staged / name)
        review_directories(source, staged, detector, chromium, font)
        staged.rename(output)


if __name__ == "__main__":
    arg = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "source", "detector", "cache", "output"):
        arg.add_argument(key, type=Path)
    arg.add_argument("--chromium", type=Path, default=Path("/usr/bin/chromium"))
    arg.add_argument(
        "--font", type=Path, default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    )
    publishable_walk(**vars(arg.parse_args()))
