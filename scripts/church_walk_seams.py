"""Remove rectangular repair edges using a reprojected background and gradient-domain blending."""

import warnings
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from build_christ_church import camera_rays


def segment_background(colors, validity, original, box):
    """Fill small sampling gaps in each donor before selecting a coherent background."""
    layers, weights = [], []
    for rank, (image, valid) in enumerate(zip(colors, validity)):
        distance, nearest = distance_transform_edt(~valid, return_indices=True)
        layers.append(image[nearest[0], nearest[1]])
        neighborhood = cv2.GaussianBlur(valid.astype("f4"), (0, 0), 3)
        weights.append(
            (np.exp(-0.5 * (distance / 2.2) ** 2) * neighborhood / (1 + 0.08 * rank)).astype("f4")
        )
    layers, weights = np.stack(layers), np.stack(weights)
    missing = weights.max(0) < 0.035
    if missing[box].mean() > 0.10:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(np.where(weights[:, :, :, None] > 0.05, layers, np.nan), 0)
    agreement = np.max(abs(layers.astype("f4") - np.nan_to_num(median)), 3)
    weights /= (1 + (agreement / 17) ** 2) ** 2
    y, x = np.indices(box.shape)
    repair = layers[np.argmax(weights, 0), y, x].copy()
    if missing.any():
        repair = cv2.inpaint(repair, missing.astype("u1"), 3, cv2.INPAINT_TELEA)
    difference = np.max(abs(original.astype("f4") - repair), 2)
    labels = np.where(box, cv2.GC_PR_FGD, cv2.GC_BGD).astype("u1")
    labels[box & (difference < 17) & ~missing] = cv2.GC_PR_BGD
    if not np.any(labels == cv2.GC_PR_FGD):
        return None
    cv2.setRNGSeed(0)
    cv2.grabCut(original, labels, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
    fg = labels % 2
    if fg.sum() < 0.10 * box.sum():
        return None
    fg = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    return cv2.medianBlur(repair, 3), fg, float(1 - missing[box].mean())


def refine_boundary(view, views, gains, model, net, existing, stats):
    from refine_church_walk import inspect_people, lookup, model_samples, neighbours, observation

    mask = view.people
    if mask.sum() < 400 or not stats.get("supported"):
        return existing
    light = gains[view.ident]
    source = np.clip(view.original * light, 0, 255).astype("u1")
    count, labels, boxes, _ = cv2.connectedComponentsWithStats(mask)
    output = existing.copy()
    accepted = []
    for i in range(1, count):
        x0, y0, width, height, area = boxes[i]
        if area < 400:
            continue
        # Keep a real background border around this component for the Poisson boundary.
        if x0 < 5 or y0 < 5 or x0 + width > mask.shape[1] - 5 or y0 + height > mask.shape[0] - 5:
            continue
        left, top = 4 * max(0, x0 - 7), 4 * max(0, y0 - 7)
        right = 4 * min(mask.shape[1], x0 + width + 7)
        bottom = 4 * min(mask.shape[0], y0 + height + 7)
        raw = source[top:bottom, left:right]
        old = output[top:bottom, left:right]
        box = labels[top // 4 : bottom // 4, left // 4 : right // 4] == i
        quarter = cv2.resize(old, box.shape[::-1], interpolation=cv2.INTER_AREA).astype("f4")
        ref = cv2.resize(raw, box.shape[::-1], interpolation=cv2.INTER_AREA).astype("f4")
        boundary = box & ~cv2.erode(box.astype("u1"), np.ones((5, 5), "u1")).astype(bool)
        loss = abs(quarter - ref).mean(2)
        if np.percentile(loss[boundary], 75) < 8 or (loss[box] > 10).mean() < 0.20:
            continue
        yy, xx = np.mgrid[top:bottom:2, left:right:2]
        y, x = yy.ravel(), xx.ravel()
        _, mesh_distance = model_samples(model, view, x, y)
        z = view.depth[y // 4, x // 4]
        agrees = (
            np.isfinite(mesh_distance)
            & (mesh_distance > 0.4)
            & ((z == 0) | (abs(z - mesh_distance) < 0.08 + 0.02 * z))
        )
        z = np.where(agrees, mesh_distance, z)
        rays = camera_rays(view.frame, x, y) @ np.asarray(view.frame["transform_matrix"])[:3, :3].T
        world = view.origin + rays * z[:, None]
        target = source[y, x]
        region = labels[y // 4, x // 4] == i
        candidates = []
        for other in neighbours(views, view, 65):
            u, v, inside, delta, dist, person = observation(other, world)
            valid = inside & (person == 0) & (delta < 0.085 + 0.018 * dist) & (z > 0.35)
            ring = valid & ~region
            if valid.mean() < 0.10 or ring.sum() < 35:
                continue
            pixels = np.clip(lookup(other.original, u, v) * gains[other.ident], 0, 255)
            mismatch = np.median(abs(pixels[ring] - target[ring]).mean(1))
            score = valid.mean() / (1 + 0.035 * mismatch)
            candidates.append(
                (score, pixels.astype("u1").reshape(*xx.shape, 3), valid.reshape(xx.shape))
            )
        if len(candidates) < 3:
            continue
        candidates.sort(key=lambda item: -item[0])
        _, colors, validity = zip(*candidates[:20])
        candidate = segment_background(
            colors, validity, target.reshape(*xx.shape, 3), region.reshape(xx.shape)
        )
        if candidate is None:
            continue
        repair, foreground, coverage = candidate
        size = (right - left, bottom - top)
        repair = cv2.resize(repair, size, interpolation=cv2.INTER_CUBIC)
        foreground = cv2.resize(foreground, size, interpolation=cv2.INTER_NEAREST)
        ident_mask = cv2.resize(box.astype("u1"), size, interpolation=cv2.INTER_NEAREST)
        x, y, w, h = cv2.boundingRect(foreground)
        proposal = cv2.seamlessClone(
            repair, raw, foreground * 255, (x + w // 2, y + h // 2), cv2.NORMAL_CLONE
        )
        # Preserve the prior repair outside this person's box (including cast shadows).
        alpha = cv2.GaussianBlur(np.maximum(foreground, ident_mask).astype("f4"), (5, 5), 0.9)[
            :, :, None
        ]
        old[:] = np.clip(old * (1 - alpha) + proposal * alpha, 0, 255).astype("u1")
        accepted.append(
            dict(box=[int(x0), int(y0), int(width), int(height)], covered=round(coverage, 4))
        )
    if not accepted:
        return existing
    people = int(inspect_people(output, net).sum())
    if people > stats["reviewAfter"]:
        return existing
    stats["boundaryRepair"] = accepted
    stats["reviewAfter"] = people
    return output
