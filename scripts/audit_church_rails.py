"""Check local repaired surfaces and through-rays against synchronized mobile LiDAR."""

import argparse
import csv
from pathlib import Path
import numpy as np
import open3d as o3d
from audit_church_geometry import samples
from church_rail_repair import BOXES
from evaluate_christ_church import glb_scene


def first_window(rays):
    origin, direction = np.split(rays, 2, axis=1)
    entry = np.full(len(rays), np.inf)
    exit_at = np.zeros(len(rays))
    for low, high in BOXES:
        parallel = abs(direction) < 1e-10
        outside = (parallel & ((origin < low) | (origin > high))).any(1)
        safe = np.where(parallel, 1.0, direction)
        a = (np.asarray(low) - origin) / safe
        b = (np.asarray(high) - origin) / safe
        start = np.maximum(0, np.where(parallel, -np.inf, np.minimum(a, b)).max(1))
        end = np.where(parallel, np.inf, np.maximum(a, b)).min(1)
        take = (~outside) & (end > start) & (start < entry)
        entry[take] = start[take]
        exit_at[take] = end[take]
    return entry, exit_at


def audit(data, original, repaired, output):
    observations = []
    for region, _, rays, distance in samples(data):
        if region != "stairs":
            continue
        entry, exit_at = first_window(rays)
        solid = (distance > entry + 0.04) & (distance < exit_at - 0.04)
        through = np.isfinite(entry) & (entry > 0.1) & (distance > exit_at + 0.30)
        observations.append((rays, distance, exit_at, solid, through))
    with output.open("w") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            [
                "version",
                "device",
                "local_hits",
                "local_within_8cm",
                "median_local_error_m",
                "through_rays",
                "through_blocked",
                "stair_rays",
                "stair_within_12cm",
                "stair_early_30cm",
                "stair_no_return",
            ]
        )
        for version, site in [("before", original), ("after", repaired)]:
            for device, suffix in [("desktop", ""), ("mobile", "-mobile")]:
                scene, _ = glb_scene(site / f"christ-church{suffix}.glb", textures=False)
                errors = []
                through_n = blocked = total = close = early = missing = 0
                for rays, measured, exit_at, solid, through in observations:
                    pred = scene.cast_rays(o3d.core.Tensor(rays), nthreads=3)["t_hit"].numpy()
                    delta = pred - measured
                    errors.append(abs(delta[solid]))
                    through_n += through.sum()
                    blocked += np.sum(pred[through] <= exit_at[through] - 0.015)
                    total += len(pred)
                    close += (abs(delta) <= 0.12).sum()
                    early += (delta < -0.30).sum()
                    missing += (~np.isfinite(pred)).sum()
                err = np.concatenate(errors)
                row = [
                    version,
                    device,
                    len(err),
                    (err < 0.08).sum(),
                    round(float(np.median(err)), 5),
                    through_n,
                    blocked,
                    total,
                    close,
                    early,
                    missing,
                ]
                print(*row, flush=True)
                writer.writerow(row)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["data", "original", "repaired", "output"]:
        p.add_argument(key, type=Path)
    audit(**vars(p.parse_args()))
