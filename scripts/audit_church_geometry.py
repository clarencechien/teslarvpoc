"""Check first hits in a GLB against Oxford's measured, image-synchronised LiDAR ranges."""

import argparse
import csv
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from build_christ_church import camera_rays
from evaluate_christ_church import glb_scene

# Published Kalibr intrinsics used by Oxford's depth image generator (cam0.yaml).
FRONT = dict(
    fl_x=698.1692094914263,
    fl_y=698.5147584873237,
    cx=712.7646767255006,
    cy=560.5751284820508,
    k1=-0.04114225009462773,
    k2=-0.002421765984053725,
    k3=0.003400355152679783,
    k4=-0.0018518822347983275,
)


def samples(data):
    frames = sorted(
        [
            v
            for v in json.loads((data / "transforms_colmap_scaled.json").read_text())["frames"]
            if "cam0_" in v["file_path"]
        ],
        key=lambda v: v["file_path"],
    )
    with zipfile.ZipFile(data / "depths_euc.zip") as archive:
        files = set(archive.namelist())
        for i, frame in enumerate(frames):
            region = "stairs" if 174 <= i <= 209 else "hall" if 18 <= i <= 75 else None
            key = "depths_euc/cam0/" + Path(frame["file_path"]).stem + ".png"
            if region is None or key not in files:
                continue
            im = cv2.imdecode(np.frombuffer(archive.read(key), "u1"), cv2.IMREAD_UNCHANGED)
            v, u = np.nonzero((im > 256 * 0.6) & (im < 256 * 22))
            dist = im[v, u].astype("f4") / 256
            pose = np.array(frame["transform_matrix"], dtype="f4")
            rays = camera_rays(FRONT, u, v) @ pose[:3, :3].T
            batch = np.column_stack([np.broadcast_to(pose[:3, 3], rays.shape), rays]).astype("f4")
            yield region, i, batch, dist


def audit(data, models, output):
    observations = list(samples(data))
    rows = []
    for path in models:
        scene, _ = glb_scene(path, textures=False)
        for region, i, rays, dist in observations:
            pred = scene.cast_rays(o3d.core.Tensor(rays), nthreads=4)["t_hit"].numpy()
            finite = np.isfinite(pred)
            # 30 cm margin deliberately exceeds typical trajectory/calibration discrepancies.
            delta = pred - dist
            rows.append(
                [
                    path.stem,
                    region,
                    i,
                    len(dist),
                    int((delta < -0.30).sum()),
                    int((finite & (delta > 0.30)).sum()),
                    int((~finite).sum()),
                    int((abs(delta) <= 0.12).sum()),
                ]
            )
    with output.open("w") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            ["model", "region", "frame", "rays", "early_30cm", "late_30cm", "no_hit", "within_12cm"]
        )
        writer.writerows(rows)
    for model in dict.fromkeys(r[0] for r in rows):
        for region in ["hall", "stairs"]:
            counts = np.array([r[3:] for r in rows if r[0] == model and r[1] == region])
            if not len(counts):
                continue
            n, *bins = counts.sum(0)
            print(model, region, len(counts), n, np.round(np.array(bins) * 100 / n, 3), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("models", type=Path, nargs="+")
    args = parser.parse_args()
    audit(args.data, args.models, args.csv)
