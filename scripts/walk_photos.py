"""Pack registered source photos with their visible mesh depth for on-demand projection."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Same binary records as camera_data(), including the half-resolution first mesh hit.
VIEW = np.dtype(
    [
        ("pose", "<f4", (12,)),
        ("bgr", "u1", (480, 640, 3)),
        ("depth", "<u2", (480, 640)),
        ("surface", "<f4", (240, 320)),
    ]
)


def write_walk_photos(work: Path, site: Path):
    with (work / "views.bin").open("rb") as stream:
        count = int(np.fromfile(stream, "<i4", 1)[0])
    views = np.memmap(
        work / "views.bin", mode="r", dtype=VIEW, offset=4, shape=(count,)
    )
    gains = np.fromfile(work / "photo-gains.bin", "<f4")
    if len(gains) != count:
        raise ValueError("Exposure gains must match the calibrated views")
    positions = views["pose"][:, :3]
    rotations = views["pose"][:, 3:].reshape(-1, 3, 3)
    forwards = rotations[:, :, 2]
    # Cover both camera positions and viewing directions. Keep distinct views of
    # nearby objects even when photographed from the same spot.
    selected = []
    nearest = np.full(count, np.inf)
    for _ in range(min(288, count)):
        index = int(nearest.argmax())
        selected.append(index)
        dist = np.sum((positions - positions[index]) ** 2, axis=1)
        dist += 2.1 * (1 - np.clip(forwards @ forwards[index], -1, 1))
        nearest = np.minimum(nearest, dist)
        nearest[index] = -1
    folder = site / "walk-photos"
    folder.mkdir(parents=True, exist_ok=True)
    catalog = []
    for index in sorted(selected):
        record = views[index]
        mesh = cv2.resize(
            record["surface"], (640, 480), interpolation=cv2.INTER_NEAREST
        )
        depth = record["depth"].astype("f4") * 0.001
        valid = (depth > 0.15) & (depth < 6.8) & np.isfinite(mesh)
        valid &= abs(depth - mesh) < (0.05 + 0.025 * depth)
        # Store the measured range in alpha. Zero explicitly means unobserved.
        alpha = np.where(valid, 255 - np.rint(depth * 32), 0).astype("u1")
        table = np.uint8(
            np.rint(
                255 * np.minimum(1, np.arange(256) * gains[index] ** 0.3 / 255) ** 0.96
            )
        )
        rgb = table[record["bgr"][:, :, ::-1]]
        filename = f"{index:03d}.webp"
        Image.fromarray(np.dstack((rgb, alpha))).save(
            folder / filename, quality=94, method=4
        )
        rotation = rotations[index]
        center = positions[index]
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = rotation.T
        world_to_camera[:3, 3] = -rotation.T @ center
        catalog.append(
            dict(
                file=filename,
                center=center.round(6).tolist(),
                forward=forwards[index].round(6).tolist(),
                matrix=world_to_camera.T.ravel().round(7).tolist(),
            )
        )
    (folder / "views.json").write_text(json.dumps(catalog, separators=(",", ":")))
