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


def supported_gaps(index, views, positions, rotations, gains):
    """Verify missing sensor depths on the proxy against two separated RGB-D cameras."""
    source = views[index]
    depth = source["depth"][1::2, 1::2]
    mesh = source["surface"]
    # Never overwrite an actual conflicting measurement. Use the first mesh hit
    # as a hypothesis, requiring neighbouring observations to confirm it.
    yy, xx = np.nonzero((depth == 0) & (mesh > 0.15) & (mesh < 6.8))
    mask = np.zeros(mesh.shape, "u1")
    if not len(xx):
        return mask
    z = mesh[yy, xx]
    ray = np.column_stack(
        ((2 * xx + 0.5 - 319.5) / 525, (2 * yy + 0.5 - 239.5) / 525, np.ones(len(xx)))
    )
    world = (ray * z[:, None]) @ rotations[index].T + positions[index]
    ref = source["bgr"][2 * yy + 1, 2 * xx + 1].astype("f4") * gains[index]
    distances = np.linalg.norm(positions - positions[index], axis=1)
    alignment = rotations[:, :, 2] @ rotations[index, :, 2]
    rank = distances + 0.55 * (1 - alignment)
    rank[(distances < 0.065) | (distances > 0.95) | (alignment < 0.58)] = np.inf
    votes = np.zeros(len(xx), "u1")
    first = np.zeros((len(xx), 3), "f4")
    for j in np.argsort(rank)[:10]:
        if not np.isfinite(rank[j]):
            break
        view = views[j]
        cam = (world - positions[j]) @ rotations[j]
        cz = cam[:, 2]
        uv = np.rint(
            cam[:, :2] * (525 / np.maximum(cz, 0.1))[:, None] + [319.5, 239.5]
        ).astype(int)
        u, v = uv.T
        ids = np.flatnonzero(
            (cz > 0.15) & (u >= 0) & (u < 640) & (v >= 0) & (v < 480) & (votes < 2)
        )
        u, v = u[ids], v[ids]
        observed = view["depth"][v, u] * 0.001
        rgb = view["bgr"][v, u].astype("f4") * gains[j]
        tolerance = 0.035 + 0.012 * cz[ids]
        ok = (abs(observed - cz[ids]) < tolerance) & (
            abs(view["surface"][v // 2, u // 2] - cz[ids]) < tolerance
        )
        ok &= np.max(abs(rgb - ref[ids]), axis=1) < 23
        ids = ids[ok]
        # Two frames at nearly the same optical centre are only one observation.
        ids = ids[
            (votes[ids] == 0)
            | (np.linalg.norm(positions[j] - first[ids], axis=1) > 0.06)
        ]
        first[ids] = positions[j]
        votes[ids] += 1
    mask[yy[votes >= 2], xx[votes >= 2]] = 1
    return mask


def write_walk_photos(source: Path, site: Path, gains, frames):
    with source.open("rb") as stream:
        count = int(np.fromfile(stream, "<i4", 1)[0])
    views = np.memmap(source, mode="r", dtype=VIEW, offset=4, shape=(count,))
    if len(gains) != count:
        raise ValueError("Exposure gains must match the calibrated views")
    positions = views["pose"][:, :3]
    rotations = views["pose"][:, 3:].reshape(-1, 3, 3)
    forwards = rotations[:, :, 2]
    folder = site / "walk-photos"
    folder.mkdir(parents=True, exist_ok=True)
    catalog = []
    supported_total = 0
    for index in range(count):
        record = views[index]
        mesh = cv2.resize(
            record["surface"], (640, 480), interpolation=cv2.INTER_NEAREST
        )
        depth = record["depth"].astype("f4") * 0.001
        valid = (depth > 0.15) & (depth < 6.8) & np.isfinite(mesh)
        valid &= abs(depth - mesh) < (0.05 + 0.025 * depth)
        gaps = (depth == 0) & (mesh > 0.15) & (mesh < 6.8)
        inferred = supported_gaps(index, views, positions, rotations, gains)
        supported = gaps & (
            cv2.resize(inferred, (640, 480), interpolation=cv2.INTER_NEAREST) > 0
        )
        alpha = np.zeros(depth.shape, "u1")
        # 255 retains source RGB for near-coincident views. It carries no metric
        # depth. Use a projected range only if neighbouring RGB-D views support it.
        alpha[gaps] = 255
        alpha[supported] = 255 - np.rint(mesh[supported] * 32).astype("u1")
        alpha[valid] = 255 - np.rint(depth[valid] * 32).astype("u1")
        supported_total += int(supported.sum())
        table = np.uint8(
            np.rint(
                255 * np.minimum(1, np.arange(256) * gains[index] ** 0.3 / 255) ** 0.96
            )
        )
        rgb = table[record["bgr"][:, :, ::-1]]
        filename = f"rgbd-{frames[index]:06d}.webp"
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
    print(
        f"{count} source views; {supported_total} missing-depth pixel observations supported",
        flush=True,
    )
    (folder / "views.json").write_text(json.dumps(catalog, separators=(",", ":")))
