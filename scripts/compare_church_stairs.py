"""Compare two baked stair models against held-out front-camera frame 198."""

import argparse
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

from build_christ_church import project, read_photo
from evaluate_christ_church import glb_scene, render, sample


def compare(data, before, after, output, font):
    frames = json.loads((data / "transforms_colmap_scaled.json").read_text())["frames"]
    front = sorted((f for f in frames if "cam0_" in f["file_path"]), key=lambda f: f["file_path"])
    frame = front[198]
    width, height, fov = 570, 335, 70
    pose = np.array(frame["transform_matrix"])
    # Match the level roll and upward pitch of the viewer's stairs-up bookmark.
    forward = -pose[:3, 2]
    yaw = np.arctan2(forward[1], forward[0])
    pitch = np.arcsin(forward[2]) + 0.24
    forward = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    pose[:3, :3] = np.column_stack([right, np.cross(right, forward), -forward])
    x, y = np.meshgrid(np.arange(width) + 0.5 - width / 2, np.arange(height) + 0.5 - height / 2)
    focal = height / (2 * np.tan(np.deg2rad(fov / 2)))
    rays = np.stack([x, -y, np.full_like(x, -focal)], axis=-1) @ pose[:3, :3].T
    u, v, _ = project((rays + pose[:3, 3]).reshape(-1, 3), frame)
    with zipfile.ZipFile(data / "images.zip") as archive:
        ref = sample(read_photo(archive, frame), u.reshape(height, width), v.reshape(height, width))
    rows = [36, height + 81]
    canvas = Image.new("RGB", (3 * width + 8, rows[1] + height + 41), "#f4f1e9")
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.truetype(str(font), 22, index=3)
    body = ImageFont.truetype(str(font), 18, index=3)

    def tile(col, row, img, label):
        left = col * (width + 4)
        canvas.paste(Image.fromarray(img[:, :, ::-1]), (left, rows[row]))
        draw.text((left + 8, rows[row] - 34), label, font=title, fill="#33372f")

    tile(0, 0, ref, "同角度實拍照片")
    for col, (path, label) in enumerate([(before, "前一版"), (after, "新版")], 1):
        scene, materials = glb_scene(path)
        for row, offset in enumerate([0.0, 0.55]):
            view = pose.copy()
            view[:3, 3] += right * offset
            img, *_ = render(
                scene, materials, {**frame, "transform_matrix": view.tolist()}, width, height, fov
            )
            tile(col, row, img, label + (" · 原拍攝位置" if row == 0 else " · 向右移動 55 公分"))
            if row == 0:
                for name, top in [("full", 0), ("lower half", height // 2)]:
                    a, b = [cv2.cvtColor(i[top:], cv2.COLOR_BGR2GRAY) for i in (ref, img)]
                    print(path.name, name, round(structural_similarity(a, b, data_range=255), 5))
        del materials, scene
    draw.multiline_text(
        (18, rows[1] + 85),
        "橫移測試\n\n右方兩張為相同虛擬視角。\n此位置沒有配對原照，僅比較模型外觀。",
        font=title,
        spacing=8,
        fill="#33372f",
    )
    draw.text(
        (8, rows[1] + height + 6),
        "石階仰視。兩版皆關閉近景照片疊加。Oxford Spires Dataset，CC BY-NC-SA 4.0。",
        font=body,
        fill="#33372f",
    )
    canvas.save(output, quality=92)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ["data", "before", "after", "output"]:
        parser.add_argument(arg, type=Path)
    parser.add_argument(
        "--font", type=Path, default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    )
    compare(**vars(parser.parse_args()))
