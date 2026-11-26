"""Audit the released meshes and show the largest gain and loss on preselected photo views."""

import argparse
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity
from audit_church_geometry import samples
from audit_church_rails import first_window
from build_christ_church import read_photo
from church_rail_repair import camera_list
from evaluate_christ_church import glb_scene, render, sample
from verify_church_edits import control_views


def ranges(data, sites):
    observations = list(samples(data))
    reports = {}
    for name in ("christ-church.glb", "christ-church-mobile.glb"):
        devices = {}
        for label, site in sites:
            scene, _ = glb_scene(site / name, textures=False)
            regions = {}
            for region, _, rays, distance in observations:
                counts = regions.setdefault(region, dict(rays=0, within12=0, early30=0, missing=0))
                pred = scene.cast_rays(o3d.core.Tensor(rays), nthreads=3)["t_hit"].numpy()
                counts["rays"] += len(pred)
                counts["within12"] += int((abs(pred - distance) <= 0.12).sum())
                counts["early30"] += int((pred < distance - 0.30).sum())
                counts["missing"] += int((~np.isfinite(pred)).sum())
                if region == "stairs":
                    entry, end = first_window(rays)
                    through = np.isfinite(entry) & (entry > 0.1) & (distance > end + 0.30)
                    counts["through"] = counts.get("through", 0) + int(through.sum())
                    counts["blocked"] = counts.get("blocked", 0) + int(
                        (pred[through] <= end[through] - 0.015).sum()
                    )
            devices[label] = regions
        reports[name] = devices
    return reports


def pictures(data, sites, *, additional=False):
    cameras = camera_list(data)
    views = json.loads((sites[0][1] / "scene.json").read_text())["views"]
    sources = {v["id"] for v in views if v.get("detail", True)}
    excluded = {v["file_path"] for v in control_views(cameras, sites[0][1])} if additional else set()
    anchors = (46, 66, 159, 214) if additional else (29, 87, 175, 203)
    scenes = [glb_scene(site / "christ-church.glb") for _, site in sites]
    results, pictures = [], []
    with zipfile.ZipFile(data / "images.zip") as photos:
        for cam, frames in enumerate(cameras):
            for anchor in anchors:
                idx = next(
                    i
                    for i in range(anchor, anchor + 20)
                    if i % 4 and cam * 1000 + i not in sources and frames[i]["file_path"] not in excluded
                )
                frame = frames[idx]
                raw = read_photo(photos, frame)
                imgs = []
                for scene, materials in scenes:
                    img, _, _, u, v = render(scene, materials, frame, 600, 400)
                    imgs.append(img)
                reference = sample(raw, u, v)

                def score(img):
                    gray = lambda arr: cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                    return round(
                        float(structural_similarity(gray(img), gray(reference), data_range=255)), 5
                    )

                scores = [score(im) for im in imgs]
                # Centre a 3x inspection crop on the largest change, irrespective
                # of whether that change helped or hurt the comparison.
                difference = cv2.absdiff(*imgs).mean(2).astype("f4")
                density = cv2.boxFilter(difference, -1, (201, 133))
                _, _, _, (x, y) = cv2.minMaxLoc(density[66:-66, 100:-100])
                focus = [round((x + 100.5) / 600, 4), round((y + 66.5) / 400, 4)]
                results.append(
                    dict(camera=cam, frame=idx, before=scores[0], after=scores[1], focus=focus)
                )
                pictures.append([reference, *imgs])
    return results, pictures


def comparison(rows, pictures, path, font):
    title = ImageFont.truetype(str(font), 34, index=3)
    body = ImageFont.truetype(str(font), 21, index=3)
    diffs = np.array([r["after"] - r["before"] for r in rows])
    indices = np.argsort(diffs, kind="stable")
    canvas = Image.new("RGB", (1512, 908), "#f4f1e9")
    draw = ImageDraw.Draw(canvas)
    ink = "#293b2f"
    draw.text((18, 8), "全室自動清理 · 留出照片核對", font=title, fill=ink)
    for c, txt in enumerate(["現場實拍", "上一版網格", "自動清理後"]):
        draw.text((18 + c * 500, 65), txt, font=body, fill=ink)
    for row, idx in enumerate([indices[-1], indices[0]]):
        item = rows[idx]
        y = 102 + row * 375
        # Same 3:2 perspective in all columns, with no nearby-photo overlays.
        for c, img in enumerate(pictures[idx]):
            tile = Image.fromarray(img[:, :, ::-1]).resize((492, 328), Image.Resampling.LANCZOS)
            canvas.paste(tile, (12 + c * 500, y))
        label = "變化最高" if row == 0 else "變化最低"
        draw.text(
            (18, y + 329),
            f'{label} · 相機 {item["camera"]} ／ 第 {item["frame"]} 幀　全圖 SSIM {item["before"]:.3f} → {item["after"]:.3f}',
            font=body,
            fill=ink,
        )
    draw.text(
        (18, 860),
        "12 個預留視角依 SSIM 變化排序。三欄使用相同相機。Oxford Spires Dataset · Yifu Tao 等人 · CC BY-NC-SA 4.0",
        font=body,
        fill=ink,
    )
    canvas.save(path, quality=92, method=6)


def review(data, original, candidate, font):
    sites = [("before", original), ("after", candidate)]
    metrics = ranges(data, sites)
    rows, images = pictures(data, sites)
    failures = []
    for model, stats in metrics.items():
        for region, before in stats["before"].items():
            after = stats["after"][region]
            # Fixed 0.15 percentage-point tolerance per region and per device.
            for key, sign in [("within12", -1), ("early30", 1), ("missing", 1), ("blocked", 1)]:
                if key not in before:
                    continue
                n = before["through"] if key == "blocked" else before["rays"]
                if sign * (after[key] - before[key]) / n > 0.0015:
                    failures.append(f"{model}/{region}/{key}")
    mean_change = float(np.mean([r["after"] - r["before"] for r in rows]))
    if mean_change <= 0:
        failures.append("mean baked photo SSIM")
    extra, _ = pictures(data, sites, additional=True)
    result = dict(
        ranges=metrics,
        photos=rows,
        meanPhotoChange=mean_change,
        additionalPhotos=extra,
        failures=failures,
    )
    views_dir = candidate / "cleanup-views"
    views_dir.mkdir(exist_ok=True)
    for row, pictures_for_view in zip(rows, images):
        for label, img in zip(("reference", "before", "after"), pictures_for_view):
            target = views_dir / f'{row["camera"]}-{row["frame"]}-{label}.webp'
            if not cv2.imwrite(str(target), img, [cv2.IMWRITE_WEBP_QUALITY, 96]):
                raise OSError(f"Could not write {target}")
    comparison(rows, images, candidate / "cleanup-review.webp", font)
    (candidate / "cleanup-review.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


DEFAULT_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "original", "candidate"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    report = review(**vars(parser.parse_args()))
    print(json.dumps(report), flush=True)
    raise SystemExit(bool(report["failures"]))
