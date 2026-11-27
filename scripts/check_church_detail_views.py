"""Compare the real walk renderer to registered photographs withheld from the two repairs."""

import argparse
import csv
from pathlib import Path
import zipfile

import cv2
import numpy as np
from playwright.sync_api import sync_playwright
from skimage.metrics import structural_similarity

from build_christ_church import UP, project, read_photo
from church_rail_repair import camera_list
from evaluate_christ_church import sample

PRIOR = ((0, 30), (1, 29), (2, 29), (0, 87), (1, 87), (2, 87), (0, 177), (0, 203), (1, 175), (2, 203))
# Distinct hall locations with a lateral offset to the nearest source photograph.
HALL = ((0, 42), (0, 37), (0, 111), (0, 55), (0, 158), (0, 150))


def reference(frame, image, shape):
    height, width = shape
    transform = np.asarray(frame["transform_matrix"])
    forward = -transform[:3, 2]
    yaw = np.arctan2(forward[1], forward[0])
    pitch = np.arcsin(forward[2]) + 0.24  # #stairs-up viewer preset
    forward = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    rotation = np.column_stack((right, np.cross(right, forward), -forward))
    x, y = np.meshgrid(np.arange(width) + 0.5 - width / 2, np.arange(height) + 0.5 - height / 2)
    rays = np.stack([x, -y, np.full_like(x, -height / (2 * np.tan(np.deg2rad(35))))], axis=-1)
    world = rays @ rotation.T + transform[:3, 3]
    u, v, _ = project(world.reshape(-1, 3), frame)
    return sample(image, u.reshape(shape), v.reshape(shape))


def capture(browser, url, frame, ident):
    page = browser.new_page(viewport=dict(width=760, height=630), device_scale_factor=1)
    errors = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)

    def place_camera(route):
        scene = route.fetch().json()
        if any(s["id"] == ident and s.get("detail", True) for s in scene["views"]):
            raise ValueError(f"Reference photograph {ident} is a source")
        pose = np.array(frame["transform_matrix"])
        pose[:3] = UP @ pose[:3]
        next(s for s in scene["views"] if s["id"] == 198)["pose"] = pose.tolist()
        route.fulfill(json=scene)

    page.route("**/scene.json*", place_camera)
    page.goto(url.rstrip("/") + "/#stairs-up", wait_until="domcontentloaded")
    page.wait_for_selector('canvas[data-photowalk="ready"]', state="attached", timeout=120000)
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(600)
    blob = page.locator(".stage").screenshot(
        type="png", style="nav,.pad,#caption{visibility:hidden!important}canvas{outline:none!important}"
    )
    page.close()
    if errors:
        raise RuntimeError(errors)
    return cv2.imdecode(np.frombuffer(blob, "u1"), 1)


def scores(image, ref):
    a, b = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[5:-5, 5:-5] for im in (image, ref)]

    def score(row):
        return round(float(structural_similarity(a[row:], b[row:], data_range=255)), 4)

    return score(0), score(a.shape[0] // 2)


def check(args):
    cameras = camera_list(args.data)
    rows = []
    with sync_playwright() as pw, zipfile.ZipFile(args.data / "images.zip") as archive:
        request = pw.request.new_context()
        try:
            for url in (args.before, args.after):
                audit = request.get(url.rstrip("/") + "/walk-light/audit.json").json()
                used = set(audit["sources"])
                for cam, i in PRIOR + HALL:
                    if str(cam * 1000 + i) in used:
                        raise ValueError(f"Reference {cam}/{i} participated in reconstruction")
        finally:
            request.dispose()
        browser = pw.chromium.launch(
            executable_path=str(args.chromium),
            args=["--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--disable-dev-shm-usage"],
        )
        for cam, i in PRIOR + HALL:
            frame = cameras[cam][i]
            previous, candidate = [
                capture(browser, url, frame, cam * 1000 + i) for url in (args.before, args.after)
            ]
            if previous.shape != candidate.shape:
                raise ValueError("Mismatched render dimensions")
            ref = reference(frame, read_photo(archive, frame), candidate.shape[:2])
            prior, lower_prior = scores(previous, ref)
            new, lower_new = scores(candidate, ref)
            rows.append(
                [
                    "hall-offset" if (cam, i) in HALL else "prior",
                    cam,
                    i,
                    candidate.shape[1],
                    candidate.shape[0],
                    prior,
                    new,
                    lower_prior,
                    lower_new,
                ]
            )
        browser.close()
    with args.out.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            [
                "set",
                "camera",
                "frame",
                "width",
                "height",
                "before_ssim",
                "after_ssim",
                "before_lower_ssim",
                "after_lower_ssim",
            ]
        )
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("out", type=Path)
    parser.add_argument("--chromium", type=Path, default=Path("/usr/bin/chromium"))
    check(parser.parse_args())
