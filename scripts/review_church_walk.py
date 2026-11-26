"""Walk three repeatable routes in the released web viewer and compare captured frames."""

import argparse
import io
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from tempfile import TemporaryDirectory
from threading import Thread
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright
from build_church_photowalk import people_visibility

ROUTES = [("大廳長桌", "wawdd"), ("樓梯平台", "wawds"), ("石階實拍路線", "wawds")]
CLOCK = """{const raf=window.requestAnimationFrame.bind(window);let count=0,last=-1;
window.requestAnimationFrame=cb=>raf(t=>{if(t!==last){count++;last=t}cb(count*1000/30)});}"""
WALK = """key=>new Promise(resolve=>{
  const canvas=document.querySelector('canvas');
  const event=type=>canvas.dispatchEvent(new KeyboardEvent(type,{code:'Key'+key.toUpperCase(),key,bubbles:true}));
  let n=0;function step(){if(n===0)event('keydown');if(n===5){event('keyup');resolve();return;}
  n++;requestAnimationFrame(step);}requestAnimationFrame(step);})"""


def record(browser, url, phone=False):
    page = browser.new_page(
        viewport=dict(width=390 if phone else 840, height=740),
        device_scale_factor=1,
        is_mobile=phone,
        has_touch=phone,
    )
    errors = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    page.add_init_script(CLOCK)
    page.goto(url.rstrip("/") + "/#stairs-up")
    page.wait_for_selector("#loading[hidden]", state="attached", timeout=120000)
    routes = []
    for name, keys in ROUTES:
        page.get_by_role("button", name=name, exact=True).click()
        frames = []
        for key in " " + keys:
            if key != " ":
                page.locator("canvas").first.focus()
                page.evaluate(WALK, key)
            page.wait_for_timeout(420)
            page.wait_for_load_state("networkidle")
            blob = page.locator(".stage").screenshot(
                type="png",
                style="nav,.pad,#caption{visibility:hidden!important} canvas{outline:none!important}",
            )
            frames.append(Image.open(io.BytesIO(blob)).convert("RGB"))
        if not any(
            np.abs(np.asarray(im, dtype=float) - np.asarray(frames[0], dtype=float)).mean() > 2
            for im in frames[1:]
        ):
            raise AssertionError(f"No visible movement on {name}")
        routes.append(frames)
    page.get_by_role("button", name="對照原照", exact=True).click()
    assert page.locator("#reference").is_visible()
    page.get_by_role("button", name="全區剖視", exact=True).click()
    assert page.locator("#overview").get_attribute("aria-pressed") == "true"
    page.close()
    if errors:
        raise RuntimeError(errors)
    return routes


def inspect_stairs(browser, url, net):
    page = browser.new_page(viewport=dict(width=725, height=710), device_scale_factor=1)
    errors = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    page.goto(url.rstrip("/") + "/#photo-stairs", wait_until="domcontentloaded")
    page.wait_for_selector('canvas[data-photowalk="ready"]', state="attached", timeout=120000)
    counts = []
    landing = None
    for i in range(12):
        if i:
            page.locator("#next-photo").click()
        page.wait_for_timeout(550)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(250)
        assert page.locator("#photo-progress").inner_text() == f"{i+1} / 12"
        blob = page.locator(".stage").screenshot(
            type="jpeg",
            quality=95,
            style="nav,.pad,#caption{visibility:hidden!important} canvas{outline:none!important}",
        )
        if i == 1:
            landing = Image.open(io.BytesIO(blob)).convert("RGB")
        pixels = cv2.imdecode(np.frombuffer(blob, "u1"), 1)
        counts.append(int((people_visibility(pixels, net, (135, 180)) < 0.92).sum()))
    page.close()
    if errors:
        raise RuntimeError(errors)
    return counts, landing


def summarize(routes, net):
    out = []
    for (name, _), frames in zip(ROUTES, routes):
        row = []
        for im in frames:
            bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
            mask = people_visibility(bgr, net, (135, 180)) < 0.92
            row.append(
                dict(
                    personBoxPixels=int(mask.sum()),
                    meanLuma=round(float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean()), 2),
                )
            )
        out.append(dict(station=name, frames=row))
    return out


def landing_comparison(before, after, path, font):
    width, height = before.size
    size = 20
    title = ImageFont.truetype(str(font), size, index=3)
    small = ImageFont.truetype(str(font), 16, index=3)
    sheet = Image.new("RGB", (2 * width + 18, height + 79), "#f3f0e8")
    draw = ImageDraw.Draw(sheet)
    for i, (image, label) in enumerate(((before, "修補前"), (after, "修補後"))):
        left = 6 + i * (width + 6)
        draw.text((left, 6), "石階第二個拍攝點　·　" + label, font=title, fill="#293b2f")
        sheet.paste(image, (left, 38))
    draw.text(
        (8, height + 47),
        "相同相機位置　·　Oxford Spires Dataset · Yifu Tao 等人 · CC BY-NC-SA 4.0",
        font=small,
        fill="#36483c",
    )
    sheet.save(path, quality=94)


def movie(before, after, path, font):
    f = ImageFont.truetype(str(font), 22, index=3)
    small = ImageFont.truetype(str(font), 15, index=3)
    frames = []
    for i, (name, _) in enumerate(ROUTES):
        for j, (old, new) in enumerate(zip(before[i], after[i])):
            w = 540
            h = round(w * old.height / old.width)
            im = Image.new("RGB", (2 * w + 18, h + 82), "#f3f0e8")
            d = ImageDraw.Draw(im)
            for col, (view, tag) in enumerate([(old, "前一版"), (new, "去人影與曝光校正後")]):
                left = 6 + col * (w + 6)
                im.paste(view.resize((w, h), Image.Resampling.LANCZOS), (left, 43))
                d.text((left + 3, 5), name + "　·　" + tag, font=f, fill="#293b2f")
            d.text(
                (10, h + 50),
                f"相同起點與按鍵　 {j+1} / 6　　 Oxford Spires Dataset · Yifu Tao 等人 · CC BY-NC-SA 4.0",
                font=small,
                fill="#36483c",
            )
            frames.append(im)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=[950, 470, 470, 470, 470, 800] * 3,
        loop=0,
        quality=86,
        method=5,
    )


def exposure_example(a, b, gains, path, font):
    from refine_church_walk import H, W, camera_rays, lookup, observation

    # Reproject the second photograph onto a commonly measured patch. Select
    # a textured crop covered in both photographs before looking at the gains.
    world = a.origin + a.rays * a.depth[:, :, None]
    u, v, inside, delta, dist, mask = observation(b, world.reshape(-1, 3))
    ok = (inside & (delta < 0.055 + 0.009 * dist) & (mask == 0)).reshape(H, W)
    ok &= (a.people == 0) & (a.depth > 0.5)
    w, h = 96, 64
    coverage = cv2.boxFilter(ok.astype("f4"), -1, (w, h))
    luma = cv2.cvtColor(cv2.resize(a.original, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    detail = cv2.boxFilter(abs(cv2.Laplacian(luma, cv2.CV_32F)), -1, (w, h))
    midtone = cv2.boxFilter(((luma > 55) & (luma < 210)).astype("f4"), -1, (w, h))
    score = np.where((coverage > 0.965) & (midtone > 0.55), detail, 0)[
        h // 2 : H - h // 2, w // 2 : W - w // 2
    ]
    if score.max() < 1:
        return None
    y, x = np.unravel_index(score.argmax(), score.shape)
    height, width = 4 * h, 4 * w
    yy, xx = np.mgrid[4 * y : 4 * y + height, 4 * x : 4 * x + width]
    rays = camera_rays(a.frame, xx, yy) @ np.asarray(a.frame["transform_matrix"])[:3, :3].T
    position = a.origin + rays * a.depth[yy // 4, xx // 4].reshape(-1, 1)
    u, v, inside, delta, dist, mask = observation(b, position)
    valid = (inside & (delta < 0.055 + 0.009 * dist) & (mask == 0)).reshape(height, width)
    valid &= (a.people[yy // 4, xx // 4] == 0) & (a.depth[yy // 4, xx // 4] > 0.5)
    photos = [a.original[yy, xx], lookup(b.original, u, v).reshape(height, width, 3)]
    margin = 12
    sheet = Image.new("RGB", (2 * width + 3 * margin, 2 * height + 166), "#f3f0e8")
    draw = ImageDraw.Draw(sheet)
    title = ImageFont.truetype(str(font), 22, index=3)
    small = ImageFont.truetype(str(font), 16, index=3)
    for row, label in enumerate(("原始曝光", "統一曝光與白平衡後")):
        top = 12 + row * (height + 57)
        for col, (view, pixels) in enumerate(zip((a, b), photos)):
            left = margin + col * (width + margin)
            patch = (
                pixels.copy() if not row else np.clip(pixels * gains[view.ident], 0, 255).astype("u1")
            )
            patch[~valid] = (75, 75, 75)
            draw.text((left, top), f"{label}　·　照片 {view.ident}", font=title, fill="#293b2f")
            sheet.paste(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)), (left, top + 37))
    draw.text(
        (margin, sheet.height - 46),
        "同一塊實測表面，依雷射深度對齊。灰色表示該處缺少共同觀測。",
        font=small,
        fill="#36483c",
    )
    draw.text(
        (margin, sheet.height - 24),
        "Oxford Spires Dataset · Yifu Tao 等人 · CC BY-NC-SA 4.0",
        font=small,
        fill="#36483c",
    )
    sheet.save(path, lossless=True)
    return [int(4 * x), int(4 * y), width, height]


def review_directories(source, staged, detector, chromium, font):
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    with TemporaryDirectory() as root:
        for name, site in (("before", source), ("after", staged)):
            (Path(root) / name).symlink_to(site.resolve(), target_is_directory=True)
        handler = partial(QuietHandler, directory=root)
        with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
            worker = Thread(target=server.serve_forever, daemon=True)
            worker.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                main(
                    argparse.Namespace(
                        before=base + "/before",
                        after=base + "/after",
                        detector=detector,
                        output=staged / "walk-light",
                        phone=False,
                        chromium=chromium,
                        font=font,
                    )
                )
            finally:
                server.shutdown()
                worker.join()


def main(args):
    net = cv2.dnn.readNet(str(args.detector))
    cv2.setNumThreads(2)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=str(args.chromium),
            headless=True,
            args=["--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--disable-dev-shm-usage"],
        )
        before, after = [record(browser, url, args.phone) for url in (args.before, args.after)]
        stops = {
            k: inspect_stairs(browser, url, net)
            for k, url in (("before", args.before), ("after", args.after))
        }
        browser.close()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "stairs-check.json").write_text(
        json.dumps({k: v[0] for k, v in stops.items()}, indent=2) + "\n"
    )
    report = {k: summarize(v, net) for k, v in [("before", before), ("after", after)]}
    (args.output / "route-check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    landing_comparison(
        stops["before"][1], stops["after"][1], args.output / "landing-comparison.webp", args.font
    )
    movie(before, after, args.output / "walk-comparison.webp", args.font)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("detector", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--phone", action="store_true")
    parser.add_argument("--chromium", type=Path, default=Path("/usr/bin/chromium"))
    parser.add_argument(
        "--font", type=Path, default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    )
    main(parser.parse_args())
