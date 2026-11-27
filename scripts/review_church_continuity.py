"""Compare the same walking and turning inputs in two live Christ Church viewers."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

from review_church_walk import CLOCK

ROUTES = {
    "hall": ("大廳長桌", "", "w" * 14 + "dddlllaaarrr"),
    "landing": ("石階第二站", "#landing-2", "wwwwwaaalllddddrrrsss"),
}
LABELS = {"w": "前進", "s": "後退", "a": "左移", "d": "右移", "l": "轉頭", "r": "回望"}
KEY = """([key,steps])=>new Promise(resolve=>{
 const canvas=document.querySelector('canvas');let n=0;
 const event=type=>canvas.dispatchEvent(new KeyboardEvent(type,{code:'Key'+key.toUpperCase(),key,bubbles:true}));
 function tick(){if(n===0)event('keydown');if(n===steps){event('keyup');resolve();return;}
 n++;requestAnimationFrame(tick);}requestAnimationFrame(tick);})"""
STYLE = "nav,.pad,#caption{visibility:hidden!important} canvas{outline:none!important}"


def record(browser, url, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for route, (_, anchor, actions) in ROUTES.items():
        page = browser.new_page(viewport={"width": 580, "height": 620}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.on("response", lambda r: errors.append(f"{r.status} {r.url}") if r.status >= 400 else None)
        page.add_init_script(CLOCK)
        page.goto(url.rstrip("/") + "/" + anchor, wait_until="domcontentloaded")
        page.wait_for_selector("canvas[data-photowalk=ready]", state="attached", timeout=120000)
        page.wait_for_load_state("networkidle")
        if route == "landing":
            assert page.locator("#photo-progress").inner_text() == "2 / 12"
        canvas = page.locator("canvas")
        canvas.focus()
        bounds = canvas.bounding_box()
        x, y = [bounds[k] + .45 * bounds[length] for k, length in (("x", "width"), ("y", "height"))]
        stage = page.locator(".stage")
        for i, action in enumerate(" " + actions):
            if action in "wasd":
                page.evaluate(KEY, [action, 4])
            elif action in "lr":
                page.mouse.move(x, y)
                page.mouse.down()
                page.mouse.move(x + (38 if action == "l" else -38), y, steps=2)
                page.mouse.up()
            stage.screenshot(path=directory / f"{route}-{i:02}.png", style=STYLE)
        page.wait_for_load_state("networkidle")
        for i in range(2):
            page.wait_for_timeout(650)
            stage.screenshot(path=directory / f"{route}-pause{i}.png", style=STYLE)
        page.close()
        if errors:
            raise RuntimeError(errors)


def record_handoff(browser, url):
    """Hold new photos while walking, release them at a fixed pose and measure the change per frame."""
    page = browser.new_page(viewport={"width": 580, "height": 620}, device_scale_factor=1)
    page.add_init_script(CLOCK)
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_selector("canvas[data-photowalk=ready]", state="attached", timeout=120000)
    page.wait_for_load_state("networkidle")
    held = []
    delaying = True

    def intercept(route):
        if delaying:
            held.append(route)
        else:
            route.continue_()

    page.route("**/walk-light/*", intercept)
    canvas = page.locator("canvas")
    canvas.focus()
    page.evaluate(KEY, ["w", 30])
    box = canvas.bounding_box()
    x, y = box["x"] + .35 * box["width"], box["y"] + .45 * box["height"]

    def turn(pixels):
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x + pixels, y, steps=3)
        page.mouse.up()

    turn(200)
    page.wait_for_timeout(600)
    if not held:
        raise AssertionError("This route did not request a new source")
    page.evaluate("""()=>{
      const canvas=document.querySelector('canvas'), sample=document.createElement('canvas');
      sample.width=180;sample.height=112;const ctx=sample.getContext('2d',{willReadFrequently:true});
      window.changes=[];let last=null;
      function tick(){
        ctx.drawImage(canvas,0,0,180,112);const data=ctx.getImageData(0,0,180,112).data;
        let total=0,error=0;
        for(let i=0;i<data.length;i+=4){
          for(let c=0;c<3;c++){total+=data[i+c];if(last)error+=Math.abs(data[i+c]-last[i+c]);}
        }
        // Ignore a drawing buffer cleared after WebGL presented it.
        if(total>50000){if(last)changes.push(error/(180*112*3));last=data;}
        requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    }""")
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x+1, y)
    page.mouse.move(x, y)
    page.mouse.up()
    page.wait_for_timeout(400)
    page.evaluate("window.changes=[]")
    delaying = False
    for route in held:
        route.continue_()
    page.wait_for_load_state("networkidle")
    page.evaluate("""()=>new Promise(done=>{let n=0;function tick(){if(++n>38)done();
        else requestAnimationFrame(tick);}tick();})""")
    changes = page.evaluate("changes")
    page.close()
    if not changes or max(changes) == 0:
        raise AssertionError("No rendered handoff was captured")
    return {
        "delayedRequests": len(held),
        "validSamples": len(changes),
        "peakMeanAbsRGB": round(max(changes), 3),
        "changingSamples": sum(x > .05 for x in changes),
        "changes": [round(x, 3) for x in changes if x > .005],
    }


def temporal_error(first, second):
    """Warp previous luma to current; exclude offscreen or inconsistent round trips."""
    def gray(image):
        return np.array(image.convert("L"))
    old, new = gray(first), gray(second)
    kwargs = (.5, 4, 21, 4, 7, 1.5, 0)
    forward = cv2.calcOpticalFlowFarneback(old, new, None, *kwargs)
    back = cv2.calcOpticalFlowFarneback(new, old, None, *kwargs)
    h, w = old.shape
    y, x = np.indices((h, w), dtype=np.float32)
    u, v = x + back[..., 0], y + back[..., 1]
    inverse = cv2.remap(forward, u, v, cv2.INTER_LINEAR)
    ok = (u >= 1) & (u < w-2) & (v >= 1) & (v < h-2) & (np.linalg.norm(back + inverse, axis=-1) < 1.5)
    warped = cv2.remap(old, u, v, cv2.INTER_LINEAR)
    residual = np.abs(warped.astype(float) - new)
    return float((residual[ok] > 12).mean()), float(ok.mean())


def summarize(before, after, output, font, handoff):
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    face = ImageFont.truetype(str(font), 17, index=3)
    small = ImageFont.truetype(str(font), 12, index=3)
    movie, durations, result = [], [], {}
    for route, (title, _, actions) in ROUTES.items():
        ids = [f"{i:02}" for i in range(len(actions)+1)] + ["pause0", "pause1"]
        labels = ["起點"] + [LABELS[c] for c in actions] + ["停步", "停步"]
        previous = {}
        series = {name: [] for name in ("before", "after")}
        for suffix, label in zip(ids, labels):
            card = Image.new("RGB", (976, 376), "#f4f1e9")
            draw = ImageDraw.Draw(card)
            for j, (name, folder) in enumerate((("before", before), ("after", after))):
                with Image.open(folder / f"{route}-{suffix}.png") as image:
                    rgb = image.convert("RGB")
                if name in previous and not suffix.startswith("pause"):
                    series[name].append(temporal_error(previous[name], rgb))
                if suffix == "pause1":
                    change = np.abs(np.asarray(previous[name], dtype=float)-np.asarray(rgb))
                    result.setdefault(route, {})[name + "StoppedMeanAbsRGB"] = round(float(change.mean()), 3)
                previous[name] = rgb
                x = 4 + 488*j
                draw.text((x, 3), f"{title}　{label}　·　" + ("修正前" if j == 0 else "修正後"), font=face, fill="#293b2f")
                card.paste(rgb.resize((480, 314), Image.Resampling.LANCZOS), (x, 31))
            draw.text((5, 350), "相同行走與轉頭操作　·　Oxford Spires Dataset / Yifu Tao 等人　·　CC BY-NC-SA 4.0", font=small, fill="#36483c")
            durations.append(650 if label in ("起點","停步") else 180)
            movie.append(card)
        for name, measurements in series.items():
            residual, coverage = np.mean(measurements, axis=0)
            result[route][name] = {
                "framePairs": len(measurements),
                "meanFractionResidualOver12": round(float(residual), 5),
                "meanFlowOverlap": round(float(coverage), 5),
            }
    movie[0].save(output / "continuity.webp", save_all=True, append_images=movie[1:], duration=durations, loop=0, quality=84, method=4)
    result["fixedPoseHandoff"] = handoff
    (output / "continuity.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("output", type=Path)
    parser.add_argument("--chromium", default="/usr/bin/chromium")
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    args = parser.parse_args()
    with TemporaryDirectory() as tmp, sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=args.chromium, args=["--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--disable-dev-shm-usage"])
        before, after = (Path(tmp) / s for s in ("before", "after"))
        record(browser, args.before, before)
        record(browser, args.after, after)
        handoff = {key: record_handoff(browser, url) for key, url in (("before", args.before), ("after", args.after))}
        browser.close()
        summarize(before, after, args.output, args.font, handoff)
