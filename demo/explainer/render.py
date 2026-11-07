"""Render explainer.html to an MP4 by seeking the GSAP timeline frame by frame.

    python demo/explainer/render.py                    # full video -> demo/explainer/how_it_works.mp4
    python demo/explainer/render.py --stills 20 33 50  # PNG stills at given seconds (for checking)

Needs: pip install playwright imageio-ffmpeg; uses the installed Microsoft Edge (or Chrome) browser.
"""
import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
PAGE = (HERE / "explainer.html").as_uri() + "?render=1"


def open_page(p):
    for channel in ("chrome", "msedge"):
        try:
            browser = p.chromium.launch(channel=channel, timeout=30000, args=["--disable-gpu"])
            print(f"browser: {channel}", flush=True)
            break
        except Exception as e:
            print(f"{channel} failed: {e}", flush=True)
    else:
        browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    page.goto(PAGE, wait_until="load", timeout=60000)
    page.wait_for_function("window.__duration > 0", timeout=30000)
    page.evaluate("document.fonts.ready.then(() => true)")
    print("page ready", flush=True)
    return browser, page


def stills(times):
    out = HERE / "stills"
    out.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser, page = open_page(p)
        page.evaluate("window.__seek(0)")
        for t in sorted(times):
            page.evaluate(f"window.__seek({t})")
            path = out / f"t{t:05.1f}.png"
            page.screenshot(path=str(path))
            print(path)
        browser.close()


def video(fps, out_path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with sync_playwright() as p:
        browser, page = open_page(p)
        duration = page.evaluate("window.__duration")
        n = int(duration * fps) + 1
        proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
             "-movflags", "+faststart", str(out_path)],
            stdin=subprocess.PIPE,
        )
        for i in range(n):
            page.evaluate(f"window.__seek({i / fps})")
            proc.stdin.write(page.screenshot(type="png"))
            if i % (fps * 5) == 0:
                print(f"{i / fps:5.1f}s / {duration:.1f}s", flush=True)
        proc.stdin.close()
        proc.wait()
        browser.close()
    print(f"wrote {out_path} ({duration:.1f}s, {fps} fps)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", nargs="*", type=float)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=str(HERE / "how_it_works.mp4"))
    a = ap.parse_args()
    if a.stills:
        stills(a.stills)
    else:
        video(a.fps, Path(a.out))
