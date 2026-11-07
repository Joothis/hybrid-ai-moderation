"""Regenerate the images used in README.md (docs/).

    streamlit run app/streamlit_app.py      # in another terminal (port 8501)
    python scripts/make_readme_assets.py

Needs: pip install playwright imageio-ffmpeg (uses the installed Chrome or Edge).
"""
import subprocess
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
VIDEO = ROOT / "demo" / "explainer" / "how_it_works_voiced.mp4"
APP = "http://localhost:8501"
FF = imageio_ffmpeg.get_ffmpeg_exe()


def gif(out, start, length, speed=1.0, width=960, fps=12):
    """Cut a looping GIF from the explainer video with a generated palette."""
    vf = f"setpts=PTS/{speed},fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run([FF, "-y", "-loglevel", "error", "-ss", str(start), "-t", str(length * speed), "-i", str(VIDEO),
                    "-filter_complex", f"[0:v]{vf},split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    "-loop", "0", str(out)], check=True)
    print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")


def frame(out, t, width=1280):
    subprocess.run([FF, "-y", "-loglevel", "error", "-ss", str(t), "-i", str(VIDEO), "-frames:v", "1",
                    "-vf", f"scale={width}:-1", str(out)], check=True)
    print("wrote", out)


def wait_idle(page):
    page.wait_for_timeout(800)
    page.wait_for_function("!document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=120000)
    page.wait_for_timeout(1500)


def app_shots():
    with sync_playwright() as p:
        browser = None
        for channel in ("chrome", "msedge"):
            try:
                browser = p.chromium.launch(channel=channel, args=["--disable-gpu"])
                break
            except Exception:
                continue
        browser = browser or p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.goto(APP, wait_until="networkidle", timeout=120000)
        page.get_by_role("tab", name="Live moderator").wait_for(timeout=120000)
        wait_idle(page)

        # Live moderator: pick the first example the hybrid escalates to the LLM, then run it.
        page.get_by_text("(type your own)").first.click()
        option = page.get_by_role("option").filter(has_text="[LLM]").first
        option.click()
        wait_idle(page)
        page.get_by_role("button", name="Moderate").click()
        page.get_by_text("Verdict:").wait_for(timeout=180000)
        wait_idle(page)
        page.screenshot(path=str(DOCS / "app_live.png"), full_page=True)
        print("wrote docs/app_live.png")

        for tab, name in (("Benchmark", "app_benchmark.png"), ("Trade-off", "app_tradeoff.png")):
            page.get_by_role("tab", name=tab).click()
            wait_idle(page)
            page.screenshot(path=str(DOCS / name), full_page=True)
            print("wrote docs/" + name)
        browser.close()


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    gif(DOCS / "demo.gif", start=18.0, length=16, speed=2.0)   # rules -> ML -> LLM walkthrough, 2x
    frame(DOCS / "architecture.png", 49.0)
    frame(DOCS / "results.png", 94.0)
    app_shots()
