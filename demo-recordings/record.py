"""Record a live demo page with the installed Chrome."""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"


def record(url: str, seconds: float, name: str) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=str(RAW),
            record_video_size={"width": 1440, "height": 900},
        )
        page = context.new_page()
        page.goto(url, wait_until="load")
        page.wait_for_timeout(int(seconds * 1000))
        video = page.video
        context.close()
        browser.close()
        target = RAW / name
        if target.exists():
            target.unlink()
        Path(video.path()).replace(target)
        print(target, flush=True)


if __name__ == "__main__":
    record(sys.argv[1], float(sys.argv[2]), sys.argv[3])
