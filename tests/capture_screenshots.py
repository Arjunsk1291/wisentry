"""Capture dashboard screenshots for the documentation.

Run from the project root (starts the app itself):

    python tests/capture_screenshots.py

Spawns `python main.py --simulate`, opens the dashboard in headless
Chromium (Playwright), waits for specific moments of the 30 s scripted
scenario by watching the live status text, and saves PNGs into
docs/screenshots/. Exits 0 on success.

Dependencies (dev-only, not in requirements.txt): playwright.
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOT_DIRECTORY = PROJECT_ROOT / "docs" / "screenshots"
DASHBOARD_URL = "http://127.0.0.1:8050"
VIEWPORT = {"width": 1500, "height": 1150}
CONTENT_HEIGHT_PX = 1060  # panels end here; crop the empty page below
APP_STARTUP_TIMEOUT_SECONDS = 40
SCENARIO_WAIT_TIMEOUT_SECONDS = 70
# Captures: filename -> (status substring, pose substring or None)
CAPTURE_PLAN = [
    ("dashboard_empty.png", "EMPTY", None),
    ("dashboard_walking.png", "OCCUPIED", "walking"),
    ("dashboard_sitting.png", "OCCUPIED", "sitting"),
    ("dashboard_lying.png", "OCCUPIED", "lying"),
]
# Respiration needs ~12 s of stillness; the scripted scenario is still from
# second 10 to 25, so the lying shot waits (briefly) for a vitals reading.
WAIT_FOR_VITALS = {"dashboard_lying.png"}
VITALS_WAIT_SECONDS = 5.0


def wait_for_vitals(page):
    """Give the respiration panel a few seconds to show a rate."""
    deadline = time.time() + VITALS_WAIT_SECONDS
    while time.time() < deadline:
        if "--" not in page.inner_text("#panel-vitals"):
            return True
        time.sleep(0.3)
    return False


def wait_for_state(page, status_substring, pose_substring):
    """Poll the rendered status bar until it shows the wanted state.

    Args:
        page: Playwright page object on the dashboard.
        status_substring (str): "OCCUPIED" or "EMPTY".
        pose_substring (str | None): Required pose text, if any.

    Returns:
        bool: True when the state appeared before the timeout.
    """
    deadline = time.time() + SCENARIO_WAIT_TIMEOUT_SECONDS
    while time.time() < deadline:
        status_text = page.inner_text("#panel-status-bar")
        flattened = re.sub(r"\s+", " ", status_text)
        if status_substring in flattened and (
            pose_substring is None or f"Pose: {pose_substring}" in flattened
        ):
            return True
        time.sleep(0.3)
    return False


def main():
    """Run the capture session; exit 0 when all screenshots saved."""
    SCREENSHOT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    app_process = subprocess.Popen(
        [sys.executable, "main.py", "--simulate"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=os.environ.get("CHROME_PATH") or None)
            page = browser.new_page(viewport=VIEWPORT)
            deadline = time.time() + APP_STARTUP_TIMEOUT_SECONDS
            while True:
                try:
                    page.goto(DASHBOARD_URL, timeout=5000)
                    page.wait_for_selector("#panel-status-bar div",
                                           timeout=20000)
                    break
                except Exception:
                    if time.time() > deadline:
                        print("FAIL: dashboard never rendered")
                        return 1
                    time.sleep(1)
            print("dashboard rendered; waiting for scenario moments...")
            page.wait_for_timeout(2500)  # let waveform history build up

            for filename, status_needed, pose_needed in CAPTURE_PLAN:
                if not wait_for_state(page, status_needed, pose_needed):
                    print(f"FAIL: never saw {status_needed}/{pose_needed}")
                    return 1
                if filename in WAIT_FOR_VITALS:
                    wait_for_vitals(page)
                page.wait_for_timeout(400)  # let all panels catch up
                page.screenshot(
                    path=str(SCREENSHOT_DIRECTORY / filename),
                    clip={"x": 0, "y": 0, "width": VIEWPORT["width"],
                          "height": CONTENT_HEIGHT_PX},
                )
                print(f"saved {filename} "
                      f"({status_needed}{'/' + pose_needed if pose_needed else ''})")
            browser.close()
        print("ALL SCREENSHOTS CAPTURED")
        return 0
    finally:
        app_process.terminate()
        app_process.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(main())
