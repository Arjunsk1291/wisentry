"""Regenerate every post/README visual from the live dashboard (simulation).

    python scripts/make_media.py --receivers 2   # and --receivers 6

Starts `main.py --simulate` with a temporary config (N simulated receivers),
records the dashboard in headless Chromium for one scripted 30 s scenario loop,
and saves per-panel PNGs at fixed scenario states plus a WebM of the run.
Output: media/sim_<N>rx/. All outputs are simulation results.
"""
import argparse, os, re, subprocess, sys, tempfile, time
from pathlib import Path
import yaml
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
URL = "http://127.0.0.1:8050"
VIEWPORT = {"width": 1500, "height": 1150}
PANELS = ["panel-room-map", "panel-pose-figure", "panel-signal-intel",
          "panel-spectrogram", "panel-vitals", "panel-heatmap", "panel-waveform",
          "panel-status-bar", "panel-device-table"]
STATES = [("empty", "EMPTY", None), ("walking", "OCCUPIED", "walking"),
          ("standing", "OCCUPIED", "standing"), ("sitting", "OCCUPIED", "sitting"),
          ("lying", "OCCUPIED", "lying")]


MIN_CONF = 70  # only keep shots where the classifier is settled, not mid-transition


def matches(text, status, pose):
    if status not in text:
        return False
    if pose is None:
        return True
    m = re.search(r"(\d+)% confidence", text)
    return bool(re.search(pose, text, re.I)) and m is not None and int(m.group(1)) >= MIN_CONF


def wait_state(page, status, pose, timeout=75):
    end = time.time() + timeout
    while time.time() < end:
        text = page.inner_text("#panel-status-bar")
        if matches(text, status, pose):
            return True
        time.sleep(0.25)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receivers", type=int, default=2)
    ap.add_argument("--video-seconds", type=float, default=40.0)
    ap.add_argument("--scale", type=float, default=2.5, help="screenshot device_scale_factor")
    args = ap.parse_args()
    out = ROOT / "media" / f"sim_{args.receivers}rx"
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["simulation"]["simulated_devices"] = args.receivers
    tmp = Path(tempfile.mkdtemp()) / "config.yaml"
    tmp.write_text(yaml.safe_dump(cfg))
    app = subprocess.Popen([sys.executable, "main.py", "--simulate", "--config", str(tmp)],
                           cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=os.environ.get("CHROME_PATH") or None)
            ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=args.scale, record_video_dir=str(out),
                                      record_video_size=VIEWPORT)
            page = ctx.new_page()
            t_page = time.time()
            end = time.time() + 40
            while True:
                try:
                    page.goto(URL, timeout=5000); page.wait_for_selector("#panel-status-bar", timeout=10000); break
                except Exception:
                    if time.time() > end: raise
                    time.sleep(1)
            time.sleep(1.5)  # let the first refresh paint before we mark the trim point
            trim_start = time.time() - t_page
            t0 = time.time()
            got = {}
            for name, status, pose in STATES:
                got[name] = False
                deadline = time.time() + 90
                while time.time() < deadline and wait_state(page, status, pose, timeout=deadline - time.time()):
                    time.sleep(0.8)
                    boxes = {pid: page.eval_on_selector(f"#{pid}", "e => {const r=e.getBoundingClientRect(); return [r.x, r.y, r.width, r.height]}")
                             for pid in PANELS}
                    full = out / f"{name}__full.png"
                    page.screenshot(path=str(full))
                    status_text = page.inner_text("#panel-status-bar")
                    # The scenario loops; if the state moved on during the settle
                    # delay, the shot is mislabelled - wait for the next pass.
                    if not matches(status_text, status, pose):
                        continue
                    img = Image.open(full); k = img.width / VIEWPORT["width"]
                    for pid, (x, y, w, h) in boxes.items():
                        img.crop((int(x * k), int(y * k), int((x + w) * k), int((y + h) * k))).save(out / f"{name}__{pid}.png")
                    (out / f"{name}__status.txt").write_text(status_text)
                    got[name] = True
                    break
            rest = args.video_seconds - (time.time() - t0)
            if rest > 0: time.sleep(rest)
            video = page.video.path(); ctx.close(); browser.close()
            Path(video).rename(out / "demo_run.webm")
            # Drop the blank page-load frames at the start of the recording.
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{trim_start:.2f}",
                            "-i", str(out / "demo_run.webm"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-crf", "20", str(out / "demo_run.mp4")], check=True)
            print(f"trimmed {trim_start:.1f}s of loading frames")
            print("states:", got)
    finally:
        app.terminate()
        try: app.wait(10)
        except Exception: app.kill()


if __name__ == "__main__":
    main()
