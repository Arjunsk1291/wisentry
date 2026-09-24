"""Compose the LinkedIn carousel (1080x1350) from media/sim_*rx panel captures.
Run after scripts/make_media.py --receivers 2 and --receivers 6."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "media"; OUT = M / "carousel"; OUT.mkdir(parents=True, exist_ok=True)
W, H = 1080, 1350
BG = (7, 12, 22); CYAN = (64, 214, 255); DIM = (140, 160, 185); WHITE = (235, 242, 250)
F = "/usr/share/fonts/truetype/noto/NotoSans-"
def font(s, w="Regular"): return ImageFont.truetype(F + w + ".ttf", s)
def base(n, title, sub=None):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    for y in range(0, H, 54): d.line([(0, y), (W, y)], fill=(12, 20, 34))
    for x in range(0, W, 54): d.line([(x, 0), (x, H)], fill=(12, 20, 34))
    d.text((60, 56), "WISENTRY", font=font(30, "Bold"), fill=CYAN)
    d.text((W - 60, 60), f"{n}/7", font=font(26), fill=DIM, anchor="ra")
    d.text((60, 120), title, font=font(52, "Bold"), fill=WHITE)
    if sub: d.text((60, 190), sub, font=font(30), fill=DIM)
    d.line([(60, H - 90), (W - 60, H - 90)], fill=(30, 50, 75), width=2)
    d.text((60, H - 70), "Simulation results", font=font(26, "SemiBold"), fill=CYAN)
    d.text((W - 60, H - 70), "WiFi CSI · ESP32 · Python", font=font(26), fill=DIM, anchor="ra")
    return im, d
def fit(p, w, h):
    im = Image.open(p).convert("RGB"); im.thumbnail((w, h), Image.LANCZOS); return im
def paste(im, p, box):
    x, y, w, h = box; t = fit(p, w, h); im.paste(t, (x + (w - t.width) // 2, y + (h - t.height) // 2))
def bullets(d, y, items, size=30):
    for it in items:
        d.text((60, y), "›", font=font(size, "Bold"), fill=CYAN); d.text((95, y), it, font=font(size), fill=WHITE); y += int(size * 1.6)
    return y
S2, S6 = M / "sim_2rx", M / "sim_6rx"
# 1 cover
im, d = base(1, "Seeing people", "with WiFi signals - no camera")
paste(im, S6 / "walking__panel-room-map.png", (40, 250, 1000, 700))
bullets(d, 980, ["Presence, pose and breathing from WiFi CSI", "ESP32 boards + a laptop, nothing else", "Round 3: 3D room, Fresnel zones, signal analytics"])
im.save(OUT / "01_cover.png")
# 2 room
im, d = base(2, "True-scale 3D room", "TX -> RX links with first Fresnel zones")
paste(im, S6 / "sitting__panel-room-map.png", (40, 250, 1000, 680))
bullets(d, 960, ["Link geometry drawn to scale in metres", "Fresnel ellipsoid shows where a body disturbs each link", "Person position estimated from per-link attenuation"])
im.save(OUT / "02_room.png")
# 3 poses
im, d = base(3, "Pose from radio", "standing · sitting · lying · walking")
for i, s in enumerate(["standing", "sitting", "lying", "walking"]):
    paste(im, S2 / f"{s}__panel-pose-figure.png", (40 + (i % 2) * 505, 250 + (i // 2) * 420, 495, 400))
bullets(d, 1110, ["Body depth/volume is a display effect, not 3D pose estimation"], 26)
im.save(OUT / "03_poses.png")
# 4 signal
im, d = base(4, "Signal intelligence", "what the radio actually sees")
paste(im, S2 / "walking__panel-spectrogram.png", (40, 250, 1000, 470))
paste(im, S2 / "walking__panel-signal-intel.png", (40, 740, 1000, 420))
im.save(OUT / "04_signal.png")
# 5 vitals
im, d = base(5, "Breathing rate", "from chest motion in the CSI phase")
paste(im, S2 / "lying__panel-vitals.png", (40, 250, 1000, 420))
paste(im, S2 / "lying__panel-heatmap.png", (40, 690, 1000, 440))
im.save(OUT / "05_vitals.png")
# 6 architecture
im, d = base(6, "How it works", "end to end, open pipeline")
boxes = [("ESP32 TX", "100 pkt/s on ch. 6"), ("ESP32 RX x2-3", "CSI, 64 subcarriers"), ("UDP wire protocol v1", "to the laptop"),
         ("Signal processing", "filter · PCA · STFT · FFT"), ("ML models", "presence · pose · skeleton"), ("Live dashboard", "Dash / Plotly, 11 panels")]
y = 260
for i, (a, b) in enumerate(boxes):
    d.rounded_rectangle([160, y, 920, y + 120], radius=18, outline=CYAN, width=3, fill=(10, 22, 38))
    d.text((540, y + 28), a, font=font(36, "Bold"), fill=WHITE, anchor="ma"); d.text((540, y + 76), b, font=font(26), fill=DIM, anchor="ma")
    if i < len(boxes) - 1: d.polygon([(525, y + 128), (555, y + 128), (540, y + 150)], fill=CYAN)
    y += 158
im.save(OUT / "06_architecture.png")
# 7 full dashboard
im, d = base(7, "The full dashboard", "live, noisy, moving data")
full = Image.open(S2 / "walking__full.png").convert("RGB"); w0, h0 = full.size
full = full.crop((0, int(h0 * 0.045), w0, h0))
full.thumbnail((1000, 900), Image.LANCZOS); im.paste(full, ((W - full.width) // 2, 260))
bullets(d, 260 + full.height + 40, ["39 automated tests, CI on Python 3.10/3.11"], 28)
im.save(OUT / "07_dashboard.png")
print("ok", sorted(p.name for p in OUT.glob("*.png")))
