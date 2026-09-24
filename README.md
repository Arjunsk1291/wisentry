# WiSentry

**WiFi CSI human presence & pose detection.** ESP32 microcontrollers sense
people by measuring how their bodies disturb WiFi radio waves (Channel State
Information). A laptop runs the ML inference and a live dashboard. No cameras,
no radar, no cloud - the runtime is local.

![CI](https://github.com/Arjunsk1291/wisentry/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

![WiSentry live dashboard: walk in, stand, sit, lie down](docs/screenshots/demo.gif)

*Live dashboard in simulation mode (physics-based CSI simulator, 2 receivers).
Stills: [walking](docs/screenshots/dashboard_walking.jpg) ·
[standing](docs/screenshots/dashboard_standing.jpg) ·
[sitting](docs/screenshots/dashboard_sitting.jpg) ·
[lying](docs/screenshots/dashboard_lying.jpg) ·
[empty room](docs/screenshots/dashboard_empty.jpg)*

## What it shows

- **Presence and pose** (standing / sitting / lying / walking) from small CNNs
  on per-band CSI attenuation features.
- **True-scale 3D room** with each TX→RX link and its first Fresnel zone, so
  you can see where a body actually disturbs the signal.
- **Breathing rate** from sub-millimetre chest motion when a person is still.
- **Motion spectrogram** (PCA + STFT, CARM-style speed profile) that separates
  still, slow and walking motion.
- **Signal intelligence** per link: packet rate, jitter, loss, Fresnel radius.
- **Device health**: per-receiver RSSI, packet counts and online status.

The 3D body is a display of the detected pose class. Depth and volume are a
rendering effect, not 3D pose estimation.

## Detection tiers

| Tier | Capability | Status |
|------|-----------|--------|
| 1 | Presence (occupied / empty) | ✅ working (synthetic-trained; calibrate on real data) |
| 2 | Pose (standing / sitting / lying / walking) | ✅ working (synthetic-trained; calibrate on real data) |
| 3 | 17-point skeleton | 🧪 experimental, demo quality |

## Quick start - simulation, no hardware

```bash
git clone https://github.com/Arjunsk1291/wisentry.git
cd wisentry
pip install -r requirements.txt
python setup_check.py          # all lines must say [ OK ]
python main.py --simulate      # then open http://localhost:8050
```

You get the full live dashboard driven by a physics-based CSI simulator: a
scripted person walks in, stands, sits, lies down, and leaves every 30 s.
This validates the software path only. It is not evidence of real-world sensing
accuracy.

## Real-hardware status

The firmware and setup path are included, but this repository does not yet
publish a reproduced real-room calibration result. Treat the hardware path and
all real-world accuracy as work to validate, not as a completed claim.

## With real hardware

2–4 × ESP32-WROOM-32 boards (~$5 each) + USB power. That's the entire BOM.

1. Flash `firmware/csi_transmitter/` to one board, `firmware/csi_receiver/`
   to the rest — step-by-step: [docs/setup.md](docs/setup.md)
2. Place them per [docs/placement_guide.md](docs/placement_guide.md)
3. `python main.py`

## How it works

```text
ESP32 TX ──100 pkt/s──> air (person disturbs multipath) ──> ESP32 RX(s)
ESP32 RX ──UDP wire protocol v1──> laptop
laptop:  udp_server → csi_parser → signal_processor (Hampel, Butterworth,
         band features) → CNN models / rule fallback → debounced detector
                        ↘ csi_analytics (respiration, PCA+STFT spectrogram,
                          link stats)
         → Dash dashboard (11 panels, 2.5 Hz)
```

- **Wire protocol v1** is pinned byte-for-byte across firmware, simulator,
  and parser ([engineering specification](docs/ENGINEERING_SPEC.md#52-wire-protocol-v1--single-source-of-truth)) with a shared test vector.
- **Training = runtime**: `models/train_all.py` generates data by pushing
  simulator physics through the same SignalProcessor used live.
- **Honest metrics**: shipped weights are synthetic-trained
  (`saved/metrics.json` is tagged `"data": "synthetic"`). Calibrate on your
  own room:
  ```bash
  python main.py --collect --label standing --duration 120   # repeat per label
  python scripts/check_capture.py                            # PASS/FAIL per receiver + label
  python models/train_real.py --eval-only                    # sim-to-real gap
  python models/train_real.py                                # fine-tune -> saved/real/
  ```

## Documentation

| Doc | What's in it |
|-----|--------------|
| [docs/user_manual.md](docs/user_manual.md) | 10 chapters, unboxing → live dashboard |
| [docs/hardware_bom.md](docs/hardware_bom.md) | exact parts, prices, where to buy |
| [docs/placement_guide.md](docs/placement_guide.md) | room diagrams, coverage tables |
| [docs/setup.md](docs/setup.md) | Windows + Ubuntu setup, ESP32 flashing, 30 troubleshooting fixes |
| [docs/ENGINEERING_SPEC.md](docs/ENGINEERING_SPEC.md) | full engineering specification |
| [docs/dev/PROJECT_LOG.md](docs/dev/PROJECT_LOG.md) | dated ledger of every test, success, and failure |

## Development

```bash
python -m pytest tests/        # unit tests (parser, DSP, detector, simulator)
python models/train_all.py     # retrain all three models
python scripts/gate_phase3.py --spawn   # end-to-end dashboard gate
CHROME_PATH=/path/to/chrome python scripts/make_media.py --receivers 2   # regenerate every screenshot + demo video
python scripts/make_carousel.py         # summary slides from those screenshots
```

Project history, including what failed and why, lives in
[docs/dev/PROJECT_LOG.md](docs/dev/PROJECT_LOG.md). CI runs the test suite on Python 3.10 and 3.11.
Model-dependent tests skip when synthetic-trained weights are absent; the status
is reported rather than treated as a verified model result.
