# WiSentry — WiFi CSI Presence & Pose Detection System
# ENGINEERING_SPEC.md — the contract for every session. Read fully before writing code.

## 0. SESSION PROTOCOL (do this first, every session)

1. Read `docs/dev/PROJECT_LOG.md`. The **Current State** section at the top tells you
   exactly which phase is active, what passed, what failed, and what to do next.
2. Run `python setup_check.py` if it exists. Fix environment drift before features.
3. Work. Follow the phase plan in §6.
4. Before the session ends (or after any milestone, success, or failure):
   - Append a dated entry to `docs/dev/PROJECT_LOG.md` (template inside that file).
   - Update its **Current State** section.
   - Commit with a conventional-commit message (§10). Never leave work uncommitted.

If anything in this file conflicts with reality (an API changed, a library is
gone), reality wins — but record the deviation in `docs/dev/PROJECT_LOG.md` and update
this file in the same commit.

---

## 1. WHAT YOU ARE BUILDING

A complete, production-ready system that detects human presence and body pose
using only WiFi Channel State Information (CSI). ESP32 microcontrollers measure
how human bodies disturb WiFi radio waves; a laptop runs the ML inference and a
live dashboard. No cameras. No radar. No cloud. 100% local.

Detection tiers (each builds on the last):

| Tier | Capability | Output | Commitment |
|------|-----------|--------|------------|
| 1 | Presence | OCCUPIED / EMPTY + confidence | **Required — must work reliably** |
| 2 | Basic pose | standing / sitting / lying / walking | **Required** |
| 3 | Skeleton | 17 COCO-style keypoints | Best effort — see §2 honesty rules |

## 2. ENGINEERING HONESTY — read before promising anything

- Presence detection from CSI is well-established and achievable (>95% in a
  calibrated room). Commit to it fully.
- 4-class pose from 1–3 ESP32s is achievable (~80%+) **after on-site data
  collection**. Models trained only on synthetic data are pipeline tests, not
  products — always label them as such in logs, reports, and the dashboard.
- 17-keypoint skeleton from commodity ESP32 CSI is research-grade (DensePose
  from WiFi used 3×3 antenna arrays). Build the pipeline, drive it with the
  best estimator you can, but the dashboard and docs must present it as
  experimental. **Never fabricate confidence.**
- Never report an accuracy number you did not measure. Synthetic-data accuracy
  must always be reported as "synthetic"; real-data accuracy as "real".
- If a validation gate fails, the phase is not done. No exceptions, no
  "mostly works".

## 3. WHO USES THIS

Windows 10/11 primary, Ubuntu secondary. Just past beginner: has never flashed
an ESP32, trained a model, or run a Python server. Therefore:

- Every command in docs must be copy-pasteable and work literally as written.
- Every error a user can plausibly hit gets a troubleshooting entry.
- Defaults must be safe; `python main.py --simulate` must work on a fresh
  machine with zero hardware and zero configuration.

## 4. HARDWARE RULES

| | |
|---|---|
| ALLOWED | ESP32-WROOM-32 / ESP32-DevKitC (classic ESP32 only — CSI API is ESP32-specific) |
| ALLOWED | USB cables for flashing/power; 1 to N ESP32s, scaling gracefully |
| ALLOWED | WiFi CSI as the ONLY sensing modality |
| BANNED | Cameras, PIR, mmWave, ultrasonic, any other sensor |
| BANNED | Cloud services — all processing on the laptop |
| BANNED | Raspberry Pi or other SBCs |

## 5. ARCHITECTURE & WIRE PROTOCOL

```
ESP32 TX ── 802.11 frames @ 100 Hz ──> air ──> ESP32 RX(s)
ESP32 RX ── UDP datagrams (binary, port 5566) ──> LAPTOP
LAPTOP: udp_server → csi_parser → signal_processor → ml_engine → detector
                                          │                        │
                                     data_logger              dashboard (Dash,
                                                              http://localhost:8050)
```

### 5.1 ESP32 CSI ground truth (do not hallucinate alternatives)

- CSI capture uses ESP-IDF's `esp_wifi_set_csi(true)`,
  `esp_wifi_set_csi_config(&cfg)`, and `esp_wifi_set_csi_rx_cb(callback, ctx)`.
  These are available from Arduino sketches via `#include "esp_wifi.h"`
  (arduino-esp32 core ≥ 2.x).
- Each CSI callback delivers `wifi_csi_info_t`: an `int8_t buf[]` of
  interleaved (imaginary, real) pairs — LLTF gives 64 subcarriers / 128 bytes
  on a 20 MHz channel. Amplitude = `sqrt(re² + im²)` per subcarrier.
- TX and RX must be on the **same WiFi channel**. Simplest reliable topology:
  TX runs as a SoftAP, RXs run as stations connected to it, and the TX emits
  traffic (e.g. broadcast UDP or ping) at the target rate. 100 Hz is the
  target; log the achieved rate honestly.
- RXs forward CSI to the laptop over the user's normal LAN **or** the TX's
  SoftAP network — config.yaml decides; document both.

### 5.2 UDP packet format (version 1 — every component must honor this)

Little-endian, one CSI frame per datagram:

| Offset | Type | Field |
|--------|------|-------|
| 0 | uint8 | protocol_version = 1 |
| 1 | uint8 | device_id (0–255, from config) |
| 2 | uint32 | sequence_number |
| 6 | uint32 | esp32_timestamp_us (truncated) |
| 10 | int8 | rssi_dbm |
| 11 | uint8 | subcarrier_count (normally 64) |
| 12 | int8[2×count] | interleaved (imag, real) CSI pairs |

`csi_parser.py` and `csi_receiver.ino` implement this identically; a shared
test vector (one known datagram, expected parse) lives in the parser's
self-test. The simulator emits this exact format so the entire downstream
pipeline cannot tell simulation from hardware.

## 6. BUILD PHASES — in order, gated, never skipped

### PHASE 0 — Environment & scaffolding
1. Create the full directory tree (§7), `requirements.txt` (pinned versions),
   `config.yaml` (every setting commented), `setup_check.py` (verifies Python
   version, imports every dependency, checks port 5566 and 8050 are free).
2. `pip install -r requirements.txt` and `python setup_check.py` must pass.
- **GATE:** `setup_check.py` exits 0.

### PHASE 1 — Backend pipeline + simulation (no hardware, no ML)
1. Build in dependency order: `config_loader → csi_parser → signal_processor
   → detector (rule-based fallback first) → data_logger → udp_server`.
2. `simulation/simulator.py`: emits §5.2 datagrams over real localhost UDP
   (so `udp_server` is genuinely exercised), driven by the scenario script
   in §9 with physically plausible CSI (see §9.1).
3. `main.py` with `--simulate`, `--headless`, `--duration N` flags.
- **GATE:** `python main.py --simulate --headless --duration 60` exits 0,
  processes ≥1000 frames, presence transitions match the scenario timeline.

### PHASE 2 — ML models
1. `models/presence_model.py` (binary), `pose_model.py` (4-class),
   `skeleton_model.py` (17×2 regressor) — small CNN/LSTM hybrids sized to run
   <40% CPU on a mid-range i5, no GPU assumed.
2. `models/train_all.py`: generates a labeled synthetic dataset from the
   simulator's physics, trains all three, saves weights + a `metrics.json`
   (accuracy, confusion matrix, clearly tagged `"data": "synthetic"`).
3. `ml_engine.py` loads the saved weights; `detector.py` switches from
   rule-based to ML with hysteresis/debouncing (no flickering states).
- **GATE:** `train_all.py` exits 0; `saved/` has 3 weight files +
  `metrics.json`; an inference smoke test loads each model and prints output
  shapes; Phase 1 gate still passes end-to-end with ML active.

### PHASE 3 — Dashboard
1. `dashboard/app.py`: 7 panels (§8), 200 ms update interval, simulation
   banner when `--simulate`.
- **GATE:** `python main.py --simulate` serves http://localhost:8050; all 7
  panels render and visibly update through one full 30 s scenario loop.

### PHASE 4 — Firmware
1. `firmware/csi_transmitter/csi_transmitter.ino` and
   `firmware/csi_receiver/csi_receiver.ino` — complete, every block commented,
   config constants (SSID, laptop IP, device_id, channel) at the top.
2. `docs/windows_setup.md`: Arduino IDE install → board manager URL → driver
   (CP210x/CH340) → board settings → flash → serial monitor sanity check.
- **GATE:** `arduino-cli compile` passes if arduino-cli is installable;
  otherwise a recorded line-by-line review against the arduino-esp32 CSI API.

### PHASE 5 — Documentation
`hardware_bom.md` (parts, realistic prices, where to buy), `placement_guide.md`
(ASCII room diagrams, coverage table), `user_manual.md` (10 chapters, unboxing
→ live dashboard), `troubleshooting.md` (≥30 real problems with fixes — UDP
blocked by Windows Firewall, wrong COM driver, channel mismatch, weak RSSI…),
`ubuntu_setup.md`, top-level `README.md` with a 5-minute Quick Start.
- **GATE:** all 7 docs exist; every command in Quick Start re-tested verbatim.

### PHASE 6 — Real-world calibration (when hardware arrives)
1. Guided data-collection mode: `python main.py --collect --label standing`
   records labeled real CSI to `logs/dataset/`.
2. `train_all.py --real` retrains on collected data; report real metrics.
3. 24 h soak test with memory/CPU sampling logged.
- **GATE:** measured presence accuracy >95%, pose >80% on a held-out real
  test set; soak test shows no leak (RSS growth <5% over 24 h).

## 7. DIRECTORY STRUCTURE

```
wisentry/
├── ENGINEERING_SPEC.md  PROJECT_LOG.md  README.md  config.yaml  requirements.txt
├── main.py  setup_check.py
├── backend/      config_loader.py udp_server.py csi_parser.py
│                 signal_processor.py ml_engine.py detector.py
│                 skeleton.py data_logger.py  (+ __init__.py)
├── simulation/   simulator.py  (+ __init__.py)
├── dashboard/    app.py        (+ __init__.py)
├── models/       presence_model.py pose_model.py skeleton_model.py
│                 train_all.py  (+ __init__.py)
├── saved/        (weights + metrics.json — git-ignored)
├── firmware/     csi_transmitter/csi_transmitter.ino
│                 csi_receiver/csi_receiver.ino
├── tests/        test_csi_parser.py test_signal_processor.py
│                 test_detector.py test_simulator.py …
├── logs/         (runtime, git-ignored)
└── docs/         user_manual.md hardware_bom.md placement_guide.md
                  windows_setup.md ubuntu_setup.md troubleshooting.md
```

## 8. DASHBOARD — 7 PANELS

1. **STATUS BAR** — full-width; green/red OCCUPIED/EMPTY card, pose label +
   confidence bar, ESP32s online count, simulation banner when applicable.
2. **CSI WAVEFORM** — scrolling line chart, one trace per ESP32, last 5 s,
   mean subcarrier amplitude on y.
3. **POSE FIGURE** — SVG stick figure with 4 stored poses; keypoint dots
   overlaid when skeleton output is available (labeled "experimental").
4. **ROOM MAP** — top-down rectangle; ESP32s as cyan dots (TX/RX-1/RX-2…),
   estimated human position as an orange dot.
5. **EVENT LOG** — last 30 events, timestamped; green=entered, red=left,
   yellow=pose change.
6. **COVERAGE ADVISOR** — live advice: 1 device → "presence only (~15 m²),
   add one for pose"; 2 → "pose active (~25 m²), add one for skeleton";
   3+ → "full system (~35 m²+)".
7. **DEVICE TABLE** — Device ID | IP | RSSI | Packets/s | Status, green/red
   per row, stale devices (no packet >2 s) flagged red.

## 9. SIMULATION SCENARIO (30 s loop)

0–5 s empty → 5–10 s person walks in (presence fires) → 10–15 s stands still
→ 15–20 s sits (pose change) → 20–25 s lies down → 25–30 s walks out
(presence clears).

### 9.1 Simulation realism requirements
Synthetic CSI must contain the physics the models will learn from: static
multipath baseline per subcarrier (Rician-like), Gaussian sensor noise,
breathing modulation ~0.2–0.5 Hz when a person is still, large erratic
amplitude swings during walking, distinct subcarrier-correlation signatures
per pose, and occasional dropped/out-of-order packets to harden the parser.

## 10. GIT DISCIPLINE

- Conventional commits: `feat(backend): …`, `fix(dashboard): …`,
  `docs: …`, `test: …`, `chore: …`. Subject ≤72 chars, body explains *why*
  and records validation evidence ("simulate --headless 60s: exit 0,
  1042 frames").
- One logical change per commit. A phase completion is its own commit whose
  body quotes the gate result.
- `docs/dev/PROJECT_LOG.md` updates ride along with the work they describe.
- Never commit `saved/`, `logs/`, datasets, or `__pycache__` (`.gitignore`
  enforces this).

## 11. CODE QUALITY — NON-NEGOTIABLE

- Module docstring (purpose, runtime, dependencies) and function docstrings
  (args with types/meaning, returns, raises) everywhere.
- No function over 50 lines — extract named helpers.
- All magic numbers are NAMED_CONSTANTS at the top of the file.
- No bare `except:`; every handler produces an actionable English message.
- `if __name__ == "__main__":` self-test in every backend/simulation module.
- Full descriptive names: `received_csi_amplitude_array`, not `raa`.
- `tests/` holds real pytest tests for parser, signal processing, detector
  state machine, and simulator packet format; `pytest -q` must pass at every
  phase gate from Phase 1 onward.

## 12. FORBIDDEN OUTPUT

Never write `# TODO`, `...rest similar to above`, `...` placeholders,
`pass` bodies, or truncate "for brevity". Every file is complete and runnable
as written. Never claim a gate passed without running it (firmware compile
being the only allowed exception, per Phase 4).

## 13. ERROR HANDLING DURING THE BUILD

- Missing package → install it, pin it in requirements.txt, continue.
- Unavailable library → substitute an equivalent, log the substitution in
  PROJECT_LOG.md.
- Gate failure → fix before proceeding. Stuck after 3 distinct attempts on
  one error → stop, write a clear blocker report in PROJECT_LOG.md, ask.

## 14. PERFORMANCE TARGETS

| Metric | Target |
|---|---|
| Presence accuracy (real data) | > 95% |
| Pose accuracy (real data) | > 80% |
| Detection latency | < 500 ms |
| Dashboard refresh | ≥ 5 Hz |
| Stability | 24 h+ no leak/crash |
| CPU | < 40% on mid-range i5 |
