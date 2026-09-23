# WiSentry — Project Log

The single source of truth for project history. Every session appends here:
every major breakthrough, every test run, every success, every failure, every
blocker. Newest entries at the bottom. Never delete an entry — if something
turns out to be wrong, add a correction entry that links back to it.

---

## CURRENT STATE (keep this section updated — it is the resume point)

- **Active phase:** Phase 6 — Real-world calibration (BLOCKED: awaiting ESP32 hardware)
- **Last gate passed:** Phase 5 (all 7 docs present, 2026-06-11); Phases 0-5 all green
- **Hardware on hand:** none yet (0 × ESP32) — see docs/hardware_bom.md
- **Next action:** buy 2-4 ESP32 boards, flash per docs/windows_setup.md,
  run --collect sessions, implement train_all.py --real, 24 h soak test
- **Open blockers:** hardware purchase (user)

---

## ENTRY TEMPLATE (copy for each new entry)

```
## YYYY-MM-DD — <short title>
**Type:** breakthrough | test | success | failure | decision | blocker | note
**Phase:** 0–6
**What happened:** <facts: what was attempted, exact commands run>
**Result:** <pass/fail, numbers, exit codes, accuracy figures — measured only>
**Why it matters / lesson:** <what changed in our understanding or plan>
**Follow-up:** <concrete next step, or "none">
```

---

# LOG

## 2026-06-11 — Repository created, spec rewritten
**Type:** decision
**Phase:** pre-0
**What happened:** Project named **WiSentry** (WiFi + sentry). Git repository
initialized on `main`. The original build prompt was rewritten into
`ENGINEERING_SPEC.md` v2 with major additions: a session protocol bound to this log; a
versioned binary UDP wire protocol (§5.2) shared by firmware, simulator, and
parser; ESP32 CSI API ground truth (§5.1) to prevent hallucinated firmware;
an engineering-honesty section separating synthetic-data results from real
ones; a Phase 0 environment gate and a Phase 6 real-world calibration phase;
a pytest-based testing requirement; and conventional-commit git discipline.
**Result:** Repo scaffolding committed. No code exists yet.
**Why it matters / lesson:** The original spec left the firmware↔backend
packet format undefined and conflated synthetic accuracy with real accuracy —
both would have caused silent failures later.
**Follow-up:** Begin Phase 0.

## 2026-06-11 — Phase 0 complete: environment verified
**Type:** success
**Phase:** 0
**What happened:** Created full directory tree, requirements.txt (pinned),
config.yaml (8 commented sections), setup_check.py. Host already had numpy
2.2.6 / scipy 1.15.3 / torch 2.10.0; installed dash 4.2.0, plotly 6.8.0,
pytest 9.0.3. Ran `python3 setup_check.py`.
**Result:** PASS — all 12 checks green, exit code 0. Ports 5566/8050 free.
**Why it matters / lesson:** torch with CUDA was preinstalled, so no CPU-wheel
substitution was needed; requirements.txt pins torch==2.10.0 generically.
**Follow-up:** Phase 1 backend build.

## 2026-06-11 — Phase 1 complete: full pipeline runs on synthetic CSI
**Type:** success
**Phase:** 1
**What happened:** Built backend (config_loader, csi_parser, signal_processor,
detector+SystemState, skeleton, ml_engine, data_logger, udp_server), the
physics-based simulator, main.py, and a 24-test pytest suite. Wire protocol v1
pinned with a byte-level test vector. Gate run:
`python3 main.py --simulate --headless --duration 60`.
**Result:** PASS — exit 0, 5956 frames received and processed (gate needs
≥1000), 0 parse errors, 585 detection windows. Presence events fired at +5 s
(entered) and +29 s (left) in both 30 s scenario loops — exactly on script.
pytest: 24/24 passed.
**Why it matters / lesson:** Two bugs caught by self-tests before they could
hide: (1) a hand-computed wire-format test vector had 2 extra hex chars —
the pinned vector now comes from verified struct output; (2) event logging
via snapshot-diff missed events once the deque was full — replaced with an
event-listener callback on SystemState.
**Failure note (accepted):** the rule-based fallback misclassifies
sitting-vs-lying (only presence is gated in Phase 1); ML models own pose
accuracy from Phase 2 on.
**Follow-up:** Phase 2 — models + training on simulator physics.

## 2026-06-11 — Phase 2: models trained; pose-flicker failure found and fixed
**Type:** failure → success
**Phase:** 2
**What happened:** Built the three models (shared 50k-param CNN backbone) and
train_all.py, which generates labeled windows by pushing simulator physics
through the real SignalProcessor. First training run: presence 97.0%, pose
90.8% (synthetic). But the end-to-end 60 s run showed severe pose flicker
(~25 spurious pose changes), even after fusing votes across devices.
**Root cause:** training sequences all started at breathing phase 0, so the
classifier never learned phase invariance — steep parts of the 0.3 Hz
breathing sinusoid classify as "walking".
**Fix:** (1) random breathing-phase offset per training sequence, (2) EMA
smoothing (α=0.3) of device-fused pose probabilities in the detector,
(3) pose_change_count 3→4. Retrained: presence 96.33%, pose 82.50%
(synthetic — lower but honest; the phase-randomized dataset is harder).
**Result:** PASS — train_all.py exit 0; saved/ has 3 weight files +
metrics.json (tagged "data": "synthetic"); inference smoke tests pass
(27/27 pytest); 60 s gate exit 0 with the pose sequence tracking the script
(walking→sitting→lying→walking, enter +5 s / leave +30 s, both loops).
**Known limitation:** the 5 s standing segment is absorbed by EMA lag after
walking; logged as a tuning item for Phase 6 real-data calibration.
**Follow-up:** Phase 3 dashboard.

## 2026-06-11 — Phase 3 complete: dashboard live with 7 panels
**Type:** success
**Phase:** 3
**What happened:** Built dashboard/app.py: status bar, CSI waveform (per-
device 5 s scroll), SVG stick-figure pose panel (genuine SVG via data-URI
img, keypoint dots overlaid, labeled experimental), top-down room map with
estimated-person dot, color-coded 30-event log, coverage advisor keyed on
live device count, and the device table with online/offline status. One
200 ms dcc.Interval drives a single 7-output callback reading
SystemState.snapshot(). Wrote tests/gate_phase3.py, which spawns
`python main.py --simulate --duration 45`, asserts all 7 panel ids in
/_dash-layout plus the simulation banner, then polls the Dash update
endpoint exactly as a browser would.
**Result:** PASS — exit 0; all panels render; status panel observed
transitioning EMPTY → OCCUPIED → EMPTY live during the scenario loop.
**Why it matters / lesson:** verifying panel updates through the real
/_dash-update-component endpoint means the gate exercises the same path a
browser does — no "it probably renders" hand-waving.
**Follow-up:** Phase 4 firmware.

## 2026-06-11 — Phase 4 complete: firmware written and compile-verified
**Type:** success
**Phase:** 4
**What happened:** Wrote csi_transmitter.ino (SoftAP "WiSentry" on channel 6,
100 Hz UDP sounding broadcasts, heartbeat LED + serial status) and
csi_receiver.ino (station mode, esp_wifi_set_csi capture, callback-to-ring-
buffer design so network sends happen in loop() not in the WiFi driver task,
packed-struct protocol-v1 datagrams broadcast to 192.168.4.255:5566, auto-
reconnect). Wrote docs/windows_setup.md (drivers CP210x/CH340, Arduino IDE,
board settings, BOOT-button workaround, serial verification). Installed
arduino-cli 1.5.1 + esp32:esp32 core 3.3.10 locally and compiled both
sketches for fqbn esp32:esp32:esp32.
**Result:** PASS — transmitter: 902,136 bytes (68% flash); receiver:
903,772 bytes (68% flash); zero warnings shown. One compile error found and
fixed: `WiFiUdp` typo for the `WiFiUDP` class.
**Why it matters / lesson:** the gate's arduino-cli path caught a real error
that a "manual syntax review" would plausibly have missed — worth the 1.5 GB
toolchain download.
**Hardware-pending caveat:** runtime behaviour (CSI rate, RSSI, broadcast
forwarding) is compile-verified only; first on-hardware validation is
Phase 6.
**Follow-up:** Phase 5 docs (already drafted in parallel).

## 2026-06-11 — Phase 5 complete: documentation suite
**Type:** success
**Phase:** 5
**What happened:** Wrote hardware_bom.md (parts, 2026 prices, classic-ESP32
buying guidance), placement_guide.md (ASCII room diagrams for 1/2/3
receivers, coverage table, degradation factors), user_manual.md (10 chapters,
unboxing → live dashboard → real-data retraining, with an honest ch. 8 about
synthetic-trained model limits), troubleshooting.md (30 numbered symptoms
with fixes across install/simulation/network/flashing/quality),
ubuntu_setup.md (dialout group, brltty gotcha, arduino-cli CLI path), and
rewrote README.md with quick start, architecture, and doc index.
**Result:** PASS — all 7 gate files exist (6 docs + README). Quick Start
commands were each executed during earlier phase gates (pip install,
setup_check, main --simulate, pytest, train_all).
**Why it matters / lesson:** docs were drafted in parallel with the Phase 4
toolchain download — no wall-clock wasted.
**Follow-up:** Phase 6 awaits real hardware: --collect runs, train_all
--real, 24 h soak test.

## 2026-06-11 — Dashboard screenshots captured and embedded in docs
**Type:** note
**Phase:** 5 (addendum)
**What happened:** Added tests/capture_screenshots.py: spawns
`main.py --simulate`, drives headless Chromium (Playwright, dev-only
dependency), watches the live status text, and captures the dashboard at
four scenario moments — empty, walking (90% conf), sitting, lying. PNGs
(1500×760, ~125 KB each) live in docs/screenshots/ and are embedded in
README.md and user_manual.md ch. 5.
**Result:** 4/4 captured on script, exit 0. Visual review confirmed every
panel renders correctly; the lying shot clearly shows the 0.3 Hz breathing
modulation in the CSI waveform — nice incidental validation of the
simulator physics.
**Follow-up:** re-run the script after any dashboard change.

## 2026-09-23 — End-to-end evaluation finds two detector bugs; fixed; dashboard redesign
**Type:** failure → success
**Phase:** 2–3 (simulation only; no hardware yet)
**What happened:** Added an attenuation-profile feature (21 auxiliary model inputs), a larger pose head, sequence-level train/val split and per-sequence domain randomisation (shadow depth ±20%, width ±20%, noise 0.8–1.6×) in train_all.py. Then wrote an offline end-to-end check that runs the scripted 30 s scenario through the real parser → SignalProcessor → MlEngine → detector (3 seeds × 60 s, 2 and 6 simulated receivers, nominal and randomised physics, 2.5 s ignored after each scripted change).
**Result:** Validation (synthetic): presence 96.33% → 97.80%, pose 82.08% → 91.03% (new split is stricter: whole sequences held out). End-to-end, the original code scored pose 52.7% (2 receivers) and 18.9–29.1% (6 receivers): standing and sitting after walking in were held as "walking". Two causes found: (1) a receiver's model voted "walking" with near-zero motion energy; fixed with a physics gate (walking needs motion energy ≥ 0.12 on some receiver). (2) With 6 receivers one receiver's presence model read the empty room as 0.99 occupied and kept breaking the off-vote streak (presence 90.2%); fixed by fusing presence as the median of each receiver's latest probability. After both fixes: presence 100% and pose 100% in all four end-to-end configurations. A class-weighted presence loss was tried and reverted (presence val fell to 89.8%). 31 tests pass; Phase 3 gate passes.
**Why it matters / lesson:** Window-level validation accuracy hid scenario-level failures; the end-to-end check is the number that matches what the dashboard shows. All figures are simulation results, not real-world accuracy.
**Follow-up:** Real ESP32 captures (Phase 6) to test whether the gate threshold and fusion hold on real CSI. Dashboard: dark theme and 3D skeleton view (depth is a display heuristic from 2D keypoints, not a 3D estimate).
