# WiSentry — Project Log

The single source of truth for project history. Every session appends here:
every major breakthrough, every test run, every success, every failure, every
blocker. Newest entries at the bottom. Never delete an entry — if something
turns out to be wrong, add a correction entry that links back to it.

---

## CURRENT STATE (keep this section updated — it is the resume point)

- **Active phase:** Phase 3 — Dashboard
- **Last gate passed:** Phase 2 (training exit 0, presence 96.3% / pose 82.5% synthetic, 2026-06-11)
- **Hardware on hand:** none yet (0 × ESP32)
- **Next action:** dashboard/app.py with the 7 panels
- **Open blockers:** none

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
