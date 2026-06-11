# WiSentry

**WiFi CSI human presence & pose detection.** ESP32 microcontrollers sense
people by measuring how their bodies disturb WiFi radio waves (Channel State
Information). A laptop runs the ML inference and a live dashboard. No cameras,
no radar, no cloud — 100% local.

## Detection tiers

| Tier | Capability | Status |
|------|-----------|--------|
| 1 | Presence (occupied / empty) | planned |
| 2 | Pose (standing / sitting / lying / walking) | planned |
| 3 | 17-point skeleton (experimental) | planned |

## Hardware

1–4 × ESP32-WROOM-32 / DevKitC boards, USB cables, a Windows or Ubuntu laptop.
That's the entire bill of materials. The system scales: one board gives
presence detection (~15 m²); each additional board unlocks the next tier.

## Status

🚧 **Pre-build.** The engineering specification is complete; implementation
begins with Phase 0. See:

- [`ENGINEERING_SPEC.md`](ENGINEERING_SPEC.md) — the full engineering specification (architecture,
  wire protocol, build phases, validation gates).
- [`PROJECT_LOG.md`](PROJECT_LOG.md) — the running ledger of every test,
  breakthrough, and failure.

## Quick start (once built)

```bash
pip install -r requirements.txt
python setup_check.py
python main.py --simulate     # full pipeline + dashboard, zero hardware needed
```

Dashboard opens at http://localhost:8050.
