# WiSentry — Project Log

The single source of truth for project history. Every session appends here:
every major breakthrough, every test run, every success, every failure, every
blocker. Newest entries at the bottom. Never delete an entry — if something
turns out to be wrong, add a correction entry that links back to it.

---

## CURRENT STATE (keep this section updated — it is the resume point)

- **Active phase:** Phase 1 — Backend pipeline + simulation
- **Last gate passed:** Phase 0 (setup_check.py exit 0, 2026-06-11)
- **Hardware on hand:** none yet (0 × ESP32)
- **Next action:** build backend modules in dependency order, then simulator
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
