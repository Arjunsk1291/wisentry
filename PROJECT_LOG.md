# WiSentry — Project Log

The single source of truth for project history. Every session appends here:
every major breakthrough, every test run, every success, every failure, every
blocker. Newest entries at the bottom. Never delete an entry — if something
turns out to be wrong, add a correction entry that links back to it.

---

## CURRENT STATE (keep this section updated — it is the resume point)

- **Active phase:** Phase 0 — Environment & scaffolding (not started)
- **Last gate passed:** none yet
- **Hardware on hand:** none yet (0 × ESP32)
- **Next action:** create directory tree, requirements.txt, config.yaml,
  setup_check.py per ENGINEERING_SPEC.md §6 Phase 0
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
