#!/usr/bin/env python3
"""Sanity-check a real ESP32 capture before training on it.

Reads what `main.py --collect` and raw logging write into the log dir:
  logs/dataset/<label>_<session>.npz   labeled feature windows
  logs/csi_raw_<session>_NNNN.npz      raw frames (needs log_raw_csi: true)

Prints per-device rate / sequence loss / RSSI, per-label window counts,
and a PASS/FAIL verdict. Exit code 0 = usable, 1 = problems found.

    python scripts/check_capture.py            # uses logs/
    python scripts/check_capture.py path/to/logs --min-windows 150
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LABELS = ["empty", "standing", "sitting", "lying", "walking"]
WINDOW_SHAPE = (50, 16)
MIN_RATE_HZ = 30.0
MAX_LOSS = 0.10
MIN_DEVICES = 2
AUX_COUNT = 21  # backend.signal_processor.build_auxiliary_vector


def check_raw(log_dir):
    """Return (problems, per-device stats) from raw csi_raw_*.npz files."""
    files = sorted(log_dir.glob("csi_raw_*.npz"))
    if not files:
        return ["no raw captures (set logging.log_raw_csi: true to record them)"], {}
    per_dev = defaultdict(lambda: {"t": [], "seq": [], "rssi": [], "amp": []})
    for f in files:
        d = np.load(f)
        for key in ("receive_times", "device_ids", "sequence_numbers", "rssi_values", "amplitudes"):
            if key not in d:
                return [f"{f.name}: missing field {key}"], {}
        for dev in np.unique(d["device_ids"]):
            m = d["device_ids"] == dev
            per_dev[int(dev)]["t"].append(d["receive_times"][m])
            per_dev[int(dev)]["seq"].append(d["sequence_numbers"][m])
            per_dev[int(dev)]["rssi"].append(d["rssi_values"][m])
            per_dev[int(dev)]["amp"].append(d["amplitudes"][m])
    problems, stats = [], {}
    for dev, v in sorted(per_dev.items()):
        t = np.concatenate(v["t"]); seq = np.concatenate(v["seq"]).astype(np.int64)
        rssi = np.concatenate(v["rssi"]); amp = np.concatenate(v["amp"])
        order = np.argsort(t); t, seq = t[order], seq[order]
        dt = np.diff(t)
        active = dt[dt < 1.0]  # ignore pauses between separate --collect runs
        span = float(active.sum())
        rate = len(active) / span if span > 0 else 0.0
        steps = np.diff(seq) % (1 << 32)  # firmware counter is uint32
        expected = int(steps[steps < 10_000].sum()) + 1 if len(steps) else len(seq)
        loss = max(0.0, 1.0 - len(seq) / expected) if expected else 0.0
        flat = float((amp.std(axis=0) < 1e-6).mean())  # subcarriers that never change
        stats[dev] = dict(frames=len(t), seconds=round(span, 1), rate_hz=round(rate, 1),
                          loss=round(loss, 3), rssi_mean=round(float(rssi.mean()), 1),
                          subcarriers=amp.shape[1], flat_subcarriers=round(flat, 2))
        if rate < MIN_RATE_HZ:
            problems.append(f"RX-{dev}: rate {rate:.1f} Hz < {MIN_RATE_HZ} Hz")
        if loss > MAX_LOSS:
            problems.append(f"RX-{dev}: sequence loss {loss:.0%} > {MAX_LOSS:.0%}")
        if flat > 0.5:
            problems.append(f"RX-{dev}: {flat:.0%} of subcarriers are constant (bad CSI?)")
    if len(stats) < MIN_DEVICES:
        problems.append(f"only {len(stats)} receiver(s) in raw capture, need {MIN_DEVICES}")
    return problems, stats


def check_dataset(log_dir, min_windows):
    """Return (problems, windows per label, session files per label)."""
    files = sorted((log_dir / "dataset").glob("*.npz"))
    counts, sessions, problems = defaultdict(int), defaultdict(int), []
    if not files:
        return ["no labeled windows in dataset/ (run main.py --collect --label ...)"], counts, sessions
    for f in files:
        d = np.load(f)
        label = str(d["label"]) if "label" in d else f.name.split("_")[0]
        band, aux = d["band_matrices"], d["auxiliary_features"]
        if label not in LABELS:
            problems.append(f"{f.name}: unknown label {label!r}")
            continue
        if band.shape[1:] != WINDOW_SHAPE:
            problems.append(f"{f.name}: window shape {band.shape[1:]} != {WINDOW_SHAPE}")
        if aux.shape != (band.shape[0], AUX_COUNT):
            problems.append(f"{f.name}: aux shape {aux.shape} != ({band.shape[0]}, {AUX_COUNT})"
                            " - recorded with an old main.py? re-collect")
        if not np.isfinite(band).all() or not np.isfinite(aux).all():
            problems.append(f"{f.name}: NaN/inf values")
        counts[label] += band.shape[0]
        sessions[label] += 1
    for label in LABELS:
        if counts[label] < min_windows:
            problems.append(f"label {label}: {counts[label]} windows < {min_windows}")
    return problems, counts, sessions


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir", nargs="?", default="logs")
    ap.add_argument("--min-windows", type=int, default=100,
                    help="windows needed per label (a 120 s --collect run gives ~390)")
    ap.add_argument("--require-raw", action="store_true", help="fail if raw captures are missing")
    args = ap.parse_args(argv)
    log_dir = Path(args.log_dir)

    raw_problems, stats = check_raw(log_dir)
    print("== receivers (raw capture) ==")
    for dev, s in stats.items():
        print(f"  RX-{dev}: " + ", ".join(f"{k}={v}" for k, v in s.items()))
    ds_problems, counts, sessions = check_dataset(log_dir, args.min_windows)
    print("== labeled windows ==")
    for label in LABELS:
        print(f"  {label:9s} {counts[label]:6d} windows in {sessions[label]} session(s)")

    problems = ds_problems + [p for p in raw_problems
                              if args.require_raw or not p.startswith("no raw captures")]
    notes = [p for p in raw_problems if p.startswith("no raw captures") and not args.require_raw]
    for n in notes:
        print(f"note: {n}")
    if problems:
        print("FAIL:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("PASS: capture looks usable for python models/train_real.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
