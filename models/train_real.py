"""Evaluate and fine-tune the models on real ESP32 captures (Phase 6).

Input: labeled windows written by `python main.py --collect --label X`
(logs/dataset/<label>_<session>.npz). Check them first with
`python scripts/check_capture.py`.

    python models/train_real.py --eval-only   # sim-to-real gap of current weights
    python models/train_real.py               # fine-tune from synthetic weights
    python models/train_real.py --scratch     # train from random init

Outputs go to saved/real/ (weights + metrics_real.json) and never overwrite
the synthetic weights in saved/. To run the dashboard on the real-data
models set `ml.model_dir: saved/real` in config.yaml. The skeleton model has
no real ground truth, so its synthetic weights are copied unchanged.

Split: whole sessions are held out when every label has 2+ sessions;
otherwise the last 20% of each session is held out, with a gap of
GAP_WINDOWS so overlapping windows do not leak into validation.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT_DIRECTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT_DIRECTORY))

from backend.ml_engine import POSE_CLASS_LABELS  # noqa: E402
from models.presence_model import PresenceModel  # noqa: E402
from models.pose_model import PoseModel  # noqa: E402
from models.train_all import (  # noqa: E402
    SAVED_DIRECTORY,
    _confusion_matrix,
    _train_model,
)

CLASS_LABELS = ["empty"] + POSE_CLASS_LABELS
REAL_DIRECTORY = SAVED_DIRECTORY / "real"
VALIDATION_FRACTION = 0.2
GAP_WINDOWS = 4          # 50-frame windows, stride 15 -> 4 windows of overlap
FINE_TUNE_EPOCHS = 25
SCRATCH_EPOCHS = 40
SEED = 7


def load_real_dataset(dataset_directory):
    """Load every labeled session file into tensors.

    Returns:
        dict: band [N,1,50,16], aux [N,21], presence [N], pose [N] (-1 empty),
            session [N] (file index), position [N] (window index in session),
            files (list[str]).
    """
    files = sorted(Path(dataset_directory).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz files in {dataset_directory}")
    bands, aux, presence, pose, session, position = [], [], [], [], [], []
    for index, path in enumerate(files):
        data = np.load(path)
        label = str(data["label"])
        if label not in CLASS_LABELS:
            raise ValueError(f"{path.name}: unknown label {label!r}")
        count = data["band_matrices"].shape[0]
        bands.append(data["band_matrices"].astype(np.float32))
        aux.append(data["auxiliary_features"].astype(np.float32))
        presence += [0.0 if label == "empty" else 1.0] * count
        pose += [POSE_CLASS_LABELS.index(label) if label != "empty" else -1] * count
        session += [index] * count
        position += list(range(count))
    return {
        "band": torch.from_numpy(np.concatenate(bands)[:, np.newaxis]).float(),
        "aux": torch.from_numpy(np.concatenate(aux)).float(),
        "presence": torch.tensor(presence, dtype=torch.float32),
        "pose": torch.tensor(pose, dtype=torch.long),
        "session": np.array(session),
        "position": np.array(position),
        "files": [p.name for p in files],
    }


def split_indices(dataset, rng):
    """Return (train_idx, val_idx, split_mode)."""
    session, position = dataset["session"], dataset["position"]
    labels = np.where(dataset["presence"].numpy() == 0, -1, dataset["pose"].numpy())
    sessions_per_label = {}
    for s in np.unique(session):
        sessions_per_label.setdefault(int(labels[session == s][0]), []).append(int(s))
    if all(len(v) >= 2 for v in sessions_per_label.values()):
        held = set()
        for v in sessions_per_label.values():
            shuffled = rng.permutation(v)
            held.update(shuffled[:max(1, int(round(len(v) * VALIDATION_FRACTION)))].tolist())
        is_val = np.array([s in held for s in session])
        return np.where(~is_val)[0], np.where(is_val)[0], "held-out sessions"
    train, val = [], []
    for s in np.unique(session):
        idx = np.where(session == s)[0]
        n = len(idx)
        cut = int(n * (1 - VALIDATION_FRACTION))
        train.extend(idx[position[idx] < cut - GAP_WINDOWS])
        val.extend(idx[position[idx] >= cut])
    return np.array(train), np.array(val), "time split within sessions"


def _load(model, path):
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return True
    return False


def evaluate(presence_model, pose_model, dataset, idx):
    """Accuracy + confusion matrices on the given window indices."""
    band, aux = dataset["band"][idx], dataset["aux"][idx]
    with torch.no_grad():
        p_pred = (torch.sigmoid(presence_model(band, aux))[:, 0] >= 0.5).long().numpy()
    p_true = dataset["presence"][idx].long().numpy()
    out = {"presence": {
        "accuracy": round(float((p_pred == p_true).mean()), 4),
        "confusion_matrix": _confusion_matrix(p_true, p_pred, 2)}}
    mask = dataset["pose"][idx].numpy() >= 0
    if mask.any():
        with torch.no_grad():
            q_pred = pose_model(band[mask], aux[mask]).argmax(dim=1).numpy()
        q_true = dataset["pose"][idx].numpy()[mask]
        out["pose"] = {"accuracy": round(float((q_pred == q_true).mean()), 4),
                       "labels": POSE_CLASS_LABELS,
                       "confusion_matrix": _confusion_matrix(q_true, q_pred, 4)}
    return out


def run(dataset_directory, eval_only=False, scratch=False, output_directory=REAL_DIRECTORY):
    torch.manual_seed(SEED)
    dataset = load_real_dataset(dataset_directory)
    n = dataset["band"].shape[0]
    print(f"train_real: {n} windows from {len(dataset['files'])} session file(s)")
    presence_model, pose_model = PresenceModel(), PoseModel()
    had_p = _load(presence_model, SAVED_DIRECTORY / "presence_model.pt")
    had_q = _load(pose_model, SAVED_DIRECTORY / "pose_model.pt")
    metrics = {"data": "real", "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "files": dataset["files"], "windows": n}
    if eval_only:
        if not (had_p and had_q):
            raise FileNotFoundError("synthetic weights missing in saved/; run models/train_all.py first")
        metrics["mode"] = "eval-only (synthetic weights on all real windows)"
        metrics.update(evaluate(presence_model, pose_model, dataset, np.arange(n)))
    else:
        if scratch:
            presence_model, pose_model = PresenceModel(), PoseModel()
        train_idx, val_idx, split_mode = split_indices(dataset, np.random.default_rng(SEED))
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError("not enough windows to split; collect longer sessions")
        epochs = SCRATCH_EPOCHS if scratch or not had_p else FINE_TUNE_EPOCHS
        metrics.update(mode="scratch" if scratch else "fine-tune from synthetic",
                       split=split_mode, samples_train=int(len(train_idx)),
                       samples_val=int(len(val_idx)), epochs=epochs)
        metrics["before"] = evaluate(presence_model, pose_model, dataset, val_idx)
        band, aux = dataset["band"][train_idx], dataset["aux"][train_idx]
        _train_model(presence_model, (band, aux),
                     dataset["presence"][train_idx].unsqueeze(1), nn.BCEWithLogitsLoss(), epochs)
        mask = dataset["pose"][train_idx] >= 0
        if mask.any():
            _train_model(pose_model, (band[mask], aux[mask]),
                         dataset["pose"][train_idx][mask], nn.CrossEntropyLoss(), epochs)
        metrics["after"] = evaluate(presence_model, pose_model, dataset, val_idx)
        output_directory.mkdir(parents=True, exist_ok=True)
        torch.save(presence_model.state_dict(), output_directory / "presence_model.pt")
        torch.save(pose_model.state_dict(), output_directory / "pose_model.pt")
        skeleton = SAVED_DIRECTORY / "skeleton_model.pt"
        if skeleton.exists():
            shutil.copy(skeleton, output_directory / "skeleton_model.pt")
    output_directory.mkdir(parents=True, exist_ok=True)
    name = "metrics_real_eval.json" if eval_only else "metrics_real.json"
    with open(output_directory / name, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("files",)}, indent=2))
    print(f"train_real: wrote {output_directory / name}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(PROJECT_ROOT_DIRECTORY / "logs" / "dataset"))
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--scratch", action="store_true")
    a = ap.parse_args()
    run(a.dataset, eval_only=a.eval_only, scratch=a.scratch)
