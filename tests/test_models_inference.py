"""Inference smoke tests: load each trained model, run one forward pass.

These tests are skipped on a fresh clone (no saved/ weights); run
`python models/train_all.py` first to enable them.
"""

import time
from pathlib import Path

import numpy as np
import pytest

from backend.ml_engine import MlEngine
from backend.signal_processor import FeatureWindow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHT_FILES = ["presence_model.pt", "pose_model.pt", "skeleton_model.pt"]
weights_present = all(
    (PROJECT_ROOT / "saved" / name).exists() for name in WEIGHT_FILES
)
pytestmark = pytest.mark.skipif(
    not weights_present,
    reason="no trained weights in saved/ — run python models/train_all.py",
)


def _fake_window(motion=2.0, deviation=1.5, breathing=0.1):
    rng = np.random.default_rng(0)
    return FeatureWindow(
        device_id=1,
        band_matrix=rng.normal(0, 1, (50, 16)).astype(np.float32),
        motion_energy=motion,
        baseline_deviation=deviation,
        breathing_energy=breathing,
        mean_amplitude=30.0,
        timestamp=time.time(),
    )


def test_engine_loads_all_three_models():
    engine = MlEngine({"model_dir": "saved", "use_ml_models": True})
    assert engine.models_available is True


def test_inference_output_shapes_and_ranges():
    engine = MlEngine({"model_dir": "saved", "use_ml_models": True})
    result = engine.infer(_fake_window())
    assert result is not None
    assert 0.0 <= result.presence_probability <= 1.0
    assert result.pose_probabilities.shape == (4,)
    assert abs(result.pose_probabilities.sum() - 1.0) < 1e-5
    assert result.skeleton_offsets.shape == (17, 2)
    assert np.abs(result.skeleton_offsets).max() <= 0.08 + 1e-6


def test_metrics_json_is_tagged_synthetic():
    """ENGINEERING_SPEC.md §2: synthetic results must never masquerade as real."""
    import json

    with open(PROJECT_ROOT / "saved" / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["data"] == "synthetic"
    assert metrics["presence"]["val_accuracy"] >= 0.95
    assert metrics["pose"]["val_accuracy"] >= 0.80
