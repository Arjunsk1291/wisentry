"""check_capture + train_real on a tiny fake capture (format only, not accuracy)."""
import numpy as np

from models import train_real
from scripts import check_capture


def _fake_logs(tmp_path, windows=12, sessions=1):
    ds = tmp_path / "dataset"
    ds.mkdir()
    rng = np.random.default_rng(0)
    for label in check_capture.LABELS:
        for s in range(sessions):
            np.savez_compressed(ds / f"{label}_2026092{s}.npz", label=label,
                                band_matrices=rng.normal(size=(windows, 50, 16)).astype(np.float32),
                                auxiliary_features=rng.normal(size=(windows, 21)).astype(np.float32))
    for dev in (1, 2):
        n = 500
        seq = np.arange(n)
        seq = np.delete(seq, [10, 20])  # 2 lost frames
        np.savez_compressed(tmp_path / f"csi_raw_s_{dev:04d}.npz",
                            receive_times=seq * 0.02, device_ids=np.full(len(seq), dev),
                            sequence_numbers=seq, rssi_values=np.full(len(seq), -50),
                            amplitudes=rng.normal(size=(len(seq), 52)))
    return tmp_path


def test_check_capture_pass_and_fail(tmp_path):
    logs = _fake_logs(tmp_path)
    assert check_capture.main([str(logs), "--min-windows", "10", "--require-raw"]) == 0
    problems, stats = check_capture.check_raw(logs)
    assert stats[1]["loss"] > 0 and stats[1]["rate_hz"] > 45
    assert check_capture.main([str(logs), "--min-windows", "50"]) == 1


def test_train_real_time_split_and_outputs(tmp_path):
    logs = _fake_logs(tmp_path, windows=30)
    data = train_real.load_real_dataset(logs / "dataset")
    tr, va, mode = train_real.split_indices(data, np.random.default_rng(0))
    assert mode.startswith("time split")
    for s in np.unique(data["session"]):
        t = data["position"][tr][data["session"][tr] == s]
        v = data["position"][va][data["session"][va] == s]
        assert t.max() + train_real.GAP_WINDOWS < v.min()
    out = tmp_path / "out"
    m = train_real.run(logs / "dataset", scratch=True, output_directory=out)
    assert (out / "pose_model.pt").exists() and m["data"] == "real"


def test_session_split_when_enough_sessions(tmp_path):
    logs = _fake_logs(tmp_path, sessions=3)
    data = train_real.load_real_dataset(logs / "dataset")
    tr, va, mode = train_real.split_indices(data, np.random.default_rng(0))
    assert mode == "held-out sessions"
    assert not set(data["session"][tr]) & set(data["session"][va])
