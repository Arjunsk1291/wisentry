"""Train all three WiSentry models on synthetic CSI.

Run from the project root:

    python models/train_all.py            # or python -m models.train_all

Dataset generation pushes the simulator's physics (simulation.simulator)
through the real SignalProcessor (backend.signal_processor), so training
windows are produced by the exact code path used at runtime. Weights are
saved to saved/*.pt with a metrics.json that is explicitly tagged
"data": "synthetic" — these numbers validate the pipeline, not real-world
performance (ENGINEERING_SPEC.md §2). Re-run with real data in Phase 6.

Runs on: any OS with Python 3.9+. Dependencies: torch, numpy, and the
backend/simulation packages.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT_DIRECTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT_DIRECTORY))

from backend.csi_parser import CsiFrame  # noqa: E402
from backend.ml_engine import POSE_CLASS_LABELS  # noqa: E402
from backend.signal_processor import SignalProcessor  # noqa: E402
from simulation.simulator import (  # noqa: E402
    SUBCARRIER_COUNT,
    make_device_baseline,
    synthesize_csi_vector,
)
from models.presence_model import PresenceModel  # noqa: E402
from models.pose_model import PoseModel  # noqa: E402
from models.skeleton_model import SkeletonModel  # noqa: E402

SAVED_DIRECTORY = PROJECT_ROOT_DIRECTORY / "saved"
CLASS_LABELS = ["empty"] + POSE_CLASS_LABELS
SEQUENCES_PER_CLASS = 60
BASELINE_FRAMES_PER_SEQUENCE = 110   # establishes the empty-room baseline
CLASS_FRAMES_PER_SEQUENCE = 130      # frames in the labeled segment
WINDOW_STRIDE = 15                   # windows kept every N frames
FRAME_INTERVAL_SECONDS = 0.02        # 50 Hz, matches config.yaml
VALIDATION_FRACTION = 0.2
TRAINING_EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
DATASET_SEED = 11
TORCH_SEED = 7
SKELETON_OFFSET_SCALE = 0.05  # synthetic ground-truth offset magnitude


def _frames_for_sequence(class_label, sequence_index, random_generator):
    """Generate one labeled frame sequence with the simulator physics.

    Args:
        class_label (str): "empty" or one of POSE_CLASS_LABELS.
        sequence_index (int): Varies the device baseline per sequence.
        random_generator (np.random.Generator): Shared RNG.

    Returns:
        tuple[list[np.ndarray], float]: (amplitude arrays per frame,
            normalized dip-center used — drives skeleton targets).
    """
    device_baseline = make_device_baseline(sequence_index)
    dip_center = random_generator.uniform(12.0, 52.0)
    # Random time origin so the model sees every breathing phase, not just
    # windows aligned to phase zero (learned the hard way: phase-locked
    # training data made still poses look like walking at zero crossings).
    breathing_phase_offset_seconds = random_generator.uniform(0.0, 10.0)
    occupied = class_label != "empty"
    pose_label = class_label if occupied else None
    amplitude_frames = []
    total_frames = BASELINE_FRAMES_PER_SEQUENCE + CLASS_FRAMES_PER_SEQUENCE
    for frame_index in range(total_frames):
        elapsed = (frame_index * FRAME_INTERVAL_SECONDS
                   + breathing_phase_offset_seconds)
        in_class_segment = frame_index >= BASELINE_FRAMES_PER_SEQUENCE
        current_dip = dip_center
        if pose_label == "walking" and in_class_segment:
            current_dip = dip_center + 12.0 * np.sin(elapsed * 0.8)
        csi_vector = synthesize_csi_vector(
            device_baseline,
            occupied and in_class_segment,
            pose_label if in_class_segment else None,
            elapsed,
            current_dip,
            random_generator,
        )
        amplitude_frames.append(np.abs(csi_vector).astype(np.float32))
    return amplitude_frames, (dip_center / SUBCARRIER_COUNT) - 0.5


def _windows_from_sequence(amplitude_frames):
    """Run a frame sequence through the real SignalProcessor.

    Args:
        amplitude_frames (list[np.ndarray]): Per-frame amplitudes.

    Returns:
        tuple[list[np.ndarray], list[list[float]]]: (band matrices,
            auxiliary scalars) for windows inside the class segment.
    """
    processor = SignalProcessor({})
    band_matrices, auxiliary_rows = [], []
    for frame_index, amplitudes in enumerate(amplitude_frames):
        frame = CsiFrame(
            device_id=1,
            sequence_number=frame_index,
            esp32_timestamp_us=int(frame_index * FRAME_INTERVAL_SECONDS * 1e6),
            rssi_dbm=-48,
            amplitudes=amplitudes,
            phases=np.zeros(SUBCARRIER_COUNT, dtype=np.float32),
        )
        window = processor.add_frame(frame)
        window_is_pure_class = frame_index >= (
            BASELINE_FRAMES_PER_SEQUENCE + processor.window_frames
        )
        if (window is not None and window_is_pure_class
                and frame_index % WINDOW_STRIDE == 0):
            band_matrices.append(window.band_matrix)
            auxiliary_rows.append([
                window.motion_energy,
                window.baseline_deviation,
                window.breathing_energy,
            ])
    return band_matrices, auxiliary_rows


def _skeleton_targets(class_index, dip_offset, sample_count, random_generator):
    """Synthetic ground-truth joint offsets for the skeleton regressor.

    The person's lateral position (dip_offset) leans the whole template;
    per-joint noise adds variety. Purely synthetic — replaced by real
    annotations in Phase 6 if skeleton work continues.

    Args:
        class_index (int): Index into CLASS_LABELS.
        dip_offset (float): Normalized dip-center offset (-0.5..0.5).
        sample_count (int): Number of windows in this sequence.
        random_generator (np.random.Generator): Shared RNG.

    Returns:
        np.ndarray: float32[sample_count, 34] offset targets.
    """
    lean = np.zeros((1, 17, 2), dtype=np.float32)
    lean[0, :, 0] = SKELETON_OFFSET_SCALE * dip_offset * 2.0
    jitter = random_generator.normal(
        0.0, 0.01, size=(sample_count, 17, 2)
    ).astype(np.float32)
    targets = np.clip(lean + jitter, -0.08, 0.08)
    if CLASS_LABELS[class_index] == "empty":
        targets[:] = 0.0
    return targets.reshape(sample_count, 34)


def generate_dataset():
    """Build the full labeled synthetic dataset.

    Returns:
        dict: band [N,1,50,16], aux [N,3], presence [N], pose [N]
            (-1 where empty), skeleton [N,34] float32 tensors.
    """
    random_generator = np.random.default_rng(DATASET_SEED)
    bands, auxiliary, presence, pose, skeleton = [], [], [], [], []
    for class_index, class_label in enumerate(CLASS_LABELS):
        for sequence_index in range(SEQUENCES_PER_CLASS):
            frames, dip_offset = _frames_for_sequence(
                class_label,
                class_index * SEQUENCES_PER_CLASS + sequence_index,
                random_generator,
            )
            band_matrices, auxiliary_rows = _windows_from_sequence(frames)
            if not band_matrices:
                continue
            count = len(band_matrices)
            bands.extend(band_matrices)
            auxiliary.extend(auxiliary_rows)
            presence.extend([0.0 if class_label == "empty" else 1.0] * count)
            pose_index = (
                POSE_CLASS_LABELS.index(class_label)
                if class_label != "empty" else -1
            )
            pose.extend([pose_index] * count)
            skeleton.append(
                _skeleton_targets(class_index, dip_offset, count,
                                  random_generator)
            )
    return {
        "band": torch.from_numpy(
            np.stack(bands)[:, np.newaxis, :, :]
        ).float(),
        "aux": torch.tensor(auxiliary, dtype=torch.float32),
        "presence": torch.tensor(presence, dtype=torch.float32),
        "pose": torch.tensor(pose, dtype=torch.long),
        "skeleton": torch.from_numpy(np.concatenate(skeleton)).float(),
    }


def _split_indices(sample_count, random_generator):
    """Shuffle indices and split train/validation."""
    indices = random_generator.permutation(sample_count)
    validation_size = int(sample_count * VALIDATION_FRACTION)
    return indices[validation_size:], indices[:validation_size]


def _train_model(model, inputs, targets, loss_function, epochs):
    """Generic mini-batch training loop.

    Args:
        model (nn.Module): Model to train (modified in place).
        inputs (tuple[torch.Tensor, torch.Tensor]): (band, aux) tensors.
        targets (torch.Tensor): Supervision matching loss_function.
        loss_function (callable): Maps (model output, targets) to a loss.
        epochs (int): Number of passes over the data.

    Returns:
        float: Final-epoch mean training loss.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    band_tensor, auxiliary_tensor = inputs
    sample_count = band_tensor.shape[0]
    final_epoch_loss = float("nan")
    for _epoch in range(epochs):
        permutation = torch.randperm(sample_count)
        epoch_losses = []
        for batch_start in range(0, sample_count, BATCH_SIZE):
            batch_indices = permutation[batch_start:batch_start + BATCH_SIZE]
            optimizer.zero_grad()
            outputs = model(band_tensor[batch_indices],
                            auxiliary_tensor[batch_indices])
            loss = loss_function(outputs, targets[batch_indices])
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        final_epoch_loss = float(np.mean(epoch_losses))
    return final_epoch_loss


def _confusion_matrix(true_labels, predicted_labels, class_count):
    """Compute a confusion matrix as a nested list (rows = truth)."""
    matrix = np.zeros((class_count, class_count), dtype=int)
    for true_value, predicted_value in zip(true_labels, predicted_labels):
        matrix[true_value, predicted_value] += 1
    return matrix.tolist()


def train_and_evaluate():
    """Generate data, train all three models, save weights and metrics.

    Returns:
        dict: The metrics that were written to saved/metrics.json.
    """
    torch.manual_seed(TORCH_SEED)
    print("train_all: generating synthetic dataset "
          f"({SEQUENCES_PER_CLASS} sequences x {len(CLASS_LABELS)} classes)...")
    generation_start = time.time()
    dataset = generate_dataset()
    sample_count = dataset["band"].shape[0]
    print(f"train_all: {sample_count} windows generated in "
          f"{time.time() - generation_start:.1f}s")
    train_idx, val_idx = _split_indices(
        sample_count, np.random.default_rng(DATASET_SEED)
    )
    train_inputs = (dataset["band"][train_idx], dataset["aux"][train_idx])
    val_inputs = (dataset["band"][val_idx], dataset["aux"][val_idx])
    SAVED_DIRECTORY.mkdir(exist_ok=True)
    metrics = {"data": "synthetic", "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "samples_train": len(train_idx), "samples_val": len(val_idx)}

    presence_model = PresenceModel()
    bce = nn.BCEWithLogitsLoss()
    presence_loss = _train_model(
        presence_model, train_inputs,
        dataset["presence"][train_idx].unsqueeze(1),
        bce, TRAINING_EPOCHS,
    )
    with torch.no_grad():
        val_probabilities = torch.sigmoid(presence_model(*val_inputs))[:, 0]
    presence_predictions = (val_probabilities >= 0.5).long().numpy()
    presence_truth = dataset["presence"][val_idx].long().numpy()
    metrics["presence"] = {
        "final_train_loss": round(presence_loss, 4),
        "val_accuracy": round(
            float((presence_predictions == presence_truth).mean()), 4),
        "confusion_matrix": _confusion_matrix(
            presence_truth, presence_predictions, 2),
    }
    torch.save(presence_model.state_dict(),
               SAVED_DIRECTORY / "presence_model.pt")
    print(f"train_all: presence val accuracy "
          f"{metrics['presence']['val_accuracy']:.2%} (synthetic)")

    pose_mask_train = dataset["pose"][train_idx] >= 0
    pose_mask_val = dataset["pose"][val_idx] >= 0
    pose_model = PoseModel()
    cross_entropy = nn.CrossEntropyLoss()
    pose_loss = _train_model(
        pose_model,
        (train_inputs[0][pose_mask_train], train_inputs[1][pose_mask_train]),
        dataset["pose"][train_idx][pose_mask_train],
        cross_entropy, TRAINING_EPOCHS,
    )
    with torch.no_grad():
        pose_logits = pose_model(val_inputs[0][pose_mask_val],
                                 val_inputs[1][pose_mask_val])
    pose_predictions = pose_logits.argmax(dim=1).numpy()
    pose_truth = dataset["pose"][val_idx][pose_mask_val].numpy()
    metrics["pose"] = {
        "final_train_loss": round(pose_loss, 4),
        "val_accuracy": round(float((pose_predictions == pose_truth).mean()), 4),
        "labels": POSE_CLASS_LABELS,
        "confusion_matrix": _confusion_matrix(pose_truth, pose_predictions, 4),
    }
    torch.save(pose_model.state_dict(), SAVED_DIRECTORY / "pose_model.pt")
    print(f"train_all: pose val accuracy "
          f"{metrics['pose']['val_accuracy']:.2%} (synthetic)")

    skeleton_model = SkeletonModel()
    mse = nn.MSELoss()
    skeleton_loss = _train_model(
        skeleton_model, train_inputs, dataset["skeleton"][train_idx],
        mse, TRAINING_EPOCHS,
    )
    with torch.no_grad():
        skeleton_outputs = skeleton_model(*val_inputs)
    skeleton_val_mse = float(
        ((skeleton_outputs - dataset["skeleton"][val_idx]) ** 2).mean()
    )
    metrics["skeleton"] = {
        "final_train_loss": round(skeleton_loss, 6),
        "val_mse": round(skeleton_val_mse, 6),
        "note": "experimental tier — synthetic offset targets only",
    }
    torch.save(skeleton_model.state_dict(),
               SAVED_DIRECTORY / "skeleton_model.pt")
    print(f"train_all: skeleton val MSE {skeleton_val_mse:.6f} (synthetic)")

    with open(SAVED_DIRECTORY / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"train_all: weights + metrics.json written to {SAVED_DIRECTORY}/")
    return metrics


if __name__ == "__main__":
    final_metrics = train_and_evaluate()
    presence_ok = final_metrics["presence"]["val_accuracy"] >= 0.95
    pose_ok = final_metrics["pose"]["val_accuracy"] >= 0.80
    if not (presence_ok and pose_ok):
        print("train_all: FAILED synthetic sanity thresholds "
              "(presence ≥95%, pose ≥80%). Investigate before proceeding.")
        sys.exit(1)
    print("train_all: PASS — all models trained and saved.")
    sys.exit(0)
