"""ML inference engine: loads and runs the three WiSentry models.

Loads presence (binary), pose (4-class), and skeleton (17x2 regression)
models from the saved/ directory if they exist. When weights are missing
(Phase 1, or a fresh clone) the engine reports itself unavailable and the
detector falls back to its rule-based logic — the system never crashes
for lack of models.

Runs on: any OS with Python 3.9+. Dependencies: torch, numpy, and the
model definitions in models/.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from backend.signal_processor import build_auxiliary_vector

PROJECT_ROOT_DIRECTORY = Path(__file__).resolve().parent.parent
PRESENCE_WEIGHTS_FILENAME = "presence_model.pt"
POSE_WEIGHTS_FILENAME = "pose_model.pt"
SKELETON_WEIGHTS_FILENAME = "skeleton_model.pt"
POSE_CLASS_LABELS = ["standing", "sitting", "lying", "walking"]
AUXILIARY_FEATURE_COUNT = 21  # 3 scalars + 16-band attenuation profile + 2 shape stats


@dataclass
class InferenceResult:
    """Outputs of one inference pass over a FeatureWindow.

    Attributes:
        presence_probability (float): P(person present), 0..1.
        pose_probabilities (np.ndarray): float[4] over POSE_CLASS_LABELS.
        skeleton_offsets (np.ndarray): float[17, 2] joint refinement
            offsets for the SkeletonEstimator.
    """

    presence_probability: float
    pose_probabilities: np.ndarray
    skeleton_offsets: np.ndarray


def _feature_window_to_tensors(feature_window):
    """Convert a FeatureWindow into model input tensors.

    Args:
        feature_window (FeatureWindow): Output of SignalProcessor.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (band tensor [1,1,W,B],
            auxiliary tensor [1,21]).
    """
    band_tensor = torch.from_numpy(
        feature_window.band_matrix[np.newaxis, np.newaxis, :, :]
    ).float()
    auxiliary_tensor = torch.from_numpy(
        build_auxiliary_vector(feature_window)[np.newaxis, :]
    ).float()
    return band_tensor, auxiliary_tensor


class MlEngine:
    """Loads the three models once and serves inference calls."""

    def __init__(self, ml_config):
        """Attempt to load all model weights.

        Args:
            ml_config (dict): The `ml` section of config.yaml
                (keys: model_dir, use_ml_models).
        """
        self.models_available = False
        self._presence_model = None
        self._pose_model = None
        self._skeleton_model = None
        if not ml_config.get("use_ml_models", True):
            print("ml_engine: ML disabled in config — rule-based mode.")
            return
        model_directory = PROJECT_ROOT_DIRECTORY / ml_config.get(
            "model_dir", "saved"
        )
        try:
            self._load_models(model_directory)
            self.models_available = True
            print(f"ml_engine: 3 models loaded from {model_directory}/")
        except FileNotFoundError:
            print(
                f"ml_engine: no trained models in {model_directory}/ — "
                "rule-based detection active. Run: python models/train_all.py"
            )
        except Exception as load_error:
            print(
                "ml_engine: failed to load models "
                f"({load_error}) — rule-based detection active."
            )

    def _load_models(self, model_directory):
        """Instantiate model classes and load their saved weights.

        Args:
            model_directory (Path): Directory holding the three .pt files.

        Raises:
            FileNotFoundError: If any weights file is absent.
        """
        from models.presence_model import PresenceModel
        from models.pose_model import PoseModel
        from models.skeleton_model import SkeletonModel

        weight_paths = {
            "presence": model_directory / PRESENCE_WEIGHTS_FILENAME,
            "pose": model_directory / POSE_WEIGHTS_FILENAME,
            "skeleton": model_directory / SKELETON_WEIGHTS_FILENAME,
        }
        for model_name, weights_path in weight_paths.items():
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"{model_name} weights missing: {weights_path}"
                )
        self._presence_model = PresenceModel()
        self._presence_model.load_state_dict(
            torch.load(weight_paths["presence"], map_location="cpu")
        )
        self._presence_model.eval()
        self._pose_model = PoseModel()
        self._pose_model.load_state_dict(
            torch.load(weight_paths["pose"], map_location="cpu")
        )
        self._pose_model.eval()
        self._skeleton_model = SkeletonModel()
        self._skeleton_model.load_state_dict(
            torch.load(weight_paths["skeleton"], map_location="cpu")
        )
        self._skeleton_model.eval()

    def infer(self, feature_window):
        """Run all three models on one FeatureWindow.

        Args:
            feature_window (FeatureWindow): Output of SignalProcessor.

        Returns:
            InferenceResult | None: None when models are unavailable.
        """
        if not self.models_available:
            return None
        band_tensor, auxiliary_tensor = _feature_window_to_tensors(
            feature_window
        )
        with torch.no_grad():
            presence_logits = self._presence_model(band_tensor, auxiliary_tensor)
            pose_logits = self._pose_model(band_tensor, auxiliary_tensor)
            skeleton_output = self._skeleton_model(band_tensor, auxiliary_tensor)
        presence_probability = float(torch.sigmoid(presence_logits)[0, 0])
        pose_probabilities = (
            torch.softmax(pose_logits, dim=1)[0].numpy().astype(np.float64)
        )
        skeleton_offsets = skeleton_output[0].numpy().reshape(17, 2)
        return InferenceResult(
            presence_probability=presence_probability,
            pose_probabilities=pose_probabilities,
            skeleton_offsets=skeleton_offsets,
        )


if __name__ == "__main__":
    engine_without_models = MlEngine({"model_dir": "saved_nonexistent"})
    assert engine_without_models.models_available is False
    assert engine_without_models.infer(None) is None
    print("ml_engine self-test: PASS (graceful no-model fallback)")
