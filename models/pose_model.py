"""Pose classification model (Tier 2): standing / sitting / lying / walking.

Shares the CsiBackbone defined in models.presence_model. Outputs four
logits in the order of backend.ml_engine.POSE_CLASS_LABELS.

Runs on: any OS with Python 3.9+. Dependencies: torch.
"""

import torch
import torch.nn as nn

from models.presence_model import (
    AUXILIARY_FEATURE_COUNT,
    BACKBONE_EMBEDDING_SIZE,
    FEATURE_BANDS,
    WINDOW_FRAMES,
    CsiBackbone,
)

POSE_CLASS_COUNT = 4  # standing, sitting, lying, walking
HEAD_HIDDEN_SIZE = 32


class PoseModel(nn.Module):
    """4-class pose classifier over the shared CSI embedding."""

    def __init__(self):
        """Build backbone + small MLP head."""
        super().__init__()
        self.backbone = CsiBackbone()
        self.head = nn.Sequential(
            nn.Linear(BACKBONE_EMBEDDING_SIZE + AUXILIARY_FEATURE_COUNT,
                      HEAD_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HEAD_HIDDEN_SIZE, POSE_CLASS_COUNT),
        )

    def forward(self, band_tensor, auxiliary_tensor):
        """Return pose logits, shape [batch, 4]."""
        return self.head(self.backbone(band_tensor, auxiliary_tensor))


if __name__ == "__main__":
    model = PoseModel()
    fake_bands = torch.randn(2, 1, WINDOW_FRAMES, FEATURE_BANDS)
    fake_auxiliary = torch.randn(2, AUXILIARY_FEATURE_COUNT)
    logits = model(fake_bands, fake_auxiliary)
    assert logits.shape == (2, POSE_CLASS_COUNT)
    print("pose_model self-test: PASS (output [2,4])")
