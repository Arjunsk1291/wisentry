"""Skeleton refinement model (Tier 3, EXPERIMENTAL).

Regresses 17x2 joint offsets that the SkeletonEstimator adds on top of
its calibrated pose template (backend.skeleton). Outputs are tanh-bounded
to MAX_OFFSET so the model can only nudge joints, never invent anatomy —
an honest framing of what commodity-ESP32 CSI supports (ENGINEERING_SPEC.md §2).

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

KEYPOINT_COUNT = 17
OUTPUT_SIZE = KEYPOINT_COUNT * 2  # (x, y) per joint
HEAD_HIDDEN_SIZE = 64
MAX_OFFSET = 0.08  # normalized units; matches skeleton.MODEL_REFINEMENT_SCALE


class SkeletonModel(nn.Module):
    """Joint-offset regressor over the shared CSI embedding."""

    def __init__(self):
        """Build backbone + bounded regression head."""
        super().__init__()
        self.backbone = CsiBackbone()
        self.head = nn.Sequential(
            nn.Linear(BACKBONE_EMBEDDING_SIZE + AUXILIARY_FEATURE_COUNT,
                      HEAD_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HEAD_HIDDEN_SIZE, OUTPUT_SIZE),
            nn.Tanh(),
        )

    def forward(self, band_tensor, auxiliary_tensor):
        """Return joint offsets, shape [batch, 34], bounded to ±MAX_OFFSET."""
        return MAX_OFFSET * self.head(
            self.backbone(band_tensor, auxiliary_tensor)
        )


if __name__ == "__main__":
    model = SkeletonModel()
    fake_bands = torch.randn(2, 1, WINDOW_FRAMES, FEATURE_BANDS)
    fake_auxiliary = torch.randn(2, AUXILIARY_FEATURE_COUNT)
    with torch.no_grad():
        offsets = model(fake_bands, fake_auxiliary)
    assert offsets.shape == (2, OUTPUT_SIZE)
    assert float(offsets.abs().max()) <= MAX_OFFSET + 1e-6
    print("skeleton_model self-test: PASS (output [2,34], bounded)")
