"""Presence detection model (Tier 1) and the shared CSI backbone.

The backbone is a small 2-D CNN over the [window_frames x feature_bands]
z-scored band matrix produced by backend.signal_processor, with the three
auxiliary scalars (motion energy, baseline deviation, breathing energy)
concatenated before the head. All three WiSentry models share this
backbone architecture; pose_model and skeleton_model import it from here
so the input contract stays in one place.

Sized for CPU inference on a mid-range laptop (<1 ms per window).

Runs on: any OS with Python 3.9+. Dependencies: torch.
"""

import torch
import torch.nn as nn

WINDOW_FRAMES = 50
FEATURE_BANDS = 16
AUXILIARY_FEATURE_COUNT = 3
CONV1_CHANNELS = 8
CONV2_CHANNELS = 16
CONV_KERNEL_SIZE = 3
POOL_SIZE = 2
BACKBONE_EMBEDDING_SIZE = 64
# After two (conv + 2x2 pool) stages: 50x16 -> 25x8 -> 12x4.
_FLATTENED_CONV_SIZE = CONV2_CHANNELS * (WINDOW_FRAMES // 4) * (FEATURE_BANDS // 4)


class CsiBackbone(nn.Module):
    """Shared feature extractor: band matrix + aux scalars -> embedding."""

    def __init__(self):
        """Build the convolutional trunk and embedding layer."""
        super().__init__()
        self.convolutional_trunk = nn.Sequential(
            nn.Conv2d(1, CONV1_CHANNELS, CONV_KERNEL_SIZE, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(POOL_SIZE),
            nn.Conv2d(CONV1_CHANNELS, CONV2_CHANNELS, CONV_KERNEL_SIZE,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(POOL_SIZE),
        )
        self.embedding_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_FLATTENED_CONV_SIZE, BACKBONE_EMBEDDING_SIZE),
            nn.ReLU(),
        )

    def forward(self, band_tensor, auxiliary_tensor):
        """Compute the shared embedding.

        Args:
            band_tensor (torch.Tensor): float[batch, 1, 50, 16] z-scored
                band matrices.
            auxiliary_tensor (torch.Tensor): float[batch, 3] scalars.

        Returns:
            torch.Tensor: float[batch, BACKBONE_EMBEDDING_SIZE + 3].
        """
        convolutional_features = self.embedding_layer(
            self.convolutional_trunk(band_tensor)
        )
        return torch.cat([convolutional_features, auxiliary_tensor], dim=1)


class PresenceModel(nn.Module):
    """Binary presence classifier: outputs one logit (sigmoid => P(person))."""

    def __init__(self):
        """Build backbone + single-logit head."""
        super().__init__()
        self.backbone = CsiBackbone()
        self.head = nn.Linear(BACKBONE_EMBEDDING_SIZE + AUXILIARY_FEATURE_COUNT, 1)

    def forward(self, band_tensor, auxiliary_tensor):
        """Return presence logits, shape [batch, 1]."""
        return self.head(self.backbone(band_tensor, auxiliary_tensor))


if __name__ == "__main__":
    model = PresenceModel()
    fake_bands = torch.randn(2, 1, WINDOW_FRAMES, FEATURE_BANDS)
    fake_auxiliary = torch.randn(2, AUXILIARY_FEATURE_COUNT)
    logits = model(fake_bands, fake_auxiliary)
    assert logits.shape == (2, 1)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"presence_model self-test: PASS (output [2,1], "
          f"{parameter_count:,} parameters)")
