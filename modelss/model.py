"""
MobileNetV2-style Semantic Segmentation CNN for LULC Classification.

Architecture highlights:
  ─ Depthwise Separable Convolution blocks
  ─ Inverted Residual Blocks (MobileNetV2)
  ─ Global Average Pooling
  ─ Lightweight decoder with bilinear upsampling

All 7 input bands (B2, B3, B4, B5, B6, B7, NDVI) are processed as
channels of the input tensor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_INPUT_CHANNELS, NUM_CLASSES, EXPANSION_RATIO, ENCODER_CHANNELS, PATCH_SIZE


# =============================================================================
# Building Blocks
# =============================================================================

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution:
      1. Depthwise conv: a single filter per input channel (groups=in_channels).
      2. Pointwise conv: 1×1 convolution to mix cross-channel information.

    Significantly reduces the number of parameters and FLOPs compared to
    standard convolutions while maintaining representational capacity.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=in_channels, bias=bias,
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)

        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=bias,
        )
        self.bn_pw = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn_dw(self.depthwise(x)))
        x = self.relu(self.bn_pw(self.pointwise(x)))
        return x


class InvertedResidualBlock(nn.Module):
    """
    Inverted Residual Block from MobileNetV2.

    Structure:
      1. Pointwise (1×1): Expand channels by expansion_ratio.
      2. Depthwise (3×3): Spatial filtering on expanded channels.
      3. Pointwise (1×1): Project back to output channels (linear, no ReLU).
      4. Residual connection when stride=1 and in_channels == out_channels.

    The "inverted" design expands the representation in the hidden layer
    (unlike traditional residual blocks that bottleneck), enabling depthwise
    convolutions to operate on a richer feature space.
    """

    def __init__(self, in_channels, out_channels, stride=1,
                 expansion_ratio=EXPANSION_RATIO):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expansion_ratio

        layers = []

        # 1. Expansion (pointwise)
        if expansion_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])

        # 2. Depthwise
        layers.extend([
            nn.Conv2d(
                hidden_dim, hidden_dim,
                kernel_size=3, stride=stride, padding=1,
                groups=hidden_dim, bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])

        # 3. Projection (pointwise, LINEAR — no activation)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        else:
            return self.block(x)


class GlobalAveragePoolingBridge(nn.Module):
    """
    Global Average Pooling bridge that:
      1. Applies GAP to produce a (B, C, 1, 1) feature vector.
      2. Processes through a small FC bottleneck.
      3. Broadcasts back to spatial dimensions via expand + concat.

    This fuses global context into the local feature map, acting as a
    bridge between the encoder and decoder.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        w = self.gap(x)          # (B, C, 1, 1)
        w = self.fc(w)           # (B, C, 1, 1)
        return x * w             # Channel-wise re-weighting


# =============================================================================
# Encoder
# =============================================================================

class Encoder(nn.Module):
    """
    MobileNetV2-inspired encoder that progressively downsamples the input.

    Stage 0: in_channels → ENCODER_CHANNELS[0]  (stride 1, no downsampling)
    Stage 1: → ENCODER_CHANNELS[1]              (stride 2, downsample)
    Stage 2: → ENCODER_CHANNELS[2]              (stride 2, downsample)
    Stage 3: → ENCODER_CHANNELS[3]              (stride 2, downsample)

    Each stage consists of one strided Inverted Residual Block followed
    by one identity Inverted Residual Block for additional representation.
    """

    def __init__(self, in_channels=NUM_INPUT_CHANNELS):
        super().__init__()

        # Initial stem: standard conv to lift raw bands to first feature dimension
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, ENCODER_CHANNELS[0], kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ENCODER_CHANNELS[0]),
            nn.ReLU6(inplace=True),
        )

        # Build encoder stages
        self.stages = nn.ModuleList()
        channels = [ENCODER_CHANNELS[0]] + list(ENCODER_CHANNELS)

        for i in range(1, len(channels)):
            c_in = channels[i - 1]
            c_out = channels[i]
            stride = 2 if i >= 1 else 1

            stage = nn.Sequential(
                InvertedResidualBlock(c_in, c_out, stride=stride),
                InvertedResidualBlock(c_out, c_out, stride=1),
            )
            self.stages.append(stage)

    def forward(self, x):
        """
        Returns
        -------
        features : list of tensors at each stage (including stem output)
        """
        features = []
        x = self.stem(x)
        features.append(x)

        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features


# =============================================================================
# Decoder
# =============================================================================

class DecoderBlock(nn.Module):
    """Single decoder block: upsample + concat skip + DepthwiseSeparableConv."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = DepthwiseSeparableConv(
            in_channels + skip_channels, out_channels,
            kernel_size=3, padding=1,
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                          align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    """
    Lightweight decoder that progressively upsamples and fuses skip connections.
    """

    def __init__(self):
        super().__init__()
        # channels: [32, 32, 64, 128, 256] from encoder features
        enc = [ENCODER_CHANNELS[0]] + list(ENCODER_CHANNELS)

        # Decoder goes from deepest to shallowest
        # Stage 3 (256) + skip from stage 2 (128) → 128
        # Stage 2 (128) + skip from stage 1 (64)  → 64
        # Stage 1 (64)  + skip from stage 0 (32)  → 32
        # Stage 0 (32)  + skip from stem (32)      → 32

        self.blocks = nn.ModuleList([
            DecoderBlock(enc[4], enc[3], enc[3]),   # 256+128 → 128
            DecoderBlock(enc[3], enc[2], enc[2]),   # 128+64  → 64
            DecoderBlock(enc[2], enc[1], enc[1]),   # 64+32   → 32
            DecoderBlock(enc[1], enc[0], enc[0]),   # 32+32   → 32
        ])

    def forward(self, features):
        """
        Parameters
        ----------
        features : list of tensors [stem, stage1, stage2, stage3, stage4]
        """
        x = features[-1]  # deepest features

        for i, block in enumerate(self.blocks):
            skip_idx = len(features) - 2 - i
            skip = features[skip_idx]
            x = block(x, skip)

        return x


# =============================================================================
# Full Segmentation Model
# =============================================================================

class LULCSegmentationNet(nn.Module):
    """
    MobileNetV2-style Semantic Segmentation Network for LULC classification.

    Architecture:
      1. Encoder with Inverted Residual Blocks (depthwise separable convolutions)
      2. Global Average Pooling bridge for global context
      3. Decoder with skip connections and bilinear upsampling
      4. Per-pixel classification head

    Input:  (B, 7, H, W) — 7 spectral bands including NDVI
    Output: (B, NUM_CLASSES, H, W) — per-pixel class logits
    """

    def __init__(self, in_channels=NUM_INPUT_CHANNELS, num_classes=NUM_CLASSES):
        super().__init__()

        self.encoder = Encoder(in_channels=in_channels)
        self.gap_bridge = GlobalAveragePoolingBridge(ENCODER_CHANNELS[-1])
        self.decoder = Decoder()

        # Classification head
        self.classifier = nn.Sequential(
            nn.Conv2d(ENCODER_CHANNELS[0], ENCODER_CHANNELS[0],
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ENCODER_CHANNELS[0]),
            nn.ReLU6(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(ENCODER_CHANNELS[0], num_classes, kernel_size=1),
        )

    def forward(self, x):
        # Encoder
        features = self.encoder(x)

        # Apply Global Average Pooling bridge on the deepest features
        features[-1] = self.gap_bridge(features[-1])

        # Decoder
        x = self.decoder(features)

        # Classification
        logits = self.classifier(x)

        return logits


# =============================================================================
# Utility: model summary
# =============================================================================

def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Quick test
    model = LULCSegmentationNet()
    total, trainable = count_parameters(model)
    print(f"Model: LULCSegmentationNet")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Input:  (B, {NUM_INPUT_CHANNELS}, {PATCH_SIZE}, {PATCH_SIZE})")
    print(f"  Output: (B, {NUM_CLASSES}, {PATCH_SIZE}, {PATCH_SIZE})")

    # Forward pass test
    x = torch.randn(2, NUM_INPUT_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    y = model(x)
    print(f"\n  Test forward pass:")
    print(f"    Input shape:  {x.shape}")
    print(f"    Output shape: {y.shape}")
    assert y.shape == (2, NUM_CLASSES, PATCH_SIZE, PATCH_SIZE), "Shape mismatch!"
    print("  ✓ Forward pass OK")
