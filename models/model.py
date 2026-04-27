
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_INPUT_CHANNELS, NUM_CLASSES, EXPANSION_RATIO, ENCODER_CHANNELS

class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride=stride, padding=padding, groups=in_ch, bias=bias)
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=bias)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn_dw(self.depthwise(x)))
        x = self.relu(self.bn_pw(self.pointwise(x)))
        return x

class InvertedResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, expansion_ratio=EXPANSION_RATIO):
        super().__init__()
        self.use_residual = ((stride == 1) and (in_ch == out_ch))
        hidden = (in_ch * expansion_ratio)
        layers = []
        if (expansion_ratio != 1):
            layers += [nn.Conv2d(in_ch, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)]
        layers += [nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)]
        layers += [nn.Conv2d(hidden, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return (x + self.block(x))
        return self.block(x)

class GlobalAveragePoolingBridge(nn.Module):

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max((channels // reduction), 16)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(channels, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU6(inplace=True), nn.Conv2d(mid, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.Sigmoid())

    def forward(self, x):
        return (x * self.fc(self.gap(x)))

class Encoder(nn.Module):

    def __init__(self, in_channels=NUM_INPUT_CHANNELS):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, ENCODER_CHANNELS[0], 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(ENCODER_CHANNELS[0]), nn.ReLU6(inplace=True))
        self.stages = nn.ModuleList()
        channels = ([ENCODER_CHANNELS[0]] + list(ENCODER_CHANNELS))
        for i in range(1, len(channels)):
            stage = nn.Sequential(InvertedResidualBlock(channels[i - 1], channels[i], stride=2), InvertedResidualBlock(channels[i], channels[i], stride=1))
            self.stages.append(stage)

    def forward(self, x):
        features = []
        x = self.stem(x)
        features.append(x)
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

class DecoderBlock(nn.Module):

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = DepthwiseSeparableConv((in_ch + skip_ch), out_ch, kernel_size=3, padding=1)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        enc = ([ENCODER_CHANNELS[0]] + list(ENCODER_CHANNELS))
        self.blocks = nn.ModuleList([DecoderBlock(enc[4], enc[3], enc[3]), DecoderBlock(enc[3], enc[2], enc[2]), DecoderBlock(enc[2], enc[1], enc[1]), DecoderBlock(enc[1], enc[0], enc[0])])

    def forward(self, features):
        x = features[- 1]
        for (i, block) in enumerate(self.blocks):
            skip = features[(len(features) - 2) - i]
            x = block(x, skip)
        return x

class LULCSegmentationNet(nn.Module):

    def __init__(self, in_channels=NUM_INPUT_CHANNELS, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.gap_bridge = GlobalAveragePoolingBridge(ENCODER_CHANNELS[- 1])
        self.decoder = Decoder()
        self.classifier = nn.Sequential(nn.Conv2d(ENCODER_CHANNELS[0], ENCODER_CHANNELS[0], 3, padding=1, bias=False), nn.BatchNorm2d(ENCODER_CHANNELS[0]), nn.ReLU6(inplace=True), nn.Dropout2d(p=0.1), nn.Conv2d(ENCODER_CHANNELS[0], num_classes, 1))

    def forward(self, x):
        features = self.encoder(x)
        features[- 1] = self.gap_bridge(features[- 1])
        x = self.decoder(features)
        return self.classifier(x)

def count_parameters(model):
    total = sum((p.numel() for p in model.parameters()))
    trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
    return (total, trainable)
