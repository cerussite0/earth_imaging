
import math
import re
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock

def initialize_decoder(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if (m.bias is not None):
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if (m.bias is not None):
                nn.init.constant_(m.bias, 0)

class Conv2dBnReLU(nn.Sequential):

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=0, stride=1):
        super().__init__(nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

class UNetDecoderBlock(nn.Module):

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = Conv2dBnReLU((in_ch + skip_ch), out_ch, kernel_size=3, padding=1)
        self.conv2 = Conv2dBnReLU(out_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x, target_h, target_w, skip=None):
        x = F.interpolate(x, size=(target_h, target_w), mode='nearest')
        if (skip is not None):
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))

class UNetDecoder(nn.Module):

    def __init__(self, encoder_channels, decoder_channels=(256, 128, 64, 32)):
        super().__init__()
        enc = list(encoder_channels[1:])[::(- 1)]
        head = enc[0]
        in_ch = ([head] + list(decoder_channels[:(- 1)]))
        skip_ch = (list(enc[1:]) + [0])
        self.blocks = nn.ModuleList()
        for (ic, sc, oc) in zip(in_ch, skip_ch, decoder_channels):
            self.blocks.append(UNetDecoderBlock(ic, sc, oc))

    def forward(self, features):
        shapes = [f.shape[2:] for f in features][::(- 1)]
        feats = features[1:][::(- 1)]
        (x, skips) = (feats[0], feats[1:])
        for (i, block) in enumerate(self.blocks):
            (h, w) = shapes[i + 1]
            skip = (skips[i] if (i < len(skips)) else None)
            x = block(x, h, w, skip)
        return x

class SegmentationHead(nn.Sequential):

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__(nn.Conv2d(in_ch, out_ch, kernel_size, padding=(kernel_size // 2)))

class ResNet18Encoder(ResNet):

    def __init__(self, in_channels=7):
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2])
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        del self.fc, self.avgpool, self.layer4
        self.out_channels = [in_channels, 64, 64, 128, 256]

    def forward(self, x):
        features = [x]
        x = self.relu(self.bn1(self.conv1(x)))
        features.append(x)
        x = self.layer1(self.maxpool(x))
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        return features
GlobalParams = collections.namedtuple('GlobalParams', ['width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate', 'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon', 'drop_connect_rate', 'depth_divisor', 'min_depth', 'include_top'])
GlobalParams.__new__.__defaults__ = ((None,) * len(GlobalParams._fields))
BlockArgs = collections.namedtuple('BlockArgs', ['num_repeat', 'kernel_size', 'stride', 'expand_ratio', 'input_filters', 'output_filters', 'se_ratio', 'id_skip'])
BlockArgs.__new__.__defaults__ = ((None,) * len(BlockArgs._fields))

def _round_filters(filters, gp):
    m = gp.width_coefficient
    if (not m):
        return filters
    d = gp.depth_divisor
    md = (gp.min_depth or d)
    f = max(md, ((int(((filters * m) + (d / 2))) // d) * d))
    if (f < ((0.9 * filters) * m)):
        f += d
    return int(f)

def _round_repeats(repeats, gp):
    m = gp.depth_coefficient
    return (int(math.ceil((m * repeats))) if m else repeats)

def _drop_connect(x, p, training):
    if (not training):
        return x
    keep = (1 - p)
    r = (keep + torch.rand([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device))
    return ((x / keep) * torch.floor(r))

class _SamePadConv2d(nn.Conv2d):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_ch, out_ch, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = (self.stride if (len(self.stride) == 2) else ([self.stride[0]] * 2))

    def forward(self, x):
        (ih, iw) = x.size()[(- 2):]
        (kh, kw) = self.weight.size()[(- 2):]
        (sh, sw) = self.stride
        pad_h = max((((((math.ceil((ih / sh)) - 1) * sh) + ((kh - 1) * self.dilation[0])) + 1) - ih), 0)
        pad_w = max((((((math.ceil((iw / sw)) - 1) * sw) + ((kw - 1) * self.dilation[1])) + 1) - iw), 0)
        if ((pad_h > 0) or (pad_w > 0)):
            x = F.pad(x, [(pad_w // 2), (pad_w - (pad_w // 2)), (pad_h // 2), (pad_h - (pad_h // 2))])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class MBConvBlock(nn.Module):

    def __init__(self, ba, gp):
        super().__init__()
        bn_mom = (1 - gp.batch_norm_momentum)
        bn_eps = gp.batch_norm_epsilon
        inp = ba.input_filters
        exp = (inp * ba.expand_ratio)
        oup = ba.output_filters
        self._has_expansion = (ba.expand_ratio != 1)
        self._has_se = ((ba.se_ratio is not None) and (0 < ba.se_ratio <= 1))
        self._has_skip = (ba.id_skip and (ba.stride == 1) and (inp == oup))
        if self._has_expansion:
            self._expand = _SamePadConv2d(inp, exp, 1, bias=False)
            self._bn0 = nn.BatchNorm2d(exp, momentum=bn_mom, eps=bn_eps)
        else:
            self._expand = self._bn0 = nn.Identity()
        self._dw = _SamePadConv2d(exp, exp, ba.kernel_size, stride=ba.stride, groups=exp, bias=False)
        self._bn1 = nn.BatchNorm2d(exp, momentum=bn_mom, eps=bn_eps)
        if self._has_se:
            sq = max(1, int((inp * ba.se_ratio)))
            self._se_reduce = _SamePadConv2d(exp, sq, 1)
            self._se_expand = _SamePadConv2d(sq, exp, 1)
        self._project = _SamePadConv2d(exp, oup, 1, bias=False)
        self._bn2 = nn.BatchNorm2d(oup, momentum=bn_mom, eps=bn_eps)
        self._swish = nn.SiLU()

    def forward(self, inputs, drop_rate=None):
        x = inputs
        if self._has_expansion:
            x = self._swish(self._bn0(self._expand(inputs)))
        x = self._swish(self._bn1(self._dw(x)))
        if self._has_se:
            s = self._swish(self._se_reduce(F.adaptive_avg_pool2d(x, 1)))
            x = (torch.sigmoid(self._se_expand(s)) * x)
        x = self._bn2(self._project(x))
        if self._has_skip:
            if (drop_rate and (drop_rate > 0)):
                x = _drop_connect(x, drop_rate, self.training)
            x = (x + inputs)
        return x

def _decode_block_string(s):
    ops = {}
    for op in s.split('_'):
        parts = re.split('(\\d.*)', op)
        if (len(parts) >= 2):
            ops[parts[0]] = parts[1]
    return BlockArgs(num_repeat=int(ops['r']), kernel_size=int(ops['k']), stride=[int(ops['s'][0])], expand_ratio=int(ops['e']), input_filters=int(ops['i']), output_filters=int(ops['o']), se_ratio=(float(ops['se']) if ('se' in ops) else None), id_skip=('noskip' not in s))

class EfficientNetB0Encoder(nn.Module):
    _BLOCKS = ['r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25', 'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25', 'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25', 'r1_k3_s11_e6_i192_o320_se0.25']

    def __init__(self, in_channels=7):
        super().__init__()
        gp = GlobalParams(width_coefficient=1.0, depth_coefficient=1.0, image_size=None, dropout_rate=0.2, num_classes=1000, batch_norm_momentum=0.99, batch_norm_epsilon=0.001, drop_connect_rate=0.2, depth_divisor=8, min_depth=None, include_top=False)
        self._gp = gp
        (bn_mom, bn_eps) = ((1 - gp.batch_norm_momentum), gp.batch_norm_epsilon)
        stem_ch = _round_filters(32, gp)
        self._conv_stem = _SamePadConv2d(in_channels, stem_ch, 3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(stem_ch, momentum=bn_mom, eps=bn_eps)
        self._swish = nn.SiLU()
        self._blocks = nn.ModuleList()
        for bs in self._BLOCKS:
            ba = _decode_block_string(bs)
            ba = ba._replace(input_filters=_round_filters(ba.input_filters, gp), output_filters=_round_filters(ba.output_filters, gp), num_repeat=_round_repeats(ba.num_repeat, gp))
            self._blocks.append(MBConvBlock(ba, gp))
            if (ba.num_repeat > 1):
                ba = ba._replace(input_filters=ba.output_filters, stride=1)
            for _ in range((ba.num_repeat - 1)):
                self._blocks.append(MBConvBlock(ba, gp))
        self._dcr = gp.drop_connect_rate
        self._out_indexes = [2, 4, 8]
        self.out_channels = [in_channels, 32, 24, 40, 112]

    def forward(self, x):
        features = [x]
        x = self._swish(self._bn0(self._conv_stem(x)))
        features.append(x)
        depth = 1
        for (i, block) in enumerate(self._blocks):
            x = block(x, ((self._dcr * i) / len(self._blocks)))
            if (i in self._out_indexes):
                features.append(x)
                depth += 1
            if (depth > 4):
                break
        return features[:5]

class MobileNetV2Encoder(nn.Module):

    def __init__(self, in_channels=7):
        super().__init__()
        import torchvision
        base = torchvision.models.mobilenet_v2(weights=None)
        base.features[0][0] = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False)
        self.features = base.features
        self._out_indexes = [1, 3, 6, 13]
        self.out_channels = [in_channels, 16, 24, 32, 96]

    def forward(self, x):
        features = [x]
        depth = 0
        for (i, module) in enumerate(self.features):
            x = module(x)
            if (i in self._out_indexes):
                features.append(x)
                depth += 1
                if (depth >= 4):
                    break
        return features

class UNet(nn.Module):

    def __init__(self, encoder, num_classes, decoder_channels=(256, 128, 64, 32), dropout_rate=0.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = UNetDecoder(encoder.out_channels, decoder_channels)
        self.dropout = (nn.Dropout2d(p=dropout_rate) if (dropout_rate > 0) else nn.Identity())
        self.segmentation_head = SegmentationHead(decoder_channels[- 1], num_classes)
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)

    def forward(self, x):
        return self.segmentation_head(self.dropout(self.decoder(self.encoder(x))))

def build_unet(arch_name, in_channels, num_classes, dropout_rate=0.0):
    encoders = {'resnet18': ResNet18Encoder, 'efficientnet-b0': EfficientNetB0Encoder, 'mobilenet_v2': MobileNetV2Encoder}
    encoder = encoders[arch_name](in_channels=in_channels)
    return UNet(encoder, num_classes, dropout_rate=dropout_rate)
if (__name__ == '__main__'):
    for name in ['resnet18', 'efficientnet-b0', 'mobilenet_v2']:
        model = build_unet(name, in_channels=7, num_classes=10)
        x = torch.randn(2, 7, 128, 128)
        y = model(x)
        params = sum((p.numel() for p in model.parameters() if p.requires_grad))
        print(f'{name:20s} | output: {tuple(y.shape)} | params: {params:,}')
    print('All architectures passed.')
