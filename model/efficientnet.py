# '''EfficientNet in PyTorch.

# Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".

# Reference: https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
# '''
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import io


# def swish(x):
#     return x * x.sigmoid()


# def drop_connect(x, drop_ratio):
#     keep_ratio = 1.0 - drop_ratio
#     mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
#     mask.bernoulli_(keep_ratio)
#     x.div_(keep_ratio)
#     x.mul_(mask)
#     return x


# class SE(nn.Module):
#     '''Squeeze-and-Excitation block with Swish.'''

#     def __init__(self, in_channels, se_channels):
#         super(SE, self).__init__()
#         self.se1 = nn.Conv2d(in_channels, se_channels,
#                              kernel_size=1, bias=True)
#         self.se2 = nn.Conv2d(se_channels, in_channels,
#                              kernel_size=1, bias=True)

#     def forward(self, x):
#         out = F.adaptive_avg_pool2d(x, (1, 1))
#         out = swish(self.se1(out))
#         out = self.se2(out).sigmoid()
#         out = x * out
#         return out


# class Block(nn.Module):
#     '''expansion + depthwise + pointwise + squeeze-excitation'''

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride,
#                  expand_ratio=1,
#                  se_ratio=0.,
#                  drop_rate=0.):
#         super(Block, self).__init__()
#         self.stride = stride
#         self.drop_rate = drop_rate
#         self.expand_ratio = expand_ratio

#         # Expansion
#         channels = expand_ratio * in_channels
#         self.conv1 = nn.Conv2d(in_channels,
#                                channels,
#                                kernel_size=1,
#                                stride=1,
#                                padding=0,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(channels)

#         # Depthwise conv
#         self.conv2 = nn.Conv2d(channels,
#                                channels,
#                                kernel_size=kernel_size,
#                                stride=stride,
#                                padding=(1 if kernel_size == 3 else 2),
#                                groups=channels,
#                                bias=False)
#         self.bn2 = nn.BatchNorm2d(channels)

#         # SE layers
#         se_channels = int(in_channels * se_ratio)
#         self.se = SE(channels, se_channels)

#         # Output
#         self.conv3 = nn.Conv2d(channels,
#                                out_channels,
#                                kernel_size=1,
#                                stride=1,
#                                padding=0,
#                                bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels)

#         # Skip connection if in and out shapes are the same (MV-V2 style)
#         self.has_skip = (stride == 1) and (in_channels == out_channels)

#     def forward(self, x):
#         out = x if self.expand_ratio == 1 else swish(self.bn1(self.conv1(x)))
#         out = swish(self.bn2(self.conv2(out)))
#         out = self.se(out)
#         out = self.bn3(self.conv3(out))
#         if self.has_skip:
#             if self.training and self.drop_rate > 0:
#                 out = drop_connect(out, self.drop_rate)
#             out = out + x
#         return out


# class EfficientNet(nn.Module):
#     def __init__(self, cfg, num_classes=10):
#         super(EfficientNet, self).__init__()
#         self.cfg = cfg
#         self.conv1 = nn.Conv2d(3,
#                                32,
#                                kernel_size=3,
#                                stride=1,
#                                padding=1,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.layers = self._make_layers(in_channels=32)
#         self.linear = nn.Linear(cfg['out_channels'][-1], num_classes)

#     def _make_layers(self, in_channels):
#         layers = []
#         cfg = [self.cfg[k] for k in ['expansion', 'out_channels', 'num_blocks', 'kernel_size',
#                                      'stride']]
#         b = 0
#         blocks = sum(self.cfg['num_blocks'])
#         for expansion, out_channels, num_blocks, kernel_size, stride in zip(*cfg):
#             strides = [stride] + [1] * (num_blocks - 1)
#             for stride in strides:
#                 drop_rate = self.cfg['drop_connect_rate'] * b / blocks
#                 layers.append(
#                     Block(in_channels,
#                           out_channels,
#                           kernel_size,
#                           stride,
#                           expansion,
#                           se_ratio=0.25,
#                           drop_rate=drop_rate))
#                 in_channels = out_channels
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = swish(self.bn1(self.conv1(x)))
#         out = self.layers(out)
#         out = F.adaptive_avg_pool2d(out, 1)
#         out = out.view(out.size(0), -1)
#         dropout_rate = self.cfg['dropout_rate']
#         if self.training and dropout_rate > 0:
#             out = F.dropout(out, p=dropout_rate)
#         out = self.linear(out)
#         return out


# def EfficientNetB0():
#     cfg = {
#         'num_blocks': [1, 2, 2, 3, 3, 4, 1],
#         'expansion': [1, 6, 6, 6, 6, 6, 6],
#         'out_channels': [16, 24, 40, 80, 112, 192, 320],
#         'kernel_size': [3, 3, 5, 3, 5, 5, 3],
#         'stride': [1, 2, 2, 2, 1, 2, 1],
#         'dropout_rate': 0.2,
#         'drop_connect_rate': 0.2,
#     }
#     return EfficientNet(cfg)

# def get_model_storage_size(model):
#     """Returns the storage size of the model in MB."""
#     # Create a buffer to save the model to
#     buffer = io.BytesIO()
    
#     # Save the model's state dict to the buffer (without saving to disk)
#     torch.save(model, buffer)
    
#     # Get the size of the buffer in bytes and convert to MB
#     buffer_size = buffer.tell()  # in bytes
#     storage_size_mb = buffer_size / (1024 * 1024)  # convert to MB
#     return storage_size_mb


# def test():
#     net = EfficientNetB0()
#     x = torch.randn(2, 3, 128, 128)
#     y = net(x)
#     print(y.shape)
#     print(net)
#     print(get_model_storage_size(net))


# if __name__ == '__main__':
#     test()
"""
Creates a EfficientNetV2 Model as defined in:
Mingxing Tan, Quoc V. Le. (2021). 
EfficientNetV2: Smaller Models and Faster Training
arXiv preprint arXiv:2104.00298.
import from https://github.com/d-li14/mobilenetv2.pytorch
"""

import torch
import torch.nn as nn
import math

__all__ = ['effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

 
class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Module):
    def __init__(self, cfgs, num_classes=10, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def effnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    # cfgs = [
    #     # t, c, n, s, SE
    #     [1,  24,  2, 1, 0],
    #     [4,  48,  4, 2, 0],
    #     [4,  64,  4, 2, 0],
    #     [4, 128,  6, 2, 1],
    #     [6, 160,  9, 1, 1],
    #     [6, 256, 15, 2, 1],
    # ]
    cfgs = [
        # t,  c,   n, s, SE
        [1,  16,  2, 1, 0],  # Stem (24 → 16)
        [3,  32,  3, 2, 0],  # Stage 2 (48 → 32, 4 → 3 blocks)
        [3,  48,  3, 2, 0],  # Stage 3 (64 → 48, 4 → 3 blocks)
        [3,  96,  4, 2, 1],  # Stage 4 (128 → 96, 6 → 4 blocks)
        [4, 128,  6, 1, 1],  # Stage 5 (160 → 128, 9 → 6 blocks)
        [4, 192,  8, 2, 1],  # Stage 6 (256 → 192, 15 → 8 blocks)
    ]
    return EffNetV2(cfgs, **kwargs)

    
import io

def get_model_storage_size(model):
    """Returns the storage size of the model in MB."""
    # Create a buffer to save the model to
    buffer = io.BytesIO()
    
    # Save the model's state dict to the buffer (without saving to disk)
    torch.save(model, buffer)
    
    # Get the size of the buffer in bytes and convert to MB
    buffer_size = buffer.tell()  # in bytes
    storage_size_mb = buffer_size / (1024 * 1024)  # convert to MB
    return storage_size_mb


def test():
    net = effnetv2_s()
    x = torch.randn(2, 3, 128, 128)
    y = net(x)
    print(y.shape)
    print(net)
    print(get_model_storage_size(net))

if __name__ == '__main__':
    test()