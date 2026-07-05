import torch.nn as nn
import torch.nn.functional as F

from . import constants


def _tf_same_pad(x, kernel_size, stride):
    """Zero-pads `x` the way TensorFlow's 'SAME' padding would for `kernel_size`/`stride`, so a
    following 'valid'-padding op (conv or pool) matches TF output size for any input size."""
    _, _, h, w = x.shape
    kh, kw = kernel_size
    sh, sw = stride
    out_h = -(-h // sh)  # ceil division
    out_w = -(-w // sw)
    pad_h = max((out_h - 1) * sh + kh - h, 0)
    pad_w = max((out_w - 1) * sw + kw - w, 0)
    return F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))  # (n, c, h + pad_h, w + pad_w)


class SamePad2d(nn.Module):
    """TF-'SAME' zero padding ahead of a stride/kernel op (e.g. the root max-pool), as a standalone
    module since MaxPool2d itself only supports symmetric 'valid'-style padding."""

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)

    def forward(self, x):
        return _tf_same_pad(x, self.kernel_size, self.stride)


class StdConv(nn.Conv2d):
    """Conv2d with weight standardization (https://github.com/joe-siyuan-qiao/WeightStandardization)
    and TF-'SAME' padding applied ahead of a 'valid' convolution — MUSIQ's ResNet stem uses this in
    place of BatchNorm-normalized convs, matching the original TF/JAX implementation."""

    def forward(self, input):
        input = _tf_same_pad(input, self.kernel_size, self.stride)
        weight = self.weight
        weight = weight - weight.mean(dim=(1, 2, 3), keepdim=True)
        weight = weight / (weight.std(dim=(1, 2, 3), keepdim=True) + constants.std_conv_weight_std_eps)
        return F.conv2d(input, weight, self.bias, self.stride)  # (n, out_channels, out_h, out_w)
