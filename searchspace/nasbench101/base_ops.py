import math

import torch.nn as nn

from common.dynamic_ops import DynamicConv2d, DynamicBatchNorm2d, DynamicSequential


def truncate(inputs, channels):
    input_channels = inputs.size(1)
    if input_channels < channels:
        raise ValueError('input channel < output channels for truncate')
    elif input_channels == channels:
        return inputs   # No truncation necessary
    else:
        # Truncation should only be necessary when channel division leads to
        # vertices with +1 channels. The input vertex should always be projected to
        # the minimum channel count.
        assert input_channels - channels == 1
        return inputs[:, :channels, :, :]


def truncated_normal_(tensor, mean=0, std=1):
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(ConvBnRelu, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_bn_relu = DynamicSequential(
            DynamicConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            DynamicBatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                truncated_normal_(m.weight.data, mean=0., std=math.sqrt(1. / fan_in))
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, **kwargs):
        return self.conv_bn_relu(x, **kwargs)


class Conv3x3BnRelu(ConvBnRelu):
    def __init__(self, in_channels, out_channels):
        super(Conv3x3BnRelu, self).__init__(in_channels, out_channels, kernel_size=3, stride=1, padding=1)


class Conv1x1BnRelu(ConvBnRelu):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1BnRelu, self).__init__(in_channels, out_channels, kernel_size=1, stride=1, padding=0)


class MaxPool3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MaxPool3x3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.maxpool = nn.MaxPool2d(3, 1, 1)

    def forward(self, x, **kwargs):
        out = self.maxpool(x)
        return out


Projection = Conv1x1BnRelu
