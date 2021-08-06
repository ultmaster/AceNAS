import torch
import torch.nn as nn
import torch.nn.functional as F

from common.dynamic_ops import DynamicBatchNorm2d, DynamicConv2d


OPS = {
    'avg_pool_3x3': lambda C: Pool('avg', C, 3, 1),
    'max_pool_2x2': lambda C: Pool('max', C, 2, 0),
    'max_pool_3x3': lambda C: Pool('max', C, 3, 1),
    'max_pool_5x5': lambda C: Pool('max', C, 5, 2),
    'max_pool_7x7': lambda C: Pool('max', C, 7, 3),
    'skip_connect': lambda C: SkipConnection(C, C),
    'sep_conv_3x3': lambda C: StackedSepConv(C, C, 3, 1),
    'sep_conv_5x5': lambda C: StackedSepConv(C, C, 5, 2),
    'sep_conv_7x7': lambda C: StackedSepConv(C, C, 7, 3),
    'dil_conv_3x3': lambda C: DilConv(C, C, 3, 2, 2),
    'dil_conv_5x5': lambda C: DilConv(C, C, 5, 4, 2),
    'dil_sep_conv_3x3': lambda C: DilSepConv(C, C, 3, 2, 2),
    'conv_1x1': lambda C: StdConv(C, C, 1, 0),
    'conv_3x1_1x3': lambda C: FacConv(C, C, 3, 1),
    'conv_3x3': lambda C: StdConv(C, C, 3, 1),
    'conv_7x1_1x7': lambda C: FacConv(C, C, 7, 3),
    'none': lambda C: Zero(),
}


class Zero(nn.Module):
    def forward(self, x, out_channels, stride):
        in_channels = x.size(1)
        if in_channels == out_channels:
            if stride == 1:
                return x.mul(0.)
            else:
                return x[:, :, ::stride, ::stride].mul(0.)
        else:
            shape = list(x.size())
            shape[1] = out_channels
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros


class StdConv(nn.Module):
    def __init__(self, max_in_channels, max_out_channels, kernel_size, padding, affine=True):
        super(StdConv, self).__init__()
        self.drop_path = DropPath_()
        self.relu = nn.ReLU()
        self.conv = DynamicConv2d(max_in_channels, max_out_channels, kernel_size, 1, padding, bias=False)
        self.bn = DynamicBatchNorm2d(max_out_channels, affine=affine)

    def forward(self, x, out_channels, stride):
        x = self.drop_path(x)
        x = self.relu(x)
        x = self.conv(x, out_channels=out_channels, stride=stride)
        x = self.bn(x)
        return x


class FactorizedReduce(nn.Module):
    def __init__(self, max_in_channels, max_out_channels):
        super(FactorizedReduce, self).__init__()
        assert max_out_channels % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = DynamicConv2d(max_in_channels, max_out_channels // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = DynamicConv2d(max_in_channels, max_out_channels // 2, 1, stride=2, padding=0, bias=False)
        self.bn = DynamicBatchNorm2d(max_out_channels)

    def forward(self, x, out_channels):
        x = self.relu(x)
        assert out_channels % 2 == 0
        out = torch.cat([self.conv_1(x, out_channels=out_channels // 2),
                         self.conv_2(x[:, :, 1:, 1:], out_channels=out_channels // 2)], dim=1)
        out = self.bn(out)
        return out


class Pool(nn.Module):
    def __init__(self, pool_type, channels, kernel_size, padding, affine=True):
        super(Pool, self).__init__()
        self.pool_type = pool_type.lower()
        self.kernel_size = kernel_size
        self.padding = padding
        self.channels = channels
        # self.bn = nn.BatchNorm2d(channels, affine=affine)
        self.drop_path = DropPath_()

    def forward(self, x, out_channels, stride):
        assert out_channels <= self.channels
        if self.pool_type == 'max':
            out = F.max_pool2d(x, self.kernel_size, stride, self.padding)
        elif self.pool_type == 'avg':
            out = F.avg_pool2d(x, self.kernel_size, stride, self.padding, count_include_pad=False)
        else:
            raise ValueError
        # out = self.bn(out)
        return self.drop_path(out)


class DilConv(nn.Module):
    def __init__(self, max_in_channels, max_out_channels, kernel_size, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.relu = nn.ReLU()
        self.dw = DynamicConv2d(max_in_channels, max_in_channels, kernel_size, 1, padding,
                                dilation=dilation, groups=max_in_channels, bias=False)
        self.pw = DynamicConv2d(max_in_channels, max_out_channels, 1, stride=1, padding=0, bias=False)
        self.bn = DynamicBatchNorm2d(max_out_channels, affine=affine)
        self.drop_path = DropPath_()

    def forward(self, x, out_channels, stride):
        x = self.relu(x)
        x = self.dw(x, out_channels=x.size(1), stride=stride)
        x = self.pw(x, out_channels=out_channels)
        x = self.bn(x)
        return self.drop_path(x)


class FacConv(nn.Module):
    """
    Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """

    def __init__(self, max_in_channels, max_out_channels, kernel_length, padding, affine=True):
        super(FacConv, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = DynamicConv2d(max_in_channels, max_in_channels, (1, kernel_length), 1, (0, padding), bias=False)
        self.conv2 = DynamicConv2d(max_in_channels, max_out_channels, (kernel_length, 1), 1, (padding, 0), bias=False)
        self.bn = DynamicBatchNorm2d(max_out_channels, affine=affine)
        self.drop_path = DropPath_()

    def forward(self, x, out_channels, stride):
        x = self.relu(x)
        x = self.conv1(x, out_channels=x.size(1), stride=(1, stride))
        x = self.conv2(x, out_channels=out_channels, stride=(stride, 1))
        x = self.bn(x)
        return self.drop_path(x)


class StackedSepConv(nn.Module):
    """
    Separable convolution stacked twice.
    """

    def __init__(self, max_in_channels, max_out_channels, kernel_size, padding, affine=True):
        super(StackedSepConv, self).__init__()
        self.relu1 = nn.ReLU(inplace=False)
        self.dw1 = DynamicConv2d(max_in_channels, max_in_channels, kernel_size=kernel_size,
                                 stride=1, padding=padding, groups=max_in_channels, bias=False)
        self.pw1 = DynamicConv2d(max_in_channels, max_in_channels, kernel_size=1, padding=0, bias=False)
        self.bn1 = DynamicBatchNorm2d(max_in_channels, affine=affine)
        self.relu2 = nn.ReLU(inplace=False)
        self.dw2 = DynamicConv2d(max_in_channels, max_in_channels, kernel_size=kernel_size,
                                 stride=1, padding=padding, groups=max_in_channels, bias=False)
        self.pw2 = DynamicConv2d(max_in_channels, max_out_channels, kernel_size=1, padding=0, bias=False)
        self.bn2 = DynamicBatchNorm2d(max_out_channels, affine=affine)
        self.drop_path = DropPath_()

    def forward(self, x, out_channels, stride):
        x = self.relu1(x)
        x = self.dw1(x, out_channels=x.size(1), stride=stride)
        x = self.pw1(x, out_channels=x.size(1))
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.dw2(x, out_channels=x.size(1))
        x = self.pw2(x, out_channels=out_channels)
        x = self.bn2(x)
        return self.drop_path(x)


class DilSepConv(nn.Module):

    def __init__(self, max_in_channels, max_out_channels, kernel_size, padding, dilation, affine=True):
        super(DilSepConv, self).__init__()
        C_in = max_in_channels
        C_out = max_out_channels
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = DynamicConv2d(
            C_in, C_in, kernel_size=kernel_size,
            padding=padding, dilation=dilation, groups=C_in, bias=False
        )
        self.conv2 = DynamicConv2d(C_in, C_in, kernel_size=1, padding=0, bias=False)
        self.bn1 = DynamicBatchNorm2d(C_in, affine=affine)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = DynamicConv2d(
            C_in, C_in, kernel_size=kernel_size,
            padding=padding, dilation=dilation, groups=C_in, bias=False
        )
        self.conv4 = DynamicConv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)
        self.bn2 = DynamicBatchNorm2d(C_out, affine=affine)
        self.drop_path = DropPath_()

    def forward(self, x, out_channels, stride):
        in_channels = x.size(1)
        x = self.relu1(x)
        x = self.conv1(x, stride=stride, out_channels=in_channels)
        x = self.conv2(x, out_channels=in_channels)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv3(x, out_channels=in_channels)
        x = self.conv4(x, out_channels=out_channels)
        x = self.bn2(x)
        return self.drop_path(x)


class SkipConnection(FactorizedReduce):
    def __init__(self, max_in_channels, max_out_channels):
        super().__init__(max_in_channels, max_out_channels)
        self.drop_path = DropPath_()

    def forward(self, x, out_channels, stride):
        if stride > 1:
            out = super(SkipConnection, self).forward(x, out_channels=out_channels)
            return self.drop_path(out)
        return x


#### utility layers ####

class DropPath_(nn.Module):
    # https://github.com/khanrc/pt.darts/blob/0.1/models/ops.py
    def __init__(self, drop_prob=0.):
        super(DropPath_, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob > 0.:
            keep_prob = 1. - self.drop_prob
            mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
            return x.div(keep_prob).mul(mask)
        return x


class ReLUConvBN(nn.Module):
    def __init__(self, max_in_channels, max_out_channels, kernel_size, padding):
        super(ReLUConvBN, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv = DynamicConv2d(max_in_channels, max_out_channels, kernel_size, padding=padding, bias=False)
        self.bn = DynamicBatchNorm2d(max_out_channels)

    def forward(self, x, out_channels):
        x = self.relu(x)
        x = self.conv(x, out_channels=out_channels)
        x = self.bn(x)
        return x
