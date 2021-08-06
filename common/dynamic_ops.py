"""
This file is to provide a wrapper for common CNN operators, to support mutable channels.
So that, we can use

y = self.conv(x, out_features=128)
"""

import torch.nn as nn
import torch.nn.functional as F


class DynamicConv2d(nn.Conv2d):
    _static_mode = False
    _dry_run_stride = 1

    def forward(self, input, **kwargs):
        if self._static_mode:
            return super().forward(input)
        out_channels = kwargs.get('out_channels', kwargs.get('out_features', self.out_channels))
        stride = kwargs.get('stride', self.stride)
        in_channels = input.size(1)
        assert out_channels <= self.out_channels and in_channels <= self.in_channels
        assert self.groups in [1, self.in_channels]
        if self.groups == 1:
            weight = self.weight[:out_channels, :in_channels].contiguous()
            groups = 1
        else:
            weight = self.weight[:out_channels].contiguous()
            groups = out_channels
        if stride is None:
            stride = self.stride
        self._dry_run_stride = stride
        return F.conv2d(input, weight, self.bias[:out_channels] if self.bias is not None else None,
                        stride, self.padding, self.dilation, groups)


class DynamicBatchNorm2d(nn.BatchNorm2d):
    _static_mode = False

    def forward(self, input, **kwargs):
        if self._static_mode:
            return super().forward(input)
        out_channels = input.size(1)
        assert out_channels <= self.num_features
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        """ Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean[:out_channels] if self.running_mean is not None and \
                (not self.training or self.track_running_stats) else None,
            self.running_var[:out_channels] if self.running_var is not None and \
                (not self.training or self.track_running_stats) else None,
            self.weight[:out_channels], self.bias[:out_channels], bn_training, exponential_average_factor, self.eps)


class DynamicLinear(nn.Linear):
    _static_mode = False

    def forward(self, input, **kwargs):
        if self._static_mode:
            return super().forward(input)
        in_features = input.size(1)
        out_features = kwargs.get('out_features', self.out_features)
        assert out_features <= self.out_features and in_features <= self.in_features
        return F.linear(input, self.weight[:out_features, :in_features].contiguous(),
                        self.bias[:out_features] if self.bias is not None else None)


class DynamicSequential(nn.Sequential):
    _static_mode = False

    def forward(self, input, **kwargs):
        if self._static_mode:
            return super().forward(input)
        for module in self:
            if isinstance(module, (DynamicConv2d, DynamicBatchNorm2d, DynamicLinear)):
                input = module(input, **kwargs)
            else:
                input = module(input)
        return input


class ResizableSequential(nn.Sequential):
    _static_mode = False

    def forward(self, input, ratio):
        if self._static_mode:
            return super().forward(input)
        for module in self:
            if isinstance(module, DynamicConv2d):
                input = module(input, out_channels=int(module.out_channels * ratio))
            else:
                input = module(input)
        return input
