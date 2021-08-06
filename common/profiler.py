import logging

import torch
import torch.nn as nn
from mmcv.utils.logging import print_log

from .dynamic_ops import DynamicConv2d, DynamicLinear, DynamicBatchNorm2d


__all__ = ['flops_params_counter']


def count_convNd(m, _, y):
    cin = m.in_channels
    kernel_ops = m.weight.size()[2] * m.weight.size()[3]
    ops_per_element = kernel_ops
    output_elements = y.nelement()
    total_ops = cin * output_elements * ops_per_element // m.groups  # cout x oW x oH
    m.total_ops = total_ops
    m.module_used = True


def count_linear(m, _, __):
    total_ops = m.in_features * m.out_features
    m.total_ops = total_ops
    m.module_used = True


def count_naive(m, _, __):
    m.module_used = True


def count_dynamic_conv2d(m, x, y):
    cin, cout = x[0].size(1), y.size(1)
    kernel_ops = m.weight.size()[2] * m.weight.size()[3]
    ops_per_element = kernel_ops
    output_elements = y.nelement()
    if m.groups == 1:
        total_ops = cin * output_elements * ops_per_element // m.groups  # cout x oW x oH
        total_params = cin * cout * kernel_ops
    else:
        # assume number of groups == number of input channels
        total_ops = output_elements * ops_per_element
        total_params = cout * kernel_ops
    if m.bias is not None:
        total_params += cout
    m.total_ops = total_ops
    m.total_params = total_params
    m.module_used = True


def count_dynamic_linear(m, x, y):
    m.total_ops = x[0].size(1) * y.size(1)
    m.total_params = x[0].size(1) * y.size(1)
    if m.bias is not None:
        m.total_params += y.size(1)
    m.module_used = True


def count_dynamic_bn2d(m, _, y):
    out_channels = y.size(1)
    m.total_params = (m.total_params * out_channels // m.num_features)
    m.module_used = True


register_hooks = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.Linear: count_linear,
    DynamicConv2d: count_dynamic_conv2d,
    DynamicLinear: count_dynamic_linear,
    DynamicBatchNorm2d: count_dynamic_bn2d
}


def flops_params_counter(model, input_size, suppress_warnings=False):
    handler_collection = []

    def add_hooks(m_):
        if len(list(m_.children())) > 0:
            return

        m_.total_ops = m_.total_params = 0
        m_.module_used = False

        for p in m_.parameters():
            m_.total_params += int(p.numel())

        m_type = type(m_)
        fn = register_hooks.get(m_type, count_naive)

        if fn is not None:
            _handler = m_.register_forward_hook(fn)
            handler_collection.append(_handler)

    def remove_buffer(m_):
        if len(list(m_.children())) > 0:
            return

        del m_.total_ops, m_.total_params, m_.module_used

    original_device = next(model.parameters()).device
    training = model.training

    model.eval()
    model.apply(add_hooks)

    assert isinstance(input_size, tuple)
    if torch.is_tensor(input_size[0]):
        x = (t.to(original_device) for t in input_size)
    else:
        x = (torch.zeros(input_size).to(original_device), )
    with torch.no_grad():
        model(*x)

    total_ops = 0
    total_params = 0
    for name, m in model.named_modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        if not m.module_used:
            if not suppress_warnings:
                print_log(f'Module {name} of type {type(m)} is not used.', __name__, logging.WARNING)
            continue
        total_ops += m.total_ops
        total_params += m.total_params
        print_log('Profiling single module %s: %.2f %.2f' % (name, m.total_ops, m.total_params), __name__, logging.DEBUG)

    model.train(training).to(original_device)
    for handler in handler_collection:
        handler.remove()
    model.apply(remove_buffer)

    return total_ops, total_params


def test_profiler():

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv = nn.Conv2d(3, 5, 1, 1)

        def forward(self, x):
            return self.conv(x)

    assert flops_params_counter(Model(), (1, 3, 2, 2)) == (60, 20)

    class Model2(nn.Module):
        def __init__(self):
            super(Model2, self).__init__()
            self.conv = DynamicConv2d(3, 5, 1, 1)

        def forward(self, x):
            return self.conv(x, out_channels=4)
    
    assert flops_params_counter(Model2(), (1, 3, 2, 2)) == (48, 20)


if __name__ == '__main__':
    test_profiler()
