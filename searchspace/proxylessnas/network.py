import functools
import math
import random

import torch
import torch.nn as nn
import tqdm
from mmcv.utils.logging import print_log

from common.searchspace import BiasedMixedOp, MixedOp, SearchSpace
from configs.searchspace import ProxylessConfig, ProxylessStageConfig
from .utils import ConvBNReLU, InvertedResidual, make_divisible


class _MbNet(nn.Module):
    def __init__(self, first_conv, blocks, feature_mix_layer, dropout_layer, classifier):
        super().__init__()
        self.first_conv = first_conv
        self.blocks = nn.Sequential(*blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout_layer = dropout_layer
        self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)
        x = self.blocks(x)
        x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout_layer(x)
        x = self.classifier(x)
        return x

    def no_weight_decay(self):
        # no regularizer to linear layer
        return {'classifier.weight', 'classifier.bias'}

    def reset_parameters(self, model_init='he_fout', init_div_groups=False,
                         bn_momentum=0.1, bn_eps=1e-5,
                         track_running_stats=True, zero_grad=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.momentum = bn_momentum
                m.eps = bn_eps
                if not track_running_stats and m.track_running_stats:
                    m.track_running_stats = False
                    delattr(m, 'running_mean')
                    delattr(m, 'running_var')
                    delattr(m, 'num_batches_tracked')
                    m.register_parameter('running_mean', None)
                    m.register_parameter('running_var', None)
                    m.register_parameter('num_batches_tracked', None)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # zero out gradients
        if zero_grad:
            for p in self.parameters():
                p.grad = torch.zeros_like(p)


class _MbMixLayer(nn.Module):
    def __init__(self, ops, **metainfo):
        super().__init__()
        for name, op in ops.items():
            self.add_module(name, op)
        self.ops = list(ops.keys())
        self.metainfo = metainfo
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.world_rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.world_rank = 0
            self.world_size = 1

        self.fixed = None

    def _sample(self):
        chosen = None
        for i in range(self.world_size):
            tmp = random.choice(self.ops)
            if i == self.world_rank:
                chosen = tmp
        assert chosen is not None
        return chosen

    def forward(self, x):
        if self.fixed is not None:
            return getattr(self, self.fixed)(x)
        return getattr(self, self._sample())(x)

    def summary(self):
        return 'MbMixLayer(' + ', '.join([f'{k}={v}' for k, v in {'ops': self.ops, **self.metainfo}.items()]) + ')'


class ProxylessNAS(_MbNet, SearchSpace):
    def __init__(self, config: ProxylessConfig, reset_parameters=True):
        stem_width = make_divisible(config.width_mult * config.stem_width, 8)

        first_conv = ConvBNReLU(3, stem_width, stride=2, norm_layer=nn.BatchNorm2d)

        last_width = stem_width
        blocks = []
        for i, stage_config in enumerate(config.stages, start=1):
            print_log(f'Building stage #{i}...', __name__)
            width = make_divisible(stage_config.width * config.width_mult, 8)
            blocks += self._build_stage(i, stage_config, last_width, width)
            last_width = width

        final_width = make_divisible(1280 * config.width_mult, 8) if config.width_mult > 1 else 1280
        dropout_layer = nn.Dropout(config.dropout_rate)
        feature_mix_layer = ConvBNReLU(last_width, final_width, kernel_size=1, norm_layer=nn.BatchNorm2d)
        classifier = nn.Linear(final_width, config.num_labels)
        super().__init__(first_conv, blocks, feature_mix_layer, dropout_layer, classifier)

        if reset_parameters:
            self.reset_parameters(track_running_stats=False, zero_grad=True)

    def _build_stage(self, stage_idx: int, config: ProxylessStageConfig, input_width: int, output_width: int):
        depth_min, depth_max = config.depth_range
        blocks = []
        for i in range(depth_max):
            stride = 2 if config.downsample and i == 0 else 1
            op_choices = {}
            for exp_ratio in config.exp_ratio_range:
                for kernel_size in config.kernel_size_range:
                    op_choices[f'k{kernel_size}e{exp_ratio}'] = InvertedResidual(input_width, output_width, stride, exp_ratio, kernel_size)
            if i >= depth_min:
                prior = [0.5 / len(op_choices)] * len(op_choices) + [0.5]
                op_choices['skip'] = nn.Identity()
                blocks.append(BiasedMixedOp(f's{stage_idx}b{i + 1}_i{input_width}o{output_width}', op_choices, prior))
                assert blocks[-1].op_candidates[-1] == 'skip'
            else:
                blocks.append(MixedOp(f's{stage_idx}b{i + 1}_i{input_width}o{output_width}', op_choices))
            print_log(f'Created block: {blocks[-1].key}: {blocks[-1].op_candidates}', __name__)
            input_width = output_width
        return blocks

    def reset_running_stats(self, dataloader, max_steps=200):
        bn_mean = {}
        bn_var = {}

        def bn_forward_hook(bn, inputs, outputs, mean_est, var_est):
            aggregate_dimensions = (0, 2, 3)
            inputs = inputs[0]  # input is a tuple of arguments
            batch_mean = inputs.mean(aggregate_dimensions, keepdim=True)  # 1, C, 1, 1
            batch_var = (inputs - batch_mean) ** 2
            batch_var = batch_var.mean(aggregate_dimensions, keepdim=True)

            batch_mean = torch.squeeze(batch_mean)
            batch_var = torch.squeeze(batch_var)

            mean_est.append(batch_mean.data)
            var_est.append(batch_var.data)

        handles = []
        for name, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                bn_mean[name] = []
                bn_var[name] = []
                handle = m.register_forward_hook(functools.partial(bn_forward_hook, mean_est=bn_mean[name], var_est=bn_var[name]))
                handles.append(handle)

        self.train()
        with torch.no_grad():
            pbar = tqdm.tqdm(range(max_steps), desc='Calibrating BatchNorm')
            for _ in pbar:
                images, _ = next(dataloader)
                self(images)

            for name, m in self.named_modules():
                if name in bn_mean and len(bn_mean[name]) > 0:
                    feature_dim = bn_mean[name][0].size(0)
                    assert isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
                    m.running_mean.data[:feature_dim].copy_(sum(bn_mean[name]) / len(bn_mean[name]))
                    m.running_var.data[:feature_dim].copy_(sum(bn_var[name]) / len(bn_var[name]))

        for handle in handles:
            handle.remove()

    def fix_sample(self, sample):
        if isinstance(sample, list):
            search_space = self.export_search_space()
            assert len(search_space) == len(sample)
            sample = {k: v for k, v in zip(search_space.keys(), sample)}
        else:
            assert len(self.export_search_space()) == len(sample)
        for name, module in self.named_modules():
            if isinstance(module, _MbMixLayer) and name in sample:
                module.fixed = sample[name]
        return sample

    def export_search_space(self):
        result = {}
        for name, module in self.named_modules():
            if isinstance(module, _MbMixLayer) and len(module.ops) > 1:
                result[name] = module.ops
        return result
