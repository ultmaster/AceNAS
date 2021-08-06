import collections
import logging

import torch
import torch.nn as nn

from common.dynamic_ops import DynamicBatchNorm2d, DynamicConv2d, DynamicLinear, ResizableSequential
from common.searchspace import SearchSpace, MixedOp, MixedInput, HyperParameter
from configs import NdsConfig, NdsModelType
from .layers import OPS, DropPath_, FactorizedReduce, ReLUConvBN


logger = logging.getLogger(__name__)


class AuxiliaryHead(nn.Module):
    def __init__(self, C, num_classes):
        super(AuxiliaryHead, self).__init__()
        if num_classes == 1000:
            # assuming input size 14x14
            self.features = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
                DynamicConv2d(C, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 768, 2, bias=False),
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True)
            )
        else:
            # assuming input size 8x8
            self.features = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
                DynamicConv2d(C, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 768, 2, bias=False),
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True)
            )
        self.classifier = nn.Linear(768, num_classes)
        for module in self.modules():
            if isinstance(module, DynamicConv2d):
                module.allow_static_op = True

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class Cell(nn.Module):
    def __init__(self, n_nodes, primitives, C_prev_prev, C_prev, C, reduction, concat_all):
        super(Cell, self).__init__()
        self.n_nodes = n_nodes
        self.reduction = reduction
        self.cell_type = 'reduce' if reduction else 'normal'
        self.concat_all = concat_all
        self.primitives = primitives
        logger.info('Cell %s created: channels %d -> %d -> %d, %d nodes',
                    self.cell_type, C_prev_prev, C_prev, C, self.n_nodes)

        self.preprocess0_reduce = FactorizedReduce(C_prev_prev, C)
        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 0)

        self.nodes = nn.ModuleDict()
        self.inputs = nn.ModuleDict()
        for i in range(self.n_nodes):
            self.nodes[f'{i}_x'] = self._build_layer_choice(f'{self.cell_type}_{i}_x', C)
            self.nodes[f'{i}_y'] = self._build_layer_choice(f'{self.cell_type}_{i}_y', C)
            self.inputs[f'{i}_x'] = self._build_input_choice(f'{self.cell_type}_{i}_x', i + 2)
            self.inputs[f'{i}_y'] = self._build_input_choice(f'{self.cell_type}_{i}_y', i + 2)

    def _build_input_choice(self, key, num_input_candidates):
        return MixedInput(key + '_input', num_input_candidates)

    def _build_layer_choice(self, key, channels):
        mapping = collections.OrderedDict()
        for name in self.primitives:
            mapping[name] = OPS[name](channels)
        return MixedOp(key + '_op', mapping)

    def forward(self, s0, s1, width):
        if s0.size(2) != s1.size(2):
            # needs to be down-sampled
            s0 = self.preprocess0_reduce(s0, width)
        else:
            s0 = self.preprocess0(s0, width)
        s1 = self.preprocess1(s1, width)
        states = [s0, s1]
        used_indices = set()
        for i in range(self.n_nodes):
            x_k, y_k = f'{i}_x', f'{i}_y'
            x_reduction = self.reduction and self.inputs[x_k].activated < 2
            y_reduction = self.reduction and self.inputs[y_k].activated < 2
            used_indices |= {self.inputs[x_k].activated, self.inputs[y_k].activated}
            t1 = self.nodes[x_k](self.inputs[x_k](states), width, 2 if x_reduction else 1)
            t2 = self.nodes[y_k](self.inputs[y_k](states), width, 2 if y_reduction else 1)
            states.append(t1 + t2)
        if self.concat_all:
            return torch.cat(states[2:], 1)
        else:
            unused_indices = [i for i in range(2, self.n_nodes + 2) if i not in used_indices]
            return torch.cat([states[i] for i in unused_indices], 1)


class NDS(SearchSpace):

    def __init__(self, config: NdsConfig):
        super(NDS, self).__init__()
        self.model_type = config.model_type
        self.num_labels = 10 if self.model_type == NdsModelType.CIFAR else 1000
        self.max_init_channels = max(config.init_channels)
        self.max_num_layers = max(config.num_layers)
        self.depth_selector = HyperParameter('depth', config.num_layers)
        self.width_selector = HyperParameter('width', config.init_channels)
        self.use_aux = config.use_aux
        C = self.max_init_channels

        if self.model_type == NdsModelType.ImageNet:
            self.stem0 = ResizableSequential(
                DynamicConv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
                DynamicBatchNorm2d(C // 2),
                nn.ReLU(inplace=True),
                DynamicConv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
                DynamicBatchNorm2d(C),
            )
            self.stem1 = ResizableSequential(
                nn.ReLU(inplace=True),
                DynamicConv2d(C, C, 3, stride=2, padding=1, bias=False),
                DynamicBatchNorm2d(C),
            )
            C_prev_prev, C_prev, C_curr = C, C, C
        elif self.model_type == NdsModelType.CIFAR:
            self.stem = ResizableSequential(
                DynamicConv2d(3, 3 * C, 3, padding=1, bias=False),
                DynamicBatchNorm2d(3 * C)
            )
            C_prev_prev, C_prev, C_curr = 3 * C, 3 * C, C

        self.stages = nn.ModuleList()
        for stage_idx in range(3):
            if stage_idx > 0:
                C_curr *= 2
            stage = nn.ModuleList()
            for i in range((self.max_num_layers + 2) // 3):
                cell = Cell(config.n_nodes, config.op_candidates, C_prev_prev, C_prev, C_curr,
                            stage_idx > 0 and i == 0, config.concat_all)
                stage.append(cell)
                C_prev_prev, C_prev = C_prev, cell.n_nodes * C_curr
            if stage_idx == 2:
                C_to_auxiliary = C_prev
            self.stages.append(stage)

        if self.use_aux:
            self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, self.num_labels)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = DynamicLinear(C_prev, self.num_labels)

    def forward(self, inputs):
        width = self.width_selector()
        depth = self.depth_selector()
        if self.model_type == NdsModelType.ImageNet:
            s0 = self.stem0(inputs, width / self.max_init_channels)
            s1 = self.stem1(s0, width / self.max_init_channels)
        else:
            s0 = s1 = self.stem(inputs, width / self.max_init_channels)

        cur_stage, cur_idx = 0, 0
        for i in range(depth):
            if i in [depth // 3, 2 * depth // 3]:
                width *= 2
                cur_stage += 1
                cur_idx = 0
            s0, s1 = s1, self.stages[cur_stage][cur_idx](s0, s1, width)
            if i == 2 * depth // 3:
                if self.training and self.use_aux:
                    logits_aux = self.auxiliary_head(s1)
            cur_idx += 1
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        if self.training and self.use_aux:
            return logits, logits_aux
        else:
            return logits

    def prune(self):
        super().prune()
        handler_collection = []

        def fn(m, _, __):
            m.module_used = True

        def add_hooks(m_):
            m_.module_used = False
            _handler = m_.register_forward_hook(fn)
            handler_collection.append(_handler)

        def dfs_and_delete(m):
            names = []
            for name, child in m.named_children():
                if child.module_used or isinstance(child, (nn.ModuleList, nn.ModuleDict)):
                    dfs_and_delete(child)
                    if isinstance(child, (DynamicConv2d, DynamicBatchNorm2d, DynamicLinear, ResizableSequential)):
                        child._static_mode = True
                    if isinstance(child, DynamicConv2d):
                        child.stride = child._dry_run_stride
                else:
                    names.append(name)
            for name in names:
                delattr(m, name)
            delattr(m, 'module_used')

        training = self.training
        self.eval()
        self.apply(add_hooks)
        with torch.no_grad():
            self(torch.zeros((1, 3, 32, 32)))
        for m in self.auxiliary_head.modules():
            m.module_used = True
        for handler in handler_collection:
            handler.remove()

        dfs_and_delete(self)
        self.train(training)

    def drop_path_prob(self, drop_prob):
        for module in self.modules():
            if isinstance(module, DropPath_):
                module.drop_prob = drop_prob
