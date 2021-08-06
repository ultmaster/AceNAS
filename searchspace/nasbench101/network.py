import functools
import logging

import numpy as np
import torch
import torch.nn as nn

from common.searchspace import SearchSpace, MixedOp, MixedInput, HyperParameter
from configs import NasBench101Config
from .base_ops import Conv3x3BnRelu, Conv1x1BnRelu, MaxPool3x3, ConvBnRelu, Projection, truncate
from .graph_util import compute_vertex_channels


logger = logging.getLogger(__name__)


class Cell(nn.Module):
    def __init__(self, max_num_vertices, max_num_edges, in_channels, out_channels):
        super(Cell, self).__init__()

        self.max_num_vertices = max_num_vertices
        self.max_num_edges = max_num_edges
        num_vertices_prior = [2 ** i for i in range(2, max_num_vertices + 1)]
        num_vertices_prior = (np.array(num_vertices_prior) / sum(num_vertices_prior)).tolist()
        self.num_vertices = HyperParameter('num_vertices',
                                           list(range(2, max_num_vertices + 1)),
                                           num_vertices_prior)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.projections = nn.ModuleList([nn.Identity()])
        self.op = nn.ModuleList([nn.Identity()])
        self.inputs = nn.ModuleList([nn.Identity()])
        for i in range(1, max_num_vertices):
            self.projections.append(Projection(in_channels, out_channels))

        for i in range(1, max_num_vertices - 1):
            self.op.append(MixedOp(f'op{i}', {
                'conv3x3-bn-relu': Conv3x3BnRelu(out_channels, out_channels),
                'conv1x1-bn-relu': Conv1x1BnRelu(out_channels, out_channels),
                'maxpool3x3': MaxPool3x3(out_channels, out_channels)
            }))
            self.inputs.append(MixedInput(f'input{i}', i, -1,
                                          functools.partial(self._intermediate_node_input_concat, current=i)))
        self.inputs.append(MixedInput(f'input{max_num_vertices - 1}', max_num_vertices - 1,
                                      -1, self._last_node_concat_and_add))

    def _build_connection_matrix(self, num_vertices):
        connections = np.zeros((num_vertices, num_vertices), dtype='int')
        for i in range(1, num_vertices - 1):
            for k in self.inputs[i].activated:
                connections[k, i] = 1
        for k in self.inputs[-1].activated:
            connections[k, -1] = 1
        return connections

    def _last_node_concat_and_add(self, tensors, activated):
        if len(activated) == 1:
            if activated[0] == 0:
                return self.projections[-1](tensors[0])
            else:
                return tensors[activated[0]]
        outputs = torch.cat([tensors[a] for a in activated if a != 0], 1)
        if 0 in activated:
            outputs += self.projections[-1](tensors[0])
        return outputs

    def _intermediate_node_input_concat(self, tensors, activated, current):
        add_in = []
        channels = self.vertex_channels[current]
        for src in activated:
            if src == 0:
                add_in.append(self.projections[current](tensors[src], out_features=channels))
            else:
                add_in.append(truncate(tensors[src], channels))
        return sum(add_in)

    def forward(self, inputs):
        num_vertices = self.num_vertices()
        connections = self._build_connection_matrix(num_vertices)
        self.vertex_channels = compute_vertex_channels(self.out_channels, connections)
        tensors = [inputs]
        for t in range(1, num_vertices - 1):
            vertex_input = self.inputs[t](tensors)
            vertex_value = self.op[t](vertex_input, out_features=self.vertex_channels[t])
            tensors.append(vertex_value)
        return self.inputs[self.max_num_vertices - 1](tensors)

    def validate(self) -> bool:
        num_vertices = self.num_vertices()
        try:
            connections = self._build_connection_matrix(num_vertices)
        except IndexError:
            return False
        if connections[-1, -1]:
            return False
        if np.sum(connections) > self.max_num_edges:
            return False
        ret = np.linalg.matrix_power(connections + np.eye(len(connections)), self.max_num_vertices)
        return np.all(ret[0, :]) and np.all(ret[:, -1])


class NasBench101(SearchSpace):
    def __init__(self, config: NasBench101Config):
        super(NasBench101, self).__init__()

        self.config = config
        # initial stem convolution
        self.stem_conv = Conv3x3BnRelu(3, config.stem_out_channels)

        layers = []
        in_channels = out_channels = config.stem_out_channels
        for stack_num in range(config.num_stacks):
            if stack_num > 0:
                downsample = nn.MaxPool2d(kernel_size=2, stride=2)
                layers.append(downsample)
                out_channels *= 2
            for _ in range(config.num_modules_per_stack):
                cell = Cell(config.max_num_vertices, config.max_num_edges, in_channels, out_channels)
                layers.append(cell)
                in_channels = out_channels

        self.features = nn.ModuleList(layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(out_channels, config.num_labels)

    def forward(self, x):
        bs = x.size(0)
        out = self.stem_conv(x)
        for layer in self.features:
            out = layer(out)
        out = self.gap(out).view(bs, -1)
        out = self.classifier(out)
        return out

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps = self.config.bn_eps
                module.momentum = self.config.bn_momentum

    def validate(self) -> bool:
        return self.features[0].validate()
