from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

from .utils import PythonConfig


class NdsModelType(str, Enum):
    CIFAR = 'cifar'
    ImageNet = 'imagenet'


@dataclass(init=False)
class NasBench101Config(PythonConfig):
    stem_out_channels: int
    num_stacks: int
    num_modules_per_stack: int
    num_labels: int
    max_num_vertices: int
    max_num_edges: int
    bn_eps: float
    bn_momentum: float


@dataclass(init=False)
class NasBench201Config(PythonConfig):
    stem_out_channels: int
    num_stacks: int
    num_modules_per_stack: int
    num_labels: int


@dataclass(init=False)
class NdsConfig(PythonConfig):
    init_channels: List[int]
    num_layers: List[int]
    model_type: NdsModelType
    concat_all: bool
    op_candidates: List[str]
    use_aux: bool
    n_nodes: int


@dataclass(init=False)
class ProxylessStageConfig(PythonConfig):
    depth_range: Tuple[int, int]
    exp_ratio_range: List[int]
    kernel_size_range: List[int]
    width: int
    downsample: bool

    def post_validate(self):
        return 1 <= self.depth_range[0] <= self.depth_range[1]


@dataclass(init=False)
class ProxylessConfig(PythonConfig):
    stages: List[ProxylessStageConfig]
    stem_width: int
    final_width: int
    width_mult: float
    num_labels: int
    dropout_rate: float = 0.
