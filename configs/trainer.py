from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from .utils import PythonConfig


class LRSchedulerType(str, Enum):
    MultiStepLR = 'multisteplr'
    CosineAnnealingLR = 'cosinelr'


class OptimizerType(str, Enum):
    SGD = 'sgd'
    Adam = 'adam'


class SamplerType(str, Enum):
    Proxyless = 'proxyless'
    Gumbel = 'gumbel'
    DropOp = 'dropop'
    Enumerate = 'enumerate'
    Naive = 'naive'
    Reinforce = 'reinforce'


class LossType(str, Enum):
    CrossEntropy = 'crossentropy'
    LabelSmoothingCrossEntropy = 'labelsmoothing'


class MetricsType(str, Enum):
    Top1 = 'top1'
    Top5 = 'top5'


@dataclass(init=False)
class LRSchedulerConfig(PythonConfig):
    scheduler_type: LRSchedulerType
    warmup_epochs: Optional[int] = None
    milestones: Optional[List[int]] = None
    gamma: Optional[float] = None
    eta_min: Optional[float] = None


@dataclass(init=False)
class OptimizerConfig(PythonConfig):
    opt_type: OptimizerType
    learning_rate: float
    momentum: float
    weight_decay: float
    grad_clip: Optional[float] = None
    betas: Optional[Tuple[float, float]] = None
    eps: Optional[float] = None


@dataclass(init=False)
class TrainerConfig(PythonConfig):
    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig
    num_epochs: int
    batch_size: int
    val_batch_size: int
    val_every_n_epoch: int
    test_every_n_epoch: int
    console_log_interval: int
    tb_log_interval: int
    save_ckpt_every_n_epoch: int
    fast_dev_run: bool = False


@dataclass(init=False)
class SamplerConfig(PythonConfig):
    sampler_type: SamplerType
    warmup_epochs: int  # epochs before first trigger of shrinking
    num_architectures_per_test: int  # architectures generated for test
    eval_on_testset: bool  # evaluate architecture once after generation
    profile_on_testset: bool  # profile params and flops on testset, overrides eval_on_testset
    learning_rate: Optional[float] = None  # very common


@dataclass(init=False)
class MetricsConfig(PythonConfig):
    loss_fn: LossType
    metrics_fn: MetricsType
    reward_key: str
