from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .utils import PythonConfig


class DatasetType(Enum):
    CIFAR10 = 'cifar10'
    CIFAR100 = 'cifar100'
    ImageNet = 'imagenet'
    ImageNet16 = 'imagenet16'


@dataclass(init=False)
class DatasetConfig(PythonConfig):
    dataset_cls: DatasetType
    data_dir: str
    num_threads: int
    test_on_val: bool
    cutout: Optional[int] = None
