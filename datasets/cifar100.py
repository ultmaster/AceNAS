import functools
import json
import os

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100

from .utils import ReproducibleDataLoader


def cutout_fn(img, length):
    h, w = img.size(1), img.size(2)
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x + length // 2, 0, w)

    mask[y1: y2, x1: x2] = 0.
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img *= mask

    return img


def load_split(config_file_path, split_names):
    with open(config_file_path, 'r') as f:
        data = json.load(f)
    result = []
    for name in split_names:
        _, arr = data[name]
        result.append(list(map(int, arr)))
    return result


def cifar100_dataloader(image_dir, split, batch_size, num_threads=6, cutout=0, distributed=False, seed=42):
    assert torch.cuda.is_available()
    MEAN = [x / 255 for x in [129.3, 124.1, 112.4]]
    STD = [x / 255 for x in [68.2, 65.4, 70.4]]
    transf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]
    if split == 'train':
        cutout_ = []
        cutout_.append(functools.partial(cutout_fn, length=cutout))
        dataset = CIFAR100(image_dir, train=True, transform=transforms.Compose(transf + normalize + cutout_))
        return ReproducibleDataLoader(dataset, batch_size=batch_size,
                                      drop_last=True, num_workers=num_threads, seed=seed)
    if split in ('val', 'test'):
        valid_split, test_split = load_split(os.path.join(image_dir, 'split-cifar100.txt'), ['xvalid', 'xtest'])
        dataset = CIFAR100(image_dir, train=False, transform=transforms.Compose(normalize))
        if split == 'val':
            dataset = Subset(dataset, valid_split)
        elif split == 'test':
            dataset = Subset(dataset, test_split)
        return ReproducibleDataLoader(dataset, distributed=distributed, batch_size=batch_size,
                                      drop_last=False, num_workers=num_threads, seed=seed)
    raise ValueError
