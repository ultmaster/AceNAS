import hashlib
import json
import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
from torchvision import transforms

from .utils import ReproducibleDataLoader


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    else:
        return check_md5(fpath, md5)


class ImageNet16(Dataset):
    # http://image-net.org/download-images
    # A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets
    # https://arxiv.org/pdf/1707.08819.pdf

    train_list = [
        ['train_data_batch_1', '27846dcaa50de8e21a7d1a35f30f0e91'],
        ['train_data_batch_2', 'c7254a054e0e795c69120a5727050e3f'],
        ['train_data_batch_3', '4333d3df2e5ffb114b05d2ffc19b1e87'],
        ['train_data_batch_4', '1620cdf193304f4a92677b695d70d10f'],
        ['train_data_batch_5', '348b3c2fdbb3940c4e9e834affd3b18d'],
        ['train_data_batch_6', '6e765307c242a1b3d7d5ef9139b48945'],
        ['train_data_batch_7', '564926d8cbf8fc4818ba23d2faac7564'],
        ['train_data_batch_8', 'f4755871f718ccb653440b9dd0ebac66'],
        ['train_data_batch_9', 'bb6dd660c38c58552125b1a92f86b5d4'],
        ['train_data_batch_10', '8f03f34ac4b42271a294f91bf480f29b'],
    ]
    valid_list = [
        ['val_data', '3410e3017fdaefba8d5073aaa65e4bd6'],
    ]

    def __init__(self, root, train, transform, num_classes=None):
        self.root = root
        self.transform = transform
        self.train = train  # training set or valid set
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.valid_list
        self.data = []
        self.targets = []

        for i, (file_name, checksum) in enumerate(downloaded_list):
            file_path = os.path.join(self.root, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 16, 16)
        self.data = np.ascontiguousarray(self.data.transpose((0, 2, 3, 1)))  # convert to HWC
        if num_classes is not None:
            assert isinstance(num_classes, int) and \
                num_classes > 0 and \
                num_classes < 1000, \
                'invalid num_classes : {:}'.format(num_classes)
            self.samples = [i for i, l in enumerate(self.targets) if 1 <= l <= num_classes]
        else:
            self.samples = list(range(len(self.targets)))

    def __getitem__(self, index):
        index = self.samples[index]
        img, target = self.data[index], self.targets[index] - 1
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.samples)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.valid_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, filename)
            if not check_integrity(fpath, md5):
                return False
        return True


def load_split(config_file_path, split_names):
    with open(config_file_path, 'r') as f:
        data = json.load(f)
    result = []
    for name in split_names:
        _, arr = data[name]
        result.append(list(map(int, arr)))
    return result


def imagenet16_dataloader(image_dir, split, batch_size, num_threads=6, distributed=False, seed=42):
    assert torch.cuda.is_available()
    MEAN = [x / 255 for x in [122.68, 116.66, 104.01]]
    STD = [x / 255 for x in [63.22,  61.26, 65.09]]
    transf = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(16, padding=2)
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]
    if split == 'train':
        dataset = ImageNet16(image_dir, train=True, transform=transforms.Compose(transf + normalize), num_classes=120)
        assert len(dataset) == 151700
        return ReproducibleDataLoader(dataset, distributed=distributed, batch_size=batch_size,
                                      drop_last=True, num_workers=num_threads, seed=seed)
    if split in ('val', 'test'):
        valid_split, test_split = load_split(os.path.join(image_dir, 'split-imagenet-16-120.txt'), ['xvalid', 'xtest'])
        dataset = ImageNet16(image_dir, train=False, transform=transforms.Compose(normalize), num_classes=120)
        assert len(dataset) == 6000
        if split == 'val':
            dataset = Subset(dataset, valid_split)
        elif split == 'test':
            dataset = Subset(dataset, test_split)
        return ReproducibleDataLoader(dataset, distributed=distributed, batch_size=batch_size,
                                      drop_last=False, num_workers=num_threads, seed=seed)
    raise ValueError
