import functools
import hashlib
import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from .cifar10 import cutout_fn


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
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


class ImageNet16(data.Dataset):
    # http://image-net.org/download-images
    # A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets
    # https://arxiv.org/pdf/1707.08819.pdf

    train_list = [
        ["train_data_batch_1", "27846dcaa50de8e21a7d1a35f30f0e91"],
        ["train_data_batch_2", "c7254a054e0e795c69120a5727050e3f"],
        ["train_data_batch_3", "4333d3df2e5ffb114b05d2ffc19b1e87"],
        ["train_data_batch_4", "1620cdf193304f4a92677b695d70d10f"],
        ["train_data_batch_5", "348b3c2fdbb3940c4e9e834affd3b18d"],
        ["train_data_batch_6", "6e765307c242a1b3d7d5ef9139b48945"],
        ["train_data_batch_7", "564926d8cbf8fc4818ba23d2faac7564"],
        ["train_data_batch_8", "f4755871f718ccb653440b9dd0ebac66"],
        ["train_data_batch_9", "bb6dd660c38c58552125b1a92f86b5d4"],
        ["train_data_batch_10", "8f03f34ac4b42271a294f91bf480f29b"],
    ]
    valid_list = [
        ["val_data", "3410e3017fdaefba8d5073aaa65e4bd6"],
    ]

    def __init__(self, root, train, transform, num_classes=None):
        self.root = root
        self.transform = transform
        self.train = train  # training set or valid set
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.valid_list
        self.data = []
        self.targets = []

        for i, (file_name, checksum) in enumerate(downloaded_list):
            file_path = os.path.join(self.root, file_name)
            with open(file_path, "rb") as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                self.targets.extend(entry["labels"])
        self.data = np.vstack(self.data).reshape(-1, 3, 16, 16)
        self.data = np.ascontiguousarray(self.data.transpose((0, 2, 3, 1)))  # convert to HWC
        if num_classes is not None:
            assert isinstance(num_classes, int) and \
                num_classes > 0 and \
                num_classes < 1000, \
                "invalid num_classes : {:}".format(num_classes)
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


def get_datasets(configs):
    if configs.dataset.startswith("cifar100"):
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif configs.dataset.startswith("cifar10"):  # cifar10 is a prefix of cifar100
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif configs.dataset.startswith('imagenet-16'):
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22,  61.26, 65.09]]
    else:
        raise NotImplementedError

    normalization = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    cutout = []
    if configs.dataset.startswith("cifar10") or configs.dataset.startswith("cifar100"):
        if hasattr(configs, "cutout") and configs.cutout > 0:
            cutout.append(functools.partial(cutout_fn, length=configs.cutout))
        augmentation = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)]
    elif configs.dataset.startswith("imagenet-16"):
        augmentation = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(16, padding=2)]
    train_transform = transforms.Compose(augmentation + normalization + cutout)
    test_transform = transforms.Compose(normalization)

    if configs.dataset.startswith("cifar100"):
        train_data = datasets.CIFAR100("data/cifar100", train=True, transform=train_transform)
        valid_data = datasets.CIFAR100("data/cifar100", train=False, transform=test_transform)
        assert len(train_data) == 50000 and len(valid_data) == 10000
    elif configs.dataset.startswith("cifar10"):
        train_data = datasets.CIFAR10("data/cifar10", train=True, transform=train_transform)
        valid_data = datasets.CIFAR10("data/cifar10", train=False, transform=test_transform)
        assert len(train_data) == 50000 and len(valid_data) == 10000
    elif configs.dataset.startswith("imagenet-16"):
        num_classes = int(configs.dataset.split("-")[-1])
        train_data = ImageNet16("data/imagenet16", train=True, transform=train_transform, num_classes=num_classes)
        valid_data = ImageNet16("data/imagenet16", train=False, transform=test_transform, num_classes=num_classes)
        assert len(train_data) == 151700 and len(valid_data) == 6000
    return train_data, valid_data


def load_split(config_file_path, split_names):
    with open(config_file_path, "r") as f:
        data = json.load(f)
    result = []
    for name in split_names:
        _, arr = data[name]
        result.append(list(map(int, arr)))
    return result


def nb201_dataloader(configs):
    train_data, valid_data = get_datasets(configs)
    split_path = "data/nb201/split-{}.txt".format(configs.dataset)
    kwargs = {"batch_size": configs.batch_size, "num_workers": configs.num_threads}
    if configs.dataset == "cifar10-valid":
        train_split, valid_split = load_split(split_path, ["train", "valid"])
        train_loader = DataLoader(train_data, sampler=SubsetRandomSampler(train_split), drop_last=True, **kwargs)
        valid_loader = DataLoader(train_data, sampler=SubsetRandomSampler(valid_split), **kwargs)
        test_loader = DataLoader(valid_data, **kwargs)
    elif configs.dataset == "cifar10":
        train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **kwargs)
        valid_loader = DataLoader(valid_data, **kwargs)
        test_loader = DataLoader(valid_data, **kwargs)
    else:
        valid_split, test_split = load_split(split_path, ["xvalid", "xtest"])
        train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **kwargs)
        valid_loader = DataLoader(valid_data, sampler=SubsetRandomSampler(valid_split), **kwargs)
        test_loader = DataLoader(valid_data, sampler=SubsetRandomSampler(test_split), **kwargs)
    return train_loader, valid_loader, test_loader
