import math
import os

import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torch
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]


class HybridTrainPipe(Pipeline):
    def __init__(self, image_dir, num_threads, seed, file_list, batch_size,
                 image_size, color_jitter, rank, world_size, enable_gpu):
        device_id = torch.cuda.current_device()
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=seed + rank)
        self.input = ops.FileReader(file_root=image_dir,
                                    shard_id=rank, num_shards=world_size, file_list=file_list,
                                    random_shuffle=True)
        self.decode = ops.ImageDecoder(device='mixed' if enable_gpu else 'cpu')
        self.res = ops.RandomResizedCrop(device='gpu' if enable_gpu else 'cpu',
                                         size=image_size,
                                         mag_filter=types.DALIInterpType.INTERP_LINEAR,
                                         min_filter=types.DALIInterpType.INTERP_TRIANGULAR)
        self.color_twist_type = color_jitter
        self.brightness = ops.BrightnessContrast(device='gpu' if enable_gpu else 'cpu')
        self.hsv = ops.Hsv(device='gpu' if enable_gpu else 'cpu')

        self.rng_0100 = ops.Uniform(range=[0.9, 1.1])
        self.rng_0125 = ops.Uniform(range=[1 - 32 / 255, 1 + 32 / 255])
        self.rng_0500 = ops.Uniform(range=[0.5, 1.5])
        self.rng_0400 = ops.Uniform(range=[0.6, 1.4])

        self.cmnp = ops.CropMirrorNormalize(device='gpu' if enable_gpu else 'cpu',
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            mean=MEAN,
                                            std=STD)
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name='Reader')
        images = self.decode(self.jpegs)
        images = self.res(images)
        if self.color_twist_type == 'tf':
            images = self.brightness(images, brightness=self.rng_0125())
            images = self.hsv(images, saturation=self.rng_0500())
        else:
            images = self.brightness(images, brightness=self.rng_0400(), contrast=self.rng_0400())
            images = self.hsv(images, saturation=self.rng_0400(), hue=self.rng_0100())
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, image_dir, num_threads, seed, file_list, batch_size, image_size, rank, world_size, enable_gpu):
        device_id = torch.cuda.current_device()
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=seed + rank)
        self.input = ops.FileReader(file_root=image_dir,
                                    shard_id=rank, num_shards=world_size, file_list=file_list,
                                    random_shuffle=True)
        self.decode = ops.ImageDecoder(device='mixed' if enable_gpu else 'cpu')
        self.res = ops.Resize(device='gpu' if enable_gpu else 'cpu',
                              resize_shorter=math.ceil(image_size / 0.875),
                              mag_filter=types.DALIInterpType.INTERP_LINEAR,
                              min_filter=types.DALIInterpType.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device='gpu' if enable_gpu else 'cpu',
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(image_size, image_size),
                                            mean=MEAN,
                                            std=STD)

    def define_graph(self):
        self.jpegs, self.labels = self.input(name='Reader')
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


class _ClassificationWrapper:
    def __init__(self, pipeline, batch_size, world_size, drop_last):
        self.num_samples = pipeline.epoch_size('Reader') // world_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.loader = DALIClassificationIterator(pipeline, size=self.num_samples,
                                                 fill_last_batch=False, auto_reset=True)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                data = next(self.loader)
                images, labels = data[0]['data'], data[0]['label'].view(-1).long()
                if self.drop_last and labels.size(0) < self.batch_size:
                    continue
                return images.cuda(), labels.cuda()
            except StopIteration:
                raise StopIteration

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size


def imagenet_dataloader(image_dir, split, batch_size, file_list_dir='data/proxyless/imagenet',
                        num_threads=6, image_size=224, distributed=True, seed=42, color_jitter='torch', enable_gpu=False):
    assert torch.cuda.is_available()
    assert os.path.exists(image_dir), f'Directory {image_dir} does not exist.'
    if distributed:
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    else:
        rank, world_size = 0, 1
    file_list = os.path.join(file_list_dir, f'{split}_files.txt')
    assert os.path.exists(file_list), f'File list {file_list} does not exist.'
    if split in ('train', 'augment'):
        pipeline = HybridTrainPipe(image_dir, num_threads, seed, file_list,
                                   batch_size, image_size, color_jitter, rank, world_size, enable_gpu)
    elif split in ('val', 'test'):
        pipeline = HybridValPipe(image_dir, num_threads, seed, file_list,
                                 batch_size, image_size, rank, world_size, enable_gpu)
    else:
        raise AssertionError
    pipeline.build()
    return _ClassificationWrapper(pipeline, batch_size, world_size, split in ('train', 'augment'))
