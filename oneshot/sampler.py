import collections
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mmcv.utils.logging import print_log
from torch.nn.modules.batchnorm import _BatchNorm

from common.searchspace import Mutable, MixedOp, MixedInput
from configs import DatasetConfig, DatasetType, MetricsConfig, RuntimeConfig, TrainerConfig, SamplerConfig
from datasets import cifar10_dataloader, cifar100_dataloader, imagenet_dataloader, imagenet16_dataloader
from .base import Trainer
from .utils import build_optimizer, build_scheduler, build_loss, build_metrics


class SamplerMixin(Trainer):
    def __init__(self, model: 'nn.Module',
                 trainer_config: TrainerConfig,
                 sampler_config: SamplerConfig,
                 runtime_config: RuntimeConfig,
                 metrics_config: MetricsConfig,
                 dataset_config: DatasetConfig):
        super(SamplerMixin, self).__init__(
            model=model,
            warmup_epochs=sampler_config.warmup_epochs,
            val_every_n_epoch=trainer_config.val_every_n_epoch,
            test_every_n_epoch=trainer_config.test_every_n_epoch,
            save_ckpt_every_n_epoch=trainer_config.save_ckpt_every_n_epoch,
            fast_dev_run=trainer_config.fast_dev_run,
            console_log_interval=trainer_config.console_log_interval,
            tb_log_interval=trainer_config.tb_log_interval,
            num_epochs=trainer_config.num_epochs,
            evaluate_only=runtime_config.evaluate_only,
            resume_from=runtime_config.resume_from,
            label_this_run=runtime_config.label_this_run,
            tb_log_dir=runtime_config.tb_log_dir,
            checkpoint_dir=runtime_config.checkpoint_dir
        )
        self.trainer_config = trainer_config
        self.sampler_config = sampler_config
        self.runtime_config = runtime_config
        self.metrics_config = metrics_config
        self.dataset_config = dataset_config
        self.loss_fn = build_loss(metrics_config)
        self.metrics_fn = build_metrics(metrics_config)
        self.reward_key = metrics_config.reward_key
        self.parse_searchspace()
        self.predefined_testset = []

        if self.distributed:
            with torch.no_grad():
                for param in self.model.module.parameters():
                    param.grad = torch.zeros_like(param)

    def parse_searchspace(self):
        def apply(m):
            for name, child in m.named_children():
                if isinstance(child, Mutable):
                    setattr(self, name, self.replace_mutable(child))
                    self.searchspace.update(child.searchspace())
                    self.mutables.append((child.key, child))
                else:
                    apply(child)

        self.searchspace = {}
        self.mutables = []
        apply(self.model_inner)

        # postprocess, disable running_stats for all BNs
        # disable finetune BN for acceleration
        # We can still enable it later
        for module in self.model_inner.modules():
            if isinstance(module, _BatchNorm):
                del module.running_mean
                del module.running_var
                del module.num_batches_tracked
                module.register_parameter('running_mean', None)
                module.register_parameter('running_var', None)
                module.register_parameter('num_batches_tracked', None)
                module.track_running_stats = False

    def configure_optimizers(self):
        optimizer = build_optimizer(self.model_weights(), self.trainer_config.optimizer)
        scheduler = build_scheduler(optimizer, self.trainer_config.lr_scheduler,
                                    len(self.train_dataloader(0)), self.trainer_config.num_epochs,
                                    torch.distributed.get_world_size() if self.distributed else 1)
        return optimizer, scheduler

    def train_dataloader(self, current_epoch):
        conf = self.dataset_config
        if conf.dataset_cls == DatasetType.CIFAR10:
            return cifar10_dataloader(conf.data_dir, 'train', self.trainer_config.batch_size,
                                      num_threads=conf.num_threads, cutout=conf.cutout,
                                      distributed=self.distributed, seed=self.runtime_config.seed + current_epoch)
        if conf.dataset_cls == DatasetType.CIFAR100:
            return cifar100_dataloader(conf.data_dir, 'train', self.trainer_config.batch_size,
                                       num_threads=conf.num_threads, cutout=conf.cutout,
                                       distributed=self.distributed, seed=self.runtime_config.seed + current_epoch)
        if conf.dataset_cls == DatasetType.ImageNet:
            return imagenet_dataloader(conf.data_dir, 'train', self.trainer_config.batch_size,
                                       num_threads=conf.num_threads, distributed=self.distributed,
                                       seed=self.runtime_config.seed + current_epoch)
        if conf.dataset_cls == DatasetType.ImageNet16:
            return imagenet16_dataloader(conf.data_dir, 'train', self.trainer_config.batch_size,
                                         num_threads=conf.num_threads, distributed=self.distributed,
                                        seed=self.runtime_config.seed + current_epoch)
        raise ValueError

    def val_dataloader(self, current_epoch):
        conf = self.dataset_config
        if conf.dataset_cls == DatasetType.CIFAR10:
            return cifar10_dataloader(conf.data_dir, 'val', self.trainer_config.val_batch_size,
                                      num_threads=conf.num_threads, cutout=0, distributed=False,
                                      seed=self.runtime_config.seed + current_epoch)
        if conf.dataset_cls == DatasetType.CIFAR100:
            return cifar100_dataloader(conf.data_dir, 'val', self.trainer_config.val_batch_size,
                                       num_threads=conf.num_threads, cutout=0, distributed=False,
                                       seed=self.runtime_config.seed + current_epoch)
        if conf.dataset_cls == DatasetType.ImageNet:
            return imagenet_dataloader(conf.data_dir, 'val', self.trainer_config.val_batch_size,
                                       num_threads=conf.num_threads, distributed=False,
                                       seed=self.runtime_config.seed + current_epoch)
        if conf.dataset_cls == DatasetType.ImageNet16:
            return imagenet16_dataloader(conf.data_dir, 'val', self.trainer_config.val_batch_size,
                                         num_threads=conf.num_threads, distributed=False,
                                        seed=self.runtime_config.seed + current_epoch)
        raise ValueError

    def test_dataloader(self):
        conf = self.dataset_config
        if conf.test_on_val:
            return self.val_dataloader(1)
        if conf.dataset_cls == DatasetType.CIFAR10:
            return cifar10_dataloader(conf.data_dir, 'test', self.trainer_config.val_batch_size,
                                      num_threads=conf.num_threads, cutout=0, distributed=False)
        if conf.dataset_cls == DatasetType.CIFAR100:
            return cifar100_dataloader(conf.data_dir, 'test', self.trainer_config.val_batch_size,
                                       num_threads=conf.num_threads, cutout=0, distributed=False)
        if conf.dataset_cls == DatasetType.ImageNet:
            return imagenet_dataloader(conf.data_dir, 'test', self.trainer_config.val_batch_size,
                                       num_threads=conf.num_threads, distributed=False)
        if conf.dataset_cls == DatasetType.ImageNet16:
            return imagenet16_dataloader(conf.data_dir, 'test', self.trainer_config.val_batch_size,
                                         num_threads=conf.num_threads, distributed=False)
        raise ValueError

    def optimizer_step(self, optimizer):
        if self.trainer_config.optimizer.grad_clip:
            nn.utils.clip_grad_norm_(self.model_weights(), self.trainer_config.optimizer.grad_clip)
        optimizer.step()

    def replace_mutable(self, mutable: Mutable) -> Mutable:
        return mutable

    def test(self, current_epoch):
        for arch_idx in range(self.sampler_config.num_architectures_per_test):
            if self.fast_dev_run and arch_idx >= 2:
                break
            if self.predefined_testset:
                # use sample from architecture set
                architecture = self.predefined_testset[arch_idx]
                self.activate(architecture)
            else:
                # randomly resample
                architecture = self.resample()
            self.print_console_log('Arch [%d/%d] %s' % (arch_idx + 1, self.sampler_config.num_architectures_per_test,
                                                        json.dumps(architecture)), current_epoch)
            evaluation_result = {}
            if self.sampler_config.profile_on_testset:
                evaluation_result.update(self.profile_single_architecture())
            if self.sampler_config.eval_on_testset:
                evaluation_result.update(self.test_single_architecutre())
            if evaluation_result is not None:
                self.print_console_log('Arch [%d/%d] %s' % (arch_idx + 1, self.sampler_config.num_architectures_per_test,
                                                            ' '.join([f'{k} = {v:.6f}' for k, v in evaluation_result.items()])),
                                       current_epoch)

    def resample(self) -> 'Dict[str, Any]':
        raise NotImplementedError

    def test_single_architecutre(self) -> 'Dict[str, Any]':
        raise NotImplementedError

    def profile_single_architecture(self) -> 'Dict[str, Any]':
        raise NotImplementedError

    def model_weights(self):
        return tuple(self.model.parameters())

    def choice_helper(self, samples):
        if isinstance(samples, list):
            return random.choice(samples)
        elif samples[-1] == -1:
            return [s for s in samples[0] if random.randrange(2)]
        elif isinstance(samples[-1], list) and len(samples[0]) == len(samples[1]):
            choices, prior = samples
            index = np.where(np.random.multinomial(1, prior))[0][0]
            return choices[index]
        raise ValueError

    def activate(self, sample):
        for key, mutable in self.mutables:
            mutable.activate(sample[key])


class Gumbel(SamplerMixin):
    pass


class Reinforce(SamplerMixin):
    pass


class Enumeration(SamplerMixin):
    pass


class ArchMinibatch(SamplerMixin):
    pass
