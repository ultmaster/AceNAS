import torch.nn as nn
import torch.optim as optim

from common.scheduler import GradualWarmupScheduler
from common.metrics import LabelSmoothingLoss, accuracy
from configs.trainer import (LRSchedulerConfig, LRSchedulerType, LossType,
                             MetricsConfig, MetricsType, OptimizerConfig, OptimizerType)


def build_optimizer(weights: 'List[nn.Parameter]', optimizer_config: OptimizerConfig) -> optim.Optimizer:
    if optimizer_config.opt_type == OptimizerType.Adam:
        return optim.Adam(weights, lr=optimizer_config.learning_rate,
                          betas=optimizer_config.betas,
                          eps=optimizer_config.eps,
                          weight_decay=optimizer_config.weight_decay)
    if optimizer_config.opt_type == OptimizerType.SGD:
        return optim.SGD(weights,
                         lr=optimizer_config.learning_rate,
                         momentum=optimizer_config.momentum,
                         weight_decay=optimizer_config.weight_decay)
    raise ValueError


def build_scheduler(optimizer: optim.Optimizer, scheduler_config: LRSchedulerConfig,
                    steps_per_epoch: int, num_epochs: int, dist_world_size: int) -> 'optim.lr_scheduler._LRScheduler':
    if scheduler_config.scheduler_type == LRSchedulerType.MultiStepLR:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[t * steps_per_epoch for t in scheduler_config.milestones],
                                                   gamma=scheduler_config.gamma)
    if scheduler_config.scheduler_type == LRSchedulerType.CosineAnnealingLR:
        if scheduler_config.warmup_epochs:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             T_max=(num_epochs - scheduler_config.warmup_epochs) * steps_per_epoch,
                                                             eta_min=scheduler_config.eta_min)
            scheduler = GradualWarmupScheduler(optimizer, dist_world_size,
                                               scheduler_config.warmup_epochs * steps_per_epoch,
                                               after_scheduler=scheduler)
            scheduler.step()
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_per_epoch * num_epochs,
                                                             eta_min=scheduler_config.eta_min)
    return scheduler


def build_loss(config: MetricsConfig):
    if config.loss_fn == LossType.CrossEntropy:
        return nn.CrossEntropyLoss()
    elif config.loss_fn == LossType.LabelSmoothingCrossEntropy:
        return LabelSmoothingLoss()
    raise ValueError


def build_metrics(config: MetricsConfig):
    if config.metrics_fn == MetricsType.Top1:
        return lambda pred, target: {'top1': accuracy(pred, target)}
    elif config.metrics_fn == MetricsType.Top5:
        return lambda pred, target: dict(zip(['top1', 'top5'], accuracy(pred, target, topk=(1, 5))))
