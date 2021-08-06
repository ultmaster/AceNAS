import functools
import json
import logging
import math
import os
import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)


class AverageMeterGroup:
    """Average meter group for multiple average meters"""

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data, n=1):
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter(k, ":4f")
            self.meters[k].update(v, n=n)

    def __getattr__(self, item):
        return self.meters[item]

    def __getitem__(self, item):
        return self.meters[item]

    def __str__(self):
        return "  ".join(str(v) for v in self.meters.values())

    def summary(self):
        return "  ".join(v.summary() for v in self.meters.values())

    def average_items(self):
        return {k: v.avg for k, v in self.meters.items()}


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        """
        Initialization of AverageMeter
        Parameters
        ----------
        name : str
            Name to display.
        fmt : str
            Format string to print the values.
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def to_cuda(obj):
    if torch.is_tensor(obj):
        return obj.cuda()
    if isinstance(obj, tuple):
        return tuple(to_cuda(t) for t in obj)
    if isinstance(obj, list):
        return [to_cuda(t) for t in obj]
    if isinstance(obj, dict):
        return {k: to_cuda(v) for k, v in obj.items()}
    if isinstance(obj, (int, float, str)):
        return obj
    raise ValueError("'%s' has unsupported type '%s'" % (obj, type(obj)))


def reduce_tensor(tensor, reduction="mean"):
    if torch.is_tensor(tensor):
        rt = tensor.clone()
    else:
        rt = torch.tensor(tensor, device="cuda")
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if reduction == "mean":
        rt /= float(os.environ["WORLD_SIZE"])
    return rt


def reduce_metrics(metrics, distributed=True):
    if distributed:
        return {k: reduce_tensor(v).item() for k, v in metrics.items()}
    return {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}


def reduce_python_object(obj, rank, distributed=True):
    # Collect object from replicas and form a list
    if not distributed:
        return [obj]
    MAX_LENGTH = 2 ** 20  # 1M
    world_size = int(os.environ["WORLD_SIZE"])
    assert 0 <= rank < world_size
    result = []
    for i in range(world_size):
        if rank == i:
            data = pickle.dumps(obj)
            data_length = len(data)
            data = data_length.to_bytes(4, "big") + data
            assert len(data) < MAX_LENGTH
            data += bytes(MAX_LENGTH - len(data))
            data = np.frombuffer(data, dtype=np.uint8)
            assert len(data) == MAX_LENGTH
            tensor = torch.from_numpy(data).cuda()
        else:
            tensor = torch.zeros(MAX_LENGTH, dtype=torch.uint8, device="cuda")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        data = tensor.cpu().numpy().tobytes()
        length = int.from_bytes(data[:4], "big")
        data = data[4:length + 4]
        result.append(pickle.loads(data))
    return result


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class AuxiliaryCrossEntropyLoss(nn.Module):
    def __init__(self, aux_weight, customize_loss=None):
        super(AuxiliaryCrossEntropyLoss, self).__init__()
        self.aux_weight = aux_weight
        if customize_loss is not None:
            self.cross_entropy = customize_loss
        else:
            self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        if isinstance(pred, tuple):
            logits, aux_logits = pred
            return self.cross_entropy(logits, target) + self.aux_weight * self.cross_entropy(aux_logits, target)
        return self.cross_entropy(pred, target)


def write_tensorboard(writer: SummaryWriter, tag: str, metrics: dict, step: int):
    if writer is None:
        return
    for k, v in metrics.items():
        writer.add_scalar(tag + "/" + k, v, global_step=step)


def adjust_learning_rate(args, optimizer, epoch: float):
    # Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    # After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
    if epoch < args.warmup_epochs:
        lr_adj = 1. / args.world_size * (epoch * (args.world_size - 1) / args.warmup_epochs + 1)
    else:
        run_epochs = epoch - args.warmup_epochs
        total_epochs = args.epochs - args.warmup_epochs
        lr_adj = 0.5 * (1 + math.cos(math.pi * run_epochs / total_epochs))

    learning_rate = args.initial_lr * args.world_size * lr_adj
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
    return learning_rate


def save_checkpoint(args, model, f, **kwargs):
    try:
        from apex.parallel import DistributedDataParallel
        if isinstance(model, DistributedDataParallel):
            data = model.module.state_dict()
        else:
            data = model.state_dict()
    except ImportError:
        data = model.state_dict()
    out = {
        "configs": vars(args),
        "state_dict": data,
        **kwargs
    }
    torch.save(out, f)


def load_checkpoint(model, f, args=None, **kwargs):
    data = torch.load(f, map_location="cuda")
    # try to be as compatible as possible
    if all(k.startswith("module.") for k in data):
        data = {k[7:]: v for k, v in data.items()}
    if "state_dict" not in data:
        data = {"configs": {}, "state_dict": data}

    logger.info("Configs of saved checkpoint (%s): %s", f, json.dumps(data["configs"]))
    if args is not None:
        ignored_args = ["distributed", "local_rank", "rank", "world_size", "output_dir", "tmp_dir",
                        "master_addr", "master_port", "is_worker_main", "is_worker_logging"]
        pruned_args = {k: v for k, v in vars(args).items() if k not in ignored_args}
        pruned_configs = {k: v for k, v in data["configs"].items() if k not in ignored_args}
        if pruned_args != pruned_configs:
            logger.error("Checkpoint not loaded. Not a match. Expected %s. Found %s.", pruned_args, pruned_configs)
            return None
    try:
        from apex.parallel import DistributedDataParallel
        if isinstance(model, DistributedDataParallel):
            model.module.load_state_dict(data["state_dict"])
        else:
            model.load_state_dict(data["state_dict"])
    except ImportError:
        model.load_state_dict(data["state_dict"])
    for k, m in kwargs.items():
        m.load_state_dict(data[k])
    return data


def set_running_statistics(configs, model, data_loader, seq2seq=False):
    bn_mean = {}
    bn_var = {}

    def bn_forward_hook(bn, inputs, outputs, mean_est, var_est):
        aggregate_dimensions = (0,) if seq2seq else (0, 2, 3)
        inputs = inputs[0]  # input is a tuple of arguments
        batch_mean = inputs.mean(aggregate_dimensions, keepdim=True)  # 1, C, 1, 1
        batch_var = (inputs - batch_mean) ** 2
        batch_var = batch_var.mean(aggregate_dimensions, keepdim=True)

        batch_mean = torch.squeeze(batch_mean)
        batch_var = torch.squeeze(batch_var)

        mean_est.update(batch_mean.data, inputs.size(0))
        var_est.update(batch_var.data, inputs.size(0))

    handles = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            bn_mean[name] = AverageMeter("mean")
            bn_var[name] = AverageMeter("var")
            handle = m.register_forward_hook(functools.partial(bn_forward_hook,
                                                               mean_est=bn_mean[name],
                                                               var_est=bn_var[name]))
            handles.append(handle)

    model.train()
    with torch.no_grad():
        if seq2seq:
            hidden = model.generate_hidden(data_loader.iterator.batch_size)
        for i in range(configs.bn_sanitize_steps):
            logger.debug("Sanitize Step: %d/%d", i + 1, configs.bn_sanitize_steps)
            images, _ = next(data_loader)
            images = images.cuda()
            if seq2seq:
                _, hidden, _, _ = model(images, hidden)
            else:
                model(images)
        # set_dynamic_bn_running_stats(False)

        for name, m in model.named_modules():
            if name in bn_mean and bn_mean[name].count > 0:
                feature_dim = bn_mean[name].avg.size(0)
                assert isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
                m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
                m.running_var.data[:feature_dim].copy_(bn_var[name].avg)

    for handle in handles:
        handle.remove()
