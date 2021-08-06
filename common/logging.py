import os
from collections import OrderedDict

from azureml.core.run import Run
from azureml.exceptions import RunEnvironmentException


class AverageMeterGroup:
    """Average meter group for multiple average meters"""

    def __init__(self):
        self.meters = OrderedDict()

    def reset(self):
        self.meters.clear()

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
        for k, v in self.meters.items():
            yield k, v.avg


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


class AzureMLWriter:
    def __init__(self):
        try:
            self.run = Run.get_context(allow_offline=False)
        except RunEnvironmentException:
            self.run = None

    def add_scalar(self, tag, metric):
        if self.run is None:
            return
        self.run.log(tag, metric)

    def add_average_meter(self, tag, avg_meters: AverageMeterGroup):
        if self.run is None:
            return
        for k, v in avg_meters.average_items():
            self.add_scalar(f'{tag}/{k}', v)


def find_available_filename(output_dir, filename, suffix):
    if not os.path.exists(os.path.join(output_dir, filename + '.' + suffix)):
        return os.path.join(output_dir, filename + '.' + suffix)
    for i in range(1, 100):
        if not os.path.exists(os.path.join(output_dir, filename + str(i) + '.' + suffix)):
            return os.path.join(output_dir, filename + str(i) + '.' + suffix)
