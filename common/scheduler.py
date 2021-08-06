import torch
from torch.optim.lr_scheduler import _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def test_scheduler():
    v = torch.randn(1)
    optim = torch.optim.SGD([v], lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 95, eta_min=0, last_epoch=-1)
    # scheduler = GradualWarmupScheduler(optim, multiplier=8, total_epoch=5, after_scheduler=scheduler)
    for epoch in range(1, 103):
        scheduler.step()
        print(epoch, optim.param_groups[0]['lr'])


if __name__ == '__main__':
    test_scheduler()
