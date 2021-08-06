import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler


class ReproducibleDataLoader(DataLoader):
    def __init__(self, dataset, distributed=False, seed=42, **kwargs):
        self.distributed_sampler = None
        if distributed:
            self.distributed_sampler = kwargs['sampler'] = DistributedSampler(dataset, seed=seed)
        else:
            g = torch.Generator()
            g.manual_seed(seed)
            kwargs['sampler'] = RandomSampler(dataset, generator=g)
        super(ReproducibleDataLoader, self).__init__(dataset, **kwargs)

    def __iter__(self):
        self.generator = super().__iter__()
        return self

    def __next__(self):
        item = next(self.generator)
        return tuple([t.cuda() for t in item])
