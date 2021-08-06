import collections
import random

import torch

from common.metrics import ArchitectureResult
from common.profiler import flops_params_counter
from .sampler import SamplerMixin


class Naive(SamplerMixin):
    def __init__(self, model: 'nn.Module',
                 trainer_config: 'TrainerConfig',
                 sampler_config: 'SamplerConfig',
                 runtime_config: 'RuntimeConfig',
                 metrics_config: 'MetricsConfig',
                 dataset_config: 'DatasetConfig'):
        super(Naive, self).__init__(model, trainer_config, sampler_config,
                                    runtime_config, metrics_config, dataset_config)
        self._profile_data_shape = None
        if self.distributed:
            self._distributed_rank = torch.distributed.get_rank()
            self._distributed_world_size = torch.distributed.get_world_size()

    def training_step(self, batch, batch_idx):
        self.resample()
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        metrics = self.metrics_fn(y_hat, y)
        return loss, metrics

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            architecture = self.resample()
            x, y = batch
            y_hat = self.model(x)
            metrics = self.metrics_fn(y_hat, y)
            return ArchitectureResult(architecture, metrics[self.reward_key]), metrics

    def test_single_architecutre(self) -> 'Dict[str, Any]':
        dataloader = self.test_dataloader()
        metrics_dict = collections.defaultdict(float)
        total = 0
        with torch.no_grad():
            for x, y in dataloader:
                y_hat = self.model(x)
                metrics = self.metrics_fn(y_hat, y)
                total += y_hat.size(0)
                for k, v in metrics.items():
                    metrics_dict[k] += v * y_hat.size(0)
        return {k: v / total for k, v in metrics_dict.items()}

    def profile_single_architecture(self) -> 'Dict[str, Any]':
        if self._profile_data_shape is None:
            self._profile_data_shape = list(next(iter(self.test_dataloader()))[0].size())
            self._profile_data_shape[0] = 1

        input_data = torch.randn(self._profile_data_shape, device='cuda')
        total_ops, total_params = flops_params_counter(self.model_inner, (input_data, ), suppress_warnings=True)
        return {'flops': total_ops / 1e6, 'params': total_params / 1e6}

    def resample(self):
        if self.distributed:
            result = None
            for i in range(self._distributed_world_size):
                t = self._resample_impl()
                if i == self._distributed_rank:
                    result = t
            assert result is not None
            self.activate(result)
            return result
        else:
            return self._resample_impl()

    def _resample_impl(self):
        num_vertices = None
        while True:
            result = {key: self.choice_helper(values) for key, values in self.searchspace.items()}

            # related to nasbench101, `num_vertices` is reserved
            if 'num_vertices' in result:
                if num_vertices is not None:
                    result['num_vertices'] = num_vertices  # avoid affecting the prior of `num_vertices`
                else:
                    num_vertices = result['num_vertices']

            self.activate(result)
            if not self.model_inner.validate():
                continue
            return result
