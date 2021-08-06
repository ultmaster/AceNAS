from dataclasses import dataclass

from common.preparation import print_config, setup_experiment
from configs import (DatasetConfig, MetricsConfig, ProxylessConfig,
                     RuntimeConfig, PythonConfig, SamplerConfig,
                     TrainerConfig, commonly_used_shortcuts, parse_command_line)
from oneshot.naive import Naive
from searchspace import ProxylessNAS


@dataclass(init=False)
class Config(PythonConfig):
    model: ProxylessConfig
    metrics: MetricsConfig
    runtime: RuntimeConfig
    sampler: SamplerConfig
    trainer: TrainerConfig
    dataset: DatasetConfig


if __name__ == '__main__':
    conf = parse_command_line(Config, shortcuts=commonly_used_shortcuts())
    setup_experiment(conf.runtime)
    print_config(conf)
    model = ProxylessNAS(conf.model)
    trainer = Naive(model, conf.trainer, conf.sampler, conf.runtime, conf.metrics, conf.dataset)
    trainer.fit()
