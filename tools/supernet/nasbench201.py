import collections
import random
from dataclasses import dataclass

from common.preparation import print_config, setup_experiment
from configs import (DatasetConfig, MetricsConfig, NasBench201Config,
                     PythonConfig, RuntimeConfig, SamplerConfig, TrainerConfig,
                     commonly_used_shortcuts, parse_command_line)
from oneshot.naive import Naive
from searchspace.nasbench201 import NasBench201


@dataclass(init=False)
class Config(PythonConfig):
    model: NasBench201Config
    metrics: MetricsConfig
    runtime: RuntimeConfig
    sampler: SamplerConfig
    trainer: TrainerConfig
    dataset: DatasetConfig


def get_architectures(number_of_arch: int):
    from nni.nas.benchmarks.nasbench201 import query_nb201_trial_stats
    all_nasbench201 = list(query_nb201_trial_stats(None, 200, 'cifar100', reduction='mean'))  # we don't care about dataset here
    random.shuffle(all_nasbench201)
    return [t['config']['arch'] for t in all_nasbench201[:number_of_arch]]


if __name__ == '__main__':
    conf = parse_command_line(Config, shortcuts=commonly_used_shortcuts())
    setup_experiment(conf.runtime)
    print_config(conf)
    model = NasBench201(conf.model)
    trainer = Naive(model, conf.trainer, conf.sampler, conf.runtime, conf.metrics, conf.dataset)
    if conf.runtime.evaluate_only:
        trainer.predefined_testset = get_architectures(conf.sampler.num_architectures_per_test)
    trainer.fit()
