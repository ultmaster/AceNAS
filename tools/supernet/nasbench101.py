import collections
import random
from dataclasses import dataclass

from common.preparation import print_config, setup_experiment
from configs import (DatasetConfig, MetricsConfig, NasBench101Config,
                     PythonConfig, RuntimeConfig, SamplerConfig, TrainerConfig,
                     commonly_used_shortcuts, parse_command_line)
from oneshot.naive import Naive
from searchspace.nasbench101 import NasBench101


@dataclass(init=False)
class Config(PythonConfig):
    model: NasBench101Config
    metrics: MetricsConfig
    runtime: RuntimeConfig
    sampler: SamplerConfig
    trainer: TrainerConfig
    dataset: DatasetConfig


def get_architectures(number_of_arch: int):
    from nni.nas.benchmarks.nasbench101 import query_nb101_trial_stats
    all_nasbench101 = list(query_nb101_trial_stats(None, 108, reduction='mean'))
    random.shuffle(all_nasbench101)
    result = []
    counter = collections.Counter([r['config']['num_vertices'] for r in all_nasbench101])
    # Counter({7: 359082, 6: 62010, 5: 2441, 4: 84, 3: 6, 2: 1})
    for r in all_nasbench101[:number_of_arch]:
        num_vertices = r['config']['num_vertices']
        cvt = {'num_vertices': num_vertices, **r['config']['arch']}
        if num_vertices < 7:
            cvt['input6'] = cvt.pop(f'input{num_vertices - 1}')
        for i in range(1, 6):
            if f'input{i}' not in cvt:
                cvt[f'input{i}'] = [0]
                cvt[f'op{i}'] = 'maxpool3x3'
        result.append(cvt)
    return result


if __name__ == '__main__':
    conf = parse_command_line(Config, shortcuts=commonly_used_shortcuts())
    setup_experiment(conf.runtime)
    print_config(conf)
    model = NasBench101(conf.model)
    trainer = Naive(model, conf.trainer, conf.sampler, conf.runtime, conf.metrics, conf.dataset)
    if conf.runtime.evaluate_only:
        trainer.predefined_testset = get_architectures(conf.sampler.num_architectures_per_test)
    trainer.fit()
