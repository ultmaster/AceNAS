import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from dgl.data import DGLDataset
from typing import List, Union
from mmcv.utils.logging import print_log

from gcn.benchmarks.models import GraphConvEmbedding, RegressionHead
from searchspace.proxylessnas import tf_indices_to_pytorch_spec


# constants

BLOCK_NAMES = [
    's2b1', 's2b2', 's2b3', 's2b4',
    's3b1', 's3b2', 's3b3', 's3b4',
    's4b1', 's4b2', 's4b3', 's4b4',
    's5b1', 's5b2', 's5b3', 's5b4',
    's6b1', 's6b2', 's6b3', 's6b4',
    's7b1'
]

STAGE_IDS = [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [5] * 1
SKIP = 'skip'
OPERATORS = ['k3e3', 'k3e6', 'k5e3', 'k5e6', 'k7e3', 'k7e6']
FEAT_DIM = len(OPERATORS) + max(STAGE_IDS) + 1  # 12

RELEVANCE_SCALE = 15.
ACCURACY_MIN = 0.73
ACCURACY_MAX = 0.76

ARCH_LIST_FILE = 'data/proxyless/proxyless-84ms-train.csv'
WS_RESULT_FILE = 'data/proxyless/proxyless-ws-results.csv'

_PROXYLESS_SEARCHSPACE = {
    's1b1_i32o16': ['k3e1'],
    's2b1_i16o24': OPERATORS,
    's2b2_i24o24': OPERATORS + [SKIP],
    's2b3_i24o24': OPERATORS + [SKIP],
    's2b4_i24o24': OPERATORS + [SKIP],
    's3b1_i24o32': OPERATORS,
    's3b2_i32o32': OPERATORS + [SKIP],
    's3b3_i32o32': OPERATORS + [SKIP],
    's3b4_i32o32': OPERATORS + [SKIP],
    's4b1_i32o64': OPERATORS,
    's4b2_i64o64': OPERATORS + [SKIP],
    's4b3_i64o64': OPERATORS + [SKIP],
    's4b4_i64o64': OPERATORS + [SKIP],
    's5b1_i64o96': OPERATORS,
    's5b2_i96o96': OPERATORS + [SKIP],
    's5b3_i96o96': OPERATORS + [SKIP],
    's5b4_i96o96': OPERATORS + [SKIP],
    's6b1_i96o160': OPERATORS,
    's6b2_i160o160': OPERATORS + [SKIP],
    's6b3_i160o160': OPERATORS + [SKIP],
    's6b4_i160o160': OPERATORS + [SKIP],
    's7b1_i160o320': OPERATORS
}

def tf_indices_to_graph(indices):
    spec = tf_indices_to_pytorch_spec(indices, _PROXYLESS_SEARCHSPACE)
    feats = []
    for name, stage_id in zip(BLOCK_NAMES, STAGE_IDS):
        name = [k for k in _PROXYLESS_SEARCHSPACE if k.startswith(name)]
        assert len(name) == 1
        name = name[0]
        if spec[name] == SKIP:
            continue
        operator = OPERATORS.index(spec[name])
        assert operator != -1
        onehot_feat = [int(i == operator) for i in range(len(OPERATORS))] + \
            [int(i == stage_id) for i in range(max(STAGE_IDS) + 1)]
        feats.append(onehot_feat)
    feats = torch.tensor(feats, dtype=torch.long)
    ret = dgl.graph((range(len(feats) - 1), range(1, len(feats))))
    ret = dgl.add_self_loop(ret)
    ret.ndata['ops'] = feats
    return ret


class ProxylessDataset(DGLDataset):
    def __init__(self, tabular_data: pd.DataFrame, metric_keys: Union[List[str], str, None]):
        self.tabular_data = tabular_data
        self.metric_keys = metric_keys
        super().__init__('nndata')

    def process(self):
        if len(self.tabular_data) < 1000:
            self._graphs = [tf_indices_to_graph(indices) for indices in self.tabular_data.indices]
        else:
            self._graphs = []
            for indices in tqdm.tqdm(self.tabular_data.indices, desc='Preprocessing data'):
                self._graphs.append(tf_indices_to_graph(indices))
        if self.metric_keys is None:
            self._metrics = None
        else:
            self._metrics = self.tabular_data[self.metric_keys].to_numpy().astype(np.float32)
            print_log(f'Dataset processed. {len(self._graphs)} graphs. Metrics shape: {self._metrics.shape}')

    def fetch_all_metrics(self, key=None):
        if key is None:
            key = self.metric_keys
        return self.tabular_data[key].to_numpy()

    def __len__(self):
        return len(self.tabular_data)

    def __getitem__(self, idx):
        if self._metrics is None:
            return self._graphs[idx]
        return self._graphs[idx], self._metrics[idx]


class ProxylessSubset:
    def __init__(self, dataset: ProxylessDataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, item):
        return self.dataset[self.indices[item]]

    def __len__(self):
        return len(self.indices)

    def fetch_all_metrics(self, key=None):
        return self.dataset.fetch_all_metrics(key=key)[self.indices]


def proxyless_collate_fn(samples):
    graphs, metrics = map(list, zip(*samples))
    if isinstance(metrics[0], dict):
        metrics = {k: torch.tensor([m[k] for m in metrics]).float() for k in metrics[0]}
    else:
        metrics = torch.tensor(metrics)
    return dgl.batch(graphs), metrics


def proxyless_pair_collate_fn(samples):
    samples1, samples2 = map(list, zip(*samples))
    return proxyless_collate_fn(samples1), proxyless_collate_fn(samples2)


def build_model(multi_head: bool, gpu: bool = False):
    model = GraphConvEmbedding(4, FEAT_DIM, 128, F.relu)
    if gpu:
        model.cuda()
    dropout_rate = 0.5
    if multi_head:
        head = [RegressionHead(128, 256, dropout_rate) for _ in range(multi_head)]
        for module in head:
            module.cuda()
    else:
        head = RegressionHead(128, 256, dropout_rate)
        if gpu:
            head.cuda()
    return model, head
