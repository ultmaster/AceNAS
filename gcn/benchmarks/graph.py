import itertools
import json
import pickle
import os
import random
from typing import List

import dgl
import networkx as nx
import numpy as np
import torch
from dgl.data import DGLDataset, load_graphs, save_graphs, load_graphs
from mmcv.utils.logging import print_log


class SingleGraph:
    def __init__(self, graph: 'nx.DiGraph', node_tags: 'List[int]'):
        self.graph = graph
        self.node_tags = node_tags

    def dump(self):
        return {
            'graph': nx.to_dict_of_lists(self.graph),
            'node_tags': self.node_tags,
        }

    @property
    def num_nodes(self):
        return self.graph.number_of_nodes()

    @property
    def num_edges(self):
        return self.graph.number_of_edges()

    @property
    def degs(self):
        return list(dict(self.graph.degree).values())

    def to_dgl_graph(self, feat_dim):
        feat = torch.zeros(self.num_nodes, feat_dim)
        feat.scatter_(1, torch.tensor(self.node_tags, dtype=torch.long).view(-1, 1), 1)
        ret = dgl.from_networkx(self.graph)
        ret = dgl.add_self_loop(ret)
        ret = dgl.to_homogeneous(ret)
        ret.ndata['ops'] = feat
        return ret

    @classmethod
    def load(cls, data):
        graph_data = {int(k): v for k, v in data['graph'].items()}
        return SingleGraph(nx.from_dict_of_lists(graph_data, create_using=nx.DiGraph), data['node_tags'])

    def check_validity(self):
        num_nodes = self.num_nodes
        for u in self.graph.nodes:
            assert 0 <= u < num_nodes, self.graph.nodes
        for u, v in self.graph.edges:
            assert 0 <= u < v < num_nodes, self.graph.edges


class GraphRecord:
    def __init__(self, graphs: 'List[SingleGraph]', hparams: 'List[float]', metrics: 'Dict[str, any]'):
        self.graphs = graphs
        self.hparams = hparams
        self.metrics = metrics

    @property
    def num_nodes(self):
        return sum([g.num_nodes for g in self.graphs])

    @property
    def num_graphs(self):
        return len(self.graphs)

    def dump(self):
        return {
            'graphs': [g.dump() for g in self.graphs],
            'hparams': self.hparams,
            'metrics': self.metrics
        }

    @classmethod
    def load(cls, data):
        return GraphRecord([SingleGraph.load(g) for g in data['graphs']], data['hparams'], data['metrics'])


class GraphDataset(DGLDataset):
    def __init__(self, file_path: str, feat_dim=None, metric_key='accuracy', force_reload=True):
        self._file_path = file_path
        self._feat_dim = feat_dim
        self._metric_key = metric_key
        self._cached_selection = dict()
        super(GraphDataset, self).__init__('nndata', force_reload=force_reload)

    def clear_selection_cache(self):
        # useful for benchmarks where each example has multiple runs
        self._cached_selection = dict()

    def has_cache(self):
        return os.path.exists(self._file_path + '.graph.bin') and os.path.exists(self._file_path + '.info.bin')

    def save(self):
        if not self._force_reload:
            save_graphs(self._file_path + '.graph.bin', list(itertools.chain(*self._graphs)))
            torch.save({
                'num_graphs_per_sample': self._num_graphs_per_sample,
                'hparams': self._hparams,
                'hparams_dim': self._hparams_dim,
                'feat_dim': self._feat_dim,
                'labels_full': self._labels_full,
                'length': len(self._graphs)
            }, self._file_path + '.info.bin')

    def load(self):
        info_data = torch.load(self._file_path + '.info.bin')
        self._num_graphs_per_sample = info_data['num_graphs_per_sample']
        flattened_graphs, _ = load_graphs(self._file_path + '.graph.bin')
        self._graphs = [flattened_graphs[i * self._num_graphs_per_sample:(i + 1) * self._num_graphs_per_sample]
                       for i in range(info_data['length'])]
        self._hparams = info_data['hparams']
        self._hparams_dim = info_data['hparams_dim']
        self._feat_dim = info_data['feat_dim']
        self._labels_full = info_data['labels_full']

    def process(self):
        if self._file_path.endswith('.pkl'):
            with open(self._file_path, 'rb') as f:
                records = [GraphRecord.load(d) for d in pickle.load(f)]
        else:
            with open(self._file_path) as f:
                records = [GraphRecord.load(d) for d in json.load(f)]
        self._num_graphs_per_sample = records[0].num_graphs
        if records[0].hparams:
            self._hparams = torch.tensor([sample.hparams for sample in records], dtype=torch.float)
            self._hparams_dim = len(records[0].hparams)
        else:
            self._hparams = None
            self._hparams_dim = 0
        if self._feat_dim is None:
            self._feat_dim = self._infer_feat_dim(records)
        for sample in records:
            assert sample.num_graphs == self._num_graphs_per_sample
            for graph in sample.graphs:
                graph.check_validity()
        self._graphs = [[g.to_dgl_graph(self._feat_dim) for g in sample.graphs] for sample in records]
        self._labels_full = [sample.metrics for sample in records]
        print_log(f'Process complete. Available keys are: {self.metric_keys}.', __name__)

    def _infer_feat_dim(self, records):
        return len(set([t for multi in records for g in multi.graphs for t in g.node_tags]))

    @property
    def num_graphs_per_sample(self):
        return self._num_graphs_per_sample

    @property
    def hparams_dim(self):
        return self._hparams_dim

    @property
    def feat_dim(self):
        return self._feat_dim

    @property
    def metric_keys(self):
        return list(self._labels_full[0].keys())

    def _get_label(self, idx, key=None):
        if key is None:
            key = self._metric_key
        all_labels = self._labels_full[idx]
        if isinstance(key, list):
            return [all_labels[k] for k in key]
        return all_labels[key]

    def __getitem__(self, idx):
        if self._hparams is None:
            return self._graphs[idx], None, self.fetch_metric(idx)
        else:
            return self._graphs[idx], self._hparams[idx], self.fetch_metric(idx)

    def __len__(self):
        return len(self._graphs)

    def fetch_metric(self, idx, key=None):
        if key is None:
            key = self._metric_key
        if isinstance(key, list):
            return [self.fetch_metric(idx, k) for k in key]
        all_labels = self._labels_full[idx]
        if isinstance(all_labels[key], list):
            if idx not in self._cached_selection:
                self._cached_selection[idx] = (random.randint(0, len(all_labels[key]) - 1), len(all_labels[key]))
            selected_idx, select_from = self._cached_selection[idx]
            assert select_from == len(all_labels[key])
            return all_labels[key][selected_idx]
        else:
            return all_labels[key]

    def fetch_all_metrics(self, key=None):
        return [self.fetch_metric(i, key) for i in range(len(self))]


class GraphSubset:
    def __init__(self, dataset: GraphDataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, item):
        return self.dataset[self.indices[item]]

    def __len__(self):
        return len(self.indices)

    def fetch_metric(self, idx, key=None):
        return self.dataset.fetch_metric(self.indices[idx], key=key)

    def fetch_all_metrics(self, key=None):
        return [self.fetch_metric(i, key) for i in range(len(self))]

    @property
    def num_graphs_per_sample(self):
        return self.dataset.num_graphs_per_sample

    @property
    def hparams_dim(self):
        return self.dataset.hparams_dim

    @property
    def feat_dim(self):
        return self.dataset.feat_dim

    @property
    def metric_keys(self):
        return self.dataset.metric_keys


def graph_collate_fn(samples):
    graphs, hparams, labels = map(list, zip(*samples))
    graphs = map(list, zip(*graphs))
    batched_graph = [dgl.batch(g) for g in graphs]
    batched_labels = torch.tensor(labels)
    if hparams[0] is not None:
        hparams = torch.stack(hparams, 0)
    else:
        hparams = None
    return batched_graph, hparams, batched_labels


class ProductDataset:
    def __init__(self, dataset: GraphDataset):
        self.dataset = dataset

    def __getitem__(self, item):
        x = item // (len(self.dataset) - 1)
        y = item % (len(self.dataset) - 1)
        if y >= x:
            y += 1
        return self.dataset[x], self.dataset[y]

    def __len__(self):
        return len(self.dataset) * (len(self.dataset) - 1)


def pair_graph_collate_fn(samples):
    samples1, samples2 = map(list, zip(*samples))
    return graph_collate_fn(samples1), graph_collate_fn(samples2)
