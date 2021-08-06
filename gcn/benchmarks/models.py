import collections
from typing import *

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import edge_softmax, GATConv, GraphConv, SortPooling, AvgPooling
from mmcv.utils.logging import print_log

from .graph import GraphDataset


class SortPoolingWithConv(nn.Module):
    def __init__(self, in_dim, out_channels, sortpooling_k=10, kernel_size=5):
        super(SortPoolingWithConv, self).__init__()
        self.sortpooling_k = sortpooling_k
        self.pooling = SortPooling(sortpooling_k)
        self.conv = nn.Sequential(collections.OrderedDict([
            ('conv1', nn.Conv1d(1, out_channels // 2, in_dim, in_dim)),
            ('relu1', nn.ReLU()),
            ('maxpool', nn.MaxPool1d(2, 2)),
            ('conv2', nn.Conv1d(out_channels // 2, out_channels, kernel_size, 1)),
            ('relu2', nn.ReLU())
        ]))

        self.in_dim = in_dim
        self.out_dim = (int((sortpooling_k - 2) / 2 + 1) - kernel_size + 1) * out_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, graph, feats):
        out = self.pooling(graph, feats).view(-1, 1, self.sortpooling_k * self.in_dim)
        out = self.conv(out)
        return out.view(-1, self.out_dim)


class GATEmbedding(nn.Module):
    def __init__(self, num_layers, in_dim, num_hidden, heads, activation, feat_drop, attn_drop,
                 negative_slope, residual, softpooling):
        super(GATEmbedding, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.softpooling = softpooling
        assert len(heads) == len(num_hidden) == self.num_layers
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden[0], heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden[l - 1] * heads[l - 1], num_hidden[l], heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # pooling layer
        if softpooling:
            self.pooling = SortPoolingWithConv(num_hidden[-1] * heads[-1], num_hidden[-1] * heads[-1])
            self.out_dim = self.pooling.out_dim
        else:
            self.pooling = AvgPooling()
            self.out_dim = num_hidden[-1] * heads[-1]
        # basic info
        self.in_dim = in_dim

        print_log('Graph attention: %d -> %d' % (self.in_dim, self.out_dim), __name__)

    def forward(self, graph, feats):
        h = feats
        for l in range(self.num_layers):
            h = self.gat_layers[l](graph, h).flatten(1)
        logits = self.pooling(graph, h)
        return logits


class GraphConvEmbedding(nn.Module):
    def __init__(self, num_layers, in_dim, num_hidden, activation):
        super(GraphConvEmbedding, self).__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.num_hidden = num_hidden
        self.activation = activation
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_dim, num_hidden, activation=activation))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.pooling = SortPoolingWithConv(num_hidden, num_hidden)

        self.in_dim = in_dim
        self.out_dim = self.pooling.out_dim
        print_log('Graph convolution: %d -> %d' % (self.in_dim, self.out_dim), __name__)

    def forward(self, graph, feats):
        h = feats
        for l in range(self.num_layers):
            h = self.layers[l](graph, h).flatten(1)
        logits = self.pooling(graph, h)
        return logits


class BRPConvEmbedding(nn.Module):
    def __init__(self, num_layers, in_dim, num_hidden, activation):
        super(BRPConvEmbedding, self).__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.num_hidden = num_hidden
        self.activation = activation
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_dim, num_hidden, activation=activation))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.pooling = AvgPooling()

        self.in_dim = in_dim
        self.out_dim = num_hidden
        print_log('Graph convolution: %d -> %d' % (self.in_dim, self.out_dim), __name__)

    def forward(self, graph, feats):
        h = feats
        for l in range(self.num_layers):
            h = self.layers[l](graph, h).flatten(1)
        logits = self.pooling(graph, h)
        return logits



class EmbeddingModel(nn.Module):
    def __init__(self, num_graphs_per_sample, hparams_dim, graph_embedding_model, hidden_size, gpu, fuse):
        super(EmbeddingModel, self).__init__()
        self.graph_embedding_model = graph_embedding_model
        if fuse:
            self.fuse = nn.Linear(graph_embedding_model.out_dim * num_graphs_per_sample + hparams_dim, hidden_size)
        else:
            assert graph_embedding_model.out_dim * num_graphs_per_sample + hparams_dim == hidden_size
            self.fuse = None
        self.gpu = gpu
        if self.gpu:
            self.graph_embedding_model.cuda()
            if fuse:
                self.fuse.cuda()

    def forward(self, batch_graph: List[DGLGraph], batch_hparams: torch.Tensor):
        tensors = []
        for graph in batch_graph:
            if self.gpu:
                embeddings = self.graph_embedding_model(graph.to('cuda'), graph.ndata['ops'].to('cuda'))
            else:
                embeddings = self.graph_embedding_model(graph, graph.ndata['ops'])
            tensors.append(embeddings)
        if batch_hparams is not None:
            if self.gpu:
                hparams_tensor = batch_hparams.cuda()
            else:
                hparams_tensor = batch_hparams
            tensors.append(hparams_tensor)
        if len(tensors) == 1:
            ret = tensors[0]
        else:
            ret = torch.cat(tensors, 1)
        if self.fuse is not None:
            return self.fuse(ret)
        return ret


class RegressionHead(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.):
        super(RegressionHead, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.h1_weights(x)
        h1 = self.relu(h1)
        h1 = self.dropout(h1)
        pred = self.h2_weights(h1).view(-1)
        return pred


def _get_model_legacy(embedding_type: str, num_hidden: int, dataset: GraphDataset):
    if embedding_type == 'gat':
        graph_embedding = GATEmbedding(4, dataset.feat_dim, [32, 32, 32, 32],
                                       [4, 4, 4, 4], F.relu, 0, 0, 0.2, False, False)
    elif embedding_type == 'gcn':
        graph_embedding = GraphConvEmbedding(4, dataset.feat_dim, 64, F.relu)
    else:
        raise ValueError(f'Embedding model type {embedding_type} not found.')
    head = RegressionHead(dataset.hparams_dim + dataset.num_graphs_per_sample * graph_embedding.out_dim,
                          num_hidden, 0.5)
    graph_embedding.cuda()
    head.cuda()
    return graph_embedding, head


def get_model(embedding_type: str, num_hidden: int, dataset: GraphDataset, multi_head: int = None, gpu: bool = True):
    dropout_rate = 0.5
    fuse = True
    if embedding_type == 'gat':
        graph_embedding = GATEmbedding(4, dataset.feat_dim, [32, 32, 32, 32],
                                       [4, 4, 4, 4], F.relu, 0, 0, 0.2, False, False)
    elif embedding_type == 'gcn':
        graph_embedding = GraphConvEmbedding(4, dataset.feat_dim, 64, F.relu)
    elif embedding_type == 'gcn256':
        graph_embedding = GraphConvEmbedding(4, dataset.feat_dim, 256, F.relu)
    elif embedding_type == 'brp':
        graph_embedding = BRPConvEmbedding(4, dataset.feat_dim, 600, F.relu)
        dropout_rate = 0.2
    elif embedding_type == 'vanilla':
        graph_embedding = BRPConvEmbedding(4, dataset.feat_dim, num_hidden, F.relu)
        dropout_rate = 0.1
    else:
        raise ValueError(f'Embedding model type {embedding_type} not found.')
    model = EmbeddingModel(dataset.num_graphs_per_sample, dataset.hparams_dim, graph_embedding, num_hidden, gpu, fuse)
    if multi_head:
        head = [RegressionHead(num_hidden, num_hidden, dropout_rate) for _ in range(multi_head)]
        if gpu:
            for module in head:
                module.cuda()
    else:
        head = RegressionHead(num_hidden, num_hidden, dropout_rate)
        if gpu:
            head.cuda()
    if gpu:
        model.cuda()
    return model, head
