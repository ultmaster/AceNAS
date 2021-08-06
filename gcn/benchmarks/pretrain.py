import argparse
import itertools
import os
import random
import time
from typing import *

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.preparation import print_config, reset_seed, setup_logger
from common.trainer import AverageMeterGroup
from mmcv.utils.logging import print_log
from scipy import stats, spatial
from torch.utils.data import DataLoader

from .graph import GraphDataset, GraphRecord, GraphSubset, graph_collate_fn
from .loss import get_criterion, RelevanceCalculator, Normalizer
from .metrics import compute_all_metrics, RELEVANCE_SCALE
from .models import get_model


def training_loop(graph_embedding, head, criterion, dataset,
                  normalizers, relevance_calculator, epochs, args):
    dataloader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True,
                            drop_last=True, collate_fn=graph_collate_fn)
    model_parameters = list(graph_embedding.parameters()) + list(itertools.chain(*[h.parameters() for h in head]))
    optimizer = optim.Adam(model_parameters, lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 0)
    device = torch.device('cpu') if args.no_cuda else torch.device('cuda')
    for module in [graph_embedding] + head:
        module.train()
    num_heads = len(args.metric_keys)
    for epoch in range(epochs):
        predict_, labels_ = [[] for _ in range(num_heads)], [[] for _ in range(num_heads)]
        loss_total = 0
        for batched_graph, batched_hparams, labels in dataloader:
            optimizer.zero_grad()
            embedding = graph_embedding(batched_graph, batched_hparams)
            loss = []
            for i in range(num_heads):
                pred = head[i](embedding)
                loss.append(criterion(pred, normalizers[i](labels[:, i].to(device))))
                predict_[i].append(pred.detach().cpu().numpy())
                labels_[i].append(labels[:, i].cpu().numpy())
            loss = sum(loss)
            loss.backward()
            loss_total += loss.item()
            optimizer.step()
        loss_group = AverageMeterGroup()
        predict_ = [normalizers[i].denormalize(np.concatenate(predict_[i])) for i in range(num_heads)]
        labels_ = [np.concatenate(labels_[i]) for i in range(num_heads)]
        loss_group.update({'loss': loss_total / len(dataloader)})
        for i in range(num_heads):
            for k, v in compute_all_metrics(predict_[i], labels_[i], relevance_calculator=relevance_calculator[i], simple=True).items():
                loss_group.update({f'{args.metric_keys[i]}_{k}': v})
        if epoch % args.log_frequency == 0 or epoch + 1 == epochs:
            print_log('Neural Predictor Epoch [%d/%d]  %s' % (epoch + 1, epochs, loss_group.summary()), __name__)
        scheduler.step()


def test(graph_embedding, head, dataset, normalizers, relevance_calculators, args):
    for module in [graph_embedding] + head:
        module.eval()
    num_heads = len(args.metric_keys)
    batched_graph, batched_hparams, all_test_labels = graph_collate_fn([dataset[i] for i in range(len(dataset))])
    all_test_data = []
    with torch.no_grad():
        test_embedding = graph_embedding(batched_graph, batched_hparams)
        for i in range(num_heads):
            test_pred = normalizers[i].denormalize(head[i](test_embedding))
            test_pred, test_labels = test_pred.cpu().numpy(), all_test_labels[:, i].cpu().numpy()
            all_test_data += [test_pred, test_labels]
            result = compute_all_metrics(test_pred, test_labels, relevance_calculator=relevance_calculators[i], simple=True)
            for k, v in result.items():
                print_log('Test result: %s_%s = %s' % (args.metric_keys[i], k, v), __name__)
    with open(os.path.join(args.output_dir, 'test.npy'), 'wb') as f:
        np.save(f, np.stack(all_test_data))
    return result


def get_train_val_split(n):
    permute = np.random.permutation(n).tolist()
    return permute[:int(n * 0.8)], permute[int(n * 0.8):]


def main(args):
    print_config(vars(args), dump_config=True, output_dir=args.output_dir)
    print_log(f'Requested to optimize metric keys: {args.metric_keys}. {args.metric_keys[0]} will be used as default.', __name__)
    gt_dataset = GraphDataset(args.test_dataset, force_reload=False)
    pretrain_dataset = GraphDataset(args.train_dataset, metric_key=args.metric_keys, feat_dim=gt_dataset.feat_dim, force_reload=True)
    train_split, valid_split = get_train_val_split(len(pretrain_dataset))
    train_dataset = GraphSubset(pretrain_dataset, train_split)
    valid_dataset = GraphSubset(pretrain_dataset, valid_split)
    graph_embedding, head = get_model(args.gnn_type, args.n_hidden, gt_dataset, multi_head=len(args.metric_keys), gpu=not args.no_cuda)
    criterion = get_criterion(args.loss)
    print_log(f'Train dataset: {len(train_dataset)} samples.', __name__)

    normalizers = [Normalizer.from_data(train_dataset.fetch_all_metrics(key)) for key in args.metric_keys]
    rel_calculators = [RelevanceCalculator.from_data(train_dataset.fetch_all_metrics(key), RELEVANCE_SCALE) for key in args.metric_keys]
    training_loop(graph_embedding, head, criterion, train_dataset, normalizers, rel_calculators, args.epochs, args)
    test(graph_embedding, head, valid_dataset, normalizers, rel_calculators, args)

    torch.save({
        'embedding': graph_embedding.state_dict(),
        'head': head[0].state_dict()  # save only the first head
    }, os.path.join(args.output_dir, 'model.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dataset', type=str)
    parser.add_argument('test_dataset', type=str)
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--log_frequency', default=10, type=int)
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--learning_rate', '--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', '--wd', default=1e-5, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', '--bs', default=20, type=int)
    parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gat', 'gcn', 'gcn256', 'brp'])
    parser.add_argument('--n_hidden', type=int, default=128)
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'ranknet'])
    parser.add_argument('--metric_keys', type=str, nargs='+', required=True)
    parser.add_argument('--emph_weight', type=float, default=1.)
    parser.add_argument('--no_cuda', default=False, action='store_true')
    args = parser.parse_args()
    if args.seed == -1:
        args.seed = int(time.time()) % 10000
    reset_seed(args.seed)
    setup_logger(args.output_dir)
    main(args)
