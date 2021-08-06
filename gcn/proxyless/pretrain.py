import argparse
import itertools
import os
import time
from typing import *

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from common.preparation import print_config, reset_seed, setup_logger
from common.trainer import AverageMeterGroup
from gcn.benchmarks.loss import Normalizer, RelevanceCalculator, get_criterion
from gcn.benchmarks.metrics import compute_all_metrics
from mmcv.utils.logging import print_log
from torch.utils.data import DataLoader

from .utils import (ARCH_LIST_FILE, WS_RESULT_FILE, RELEVANCE_SCALE, ProxylessDataset,
                    ProxylessSubset, build_model, proxyless_collate_fn)



def training_loop(model, head, criterion, dataset,
                  normalizers, relevance_calculator, epochs, args):
    dataloader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True,
                            drop_last=True, collate_fn=proxyless_collate_fn)
    model_parameters = list(model.parameters()) + list(itertools.chain(*[h.parameters() for h in head]))
    optimizer = optim.Adam(model_parameters, lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 0)
    device = torch.device('cpu') if args.no_cuda else torch.device('cuda')
    for module in [model] + head:
        module.train()
    num_heads = len(args.metric_keys)
    for epoch in range(epochs):
        predict_, labels_ = [[] for _ in range(num_heads)], [[] for _ in range(num_heads)]
        loss_total = 0
        for graph, labels in dataloader:
            optimizer.zero_grad()
            embedding = model(graph.to(device), graph.ndata['ops'].to(device))
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


def test(model, head, dataset, normalizers, relevance_calculators, args):
    for module in [model] + head:
        module.eval()
    num_heads = len(args.metric_keys)
    batched_graph, all_test_labels = proxyless_collate_fn([dataset[i] for i in range(len(dataset))])
    device = torch.device('cpu') if args.no_cuda else torch.device('cuda')
    all_test_data = {}
    with torch.no_grad():
        test_embedding = model(batched_graph.to(device), batched_graph.ndata['ops'].to(device))
        for i in range(num_heads):
            test_pred = normalizers[i].denormalize(head[i](test_embedding))
            test_pred, test_labels = test_pred.cpu().numpy(), all_test_labels[:, i].cpu().numpy()
            all_test_data[args.metric_keys[i]] = test_labels
            all_test_data[args.metric_keys[i] + '_pred'] = test_pred
            result = compute_all_metrics(test_pred, test_labels, relevance_calculator=relevance_calculators[i], simple=True)
            for k, v in result.items():
                print_log('Test result: %s_%s = %s' % (args.metric_keys[i], k, v), __name__)
    return result, pd.DataFrame(all_test_data)


def get_train_val_split(n):
    permute = np.random.permutation(n).tolist()
    return permute[:int(n * 0.8)], permute[int(n * 0.8):]


def main(args):
    print_config(vars(args), dump_config=True, output_dir=args.output_dir)
    print_log(f'Requested to optimize metric keys: {args.metric_keys}. {args.metric_keys[0]} will be used as default.', __name__)

    architectures = pd.read_csv(ARCH_LIST_FILE)
    legal_indices = architectures[architectures.src == 'random'].indices
    ws_results = pd.read_csv(WS_RESULT_FILE)
    ws_results = ws_results[ws_results.indices.isin(legal_indices)]
    ws_results = ws_results.merge(architectures[['indices', 'simulated_pixel1_time_ms']])

    pretrain_dataset = ProxylessDataset(ws_results, metric_keys=args.metric_keys)
    train_split, valid_split = get_train_val_split(len(pretrain_dataset))
    train_dataset = ProxylessSubset(pretrain_dataset, train_split)
    valid_dataset = ProxylessSubset(pretrain_dataset, valid_split)
    model, head = build_model(len(args.metric_keys), gpu=not args.no_cuda)
    criterion = get_criterion(args.loss)
    print_log(f'Train dataset: {len(train_dataset)} samples.', __name__)

    normalizers = [Normalizer.from_data(train_dataset.fetch_all_metrics(key)) for key in args.metric_keys]
    rel_calculators = [RelevanceCalculator.from_data(train_dataset.fetch_all_metrics(key), RELEVANCE_SCALE) for key in args.metric_keys]
    training_loop(model, head, criterion, train_dataset, normalizers, rel_calculators, args.epochs, args)
    _, predictions = test(model, head, valid_dataset, normalizers, rel_calculators, args)

    predictions.to_csv(os.path.join(args.output_dir, 'proxyless_gcn_pretrain_pred.csv'), index=False)
    torch.save({
        'embedding': model.state_dict(),
        'head': head[0].state_dict()  # save only the first head
    }, os.path.join(args.output_dir, 'proxyless_gcn_pretrain.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--log_frequency', default=10, type=int)
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--learning_rate', '--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', '--wd', default=1e-5, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', '--bs', default=20, type=int)
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'ranknet'])
    parser.add_argument('--metric_keys', type=str, nargs='+', required=True)
    parser.add_argument('--no_cuda', default=False, action='store_true')
    args = parser.parse_args()
    if args.seed == -1:
        args.seed = int(time.time()) % 10000
    reset_seed(args.seed)
    setup_logger(args.output_dir)
    main(args)
