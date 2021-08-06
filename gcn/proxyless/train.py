import argparse
import copy
import os
import random
import time
from typing import *

import dgl
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import tqdm
from common.preparation import print_config, reset_seed, setup_logger
from common.trainer import AverageMeterGroup
from gcn.benchmarks.graph import ProductDataset
from gcn.benchmarks.loss import Normalizer, RelevanceCalculator, get_criterion
from gcn.benchmarks.metrics import compute_all_metrics
from gcn.benchmarks.train import find_resume, resume
from mmcv.utils.logging import print_log
from torch.utils.data import DataLoader

from .utils import (ACCURACY_MAX, ACCURACY_MIN, RELEVANCE_SCALE,
                    ProxylessDataset, build_model,
                    proxyless_collate_fn, proxyless_pair_collate_fn)


def predict(model, head, dataset, normalizer, args):
    device = torch.device('cpu') if args.no_cuda else torch.device('cuda')
    test_pred = []
    for i in tqdm.tqdm(range(0, len(dataset), 1000), desc='Predicting'):
        batched_graph = dgl.batch([dataset[k] for k in range(i, min(len(dataset), i + 1000))])
        with torch.no_grad():
            test_embedding = model(batched_graph.to(device), batched_graph.ndata['ops'].to(device))
            pred = normalizer.denormalize(head(test_embedding))
        test_pred.append(pred)
    test_pred = torch.cat(test_pred, 0).cpu().numpy()
    return pd.DataFrame({'indices': dataset.tabular_data.indices, 'pred_score': test_pred})


def test(model, head, dataset, normalizer, relevance_calculator, args, return_metric=None):
    device = torch.device('cpu') if args.no_cuda else torch.device('cuda')
    model.eval()
    head.eval()
    batched_graph, test_labels = proxyless_collate_fn([dataset[i] for i in range(len(dataset))])
    with torch.no_grad():
        test_embedding = model(batched_graph.to(device), batched_graph.ndata['ops'].to(device))
        test_pred = normalizer.denormalize(head(test_embedding))
    test_pred, test_labels = test_pred.cpu().numpy(), test_labels.cpu().numpy()
    assert len(test_labels) == len(dataset)
    metrics = compute_all_metrics(test_pred, test_labels, relevance_calculator=relevance_calculator)
    if return_metric is not None:
        return metrics[return_metric]
    for k, v in metrics.items():
        print_log('Test result: %s = %s' % (k, v), __name__)
    return pd.DataFrame({'indices': dataset.tabular_data.indices, 'pred_score': test_pred, 'pred_labels': test_labels})


def training_loop(model, head, criterion, dataset, val_dataset,
                  normalizer, relevance_calculator, epochs, args):
    if args.epochs == 0:
        return
    device = torch.device('cpu') if args.no_cuda else torch.device('cuda')
    optimization_target = 'kendalltau'
    dataloader = None
    if args.loss == 'brp':
        dataset = ProductDataset(dataset)
        dataloader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True,
                                drop_last=True, collate_fn=proxyless_pair_collate_fn)
    elif args.loss == 'lambdarank':
        optimization_target = 'ndcg'
    if dataloader is None:
        dataloader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True,
                                drop_last=True, collate_fn=proxyless_collate_fn)
    model_parameters = tuple(model.parameters()) + tuple(head.parameters())
    if args.opt_type == 'sgd':
        optimizer = optim.SGD(model_parameters, lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    elif args.opt_type == 'adam':
        optimizer = optim.Adam(model_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.opt_type == 'adamw':
        optimizer = optim.AdamW(model_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f'No optimizer named {args.opt_type} found.')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 0)
    best_metric, patience = 0., 0
    best_state = dict()
    for epoch in range(epochs):
        model.train()
        head.train()
        predict_, labels_ = [], []
        loss_total = 0
        for batch in dataloader:
            optimizer.zero_grad()
            if args.loss == 'brp':
                (batched_graph1, labels1), (batched_graph2, labels2) = batch
                pred1 = head(model(batched_graph1.to(device), batched_graph1.ndata['ops'].to(device)))
                pred2 = head(model(batched_graph2.to(device), batched_graph2.ndata['ops'].to(device)))
                loss = criterion(pred1, pred2, labels1.to(device), labels2.to(device))
                loss.backward()
                loss_total += loss.item()
                pred = torch.cat((pred1, pred2), 0)
                labels = torch.cat((labels1, labels2), 0)
            else:
                batched_graph, labels = batch
                embedding = model(batched_graph.to(device), batched_graph.ndata['ops'].to(device))
                pred = head(embedding)
                if args.loss == 'lambdarank':
                    criterion(pred, relevance_calculator(labels.to(device)))
                else:
                    loss = criterion(pred, normalizer(labels.to(device)))
                    loss.backward()
                    loss_total += loss.item()
            optimizer.step()
            predict_.append(pred.detach().cpu().numpy())
            labels_.append(labels.cpu().numpy())
        loss_group = AverageMeterGroup()
        predict_ = normalizer.denormalize(np.concatenate(predict_))
        labels_ = np.concatenate(labels_)
        loss_group.update({'loss': loss_total / len(dataloader),
                           **compute_all_metrics(predict_, labels_, relevance_calculator=relevance_calculator)})
        if epoch % args.log_frequency == 0 or epoch + 1 == epochs:
            print_log('Neural Predictor Epoch [%d/%d]  %s' % (epoch + 1, epochs, loss_group.summary()), __name__)
        scheduler.step()
        if args.early_stop_patience > 0:
            current_metric = test(model, head, val_dataset, normalizer, relevance_calculator, args, return_metric=optimization_target)
            if current_metric > best_metric:
                best_metric = current_metric
                print_log('Best optimized target (%s) found: %s' % (optimization_target, best_metric), __name__)
                best_state = {
                    'embedding': copy.deepcopy(model.state_dict()),
                    'head': copy.deepcopy(head.state_dict())
                }
                patience = 0
            else:
                patience += 1
                if patience >= args.early_stop_patience:
                    print_log('Running out of patience. Break.', __name__)
                    if 'embedding' in best_state:
                        model.load_state_dict(best_state['embedding'])
                        head.load_state_dict(best_state['head'])
                    break
    if args.early_stop_patience > 0 and 'embedding' in best_state:
        model.load_state_dict(best_state['embedding'])
        head.load_state_dict(best_state['head'])


def _get_rel_calculator():
    return RelevanceCalculator(ACCURACY_MIN, ACCURACY_MAX, RELEVANCE_SCALE)


def train_and_eval(train_df, valid_df, test_df, args):
    train_dataset = ProxylessDataset(train_df, metric_keys='90epoch_validation_accuracy')
    valid_dataset = ProxylessDataset(valid_df, metric_keys='90epoch_validation_accuracy')
    test_dataset = ProxylessDataset(test_df, metric_keys='90epoch_validation_accuracy')
    device = torch.device('cpu') if args.no_cuda else torch.device('cuda')
    model, head = build_model(False, gpu=not args.no_cuda)
    resume(args.resume, model, head, device=device)
    criterion = get_criterion(args.loss)

    normalizer = Normalizer.from_data(train_dataset.fetch_all_metrics())
    rel_calculator = _get_rel_calculator()

    print_log(f'Train: {len(train_dataset)} samples. Valid: {len(valid_dataset)} samples. Test: {len(test_dataset)} samples.', __name__)
    training_loop(model, head, criterion, train_dataset, valid_dataset, normalizer, rel_calculator, args.epochs, args)

    if args.mode == 'predict':
        predictions = predict(model, head, test_dataset, normalizer, args)
    else:
        predictions = test(model, head, test_dataset, normalizer, rel_calculator, args)

    return predictions, (model, head)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--log_frequency', default=10, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--opt_type', default='adam', choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--learning_rate', '--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', '--wd', default=1e-5, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--budget', default=100, type=int)
    parser.add_argument('--batch_size', '--bs', default=20, type=int)
    parser.add_argument('--loss', type=str, default='ranknet', choices=['mse', 'ranknet', 'lambdarank', 'brp'])
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--early_stop_patience', default=-1, type=int)
    parser.add_argument('--no_cuda', default=False, action='store_true')
    parser.add_argument('--kfold', default=0, type=int)
    args = parser.parse_args()
    if args.seed == -1:
        args.seed = int(time.time() + random.randint(0, 100)) % 10000
    reset_seed(args.seed)
    setup_logger(args.output_dir)

    if args.resume:
        args.resume = find_resume(args.resume)

    train_df = pd.read_csv('data/proxyless/tunas-proxylessnas-search.csv').iloc[:200]
    valid_df = pd.read_csv('data/proxyless/tunas-proxylessnas-search.csv').iloc[200:250]
    eval_df = pd.read_csv('data/proxyless/tunas-proxylessnas-search.csv')

    print_config(vars(args), dump_config=True, output_dir=args.output_dir)

    output_csv_path = os.path.join(args.output_dir, 'proxyless_gcn_pred.csv')
    predictions, _ = train_and_eval(train_df, valid_df, eval_df, args)
    predictions['partition'] = 'test'
    predictions.loc[predictions.indices.isin(train_df.indices), 'partition'] = 'train'
    predictions.loc[predictions.indices.isin(valid_df.indices), 'partition'] = 'val'
    predictions.to_csv(output_csv_path, index=False)


if __name__ == '__main__':
    main()
