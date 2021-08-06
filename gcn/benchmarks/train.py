import argparse
import copy
import logging
import os
import random
import string
import time
from datetime import datetime
from typing import *

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.logging import find_available_filename
from common.preparation import print_config, reset_seed, setup_logger
from common.trainer import AverageMeterGroup
from mmcv.utils.logging import print_log
from scipy import stats, spatial
from torch.utils.data import DataLoader

from .graph import GraphDataset, GraphRecord, GraphSubset, ProductDataset, graph_collate_fn, pair_graph_collate_fn
from .loss import get_criterion, RelevanceCalculator, Normalizer
from .metrics import compute_all_metrics, RELEVANCE_SCALE
from .models import get_model


def test(graph_embedding, head, dataset, normalizer, relevance_calculator, train_indices, args, final=False):
    graph_embedding.eval()
    head.eval()
    test_pred, test_labels = [], []
    for i in range(0, len(dataset), 1000):
        batched_graph, batched_hparams, labels = graph_collate_fn([dataset[i] for i in range(i, min(len(dataset), i + 1000))])
        with torch.no_grad():
            pred = normalizer.denormalize(head(graph_embedding(batched_graph, batched_hparams)))
        test_pred.append(pred)
        test_labels.append(labels)
    del batched_graph, batched_hparams, labels, pred
    test_pred = torch.cat(test_pred, 0)
    test_labels = torch.cat(test_labels, 0)
    test_pred, test_labels = test_pred.cpu().numpy(), test_labels.cpu().numpy()
    assert len(test_labels) == len(dataset)
    if 'test_acc' in dataset.metric_keys:
        test_acc = dataset.fetch_all_metrics('test_acc')
    else:
        test_acc = test_labels
    result = compute_all_metrics(test_pred, test_labels, test_labels=test_acc,
                                 relevance_calculator=relevance_calculator,
                                 free_index=train_indices if args.use_train_samples else None)
    for k in sorted(result.keys()):
        print_log('%s result: %s = %s' % ('Test' if final else 'Intermediate', k, result[k]), __name__)
    if final:
        with open(find_available_filename(args.output_dir, 'test', 'npy'), 'wb') as f:
            print_log(f'Writing to {f.name}.', __name__)
            train_indices = set(train_indices)
            np.save(f, np.stack((np.array([i in train_indices for i in range(len(test_pred))]),
                                 test_pred, test_labels, test_acc)))
    if hasattr(args, 'nni') and args.nni:
        result_key = 'full_top10'
        import nni
        nni.report_intermediate_result(result[result_key])
        if final:
            nni.report_final_result(result[result_key])
    return test_pred


def resume(fp, embedding, head, device='cuda'):
    if not fp:
        return
    data = torch.load(fp, map_location=device)
    embedding.load_state_dict(data['embedding'])
    head.load_state_dict(data['head'])


def sample_train_dataset(dataset, embedding_model, args):
    if not args.greedy:
        return np.random.permutation(len(dataset))[:args.budget].tolist()
    print_log('Use greedy strategy to pick samples...', __name__)
    pool_size = min(len(dataset), 2000)
    pool_indices = np.random.permutation(len(dataset))[:pool_size].tolist()
    dataset = GraphSubset(dataset, pool_indices)
    embedding_model.eval()
    batched_graph, batched_hparams, _ = graph_collate_fn([dataset[i] for i in range(len(dataset))])
    with torch.no_grad():
        embedding = embedding_model(batched_graph, batched_hparams).cpu().numpy()
    print_log(f'Finish with embedding. Embedding shape: {embedding.shape}', __name__)

    insert_order = []
    train_indices = np.zeros(pool_size, dtype=np.bool)

    def push(idx):
        assert not train_indices[idx]
        insert_order.append(pool_indices[idx])
        train_indices[idx] = 1
    distance_matrix = spatial.distance_matrix(embedding, embedding)
    centroid = np.mean(embedding, 0)
    dist_centroid = spatial.distance_matrix(embedding, centroid.reshape((1, -1)))
    push(np.argmin(dist_centroid).item())
    for _ in range(1, args.budget):
        best = np.argmax(np.min(distance_matrix[~train_indices][:, train_indices], 1))
        push(np.where(~train_indices)[0][best].item())
    print_log(f'Greedy sampler takes: {insert_order}', __name__)
    return insert_order


def find_resume(resume_prefix):
    if not os.path.exists(resume_prefix):
        fdir = os.path.dirname(resume_prefix)
        files = [f for f in os.listdir(fdir) if f.startswith(os.path.basename(resume_prefix))]
        print_log('To choose a file from: %s' % files, __name__)
        return os.path.join(fdir, random.choice(files))
    return resume_prefix


def training_loop(graph_embedding, head, criterion, dataset,
                  normalizer, relevance_calculator, epochs, args):
    device = torch.device('cpu') if args.no_cuda else torch.device('cuda')
    optimization_target = 'kendalltau'
    dataloader = None
    if args.loss == 'brp':
        dataset = ProductDataset(dataset)
        dataloader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True,
                                drop_last=True, collate_fn=pair_graph_collate_fn)
    elif args.loss == 'lambdarank':
        optimization_target = 'ndcg'
    if dataloader is None:
        dataloader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True,
                                drop_last=True, collate_fn=graph_collate_fn)
    model_parameters = tuple(graph_embedding.parameters()) + tuple(head.parameters())
    if args.opt_type == 'sgd':
        optimizer = optim.SGD(model_parameters, lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    elif args.opt_type == 'adam':
        optimizer = optim.Adam(model_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.opt_type == 'adamw':
        optimizer = optim.AdamW(model_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f'No optimizer named {args.opt_type} found.')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 0)
    graph_embedding.train()
    head.train()
    best_metric, patience = 0., 0
    best_state = dict()
    for epoch in range(epochs):
        predict_, labels_ = [], []
        loss_total = 0
        for batch in dataloader:
            optimizer.zero_grad()
            if args.loss == 'brp':
                (batched_graph1, batched_hparams1, labels1), (batched_graph2, batched_hparams2, labels2) = batch
                pred1 = head(graph_embedding(batched_graph1, batched_hparams1))
                pred2 = head(graph_embedding(batched_graph2, batched_hparams2))
                loss = criterion(pred1, pred2, labels1.to(device), labels2.to(device))
                loss.backward()
                loss_total += loss.item()
                pred = torch.cat((pred1, pred2), 0)
                labels = torch.cat((labels1, labels2), 0)
            else:
                batched_graph, batched_hparams, labels = batch
                embedding = graph_embedding(batched_graph, batched_hparams)
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
            if loss_group[optimization_target].avg > best_metric:
                best_metric = loss_group[optimization_target].avg
                print_log('Best optimized target (%s) found: %s' % (optimization_target, best_metric), __name__)
                best_state = {
                    'embedding': copy.deepcopy(graph_embedding.state_dict()),
                    'head': copy.deepcopy(head.state_dict())
                }
                patience = 0
            else:
                patience += 1
                if patience >= args.early_stop_patience:
                    print_log('Running out of patience. Break.', __name__)
                    if 'embedding' in best_state:
                        graph_embedding.load_state_dict(best_state['embedding'])
                        head.load_state_dict(best_state['head'])
                    break
    if args.early_stop_patience > 0 and 'embedding' in best_state:
        graph_embedding.load_state_dict(best_state['embedding'])
        head.load_state_dict(best_state['head'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--log_frequency', default=10, type=int)
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--opt_type', default='adam', choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--learning_rate', '--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', '--wd', default=1e-5, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--budget', default=100, type=int)
    parser.add_argument('--batch_size', '--bs', default=20, type=int)
    parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gat', 'gcn', 'gcn256', 'brp', 'vanilla'])
    parser.add_argument('--n_hidden', type=int, default=128)
    parser.add_argument('--test_dataset', type=str, default=None)
    parser.add_argument('--loss', type=str, default='ranknet', choices=['mse', 'ranknet', 'lambdarank', 'brp'])
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--gridsearch', default=False, action='store_true')
    parser.add_argument('--repeatrun', default=0, type=int)
    parser.add_argument('--greedy', default=False, action='store_true')
    parser.add_argument('--iteration', default=1, type=int)
    parser.add_argument('--use_train_samples', default=False, action='store_true')
    parser.add_argument('--reload_weight', default=False, action='store_true')
    parser.add_argument('--early_stop_patience', default=-1, type=int)
    parser.add_argument('--nni', default=False, action='store_true')
    parser.add_argument('--no_cuda', default=False, action='store_true')
    args = parser.parse_args()
    if args.nni:
        import nni
        nni.utils.merge_parameter(args, nni.get_next_parameter())
        current_datetime_str = datetime.now().strftime("%m%d%H%M%S") + ''.join([random.choice(string.digits) for _ in range(3)])
        args.output_dir = os.path.join(args.output_dir, 'archive', current_datetime_str)
    if args.seed == -1:
        args.seed = int(time.time() + random.randint(0, 100)) % 10000
    reset_seed(args.seed)
    setup_logger(args.output_dir)

    device = torch.device('cpu') if args.no_cuda else torch.device('cuda')
    if args.resume:
        args.resume = find_resume(args.resume)

    print_config(vars(args), dump_config=True, output_dir=args.output_dir)
    test_dataset = GraphDataset(args.test_dataset, force_reload=False)
    graph_embedding, head = get_model(args.gnn_type, args.n_hidden, test_dataset, gpu=not args.no_cuda)
    resume(args.resume, graph_embedding, head, device=device)
    criterion = get_criterion(args.loss)
    print_log(f'Test dataset: {len(test_dataset)} samples.', __name__)

    if args.budget == 0:
        print_log('Training dataset is empty. Skip training.', __name__, logging.WARNING)
        test(graph_embedding, head, test_dataset, Normalizer(0., 1.), None, [], args, final=True)
    elif args.iteration == 1:
        train_indices = sample_train_dataset(test_dataset, graph_embedding, args)
        train_dataset = GraphSubset(test_dataset, train_indices)
        normalizer = Normalizer.from_data(train_dataset.fetch_all_metrics())
        rel_calculator = RelevanceCalculator.from_data(train_dataset.fetch_all_metrics(), RELEVANCE_SCALE)

        print_log(f'Train dataset: {len(train_dataset)} samples.', __name__)
        training_loop(graph_embedding, head, criterion, train_dataset, normalizer, rel_calculator, args.epochs, args)
        test(graph_embedding, head, test_dataset, normalizer, None, train_indices, args, final=True)
    else:
        assert not args.greedy
        train_indices = []
        for i in range(args.iteration):
            if args.reload_weight:
                print_log('Reload weight from pretraining...', __name__)
                resume(args.resume, graph_embedding, head, device=device)
            last_iteration = i + 1 == args.iteration
            current_allowance = args.budget // args.iteration * (i + 1) if not last_iteration else args.budget
            random_permutation = np.random.permutation(len(test_dataset)).tolist()

            # fill the rest with random
            for index in random_permutation:
                if len(train_indices) < current_allowance:
                    if index not in train_indices:
                        train_indices.append(index)
                else:
                    break
            print_log('Current training indices (%d): %s' % (len(train_indices), train_indices), __name__)
            train_dataset = GraphSubset(test_dataset, train_indices)
            normalizer = Normalizer.from_data(train_dataset.fetch_all_metrics())
            rel_calculator = RelevanceCalculator.from_data(train_dataset.fetch_all_metrics(), RELEVANCE_SCALE)

            print_log(f'Train dataset: {len(train_dataset)} samples.', __name__)
            training_loop(graph_embedding, head, criterion, train_dataset, normalizer, rel_calculator, args.epochs, args)
            labels = test(graph_embedding, head, test_dataset, normalizer, None, train_indices, args, final=last_iteration)

            # fill the half of the budget in the next iteration with predicted top models
            for index in np.argsort(-labels).tolist():
                if len(train_indices) < current_allowance + args.budget // args.iteration // 2:
                    if index not in train_indices:
                        train_indices.append(index)
                else:
                    break
            print_log('Adding training indices that is from the top: %s' % train_indices[current_allowance:], __name__)


if __name__ == '__main__':
    main()
