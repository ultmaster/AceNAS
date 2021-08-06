import argparse
import glob
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils
from common.metrics import accuracy
from common.preparation import print_config, reset_seed, setup_logger
from common.profiler import flops_params_counter
from common.trainer import AverageMeter
from datasets.cifar10 import cifar10_dataloader
from mmcv.utils.logging import print_log
from searchspace.nds import create_darts_model


def main():
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('arch', help='which architecture to use')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='path to save the model')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    args = parser.parse_args()

    setup_logger(args.output_dir)
    reset_seed(args.seed)

    print_config(vars(args), output_dir=args.output_dir)
    model = create_darts_model(args.arch)
    model = model.cuda()
    flops, params = flops_params_counter(model, (1, 3, 32, 32), suppress_warnings=True)
    print_log(f'FLOPS = {flops} Params = {params}', __name__)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    summary_path = os.path.join(args.output_dir, 'summary.csv')
    write_summary(summary_path, 'epoch', 'train_loss', 'train_acc', 'eval_loss', 'eval_acc')

    for epoch in range(args.epochs):
        print_log('Epoch [%d/%d] lr = %e' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']), __name__)
        model.drop_path_prob(args.drop_path_prob * epoch / args.epochs)

        train_acc, train_obj = train(model, criterion, optimizer, epoch + 1, args)
        print_log('Epoch [%d/%d] train loss = %.6f acc = %.6f' % (epoch + 1, args.epochs, train_obj, train_acc), __name__)

        valid_acc, valid_obj = test(model, criterion, args)
        print_log('Epoch [%d/%d] valid loss = %.6f acc = %.6f' % (epoch + 1, args.epochs, valid_obj, valid_acc), __name__)

        scheduler.step()

        write_summary(summary_path, epoch + 1, train_obj, train_acc, valid_obj, valid_acc)


def write_summary(output_path, *content):
    with open(output_path, 'a') as f:
        print(*content, sep=',', file=f)


def train(model, criterion, optimizer, epoch, args):
    objs = AverageMeter('loss')
    top1 = AverageMeter('top1')
    model.train()

    train_queue = cifar10_dataloader('data/cifar10', 'augment', args.batch_size, cutout=args.cutout_length,
                                     seed=args.seed + epoch)
    for step, (input, target) in enumerate(train_queue):
        input, target = input.cuda(), target.cuda()

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        loss_aux = criterion(logits_aux, target)
        loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1 = accuracy(logits, target)
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1, n)

        if step % args.report_freq == 0 or step + 1 == len(train_queue):
            print_log('Epoch [%d/%d] train [%03d/%d] %s %s' % (
                epoch, args.epochs, step, len(train_queue), objs, top1), __name__)

    return top1.avg, objs.avg


def test(model, criterion, args):
    objs = AverageMeter('loss')
    top1 = AverageMeter('top1')
    model.eval()

    valid_queue = cifar10_dataloader('data/cifar10', 'test', 500)
    with torch.no_grad():
        for _, (input, target) in enumerate(valid_queue):
            input, target = input.cuda(), target.cuda()

            logits = model(input)
            loss = criterion(logits, target)

            prec1 = accuracy(logits, target)
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1, n)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
