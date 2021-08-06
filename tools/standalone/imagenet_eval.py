import argparse
import os

import torch
import tqdm
from datasets.imagenet import imagenet_dataloader
from searchspace.proxylessnas import create_proxylessnas_model


ACENAS_MOBILE = {
    # top-1 75.254 latency: 84.60
    'acenas-m1': '0:0:0:0:1:0:0:0:0:0:1:1:1:1:1:0:1:0:0:0:0:0:1:1:1:1:0:0:0:0:1:0:0:0:1:1:0:0:2:0:1:0:2:1:0:0:1:0:0:1:0:2:0:0:0:2:1:0:2:1:0:2:0:0:2:0:0:0:2:1:0:0:0',
    # top-1 75.07 latency: 84.59
    'acenas-m2': '0:0:0:1:0:0:2:0:1:0:0:1:2:1:1:0:0:1:0:0:0:0:2:0:1:1:0:0:0:2:1:0:2:1:1:1:1:0:2:0:1:0:1:1:0:1:0:0:1:1:0:0:1:0:0:2:1:0:1:1:0:1:0:0:1:1:0:0:2:1:0:0:0',
    # top-1 75.11 latency: 84.92
    'acenas-m3': '0:0:0:0:0:0:0:1:0:1:1:1:0:0:1:0:1:0:0:0:0:1:1:0:1:0:0:0:0:1:1:0:2:1:0:0:1:0:2:1:1:0:2:1:0:2:0:0:2:0:0:1:0:0:0:2:1:0:2:0:0:2:1:0:0:0:0:0:1:1:0:0:0'
}


def evaluate(model, imagenet_dir):
    dataloader = imagenet_dataloader(imagenet_dir, 'test', 512, image_size=224, distributed=False)
    model.cuda()
    model.eval()
    with torch.no_grad():
        correct = total = 0
        pbar = tqdm.tqdm(dataloader, desc='Evaluating on ImageNet')
        for inputs, targets in pbar:
            logits = model(inputs)
            _, predict = torch.max(logits, 1)
            correct += (predict == targets).cpu().sum().item()
            total += targets.size(0)
            pbar.set_postfix({'correct': correct, 'total': total, 'acc': correct / total * 100})
    print('Overall accuracy (top-1):', correct / total * 100)


def estimate_latency(indices):
    from tunas.mobile_cost_model import estimate_cost
    from tunas.mobile_search_space_v3 import PROXYLESSNAS_SEARCH
    print('Latency:', estimate_cost([int(s) for s in indices.split(':')], PROXYLESSNAS_SEARCH))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('net', choices=list(ACENAS_MOBILE.keys()))
    parser.add_argument('imagenet_dir')
    parser.add_argument('--latency', default=False, action='store_true')
    args = parser.parse_args()
    model = create_proxylessnas_model(ACENAS_MOBILE[args.net])
    model.eval()
    model.load_state_dict(torch.load(os.path.join('data', 'checkpoints', args.net + '.pth.tar')))
    if args.latency:
        estimate_latency(ACENAS_MOBILE[args.net])
    evaluate(model, args.imagenet_dir)
