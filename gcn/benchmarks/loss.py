import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils.logging import print_log

from .metrics import RelevanceCalculator


class RankNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        split_size = input.size(0) // 2
        pred_diff = input[:split_size] - input[-split_size:]
        targ_diff = (target[:split_size] - target[-split_size:] > 0).float()
        return self.bce_loss(pred_diff, targ_diff)


class BRPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, input1, input2, target1, target2):
        input = F.log_softmax(torch.stack((input1, input2), 1), 1)
        target = F.softmax(torch.stack((target1, target2), 1), 1)
        return self.kl_div_loss(input, target)


class Normalizer:
    def __init__(self, mean, std):
        print_log(f'Creating normalizer: mean = {mean:.6f}, std = {std:.6f}', __name__)
        self.mean = mean
        self.std = std

    @classmethod
    def from_data(cls, x):
        if torch.is_tensor(x):
            return cls(torch.mean(x).item(), torch.std(x).item())
        return cls(np.mean(x).item(), np.std(x).item())

    def __call__(self, x, denormalize=False):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


class DCG(object):

    def __init__(self, k=10, gain_type='exp2'):
        """
        :param k: int DCG@k
        :param gain_type: 'exp2' or 'identity'
        """
        self.k = k
        self.discount = self._make_discount(256)
        if gain_type in ['exp2', 'identity']:
            self.gain_type = gain_type
        else:
            raise ValueError('gain type not equal to exp2 or identity')

    def evaluate(self, targets):
        """
        :param targets: ranked list with relevance
        :return: float
        """
        gain = self._get_gain(targets)
        discount = self._get_discount(min(self.k, len(gain)))
        return np.sum(np.divide(gain, discount))

    def _get_gain(self, targets):
        t = targets[:self.k]
        if self.gain_type == 'exp2':
            return np.power(2.0, t) - 1.0
        else:
            return t

    def _get_discount(self, k):
        if k > len(self.discount):
            self.discount = self._make_discount(2 * len(self.discount))
        return self.discount[:k]

    @staticmethod
    def _make_discount(n):
        x = np.arange(1, n+1, 1)
        discount = np.log2(x + 1)
        return discount


class NDCG(DCG):
    """
    NDCG:
    https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
    """

    def __init__(self, k=10, gain_type='exp2'):
        """
        :param k: int NDCG@k
        :param gain_type: 'exp2' or 'identity'
        """
        super(NDCG, self).__init__(k, gain_type)

    def evaluate(self, targets):
        """
        :param targets: ranked list with relevance
        :return: float
        """
        dcg = super(NDCG, self).evaluate(targets)
        ideal = np.sort(targets)[::-1]
        idcg = super(NDCG, self).evaluate(ideal)
        return dcg / idcg

    def maxDCG(self, targets):
        """
        :param targets: ranked list with relevance
        :return:
        """
        ideal = np.sort(targets)[::-1]
        return super(NDCG, self).evaluate(ideal)


class LambdaRankLoss(nn.Module):
    def __init__(self):
        super(LambdaRankLoss, self).__init__()
        self.ndcg_gain_in_train = 'exp2'
        self.ideal_dcg = NDCG(2 ** 9, self.ndcg_gain_in_train)
        self.sigma = 1.0

    def forward(self, prediction, target):
        # target should have been relevance-computed
        target_npy = target.cpu().numpy()
        prediction = prediction.view(-1, 1)
        target = target.view(-1, 1)

        N = 1.0 / self.ideal_dcg.maxDCG(target_npy)

        # compute the rank order of each document
        rank_df = pd.DataFrame({'Y': target_npy, 'doc': np.arange(target_npy.shape[0])})
        rank_df = rank_df.sort_values('Y').reset_index(drop=True)
        rank_order = rank_df.sort_values('doc').index.values + 1

        with torch.no_grad():
            pos_pairs_score_diff = 1.0 + torch.exp(self.sigma * (prediction - prediction.t()))

            rel_diff = target - target.t()
            pos_pairs = (rel_diff > 0).float()
            neg_pairs = (rel_diff < 0).float()
            Sij = pos_pairs - neg_pairs
            if self.ndcg_gain_in_train == 'exp2':
                gain_diff = torch.pow(2.0, target) - torch.pow(2.0, target.t())
            elif self.ndcg_gain_in_train == 'identity':
                gain_diff = target - target.t()

            rank_order_tensor = torch.tensor(rank_order, dtype=torch.float, device=prediction.device).view(-1, 1)
            decay_diff = 1.0 / torch.log2(rank_order_tensor + 1.0) - 1.0 / torch.log2(rank_order_tensor.t() + 1.0)

            delta_ndcg = torch.abs(N * gain_diff * decay_diff)
            lambda_update = self.sigma * (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambda_update = torch.sum(lambda_update, 1, keepdim=True)

            assert lambda_update.shape == prediction.shape

        # print(lambda_update)
        prediction.backward(lambda_update)


def get_criterion(loss_fn):
    if loss_fn == 'ranknet':
        criterion = RankNetLoss()
    elif loss_fn == 'mse':
        criterion = nn.MSELoss()
    elif loss_fn == 'lambdarank':
        criterion = LambdaRankLoss()
    elif loss_fn == 'brp':
        criterion = BRPLoss()
    else:
        raise ValueError(f'Criterion type {loss_fn} not found.')
    return criterion
