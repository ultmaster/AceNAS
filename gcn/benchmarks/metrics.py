import numpy as np
import torch
from scipy import stats
from sklearn.metrics import r2_score
from mmcv.utils.logging import print_log


RELEVANCE_SCALE = 20.


def predict_topk_with_val(pred, val, target, k=1, free_index=None, return_dict=False):
    if return_dict:
        ret = {f'top{k}': predict_topk_with_val(pred, val, target, k=k, free_index=None)}
        if free_index is not None:
            ret.update({f'full_top{k}': predict_topk_with_val(pred, val, target, k=k, free_index=free_index)})
        return ret
    rank = np.argsort(pred)[::-1]
    if free_index is None:
        pool = []
    else:
        pool = [(val[i], i) for i in free_index]
    for i in range(k):
        if i < len(rank):
            pool.append((val[rank[i]], rank[i]))
    _, best_idx = max(pool)
    return target[best_idx]


class RelevanceCalculator:
    def __init__(self, lower, upper, scale):
        print_log(f'Creating relevance calculator: lower = {lower:.6f}, upper = {upper:.6f}, scale = {scale:.2f}', __name__)
        self.lower = lower
        self.upper = upper
        self.scale = scale

    @classmethod
    def from_data(cls, x, scale):
        # TODO use 20% for now
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        lower = np.quantile(x, 0.2)
        upper = np.max(x)
        return cls(lower, upper, scale)

    def __call__(self, x):
        if torch.is_tensor(x):
            return torch.clamp((x - self.lower) / (self.upper - self.lower), 0, 1) * self.scale
        else:
            return np.clip((x - self.lower) / (self.upper - self.lower), 0, 1) * self.scale


def dcg_score(y_true, y_score, k=10, gains='exponential'):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == 'exponential':
        gains = 2 ** y_true - 1
    elif gains == 'linear':
        gains = y_true
    else:
        raise ValueError('Invalid gains option.')

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains='exponential'):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best


def compute_all_metrics(predict, validation, test_labels=None, relevance_calculator=None, simple=False,
                        free_index=None):
    if relevance_calculator is None:
        print_log(f'Relevance calculator is none. Creating one from data ({len(validation)} samples).', __name__)
        relevance_calculator = RelevanceCalculator.from_data(validation, RELEVANCE_SCALE)

    correlation = stats.kendalltau(predict, validation)[0]
    correlation_sp = stats.spearmanr(predict, validation)[0]
    rmse = np.sqrt(np.mean((predict - validation) ** 2)).item()
    r2 = r2_score(validation, predict)
    result = {
        'rmse': rmse if not np.isnan(rmse) else 0.,
        'r2_score': r2 if not np.isnan(r2) else 0.,
        'kendalltau': correlation if not np.isnan(correlation) else -1.,
        'spearman': correlation_sp if not np.isnan(correlation_sp) else -1.,
    }
    if simple:
        return result
    result.update({
        'ndcg10': ndcg_score(relevance_calculator(validation), predict, k=10),
        'ndcg30': ndcg_score(relevance_calculator(validation), predict, k=30),
        'ndcg': ndcg_score(relevance_calculator(validation), predict, k=len(predict)),
    })

    for k in [1, 3, 10, 20, 30, 40, 60, 80, 100, 120, 160, 200]:
        val_top = predict_topk_with_val(predict, validation, validation, k=k, free_index=free_index, return_dict=True)
        if test_labels is not None:
            test_top = predict_topk_with_val(predict, validation, test_labels, k=k, free_index=free_index, return_dict=True)
            result.update(test_top)
            result.update({'val_' + k: v for k, v in val_top.items()})
        else:
            result.update(val_top)
    return result
