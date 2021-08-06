import collections

import numpy as np

from .naive import Naive


class DropOp(Naive):
    def __init__(self, criterion, metric_fn, reward_key, drop_k):
        super(DropOp, self).__init__()
        self.criterion = criterion
        self.metric_fn = metric_fn
        self.reward_key = reward_key
        self.drop_k = drop_k

    def on_validation_epoch_end(self, predictions, current_epoch):
        scores = collections.defaultdict(list)
        for prediction in predictions:
            for key, value in prediction.architecture.items():
                scores[key, value].append(prediction.reward)
        scores = {key: np.mean(values).item() for key, values in scores.items()}
        for (k, v), reward in sorted(scores.items(), key=lambda x: x[-1])[:self.drop_k]:
            self.print_console_log('Drop search space: %s = %s (reward = %.6f)' % (k, v, reward),
                                   current_epoch, 'val')
            self.searchspace[k].remove(v)
