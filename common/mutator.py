import copy
import functools
import itertools
import logging

import numpy as np
import torch
from nni.nas.pytorch.mutables import LayerChoice, InputChoice
from nni.nas.pytorch.mutator import Mutator


logger = logging.getLogger(__name__)


def _onehot_tensor(k, n):
    assert 0 <= k < n
    return torch.tensor([i == k for i in range(n)], dtype=torch.bool)


class PrunableMutator(Mutator):
    def __init__(self, model, validation_size, pruned=None, seed=0):
        super().__init__(model)
        self.pruned = dict()
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                self.pruned[mutable.key] = list(range(mutable.length))
            elif isinstance(mutable, InputChoice):
                self.pruned[mutable.key] = list(range(mutable.n_candidates))
        _original_search_space_size = self.search_space_size(self.pruned)
        if pruned is not None:
            for k, v in pruned.items():
                if isinstance(v, int):
                    v = [v]
                self.pruned[k] = v
        self.total_size = self.search_space_size(self.pruned)
        logger.info("Pruned from %d (%.2e) architectures to %d (%.2e).",
                    _original_search_space_size, float(_original_search_space_size),
                    self.total_size, float(self.total_size))
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.validation_size = validation_size
        self.archset = None
        self.val_archset = self.select_validation_archset()

    def select_validation_archset(self):
        ss_size = self.search_space_size(self.pruned)
        if ss_size > self.validation_size * 10:
            return [self._random_sample() for _ in range(self.validation_size)]
        logger.warning("Attempting to choose %d architectures from %d.", self.validation_size, ss_size)
        all_candidates = list(self._all_permutations())
        assert len(all_candidates) == ss_size
        return [all_candidates[i] for i in self.random_state.permutation(ss_size)[:self.validation_size]]

    def reset(self, arch_index=None):
        if arch_index is not None:
            assert not self.training, "Training mode doesn't support index reset."
            arch = self.val_archset[arch_index]
        else:
            if self.training:
                arch = self._random_sample()
            else:
                arch = self.val_archset[self.random_state.randint(len(self.val_archset))]
        self._cache = copy.deepcopy(arch)
        return {"arch": arch}

    def on_forward_layer_choice(self, mutable, *inputs):
        return mutable.choices[self._cache[mutable.key]](*inputs), [i == self._cache[mutable.key] for i in range(mutable.length)]

    def on_forward_input_choice(self, mutable, tensor_list):
        return tensor_list[self._cache[mutable.key]], [i == self._cache[mutable.key] for i in range(mutable.n_candidates)]

    def iterative_reset(self):
        assert not self.training, "Training mode doesn't support iterative reset."
        for i in range(len(self.val_archset)):
            yield self.reset(i)

    def _all_permutations(self):
        keys, vals = zip(*self.pruned.items())
        for val in itertools.product(*vals):
            yield dict(zip(keys, val))

    def _random_sample(self):
        return {k: self.random_state.choice(r).item() for k, r in self.pruned.items()}

    def search_space_size(self, space_dict: dict):
        return functools.reduce(lambda a, b: a * len(b), space_dict.values(), 1)

    def _get_current_archset(self):
        return self.trn_archset if self.training else self.val_archset

    def __len__(self):
        return self.total_size if self.training else len(self.val_archset)
