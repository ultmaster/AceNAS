from typing import List

import torch.nn as nn


class SearchSpace(nn.Module):
    def validate(self) -> bool:
        return True

    # the following implemetations are for ease of use
    # trainers can have their own way to parse search space and activate

    def searchspace(self) -> dict:
        # quickly retrieve the search space.
        # doesn't mean trainer should use this.
        searchspace = {}
        for child in self.modules():
            if isinstance(child, Mutable):
                searchspace.update(child.searchspace())
        return searchspace

    def activate(self, sample) -> None:
        for child in self.modules():
            if isinstance(child, Mutable):
                child.activate(sample[child.key])

    def prune(self) -> None:
        for child in self.modules():
            if isinstance(child, Mutable):
                child.prune()


class Mutable(nn.Module):
    def __init__(self, key: str):
        super(Mutable, self).__init__()
        self._key = key
        self._value = None

    @property
    def key(self) -> str:
        return self._key

    @property
    def activated(self) -> 'Any':
        return self._value

    def searchspace(self) -> 'Dict[str, Union[List[Any], Tuple[List[Any], int]]]':
        raise NotImplementedError

    def activate(self, value: 'Union[Any, List[Any]]') -> None:
        self._value = value

    def prune(self) -> None:
        pass


class HyperParameter(Mutable):
    # forward will be implemented by sampler

    def __init__(self, key: str, options: 'List[Any]', prior: 'List[Any]' = None):
        super(HyperParameter, self).__init__(key)
        self.options = options
        self.prior = prior
        if self.prior is not None:
            assert len(self.options) == len(self.prior)

    def searchspace(self):
        if self.prior is not None:
            return {self.key: (self.options, self.prior)}
        return {self.key: self.options}

    def execute(self):
        return self.activated

    def forward(self):
        return self.execute()


class MixedOp(Mutable):
    # forward will be implemented by sampler

    def __init__(self, key: str, ops: 'List[torch.nn.Module]'):
        super(MixedOp, self).__init__(key)
        self.num_op_candidates = len(ops)
        self._pruned = False
        if isinstance(ops, nn.Module):
            self.ops = ops
        elif isinstance(ops, dict):
            self.ops = nn.ModuleDict(ops)
        else:
            self.ops = nn.ModuleList(ops)

    def execute(self, *args, **kwargs):
        if self._pruned:
            return self.ops(*args, **kwargs)
        return self.ops[self.activated](*args, **kwargs)

    def prune(self):
        self._op_candidates = self.op_candidates
        assert self._value is not None
        if isinstance(self.ops, nn.ModuleList):
            self.ops = self.ops[self._value]
        elif isinstance(self.ops, nn.ModuleDict):
            self.ops = self.ops[self._value]
        self._pruned = True

    @property
    def op_candidates(self):
        if self._pruned:
            return self._op_candidates
        if isinstance(self.ops, nn.ModuleList):
            return list(range(len(self.ops)))
        return list(self.ops.keys())

    def searchspace(self):
        return {self.key: self.op_candidates}

    def forward(self, *args, **kwargs):
        return self.execute(*args, **kwargs)


class BiasedMixedOp(MixedOp):
    def __init__(self, key: str, ops: 'List[torch.nn.Module]', prior: List[float]):
        super(BiasedMixedOp, self).__init__(key, ops)
        self.prior = prior
        assert sum(prior) == 1

    def searchspace(self):
        return {self.key: (self.op_candidates, self.prior)}


class MixedInput(Mutable):
    # forward will be implemented by sampler

    def __init__(self, key: str, num_input_candidates: int, num_slots: int = 1,
                 custom_reduce: 'Function' = None):
        super(MixedInput, self).__init__(key)
        self.num_input_candidates = num_input_candidates
        assert num_slots in [-1, 1], 'Must choose any number or exactly one.'
        self.num_slots = num_slots
        self.custom_reduce = custom_reduce

    def execute(self, inputs):
        if self.custom_reduce is not None:
            return self.custom_reduce(inputs, self.activated)
        if isinstance(self.activated, int):
            return inputs[self.activated]
        elif len(self.activated) == 1:
            return inputs[self.activated[0]]
        elif len(self.activated) > 1:
            return sum([inputs[i] for i in self.activated])

    def searchspace(self):
        if self.num_slots == 1:
            return {self.key: list(range(self.num_input_candidates))}
        else:
            return {self.key: (list(range(self.num_input_candidates)), self.num_slots)}

    def forward(self, inputs):
        return self.execute(inputs)
