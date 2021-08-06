import copy

_CONTEXT = dict()


def get_context(key: str) -> 'Any':
    return _CONTEXT[key]


def update_context(kv_pairs: dict) -> None:
    global _CONTEXT
    _CONTEXT = copy.deepcopy(kv_pairs)
