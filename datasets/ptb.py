import os
import logging
from collections import Counter
from typing import Union, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def _batchify(data, batch_size):
    num_batch = data.size(0) // batch_size
    data = data[:num_batch * batch_size]
    data = data.view(batch_size, -1).t().contiguous()
    return data


class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class CorpusDataLoader:
    def __init__(self, source, batch_size: int, seq_len: Union[int, Tuple[int, int, int, int]]):
        self.source = _batchify(source, batch_size)
        self.pos = 0
        self.seq_len = seq_len
        self.batch_size = batch_size

    def __iter__(self):
        self.pos = 0
        return self

    @staticmethod
    def _sample_sequence_length(mean, std, low, high):
        if np.random.random() > 0.95:
            mean /= 2.  # copied from original code
        return min(high, max(low, int(np.random.normal(mean, std))))

    def _get_batch(self, i, seq_len):
        # get the batch starting from i
        if i + 1 + seq_len > self.source.size(0):  # minus one for space of target
            seq_len = self.source.size(0) - i - 1
        return self.source[i:i + seq_len], self.source[i + 1:i + 1 + seq_len]

    def __len__(self):
        # this is just an estimation
        est_seq_len = self.seq_len
        if not isinstance(self.seq_len, int):
            est_seq_len = self.seq_len[0]
        return (len(self.source) + est_seq_len - 1) // est_seq_len

    def __next__(self):
        if self.pos >= self.source.size(0) - 2:  # ensure data and target both have length >= 2
            raise StopIteration
        if isinstance(self.seq_len, int):
            length = self.seq_len
        else:
            length = self._sample_sequence_length(*self.seq_len)
        cur_pos = self.pos
        self.pos += length  # prepare for next step
        # batch is already on cuda
        return self._get_batch(cur_pos, length)


class Corpus:
    def __init__(self, data_dir):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(data_dir, "train.txt"))
        self.valid = self.tokenize(os.path.join(data_dir, "valid.txt"))
        self.test = self.tokenize(os.path.join(data_dir, "test.txt"))

    @property
    def n_tokens(self):
        return len(self.dictionary)

    def tokenize(self, path):
        assert os.path.exists(path), f"Data file {path} not found."
        tokens = []

        # Add words to the dictionary and tokenize file content
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                words = [*line.split(), "<eos>"]
                for word in words:
                    self.dictionary.add_word(word)
                    tokens.append(self.dictionary.word2idx[word])
        logger.info("Total number of tokens in %s: %s", path, len(tokens))
        tokens = torch.tensor(tokens, dtype=torch.long, device="cuda")
        return tokens

    def data_loader(self, source: str, batch_size: int, seq_len: Union[int, Tuple[int, int, int, int]]):
        assert source in ["train", "valid", "test"]
        return CorpusDataLoader(getattr(self, source), batch_size, seq_len)
