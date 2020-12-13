import torch

from collections.abc import Iterable

def to_one_hot(idx, vocab_size):
    if not isinstance(idx, int):
        raise TypeError('idx must be an interger!')

    if not isinstance(vocab_size, int):
        raise TypeError('vocab_size must be an interger!')

    out = torch.zeros(vocab_size).float()
    out[idx] = 1.0

    return out

def to_multi_hot(idx_iter, vocab_size):
    if not isinstance(idx_iter, Iterable):
        raise TypeError('idx_iter must be an interable!')

    if not isinstance(vocab_size, int):
        raise TypeError('vocab_size must be an interger!')

    out = torch.zeros(vocab_size).float()
    for idx in idx_iter:
        out[idx] = 1.0

    return out
