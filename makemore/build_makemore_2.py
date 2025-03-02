words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
"""Get alphabet."""
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
"""Letters to int. 28 Elements"""
itos = {i:s for s,i in stoi.items()}

import numpy as np
import torch

N = torch.ones(27, 27)
"""27x27 Matrix since we have 27 symbols.
Start each bytegram count as one to not get -inf on log function."""

xs, ys = [], []

for w in words[:1]:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)
  
xs = torch.tensor(xs)
ys = torch.tensor(ys)

import torch.nn.functional as F

xenc = F.one_hot(xs, num_classes=27).float()
"""One hot encoding makes a vector of 27 elements and assign probability on one to represent each element.
This becomes the linear representation of the input."""
print(xenc)

W = torch.randn((27, 1))
print(xenc.shape, W.shape)
"""(5, 27) x (27, 1) will become (5, 1)"""
print((xenc @ W).shape)
print((xenc @ W))
