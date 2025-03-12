words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
"""Get alphabet."""
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
"""Letters to int. 27 Elements"""

import numpy as np
import torch

xs, ys = [], []

for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)
  
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
"""(27, 1) means a neuron with vector of 27 wigths"""
print(W.shape)

import torch.nn.functional as F

for k in range(100):
  """Important to note. This structure usumes logits because the layer does basically a discrete class
  selection. So, this should be the last layer of a the network"""
  xenc = F.one_hot(xs, num_classes=27).float()
  """One hot encoding makes a vector of 27 elements and assign one on the element index.
  This becomes the linear representation of the input"""

  logits = xenc @ W
  """These are the 'log counts'.
  Here, we are assuming that the output of the activation are units of exp (e) because we need
  probability and this will make negative numbers close to zero and positive numbers more positive. Very nice!"""

  counts = logits.exp()
  """The actual counts."""

  probs = counts / counts.sum(1, keepdims=True)
  """Normalize the counts to make them probability values."""

  loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
  """0.01*(W**2).mean() Increases the loss which in turns forces a reduction on the weigths to make them
  smallers."""
  print(loss.item())

  W.grad = None # Zero grads
  loss.backward()
  """Backward pass."""

  W.data += -0.1 * W.grad
  """Update.""" 



