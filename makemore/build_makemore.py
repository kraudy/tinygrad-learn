words = open('names.txt', 'r').read().splitlines()
print(words[:10])
print(f"len: {len(words)}")
print(f"min len word: {min(len(w) for w in words)}")
print(f"max len word: {max(len(w) for w in words)}")


chars = sorted(list(set(''.join(words))))
"""Get alphabet."""
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
"""Letters to int. 28 Elements"""
itos = {i:s for s,i in stoi.items()}

import numpy as np

# Use NumPy array for counting
N_np = np.zeros((27, 27), dtype=np.int32)
"""27x27 Matrix since we have 27 symbols."""

for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N_np[ix1, ix2] += 1
    """Get letter occurrence. This looks like a probability distribution over each letter"""

import torch

#p = N_np[0].astype(float)
#p = p / p.sum()
#print(p)

g = torch.Generator().manual_seed(2147483647)
ix = 0
while True:
    p = N_np[ix].astype(float)
    p = p / p.sum()
    ix = torch.multinomial(torch.from_numpy(p), num_samples=1, replacement=True, generator=g).item()
    print(itos[ix])
    if ix == 0: break
    
print(f"ix: {ix}")
print(f"itos[ix]: {itos[ix]}")

from tinygrad import Tensor, nn, TinyJit, dtypes

N = Tensor(N_np, dtype=dtypes.int32)
print(N.shape)
#print(N.numpy())

