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

from tinygrad import Tensor, nn, TinyJit, dtypes

N = Tensor(N_np, dtype=dtypes.int32)
print(N.shape)
print(N.numpy())

