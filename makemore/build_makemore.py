words = open('names.txt', 'r').read().splitlines()
print(words[:10])
print(f"len: {len(words)}")
print(f"min len word: {min(len(w) for w in words)}")
print(f"max len word: {max(len(w) for w in words)}")


chars = sorted(list(set(''.join(words))))
stoi = {s:i for i,s in enumerate(chars)}
stoi['<S>'] = 26
stoi['<E>'] = 27


import numpy as np

# Use NumPy array for counting
N_np = np.zeros((28, 28), dtype=np.int32)

b = {}
for w in words:
  chs = ['<S>'] + list(w) + ['<E>']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    #N[ix1, ix2] += 1
    N_np[ix1, ix2] += 1

from tinygrad import Tensor, nn, TinyJit, dtypes

N = Tensor(N_np, dtype=dtypes.int32)
print(N.shape)
print(N.numpy())
