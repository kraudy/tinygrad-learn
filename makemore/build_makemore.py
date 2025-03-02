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
    """Get letter occurrence. This will be used for the probability distribution over each bytegram"""

import torch

P = torch.from_numpy(N_np.astype(float))
P /= P.sum(1, keepdims=True)
"""We get the sum across rows and keep dimesions to get matrix of shape (27,1) so when the broadcasting
is done for the division, the sum of the column get expanded to the 27 elemnts getting (27, 27)"""
print(f"P[0] Prob: {P[0].sum().item()}")

g = torch.Generator().manual_seed(2147483647)
for i in range(50):
  out = []
  ix = 0
  while True:
      p = P[ix]
      ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
      """This takes the probability distribution and makes a generator.
      So, a neural network can be viewed as a generator with an underlying probability distribution."""
      out.append(itos[ix])
      if ix == 0: break
  print(''.join(out))

"""At this point, we have basically 'train' a nueral network since we have the probability distribution
that describes the input data. This will usually be a distributed representations on the weights."""

log_likelihood = 0.0
n = 0

#for w in words[:3]:
for w in ['roberto']:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]
    """Here we are basically evaluating the probability distribution with an specific value. This is the
    forward pass"""
    logprop = torch.log(prob)
    """Now we get the loglikelyhood"""
    log_likelihood += logprop
    """Sumatory of the loglikelyhood to get the 'model' prediction"""
    n += 1
    print(f"{ch1}{ch2}: {prob:.4f} {-logprop:.4f}")

print(f"loss: {-log_likelihood}")

from tinygrad import Tensor, nn, TinyJit, dtypes

N = Tensor(N_np, dtype=dtypes.int32)
print(N.shape)
#print(N.numpy())

