words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
"""Get alphabet."""
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
"""Letters to int. 27 Elements"""
itos = {i:s for s,i in stoi.items()}

import torch

block_size = 3
X, Y = [], []

for w in words[:5]:
  print(w)
  context = [0] * block_size
  for ch in w + '.':
    ix = stoi[ch]
    X.append(context)
    Y.append(ix)
    print(''.join(itos[i] for i in context), '---->', itos[ix])
    context = context[1:] + [ix] # crop and append

X = torch.tensor(X)
Y = torch.tensor(Y)
print(X); print(X.shape)
print(Y); print(Y.shape)

C = torch.randn(27, 2)
"""Embedding"""

print(C[5])
print(C[X].shape)

print(X[10])
"""[ 9, 22,  9]"""
print(C[X[10]])
print(C[9]); print(C[22]); print(C[9])
"""Intead of using a one_hot encoding we just index the layer matrix to get out each
index 2d representation"""

emb = C[X]
print(emb.shape)
W1 = torch.randn(6, 100)
b1 = torch.randn(100)

emb @ W1 + b1

