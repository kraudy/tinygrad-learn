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
    """This is like a sliding window."""

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
"""
[ [ 1.2605,  0.8640], => 9
  [-1.5719, -0.7918], => 22
  [ 1.2605,  0.8640]  => 9 ]
"""
print(C[9]); print(C[22]); print(C[9])
"""
[ 1.2605,  0.8640], [-1.5719, -0.7918], [ 1.2605,  0.8640]  ]
Intead of using a one_hot encoding we just index the layer matrix to get out each
index 2d representation"""

emb = C[X]
print(emb.shape)
"""[32, 3, 2]"""
W1 = torch.randn(3*2, 100)
"""This means a layer with 100 neurons and 6 weigths per neuron."""
b1 = torch.randn(100)

#print(W1)

h = emb.view(32,6) @ W1 + b1
"""A tensor is basically a 1d vector array with shapes, strides and size that describes a view which 
determines how the data is represented."""
print(h.shape)

