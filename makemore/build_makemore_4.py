words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
"""Get alphabet."""
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
"""Letters to int. 27 Elements"""
itos = {i:s for s,i in stoi.items()}

import torch
import torch.nn.functional as F


block_size = 3
X, Y = [], []

for w in words:
  #print(w)
  context = [0] * block_size
  for ch in w + '.':
    ix = stoi[ch]
    X.append(context)
    Y.append(ix)
    #print(''.join(itos[i] for i in context), '---->', itos[ix])
    context = context[1:] + [ix] # crop and append
    """This is like a sliding window."""

X = torch.tensor(X)
Y = torch.tensor(Y)
#print(X); 
print(X.shape) # [32, 3]
#print(Y); 
print(Y.shape) # [32]

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)
"""Embedding"""
W1 = torch.randn((3*2, 100), generator=g)
"""This means a layer with 100 neurons and 6 weigths per neuron."""
b1 = torch.randn(100, generator=g)

W2 = torch.randn((100, 27), generator=g)
"""This means a layer with 27 neurons and 100 weigths per neuron."""
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
print(sum(p.nelement() for p in parameters))
"""3481"""

"""
[ 9, 22,  9]

[ [ 1.2605,  0.8640], => 9
  [-1.5719, -0.7918], => 22
  [ 1.2605,  0.8640]  => 9 ]

[ 1.2605,  0.8640], [-1.5719, -0.7918], [ 1.2605,  0.8640]  ]
"""

for p in parameters: p.requires_grad = True

for _ in range(100):
  #minibatch
  ix = torch.randint(0, X.shape[0], (32, ))
  emb = C[X[ix]]
  """[32, 3, 2]
  Intead of using a one_hot encoding we just index the layer matrix to get out each
  index 2d representation"""

  #(32, 6)
  h = (emb.view(-1,6) @ W1 + b1).tanh()
  """A tensor is basically a 1d vector array with shapes, strides and size that describes a view which 
  determines how the data is represented."""
  """h.shape: [32, 100]"""

  logits = (h @ W2 + b2)
  loss = F.cross_entropy(logits, Y[ix]); 

  print(loss.item())

  for p in parameters: p.grad = None
  loss.backward()
  for p in parameters: p.data -= 0.1 * p.grad

