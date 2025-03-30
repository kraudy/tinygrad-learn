words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
"""Get alphabet."""
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
"""Letters to int. 27 Elements"""
itos = {i:s for s,i in stoi.items()}

import torch
import torch.nn.functional as F

block_size = 4
v_size = 30
"""Encoding feature vector size"""

def build_dataset(words):
  """Three words context"""
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
  print(X.shape) # [32, 3]
  print(Y.shape) # [32]

  return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
"""Here we separate the data into 80%, 10%, 10%"""

#X,Y = build_dataset(words)


g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, v_size), generator=g)
"""Embedding"""
W1 = torch.randn((block_size*v_size, 200), generator=g)
"""This means a layer with 200 neurons and 6 weigths per neuron.
Which is kinda confusing, it would look better like
W1 = torch.randn((200, block_size*2), generator=g)
"""
b1 = torch.randn(200, generator=g)

W2 = torch.randn((200, 27), generator=g)
"""This means a layer with 27 neurons and 200 weigths per neuron."""
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]


"""
[ 9, 22,  9]

[ [ 1.2605,  0.8640], => 9
  [-1.5719, -0.7918], => 22
  [ 1.2605,  0.8640]  => 9 ]

[ 1.2605,  0.8640], [-1.5719, -0.7918], [ 1.2605,  0.8640]  ]
"""

for p in parameters: p.requires_grad = True

for _ in range(300000):
  #minibatch
  ix = torch.randint(0, Xtr.shape[0], (32, )); #print(ix.shape)
  emb = C[Xtr[ix]]
  """[32, block_size, 2]
  Intead of using a one_hot encoding we just index the layer matrix to get out each
  index 2d representation"""

  #(32, 6)
  h = (emb.view(-1, block_size*v_size) @ W1 + b1).tanh()
  """A tensor is basically a 1d vector array with shapes, strides and size that describes a view which 
  determines how the data is represented."""
  """h.shape: [32, 100]"""

  logits = (h @ W2 + b2)
  loss = F.cross_entropy(logits, Ytr[ix]); 

  #print(loss.item())

  for p in parameters: p.grad = None
  loss.backward()
  #lr = 0.1 if _ < 100000 else 0.01
  if _ < 100000:
    lr = 0.1
  elif _ > 100000 and _ < 200000:
    lr = 0.01
  elif _ > 200000:
    lr = 0.001

  for p in parameters: p.data -= lr * p.grad


print(f"Model Params: {sum(p.nelement() for p in parameters)}")
print(f"Block size: {block_size}")
print(f"Vector size: {v_size}")

emb = C[Xdev]
h = torch.tanh(emb.view(-1, block_size*v_size) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print(f"Dev loss: {loss.item()}")

emb = C[Xte]
h = torch.tanh(emb.view(-1, block_size*v_size) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Yte)
print(f"Test loss: {loss.item()}")

"""
My best:

v_size = 3
Dev loss: 2.2214629650115967
Test loss: 2.2182235717773438

v_size = 5
Dev loss: 2.1995632648468018
Test loss: 2.1936023235321045

v_size = 7
Dev loss: 2.1805851459503174
Test loss: 2.1771156787872314

Model Params: 11897
Vector size: 10
Dev loss: 2.16522479057312
Test loss: 2.1652801036834717

Model Params: 15032
Vector size: 15
Dev loss: 2.1518328189849854
Test loss: 2.149149179458618

Model Params: 18167
Vector size: 20
Dev loss: 2.1452529430389404
Test loss: 2.1440892219543457

Model Params: 24437
Vector size: 30
Dev loss: 2.1433637142181396
Test loss: 2.142561435699463

============================

Model Params: 13897
Block size: 4
Vector size: 10
Dev loss: 2.1790213584899902
Test loss: 2.174254894256592

=> **** Best time **** <=
Block size: 4
Vector size: 30
Dev loss: 2.142103672027588
Test loss: 2.1336076259613037

Block size: 4
Vector size: 35
Dev loss: 2.1501615047454834
Test loss: 2.14841890335083

Block size: 4
Vector size: 40
Dev loss: 2.1450135707855225
Test loss: 2.1384081840515137

Model Params: 15897
Block size: 5
Vector size: 10
Dev loss: 2.1916513442993164
Test loss: 2.1935253143310547

Block size: 5
Vector size: 30
Dev loss: 2.223001718521118
Test loss: 2.2083425521850586

============================

Krapathy's Target:
Test loss: 2.1701
"""
