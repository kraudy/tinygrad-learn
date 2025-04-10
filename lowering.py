from tinygrad import Tensor, nn

Tensor.manual_seed(42)

n_symbols = 28
v_size = 3
context_size = 3
elements_size = 1000

X = Tensor.randint(elements_size, context_size, low=0, high=n_symbols-1)
"""Sequence of context_size words"""
Y = Tensor.randint(elements_size, low=0, high=n_symbols-1)
"""Word to predict from context"""
C = Tensor.randn(n_symbols, v_size)
"""Vector representatin for each word"""


L1_neurons = 100
W1 = Tensor.randn(v_size*context_size, L1_neurons)
b1 = Tensor.randn(L1_neurons)

L2_neurons = n_symbols
W2 = Tensor.randn(L1_neurons, L2_neurons)
b2 = Tensor.randn(L2_neurons)

parameters = [C, W1, b1, W2, b2]
lr = 0.1
optimizer = nn.optim.SGD(parameters, lr)

print(X[0].numpy())
print(C[X[:2]].flatten(1).numpy())
print(Y[0].numpy())


for i in range(100):
  Tensor.training = True

  logits = (C[X].flatten(1)).matmul(W1).add(b1).tanh()
  loss = logits.matmul(W2).add(b2).cross_entropy(Y)

  optimizer.zero_grad()

  loss.backward()

  optimizer.step()

  if i % 10 == 0:
    print(f"Loss: {loss.numpy()}")