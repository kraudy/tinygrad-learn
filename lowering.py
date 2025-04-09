from tinygrad import Tensor

Tensor.manual_seed(42)

n_symbols = 28
v_size = 3
context_size = 3
elements_size = 100

X = Tensor.randint(elements_size, context_size, low=0, high=n_symbols-1)
"""Sequence of context_size words"""
Y = Tensor.randint(elements_size, low=0, high=n_symbols-1)
"""Word to predict from context"""
C = Tensor.randn(n_symbols, v_size)
"""Vector representatin for each word"""



print(X[0].numpy())
print(C[X[0]].numpy())
print(Y[0].numpy())

