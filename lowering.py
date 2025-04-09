from tinygrad import Tensor

Tensor.manual_seed(42)

n_symbols = 28
v_size = 3

words = Tensor.randint(100, low=0, high=n_symbols-1)
"""Words as randoms int"""
C = Tensor.randn(n_symbols, v_size)
"""Vector representatin for each word"""

print(C[words[:2]].numpy())


