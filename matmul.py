from tinygrad import Tensor

print((Tensor.empty(4,4) * Tensor.empty(4, 4)).shape)
