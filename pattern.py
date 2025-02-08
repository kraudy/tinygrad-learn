from tinygrad import Tensor

a = Tensor.empty(4,4)
b = a + 1
b.realize()
