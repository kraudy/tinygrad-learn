from tinygrad import Tensor


t = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
t.sum().backward()
