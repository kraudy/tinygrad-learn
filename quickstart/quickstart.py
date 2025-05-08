from tinygrad import Tensor
from tinygrad import nn


class TinyNet:
  def __init__(self):
    self.l1 = nn.Linear(784, 128, bias=False)
    self.l2 = nn.Linear(128, 10, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    x = self.l1(x).leaky_relu()
    x = self.l2(x)
    return x
  
net = TinyNet()

optim = nn.optim.SGD([net.l1.weight, net.l2.weight], lr=3e-4)