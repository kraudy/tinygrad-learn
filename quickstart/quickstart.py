from tinygrad import Linear

class TinyNet:
  def __init__(self):
    self.l1 = Linear(784, 128, bias=False)
    self.l2 = Linear(128, 10, bias=False)

  def __call__(self, x):
    x = self.l1(x).leaky_relu()
    x = self.l2(x)
    return x