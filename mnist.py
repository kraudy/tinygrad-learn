from tinygrad import Tensor, nn

class Model:
  def __init__(self):
    self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3))     
    self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
    self.l3 = nn.Linear(1600, 10)

  def __call__(self, x: Tensor) -> Tensor:
    x = self.l1(x).relu().max_pool2d((2,2))
    x = self.l2(x).relu().max_pool2d((2,2))
    return self.l3(x.flatten(1).dropout(0.5))

from tinygrad.nn.datasets import mnist
X_train, Y_train, X_test, Y_test = mnist()
print(X_train.shape, X_train.dtype, Y_train.shape, Y_train.dtype)
"""
(60000, 1, 28, 28) dtypes.uchar (60000,) dtypes.uchar
"""

model = Model()
acc = (model(X_test).argmax(axis=1) == Y_test).mean()

print(acc.item())

optim = nn.optim.Adam(nn.state.get_parameters(model))
batch_size = 128
def step():
  Tensor.training = True  # makes dropout work
  samples = Tensor.randint(batch_size, high=X_train.shape[0])
  X, Y = X_train[samples], Y_train[samples]
  optim.zero_grad()
  loss = model(X).sparse_categorical_crossentropy(Y).backward()
  optim.step()
  return loss

import timeit
timeit.repeat(step, repeat=5, number=1)

from tinygrad import GlobalCounters, Context
GlobalCounters.reset()
with Context(DEBUG=2): step()


from tinygrad import TinyJit
jit_step = TinyJit(step)
timeit.repeat(jit_step, repeat=5, number=1)