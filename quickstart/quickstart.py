from tinygrad import Tensor
from tinygrad import nn
import numpy as np

class TinyNet:
  def __init__(self):
    self.l1 = nn.Linear(28*28, 128, bias=False)
    self.l2 = nn.Linear(128, 10, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    x = x.flatten(1)
    x = self.l1(x).leaky_relu()
    x = self.l2(x)
    return x
  
net = TinyNet()

optim = nn.optim.SGD([net.l1.weight, net.l2.weight], lr=3e-4)

X_train, Y_train, X_test, Y_test = nn.datasets.mnist()

for step in range(1000):
  Tensor.training = True
  samp = Tensor.randint(64, low=0, high=X_train.shape[0])
  batch = X_train[samp]
  labels = Y_train[samp]

  out = net(batch)
  loss = out.sparse_categorical_crossentropy(labels)

  optim.zero_grad()
  loss.backward()
  optim.step()

  pred = out.argmax(axis=-1)
  acc = (pred == labels).mean()

  if step % 100 == 0:
    print(f"Step {step+1} | Loss: {loss.numpy()} | Accuracy: {acc.numpy()}")
