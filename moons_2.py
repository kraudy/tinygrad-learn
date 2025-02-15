from tinygrad import Tensor, nn

class Model:
  def __init__(self):
    self.l1 = nn.Linear(2, 16)  
    self.l2 = nn.Linear(16, 1) 
  
  def __call__(self, x: Tensor) -> Tensor:
    x = self.l1(x).relu()#.sum(0)
    return self.l2(x).relu().sum(0)

from sklearn.datasets import make_moons
x, y = make_moons(n_samples=100, noise=0.1)
y = y*2 - 1 # make y be -1 or 1

print(x.shape, y.shape)

x = Tensor(x)
y = Tensor(y)
print(x.shape, y.shape)

model = Model()
acc = (model(x) == y).mean()

print(acc.item())


