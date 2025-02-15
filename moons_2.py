from tinygrad import Tensor, nn, TinyJit

class Model:
  def __init__(self):
    self.l1 = nn.Linear(2, 16)  
    self.l2 = nn.Linear(16, 1) 
  
  def __call__(self, x: Tensor) -> Tensor:
    x = self.l1(x).relu()
    """
    This activation applies a linear transform that does the scalar multiplication and so sums
    the output, so no sum(0) reduction needed
    """
    return self.l2(x)

from sklearn.datasets import make_moons
x, y = make_moons(n_samples=100, noise=0.1)
y = y*2 - 1 # make y be -1 or 1

print(x.shape, y.shape)

x = Tensor(x)
y = Tensor(y)
print(x.shape, y.shape)

model = Model()

optim = nn.optim.Adam(nn.state.get_parameters(model))
@TinyJit
def train():
  Tensor.training = True # Needed for Optimizer.step()
  optim.zero_grad()
  loss = model(x).sigmoid().binary_crossentropy(y).backward()
  optim.step()
  if i % 100 == 0:  # Print every 100th iteration
    print(f"Iteration {i}: Accuracy = {100 - loss.numpy() * 100}")

for i in range(1001):
  train()
