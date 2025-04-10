from tinygrad import Tensor, nn
#import pdb 

Tensor.manual_seed(42)

#pdb.set_trace()

X = Tensor.randn(2,3)
Y = Tensor.randint(2, low=0, high=1)

W = Tensor.randn(3,2)
b = Tensor.randn(2)

params = [X, W, b]
optim = nn.optim.SGD(params, lr=0.1)

for i in range(3):
  Tensor.training = True
  optim.zero_grad()
  loss = X.matmul(W).add(b).cross_entropy(Y)
  loss.backward()
  print(loss.numpy().item())
  optim.step()