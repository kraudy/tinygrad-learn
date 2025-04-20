"""
Classic Titanic https://www.kaggle.com/competitions/titanic/overview
"""
from tinygrad import Tensor, nn, TinyJit
import numpy as np
import pandas as pd
import pdb
import time

#pdb.set_trace()

X = pd.read_csv("./train.csv", usecols=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
"""Get data"""

X['Age'] = X['Age'].fillna(X['Age'].median())
X['Fare'] = X['Fare'].fillna(X['Fare'].median())
"""Handling NaN"""

X['Sex'] = X['Sex'].replace({'male':1, "female":2})
"""Sex to numeric"""

X = (X - X.mean()) / X.std()
"""Normalizing data"""

print(X.head())

Y = pd.read_csv("./train.csv", usecols=['Survived'])
print(Y.head())
# We need to make this [1, 0] or [0, 1] for alive and death

"""
All columns
['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
"""

x_cols = X.shape[1]
"""6"""
classes = 2

X = Tensor(X.values, dtype='float32')
Y = Tensor(Y.values.flatten(), dtype='int32').one_hot(classes)
"""Tensor from 'numpy.ndarray'"""
print(Y.shape)

class Model:
  def __init__(self):
    self.L1_neurons = 50
    self.W1 = Tensor.randn(x_cols, self.L1_neurons)
    self.b1 = Tensor.randn(self.L1_neurons)

    self.L2_neurons = 2
    self.W2 = Tensor.randn(self.L1_neurons, self.L2_neurons)
    self.b2 = Tensor.randn(self.L2_neurons)
  
  # def training

class M_relu(Model):
  def __init__(self):
    super().__init__()
    # Define weigths for RELU
    self.W1 *= (2 / x_cols)**0.5 
    self.W2 *= (2 / self.L1_neurons)**0.5
  
  def __call__(self, X: Tensor) ->Tensor:
    # return logits
    return X.matmul(self.W1).add(self.b1).relu().matmul(self.W2).add(self.b2)

class M_tanh(Model):
  def __init__(self):
      super().__init__()
  
  def __call__(self, X: Tensor) ->Tensor:
    # return logits
    return X.matmul(self.W1).add(self.b1).tanh().matmul(self.W2).add(self.b2)

RELU = M_relu()
TANH = M_tanh()

lr_sgd_relu = 0.1 #validate
optim_sgd_relu = nn.optim.SGD(nn.state.get_parameters(RELU), lr_sgd_relu)

lr_sgd_tanh = 0.1 #validate
optim_sgd_tanh = nn.optim.SGD(nn.state.get_parameters(TANH), lr_sgd_tanh)

"""Optimizer"""

num_epochs = 101
batch_size = 32

s = time.time()

for epoch in range(num_epochs):
  Tensor.training=True

  indices = np.random.permutation(X.shape[0])
  """This just give us the indexes numbers from 0 to X.shape[0]
  in random order, which we'll use to sample"""

  for i in range(0, X.shape[0], batch_size):
    """This is like a sliding window of indixes"""
    batch_idx = indices[i: i+batch_size].tolist()
    X_batch = X[batch_idx]
    Y_batch = Y[batch_idx]

    #logits = X_batch.matmul(W1).add(b1).tanh() # check relu, sigmoid, etc
    loss_relu = RELU(X_batch).cross_entropy(Y_batch)
    optim_sgd_relu.zero_grad()
    loss_relu.backward()
    optim_sgd_relu.step()

    loss_tanh = TANH(X_batch).cross_entropy(Y_batch)
    optim_sgd_tanh.zero_grad()
    loss_tanh.backward()
    optim_sgd_tanh.step()

  if epoch % 10 == 0 : 
      loss_relu = RELU(X).cross_entropy(Y)
      print(f"Epoch {epoch}, SGD | Loss RELU: {loss_relu.numpy()}")

      loss_tanh = TANH(X).cross_entropy(Y)
      print(f"Epoch {epoch}, SGD | Loss TANH: {loss_tanh.numpy()}")

e = time.time()
print(f"took {(e-s)*1000:.2f}ms")

#@TinyJit
#def forward(X: Tensor) -> Tensor:
#  return X.matmul(W1).add(b1).relu().matmul(W2).add(b2)
 
# Do inference

"""
Epoch 0, SGD | Loss RELU: 0.4353294372558594
Epoch 0, SGD | Loss TANH: 0.9595557451248169
Epoch 10, SGD | Loss RELU: 0.42728522419929504
Epoch 10, SGD | Loss TANH: 0.4000791609287262
Epoch 20, SGD | Loss RELU: 0.4008325934410095
Epoch 20, SGD | Loss TANH: 0.36548030376434326
Epoch 30, SGD | Loss RELU: 0.4011451303958893
Epoch 30, SGD | Loss TANH: 0.3567061126232147
Epoch 40, SGD | Loss RELU: 0.4163704216480255
Epoch 40, SGD | Loss TANH: 0.35577359795570374
Epoch 50, SGD | Loss RELU: 0.418885201215744
Epoch 50, SGD | Loss TANH: 0.35098880529403687
Epoch 60, SGD | Loss RELU: 0.4408760368824005
Epoch 60, SGD | Loss TANH: 0.4212713837623596
Epoch 70, SGD | Loss RELU: 0.39157143235206604
Epoch 70, SGD | Loss TANH: 0.3345852494239807
Epoch 80, SGD | Loss RELU: 0.38610124588012695
Epoch 80, SGD | Loss TANH: 0.33571767807006836
Epoch 90, SGD | Loss RELU: 0.40522268414497375
Epoch 90, SGD | Loss TANH: 0.3348316252231598
Epoch 100, SGD | Loss RELU: 0.3779715895652771
Epoch 100, SGD | Loss TANH: 0.34364986419677734
took 222516.03ms
"""