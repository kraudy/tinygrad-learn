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

lr_sgd = 0.1 #validate
optim_sgd = nn.optim.SGD(nn.state.get_parameters(RELU), lr_sgd)

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
    #logits = X_batch.matmul(W1).add(b1).relu()

    #logits = X_batch.matmul(W1).add(b1).relu().matmul(W2).add(b2)
    #loss = logits.cross_entropy(Y_batch)
    loss_relu = RELU(X_batch).cross_entropy(Y_batch)

    #zero grads
    optim_sgd.zero_grad()

    #backward
    loss_relu.backward()

    # update
    optim_sgd.step()

  if epoch % 10 == 0 : 
      loss_relu = RELU(X).cross_entropy(Y)
      print(f"Epoch {epoch}, Loss RELU: {loss_relu.numpy()}")

e = time.time()
print(f"took {(e-s)*1000:.2f}ms")

#@TinyJit
#def forward(X: Tensor) -> Tensor:
#  return X.matmul(W1).add(b1).relu().matmul(W2).add(b2)
 
# Do inference

"""
Relu
Epoch 0, Loss RELU: 0.44249653816223145
Epoch 10, Loss RELU: 0.4010377526283264
Epoch 20, Loss RELU: 0.3978310823440552
Epoch 30, Loss RELU: 0.3859252333641052
Epoch 40, Loss RELU: 0.38797450065612793
Epoch 50, Loss RELU: 0.3843986690044403
Epoch 60, Loss RELU: 0.41583114862442017
Epoch 70, Loss RELU: 0.38714471459388733
Epoch 80, Loss RELU: 0.4209233224391937
Epoch 90, Loss RELU: 0.3701449930667877
Epoch 100, Loss RELU: 0.3612000644207001
took 119178.16ms

Tanh
Loss: 1.1645634174346924
Loss: 0.3553141951560974
Loss: 0.29564711451530457
Loss: 0.4266332685947418
Loss: 0.5494696497917175
Loss: 0.2601075768470764
Loss: 0.4603317379951477
Loss: 0.22537310421466827
Loss: 0.6060318946838379
Loss: 0.25036874413490295
Loss: 0.39255309104919434
"""