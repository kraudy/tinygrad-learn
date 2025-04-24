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
    self.b1 = Tensor.zeros(self.L1_neurons)

    self.L2_neurons = 2
    self.W2 = Tensor.randn(self.L1_neurons, self.L2_neurons)
    self.b2 = Tensor.zeros(self.L2_neurons)
  
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
    # Define weigths for TANH
    self.W1 *= (1 / x_cols)**0.5 
    self.W2 *= (1 / self.L1_neurons)**0.5
  
  def __call__(self, X: Tensor) ->Tensor:
    # return logits
    return X.matmul(self.W1).add(self.b1).tanh().matmul(self.W2).add(self.b2)



# SGD optimizer
RELU_SGD = M_relu()
TANH_SGD = M_tanh()

lr_sgd_relu = 0.1
optim_sgd_relu = nn.optim.SGD(nn.state.get_parameters(RELU_SGD), lr_sgd_relu)

lr_sgd_tanh = 0.1
optim_sgd_tanh = nn.optim.SGD(nn.state.get_parameters(TANH_SGD), lr_sgd_tanh)

# Adam optimizers
RELU_ADAM = M_relu()
TANH_ADAM = M_tanh()

lr_adam_relu = 0.001  # Smaller lr for Adam
optim_adam_relu = nn.optim.Adam(nn.state.get_parameters(RELU_ADAM), lr_adam_relu)

lr_adam_tanh = 0.001  # Smaller lr for Adam
optim_adam_tanh = nn.optim.Adam(nn.state.get_parameters(TANH_ADAM), lr_adam_tanh)

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

    # SGD
    loss_relu_sgd = RELU_SGD(X_batch).cross_entropy(Y_batch)
    optim_sgd_relu.zero_grad()
    loss_relu_sgd.backward()
    optim_sgd_relu.step()

    loss_tanh_sgd = TANH_SGD(X_batch).cross_entropy(Y_batch)
    optim_sgd_tanh.zero_grad()
    loss_tanh_sgd.backward()
    optim_sgd_tanh.step()
    # ADAM
    

  if epoch % 10 == 0 : 
      loss_relu_sgd = RELU_SGD(X).cross_entropy(Y)
      print(f"Epoch {epoch}, SGD | Loss RELU: {loss_relu_sgd.numpy()}")

      loss_tanh_sgd = TANH_SGD(X).cross_entropy(Y)
      print(f"Epoch {epoch}, SGD | Loss TANH: {loss_tanh_sgd.numpy()}")

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

Bias to Zero
Epoch 0, SGD | Loss RELU: 0.4649984836578369
Epoch 0, SGD | Loss TANH: 0.6744505167007446
Epoch 10, SGD | Loss RELU: 0.4090166389942169
Epoch 10, SGD | Loss TANH: 0.40580445528030396
Epoch 20, SGD | Loss RELU: 0.41389477252960205
Epoch 20, SGD | Loss TANH: 0.388142466545105
Epoch 30, SGD | Loss RELU: 0.4087371826171875
Epoch 30, SGD | Loss TANH: 0.35230016708374023
Epoch 40, SGD | Loss RELU: 0.4085529148578644
Epoch 40, SGD | Loss TANH: 0.33915606141090393
Epoch 50, SGD | Loss RELU: 0.39890801906585693
Epoch 50, SGD | Loss TANH: 0.354353129863739
Epoch 60, SGD | Loss RELU: 0.3790639638900757
Epoch 60, SGD | Loss TANH: 0.33305278420448303
Epoch 70, SGD | Loss RELU: 0.37915492057800293
Epoch 70, SGD | Loss TANH: 0.3537832498550415
Epoch 80, SGD | Loss RELU: 0.37857958674430847
Epoch 80, SGD | Loss TANH: 0.32952880859375
Epoch 90, SGD | Loss RELU: 0.3717404007911682
Epoch 90, SGD | Loss TANH: 0.3393588960170746
Epoch 100, SGD | Loss RELU: 0.3680575489997864
Epoch 100, SGD | Loss TANH: 0.3238118588924408
took 210297.77ms

Tanh weigth regularization 
Epoch 0, SGD | Loss RELU: 0.4452938735485077
Epoch 0, SGD | Loss TANH: 0.4436724781990051
Epoch 10, SGD | Loss RELU: 0.40936824679374695
Epoch 10, SGD | Loss TANH: 0.41842639446258545
Epoch 20, SGD | Loss RELU: 0.3979603350162506
Epoch 20, SGD | Loss TANH: 0.4023357331752777
Epoch 30, SGD | Loss RELU: 0.39638784527778625
Epoch 30, SGD | Loss TANH: 0.39808332920074463
Epoch 40, SGD | Loss RELU: 0.37979522347450256
Epoch 40, SGD | Loss TANH: 0.38466495275497437
Epoch 50, SGD | Loss RELU: 0.3759876489639282
Epoch 50, SGD | Loss TANH: 0.37975436449050903
Epoch 60, SGD | Loss RELU: 0.37766996026039124
Epoch 60, SGD | Loss TANH: 0.3783160448074341
Epoch 70, SGD | Loss RELU: 0.3724023997783661
Epoch 70, SGD | Loss TANH: 0.3751338720321655
Epoch 80, SGD | Loss RELU: 0.3876665234565735
Epoch 80, SGD | Loss TANH: 0.38270702958106995
Epoch 90, SGD | Loss RELU: 0.37578684091567993
Epoch 90, SGD | Loss TANH: 0.3695245683193207
Epoch 100, SGD | Loss RELU: 0.3731934726238251
Epoch 100, SGD | Loss TANH: 0.3674921691417694
took 225023.95ms


"""