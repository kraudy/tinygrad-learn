"""
Classic Titanic https://www.kaggle.com/competitions/titanic/overview
"""
from tinygrad import Tensor, nn
import numpy as np
import pandas as pd
import pdb

#pdb.set_trace()

X = pd.read_csv("./train.csv", usecols=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
X['Sex'] = X['Sex'].replace({'male':1, "female":2})
print(X.head())

Y = pd.read_csv("./train.csv", usecols=['Survived'])
print(Y.head())
# We need to make this [1, 0] or [0, 1] for alive and death

"""
All columns
['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
"""

x_cols = len(X.columns)
"""6"""
classes = 2

X = Tensor(X.values)
Y = Tensor(Y.values).one_hot(classes)
"""Tensfor from 'numpy.ndarray'"""

L1_neurons = 50 #make it 100
W1 = Tensor.randn(x_cols, L1_neurons)
b1 = Tensor.randn(L1_neurons)

L2_neurons = 2
W2 = Tensor.randn(L1_neurons, L2_neurons)
b2 = Tensor.randn(L2_neurons)
"""Defined network"""

params = [X, W1, b1, W2, b2]
lr = 0.001 #validate
optim = nn.optim.SGD(params, lr)
"""Optimizer"""

for i in range(101):
  Tensor.training=True

  logits = X.matmul(W1).add(b1).relu()#.tanh() # check relu, sigmoid, etc
  loss = logits.matmul(W2).add(b2).cross_entropy(Y)

  #zero grads
  optim.zero_grad()

  #backward
  loss.backward()

  # update
  optim.step()

  if i % 10 == 0 : print(f"Loss: {loss.numpy()}")
