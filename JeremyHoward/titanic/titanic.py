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

"""
All columns
['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
"""

x_cols = len(X.columns)
"""6"""
y_cols = len(Y.columns)
"""1"""

X = Tensor(X.values)
Y = Tensor(Y.values)
"""Tensfor from 'numpy.ndarray'"""

L1_neurons = 24 #make it 100
W1 = Tensor.randn(x_cols, L1_neurons)
b1 = Tensor.randn(L1_neurons)

L2_neurons = y_cols
W2 = Tensor.randn(L1_neurons, L1_neurons)
b2 = Tensor.randn(L1_neurons)

logits = X.matmul(W1).add(b1).tanh() # check relu, sigmoid, etc
