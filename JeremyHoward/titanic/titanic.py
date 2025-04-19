"""
Classic Titanic https://www.kaggle.com/competitions/titanic/overview
"""
from tinygrad import Tensor, nn
import numpy as np
import pandas as pd

X = pd.read_csv("./train.csv", usecols=['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
X['Sex'] = X['Sex'].replace({'male':1, "female":2})
print(X.head())

Y = pd.read_csv("./train.csv", usecols=['Survived'])
print(Y.head())

print(type(Y.values))

"""
All columns
['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
"""

