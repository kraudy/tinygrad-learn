"""
Classic Titanic https://www.kaggle.com/competitions/titanic/overview
"""
from tinygrad import Tensor, nn
import numpy as np
import pandas as pd

X = pd.read_csv("./train.csv")

print(X.head())

