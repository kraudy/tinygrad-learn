"""
Classic Titanic https://www.kaggle.com/competitions/titanic/overview
"""
from tinygrad import Tensor, nn
import numpy as np
import pandas as pd
import pdb

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

L1_neurons = 50 #make it 100
W1 = Tensor.randn(x_cols, L1_neurons)
b1 = Tensor.randn(L1_neurons)

L2_neurons = 2
W2 = Tensor.randn(L1_neurons, L2_neurons)
b2 = Tensor.randn(L2_neurons)
"""Defined network"""

params = [W1, b1, W2, b2]
lr = 0.01 #validate
optim = nn.optim.SGD(params, lr)
"""Optimizer"""

num_epochs = 101
batch_size = 32

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


    logits = X_batch.matmul(W1).add(b1).tanh() # check relu, sigmoid, etc
    #logits = X.matmul(W1).add(b1).relu()
    loss = logits.matmul(W2).add(b2).cross_entropy(Y_batch)

    #zero grads
    optim.zero_grad()

    #backward
    loss.backward()

    # update
    optim.step()

  if epoch % 10 == 0 : print(f"Loss: {loss.numpy()}")

"""
Relu
Loss: 2.5989575386047363
Loss: 1.7362608909606934
Loss: 1.3206514120101929
Loss: 1.1132731437683105
Loss: 0.9956077933311462
Loss: 0.9190303683280945
Loss: 0.8678664565086365
Loss: 0.8322211503982544
Loss: 0.8064774870872498
Loss: 0.78732830286026
Loss: 0.771884024143219

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