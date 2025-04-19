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
    logits = X_batch.matmul(W1).add(b1).relu().matmul(W2).add(b2)
    loss = logits.cross_entropy(Y_batch)

    #zero grads
    optim.zero_grad()

    #backward
    loss.backward()

    # update
    optim.step()

  if epoch % 10 == 0 : print(f"Epoch {epoch}, Loss: {loss.numpy()}")

e = time.time()
print(f"took {(e-s)*1000:.2f}ms")

@TinyJit
def forward(X: Tensor) -> Tensor:
  return X.matmul(W1).add(b1).relu().matmul(W2).add(b2)
 
# Do inference

"""
Relu
Epoch 0, Loss: 6.285268783569336
Epoch 10, Loss: 0.9767337441444397
Epoch 20, Loss: 0.667522132396698
Epoch 30, Loss: 0.4770074486732483
Epoch 40, Loss: 0.5852214097976685
Epoch 50, Loss: 0.34624993801116943
Epoch 60, Loss: 0.7595096826553345
Epoch 70, Loss: 0.47017794847488403
Epoch 80, Loss: 0.3741145431995392
Epoch 90, Loss: 0.5492170453071594
Epoch 100, Loss: 0.34869301319122314

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