from tinygrad import Tensor, nn
import numpy as np
import pdb
import psutil

cache_path="./data/flow_images.npy"

print(f"Defining network")
class Model():
  def __init__(self):
    # This first layer is kinda big
    self.W1 = Tensor.randn(480*640*3, 600) * (1 / 480*640*3) ** 0.5
    self.b1 = Tensor.zeros(600)

    self.W2 = Tensor.randn(600, 400) * (1 / 600) ** 0.5
    self.b2 = Tensor.zeros(400)

    self.W3 = Tensor.randn(400, 200) * (1 / 400) ** 0.5
    self.b3 = Tensor.zeros(200)

    self.W4 = Tensor.randn(200, 100) * (1 / 200) ** 0.5
    self.b4 = Tensor.zeros(100)

    self.W5 = Tensor.randn(100, 1) * (1 / 100) ** 0.5
    self.b5 = Tensor.zeros(1)

  def __call__(self, X: Tensor) -> Tensor:
    X = X.flatten(1)
    X = X.matmul(self.W1).add(self.b1).tanh().dropout()
    X = X.matmul(self.W2).add(self.b2).tanh().dropout()
    X = X.matmul(self.W3).add(self.b3).tanh().dropout()
    X = X.matmul(self.W4).add(self.b4).tanh().dropout()
    X = X.matmul(self.W5).add(self.b5)
    return X


model = Model()

print(f"Declaring optimizer")
#params = nn.state.get_parameters(model)
optim = nn.optim.SGD(nn.state.get_parameters(model), lr=0.01)
"""SGD may be leading to overfit"""

# Load size
chunk = 500
batch = 32

n_chunk = 0
total_chunk = (20399 - 10200) // chunk

print(f"Training")

total_loss = 0

print(f"Memory usage pre training: {psutil.Process().memory_info().rss / 1024**3:.4f} GB")
Y = np.array([float(line) for line in open("./data/train.txt")])[:-1]  # 10200 labels

for i in range (10200, 20399, chunk):
  chunk_X = np.load("./data/flow_images.npy", mmap_mode='r')[i:i+chunk].astype(np.float32) / 255.0
  chunk_X = Tensor(chunk_X, dtype='float32')
  chunk_Y = Y[i:i + chunk]
  chunk_Y = Tensor(chunk_Y, dtype='float32')
  print(f"Memory usage outer loop: {psutil.Process().memory_info().rss / 1024**3:.4f} GB")
  Tensor.training=True
  
  # Mini-batch training within chunk
  chunk_loss = 0
  indices = np.random.permutation(chunk_X.shape[0])
  for j in range(0, chunk_X.shape[0], batch):
    print(f"Memory usage inner loop: {psutil.Process().memory_info().rss / 1024**3:.4f} GB")
    batch_idx = indices[j:j + batch].tolist()
    X_batch = chunk_X[batch_idx]
    Y_batch = chunk_Y[batch_idx]

    print("Doing forward")
    # Not needed for now
    # Add l2 loss regularization
    # l2_loss = sum(p.square().sum() for p in params) * 0.01
    loss = model(X_batch).sub(Y_batch).square().mean() # + l2_loss # MSE Loss
    optim.zero_grad()
    loss.backward()
    optim.step()

    chunk_loss += loss.numpy()
    print(f"Chunk {n_chunk + 1}-{total_chunk} % proccessed: {((j / chunk_X.shape[0]) * 100):.4f} | Loss: {loss.numpy():.4f}")

  chunk_loss /= (chunk // batch)
  print("="*40)
  print(f"Mean Chunk loss {chunk_loss:.4f}")
  n_chunk += 1
  print(f"Total data proccessed %: {(n_chunk / total_chunk):.4f}")
  print("="*40)

  total_loss += chunk_loss
  #gc.collect() # Check for garbaje collection
  break

total_loss /= total_chunk
print(f"Mean total loss {total_loss}")

# Save model parameters
model_save_path = "./models/linear_sgd_half.npz"
print(f"Saving model to {model_save_path}")

# Get the model's parameters
param_dict = {}
for i, param in enumerate(nn.state.get_parameters(model)):
    param_dict[f'param_{i}'] = param.numpy()

# Save all parameters to a single .npz file
np.savez(model_save_path, **param_dict)
print(f"Model linear SGD saved successfully to {model_save_path}")
