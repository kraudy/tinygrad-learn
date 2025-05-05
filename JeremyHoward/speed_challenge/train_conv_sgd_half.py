from tinygrad import Tensor, nn
import numpy as np
import pdb
import psutil

cache_path="./data/flow_images.npy"

"""
X = shape=(20399, 480, 640, 3)
Y = shape=(20399,)
"""

# Load size
chunk = 500
batch = 32

n_chunk = 0
# 10200
#total_chunk = 20399 // chunk
total_chunk = 10200 // chunk

print(f"Defining network")
class Model():
  def __init__(self):

    # so this expects 3 channels
    self.W1 = Tensor.randn(16, 3, 3, 3) * (2.0 / (3 * 3 * 3)) ** 0.5
    self.b1 = Tensor.zeros(1, 16, 1, 1)

    self.W2 = Tensor.randn(32, 16, 3, 3) * (2.0 / (16 * 3 * 3)) ** 0.5
    self.b2 = Tensor.zeros(1, 32, 1, 1)

    # Fully connected layers
    # Flatten: 16 * 480 * 640 = 19,660,800
    # FC1: 19,660,800 -> 128

    self.W_fc1 = Tensor.randn(32 * 480 * 640, 128) * (2.0 / (32 * 480 * 640)) ** 0.5
    self.b_fc1 = Tensor.zeros(128)

    # FC2: 128 -> 1 (single output for regression)
    self.W_fc2 = Tensor.randn(128, 1) * (2.0 / 128) ** 0.5
    self.b_fc2 = Tensor.zeros(1)


  def __call__(self, X: Tensor) -> Tensor:
    # X shape: (batch, 480, 640, 3)
    # Transpose to (batch, 3, 480, 640) for conv2d
    X = X.permute(0, 3, 1, 2)  # Now (batch, channels, height, width)
    # padding help us keep shape (1, 16, 480, 640)
    X = X.conv2d(self.W1, self.b1, padding=1).relu().max_pool2d((2,2))
    X = X.conv2d(self.W2, self.b2, padding=1).relu().max_pool2d((2,2))
    X = X.flatten(1).dropout() # Flatten to (batch, 16 * 480 * 640)
    X = X.matmul(self.W_fc1).add(self.b_fc1).tanh()
    X = X.matmul(self.W_fc2).add(self.b_fc2)
    return X


model = Model()

print(f"Declaring optimizer")
params = nn.state.get_parameters(model)
optim = nn.optim.SGD(nn.state.get_parameters(params), lr=0.001)


print(f"Training")

total_loss = 0

print(f"Memory usage pre training: {psutil.Process().memory_info().rss / 1024**3:.4f} GB")
Y = np.array([float(line) for line in open("./data/train.txt")])[:-1]  # 10200 labels

for i in range (0, 10200, chunk):
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
    # l2_loss = sum(p.square().sum() for p in params) * 0.0001
    loss = model(X_batch).sub(Y_batch).square().mean() #.add(l2_loss) # MSE Loss
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
model_save_path = "./models/conv_sgd_half.npz"
print(f"Saving model to {model_save_path}")

# Get the model's parameters
param_dict = {}
for i, param in enumerate(nn.state.get_parameters(model)):
    param_dict[f'param_{i}'] = param.numpy()

# Save all parameters to a single .npz file
np.savez(model_save_path, **param_dict)
print(f"Model conv SGD half saved successfully to {model_save_path}")


"""
Some conv2d ops

>>> print(Tensor.randn(1,480,640,3).permute(0,3,1,2).conv2d(Tensor.randn(16,3,3,3), Tensor.zeros(1,16,1,1), padding=1).shape)
(1, 16, 480, 640)
>>> print(Tensor.randn(1,480,640,3).permute(0,3,1,2).conv2d(Tensor.randn(32,3,3,3), Tensor.zeros(1,32,1,1), padding=1).shape)
(1, 32, 480, 640)

>>> print(Tensor.randn(1,480,640,3).permute(0,3,1,2).conv2d(Tensor.randn(64,3,3,3), Tensor.zeros(1,64,1,1), padding=1).shape)
(1, 64, 480, 640)
>>> print(Tensor.randn(10,480,640,3).permute(0,3,1,2).conv2d(Tensor.randn(64,3,3,3), Tensor.zeros(1,64,1,1), padding=1).shape)
(10, 64, 480, 640) 

Error, channels are not the same
>>> print(Tensor.randn(10,480,640,3).permute(0,3,1,2).conv2d(Tensor.randn(64,7,3,3), Tensor.zeros(1,64,1,1), padding=1).shape)
AssertionError: Input Tensor shape (10, 3, 480, 640) does not match the shape of the weights (64, 7, 3, 3). (7 vs. 3)

"""
