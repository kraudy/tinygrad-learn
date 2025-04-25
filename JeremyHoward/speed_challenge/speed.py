from tinygrad import Tensor, nn
import numpy as np
import pdb

cache_path="./data/flow_images.npy"

"""
X = np.load(cache_path)
  ...,
  [ 94, 255,   0],
  [ 94, 255,   0],
  [ 94, 255,   0]]]], shape=(20399, 480, 640, 3), dtype=uint8)
"""

#pdb.set_trace()

"""
Y = np.array([float(line) for line in open("./data/train.txt")])[: -1]
array([28.105569, 28.105569, 28.106527, ...,  2.289795,  2.292917,
2.2606  ], shape=(20399,))
"""

print(f"Defining network")
W1 = Tensor.randn(480*640*3, 100) * (1 / 480*640*3) ** 0.5
b1 = Tensor.zeros(100)

W2 = Tensor.randn(100, 1) * (1 / 100) ** 0.5
b2 = Tensor.zeros(1)

print(f"Declaring optimizer")
params = [W1, b1, W2, b2]

lr = 0.1
optim = nn.optim.SGD(params, lr)

# Load size
chunk = 1000
batch = 32

# Load data in chunks
def load_data_in_chunks():
    print("Loading data in chunks")
    Y = np.array([float(line) for line in open("./data/train.txt")])[:-1]  # 20399 labels
    X_memmap = np.load("./data/flow_images.npy", mmap_mode='r')  # Memory-mapped array
    num_frames = X_memmap.shape[0]
    
    for i in range(0, num_frames, chunk):
        chunk_X = X_memmap[i:i + chunk].astype(np.float32) / 255.0
        chunk_Y = Y[i:i + chunk]
        yield Tensor(chunk_X, dtype='float32'), Tensor(chunk_Y, dtype='float32')

print(f"Training")


for chunk_X, chunk_Y in load_data_in_chunks():
  Tensor.training=True
  
  # Mini-batch training within chunk
  indices = np.random.permutation(chunk_X.shape[0])
  for i in range(0, chunk_X.shape[0], batch):
    batch_idx = indices[i:i + batch].tolist()
    X_batch = chunk_X[batch_idx]
    Y_batch = chunk_Y[batch_idx]

    print("Doing forward")
    pred = X_batch.flatten(1).matmul(W1).add(b1).tanh().matmul(W2).add(b2)
    loss = pred.sub(Y_batch).square().mean() # MSE Loss
    #loss = loss_fn(pred, Y_batch)
    optim.zero_grad()
    loss.backward()
    optim.step()

    print(f"Loss: {loss.numpy()}")

  break


"""
With RELU

Doing forward
Loss: 13507033.0
Doing forward
Loss: 3.371206309336986e+30
Doing forward
Loss: inf
Doing forward
Loss: nan
Doing forward

==============
Changed to TANH

Loss: 713.3308715820312
Doing forward
Loss: 122493.65625
Doing forward
Loss: 23008048.0
Doing forward
Loss: 5476477952.0
Doing forward
Loss: 1723970486272.0
Doing forward
Loss: 625388577357824.0
Doing forward
Loss: 2.0519178768849306e+17

==============
Normalize weigths

Loading data in chunks
Doing forward
Loss: 410.64361572265625
Doing forward
Loss: 93295.6953125
Doing forward
Loss: 18867530.0
Doing forward
Loss: 5181755904.0

"""