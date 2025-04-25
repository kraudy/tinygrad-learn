from tinygrad import Tensor, nn
import numpy as np
import pdb

cache_path="./data/flow_images.npy"
 
print(f"Loading X")
X = np.load(cache_path)
"""
  ...,
  [ 94, 255,   0],
  [ 94, 255,   0],
  [ 94, 255,   0]]]], shape=(20399, 480, 640, 3), dtype=uint8)
"""

#pdb.set_trace()

print("Loading Y")
Y = np.array([float(line) for line in open("./data/train.txt")])[: -1]
"""
array([28.105569, 28.105569, 28.106527, ...,  2.289795,  2.292917,
2.2606  ], shape=(20399,))
"""

# Normalizing
X = X / 255

X = Tensor(X, dtype='float32')
Y = Tensor(Y, dtype='float32')