from tinygrad import Tensor, nn
import numpy as np
import pdb

cache_path="./data/flow_images.npy"

X = np.load(cache_path)
"""
  ...,
  [ 94, 255,   0],
  [ 94, 255,   0],
  [ 94, 255,   0]]]], shape=(20399, 480, 640, 3), dtype=uint8)
"""

pdb.set_trace()
Y = np.array([float(line) for line in open("./data/train.txt")])[: -1]

