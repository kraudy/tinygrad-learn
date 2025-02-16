from tinygrad import Tensor
import math

from sklearn.datasets import make_moons
x, y = make_moons(n_samples=100, noise=0.1)
y = y*2 - 1 # make y be -1 or 1

