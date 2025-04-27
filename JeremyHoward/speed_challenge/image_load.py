from PIL import Image
import numpy as np

"""
This works with images, not videos,
could be used for MNIST
"""

def image_load(path):
  na = np.array(Image.open(path))