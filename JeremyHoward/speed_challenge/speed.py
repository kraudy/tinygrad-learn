from tinygrad import Tensor, nn
import cv2
import pdb
"""This is intalled with opencv-python"""

pdb.set_trace()

cap = cv2.VideoCapture('./data/train.mp4')
"""< cv2.VideoCapture 0x7fd09bdd3cb0>"""
ret, prev = cap.read()
"""
(Pdb) ret
True

(Pdb) prev
array([[[3, 0, 1],
        [3, 0, 1],
        [3, 0, 1],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],

       [[3, 0, 1],
        [3, 0, 1],
        [3, 0, 1],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],

       [[5, 2, 3],
        [5, 2, 3],
        [5, 2, 3],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],

       ...,
, shape=(480, 640, 3), dtype=uint8)
"""

prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
"""
(Pdb) prev_gray
array([[1, 1, 1, ..., 0, 0, 0],
       [1, 1, 1, ..., 0, 0, 0],
       [3, 3, 3, ..., 0, 0, 0],
       ...,
       [2, 2, 2, ..., 3, 3, 3],
       [2, 2, 2, ..., 3, 3, 3],
       [2, 2, 2, ..., 3, 3, 3]], shape=(480, 640), dtype=uint8)
"""
flow_images = []
