from tinygrad import Tensor, nn
import cv2
import pdb
"""This is intalled with opencv-python"""


capture = cv2.VideoCapture('./data/train.mp4')
"""
Get video capture
< cv2.VideoCapture 0x7fd09bdd3cb0>
"""

ret, prev = capture.read()
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


#pdb.set_trace()

while capture.isOpened():
  ret, frame = capture.read()
  if not ret: break

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

  # Convert to HSV for training


capture.release()
