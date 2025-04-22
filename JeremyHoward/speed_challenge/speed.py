from tinygrad import Tensor, nn
import cv2
import pdb
import numpy as np
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



while capture.isOpened():
  ret, frame = capture.read()
  """
  (Pdb) type(frame)
  <class 'numpy.ndarray'>
  (Pdb) frame
  array([[[4, 1, 2],
          [4, 1, 2],
          [4, 1, 2],
          ...,
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]],
        ...,
          , shape=(480, 640, 3), dtype=uint8)
  A color frame is just a 3d tensor
  """
  if not ret: break

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  """
  (Pdb) type(gray)
  <class 'numpy.ndarray'>
  (Pdb) gray
  array([[2, 2, 2, ..., 0, 0, 0],
        [3, 3, 3, ..., 0, 0, 0],
        [5, 5, 5, ..., 0, 0, 0],
        ...,
        [2, 2, 2, ..., 2, 2, 2],
        [2, 2, 2, ..., 2, 2, 2],
        [2, 2, 2, ..., 2, 2, 2]], shape=(480, 640), dtype=uint8)
  Making the frame gray leave us a 2d tensor.
  """
  flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
  """
  (Pdb) type(flow)
  <class 'numpy.ndarray'>
  (Pdb) flow
  array([[[-3.65335745e-06, -7.98011115e-07],
        [-7.87143199e-06, -1.32266496e-06],
        [-1.19714523e-05, -2.28082763e-06],
        ...,
        [-2.66591286e-07, -6.40445194e-07],
        [ 1.67395449e-08, -3.06254208e-07],
        [ 1.50439249e-07, -7.09934938e-08]],
       ...,,
      shape=(480, 640, 2), dtype=float32)
  Are these like coordinates?
  """
  #pdb.set_trace()
  # Convert to HSV for training
  mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
  """
  (Pdb) mag
  array([[3.7394977e-06, 7.9817846e-06, 1.2186790e-05, ..., 6.9371532e-07,
          3.0671134e-07, 1.6634915e-07],
        [8.7670687e-06, 1.7663959e-05, 2.6312775e-05, ..., 2.8885609e-06,
          2.3231823e-06, 2.6260589e-06],
        [1.2164734e-05, 2.1700091e-05, 3.0020656e-05, ..., 7.5320550e-06,
          7.2712442e-06, 9.0219219e-06],
        ...,
        , shape=(480, 640), dtype=float32)
  (Pdb) ang
  array([[3.3566594, 3.3080654, 3.32986  , ..., 4.3178687, 4.766983 ,
        5.8422074],
       [3.3233955, 3.2977467, 3.3217387, ..., 4.8974357, 5.5845795,
        6.170016 ],
       [3.4098787, 3.427463 , 3.4823477, ..., 5.4508605, 5.8530583,
        6.22857  ],
       ..., shape=(480, 640), dtype=float32)
  """
  hsv = np.zeros_like(prev)
  """
  (Pdb) hsv
  array([[[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          ..., 
          shape=(480, 640, 3), dtype=uint8)
  """
  hsv[..., 1] = 255
  """
  array([[[  0, 255,   0],
          [  0, 255,   0],
          [  0, 255,   0],
        ..., 
        shape=(480, 640, 3), dtype=uint8)
  """
  hsv[..., 0] = ang * 180 / np.pi / 2
  """
  (Pdb) hsv
  array([[[ 96, 255,   0],
          [ 94, 255,   0],
          [ 95, 255,   0],
        ..., 
        shape=(480, 640, 3), dtype=uint8)
  """
  hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

capture.release()
