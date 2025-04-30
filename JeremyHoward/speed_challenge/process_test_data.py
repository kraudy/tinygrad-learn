import cv2
import pdb
import numpy as np
"""This is intalled with opencv-python"""

capture = cv2.VideoCapture('./data/test.mp4')
"""< cv2.VideoCapture 0x7fb6672551b0>"""

ret, prev = capture.read()
"""
(Pdb) ret
True
(Pdb) prev
array([[[6, 1, 0],
        [6, 1, 0],
        [6, 1, 0],
        ...,
        [4, 1, 2],
        [4, 1, 2],
        [4, 1, 2]]], shape=(480, 640, 3), dtype=uint8)
"""
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
"""
(Pdb) prev_gray
array([[1, 1, 1, ..., 0, 0, 0],
       [2, 2, 2, ..., 0, 0, 0],
       [2, 2, 2, ..., 0, 0, 0],
       ...,
       [1, 1, 1, ..., 2, 2, 2],
       [1, 1, 1, ..., 2, 2, 2],
       [1, 1, 1, ..., 2, 2, 2]], shape=(480, 640), dtype=uint8)
"""

crop_h=480
crop_w=640

frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1  # xxx flow images
"""10797"""
test_flow_images = np.zeros((frame_count, crop_h, crop_w, 3), dtype=np.uint8)
"""
(Pdb) test_flow_images
array([[[
         ...,
         [0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]]], shape=(10797, 480, 640, 3), dtype=uint8)
"""

pdb.set_trace()

for i in range(frame_count):
  ret, frame = capture.read()
  """
  (Pdb) ret 
  True
  (Pdb) frame
  array([[[6, 1, 0],
          [6, 1, 0],
          [6, 1, 0],
          ...,
          [3, 0, 1],
          [3, 0, 1],
          [3, 0, 1]]], shape=(480, 640, 3), dtype=uint8)
  """

  if not ret: break

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  """
  (Pdb) gray
  array([[1, 1, 1, ..., 0, 0, 0],
        [2, 2, 2, ..., 0, 0, 0],
        [2, 2, 2, ..., 0, 0, 0],
        ...,
        [1, 1, 1, ..., 1, 1, 1],
        [1, 1, 1, ..., 1, 1, 1],
        [1, 1, 1, ..., 1, 1, 1]], shape=(480, 640), dtype=uint8)
  """

  flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
  """
  (Pdb) flow
  array([
        [[-4.2749593e-10,  6.7343692e-10],
          [-1.0103485e-09,  1.2197339e-09],
          [-3.1918730e-09,  2.0528939e-09],
          ...,
          [ 2.6790312e-09,  1.1552592e-10],
          [ 2.0028936e-09, -6.4602775e-11],
          [ 1.7798606e-09, -6.8777088e-11]]],
        shape=(480, 640, 2), dtype=float32)
  """
  #pdb.set_trace()
  # Convert to HSV for training
  mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
  """
  (Pdb) mag
  array([
       ...,
       [7.7528839e-10, 1.7805237e-09, 4.8226543e-09, ..., 3.0168870e-09, 2.2580127e-09, 2.0090893e-09],
       [6.5920880e-10, 1.4281123e-09, 3.7926458e-09, ..., 2.7014591e-09, 2.0209461e-09, 1.7991437e-09],
       [7.9766538e-10, 1.5838417e-09, 3.7950532e-09, ..., 2.6815210e-09, 2.0039352e-09, 1.7811890e-09]], 
       shape=(480, 640), dtype=float32)
  (Pdb) ang
  array([
       ...,
       [2.9602559 , 2.9970813 , 3.0511973 , ..., 0.04179059, 6.251979  , 6.245869  ],
       [2.504745  , 2.6209    , 2.8205998 , ..., 0.04308855, 6.2509413 , 6.2445507 ],
       [2.136312  , 2.2625918 , 2.5701435 , ..., 0.04308702, 6.2509484 , 6.2445703 ]],
        shape=(480, 640), dtype=float32)
  """
  hsv = np.zeros_like(prev)
  """
  (Pdb) hsv
  array([
       [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]], shape=(480, 640, 3), dtype=uint8)
  """

  hsv[..., 1] = 255
  """
  (Pdb) hsv[:,1]
  array([...,
        [  0, 255,   0],
        [  0, 255,   0],
        [  0, 255,   0]], shape=(480, 3), dtype=uint8)
  """

  hsv[..., 0] = ang * 180 / np.pi / 2
  """
  (Pdb) hsv[:,0]
  array([...,
        [ 84, 255,   0],
        [ 71, 255,   0],
        [ 61, 255,   0]], shape=(480, 3), dtype=uint8)
  """

  hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
  """
  (Pdb) hsv[:,0]
  array([...,
        [ 84, 255,   0],
        [ 71, 255,   0],
        [ 61, 255,   0]], shape=(480, 3), dtype=uint8)
  """

  hsv = hsv[:crop_h, :crop_w]
  """
  (Pdb) hsv
  array([...,
          [  1, 255,   0],
          [179, 255,   0],
          [178, 255,   0]]], shape=(480, 640, 3), dtype=uint8)
  """

  test_flow_images[i] = hsv

  if (i + 1) % 100 == 0: print(f"Step {i}")

  prev_gray = gray

cache_path="./data/test_flow_images.npy"
print(f"Saving test flow images to {cache_path}")
np.save(cache_path, test_flow_images)

capture.release()
