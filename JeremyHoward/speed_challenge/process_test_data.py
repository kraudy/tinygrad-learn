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

  if not ret: break

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

  #pdb.set_trace()
  # Convert to HSV for training
  mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

  hsv = np.zeros_like(prev)

  hsv[..., 1] = 255

  hsv[..., 0] = ang * 180 / np.pi / 2

  hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

  hsv = hsv[:crop_h, :crop_w]
  # Consider normalizing this thing and set dtype to float32 to reduce processing in training
  test_flow_images[i] = hsv

  if (i + 1) % 100 == 0: print(f"Step {i}")

  prev_gray = gray

cache_path="./data/test_flow_images.npy"
print(f"Saving test flow images to {cache_path}")
np.save(cache_path, test_flow_images)

capture.release()
