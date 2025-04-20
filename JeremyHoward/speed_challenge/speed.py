from tinygrad import Tensor, nn
import cv2
"""This is intalled with opencv-python"""

cap = cv2.VideoCapture('./data/train.mp4')
ret, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
flow_images = []
