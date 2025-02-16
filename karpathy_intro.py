import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
  return 3*x**2 - 4*x + 5

print(f(3.0))
xs = np.arange(-5, 5, 0.25)
ys = f(xs)

h = 0.000000001

print(((f(xs + h) - f(xs))/h))

def fd(a, b, c):
  return a*b + c

a = 2.0
b = -3.0
c = 10.0
d = fd(a,b,c)
print(d)

h = 0.0001
d1 = fd(a,b,c)
d2 = fd(a+h,b,c)
print("d1: ", d1, " d2: ", d2)
print(f"slope {(d2-d1)/h}")
