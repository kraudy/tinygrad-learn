import math
import numpy as np
import matplotlib.pyplot as plt

def f1(x):
  return 3*x**2 - 4*x + 5

print(f1(3.0))
xs = np.arange(-5, 5, 0.25)
ys = f1(xs)

h = 0.000000001

print(((f1(xs + h) - f1(xs))/h))

#==============================

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
"""
Increment a
d1:  4.0  d2:  3.999699999999999
slope -3.000000000010772 # a affects f(x) by -3 times h
"""

d2 = fd(a,b+h,c)
print("d1: ", d1, " d2: ", d2)
print(f"slope {(d2-d1)/h}")
"""
Increment b
d1:  4.0  d2:  4.0002
slope 2.0000000000042206 # b affects f(x) by 2 times h
"""

d2 = fd(a,b,c+h)
print("d1: ", d1, " d2: ", d2)
print(f"slope {(d2-d1)/h}")
"""
Increment c
d1:  4.0  d2:  4.0001
slope 0.9999999999976694 # c affects (fx) by 0.9 times h or by 1 times c
"""

print("="*25, " Manual grad ", "="*25)

def lol(a, b, c, e, d, f, L):
  e += a * b
  d += e + c
  L += d * f
  return L

d1 = lol(a=2, b=-3, c=10, e=0, d=0, f=-2, L=0)

d2 = lol(a=2, b=-3, c=10, e=0, d=0, f=-2, L=0) + h
print("L | d1: ", d1, " d2: ", d2)
print(f"slope {(d2-d1)/h}")
"""
Increment L
d1:  -16  d2:  -15.9999
slope 0.9999999999976694 # Increments by 1 h
"""

d2 = lol(a=2, b=-3, c=10, e=0, d=0, f=-2 + h, L=0)
print("f | d1: ", d1, " d2: ", d2)
print(f"slope {(d2-d1)/h}")
"""
f | d1:  -8  d2:  -7.9996
slope 3.9999999999995595
"""

d2 = lol(a=2, b=-3, c=10, e=0, d=0 + h, f=-2, L=0)
print("d | d1: ", d1, " d2: ", d2)
print(f"slope {(d2-d1)/h}")
"""
d | d1:  -8  d2:  -8.0002
slope -1.9999999999953388
"""

d2 = lol(a=2, b=-3, c=10 + h, e=0, d=0, f=-2, L=0)
print("c | d1: ", d1, " d2: ", d2)
print(f"slope {(d2-d1)/h}")
"""
#   dl/dc = dl/dd * dd/dc | Chain rule
c | d1:  -8  d2:  -8.0002
slope -1.9999999999953388
"""


d2 = lol(a=2 + h, b=-3, c=10, e=0, d=0, f=-2, L=0)
print("d1: ", d1, " d2: ", d2)
print(f"slope {(d2-d1)/h}")
"""
d1:  -16  d2:  -16.000300000000003
slope -3.0000000000285354
"""