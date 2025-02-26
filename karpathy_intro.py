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
#   dl/dc = 2 * 1  
c | d1:  -8  d2:  -8.0002
slope -1.9999999999953388
"""

d2 = lol(a=2, b=-3, c=10, e=0 + h, d=0, f=-2, L=0)
print("e| d1: ", d1, " d2: ", d2)
print(f"slope {(d2-d1)/h}")
"""
#   dl/de = dl/dd * dd/de | Chain rule
#   dl/de = -2 * 1  
e| d1:  -8  d2:  -8.0002
slope -1.9999999999953388
"""


d2 = lol(a=2 + h, b=-3, c=10, e=0, d=0, f=-2, L=0)
print("a | d1: ", d1, " d2: ", d2)
print(f"slope {(d2-d1)/h}")
"""
#   dl/da = dl/dd * dd/de * de/da | Chain rule
#   dl/da = -2 * 1 * -3
a | d1:  -8  d2:  -7.999399999999998
slope 6.000000000021544
"""

d2 = lol(a=2, b=-3 + h, c=10, e=0, d=0, f=-2, L=0)
print("b | d1: ", d1, " d2: ", d2)
print(f"slope {(d2-d1)/h}")
"""
#   dl/db = dl/dd * dd/de * de/db | Chain rule
#   dl/db = -2 * 1 * 2
b | d1:  -8  d2:  -8.0004
slope -4.000000000008441
"""


import torch

x1 = torch.Tensor([2.0]).double()   ; x1.requires_grad=True
x2 = torch.Tensor([0.0]).double()   ; x2.requires_grad=True
w1 = torch.Tensor([-3.0]).double()  ; w1.requires_grad=True
w2 = torch.Tensor([1.0]).double()   ; w2.requires_grad=True
b = torch.Tensor([6.8813735870195432]).double()
n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

print(o.data.item())
o.backward()

print("="*50)
print(f"x2: {x2.grad.item()}")
print(f"w2: {w2.grad.item()}")
print(f"x1: {x1.grad.item()}")
print(f"w1: {w1.grad.item()}")


print("="*25," MLP ", "="*25)

from micrograd.nn import MLP

n = MLP(3, [4, 4, 1])
"""
3 Features input,
Layer 1: w=3, n = 4
Layer 2: w=4, n = 4
Layer 3: w=4, n = 1"""

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]

for k in range(50):
  output= [n(x) for x in xs]
  #print([o.data for o in output])

  loss = sum([(output[i] - ys[i])**2 for i in range(len(output))])

  for p in n.parameters():
    p.grad = 0.0

  loss.backward()
  print(loss)
  
  for p in n.parameters():
      p.data -= 0.001 * p.grad
                        


