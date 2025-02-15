from tinygrad import Tensor, nn

class Model:
  def __init__(self):
    self.l1 = nn.Linear(1, 16)  
    self.l1 = nn.Linear(1, 16) 


lin = nn.Linear(2, 16)
t = Tensor.rand(10, 2)
print(t.numpy()) 
print(t.shape) 
"""
[[0.28873467 0.88722086]
 [0.08031988 0.06844091]
 [0.5734875  0.97716963]
 [0.66889775 0.49414122]
 [0.05416346 0.9082408 ]
 [0.9855571  0.59074223]
 [0.8551377  0.6917448 ]
 [0.8651501  0.14550364]
 [0.10968459 0.12747383]
 [0.36962068 0.14609933]]
"""

t = lin(t).relu()
print(t.numpy())
print(t.shape)
"""
shape: (10, 16)
"""
t = t.sum(0)
print(t.numpy())
print(t.shape)
"""
[ 0.          0.          0.          6.6112766   0.19170047  0.43650466
  0.         11.394417    6.107524    0.          0.          5.8423243
  0.          0.          0.          3.224327  ]

shape: (16,)
"""

print("="*25, ' relu2 ', "="*25)

lin2 = nn.Linear(16, 1)
t = lin2(t).relu()
print(t.numpy())
print(t.shape)
"""
[0.73632187]
(1,)
"""

t = t.sum(0)
print(t.numpy())
print(t.shape)
"""
0.73632187
()
"""

