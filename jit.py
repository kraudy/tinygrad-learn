from tinygrad import Tensor, TinyJit
from tinygrad.ops import UOp
import time

w = Tensor.empty(4,4)

@TinyJit
def forward(x: Tensor):
  c = (x * w).contiguous()
  """dot product without reduce"""
  print(f"shape of c {c.shape}")
  c.sum(0).realize()
  """reduce"""

for i in range(4):
  start = time.time()
  x = Tensor.empty(4,4)
  """y stays static"""
  forward(x)
  end = time.time()
  print(f"Iteration {i} took {(end - start)*1000:.2f}ms")


w = Tensor.empty(4,4)

@TinyJit
def forward(x: Tensor):
  return w.matmul(x).contiguous().sum().realize()

for i in range(4):
  start = time.time()
  dim = UOp.variable("dim", 1, 4).bind(i+1)
  x = Tensor.empty(4, dim)
  """note how y changes"""
  forward(x)
  end = time.time()
  print(f"Iteration {i} took {(end - start)*1000:.2f}ms")


print("="*50)

import tinygrad.nn as nn
from tinygrad.device import Device

shards = [f"{Device.DEFAULT}", f"{Device.DEFAULT}:1"]
weight = Tensor.empty(4, 4).shard_(shards)

@TinyJit
def forward(x: Tensor):
  return x.mul(weight).contiguous().mul(weight)

for i in range(4):
  print(f"\niteration {i}")
  s = time.time()
  x = Tensor.empty(4, 4).shard_(shards).realize()
  ret = forward(x).realize()
  e = time.time()
  print(f"took {(e-s)*1000:.2f}ms")

print("="*50)

w = Tensor.empty(4, 4)

def forward(x: Tensor):
  x2 = Tensor.empty(32, 32).contiguous().realize()
  x = (x + x2[:4, :4]).contiguous()
  return x.mul(w).contiguous().sum(0)

for i in range(4):
  s = time.time()
  x = Tensor.empty(4, 4)
  ret = forward(x).realize()
  e = time.time()
  print(f"took {(e-s)*1000:.2f}ms")

print("="*50)

w = Tensor.empty(4, 4)

@TinyJit
def forward(x: Tensor):
  x2 = Tensor.empty(32, 32).contiguous().realize()
  x = (x + x2[:4, :4]).contiguous()
  return x.mul(w).contiguous().sum(0)

for i in range(4):
  s = time.time()
  x = Tensor.empty(4, 4)
  ret = forward(x).realize()
  e = time.time()
  print(f"took {(e-s)*1000:.2f}ms")

print("="*25, " Lastt ", "="*25)

import math

@TinyJit
def replace_if_zero(tokens: Tensor):
  tokens = (tokens == 0).where(-math.inf, tokens).contiguous()
  _ = (tokens == 0).all()
  return tokens, _ 

ctx = Tensor([
  [0, 0]
])

_a = [
    [1, 7],
    [2, 0],
    [3, 2],
    [4, 0],
]

for i in range(0, len(_a)):
  a = Tensor(_a[i]).reshape((1, -1))
  a, _ = replace_if_zero(a)
  ctx = ctx.cat(a, dim=0)

print(ctx.numpy())