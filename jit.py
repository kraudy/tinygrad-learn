from tinygrad import Tensor, TinyJit
import time

w = Tensor.empty(4,4)

@TinyJit
def forward(x: Tensor):
  c = (x * w).contiguous()
  """dot product without reduce"""
  c.sum(0).realize()
  """reduce"""

for i in range(4):
  start = time.time()
  x = Tensor.empty(4,4)
  forward(x)
  end = time.time()
  print(f"Iteration {i} took {(end - start)*1000:.2f}ms")




