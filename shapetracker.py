from tinygrad.shape.view import View
from tinygrad.shape.view import unravel, sint_to_uop
from tinygrad.ops import UOp, Ops
from tinygrad import dtypes

"""Is called view because it represents how the 2x2 continuos memory data
is going to be 'view' by the stride """
a = View.create(shape=(2,2), strides=(2,1))
"""Note how a view is not a tensor"""

idx, valid = a.to_indexed_uops()
"""
idx: acces strategy, AST tree
valid: mask.

UOp(Ops.ADD, dtypes.int, arg=None, src=(
  UOp(Ops.ADD, dtypes.int, arg=None, src=( # (ridx0*2)
    x0:=UOp(Ops.CONST, dtypes.int, arg=0, src=()), # x0 = const 0
    UOp(Ops.MUL, dtypes.int, arg=None, src=( # ridx0
      UOp(Ops.RANGE, dtypes.int, arg=-1, src=( # Note -1
         x0,
        x3:=UOp(Ops.CONST, dtypes.int, arg=2, src=()),)), # to 2
       x3,)),)),  # 2
  UOp(Ops.MUL, dtypes.int, arg=None, src=(  # ridx1
    UOp(Ops.RANGE, dtypes.int, arg=0, src=(
       x0,    # from 
       x3,)),
    UOp(Ops.CONST, dtypes.int, arg=0, src=()),)),))
"""


print(idx)

print("kernel: ", idx.render())
"""Show kernel
row * 2 + col
This is just for debug. The UOp AST is actually used render when doing the kernel computation

((ridx0*2)+ridx1)
"""

a = View.create(shape=(3, 2), strides=(2, 1))

print(a.shape) # (3, 2)
print(a.strides) # (2 ,1)
"""Original shapes."""

#transpose
a = a.permute((1,0))
print(a.shape) # (2, 3)
print(a.strides) # (1, 2)
"""Permuted, just swap values"""

a = a.reshape((3, 2))
print(a)
"""Oho. None. This is because this can not be accomplish with the same view"""

from tinygrad.shape.shapetracker import ShapeTracker

a = ShapeTracker.from_shape((3,2))
print(a.shape)
a = a.permute((1,0))
print(a.shape)
a = a.reshape((3,2))
print(a)
"""
Note, now we got 2 views with its shape and offset
ShapeTracker(views=(
View(shape=(3, 2), strides=(2, 1), offset=0, mask=None, contiguous=True))
View(shape=(2, 3), strides=(1, 2), offset=0, mask=None, contiguous=False), 
)"""

idx, valid = a.to_indexed_uops()
print(idx)
"""
UOp(Ops.ADD, dtypes.int, arg=None, src=(
  UOp(Ops.MUL, dtypes.int, arg=None, src=(
    UOp(Ops.MOD, dtypes.int, arg=None, src=(
      x2:=UOp(Ops.ADD, dtypes.int, arg=None, src=(
        UOp(Ops.MUL, dtypes.int, arg=None, src=(
          UOp(Ops.RANGE, dtypes.int, arg=0, src=( # maybe this initializes in 0?
            x5:=UOp(Ops.CONST, dtypes.int, arg=0, src=()), #dim 0, from 0 to 2 (3)
            x6:=UOp(Ops.CONST, dtypes.int, arg=3, src=()),)),
          x7:=UOp(Ops.CONST, dtypes.int, arg=2, src=()),)), # dim 1 = dim0 * 2
        UOp(Ops.RANGE, dtypes.int, arg=1, src=(
           x5, # from 0 to 1 (2)
           x7,)),)),
       x6,)),
     x7,)),
  UOp(Ops.IDIV, dtypes.int, arg=None, src=(
     x2,
     x6,)),))
"""
print("kernel: ", idx.render())
"""
(((((ridx0*2)+ridx1)%3)*2)+(((ridx0*2)+ridx1)//3))
"""

from typing import Optional

views = [View.create(shape=(3, 2), strides=(2, 1)), View.create(shape=(2,3), strides=(1,2))]

print("="*50)
for v in reversed(views[0: -1]):
    idx, valid = v.to_indexed_uops()
    print("idx ", idx)
    print("kernel ", idx.render())
    v = v.minify()
    print("="*30)
    idx, valid = v.to_indexed_uops()
    print("minify idx ", idx)
    print("minify kernel ", idx.render())
    #print("="*20)
    #for i in unravel(v.shape, idx): print(i)
    idx, valid = v.to_indexed_uops([sint_to_uop(i) for i in unravel(v.shape, idx)], valid)
    print("="*20)
    print("after op idx ", idx)
    print("after op kernel ", idx.render())
