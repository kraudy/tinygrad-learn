from tinygrad.shape.view import View
from tinygrad.shape.view import unravel, sint_to_uop
from tinygrad.ops import UOp, Ops
from tinygrad import dtypes

"""Is called view because it represents how the 2x2 continuos memory data
is going to be 'view' by the stride """
a = View.create(shape=(2,2), strides=(2,1))
"""Note how a view is not a tensor
"""

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
"""Oho. None. This is because this can not be accomplish with the same view.
Transposing and reshapeing a view requires another view because another shape and stride (or even only a stride)
is needed, that's where shapetracker gets in.
What we want is to proyect the indexes of the first view to the last view.
"""

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
    """
    UOp(Ops.ADD, dtypes.int, arg=None, src=(
    UOp(Ops.ADD, dtypes.int, arg=None, src=(
      x1:=UOp(Ops.CONST, dtypes.int, arg=0, src=()),
      UOp(Ops.MUL, dtypes.int, arg=None, src=(
        UOp(Ops.RANGE, dtypes.int, arg=0, src=(
          x1, #dim0 3 elem
          UOp(Ops.CONST, dtypes.int, arg=3, src=()),)),
        x5:=UOp(Ops.CONST, dtypes.int, arg=2, src=()),)),)), #stride 2
    UOp(Ops.MUL, dtypes.int, arg=None, src=(
      UOp(Ops.RANGE, dtypes.int, arg=1, src=(
        x1,   #dim1 2 elem
        x5,)),
      UOp(Ops.CONST, dtypes.int, arg=1, src=()),)),)) #stride 1
    kernel  ((ridx0*2)+ridx1) # *1 omited
    """
    v = v.minify()
    print("="*30)
    idx, valid = v.to_indexed_uops()
    """
    #This collapeses shape (3,2) to (6) and stride to (1)
    UOp(Ops.ADD, dtypes.int, arg=None, src=(
    x0:=UOp(Ops.CONST, dtypes.int, arg=0, src=()),
    UOp(Ops.MUL, dtypes.int, arg=None, src=(
      UOp(Ops.RANGE, dtypes.int, arg=0, src=(
        x0,
        UOp(Ops.CONST, dtypes.int, arg=6, src=()),)), # dim0
      UOp(Ops.CONST, dtypes.int, arg=1, src=()),)),)) # strides
    minify kernel  ridx0 # note how simple the kernel for traversing becomes for dim0
    """
    idx, valid = v.to_indexed_uops([sint_to_uop(i) for i in unravel(v.shape, idx)], valid)
    print("="*20)
    print("after op idx ", idx)
    print("after op kernel ", idx.render())
    """
    UOp(Ops.ADD, dtypes.int, arg=None, src=(
    x0:=UOp(Ops.CONST, dtypes.int, arg=0, src=()),
      UOp(Ops.MUL, dtypes.int, arg=None, src=( #dim0 * 6 ? review a x8 is missing
        UOp(Ops.MOD, dtypes.int, arg=None, src=( #dim0 mod 1
          UOp(Ops.IDIV, dtypes.int, arg=None, src=( #dim0 / 6
            UOp(Ops.ADD, dtypes.int, arg=None, src=( #0 + dim0
              x0, # 0 + dim0
              UOp(Ops.MUL, dtypes.int, arg=None, src=( #dim0 * 1
                UOp(Ops.RANGE, dtypes.int, arg=0, src=(
                    x0,
                  x7:=UOp(Ops.CONST, dtypes.int, arg=6, src=()),)),   #dim0 6 elem
                x8:=UOp(Ops.CONST, dtypes.int, arg=1, src=()),)),)),  #strides 1
            x8,)),
          x7,)),
        x8,)),))
    after op kernel  ridx0 # Note how the AST is reduced when doing makeing the kernel, like a compiler
    """ 

views = [View.create(shape=(2,3), strides=(1,2)), View.create(shape=(3, 2), strides=(2, 1))]

print("="*30)
idx, valid = views[-1].to_indexed_uops()
for view in reversed(views[0: -1]):
    view = view.minify()
    """We flatten the last view. We are going backward like the reversed topo."""
    acc, idxs = 1, []
    for d in reversed(view.shape):
        idxs.append((idx//acc)%d)
        acc *= d
    print("beofre op idx ", idx)
    print("beofre op kernel ", idx.render())
    idx, valid = view.to_indexed_uops(idxs[::1], valid)
    print("after op idx ", idx)
    print("after op kernel ", idx.render())
    print("="*20)

"""
after op idx  UOp(Ops.ADD, dtypes.int, arg=None, src=(
  x0:=UOp(Ops.CONST, dtypes.int, arg=0, src=()),
  UOp(Ops.MUL, dtypes.int, arg=None, src=(
    UOp(Ops.MOD, dtypes.int, arg=None, src=(
      UOp(Ops.IDIV, dtypes.int, arg=None, src=(
        UOp(Ops.ADD, dtypes.int, arg=None, src=(
           x0,
          UOp(Ops.MUL, dtypes.int, arg=None, src=(
            UOp(Ops.RANGE, dtypes.int, arg=0, src=(
               x0,
              x7:=UOp(Ops.CONST, dtypes.int, arg=6, src=()),)),
            x8:=UOp(Ops.CONST, dtypes.int, arg=1, src=()),)),)),
         x8,)),
       x7,)),
     x8,)),))
after op kernel  ridx0
==============================
# First view before change
beofre op idx  UOp(Ops.ADD, dtypes.int, arg=None, src=(
  UOp(Ops.ADD, dtypes.int, arg=None, src=(
    x1:=UOp(Ops.CONST, dtypes.int, arg=0, src=()),
    UOp(Ops.MUL, dtypes.int, arg=None, src=(
      UOp(Ops.RANGE, dtypes.int, arg=0, src=(
         x1,
        UOp(Ops.CONST, dtypes.int, arg=3, src=()),)),
      x5:=UOp(Ops.CONST, dtypes.int, arg=2, src=()),)),)),
  UOp(Ops.MUL, dtypes.int, arg=None, src=(
    UOp(Ops.RANGE, dtypes.int, arg=1, src=(
       x1,
       x5,)),
    UOp(Ops.CONST, dtypes.int, arg=1, src=()),)),))

beofre op kernel  ((ridx0*2)+ridx1)

# First view changed
after op idx  UOp(Ops.ADD, dtypes.int, arg=None, src=(
  UOp(Ops.ADD, dtypes.int, arg=None, src=(
    x1:=UOp(Ops.CONST, dtypes.int, arg=0, src=()),
    UOp(Ops.MUL, dtypes.int, arg=None, src=(
      UOp(Ops.MOD, dtypes.int, arg=None, src=(
        UOp(Ops.IDIV, dtypes.int, arg=None, src=(
          x5:=UOp(Ops.ADD, dtypes.int, arg=None, src=(
            UOp(Ops.ADD, dtypes.int, arg=None, src=(
               x1,
              UOp(Ops.MUL, dtypes.int, arg=None, src=(
                UOp(Ops.RANGE, dtypes.int, arg=0, src=(
                   x1,
                  x9:=UOp(Ops.CONST, dtypes.int, arg=3, src=()),)),
                x10:=UOp(Ops.CONST, dtypes.int, arg=2, src=()),)),)),
            UOp(Ops.MUL, dtypes.int, arg=None, src=(
              UOp(Ops.RANGE, dtypes.int, arg=1, src=(
                 x1,
                 x10,)),
              x13:=UOp(Ops.CONST, dtypes.int, arg=1, src=()),)),)),
           x13,)),
         x9,)),
       x13,)),)),
  UOp(Ops.MUL, dtypes.int, arg=None, src=(
    UOp(Ops.MOD, dtypes.int, arg=None, src=(
      UOp(Ops.IDIV, dtypes.int, arg=None, src=(
         x5,
         x9,)),
       x10,)),
     x10,)),))

after op kernel  (((((ridx0*2)+ridx1)//3)*2)+(((ridx0*2)+ridx1)%3))
"""

