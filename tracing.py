from tinygrad import Tensor, nn


X = Tensor.randn(3)

"""

tensory.py
  _getitem
  (1141)<genexpr>
  (3900)_wrapper()
  (1012)shrink()
  (1029)<listcomp>()
  (227)shape()

ops.py
  (315)shape()
  (289)st()

helpers.py
  (60)unwrap()

shapetracker.py
  (84)shape()
  @property


(Pdb) self.lazydata
UOp(Ops.VIEW, dtypes.uint, arg=ShapeTracker(views=(View(shape=(2,), strides=(1,), offset=0, mask=None, contiguous=True),)), src=(
  UOp(Ops.COPY, dtypes.uint, arg=False, src=(
    UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),
    UOp(Ops.BUFFER, dtypes.uint, arg=(0, 2), src=(
      UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),)),))

(Pdb) self.op
<Ops.SHRINK: 18>
         
(Pdb) self
ShapeTracker(views=(View(shape=(2,), strides=(1,), offset=0, mask=None, contiguous=True),))
Pdb) self.views
(View(shape=(2,), strides=(1,), offset=0, mask=None, contiguous=True),) 

(Pdb) type(self)
<class 'tinygrad.tensor.Tensor'>
(Pdb) self
<Tensor <UOp CLANG (2,) uint ShapeTracker(views=(View(shape=(2,), strides=(1,), offset=0, mask=None, contiguous=True),))> on CLANG with grad None>
(Pdb) type(self.lazydata)
<class 'tinygrad.ops.UOp'>
(Pdb) type(self.lazydata.shape)
<class 'tuple'>
(Pdb) type(self.st)
<class 'tinygrad.shape.shapetracker.ShapeTracker'>
(Pdb) type(self.views)
<class 'tuple'>


UOp(Ops.CONST, dtypes.uint, arg=1, src=(
  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
    UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),)),))

> /home/kraudy/tinygrad/tinygrad/tinygrad/tensor.py(138)__init__()
-> if dtype is not None: dtype = to_dtype(dtype)
(Pdb) self
*** AttributeError: 'Tensor' object has no attribute 'lazydata'
(Pdb) data
UOp(Ops.RESHAPE, dtypes.uint, arg=(1,), src=(
  UOp(Ops.CONST, dtypes.uint, arg=1, src=(
    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
      UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),)),)),))

/home/kraudy/tinygrad/tinygrad/tinygrad/tensor.py(572)full()-><Tensor <UOp ...ith grad None>
-> return Tensor(fill_value, **kwargs).reshape((1, )*len(new_shape := argfix(shape))).expand(new_shape)    

full() is the last function on Tensor
transpose()
permute()
tensor()
pool()
apply_uop()
flatten()
add()
tensor.py(3900)_wrapper()
ops.py(44)__add__()
tensor.py(3900)_wrapper()
tensor.py(537)rand()
tensor.py(3900)_wrapper()
tensor.py(492)_threefry_random_bits()
tensor.py(3713)cast()
dtype.py(156)to_dtype()
tensor.py(190)_apply_uop()
-> new_uop: UOp = fxn(*[t.lazydata for t in (self,)+x], **kwargs)

rand()

> /home/kraudy/tinygrad/tinygrad/tinygrad/tensor.py(572)full()-><Tensor <UOp ...ith grad None>
-> return Tensor(fill_value, **kwargs).reshape((1, )*len(new_shape := argfix(shape))).expand(new_shape)


This _apply_uop () Is used a lot in the tensor.py

So, this thing returns a created Tensor

> /home/kraudy/tinygrad/tinygrad/tinygrad/ops.py(235)__call__()
-> return created
(Pdb) created
UOp(Ops.BITCAST, dtypes.uint, arg=None, src=(
  UOp(Ops.EXPAND, dtypes.float, arg=(6,), src=(
    UOp(Ops.RESHAPE, dtypes.float, arg=(1,), src=(
      UOp(Ops.CONST, dtypes.float, arg=1.0, src=(
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
          UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),)),)),)),)),))

(Pdb) p s
UOp(Ops.EXPAND, dtypes.float, arg=(6,), src=(
  UOp(Ops.RESHAPE, dtypes.float, arg=(1,), src=(
    UOp(Ops.CONST, dtypes.float, arg=1.0, src=(
      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
        UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),)),)),)),))
(Pdb) p s.src
(UOp(Ops.RESHAPE, dtypes.float, arg=(1,), src=(
  UOp(Ops.CONST, dtypes.float, arg=1.0, src=(
    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
      UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),)),)),)),)

-> def _apply_broadcasted_uop(self, fxn:Callable, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
(Pdb) type(self)
<class 'tinygrad.tensor.Tensor'>
(Pdb) self.lazydata
UOp(Ops.EXPAND, dtypes.float, arg=(6,), src=(
  UOp(Ops.RESHAPE, dtypes.float, arg=(1,), src=(
    UOp(Ops.CONST, dtypes.float, arg=1.0, src=(
      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
        UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),)),)),)),))     
"""


"""
Tensor Lazy data, i dont' know why is so big

(Pdb) self.lazydata
UOp(Ops.ADD, dtypes.uint, arg=None, src=(
  UOp(Ops.PAD, dtypes.uint, arg=((0, 3),), src=(
    UOp(Ops.CAST, dtypes.uint, arg=None, src=(
      UOp(Ops.AND, dtypes.ulong, arg=None, src=(
        x3:=UOp(Ops.THREEFRY, dtypes.ulong, arg=None, src=(
          UOp(Ops.OR, dtypes.ulong, arg=None, src=(
            UOp(Ops.MUL, dtypes.ulong, arg=None, src=(
              UOp(Ops.CAST, dtypes.ulong, arg=None, src=(
                UOp(Ops.ADD, dtypes.uint, arg=None, src=(
                  x8:=UOp(Ops.ADD, dtypes.uint, arg=None, src=(
                    UOp(Ops.ADD, dtypes.uint, arg=None, src=(
                      UOp(Ops.RESHAPE, dtypes.uint, arg=(3,), src=(
                        UOp(Ops.REDUCE_AXIS, dtypes.uint, arg=(Ops.ADD, (1,)), src=(
                          UOp(Ops.PERMUTE, dtypes.uint, arg=(1, 0), src=(
                            UOp(Ops.RESHAPE, dtypes.uint, arg=(3, 3), src=(
                              UOp(Ops.RESHAPE, dtypes.uint, arg=(3, 3, 1), src=(
                                UOp(Ops.SHRINK, dtypes.uint, arg=((0, 3), (0, 3)), src=(
                                  UOp(Ops.RESHAPE, dtypes.uint, arg=(3, 6), src=(
                                    UOp(Ops.SHRINK, dtypes.uint, arg=((0, 18),), src=(
                                      UOp(Ops.RESHAPE, dtypes.uint, arg=(20,), src=(
                                        UOp(Ops.EXPAND, dtypes.uint, arg=(4, 5), src=(
                                          UOp(Ops.RESHAPE, dtypes.uint, arg=(1, 5), src=(
                                            UOp(Ops.PAD, dtypes.uint, arg=((2, 0),), src=(
                                              UOp(Ops.EXPAND, dtypes.uint, arg=(3,), src=(
                                                UOp(Ops.RESHAPE, dtypes.uint, arg=(1,), src=(
                                                  UOp(Ops.CONST, dtypes.uint, arg=1, src=(
                                                    x25:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
                                                      x26:=UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),)),)),)),)),)),)),)),)),)),)),)),)),)),)),)),)),
                      UOp(Ops.EXPAND, dtypes.uint, arg=(3,), src=(
                        UOp(Ops.RESHAPE, dtypes.uint, arg=(1,), src=(
                          UOp(Ops.CONST, dtypes.uint, arg=-1, src=(
                             x25,)),)),)),)),
                    UOp(Ops.EXPAND, dtypes.uint, arg=(3,), src=(
                      UOp(Ops.VIEW, dtypes.uint, arg=ShapeTracker(views=(View(shape=(1,), strides=(0,), offset=0, mask=None, contiguous=True),)), src=(
                        UOp(Ops.COPY, dtypes.uint, arg=False, src=(
                           x26,
                          UOp(Ops.BUFFER, dtypes.uint, arg=(1, 1), src=(
                            x34:=UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),)),)),)),)),
                  UOp(Ops.EXPAND, dtypes.uint, arg=(3,), src=(
                    UOp(Ops.RESHAPE, dtypes.uint, arg=(1,), src=(
                      UOp(Ops.CONST, dtypes.uint, arg=3, src=(
                         x25,)),)),)),)),)),
              x38:=UOp(Ops.EXPAND, dtypes.ulong, arg=(3,), src=(
                UOp(Ops.RESHAPE, dtypes.ulong, arg=(1,), src=(
                  UOp(Ops.CONST, dtypes.ulong, arg=4294967296, src=(
                     x25,)),)),)),)),
            UOp(Ops.CAST, dtypes.ulong, arg=None, src=(
               x8,)),)),
          UOp(Ops.OR, dtypes.ulong, arg=None, src=(
            UOp(Ops.MUL, dtypes.ulong, arg=None, src=(
              UOp(Ops.CAST, dtypes.ulong, arg=None, src=(
                UOp(Ops.EXPAND, dtypes.uint, arg=(3,), src=(
                  UOp(Ops.RESHAPE, dtypes.uint, arg=(1,), src=(
                    UOp(Ops.RESHAPE, dtypes.uint, arg=(), src=(
                      UOp(Ops.SHRINK, dtypes.uint, arg=((1, 2),), src=(
                        x49:=UOp(Ops.VIEW, dtypes.uint, arg=ShapeTracker(views=(View(shape=(2,), strides=(1,), offset=0, mask=None, contiguous=True),)), src=(
                          UOp(Ops.COPY, dtypes.uint, arg=False, src=(
                             x26,
                            UOp(Ops.BUFFER, dtypes.uint, arg=(0, 2), src=(
                               x34,)),)),)),)),)),)),)),)),
               x38,)),
            UOp(Ops.CAST, dtypes.ulong, arg=None, src=(
              UOp(Ops.EXPAND, dtypes.uint, arg=(3,), src=(
                UOp(Ops.RESHAPE, dtypes.uint, arg=(1,), src=(
                  UOp(Ops.RESHAPE, dtypes.uint, arg=(), src=(
                    UOp(Ops.SHRINK, dtypes.uint, arg=((0, 1),), src=(
                       x49,)),)),)),)),)),)),)),
        x57:=UOp(Ops.EXPAND, dtypes.ulong, arg=(3,), src=(
          UOp(Ops.RESHAPE, dtypes.ulong, arg=(1,), src=(
            UOp(Ops.CONST, dtypes.ulong, arg=4294967295, src=(
               x25,)),)),)),)),)),)),
  UOp(Ops.PAD, dtypes.uint, arg=((3, 0),), src=(
    UOp(Ops.CAST, dtypes.uint, arg=None, src=(
      UOp(Ops.AND, dtypes.ulong, arg=None, src=(
        UOp(Ops.IDIV, dtypes.ulong, arg=None, src=(
           x3,
           x38,)),
         x57,)),)),)),))
"""