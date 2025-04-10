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

"""