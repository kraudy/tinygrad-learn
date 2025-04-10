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

"""