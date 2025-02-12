from tinygrad import Tensor

print((Tensor.empty(4,4) * Tensor.empty(4, 4)).shape)
"""
So, a Tensor is basically an AST with an specific device.
Which is needed to do the actual operations, which means, it also
has to have a JIT compiler for the AST to the tensor device; CLANG in this case.
"""

a = Tensor.empty(4,4)
ld = a.lazydata
print(a)
"""
<Tensor <UOp CLANG (4, 4) float 
ShapeTracker(views=(View(shape=(4, 4), strides=(4, 1), 
offset=0, mask=None, contiguous=True),))> on CLANG with grad None>

Interesting how the tensor has the ShapeTraker and the target device (CLANG)
This means, a Tenser needs to have a target device for the AST 
"""

print(a.lazydata)
"""
UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(4, 4), strides=(4, 1), offset=0, mask=None, contiguous=True),)), src=(
  UOp(Ops.BUFFER, dtypes.float, arg=(2, 16), src=(
    UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),)),))
"""
print(a.device)
"""CLANG"""
print(ld.st if ld.base is not ld else (ld.op, ld.realized))
print(ld.st)
"""ShapeTracker(views=(View(shape=(4, 4), strides=(4, 1), offset=0, mask=None, contiguous=True),))"""
print((ld.op, ld.realized))
"""(<Ops.VIEW: 22>, None)"""



