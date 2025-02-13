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
<Tensor <UOp CLANG (4, 4) float sShapeTracker(views=(View(shape=(4, 4), strides=(4, 1), 
offset=0, mask=None, contiguous=True),))> on CLANG with grad None>

Interesting how the tensor has the ShapeTraker and the target device (CLANG)
This means, a Tensor needs to have a target device for the AST 
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

print(Tensor.empty(4, 4).sum(0).shape)
"""(4,)"""
b = Tensor.empty(4,4)
print(b)
b = b.sum(0)
print(b)
"""
<Tensor <UOp CLANG (4, 4) float ShapeTracker(views=(View(shape=(4, 4), strides=(4, 1), offset=0, mask=None, contiguous=True),))> on CLANG with grad None>
<Tensor <UOp CLANG (4,) float ShapeTracker(views=(View(shape=(4,), strides=(1,), offset=0, mask=None, contiguous=True),))> on CLANG with grad None>

Interesting how the output of the reduction operation is itself another View, which make sense.
"""

b = Tensor.empty(4)
print(b)
b = b.sum(0)
print(b)

"""
Since a Tensor is made of View, the View got the Strides, with the Strides we know how to traverse each element
shape = (row, col)
stride = (x, y). # Remeber, a stride is like a x,y coordinate to identify points on a plane.
now we can traverse each element in the following manner
row = range(row) , col = range(col)
i = row*x + b*col
Why do we need this? We want to make operations of this thing and for that, we need a way to traverse through each
element.
So, the same View that we define helps us when we need to do the operations
"""

"""
                Add new dim   Expand to 3 
a = (2, 4) --> (2, 1, 4) ----------------> (2, 3, 4)
Here we are basically expanding the two rows 3 times each to mach the 3 columns of b
and do a vector dot product between each vector. Note dim 1 (3) is equal b dim 1 (3)

                Add new dim   Transpose
b = (4, 3) --> (1, 4, 3) --> (1, 3, 4) --> (2, 3, 4)
Note how we do the transform when the dim0 is 1, so when dim0 is 2, the (3,4) matrix gets expanded and then
we can do vector by vector dot product

Incrementing dim1, expands dim2
Incrementing dim0, expands dim1,dim2

At the end, a and b have the same shape but different data. After the reducion
c = (2, 3). dim2 dissapears

Now to the dot product a@b and get c = (2, 3)
We got to the operation just manipulating the View, doing a column-wise operation and then a reduce sum

A matrix dot product is just many vector dot product or scalar product where each scalar = (x, y) cell
"""
