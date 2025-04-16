from tinygrad import Tensor, nn


a = Tensor([1., 2., 3., 4.])

"""
This data can be Numeric or UOp
(Pdb) data
[1.0, 2.0, 3.0, 4.0]

> /home/kraudy/tinygrad/tinygrad/tinygrad/device.py(21)canonicalize()->'CLANG'
-> def canonicalize(self, device:Optional[str]) -> str: return self._canonicalize(device) if device is not None else Device.DEFAULT

Here is the thing, since data is a Union of multiple types, that is why we need to
check isinstances for every data type afterwards

On this case it enters
> /home/kraudy/tinygrad/tinygrad/tinygrad/tensor.py(160)__init__()
-> elif isinstance(data, (list, tuple)):
(Pdb) --KeyboardInterrupt--
(Pdb) isinstance(data, (list, tuple))

> /home/kraudy/tinygrad/tinygrad/tinygrad/helpers.py(35)fully_flatten()
-> def fully_flatten(l):

> /home/kraudy/tinygrad/tinygrad/tinygrad/helpers.py(41)fully_flatten()->[1.0]
> /home/kraudy/tinygrad/tinygrad/tinygrad/helpers.py(41)fully_flatten()->[2.0]
> /home/kraudy/tinygrad/tinygrad/tinygrad/helpers.py(41)fully_flatten()->[3.0]
> /home/kraudy/tinygrad/tinygrad/tinygrad/helpers.py(41)fully_flatten()->[4.0]

It gets every elemnt out one at a time

> /home/kraudy/tinygrad/tinygrad/tinygrad/helpers.py(40)fully_flatten()->[1.0, 2.0, 3.0, 4.0]
-> return flattened

After that there some operatios to get the data type

Looks like this genereates UOps
> /home/kraudy/tinygrad/tinygrad/tinygrad/tensor.py(73)_frompy()
-> def _frompy(x:Union[list, tuple, bytes], dtype:DType) -> UOp:

After _frompy() it does to other options, i didn't know that even these explicit
tensors where turned to Uops
(Pdb) w
  /home/kraudy/tinygrad/tinygrad-learn/tracing_tensor.py(4)<module>()
-> a = Tensor([1., 2., 3., 4.])
  /home/kraudy/tinygrad/tinygrad/tinygrad/tensor.py(165)__init__()
-> else: data = _frompy(data, dtype)
  /home/kraudy/tinygrad/tinygrad/tinygrad/tensor.py(76)_frompy()

Looks like it needs the Uops to crate the views of the tensor

UOp can be called 

(Pdb) w
  /home/kraudy/tinygrad/tinygrad-learn/tracing_tensor.py(4)<module>()
-> a = Tensor([1., 2., 3., 4.])
  /home/kraudy/tinygrad/tinygrad/tinygrad/tensor.py(165)__init__()
-> else: data = _frompy(data, dtype)
  /home/kraudy/tinygrad/tinygrad/tinygrad/tensor.py(76)_frompy()
-> ret = UOp.metaop(Ops.EMPTY, get_shape(x), dtype, "PYTHON")
  /home/kraudy/tinygrad/tinygrad/tinygrad/ops.py(494)metaop()
-> return UOp(Ops.VIEW, dtype, (UOp.new_buffer(device, (st:=ShapeTracker.from_shape(shape)).size, dtype),), st)
  /home/kraudy/tinygrad/tinygrad/tinygrad/ops.py(535)new_buffer()
-> def new_buffer(device:str, size:int, dtype:DType): return UOp(Ops.BUFFER, dtype, (UOp(Ops.DEVICE, arg=device),), (next(UOp.buffer_num), size))
> /home/kraudy/tinygrad/tinygrad/tinygrad/ops.py(235)__call__()
-> return created

Got back to _frompy()
(Pdb) ret
UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(4,), strides=(1,), offset=0, mask=None, contiguous=True),)), src=(
  UOp(Ops.BUFFER, dtypes.float, arg=(2, 4), src=(
    UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),))
 
Looks like it does some hex conversion
data = struct.pack(f"@{ret.size}{dtype.fmt}", *[truncate_function(xi) for xi in fully_flatten(x)])
(Pdb) data
b'\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@'

After the correspoding path data has to be converted to a UOp
So all the previous options are to convert the data to a UOp


So this is the operations Graph, that is why they use tupple because each node has at max
2 parents and this also helps the toposort algorithm

(Pdb) data
UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(4,), strides=(1,), offset=0, mask=None, contiguous=True),)), src=(
  UOp(Ops.BUFFER, dtypes.float, arg=(2, 4), src=(
    UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),))
(Pdb) data.src[0]
UOp(Ops.BUFFER, dtypes.float, arg=(2, 4), src=(
  UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),))
(Pdb) data.src[0].src[0]
UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=())
(Pdb) data.src[0].src[0].src
()

Lazydata is of type UOp

Copying data from device PYTHON to CLANG
(Pdb) ret
UOp(Ops.COPY, dtypes.float, arg=False, src=(
  UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),
  UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
    UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),))

    
(Pdb) n
> /home/kraudy/tinygrad/tinygrad/tinygrad/ops.py(233)__call__()
-> for s in src: s.children.add(ref)
(Pdb) src
(UOp(Ops.VIEW, dtypes.int, arg=ShapeTracker(views=(View(shape=(4,), strides=(1,), offset=0, mask=None, contiguous=True),)), src=(
  UOp(Ops.COPY, dtypes.int, arg=False, src=(
    UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),
    UOp(Ops.BUFFER, dtypes.int, arg=(0, 4), src=(
      UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),)),)), UOp(Ops.EXPAND, dtypes.int, arg=(4,), src=(
  UOp(Ops.RESHAPE, dtypes.int, arg=(1,), src=(
    UOp(Ops.CONST, dtypes.int, arg=5, src=(
      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
        UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),)),)),)),)))


CALL Stack
        
/home/kraudy/tinygrad/tinygrad/tinygrad/tensor.py(3923)_wrapper()
-> ret = fn(*args, **kwargs)
  /home/kraudy/tinygrad/tinygrad/tinygrad/tensor.py(3103)add()
-> return self._apply_broadcasted_uop(UOp.add, x, reverse)
  /home/kraudy/tinygrad/tinygrad/tinygrad/tensor.py(3898)_wrapper()
-> if _METADATA.get() is not None: return fn(*args, **kwargs)
  /home/kraudy/tinygrad/tinygrad/tinygrad/tensor.py(194)_apply_broadcasted_uop()
-> return lhs._apply_uop(fxn, rhs)
  /home/kraudy/tinygrad/tinygrad/tinygrad/tensor.py(3898)_wrapper()
-> if _METADATA.get() is not None: return fn(*args, **kwargs)
  /home/kraudy/tinygrad/tinygrad/tinygrad/tensor.py(188)_apply_uop()
-> new_uop: UOp = fxn(*[t.lazydata for t in (self,)+x], **kwargs)
  /home/kraudy/tinygrad/tinygrad/tinygrad/ops.py(37)add()
-> return self._binop(Ops.ADD, x, reverse)
  /home/kraudy/tinygrad/tinygrad/tinygrad/ops.py(29)_binop()
-> def _binop(self, op, x, reverse): return self.ufix(x).alu(op, self) if reverse else self.alu(op, self.ufix(x))
  /home/kraudy/tinygrad/tinygrad/tinygrad/ops.py(391)alu()
-> return UOp(arg, out_dtype, (self,)+src)
> /home/kraudy/tinygrad/tinygrad/tinygrad/ops.py(240)__call__()->UOp(Ops.ADD, ...)),)),)),)),))
-> return created

(Pdb) created
UOp(Ops.ADD, dtypes.int, arg=None, src=(
  UOp(Ops.VIEW, dtypes.int, arg=ShapeTracker(views=(View(shape=(4,), strides=(1,), offset=0, mask=None, contiguous=True),)), src=(
    UOp(Ops.COPY, dtypes.int, arg=False, src=(
      x2:=UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),
      UOp(Ops.BUFFER, dtypes.int, arg=(0, 4), src=(
        UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),)),)),
  UOp(Ops.EXPAND, dtypes.int, arg=(4,), src=(
    UOp(Ops.RESHAPE, dtypes.int, arg=(1,), src=(
      UOp(Ops.CONST, dtypes.int, arg=5, src=(
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
           x2,)),)),)),)),))

"""
