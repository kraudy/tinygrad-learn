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

"""
