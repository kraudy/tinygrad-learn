from tinygrad import Tensor


t = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
t.sum().backward()

"""
(Pdb) w
  /home/kraudy/tinygrad/tinygrad-learn/tracing_backwards.py(5)<module>()
-> t.sum().backward()
  /home/kraudy/tinygrad/tinygrad/tinygrad/tensor.py(929)backward()
-> all_uops = self.lazydata.toposort
  /home/kraudy/tinygrad/tinygrad/tinygrad/ops.py(282)toposort()
-> return _toposort(self, cache=set())
> /home/kraudy/tinygrad/tinygrad/tinygrad/ops.py(274)_toposort()
-> def _toposort(u:UOp, cache:set[UOp]):



(Pdb) p u
UOp(Ops.COPY, dtypes.float, arg=False, src=(
  UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),
  UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
    UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),))

UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=())

So, the toposort makes a dictionary with the operatios
and uses a Set to keep track of visited.
So, each _toposort() returns a dictionary with its Uop
(Pdb) nodes
{}

Some nice recursion call stack
(Pdb) w
  /home/kraudy/tinygrad/tinygrad-learn/tracing_backwards.py(5)<module>()
-> t.sum().backward()
  /home/kraudy/tinygrad/tinygrad/tinygrad/tensor.py(929)backward()
-> all_uops = self.lazydata.toposort
  /home/kraudy/tinygrad/tinygrad/tinygrad/ops.py(282)toposort()
-> return _toposort(self, cache=set())
  /home/kraudy/tinygrad/tinygrad/tinygrad/ops.py(278)_toposort()
-> for parent in u.src: nodes.update(_toposort(parent, cache))
  /home/kraudy/tinygrad/tinygrad/tinygrad/ops.py(278)_toposort()
-> for parent in u.src: nodes.update(_toposort(parent, cache))
  /home/kraudy/tinygrad/tinygrad/tinygrad/ops.py(278)_toposort()
-> for parent in u.src: nodes.update(_toposort(parent, cache))
  /home/kraudy/tinygrad/tinygrad/tinygrad/ops.py(278)_toposort()
-> for parent in u.src: nodes.update(_toposort(parent, cache))
  /home/kraudy/tinygrad/tinygrad/tinygrad/ops.py(278)_toposort()
-> for parent in u.src: nodes.update(_toposort(parent, cache))
> /home/kraudy/tinygrad/tinygrad/tinygrad/ops.py(281)_toposort()->{UOp(Ops.DEVIC...THON', src=()): None}

(Pdb) nodes
{UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()): None, UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
  UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)): None}

  (Pdb) nodes
{UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()): None, UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()): None, UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
  UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)): None}

(Pdb) nodes
{UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()): None, UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()): None, UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
  UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)): None, UOp(Ops.COPY, dtypes.float, arg=False, src=(
  UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),
  UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
    UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),)): None}

(Pdb) nodes
{UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()): None, UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()): None, UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
  UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)): None, UOp(Ops.COPY, dtypes.float, arg=False, src=(
  UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),
  UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
    UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),)): None}

(Pdb) nodes
{UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()): None, UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()): None, UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
  UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)): None, UOp(Ops.COPY, dtypes.float, arg=False, src=(
  UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),
  UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
    UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),)): None}


(Pdb) cache
{UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()), UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()), UOp(Ops.COPY, dtypes.float, arg=False, src=(
  UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),
  UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
    UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),)), UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
  UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),))}

(Pdb) nodes
{UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()): None, UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()): None, UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
  UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)): None, UOp(Ops.COPY, dtypes.float, arg=False, src=(
  UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),
  UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
    UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),)): None, UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(4,), strides=(1,), offset=0, mask=None, contiguous=True),)), src=(
  UOp(Ops.COPY, dtypes.float, arg=False, src=(
    UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),
    UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
      UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),)),)): None}

(Pdb) nodes
{UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()): None, UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()): None, UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
  UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)): None, UOp(Ops.COPY, dtypes.float, arg=False, src=(
  UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),
  UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
    UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),)): None, UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(4,), strides=(1,), offset=0, mask=None, contiguous=True),)), src=(
  UOp(Ops.COPY, dtypes.float, arg=False, src=(
    UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),
    UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
      UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),)),)): None, UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (0,)), src=(
  UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(4,), strides=(1,), offset=0, mask=None, contiguous=True),)), src=(
    UOp(Ops.COPY, dtypes.float, arg=False, src=(
      UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),
      UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
        UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),)),)),)): None}      

(Pdb) all_uops
{UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()): None, UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()): None, UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
  UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)): None, UOp(Ops.COPY, dtypes.float, arg=False, src=(
  UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),
  UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
    UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),)): None, UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(4,), strides=(1,), offset=0, mask=None, contiguous=True),)), src=(
  UOp(Ops.COPY, dtypes.float, arg=False, src=(
    UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),
    UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
      UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),)),)): None, UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (0,)), src=(
  UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(4,), strides=(1,), offset=0, mask=None, contiguous=True),)), src=(
    UOp(Ops.COPY, dtypes.float, arg=False, src=(
      UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),
      UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
        UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),)),)),)): None, UOp(Ops.RESHAPE, dtypes.float, arg=(), src=(
  UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (0,)), src=(
    UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(4,), strides=(1,), offset=0, mask=None, contiguous=True),)), src=(
      UOp(Ops.COPY, dtypes.float, arg=False, src=(
        UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),
        UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
          UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),)),)),)),)): None}        

Remember that gradients are tensors.
if gradient is None: gradient = Tensor(1.0, dtype=self.dtype, device=self.device, requires_grad=False)          

(Pdb) gradient
<Tensor <UOp CLANG () float (<Ops.CONST: 72>, None)> on CLANG with grad None>

These are the built grads
(Pdb) grads
{UOp(Ops.RESHAPE, dtypes.float, arg=(), src=(
  UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (0,)), src=(
    UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(4,), strides=(1,), offset=0, mask=None, contiguous=True),)), src=(
      UOp(Ops.COPY, dtypes.float, arg=False, src=(
        UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),
        UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
          UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),)),)),)),)): UOp(Ops.CONST, dtypes.float, arg=1.0, src=(
  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
    UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),)),)), UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (0,)), src=(
  UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(4,), strides=(1,), offset=0, mask=None, contiguous=True),)), src=(
    UOp(Ops.COPY, dtypes.float, arg=False, src=(
      UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),
      UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
        UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),)),)),)): UOp(Ops.RESHAPE, dtypes.float, arg=(1,), src=(
  UOp(Ops.CONST, dtypes.float, arg=1.0, src=(
    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
      UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),)),)),)), UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(4,), strides=(1,), offset=0, mask=None, contiguous=True),)), src=(
  UOp(Ops.COPY, dtypes.float, arg=False, src=(
    UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),
    UOp(Ops.BUFFER, dtypes.float, arg=(0, 4), src=(
      UOp(Ops.DEVICE, dtypes.void, arg='PYTHON', src=()),)),)),)): UOp(Ops.EXPAND, dtypes.float, arg=(4,), src=(
  UOp(Ops.RESHAPE, dtypes.float, arg=(1,), src=(
    UOp(Ops.CONST, dtypes.float, arg=1.0, src=(
      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
        UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),)),)),)),))}

        
[UOp(Ops.EXPAND, dtypes.float, arg=(4,), src=(
  UOp(Ops.RESHAPE, dtypes.float, arg=(1,), src=(
    UOp(Ops.CONST, dtypes.float, arg=1.0, src=(
      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
        UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),)),)),)),)), UOp(Ops.CONST, dtypes.float, arg=1.0, src=(
  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
    UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),)),))]        


This is the Tensor after Backwards
  4     t = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
  5  -> t.sum().backward()
[EOF]
(Pdb) t
<Tensor <UOp CLANG (4,) float ShapeTracker(views=(View(shape=(4,), strides=(1,), offset=0, mask=None, contiguous=True),))> on CLANG with grad UOp(Ops.EXPAND, dtypes.float, arg=(4,), src=(
  UOp(Ops.RESHAPE, dtypes.float, arg=(1,), src=(
    UOp(Ops.CONST, dtypes.float, arg=1.0, src=(
      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),)), src=(
        UOp(Ops.DEVICE, dtypes.void, arg='CLANG', src=()),)),)),)),))>    
"""