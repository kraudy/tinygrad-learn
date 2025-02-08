from tinygrad.renderer.cstyle import MetalRenderer
from tinygrad.ops import UOp, Ops
from tinygrad import dtypes


const = UOp(Ops.CONST, dtypes.float, arg=1.0)
add = UOp(Ops.ADD, dtypes.float, src=(const, const), arg=None)

print(add)
print(MetalRenderer().render("example",[
  const,
  add
]))

print(MetalRenderer().render("example",[
  UOp(Ops.SPECIAL, dtypes.int, arg=("gidx0", 16))
]))
