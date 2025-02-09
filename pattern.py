from tinygrad import Tensor, dtypes#, UPat, Ops
from tinygrad.ops import UPat, Ops, UOp, PatternMatcher
from tinygrad.renderer.cstyle import MetalRenderer

metal_render = MetalRenderer()

const = UOp(Ops.CONST, dtypes.float, arg=1.0)
#define_global = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0)
# looks like the .ptr() is not needed
define_global = UOp(Ops.DEFINE_GLOBAL, dtypes.float, arg=0)
special = UOp(Ops.SPECIAL, dtypes.int, arg=('gidx0', 16), src=())
added = UOp(Ops.ADD, dtypes.long, arg=None, src=(define_global, special))
store = UOp(Ops.STORE, dtypes.void, arg=None, src=(added, const))
uops = [const, define_global, special, added, store]

rendered = metal_render.render("rendered", uops)
print(rendered)

const_1 = UOp(Ops.CONST, dtypes.float, arg=0.5)
const_2 = UOp(Ops.CONST, dtypes.float, arg=0.5)

matcher = PatternMatcher([
  (UPat(Ops.CONST, dtypes.float, name="x"), lambda ctx, x: UOp(Ops.ADD, dtypes.float, src=(const_1, const_2)))
])

const = UOp(Ops.CONST, dtypes.float, arg=1.0)
conts_rewrite = matcher.rewrite(const)
