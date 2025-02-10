from tinygrad import Tensor, dtypes#, UPat, Ops
from tinygrad.ops import UPat, Ops, UOp, PatternMatcher
from tinygrad.renderer.cstyle import MetalRenderer


const_1 = UOp(Ops.CONST, dtypes.float, arg=0.5)
const_2 = UOp(Ops.CONST, dtypes.float, arg=0.5)

matcher = PatternMatcher([
  (UPat(Ops.CONST, dtypes.float, name="x"), lambda ctx, x: UOp(Ops.ADD, dtypes.float, src=(const_1, const_2))),
])

metal_renderer = MetalRenderer()
const = UOp(Ops.CONST, dtypes.float, arg=1.0)

const_rewritten = matcher.rewrite(const)

define_global = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0)
special = UOp(Ops.SPECIAL, dtypes.int, arg=('gidx0', 16), src=())
added = UOp(Ops.ADD, dtypes.long, arg=None, src=(define_global, special))
store = UOp(Ops.STORE, dtypes.void, arg=None, src=(added, const_rewritten))
uops = [const_1, const_2, const_rewritten, define_global, special, added, store]

rendered = metal_renderer.render("rendered", uops)
print(rendered)
