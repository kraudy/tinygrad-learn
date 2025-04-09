from tinygrad import Tensor

a = Tensor([1,2,3])
b = Tensor([4,5,6])

c = a + b

print(c)
"""<Tensor <UOp CLANG (3,) int (<Ops.ADD: 46>, None)> on CLANG with grad None>"""
c.realize()
print(c.numpy())

