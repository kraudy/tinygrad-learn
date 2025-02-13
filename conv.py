from tinygrad import Tensor

#a = Tensor([i for i in range(16)]).reshape((4,4))
a = Tensor([i for i in range(16)]).reshape((1,1,4,4))
print(a.numpy())
"""
[[[[ 0  1  2  3]
   [ 4  5  6  7]
   [ 8  9 10 11]
   [12 13 14 15]]]]
"""

weight = Tensor.ones(1, 1, 3, 3)
print(weight.numpy())
"""
[[[[1. 1. 1.]
   [1. 1. 1.]
   [1. 1. 1.]]]]
"""

out = a.conv2d(weight)
print(out.shape)
print(out.numpy())
"""
[[[[45. 54.]
   [81. 90.]]]]
"""

pooled = a._pool(k_=(3,3), stride=1, dilation=1)
print(pooled.shape)
print(pooled.numpy())
"""
[[[[[[ 0  1  2]
     [ 4  5  6]
     [ 8  9 10]]

    [[ 1  2  3]
     [ 5  6  7]
     [ 9 10 11]]]


   [[[ 4  5  6]
     [ 8  9 10]
     [12 13 14]]

    [[ 5  6  7]
     [ 9 10 11]
     [13 14 15]]]]]]
"""

"""
Note how each pool operation is a matrix mul and the output is the sum reduction
_pool is a series of expand, permute and shrink like the matmul
"""

pooled = a._pool(k_=(2,2), stride=2, dilation=1)
print(pooled.numpy())
"""
[[[[[[ 0  1]
     [ 4  5]]

    [[ 2  3]
     [ 6  7]]]


   [[[ 8  9]
     [12 13]]

    [[10 11]
     [14 15]]]]]]
"""

"""
Note how we can get the same number of output but with different results changin the kernel shape and stride
"""

pooled = a._pool(k_=(2,2), stride=1, dilation=1)
print(pooled.numpy())
"""
[[[[[[ 0  1]
     [ 4  5]]

    [[ 1  2]
     [ 5  6]]

    [[ 2  3]
     [ 6  7]]]


   [[[ 4  5]
     [ 8  9]]

    [[ 5  6]
     [ 9 10]]

    [[ 6  7]
     [10 11]]]


   [[[ 8  9]
     [12 13]]

    [[ 9 10]
     [13 14]]

    [[10 11]
     [14 15]]]]]]
"""

"""
Note how reducing the kernel size and the stride increases the output
"""

pooled = a._pool(k_=(2, 2), stride=1, dilation=2) # dilation=2
print(pooled.numpy())

"""
[[[[[[ 0  2]
     [ 8 10]]

    [[ 1  3]
     [ 9 11]]]


   [[[ 4  6]
     [12 14]]

    [[ 5  7]
     [13 15]]]]]]
"""

pooled = a._pool(k_=(2, 2), stride=2, dilation=2) # dilation=2
print(pooled.numpy())
"""
[[[[[[ 0  2]
     [ 8 10]]]]]]
"""

"""
Looks like dilation 'skips' the middle values
"""

Tensor.arange(15).realize()

""" No optimize
void r_15_15(int* restrict data0) {
  for (int ridx0 = 0; ridx0 < 15; ridx0++) {
    *(data0+ridx0) = ridx0;
  }
}
"""

""" Optimized
void r_15_15(int* restrict data0) {
  for (int ridx0 = 0; ridx0 < 15; ridx0++) {
    *(data0+ridx0) = ((((ridx0<13)!=1)?1:0)+(((ridx0<14)!=1)?1:0)+(((ridx0<12)!=1)?1:0)+(((ridx0<11)!=1)?1:0)+(((ridx0<10)!=1)?1:0)+(((ridx0<9)!=1)?1:0)+(((ridx0<8)!=1)?1:0)+(((ridx0<7)!=1)?1:0)+(((ridx0<6)!=1)?1:0)+(((ridx0<5)!=1)?1:0)+(((ridx0<4)!=1)?1:0)+(((ridx0<3)!=1)?1:0)+(((ridx0<2)!=1)?1:0)+(((ridx0<1)!=1)?1:0));
  }
}
"""

"""
Generates something like this. Looking to implement vector operation and hardware paralelization
[[0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [0 0 0 0 0 0 0 0 0 0 0 0 1 1 1]
 [0 0 0 0 0 0 0 0 0 0 0 1 1 1 1]
 [0 0 0 0 0 0 0 0 0 0 1 1 1 1 1]
 [0 0 0 0 0 0 0 0 0 1 1 1 1 1 1]
 [0 0 0 0 0 0 0 0 1 1 1 1 1 1 1]
 [0 0 0 0 0 0 0 1 1 1 1 1 1 1 1]
 [0 0 0 0 0 0 1 1 1 1 1 1 1 1 1]
 [0 0 0 0 0 1 1 1 1 1 1 1 1 1 1]
 [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1]
 [0 0 0 1 1 1 1 1 1 1 1 1 1 1 1]
 [0 0 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]]
 """
print(Tensor.arange(15).numpy())