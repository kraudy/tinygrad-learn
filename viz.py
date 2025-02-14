from tinygrad import Tensor

a = Tensor.empty(4)
a.sum(0).realize()
"""
Loop with no optimization
void r_4(float* restrict data0, float* restrict data1) {
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float val0 = *(data1+ridx0);
    acc0 = (acc0+val0);
  }
  *(data0+0) = acc0;
}
"""

"""
Optimized loop
void r_4(float* restrict data0, float* restrict data1) {
  float4 val0 = *((float4*)((data1+0)));
  *(data0+0) = (val0[3]+val0[2]+val0[0]+val0[1]);
}
"""
