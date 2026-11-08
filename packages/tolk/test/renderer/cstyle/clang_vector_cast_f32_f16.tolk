# tolk D141 D151 golden=765c0c15dcca0457647b363ee543d2d3
typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
typedef __fp16 __fp164 __attribute__((aligned(2),ext_vector_type(4)));
__attribute__((min_vector_width(512))) void cast(__fp16* restrict data0_4, float* restrict data1_4) {
  float4 val0 = (*((float4*)((data1_4+0))));
  __fp164 cast0 = __builtin_convertvector(val0, __fp164);
  *((__fp164*)((data0_4+0))) = cast0;
}

