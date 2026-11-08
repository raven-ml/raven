# tolk D141 D151 golden=fb035a49cd41e405bf9993227f5485ff
typedef __fp16 __fp164 __attribute__((aligned(2),ext_vector_type(4)));
typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
__attribute__((min_vector_width(512))) void cast(float* restrict data0_4, __fp16* restrict data1_4) {
  __fp164 val0 = (*((__fp164*)((data1_4+0))));
  float4 cast0 = __builtin_convertvector(val0, float4);
  *((float4*)((data0_4+0))) = cast0;
}

