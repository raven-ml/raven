# tolk D141 D151 golden=2eed21202a99e05d2f91b74845849d88
typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
#if !__has_builtin(__builtin_elementwise_sqrt)
#define __builtin_elementwise_sqrt tolk_lanes_sqrt
static inline __attribute__((overloadable, always_inline)) float4 tolk_lanes_sqrt(float4 a) { return (float4){__builtin_sqrtf(a[0]), __builtin_sqrtf(a[1]), __builtin_sqrtf(a[2]), __builtin_sqrtf(a[3])}; }
#endif
__attribute__((min_vector_width(512))) void sqrt(float* restrict data0_4, float* restrict data1_4) {
  float4 val0 = (*((float4*)((data1_4+0))));
  *((float4*)((data0_4+0))) = __builtin_elementwise_sqrt(val0);
}

