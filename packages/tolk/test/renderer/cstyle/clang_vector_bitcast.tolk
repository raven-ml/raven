# tolk D151 golden=4ca9bf9eccad0b4709ce8ac8d497b2b7
typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
typedef unsigned int unsigned_int4 __attribute__((aligned(4),ext_vector_type(4)));
__attribute__((min_vector_width(512))) void bitcast(unsigned int* restrict data0_4, float* restrict data1_4) {
  float4 val0 = (*((float4*)((data1_4+0))));
  *((unsigned_int4*)((data0_4+0))) = ((unsigned_int4)((float4)(val0)));
}

