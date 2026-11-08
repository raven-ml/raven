# tolk D151 golden=668d737c3713c2e6f461d7cdc0163a17
typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));
__attribute__((min_vector_width(512))) void mask(float* restrict data0_4, float* restrict data1_4, float* restrict data2_4) {
  float4 val0 = (*((float4*)((data1_4+0))));
  int4 cast0 = (val0!=0);
  float4 val1 = (*((float4*)((data2_4+0))));
  float4 alu0 = (cast0?val0:val1);
  *((float4*)((data0_4+0))) = alu0;
}

