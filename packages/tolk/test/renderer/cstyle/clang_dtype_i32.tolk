# tolk D141 D151 golden=1df679bd7e26f4bac2c63e28f7db58e8
typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));
typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
__attribute__((min_vector_width(512))) void E_4_4(float* restrict data0_16, float* restrict data1_16, int* restrict data2_16) {
  for (int Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    int alu0 = (Lidx0<<2);
    int4 val0 = (*((int4*)((data2_16+alu0))));
    float4 val1 = (*((float4*)((data1_16+alu0))));
    *((float4*)((data0_16+alu0))) = (float4){((float)(((((int)(val1[0]))+val0[0])*3))),((float)(((((int)(val1[1]))+val0[1])*3))),((float)(((((int)(val1[2]))+val0[2])*3))),((float)(((((int)(val1[3]))+val0[3])*3)))};
  }
}

