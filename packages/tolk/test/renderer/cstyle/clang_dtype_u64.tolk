# tolk D141 D151 D342 golden=663326c13580b21377631674a9beea6a
typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
__attribute__((min_vector_width(512))) void E_4_4(float* restrict data0_16, float* restrict data1_16, unsigned long* restrict data2_16) {
  for (int Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    int alu0 = (Lidx0<<2);
    unsigned long val0 = (*(data2_16+(alu0+1)));
    unsigned long val1 = (*(data2_16+(alu0+2)));
    unsigned long val2 = (*(data2_16+(alu0+3)));
    unsigned long val3 = (*(data2_16+alu0));
    float4 val4 = (*((float4*)((data1_16+alu0))));
    *((float4*)((data0_16+alu0))) = (float4){((float)(((((unsigned long)(val4[0]))+val3)*3ul))),((float)(((((unsigned long)(val4[1]))+val0)*3ul))),((float)(((((unsigned long)(val4[2]))+val1)*3ul))),((float)(((((unsigned long)(val4[3]))+val2)*3ul)))};
  }
}

