# tolk D141 D151 D342 golden=2e8d4206893c4e2364df9dd5fd58fdb9
typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
__attribute__((min_vector_width(512))) void E_4_4(float* restrict data0_16, float* restrict data1_16, unsigned char* restrict data2_16) {
  for (int Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    int alu0 = (Lidx0<<2);
    unsigned char val0 = (*(data2_16+(alu0+1)));
    unsigned char val1 = (*(data2_16+(alu0+2)));
    unsigned char val2 = (*(data2_16+(alu0+3)));
    unsigned char val3 = (*(data2_16+alu0));
    float4 val4 = (*((float4*)((data1_16+alu0))));
    *((float4*)((data0_16+alu0))) = (float4){((float)(((unsigned char)((((unsigned int)(((unsigned char)((((unsigned int)(((unsigned char)(val4[0]))))+((unsigned int)(val3)))))))*3u))))),((float)(((unsigned char)((((unsigned int)(((unsigned char)((((unsigned int)(((unsigned char)(val4[1]))))+((unsigned int)(val0)))))))*3u))))),((float)(((unsigned char)((((unsigned int)(((unsigned char)((((unsigned int)(((unsigned char)(val4[2]))))+((unsigned int)(val1)))))))*3u))))),((float)(((unsigned char)((((unsigned int)(((unsigned char)((((unsigned int)(((unsigned char)(val4[3]))))+((unsigned int)(val2)))))))*3u)))))};
  }
}

