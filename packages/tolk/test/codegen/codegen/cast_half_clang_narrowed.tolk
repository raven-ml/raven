# tolk D141 golden=e67709fd8ba0dd512df3c7b5e5d4fb2e
typedef __fp16 __fp168 __attribute__((aligned(2),ext_vector_type(8)));
typedef __fp16 __fp1616 __attribute__((aligned(2),ext_vector_type(16)));
typedef float float16 __attribute__((aligned(4),ext_vector_type(16)));
typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
__attribute__((min_vector_width(512))) void E_16(unsigned char* restrict args_16) {
  unsigned long val0 = (*(((unsigned long*)(args_16))+0));
  unsigned long val1 = (*(((unsigned long*)(args_16))+1));
  __fp168 val2 = (*((__fp168*)((((__fp16*)(val1))+0))));
  __fp168 val3 = (*((__fp168*)((((__fp16*)(val1))+8))));
  float16 cast0 = __builtin_convertvector(((__fp1616){val2[0],val2[1],val2[2],val2[3],val2[4],val2[5],val2[6],val2[7],val3[0],val3[1],val3[2],val3[3],val3[4],val3[5],val3[6],val3[7]}*((__fp16)(3.0f))), float16);
  *((float4*)((((float*)(val0))+0))) = (float4){cast0[0],cast0[1],cast0[2],cast0[3]};
  *((float4*)((((float*)(val0))+12))) = (float4){cast0[12],cast0[13],cast0[14],cast0[15]};
  *((float4*)((((float*)(val0))+4))) = (float4){cast0[4],cast0[5],cast0[6],cast0[7]};
  *((float4*)((((float*)(val0))+8))) = (float4){cast0[8],cast0[9],cast0[10],cast0[11]};
}

