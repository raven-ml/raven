# tolk D141 golden=975eb0a1e5d93345f4da14a3109f84e8
typedef __fp16 __fp168 __attribute__((aligned(2),ext_vector_type(8)));
typedef __fp16 __fp1616 __attribute__((aligned(2),ext_vector_type(16)));
typedef float float16 __attribute__((aligned(4),ext_vector_type(16)));
typedef int int16 __attribute__((aligned(4),ext_vector_type(16)));
typedef unsigned int unsigned_int16 __attribute__((aligned(4),ext_vector_type(16)));
typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
__attribute__((min_vector_width(512))) void E_16(unsigned char* restrict args_24) {
  unsigned long val0 = (*(((unsigned long*)(args_24))+0));
  unsigned long val1 = (*(((unsigned long*)(args_24))+1));
  unsigned long val2 = (*(((unsigned long*)(args_24))+2));
  __fp168 val3 = (*((__fp168*)((((__fp16*)(val1))+0))));
  __fp168 val4 = (*((__fp168*)((((__fp16*)(val1))+8))));
  __fp168 val5 = (*((__fp168*)((((__fp16*)(val2))+0))));
  __fp168 val6 = (*((__fp168*)((((__fp16*)(val2))+8))));
  float16 cast0 = __builtin_convertvector((__fp1616){val3[0],val3[1],val3[2],val3[3],val3[4],val3[5],val3[6],val3[7],val4[0],val4[1],val4[2],val4[3],val4[4],val4[5],val4[6],val4[7]}, float16);
  int16 alu0 = (cast0!=cast0);
  float16 alu1 = ((cast0!=((float)(-__builtin_inff())))?cast0:0.0f);
  float16 alu2 = (alu0?0.0f:alu1);
  float16 alu3 = ((cast0!=((float)(__builtin_inff())))?alu2:0.0f);
  float16 alu4 = ((alu3<0.0f)?((float16)(-0.5f)):0.5f);
  int16 cast1 = __builtin_convertvector((alu3+alu4), int16);
  float16 cast2 = __builtin_convertvector(cast1, float16);
  unsigned_int16 cast3 = __builtin_convertvector(cast1, unsigned_int16);
  int16 alu5 = (cast1>>1);
  unsigned_int16 cast4 = __builtin_convertvector(alu5, unsigned_int16);
  int16 cast5 = __builtin_convertvector((cast3-cast4), int16);
  unsigned_int16 cast6 = __builtin_convertvector(cast5, unsigned_int16);
  int16 cast7 = __builtin_convertvector((cast6+127u), int16);
  unsigned_int16 cast8 = __builtin_convertvector(cast7, unsigned_int16);
  int16 cast9 = __builtin_convertvector((cast8<<23u), int16);
  unsigned_int16 cast10 = __builtin_convertvector((alu5+127), unsigned_int16);
  int16 cast11 = __builtin_convertvector((cast10<<23u), int16);
  float16 alu6 = (alu3-cast2);
  float16 alu7 = ((~((cast0<128.0f)^0))?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu6)+0.0013392627006396651f)*alu6)+0.009618384763598442f)*alu6)+0.055503472685813904f)*alu6)+0.24022644758224487f)*alu6)+0.6931471824645996f)*alu6)+1.0f)*((float16)((int16)(cast11)))*((float16)((int16)(cast9)))));
  float16 alu8 = ((cast0<-150.0f)?0.0f:alu7);
  float16 alu9 = (alu0?__builtin_bit_cast(float, (unsigned int)(0x7fc00000u)):alu8);
  __fp1616 cast12 = __builtin_convertvector(alu9, __fp1616);
  float16 cast13 = __builtin_convertvector((cast12+__builtin_elementwise_sqrt((__fp1616){val5[0],val5[1],val5[2],val5[3],val5[4],val5[5],val5[6],val5[7],val6[0],val6[1],val6[2],val6[3],val6[4],val6[5],val6[6],val6[7]})), float16);
  *((float4*)((((float*)(val0))+0))) = (float4){cast13[0],cast13[1],cast13[2],cast13[3]};
  *((float4*)((((float*)(val0))+12))) = (float4){cast13[12],cast13[13],cast13[14],cast13[15]};
  *((float4*)((((float*)(val0))+4))) = (float4){cast13[4],cast13[5],cast13[6],cast13[7]};
  *((float4*)((((float*)(val0))+8))) = (float4){cast13[8],cast13[9],cast13[10],cast13[11]};
}

