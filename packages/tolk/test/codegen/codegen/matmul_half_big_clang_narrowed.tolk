# tolk D141 golden=1b664a2a2181a435f22f4b60dc2a40b7
typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));
typedef __fp16 __fp168 __attribute__((aligned(2),ext_vector_type(8)));
typedef __fp16 __fp1632 __attribute__((aligned(2),ext_vector_type(32)));
typedef float float32 __attribute__((aligned(4),ext_vector_type(32)));
__attribute__((min_vector_width(512))) void r_256_8_16_2_256(unsigned char* restrict args_40) {
  float buf0[32];
  int val0 = (*(((int*)(args_40))+6));
  int val1 = (*(((int*)(args_40))+7));
  unsigned long val2 = (*(((unsigned long*)(args_40))+0));
  unsigned long val3 = (*(((unsigned long*)(args_40))+1));
  unsigned long val4 = (*(((unsigned long*)(args_40))+2));
  float4 cast0 = (float4){0.0f,0.0f,0.0f,0.0f};
  for (int Lidx1 = 0; Lidx1 < (val1-val0); Lidx1++) {
    int alu0 = (val0+Lidx1);
    int alu1 = (alu0<<8);
    for (int Lidx2 = 0; Lidx2 < 8; Lidx2++) {
      int alu2 = (Lidx2<<5);
      *((float4*)((buf0+0))) = cast0;
      *((float4*)((buf0+12))) = cast0;
      *((float4*)((buf0+16))) = cast0;
      *((float4*)((buf0+20))) = cast0;
      *((float4*)((buf0+24))) = cast0;
      *((float4*)((buf0+28))) = cast0;
      *((float4*)((buf0+4))) = cast0;
      *((float4*)((buf0+8))) = cast0;
      for (int Lidx0 = 0; Lidx0 < 256; Lidx0++) {
        __fp16 val5 = (*(((__fp16*)(val3))+(alu1+Lidx0)));
        int alu11 = ((Lidx0<<8)+alu2);
        __fp168 val6 = (*((__fp168*)((((__fp16*)(val4))+(alu11+16)))));
        __fp168 val7 = (*((__fp168*)((((__fp16*)(val4))+(alu11+24)))));
        __fp168 val8 = (*((__fp168*)((((__fp16*)(val4))+(alu11+8)))));
        __fp168 val9 = (*((__fp168*)((((__fp16*)(val4))+alu11))));
        float4 val10 = (*((float4*)((buf0+0))));
        float4 val11 = (*((float4*)((buf0+12))));
        float4 val12 = (*((float4*)((buf0+16))));
        float4 val13 = (*((float4*)((buf0+20))));
        float4 val14 = (*((float4*)((buf0+24))));
        float4 val15 = (*((float4*)((buf0+28))));
        float4 val16 = (*((float4*)((buf0+4))));
        float4 val17 = (*((float4*)((buf0+8))));
        float32 cast1 = __builtin_convertvector(((__fp1632){val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5}*(__fp1632){val9[0],val9[1],val9[2],val9[3],val9[4],val9[5],val9[6],val9[7],val8[0],val8[1],val8[2],val8[3],val8[4],val8[5],val8[6],val8[7],val6[0],val6[1],val6[2],val6[3],val6[4],val6[5],val6[6],val6[7],val7[0],val7[1],val7[2],val7[3],val7[4],val7[5],val7[6],val7[7]}), float32);
        float32 alu12 = ((float32){val10[0],val10[1],val10[2],val10[3],val16[0],val16[1],val16[2],val16[3],val17[0],val17[1],val17[2],val17[3],val11[0],val11[1],val11[2],val11[3],val12[0],val12[1],val12[2],val12[3],val13[0],val13[1],val13[2],val13[3],val14[0],val14[1],val14[2],val14[3],val15[0],val15[1],val15[2],val15[3]}+cast1);
        *((float4*)((buf0+0))) = (float4){alu12[0],alu12[1],alu12[2],alu12[3]};
        *((float4*)((buf0+12))) = (float4){alu12[12],alu12[13],alu12[14],alu12[15]};
        *((float4*)((buf0+16))) = (float4){alu12[16],alu12[17],alu12[18],alu12[19]};
        *((float4*)((buf0+20))) = (float4){alu12[20],alu12[21],alu12[22],alu12[23]};
        *((float4*)((buf0+24))) = (float4){alu12[24],alu12[25],alu12[26],alu12[27]};
        *((float4*)((buf0+28))) = (float4){alu12[28],alu12[29],alu12[30],alu12[31]};
        *((float4*)((buf0+4))) = (float4){alu12[4],alu12[5],alu12[6],alu12[7]};
        *((float4*)((buf0+8))) = (float4){alu12[8],alu12[9],alu12[10],alu12[11]};
      }
      float4 val18 = (*((float4*)((buf0+0))));
      float4 val19 = (*((float4*)((buf0+12))));
      float4 val20 = (*((float4*)((buf0+16))));
      float4 val21 = (*((float4*)((buf0+20))));
      float4 val22 = (*((float4*)((buf0+24))));
      float4 val23 = (*((float4*)((buf0+28))));
      float4 val24 = (*((float4*)((buf0+4))));
      float4 val25 = (*((float4*)((buf0+8))));
      __fp1632 cast2 = __builtin_convertvector((float32){val18[0],val18[1],val18[2],val18[3],val24[0],val24[1],val24[2],val24[3],val25[0],val25[1],val25[2],val25[3],val19[0],val19[1],val19[2],val19[3],val20[0],val20[1],val20[2],val20[3],val21[0],val21[1],val21[2],val21[3],val22[0],val22[1],val22[2],val22[3],val23[0],val23[1],val23[2],val23[3]}, __fp1632);
      int alu22 = (alu1+alu2);
      *((__fp168*)((((__fp16*)(val2))+(alu22+16)))) = (__fp168){cast2[16],cast2[17],cast2[18],cast2[19],cast2[20],cast2[21],cast2[22],cast2[23]};
      *((__fp168*)((((__fp16*)(val2))+(alu22+24)))) = (__fp168){cast2[24],cast2[25],cast2[26],cast2[27],cast2[28],cast2[29],cast2[30],cast2[31]};
      *((__fp168*)((((__fp16*)(val2))+(alu22+8)))) = (__fp168){cast2[8],cast2[9],cast2[10],cast2[11],cast2[12],cast2[13],cast2[14],cast2[15]};
      *((__fp168*)((((__fp16*)(val2))+alu22))) = (__fp168){cast2[0],cast2[1],cast2[2],cast2[3],cast2[4],cast2[5],cast2[6],cast2[7]};
    }
  }
}

