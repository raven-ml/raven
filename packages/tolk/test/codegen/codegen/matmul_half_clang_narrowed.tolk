# tolk D141 golden=405100d7fed39318a103564acea22086
typedef __fp16 __fp168 __attribute__((aligned(2),ext_vector_type(8)));
typedef __fp16 __fp1632 __attribute__((aligned(2),ext_vector_type(32)));
typedef float float32 __attribute__((aligned(4),ext_vector_type(32)));
__attribute__((min_vector_width(512))) void r_64_2_16_2_64(unsigned char* restrict args_40) {
  float buf0[32];
  int val0 = (*(((int*)(args_40))+6));
  int val1 = (*(((int*)(args_40))+7));
  unsigned long val2 = (*(((unsigned long*)(args_40))+0));
  unsigned long val3 = (*(((unsigned long*)(args_40))+1));
  unsigned long val4 = (*(((unsigned long*)(args_40))+2));
  for (int Lidx1 = 0; Lidx1 < (val1-val0); Lidx1++) {
    int alu0 = (val0+Lidx1);
    int alu1 = (alu0<<6);
    for (int Lidx2 = 0; Lidx2 < 2; Lidx2++) {
      int alu2 = (Lidx2<<5);
      *(buf0+0) = 0.0f;
      *(buf0+1) = 0.0f;
      *(buf0+10) = 0.0f;
      *(buf0+11) = 0.0f;
      *(buf0+12) = 0.0f;
      *(buf0+13) = 0.0f;
      *(buf0+14) = 0.0f;
      *(buf0+15) = 0.0f;
      *(buf0+16) = 0.0f;
      *(buf0+17) = 0.0f;
      *(buf0+18) = 0.0f;
      *(buf0+19) = 0.0f;
      *(buf0+2) = 0.0f;
      *(buf0+20) = 0.0f;
      *(buf0+21) = 0.0f;
      *(buf0+22) = 0.0f;
      *(buf0+23) = 0.0f;
      *(buf0+24) = 0.0f;
      *(buf0+25) = 0.0f;
      *(buf0+26) = 0.0f;
      *(buf0+27) = 0.0f;
      *(buf0+28) = 0.0f;
      *(buf0+29) = 0.0f;
      *(buf0+3) = 0.0f;
      *(buf0+30) = 0.0f;
      *(buf0+31) = 0.0f;
      *(buf0+4) = 0.0f;
      *(buf0+5) = 0.0f;
      *(buf0+6) = 0.0f;
      *(buf0+7) = 0.0f;
      *(buf0+8) = 0.0f;
      *(buf0+9) = 0.0f;
      for (int Lidx0 = 0; Lidx0 < 64; Lidx0++) {
        __fp16 val5 = (*(((__fp16*)(val3))+(alu1+Lidx0)));
        int alu35 = ((Lidx0<<6)+alu2);
        __fp168 val6 = (*((__fp168*)((((__fp16*)(val4))+(alu35+16)))));
        __fp168 val7 = (*((__fp168*)((((__fp16*)(val4))+(alu35+24)))));
        __fp168 val8 = (*((__fp168*)((((__fp16*)(val4))+(alu35+8)))));
        __fp168 val9 = (*((__fp168*)((((__fp16*)(val4))+alu35))));
        float32 cast0 = __builtin_convertvector(((__fp1632){val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5,val5}*(__fp1632){val9[0],val9[1],val9[2],val9[3],val9[4],val9[5],val9[6],val9[7],val8[0],val8[1],val8[2],val8[3],val8[4],val8[5],val8[6],val8[7],val6[0],val6[1],val6[2],val6[3],val6[4],val6[5],val6[6],val6[7],val7[0],val7[1],val7[2],val7[3],val7[4],val7[5],val7[6],val7[7]}), float32);
        *(buf0+0) = ((*(buf0+0))+cast0[0]);
        *(buf0+1) = ((*(buf0+1))+cast0[1]);
        *(buf0+10) = ((*(buf0+10))+cast0[10]);
        *(buf0+11) = ((*(buf0+11))+cast0[11]);
        *(buf0+12) = ((*(buf0+12))+cast0[12]);
        *(buf0+13) = ((*(buf0+13))+cast0[13]);
        *(buf0+14) = ((*(buf0+14))+cast0[14]);
        *(buf0+15) = ((*(buf0+15))+cast0[15]);
        *(buf0+16) = ((*(buf0+16))+cast0[16]);
        *(buf0+17) = ((*(buf0+17))+cast0[17]);
        *(buf0+18) = ((*(buf0+18))+cast0[18]);
        *(buf0+19) = ((*(buf0+19))+cast0[19]);
        *(buf0+2) = ((*(buf0+2))+cast0[2]);
        *(buf0+20) = ((*(buf0+20))+cast0[20]);
        *(buf0+21) = ((*(buf0+21))+cast0[21]);
        *(buf0+22) = ((*(buf0+22))+cast0[22]);
        *(buf0+23) = ((*(buf0+23))+cast0[23]);
        *(buf0+24) = ((*(buf0+24))+cast0[24]);
        *(buf0+25) = ((*(buf0+25))+cast0[25]);
        *(buf0+26) = ((*(buf0+26))+cast0[26]);
        *(buf0+27) = ((*(buf0+27))+cast0[27]);
        *(buf0+28) = ((*(buf0+28))+cast0[28]);
        *(buf0+29) = ((*(buf0+29))+cast0[29]);
        *(buf0+3) = ((*(buf0+3))+cast0[3]);
        *(buf0+30) = ((*(buf0+30))+cast0[30]);
        *(buf0+31) = ((*(buf0+31))+cast0[31]);
        *(buf0+4) = ((*(buf0+4))+cast0[4]);
        *(buf0+5) = ((*(buf0+5))+cast0[5]);
        *(buf0+6) = ((*(buf0+6))+cast0[6]);
        *(buf0+7) = ((*(buf0+7))+cast0[7]);
        *(buf0+8) = ((*(buf0+8))+cast0[8]);
        *(buf0+9) = ((*(buf0+9))+cast0[9]);
      }
      int alu69 = (alu1+alu2);
      *((__fp168*)((((__fp16*)(val2))+(alu69+16)))) = (__fp168){((__fp16)((*(buf0+16)))),((__fp16)((*(buf0+17)))),((__fp16)((*(buf0+18)))),((__fp16)((*(buf0+19)))),((__fp16)((*(buf0+20)))),((__fp16)((*(buf0+21)))),((__fp16)((*(buf0+22)))),((__fp16)((*(buf0+23))))};
      *((__fp168*)((((__fp16*)(val2))+(alu69+24)))) = (__fp168){((__fp16)((*(buf0+24)))),((__fp16)((*(buf0+25)))),((__fp16)((*(buf0+26)))),((__fp16)((*(buf0+27)))),((__fp16)((*(buf0+28)))),((__fp16)((*(buf0+29)))),((__fp16)((*(buf0+30)))),((__fp16)((*(buf0+31))))};
      *((__fp168*)((((__fp16*)(val2))+(alu69+8)))) = (__fp168){((__fp16)((*(buf0+8)))),((__fp16)((*(buf0+9)))),((__fp16)((*(buf0+10)))),((__fp16)((*(buf0+11)))),((__fp16)((*(buf0+12)))),((__fp16)((*(buf0+13)))),((__fp16)((*(buf0+14)))),((__fp16)((*(buf0+15))))};
      *((__fp168*)((((__fp16*)(val2))+alu69))) = (__fp168){((__fp16)((*(buf0+0)))),((__fp16)((*(buf0+1)))),((__fp16)((*(buf0+2)))),((__fp16)((*(buf0+3)))),((__fp16)((*(buf0+4)))),((__fp16)((*(buf0+5)))),((__fp16)((*(buf0+6)))),((__fp16)((*(buf0+7))))};
    }
  }
}

