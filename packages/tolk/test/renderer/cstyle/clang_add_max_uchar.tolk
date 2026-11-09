# tolk D141 D151 D342 golden=97f7906be01ca175869ee747e982920a

__attribute__((min_vector_width(512))) void r_256_4(unsigned char* restrict data0_1, unsigned char* restrict data1_1024, unsigned char* restrict data2_1024) {
  unsigned char buf0[1];
  *(buf0+0) = ((unsigned char)(0u));
  for (int Lidx0 = 0; Lidx0 < 256; Lidx0++) {
    int alu1 = (Lidx0<<2);
    int alu2 = (alu1+1);
    unsigned char val0 = (*(data1_1024+alu2));
    int alu3 = (alu1+2);
    unsigned char val1 = (*(data1_1024+alu3));
    int alu4 = (alu1+3);
    unsigned char val2 = (*(data1_1024+alu4));
    unsigned char val3 = (*(data1_1024+alu1));
    unsigned char val4 = (*(data2_1024+alu2));
    unsigned char val5 = (*(data2_1024+alu3));
    unsigned char val6 = (*(data2_1024+alu4));
    unsigned char val7 = (*(data2_1024+alu1));
    unsigned char val8 = (*(buf0+0));
    unsigned char cast0 = ((unsigned char)((((unsigned int)(val0))+((unsigned int)(val4)))));
    unsigned char cast1 = ((unsigned char)((((unsigned int)(val1))+((unsigned int)(val5)))));
    unsigned char cast2 = ((unsigned char)((((unsigned int)(val2))+((unsigned int)(val6)))));
    unsigned char cast3 = ((unsigned char)((((unsigned int)(val3))+((unsigned int)(val7)))));
    unsigned char alu5 = ((cast3<cast0)?cast0:cast3);
    unsigned char alu6 = ((alu5<cast1)?cast1:alu5);
    unsigned char alu7 = ((alu6<cast2)?cast2:alu6);
    unsigned char alu8 = ((val8<alu7)?alu7:val8);
    *(buf0+0) = alu8;
  }
  *(data0_1+0) = (*(buf0+0));
}

