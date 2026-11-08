# tolk D141 D151 D342 golden=97f7906be01ca175869ee747e982920a
typedef unsigned char unsigned_char4 __attribute__((aligned(1),ext_vector_type(4)));
typedef unsigned int unsigned_int4 __attribute__((aligned(4),ext_vector_type(4)));
__attribute__((min_vector_width(512))) void r_256_4(unsigned char* restrict data0_1, unsigned char* restrict data1_1024, unsigned char* restrict data2_1024) {
  unsigned char buf0[1];
  *(buf0+0) = ((unsigned char)(0u));
  for (int Lidx0 = 0; Lidx0 < 256; Lidx0++) {
    unsigned char val0 = (*(buf0+0));
    int alu1 = (Lidx0<<2);
    unsigned_char4 val1 = (*((unsigned_char4*)((data1_1024+alu1))));
    unsigned_char4 val2 = (*((unsigned_char4*)((data2_1024+alu1))));
    unsigned_int4 cast0 = __builtin_convertvector(val1, unsigned_int4);
    unsigned_int4 cast1 = __builtin_convertvector(val2, unsigned_int4);
    unsigned_char4 cast2 = __builtin_convertvector((cast0+cast1), unsigned_char4);
    unsigned char alu2 = ((cast2[0]<cast2[1])?cast2[1]:cast2[0]);
    unsigned char alu3 = ((alu2<cast2[2])?cast2[2]:alu2);
    unsigned char alu4 = ((alu3<cast2[3])?cast2[3]:alu3);
    unsigned char alu5 = ((val0<alu4)?alu4:val0);
    *(buf0+0) = alu5;
  }
  *(data0_1+0) = (*(buf0+0));
}

