# tolk D141 D151 D342 golden=0047a5c3b3e9cf1d454391c289d545fe

__attribute__((min_vector_width(512))) void E_4_4(unsigned short* restrict data0_16, unsigned short* restrict data1_16) {
  for (int Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    int alu0 = (Lidx0<<2);
    int alu1 = (alu0+1);
    unsigned short val0 = (*(data1_16+alu1));
    int alu2 = (alu0+2);
    unsigned short val1 = (*(data1_16+alu2));
    int alu3 = (alu0+3);
    unsigned short val2 = (*(data1_16+alu3));
    unsigned short val3 = (*(data1_16+alu0));
    float cast0 = __builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (unsigned short)(val0))))<<16u)));
    unsigned short alu4 = ((0.0f<cast0)?val0:32640u);
    unsigned short alu5 = ((cast0<-1.0f)?32704u:val0);
    unsigned int cast1 = __builtin_bit_cast(unsigned int, (float)((__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (unsigned short)(alu4))))<<16u)))+__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (unsigned short)(alu5))))<<16u))))));
    unsigned int alu6 = (((cast1&65535u)!=0u)?(cast1|65536u):cast1);
    unsigned int alu7 = (((-cast1&2139095040u)!=0u)?(cast1+((cast1>>16u)&1u)+32767u):alu6);
    unsigned short alu8 = ((cast0<1.0f)?65408u:val0);
    unsigned int cast2 = __builtin_bit_cast(unsigned int, (float)((__builtin_bit_cast(float, (unsigned int)((((unsigned int)(((unsigned short)((alu7>>16u)))))<<16u)))+__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (unsigned short)(alu8))))<<16u))))));
    unsigned int alu9 = (((cast2&65535u)!=0u)?(cast2|65536u):cast2);
    unsigned int alu10 = (((-cast2&2139095040u)!=0u)?(cast2+((cast2>>16u)&1u)+32767u):alu9);
    float cast3 = __builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (unsigned short)(val1))))<<16u)));
    unsigned short alu11 = ((0.0f<cast3)?val1:32640u);
    unsigned short alu12 = ((cast3<-1.0f)?32704u:val1);
    unsigned int cast4 = __builtin_bit_cast(unsigned int, (float)((__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (unsigned short)(alu11))))<<16u)))+__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (unsigned short)(alu12))))<<16u))))));
    unsigned int alu13 = (((cast4&65535u)!=0u)?(cast4|65536u):cast4);
    unsigned int alu14 = (((-cast4&2139095040u)!=0u)?(cast4+((cast4>>16u)&1u)+32767u):alu13);
    unsigned short alu15 = ((cast3<1.0f)?65408u:val1);
    unsigned int cast5 = __builtin_bit_cast(unsigned int, (float)((__builtin_bit_cast(float, (unsigned int)((((unsigned int)(((unsigned short)((alu14>>16u)))))<<16u)))+__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (unsigned short)(alu15))))<<16u))))));
    unsigned int alu16 = (((cast5&65535u)!=0u)?(cast5|65536u):cast5);
    unsigned int alu17 = (((-cast5&2139095040u)!=0u)?(cast5+((cast5>>16u)&1u)+32767u):alu16);
    float cast6 = __builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (unsigned short)(val2))))<<16u)));
    unsigned short alu18 = ((0.0f<cast6)?val2:32640u);
    unsigned short alu19 = ((cast6<-1.0f)?32704u:val2);
    unsigned int cast7 = __builtin_bit_cast(unsigned int, (float)((__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (unsigned short)(alu18))))<<16u)))+__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (unsigned short)(alu19))))<<16u))))));
    unsigned int alu20 = (((cast7&65535u)!=0u)?(cast7|65536u):cast7);
    unsigned int alu21 = (((-cast7&2139095040u)!=0u)?(cast7+((cast7>>16u)&1u)+32767u):alu20);
    unsigned short alu22 = ((cast6<1.0f)?65408u:val2);
    unsigned int cast8 = __builtin_bit_cast(unsigned int, (float)((__builtin_bit_cast(float, (unsigned int)((((unsigned int)(((unsigned short)((alu21>>16u)))))<<16u)))+__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (unsigned short)(alu22))))<<16u))))));
    unsigned int alu23 = (((cast8&65535u)!=0u)?(cast8|65536u):cast8);
    unsigned int alu24 = (((-cast8&2139095040u)!=0u)?(cast8+((cast8>>16u)&1u)+32767u):alu23);
    float cast9 = __builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (unsigned short)(val3))))<<16u)));
    unsigned short alu25 = ((0.0f<cast9)?val3:32640u);
    unsigned short alu26 = ((cast9<-1.0f)?32704u:val3);
    unsigned int cast10 = __builtin_bit_cast(unsigned int, (float)((__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (unsigned short)(alu25))))<<16u)))+__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (unsigned short)(alu26))))<<16u))))));
    unsigned int alu27 = (((cast10&65535u)!=0u)?(cast10|65536u):cast10);
    unsigned int alu28 = (((-cast10&2139095040u)!=0u)?(cast10+((cast10>>16u)&1u)+32767u):alu27);
    unsigned short alu29 = ((cast9<1.0f)?65408u:val3);
    unsigned int cast11 = __builtin_bit_cast(unsigned int, (float)((__builtin_bit_cast(float, (unsigned int)((((unsigned int)(((unsigned short)((alu28>>16u)))))<<16u)))+__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (unsigned short)(alu29))))<<16u))))));
    unsigned int alu30 = (((cast11&65535u)!=0u)?(cast11|65536u):cast11);
    unsigned int alu31 = (((-cast11&2139095040u)!=0u)?(cast11+((cast11>>16u)&1u)+32767u):alu30);
    *(data0_16+alu1) = __builtin_bit_cast(unsigned short, (unsigned short)(((unsigned short)((alu10>>16u)))));
    *(data0_16+alu2) = __builtin_bit_cast(unsigned short, (unsigned short)(((unsigned short)((alu17>>16u)))));
    *(data0_16+alu3) = __builtin_bit_cast(unsigned short, (unsigned short)(((unsigned short)((alu24>>16u)))));
    *(data0_16+alu0) = __builtin_bit_cast(unsigned short, (unsigned short)(((unsigned short)((alu31>>16u)))));
  }
}

