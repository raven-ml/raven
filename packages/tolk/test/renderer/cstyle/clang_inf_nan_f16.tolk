# tolk D27 D88 D151 D342 golden=2ac991b6551c027ca4a6ae156f468dec
typedef __fp16 __fp164 __attribute__((aligned(2),ext_vector_type(4)));
__attribute__((min_vector_width(512))) void E_4_4(__fp16* restrict data0_16, __fp16* restrict data1_16) {
  for (int Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    int alu0 = (Lidx0<<2);
    __fp164 val0 = (*((__fp164*)((data1_16+alu0))));
    __fp16 alu1 = __builtin_bit_cast(__fp16, (unsigned short)(((val0[0]<((__fp16)(-1.0f)))?__builtin_bit_cast(unsigned short, (__fp16)(__builtin_bit_cast(__fp16, (unsigned short)(0x7e00u)))):__builtin_bit_cast(unsigned short, (__fp16)(val0[0])))));
    __fp16 alu2 = __builtin_bit_cast(__fp16, (unsigned short)(((((__fp16)(({float z = ((__fp16)(0.0f)); __asm__("" : "+x"(z)); z;})))<val0[0])?__builtin_bit_cast(unsigned short, (__fp16)(val0[0])):__builtin_bit_cast(unsigned short, (__fp16)(((__fp16)(__builtin_inff())))))));
    __fp16 alu3 = __builtin_bit_cast(__fp16, (unsigned short)(((val0[0]<((__fp16)(1.0f)))?__builtin_bit_cast(unsigned short, (__fp16)(((__fp16)(-__builtin_inff())))):__builtin_bit_cast(unsigned short, (__fp16)(val0[0])))));
    __fp16 alu4 = __builtin_bit_cast(__fp16, (unsigned short)(((val0[1]<((__fp16)(-1.0f)))?__builtin_bit_cast(unsigned short, (__fp16)(__builtin_bit_cast(__fp16, (unsigned short)(0x7e00u)))):__builtin_bit_cast(unsigned short, (__fp16)(val0[1])))));
    __fp16 alu5 = __builtin_bit_cast(__fp16, (unsigned short)(((((__fp16)(({float z = ((__fp16)(0.0f)); __asm__("" : "+x"(z)); z;})))<val0[1])?__builtin_bit_cast(unsigned short, (__fp16)(val0[1])):__builtin_bit_cast(unsigned short, (__fp16)(((__fp16)(__builtin_inff())))))));
    __fp16 alu6 = __builtin_bit_cast(__fp16, (unsigned short)(((val0[1]<((__fp16)(1.0f)))?__builtin_bit_cast(unsigned short, (__fp16)(((__fp16)(-__builtin_inff())))):__builtin_bit_cast(unsigned short, (__fp16)(val0[1])))));
    __fp16 alu7 = __builtin_bit_cast(__fp16, (unsigned short)(((val0[2]<((__fp16)(-1.0f)))?__builtin_bit_cast(unsigned short, (__fp16)(__builtin_bit_cast(__fp16, (unsigned short)(0x7e00u)))):__builtin_bit_cast(unsigned short, (__fp16)(val0[2])))));
    __fp16 alu8 = __builtin_bit_cast(__fp16, (unsigned short)(((((__fp16)(({float z = ((__fp16)(0.0f)); __asm__("" : "+x"(z)); z;})))<val0[2])?__builtin_bit_cast(unsigned short, (__fp16)(val0[2])):__builtin_bit_cast(unsigned short, (__fp16)(((__fp16)(__builtin_inff())))))));
    __fp16 alu9 = __builtin_bit_cast(__fp16, (unsigned short)(((val0[2]<((__fp16)(1.0f)))?__builtin_bit_cast(unsigned short, (__fp16)(((__fp16)(-__builtin_inff())))):__builtin_bit_cast(unsigned short, (__fp16)(val0[2])))));
    __fp16 alu10 = __builtin_bit_cast(__fp16, (unsigned short)(((val0[3]<((__fp16)(-1.0f)))?__builtin_bit_cast(unsigned short, (__fp16)(__builtin_bit_cast(__fp16, (unsigned short)(0x7e00u)))):__builtin_bit_cast(unsigned short, (__fp16)(val0[3])))));
    __fp16 alu11 = __builtin_bit_cast(__fp16, (unsigned short)(((((__fp16)(({float z = ((__fp16)(0.0f)); __asm__("" : "+x"(z)); z;})))<val0[3])?__builtin_bit_cast(unsigned short, (__fp16)(val0[3])):__builtin_bit_cast(unsigned short, (__fp16)(((__fp16)(__builtin_inff())))))));
    __fp16 alu12 = __builtin_bit_cast(__fp16, (unsigned short)(((val0[3]<((__fp16)(1.0f)))?__builtin_bit_cast(unsigned short, (__fp16)(((__fp16)(-__builtin_inff())))):__builtin_bit_cast(unsigned short, (__fp16)(val0[3])))));
    *((__fp164*)((data0_16+alu0))) = (__fp164){((__fp16)((((__fp16)((alu2+alu1)))+alu3))),((__fp16)((((__fp16)((alu5+alu4)))+alu6))),((__fp16)((((__fp16)((alu8+alu7)))+alu9))),((__fp16)((((__fp16)((alu11+alu10)))+alu12)))};
  }
}

