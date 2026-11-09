# tolk D151 D342 golden=6e8d064aa0c3ffe7e7f6b7253e4e189f

__attribute__((min_vector_width(512))) void selected_half(__fp16* restrict data0_2, __fp16* restrict data1_1, float* restrict data2_1, _Bool* restrict data3_1, __fp16* restrict data4_2) {
  _Bool val0 = (*(data3_1+0));
  __fp16 val1 = (*(data1_1+0));
  __fp16 val2 = __builtin_bit_cast(__fp16, (unsigned short)((val0?__builtin_bit_cast(unsigned short, (__fp16)(*(data4_2+1))):__builtin_bit_cast(unsigned short, (__fp16)(((__fp16)(0.0f)))))));
  float val3 = (*(data2_1+0));
  __fp16 alu0 = __builtin_bit_cast(__fp16, (unsigned short)((val0?__builtin_bit_cast(unsigned short, (__fp16)(val1)):__builtin_bit_cast(unsigned short, (__fp16)(((__fp16)(val3)))))));
  *(data0_2+0) = alu0;
  *(data0_2+1) = val2;
}

