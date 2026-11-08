# tolk D151 golden=a4518b11dc3f39c800018fb14a495be2

__attribute__((min_vector_width(512))) void test(int* restrict data0_16) {
  for (int Lidx0 = 0; Lidx0 < 16; Lidx0++) {
    if ((Lidx0<8)) {
      *(data0_16+Lidx0) = Lidx0;
    }
  }
}

