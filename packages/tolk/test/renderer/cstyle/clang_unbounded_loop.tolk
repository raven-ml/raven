# tolk D151 golden=64d503a6d9e025d116eddcfa27e22476

__attribute__((min_vector_width(512))) void unbounded(int* restrict data0_1) {
  for (;;) {
    int val0 = (*(data0_1+0));
    *(data0_1+0) = (val0+1);
    int val1 = (*(data0_1+0));
    if (!((val1<10))) { break; }
  }
}

