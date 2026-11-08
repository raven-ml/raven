# tolk D151 golden=96f9f943a0a017342f8a11d0c402898c

__attribute__((min_vector_width(512))) void packed(signed char* restrict data0_4) {
  unsigned int val0 = (*((unsigned int*)(((unsigned int*)((data0_4+0))))));
  *(data0_4+0) = (val0&4294967040u);
}

