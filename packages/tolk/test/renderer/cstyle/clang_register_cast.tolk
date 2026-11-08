# tolk D151 golden=9ce4597160e2240127bb336edde85926

__attribute__((min_vector_width(512))) void register_cast(unsigned long* restrict data0_1) {
  unsigned int buf0[2];
  *(buf0+0) = 1u;
  *(buf0+1) = 2u;
  *(data0_1+0) = (*((unsigned long*)(((unsigned long)((buf0+0))))));
}

