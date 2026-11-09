# tolk D151 D342 golden=93401f18b7c9d2bb2018f0da5af8d701

extern void sqrt();
__attribute__((min_vector_width(512))) void call_ret(int* restrict data0_1) {
  double alu0 = ((double(*)(double))(sqrt))((double)(16.0));
  *(data0_1+0) = (((int)(alu0))<<1);
}

