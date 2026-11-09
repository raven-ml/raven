# tolk D151 golden=3222893066d1c40d5479fe9b1351fb2c

__attribute__((min_vector_width(512))) void E_28n2A429(float* restrict data0_256, float* restrict data1_256, const int n_) {
  for (int Lidx0 = 0; Lidx0 < (n_<<2); Lidx0++) {
    float val0 = (*(data1_256+Lidx0));
    *(data0_256+Lidx0) = (val0+1.0f);
  }
}

