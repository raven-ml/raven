# tolk D151 golden=e23859f2231ea9778bf235c574c9b78e

__attribute__((min_vector_width(512))) void scalars(long* restrict data0_4, const int start_, const long offset_) {
  for (int Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    *(data0_4+Lidx0) = (((long)((Lidx0+start_)))+offset_);
  }
}

