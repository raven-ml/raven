# tolk D151 golden=cafc1347af115f25edd3232cf6192c1d

__attribute__((min_vector_width(512))) void named(float* restrict input_tile_4, float* restrict output_tile_4, const int for_) {
  for (int Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    float val0 = (*(input_tile_4+Lidx0));
    *(output_tile_4+Lidx0) = (val0+((float)(for_)));
  }
}

