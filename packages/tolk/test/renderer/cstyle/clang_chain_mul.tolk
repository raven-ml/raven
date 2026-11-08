# tolk D151 golden=ca90ec4eb17b1a9bb8e2e0433f5cbb3d

__attribute__((min_vector_width(512))) void E(float* restrict data0_1, float* restrict data1_1, float* restrict data2_1, float* restrict data3_1, float* restrict data4_1, float* restrict data5_1, float* restrict data6_1) {
  float val0 = (*(data1_1+0));
  float val1 = (*(data2_1+0));
  float val2 = (*(data3_1+0));
  float val3 = (*(data4_1+0));
  float val4 = (*(data5_1+0));
  float val5 = (*(data6_1+0));
  *(data0_1+0) = (val0*val1*val2*val3*val4*val5);
}

