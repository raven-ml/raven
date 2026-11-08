# tolk D151 golden=9018667116c96848929c07b3737d69b6

__attribute__((min_vector_width(512))) void test(int* restrict data0_1, int* restrict data1_1) {
  int val0 = (*(data1_1+0));
  int alu0 = ((val0<-2147483647)?-2147483647:val0);
  *(data0_1+0) = alu0;
}

