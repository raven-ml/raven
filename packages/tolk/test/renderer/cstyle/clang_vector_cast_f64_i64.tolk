# tolk D141 D151 golden=546d2d4e3117fea0dda6df7799126cb1
typedef double double4 __attribute__((aligned(8),ext_vector_type(4)));
typedef long long4 __attribute__((aligned(8),ext_vector_type(4)));
__attribute__((min_vector_width(512))) void cast(long* restrict data0_4, double* restrict data1_4) {
  double4 val0 = (*((double4*)((data1_4+0))));
  long4 cast0 = __builtin_convertvector(val0, long4);
  *((long4*)((data0_4+0))) = cast0;
}

