# tolk D141 D151 golden=88ece6eb2382531f84217f62dfb035e4
typedef long long4 __attribute__((aligned(8),ext_vector_type(4)));
typedef double double4 __attribute__((aligned(8),ext_vector_type(4)));
__attribute__((min_vector_width(512))) void cast(double* restrict data0_4, long* restrict data1_4) {
  long4 val0 = (*((long4*)((data1_4+0))));
  double4 cast0 = __builtin_convertvector(val0, double4);
  *((double4*)((data0_4+0))) = cast0;
}

