# tolk D141 D151 golden=223a89f9f1ecce8648c192b50f979732
typedef double double4 __attribute__((aligned(8),ext_vector_type(4)));
__attribute__((min_vector_width(512))) void trunc(double* restrict data0_4, double* restrict data1_4) {
  double4 val0 = (*((double4*)((data1_4+0))));
  *((double4*)((data0_4+0))) = __builtin_elementwise_trunc(val0);
}

