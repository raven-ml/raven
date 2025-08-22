#include "nx_c_shared.h"

// Parallel iteration helper for non-contiguous arrays
// This function iterates over all elements except the outermost dimension
static inline void iterate_inner_dims(
    const ndarray_t *x, const ndarray_t *y, const ndarray_t *z, long outer_idx,
    void (*op)(void *, void *, void *, long, long, long), void *x_data,
    void *y_data, void *z_data) {
  if (x->ndim == 1) {
    // Base case: 1D array, just apply the operation
    long x_off = x->offset + outer_idx * x->strides[0];
    long y_off = y ? (y->offset + outer_idx * y->strides[0]) : 0;
    long z_off = z->offset + outer_idx * z->strides[0];
    op(x_data, y_data, z_data, x_off, y_off, z_off);
  } else if (x->ndim == 2) {
    // Common case: 2D array
    long x_base = x->offset + outer_idx * x->strides[0];
    long y_base = y ? (y->offset + outer_idx * y->strides[0]) : 0;
    long z_base = z->offset + outer_idx * z->strides[0];

    for (long j = 0; j < x->shape[1]; j++) {
      long x_off = x_base + j * x->strides[1];
      long y_off = y ? (y_base + j * y->strides[1]) : 0;
      long z_off = z_base + j * z->strides[1];
      op(x_data, y_data, z_data, x_off, y_off, z_off);
    }
  } else {
    // General case: use iterator for inner dimensions
    nd_iterator_t it;
    it.ndim = x->ndim - 1;
    it.shape = x->shape + 1;  // Skip first dimension
    it.x_strides = x->strides + 1;
    it.y_strides = y ? (y->strides + 1) : NULL;
    it.z_strides = z->strides + 1;
    it.x_offset = x->offset + outer_idx * x->strides[0];
    it.y_offset = y ? (y->offset + outer_idx * y->strides[0]) : 0;
    it.z_offset = z->offset + outer_idx * z->strides[0];

    if (it.ndim <= 16) {
      it.heap_indices = NULL;
      for (int i = 0; i < it.ndim; i++) it.indices[i] = 0;
    } else {
      it.heap_indices = calloc(it.ndim, sizeof(long));
      if (it.heap_indices == NULL) {
        caml_failwith("iterate_inner_dims: failed to allocate memory");
      }
    }

    do {
      long x_off, y_off, z_off;
      nd_iterator_get_offsets(&it, &x_off, y ? &y_off : NULL, &z_off);
      op(x_data, y_data, z_data, x_off, y_off, z_off);
    } while (nd_iterator_next(&it));

    nd_iterator_destroy(&it);
  }
}

// Helper functions for Int4/UInt4 packed value access
static inline uint8_t get_int4_value(const void *data, long index) {
  const uint8_t *bytes = (const uint8_t *)data;
  long byte_idx = index / 2;
  int nibble_idx = index % 2;
  uint8_t byte = bytes[byte_idx];
  if (nibble_idx == 0) {
    // Lower nibble
    return byte & 0x0F;
  } else {
    // Upper nibble
    return (byte >> 4) & 0x0F;
  }
}

static inline void set_int4_value(void *data, long index, uint8_t value) {
  uint8_t *bytes = (uint8_t *)data;
  long byte_idx = index / 2;
  int nibble_idx = index % 2;
  uint8_t byte = bytes[byte_idx];
  if (nibble_idx == 0) {
    // Set lower nibble
    bytes[byte_idx] = (byte & 0xF0) | (value & 0x0F);
  } else {
    // Set upper nibble
    bytes[byte_idx] = (byte & 0x0F) | ((value & 0x0F) << 4);
  }
}

static void nx_c_copy_int4(const ndarray_t *x, ndarray_t *z) {
  long total = total_elements(x);
  if (total == 0) return;

  nd_iterator_t it;
  nd_iterator_init(&it, x, NULL, z);

  do {
    long x_off, z_off;
    nd_iterator_get_offsets(&it, &x_off, NULL, &z_off);
    uint8_t value = get_int4_value(x->data, x->offset + x_off);
    set_int4_value(z->data, z->offset + z_off, value);
  } while (nd_iterator_next(&it));

  nd_iterator_destroy(&it);
}

static void nx_c_generic_copy(const ndarray_t *x, ndarray_t *z,
                              size_t elem_size) {
  long total = total_elements(x);
  if (total == 0) return;

  if (is_c_contiguous(x) && is_c_contiguous(z)) {
    memcpy((char *)z->data + z->offset * elem_size,
           (char *)x->data + x->offset * elem_size, total * elem_size);
  } else {
    // Use optimized iterator for non-contiguous case
    nd_iterator_t it;
    nd_iterator_init(&it, x, NULL, z);

    char *x_data = (char *)x->data;
    char *z_data = (char *)z->data;

    do {
      long x_off, z_off;
      nd_iterator_get_offsets(&it, &x_off, NULL, &z_off);
      memcpy(z_data + z_off * elem_size, x_data + x_off * elem_size, elem_size);
    } while (nd_iterator_next(&it));

    nd_iterator_destroy(&it);
  }
}

// Cast operation kernel for use with parallel iterator
#define DEFINE_CAST_OP_KERNEL(T_SRC, T_DST, CAST_OP)                    \
  static void nx_c_cast_##T_SRC##_to_##T_DST##_kernel(                  \
      void *x_data, void *y_data, void *z_data, long x_off, long y_off, \
      long z_off) {                                                     \
    (void)y_data;                                                       \
    (void)y_off; /* Unused for cast ops */                              \
    T_SRC *x = (T_SRC *)x_data;                                         \
    T_DST *z = (T_DST *)z_data;                                         \
    z[z_off] = CAST_OP(x[x_off]);                                       \
  }

#define DEFINE_CAST_OP(T_SRC, T_DST, CAST_OP)                               \
  DEFINE_CAST_OP_KERNEL(T_SRC, T_DST, CAST_OP)                              \
  static void nx_c_cast_##T_SRC##_to_##T_DST(const ndarray_t *x,            \
                                             ndarray_t *z) {                \
    T_SRC *x_data = (T_SRC *)x->data;                                       \
    T_DST *z_data = (T_DST *)z->data;                                       \
    long total = total_elements(x);                                         \
    if (total == 0) return;                                                 \
                                                                            \
    if (is_c_contiguous(x) && is_c_contiguous(z)) {                         \
      T_SRC *x_ptr = x_data + x->offset;                                    \
      T_DST *z_ptr = z_data + z->offset;                                    \
      _Pragma("omp parallel for simd") for (long i = 0; i < total; i++) {   \
        z_ptr[i] = CAST_OP(x_ptr[i]);                                       \
      }                                                                     \
    } else if (x->ndim > 0) {                                               \
      /* Parallelize over the outermost dimension */                        \
      _Pragma("omp parallel for") for (long i = 0; i < x->shape[0]; i++) {  \
        iterate_inner_dims(x, NULL, z, i,                                   \
                           nx_c_cast_##T_SRC##_to_##T_DST##_kernel, x_data, \
                           NULL, z_data);                                   \
      }                                                                     \
    } else {                                                                \
      /* Scalar case */                                                     \
      z_data[z->offset] = CAST_OP(x_data[x->offset]);                       \
    }                                                                       \
  }

DEFINE_CAST_OP(float, double, (double))
DEFINE_CAST_OP(double, float, (float))
DEFINE_CAST_OP(int32_t, float, (float))
DEFINE_CAST_OP(int64_t, float, (float))
DEFINE_CAST_OP(float, int32_t, (int32_t))
DEFINE_CAST_OP(float, int64_t, (int64_t))
DEFINE_CAST_OP(double, int32_t, (int32_t))
DEFINE_CAST_OP(double, int64_t, (int64_t))
DEFINE_CAST_OP(int32_t, double, (double))
DEFINE_CAST_OP(int64_t, double, (double))
DEFINE_CAST_OP(int16_t, float, (float))
DEFINE_CAST_OP(int16_t, double, (double))
DEFINE_CAST_OP(int16_t, int32_t, (int32_t))
DEFINE_CAST_OP(int16_t, int64_t, (int64_t))
DEFINE_CAST_OP(int32_t, int16_t, (int16_t))
DEFINE_CAST_OP(int64_t, int16_t, (int16_t))
DEFINE_CAST_OP(float, int16_t, (int16_t))
DEFINE_CAST_OP(double, int16_t, (int16_t))
DEFINE_CAST_OP(uint8_t, int32_t, (int32_t))
DEFINE_CAST_OP(uint8_t, float, (float))
DEFINE_CAST_OP(uint8_t, double, (double))
DEFINE_CAST_OP(int32_t, uint8_t, (uint8_t))
DEFINE_CAST_OP(float, uint8_t, (uint8_t))
DEFINE_CAST_OP(double, uint8_t, (uint8_t))

// int8_t conversions
DEFINE_CAST_OP(int8_t, int16_t, (int16_t))
DEFINE_CAST_OP(int8_t, uint8_t, (uint8_t))
DEFINE_CAST_OP(int8_t, uint16_t, (uint16_t))
DEFINE_CAST_OP(int8_t, int32_t, (int32_t))
DEFINE_CAST_OP(int8_t, int64_t, (int64_t))
DEFINE_CAST_OP(int8_t, float, (float))
DEFINE_CAST_OP(int8_t, double, (double))
DEFINE_CAST_OP(int16_t, int8_t, (int8_t))
DEFINE_CAST_OP(uint8_t, int8_t, (int8_t))
DEFINE_CAST_OP(uint16_t, int8_t, (int8_t))
DEFINE_CAST_OP(int32_t, int8_t, (int8_t))
DEFINE_CAST_OP(int64_t, int8_t, (int8_t))
DEFINE_CAST_OP(float, int8_t, (int8_t))
DEFINE_CAST_OP(double, int8_t, (int8_t))

// uint16_t conversions
DEFINE_CAST_OP(uint16_t, int16_t, (int16_t))
DEFINE_CAST_OP(uint16_t, uint8_t, (uint8_t))
DEFINE_CAST_OP(uint16_t, int32_t, (int32_t))
DEFINE_CAST_OP(uint16_t, int64_t, (int64_t))
DEFINE_CAST_OP(uint16_t, float, (float))
DEFINE_CAST_OP(uint16_t, double, (double))
DEFINE_CAST_OP(int16_t, uint16_t, (uint16_t))
DEFINE_CAST_OP(uint8_t, uint16_t, (uint16_t))
DEFINE_CAST_OP(int32_t, uint16_t, (uint16_t))
DEFINE_CAST_OP(int64_t, uint16_t, (uint16_t))
DEFINE_CAST_OP(float, uint16_t, (uint16_t))
DEFINE_CAST_OP(double, uint16_t, (uint16_t))

// Additional integer conversions
DEFINE_CAST_OP(int64_t, int32_t, (int32_t))
DEFINE_CAST_OP(int32_t, int64_t, (int64_t))
DEFINE_CAST_OP(uint8_t, int16_t, (int16_t))
DEFINE_CAST_OP(uint8_t, int64_t, (int64_t))
DEFINE_CAST_OP(int16_t, uint8_t, (uint8_t))
DEFINE_CAST_OP(int64_t, uint8_t, (uint8_t))

// intnat conversions
DEFINE_CAST_OP(intnat, int8_t, (int8_t))
DEFINE_CAST_OP(intnat, uint8_t, (uint8_t))
DEFINE_CAST_OP(intnat, int16_t, (int16_t))
DEFINE_CAST_OP(intnat, uint16_t, (uint16_t))
DEFINE_CAST_OP(intnat, int32_t, (int32_t))
DEFINE_CAST_OP(intnat, int64_t, (int64_t))
DEFINE_CAST_OP(intnat, float, (float))
DEFINE_CAST_OP(intnat, double, (double))
DEFINE_CAST_OP(int8_t, intnat, (intnat))
DEFINE_CAST_OP(uint8_t, intnat, (intnat))
DEFINE_CAST_OP(int16_t, intnat, (intnat))
DEFINE_CAST_OP(uint16_t, intnat, (intnat))
DEFINE_CAST_OP(int32_t, intnat, (intnat))
DEFINE_CAST_OP(int64_t, intnat, (intnat))
DEFINE_CAST_OP(float, intnat, (intnat))
DEFINE_CAST_OP(double, intnat, (intnat))

// float16 conversions
DEFINE_CAST_OP(float16_t, int8_t, (int8_t))
DEFINE_CAST_OP(float16_t, uint8_t, (uint8_t))
DEFINE_CAST_OP(float16_t, int16_t, (int16_t))
DEFINE_CAST_OP(float16_t, uint16_t, (uint16_t))
DEFINE_CAST_OP(float16_t, int32_t, (int32_t))
DEFINE_CAST_OP(float16_t, int64_t, (int64_t))
DEFINE_CAST_OP(float16_t, intnat, (intnat))
DEFINE_CAST_OP(float16_t, float, (float))
DEFINE_CAST_OP(float16_t, double, (double))
DEFINE_CAST_OP(float16_t, qint8_t, (qint8_t))
DEFINE_CAST_OP(float16_t, quint8_t, (quint8_t))

// Conversions to float16
DEFINE_CAST_OP(int8_t, float16_t, (float16_t))
DEFINE_CAST_OP(uint8_t, float16_t, (float16_t))
DEFINE_CAST_OP(int16_t, float16_t, (float16_t))
DEFINE_CAST_OP(uint16_t, float16_t, (float16_t))
DEFINE_CAST_OP(int32_t, float16_t, (float16_t))
DEFINE_CAST_OP(int64_t, float16_t, (float16_t))
DEFINE_CAST_OP(intnat, float16_t, (float16_t))
DEFINE_CAST_OP(float, float16_t, (float16_t))
DEFINE_CAST_OP(double, float16_t, (float16_t))
DEFINE_CAST_OP(qint8_t, float16_t, (float16_t))
DEFINE_CAST_OP(quint8_t, float16_t, (float16_t))

// qint8 conversions
DEFINE_CAST_OP(qint8_t, int8_t, (int8_t))
DEFINE_CAST_OP(qint8_t, uint8_t, (uint8_t))
DEFINE_CAST_OP(qint8_t, int16_t, (int16_t))
DEFINE_CAST_OP(qint8_t, uint16_t, (uint16_t))
DEFINE_CAST_OP(qint8_t, int32_t, (int32_t))
DEFINE_CAST_OP(qint8_t, int64_t, (int64_t))
DEFINE_CAST_OP(qint8_t, intnat, (intnat))
DEFINE_CAST_OP(qint8_t, float, (float))
DEFINE_CAST_OP(qint8_t, double, (double))
DEFINE_CAST_OP(qint8_t, quint8_t, (quint8_t))

// Conversions to qint8
DEFINE_CAST_OP(int8_t, qint8_t, (qint8_t))
DEFINE_CAST_OP(uint8_t, qint8_t, (qint8_t))
DEFINE_CAST_OP(int16_t, qint8_t, (qint8_t))
DEFINE_CAST_OP(uint16_t, qint8_t, (qint8_t))
DEFINE_CAST_OP(int32_t, qint8_t, (qint8_t))
DEFINE_CAST_OP(int64_t, qint8_t, (qint8_t))
DEFINE_CAST_OP(intnat, qint8_t, (qint8_t))
DEFINE_CAST_OP(float, qint8_t, (qint8_t))
DEFINE_CAST_OP(double, qint8_t, (qint8_t))
DEFINE_CAST_OP(quint8_t, qint8_t, (qint8_t))

// quint8 conversions
DEFINE_CAST_OP(quint8_t, int8_t, (int8_t))
DEFINE_CAST_OP(quint8_t, uint8_t, (uint8_t))
DEFINE_CAST_OP(quint8_t, int16_t, (int16_t))
DEFINE_CAST_OP(quint8_t, uint16_t, (uint16_t))
DEFINE_CAST_OP(quint8_t, int32_t, (int32_t))
DEFINE_CAST_OP(quint8_t, int64_t, (int64_t))
DEFINE_CAST_OP(quint8_t, intnat, (intnat))
DEFINE_CAST_OP(quint8_t, float, (float))
DEFINE_CAST_OP(quint8_t, double, (double))

// Conversions to quint8
DEFINE_CAST_OP(int8_t, quint8_t, (quint8_t))
DEFINE_CAST_OP(uint8_t, quint8_t, (quint8_t))
DEFINE_CAST_OP(int16_t, quint8_t, (quint8_t))
DEFINE_CAST_OP(uint16_t, quint8_t, (quint8_t))
DEFINE_CAST_OP(int32_t, quint8_t, (quint8_t))
DEFINE_CAST_OP(int64_t, quint8_t, (quint8_t))
DEFINE_CAST_OP(intnat, quint8_t, (quint8_t))
DEFINE_CAST_OP(float, quint8_t, (quint8_t))
DEFINE_CAST_OP(double, quint8_t, (quint8_t))

// Unary operation kernel for use with parallel iterator
#define DEFINE_UNARY_OP_KERNEL(name, T, OP)                                    \
  static void nx_c_##name##_##T##_kernel(void *x_data, void *y_data,           \
                                         void *z_data, long x_off, long y_off, \
                                         long z_off) {                         \
    (void)y_data;                                                              \
    (void)y_off; /* Unused for unary ops */                                    \
    T *x = (T *)x_data;                                                        \
    T *z = (T *)z_data;                                                        \
    z[z_off] = OP(x[x_off]);                                                   \
  }

#define DEFINE_UNARY_OP(name, T, OP)                                          \
  DEFINE_UNARY_OP_KERNEL(name, T, OP)                                         \
  static void nx_c_##name##_##T(const ndarray_t *x, ndarray_t *z) {           \
    T *x_data = (T *)x->data;                                                 \
    T *z_data = (T *)z->data;                                                 \
    long total = total_elements(x);                                           \
    if (total == 0) return;                                                   \
                                                                              \
    if (is_c_contiguous(x) && is_c_contiguous(z)) {                           \
      T *x_ptr = x_data + x->offset;                                          \
      T *z_ptr = z_data + z->offset;                                          \
      _Pragma("omp parallel for simd") for (long i = 0; i < total; i++) {     \
        z_ptr[i] = OP(x_ptr[i]);                                              \
      }                                                                       \
    } else if (x->ndim > 0) {                                                 \
      /* Parallelize over the outermost dimension */                          \
      _Pragma("omp parallel for") for (long i = 0; i < x->shape[0]; i++) {    \
        iterate_inner_dims(x, NULL, z, i, nx_c_##name##_##T##_kernel, x_data, \
                           NULL, z_data);                                     \
      }                                                                       \
    } else {                                                                  \
      /* Scalar case */                                                       \
      z_data[z->offset] = OP(x_data[x->offset]);                              \
    }                                                                         \
  }

// -- Negation Implementations --
DEFINE_UNARY_OP(neg_generic, float, -)
DEFINE_UNARY_OP(neg_generic, double, -)
DEFINE_UNARY_OP(neg_generic, c32_t, -)
DEFINE_UNARY_OP(neg_generic, c64_t, -)

static void nx_c_neg_float(const ndarray_t *x, ndarray_t *z) {
  if (is_c_contiguous(x) && is_c_contiguous(z)) {
    long total = total_elements(x);
    float *x_ptr = (float *)x->data + x->offset;
    float *z_ptr = (float *)z->data + z->offset;

    _Pragma("omp parallel for simd") for (long i = 0; i < total; i++) {
      z_ptr[i] = -x_ptr[i];
    }
  } else {
    nx_c_neg_generic_float(x, z);
  }
}
static void nx_c_neg_double(const ndarray_t *x, ndarray_t *z) {
  if (is_c_contiguous(x) && is_c_contiguous(z)) {
    long total = total_elements(x);
    double *x_ptr = (double *)x->data + x->offset;
    double *z_ptr = (double *)z->data + z->offset;

    _Pragma("omp parallel for simd") for (long i = 0; i < total; i++) {
      z_ptr[i] = -x_ptr[i];
    }
  } else {
    nx_c_neg_generic_double(x, z);
  }
}
DEFINE_UNARY_OP(neg, int8_t, -)
DEFINE_UNARY_OP(neg, int16_t, -)
DEFINE_UNARY_OP(neg, int32_t, -)
DEFINE_UNARY_OP(neg, int64_t, -)
DEFINE_UNARY_OP(neg, intnat, -)  // Covers both int and nativeint
DEFINE_UNARY_OP(neg, c32_t, -)
DEFINE_UNARY_OP(neg, c64_t, -)
// float16 and quantized negation
DEFINE_UNARY_OP(neg, float16_t, -)
DEFINE_UNARY_OP(neg, qint8_t, -)

// Extended types negation
DEFINE_UNARY_OP(neg, bool_t, !)
static inline bfloat16_t neg_bf16(bfloat16_t x) {
  return float_to_bfloat16(-bfloat16_to_float(x));
}
DEFINE_UNARY_OP(neg, bfloat16_t, neg_bf16)
static inline fp8_e4m3_t neg_fp8_e4m3(fp8_e4m3_t x) {
  return float_to_fp8_e4m3(-fp8_e4m3_to_float(x));
}
DEFINE_UNARY_OP(neg, fp8_e4m3_t, neg_fp8_e4m3)
static inline fp8_e5m2_t neg_fp8_e5m2(fp8_e5m2_t x) {
  return float_to_fp8_e5m2(-fp8_e5m2_to_float(x));
}
DEFINE_UNARY_OP(neg, fp8_e5m2_t, neg_fp8_e5m2)
static inline complex16_t neg_c16(complex16_t x) {
  complex16_t result;
  result.real = float_to_bfloat16(-bfloat16_to_float(x.real));
  result.imag = float_to_bfloat16(-bfloat16_to_float(x.imag));
  return result;
}
DEFINE_UNARY_OP(neg, complex16_t, neg_c16)

// -- Other Unary Implementations --
DEFINE_UNARY_OP(sqrt, float, sqrtf)
DEFINE_UNARY_OP(sqrt, double, sqrt)
DEFINE_UNARY_OP(sqrt, c32_t, csqrtf)
DEFINE_UNARY_OP(sqrt, c64_t, csqrt)
// float16 sqrt (using cast through float)
static inline float16_t sqrtf16(float16_t x) { return (float16_t)sqrtf((float)x); }
DEFINE_UNARY_OP(sqrt, float16_t, sqrtf16)

// Extended types sqrt
static inline bfloat16_t sqrt_bf16(bfloat16_t x) {
  return float_to_bfloat16(sqrtf(bfloat16_to_float(x)));
}
DEFINE_UNARY_OP(sqrt, bfloat16_t, sqrt_bf16)
static inline fp8_e4m3_t sqrt_fp8_e4m3(fp8_e4m3_t x) {
  return float_to_fp8_e4m3(sqrtf(fp8_e4m3_to_float(x)));
}
DEFINE_UNARY_OP(sqrt, fp8_e4m3_t, sqrt_fp8_e4m3)
static inline fp8_e5m2_t sqrt_fp8_e5m2(fp8_e5m2_t x) {
  return float_to_fp8_e5m2(sqrtf(fp8_e5m2_to_float(x)));
}
DEFINE_UNARY_OP(sqrt, fp8_e5m2_t, sqrt_fp8_e5m2)
static inline complex16_t sqrt_c16(complex16_t x) {
  float complex fc = bfloat16_to_float(x.real) + bfloat16_to_float(x.imag) * I;
  float complex result_c = csqrtf(fc);
  complex16_t result;
  result.real = float_to_bfloat16(crealf(result_c));
  result.imag = float_to_bfloat16(cimagf(result_c));
  return result;
}
DEFINE_UNARY_OP(sqrt, complex16_t, sqrt_c16)


DEFINE_UNARY_OP(sin, float, sinf)
DEFINE_UNARY_OP(sin, double, sin)
DEFINE_UNARY_OP(sin, c32_t, csinf)
DEFINE_UNARY_OP(sin, c64_t, csin)
// float16 sin
static inline float16_t sinf16(float16_t x) { return (float16_t)sinf((float)x); }
DEFINE_UNARY_OP(sin, float16_t, sinf16)

DEFINE_UNARY_OP(exp2, float, exp2f)
DEFINE_UNARY_OP(exp2, double, exp2)
// Note: exp2 is not in the C standard for complex, must be built from exp
static inline c32_t cexp2f_op(c32_t v) { return cexpf(v * M_LN2); }
static inline c64_t cexp2_op(c64_t v) { return cexp(v * M_LN2); }
DEFINE_UNARY_OP(exp2, c32_t, cexp2f_op)
DEFINE_UNARY_OP(exp2, c64_t, cexp2_op)
// float16 exp2
static inline float16_t exp2f16(float16_t x) { return (float16_t)exp2f((float)x); }
DEFINE_UNARY_OP(exp2, float16_t, exp2f16)

DEFINE_UNARY_OP(log2, float, log2f)
DEFINE_UNARY_OP(log2, double, log2)
// Note: log2 is not in the C standard for complex, must be built from log
static inline c32_t clog2f_op(c32_t v) { return clogf(v) / M_LN2; }
static inline c64_t clog2_op(c64_t v) { return clog(v) / M_LN2; }
DEFINE_UNARY_OP(log2, c32_t, clog2f_op)
DEFINE_UNARY_OP(log2, c64_t, clog2_op)
// float16 log2
static inline float16_t log2f16(float16_t x) { return (float16_t)log2f((float)x); }
DEFINE_UNARY_OP(log2, float16_t, log2f16)

static inline float recipf_op(float v) { return 1.0f / v; }
static inline double recip_op(double v) { return 1.0 / v; }
static inline c32_t crecipf_op(c32_t v) { return 1.0f / v; }
static inline c64_t crecip_op(c64_t v) { return 1.0 / v; }
DEFINE_UNARY_OP(recip, float, recipf_op)
DEFINE_UNARY_OP(recip, double, recip_op)
DEFINE_UNARY_OP(recip, c32_t, crecipf_op)
DEFINE_UNARY_OP(recip, c64_t, crecip_op)
// float16 recip
static inline float16_t recipf16(float16_t x) { return (float16_t)(1.0f / (float)x); }
DEFINE_UNARY_OP(recip, float16_t, recipf16)

#define ADD_OP(a, b) ((a) + (b))
#define SUB_OP(a, b) ((a) - (b))
#define MUL_OP(a, b) ((a) * (b))
#define DIV_OP(a, b) ((a) / (b))
#define MOD_OP(a, b) ((a) % (b))
#define MAX_OP(a, b) ((a) > (b) ? (a) : (b))
#define POW_OP(a, b) pow(a, b)
#define FMAX_OP(a, b) fmax(a, b)
#define MAX_NAN_OP(a, b) (isnan(a) || isnan(b) ? NAN : fmax(a, b))
#define MIN_NAN_OP(a, b) (isnan(a) || isnan(b) ? NAN : fmin(a, b))
#define FMOD_OP(a, b) fmod(a, b)
#define XOR_OP(a, b) ((a) ^ (b))
#define OR_OP(a, b) ((a) | (b))
#define AND_OP(a, b) ((a) & (b))
#define CMPLT_OP(a, b) ((a) < (b))
#define CMPNE_OP(a, b) ((a) != (b))

static inline int32_t idiv_op_i32(int32_t a, int32_t b) {
  return b == 0 ? 0 : a / b;
}
static inline int64_t idiv_op_i64(int64_t a, int64_t b) {
  return b == 0 ? 0 : a / b;
}

// Binary operation kernel for use with parallel iterator
#define DEFINE_BINARY_OP_KERNEL(name, T, OP)                                   \
  static void nx_c_##name##_##T##_kernel(void *x_data, void *y_data,           \
                                         void *z_data, long x_off, long y_off, \
                                         long z_off) {                         \
    T *x = (T *)x_data;                                                        \
    T *y = (T *)y_data;                                                        \
    T *z = (T *)z_data;                                                        \
    z[z_off] = OP(x[x_off], y[y_off]);                                         \
  }

// Comparison operation kernel that returns uint8_t
#define DEFINE_COMPARE_OP_KERNEL(name, T, OP)                                  \
  static void nx_c_##name##_##T##_kernel(void *x_data, void *y_data,           \
                                         void *z_data, long x_off, long y_off, \
                                         long z_off) {                         \
    T *x = (T *)x_data;                                                        \
    T *y = (T *)y_data;                                                        \
    uint8_t *z = (uint8_t *)z_data;                                            \
    z[z_off] = OP(x[x_off], y[y_off]) ? 1 : 0;                                 \
  }

// The DEFINE_BINARY_OP macro with parallel support for non-contiguous arrays
#define DEFINE_BINARY_OP(name, T, OP)                                      \
  DEFINE_BINARY_OP_KERNEL(name, T, OP)                                     \
  static void nx_c_##name##_##T(const ndarray_t *x, const ndarray_t *y,    \
                                ndarray_t *z) {                            \
    T *x_data = (T *)x->data;                                              \
    T *y_data = (T *)y->data;                                              \
    T *z_data = (T *)z->data;                                              \
    long total = total_elements(x);                                        \
    if (total == 0) return;                                                \
    if (is_c_contiguous(x) && is_c_contiguous(y) && is_c_contiguous(z)) {  \
      T *x_ptr = x_data + x->offset;                                       \
      T *y_ptr = y_data + y->offset;                                       \
      T *z_ptr = z_data + z->offset;                                       \
      _Pragma("omp parallel for simd") for (long i = 0; i < total; i++) {  \
        z_ptr[i] = OP(x_ptr[i], y_ptr[i]);                                 \
      }                                                                    \
    } else if (x->ndim > 0) {                                              \
      /* Parallelize over the outermost dimension */                       \
      _Pragma("omp parallel for") for (long i = 0; i < x->shape[0]; i++) { \
        iterate_inner_dims(x, y, z, i, nx_c_##name##_##T##_kernel, x_data, \
                           y_data, z_data);                                \
      }                                                                    \
    } else {                                                               \
      /* Scalar case */                                                    \
      z_data[z->offset] = OP(x_data[x->offset], y_data[y->offset]);        \
    }                                                                      \
  }

// BLAS-able ops
DEFINE_BINARY_OP(add_generic, float, ADD_OP)
static void nx_c_add_float(const ndarray_t *x, const ndarray_t *y,
                           ndarray_t *z) {
  if (is_c_contiguous(x) && is_c_contiguous(y) && is_c_contiguous(z)) {
    long total = total_elements(x);
    float *x_ptr = (float *)x->data + x->offset;
    float *y_ptr = (float *)y->data + y->offset;
    float *z_ptr = (float *)z->data + z->offset;

    _Pragma("omp parallel for simd") for (long i = 0; i < total; i++) {
      z_ptr[i] = x_ptr[i] + y_ptr[i];
    }
  } else {
    nx_c_add_generic_float(x, y, z);
  }
}
DEFINE_BINARY_OP(sub_generic, float, SUB_OP)
static void nx_c_sub_float(const ndarray_t *x, const ndarray_t *y,
                           ndarray_t *z) {
  if (is_c_contiguous(x) && is_c_contiguous(y) && is_c_contiguous(z)) {
    long total = total_elements(x);
    float *x_ptr = (float *)x->data + x->offset;
    float *y_ptr = (float *)y->data + y->offset;
    float *z_ptr = (float *)z->data + z->offset;

    _Pragma("omp parallel for simd") for (long i = 0; i < total; i++) {
      z_ptr[i] = x_ptr[i] - y_ptr[i];
    }
  } else {
    nx_c_sub_generic_float(x, y, z);
  }
}

// FIX: Use the new operator macros
// uint8 operations
DEFINE_BINARY_OP(add, uint8_t, ADD_OP)
DEFINE_BINARY_OP(sub, uint8_t, SUB_OP)
DEFINE_BINARY_OP(mul, uint8_t, MUL_OP)
DEFINE_BINARY_OP(max, uint8_t, MAX_OP)
DEFINE_BINARY_OP(xor, uint8_t, XOR_OP)
DEFINE_BINARY_OP(or, uint8_t, OR_OP)
DEFINE_BINARY_OP(and, uint8_t, AND_OP)

DEFINE_BINARY_OP(add, int32_t, ADD_OP)
DEFINE_BINARY_OP(sub, int32_t, SUB_OP)
DEFINE_BINARY_OP(mul, int32_t, MUL_OP)
DEFINE_BINARY_OP(idiv, int32_t, idiv_op_i32)
DEFINE_BINARY_OP(max, int32_t, MAX_OP)
DEFINE_BINARY_OP(mod, int32_t, MOD_OP)
DEFINE_BINARY_OP(xor, int32_t, XOR_OP)
DEFINE_BINARY_OP(or, int32_t, OR_OP)
DEFINE_BINARY_OP(and, int32_t, AND_OP)

DEFINE_BINARY_OP(add, int64_t, ADD_OP)
DEFINE_BINARY_OP(sub, int64_t, SUB_OP)
DEFINE_BINARY_OP(mul, int64_t, MUL_OP)
DEFINE_BINARY_OP(idiv, int64_t, idiv_op_i64)
DEFINE_BINARY_OP(max, int64_t, MAX_OP)
DEFINE_BINARY_OP(mod, int64_t, MOD_OP)
DEFINE_BINARY_OP(xor, int64_t, XOR_OP)
DEFINE_BINARY_OP(or, int64_t, OR_OP)
DEFINE_BINARY_OP(and, int64_t, AND_OP)

DEFINE_BINARY_OP(mul, float, MUL_OP)
DEFINE_BINARY_OP(fdiv, float, DIV_OP)
DEFINE_BINARY_OP(max, float, FMAX_OP)
DEFINE_BINARY_OP(mod, float, FMOD_OP)
DEFINE_BINARY_OP(pow, float, POW_OP)

// Add float64 (double) binary operations
DEFINE_BINARY_OP(add_generic, double, ADD_OP)
static void nx_c_add_double(const ndarray_t *x, const ndarray_t *y,
                            ndarray_t *z) {
  if (is_c_contiguous(x) && is_c_contiguous(y) && is_c_contiguous(z)) {
    long total = total_elements(x);
    double *x_ptr = (double *)x->data + x->offset;
    double *y_ptr = (double *)y->data + y->offset;
    double *z_ptr = (double *)z->data + z->offset;

    _Pragma("omp parallel for simd") for (long i = 0; i < total; i++) {
      z_ptr[i] = x_ptr[i] + y_ptr[i];
    }
  } else {
    nx_c_add_generic_double(x, y, z);
  }
}

DEFINE_BINARY_OP(sub_generic, double, SUB_OP)
static void nx_c_sub_double(const ndarray_t *x, const ndarray_t *y,
                            ndarray_t *z) {
  if (is_c_contiguous(x) && is_c_contiguous(y) && is_c_contiguous(z)) {
    long total = total_elements(x);
    double *x_ptr = (double *)x->data + x->offset;
    double *y_ptr = (double *)y->data + y->offset;
    double *z_ptr = (double *)z->data + z->offset;

    _Pragma("omp parallel for simd") for (long i = 0; i < total; i++) {
      z_ptr[i] = x_ptr[i] - y_ptr[i];
    }
  } else {
    nx_c_sub_generic_double(x, y, z);
  }
}

DEFINE_BINARY_OP(mul, double, MUL_OP)
DEFINE_BINARY_OP(fdiv, double, DIV_OP)
DEFINE_BINARY_OP(max, double, FMAX_OP)
DEFINE_BINARY_OP(mod, double, FMOD_OP)
DEFINE_BINARY_OP(pow, double, POW_OP)

// Add complex32 binary operations
DEFINE_BINARY_OP(add, c32_t, ADD_OP)
DEFINE_BINARY_OP(sub, c32_t, SUB_OP)
DEFINE_BINARY_OP(mul, c32_t, MUL_OP)
DEFINE_BINARY_OP(fdiv, c32_t, DIV_OP)

// Add complex64 binary operations
DEFINE_BINARY_OP(add, c64_t, ADD_OP)
DEFINE_BINARY_OP(sub, c64_t, SUB_OP)
DEFINE_BINARY_OP(mul, c64_t, MUL_OP)
DEFINE_BINARY_OP(fdiv, c64_t, DIV_OP)

// float16 operations
DEFINE_BINARY_OP(add, float16_t, ADD_OP)
DEFINE_BINARY_OP(sub, float16_t, SUB_OP)
DEFINE_BINARY_OP(mul, float16_t, MUL_OP)
DEFINE_BINARY_OP(fdiv, float16_t, DIV_OP)
DEFINE_BINARY_OP(max, float16_t, FMAX_OP)
DEFINE_BINARY_OP(mod, float16_t, FMOD_OP)
DEFINE_BINARY_OP(pow, float16_t, POW_OP)

// qint8 operations
DEFINE_BINARY_OP(add, qint8_t, ADD_OP)
DEFINE_BINARY_OP(sub, qint8_t, SUB_OP)
DEFINE_BINARY_OP(mul, qint8_t, MUL_OP)
DEFINE_BINARY_OP(max, qint8_t, MAX_OP)
DEFINE_BINARY_OP(mod, qint8_t, MOD_OP)
DEFINE_BINARY_OP(xor, qint8_t, XOR_OP)
DEFINE_BINARY_OP(or, qint8_t, OR_OP)
DEFINE_BINARY_OP(and, qint8_t, AND_OP)

// quint8 operations
DEFINE_BINARY_OP(add, quint8_t, ADD_OP)
DEFINE_BINARY_OP(sub, quint8_t, SUB_OP)
DEFINE_BINARY_OP(mul, quint8_t, MUL_OP)
DEFINE_BINARY_OP(max, quint8_t, MAX_OP)
DEFINE_BINARY_OP(xor, quint8_t, XOR_OP)
DEFINE_BINARY_OP(or, quint8_t, OR_OP)
DEFINE_BINARY_OP(and, quint8_t, AND_OP)

// int8_t operations
DEFINE_BINARY_OP(add, int8_t, ADD_OP)
DEFINE_BINARY_OP(sub, int8_t, SUB_OP)
DEFINE_BINARY_OP(mul, int8_t, MUL_OP)
DEFINE_BINARY_OP(max, int8_t, MAX_OP)
DEFINE_BINARY_OP(mod, int8_t, MOD_OP)
DEFINE_BINARY_OP(xor, int8_t, XOR_OP)
DEFINE_BINARY_OP(or, int8_t, OR_OP)
DEFINE_BINARY_OP(and, int8_t, AND_OP)

// int16_t operations
DEFINE_BINARY_OP(add, int16_t, ADD_OP)
DEFINE_BINARY_OP(sub, int16_t, SUB_OP)
DEFINE_BINARY_OP(mul, int16_t, MUL_OP)
DEFINE_BINARY_OP(max, int16_t, MAX_OP)
DEFINE_BINARY_OP(mod, int16_t, MOD_OP)
DEFINE_BINARY_OP(xor, int16_t, XOR_OP)
DEFINE_BINARY_OP(or, int16_t, OR_OP)
DEFINE_BINARY_OP(and, int16_t, AND_OP)

// uint16_t operations
DEFINE_BINARY_OP(add, uint16_t, ADD_OP)
DEFINE_BINARY_OP(sub, uint16_t, SUB_OP)
DEFINE_BINARY_OP(mul, uint16_t, MUL_OP)
DEFINE_BINARY_OP(max, uint16_t, MAX_OP)
DEFINE_BINARY_OP(xor, uint16_t, XOR_OP)
DEFINE_BINARY_OP(or, uint16_t, OR_OP)
DEFINE_BINARY_OP(and, uint16_t, AND_OP)

// intnat operations
DEFINE_BINARY_OP(add, intnat, ADD_OP)
DEFINE_BINARY_OP(sub, intnat, SUB_OP)
DEFINE_BINARY_OP(mul, intnat, MUL_OP)
DEFINE_BINARY_OP(max, intnat, MAX_OP)
DEFINE_BINARY_OP(mod, intnat, MOD_OP)
DEFINE_BINARY_OP(xor, intnat, XOR_OP)
DEFINE_BINARY_OP(or, intnat, OR_OP)
DEFINE_BINARY_OP(and, intnat, AND_OP)

// Comparison operation macro that returns uint8_t results
#define DEFINE_COMPARE_OP(name, T, OP)                                      \
  DEFINE_COMPARE_OP_KERNEL(name, T, OP)                                     \
  static void nx_c_##name##_##T(const ndarray_t *x, const ndarray_t *y,     \
                                ndarray_t *z) {                             \
    T *x_data = (T *)x->data;                                               \
    T *y_data = (T *)y->data;                                               \
    uint8_t *z_data = (uint8_t *)z->data;                                   \
    long total = total_elements(x);                                         \
    if (total == 0) return;                                                 \
    if (is_c_contiguous(x) && is_c_contiguous(y) && is_c_contiguous(z)) {   \
      T *x_ptr = x_data + x->offset;                                        \
      T *y_ptr = y_data + y->offset;                                        \
      uint8_t *z_ptr = z_data + z->offset;                                  \
      _Pragma("omp parallel for simd") for (long i = 0; i < total; i++) {   \
        z_ptr[i] = OP(x_ptr[i], y_ptr[i]) ? 1 : 0;                          \
      }                                                                     \
    } else if (x->ndim > 0) {                                               \
      /* Parallelize over the outermost dimension */                        \
      _Pragma("omp parallel for") for (long i = 0; i < x->shape[0]; i++) {  \
        iterate_inner_dims(x, y, z, i, nx_c_##name##_##T##_kernel, x_data,  \
                           y_data, z_data);                                 \
      }                                                                     \
    } else {                                                                \
      /* Scalar case */                                                     \
      z_data[z->offset] = OP(x_data[x->offset], y_data[y->offset]) ? 1 : 0; \
    }                                                                       \
  }

// Define comparison operations for all numeric types
DEFINE_COMPARE_OP(cmplt, int8_t, CMPLT_OP)
DEFINE_COMPARE_OP(cmplt, uint8_t, CMPLT_OP)
DEFINE_COMPARE_OP(cmplt, int16_t, CMPLT_OP)
DEFINE_COMPARE_OP(cmplt, uint16_t, CMPLT_OP)
DEFINE_COMPARE_OP(cmplt, int32_t, CMPLT_OP)
DEFINE_COMPARE_OP(cmplt, int64_t, CMPLT_OP)
DEFINE_COMPARE_OP(cmplt, intnat, CMPLT_OP)
DEFINE_COMPARE_OP(cmplt, float, CMPLT_OP)
DEFINE_COMPARE_OP(cmplt, double, CMPLT_OP)
// float16 and quantized comparison
DEFINE_COMPARE_OP(cmplt, float16_t, CMPLT_OP)
DEFINE_COMPARE_OP(cmplt, qint8_t, CMPLT_OP)
DEFINE_COMPARE_OP(cmplt, quint8_t, CMPLT_OP)

DEFINE_COMPARE_OP(cmpne, int8_t, CMPNE_OP)
DEFINE_COMPARE_OP(cmpne, uint8_t, CMPNE_OP)
DEFINE_COMPARE_OP(cmpne, int16_t, CMPNE_OP)
DEFINE_COMPARE_OP(cmpne, uint16_t, CMPNE_OP)
DEFINE_COMPARE_OP(cmpne, int32_t, CMPNE_OP)
DEFINE_COMPARE_OP(cmpne, int64_t, CMPNE_OP)
DEFINE_COMPARE_OP(cmpne, intnat, CMPNE_OP)
DEFINE_COMPARE_OP(cmpne, float, CMPNE_OP)
DEFINE_COMPARE_OP(cmpne, double, CMPNE_OP)
DEFINE_COMPARE_OP(cmpne, c32_t, CMPNE_OP)
DEFINE_COMPARE_OP(cmpne, c64_t, CMPNE_OP)
// float16 and quantized comparison
DEFINE_COMPARE_OP(cmpne, float16_t, CMPNE_OP)
DEFINE_COMPARE_OP(cmpne, qint8_t, CMPNE_OP)
DEFINE_COMPARE_OP(cmpne, quint8_t, CMPNE_OP)

// Reduce operation helpers
static void compute_reduction_params(const ndarray_t *x, const int *axes,
                                     int num_axes, int *reduce_shape,
                                     int *reduce_strides, int *outer_shape,
                                     int *outer_strides, int *result_shape,
                                     int *result_strides) {
  // Initialize all dimensions as kept
  for (int i = 0; i < x->ndim; i++) {
    outer_shape[i] = x->shape[i];
    outer_strides[i] = x->strides[i];
    result_shape[i] = x->shape[i];
    result_strides[i] = 0;  // Will be computed based on result layout
  }

  // Mark axes to reduce
  for (int i = 0; i < num_axes; i++) {
    int axis = axes[i];
    if (axis < 0) axis += x->ndim;  // Handle negative indices
    reduce_shape[i] = x->shape[axis];
    reduce_strides[i] = x->strides[axis];
    outer_shape[axis] = 1;
    result_shape[axis] = 1;
  }

  // Compute result strides for C-contiguous layout
  int stride = 1;
  for (int i = x->ndim - 1; i >= 0; i--) {
    result_strides[i] = stride;
    stride *= result_shape[i];
  }
}

// Generic reduce kernel
#define DEFINE_REDUCE_OP_KERNEL(name, T, INIT, OP)                            \
  static void nx_c_reduce_##name##_##T##_kernel(                              \
      const ndarray_t *x, ndarray_t *z, const int *axes, int num_axes) {      \
    T *x_data = (T *)x->data;                                                 \
    T *z_data = (T *)z->data;                                                 \
                                                                              \
    int reduce_shape[16], reduce_strides[16];                                 \
    int outer_shape[16], outer_strides[16];                                   \
    int result_shape[16], result_strides[16];                                 \
                                                                              \
    compute_reduction_params(x, axes, num_axes, reduce_shape, reduce_strides, \
                             outer_shape, outer_strides, result_shape,        \
                             result_strides);                                 \
                                                                              \
    /* Initialize output to identity element */                               \
    long z_total = 1;                                                         \
    for (int i = 0; i < x->ndim; i++) z_total *= result_shape[i];             \
    _Pragma("omp parallel for simd") for (long i = 0; i < z_total; i++) {     \
      z_data[z->offset + i] = INIT;                                           \
    }                                                                         \
                                                                              \
    /* Perform reduction */                                                   \
    long total_outer = 1;                                                     \
    for (int i = 0; i < x->ndim; i++) {                                       \
      if (outer_shape[i] > 1) total_outer *= outer_shape[i];                  \
    }                                                                         \
                                                                              \
    _Pragma("omp parallel for") for (long outer_idx = 0;                      \
                                     outer_idx < total_outer; outer_idx++) {  \
      /* Compute position in non-reduced dimensions */                        \
      long temp = outer_idx;                                                  \
      long x_outer_offset = x->offset;                                        \
      long z_offset = z->offset;                                              \
                                                                              \
      for (int d = x->ndim - 1; d >= 0; d--) {                                \
        if (outer_shape[d] > 1) {                                             \
          long dim_idx = temp % outer_shape[d];                               \
          x_outer_offset += dim_idx * outer_strides[d];                       \
          z_offset += dim_idx * result_strides[d];                            \
          temp /= outer_shape[d];                                             \
        }                                                                     \
      }                                                                       \
                                                                              \
      /* Iterate over axes to reduce */                                       \
      if (num_axes == 1) {                                                    \
        /* Common case: single axis reduction */                              \
        int axis = axes[0];                                                   \
        if (axis < 0) axis += x->ndim;                                        \
        T acc = INIT;                                                         \
        for (int i = 0; i < x->shape[axis]; i++) {                            \
          long x_idx = x_outer_offset + i * x->strides[axis];                 \
          acc = OP(acc, x_data[x_idx]);                                       \
        }                                                                     \
        z_data[z_offset] = acc;                                               \
      } else {                                                                \
        /* General case: multiple axes */                                     \
        long reduce_total = 1;                                                \
        for (int i = 0; i < num_axes; i++) {                                  \
          reduce_total *= reduce_shape[i];                                    \
        }                                                                     \
                                                                              \
        T acc = INIT;                                                         \
        for (long i = 0; i < reduce_total; i++) {                             \
          long reduce_offset = 0;                                             \
          long temp_i = i;                                                    \
          for (int j = num_axes - 1; j >= 0; j--) {                           \
            reduce_offset += (temp_i % reduce_shape[j]) * reduce_strides[j];  \
            temp_i /= reduce_shape[j];                                        \
          }                                                                   \
          acc = OP(acc, x_data[x_outer_offset + reduce_offset]);              \
        }                                                                     \
        z_data[z_offset] = acc;                                               \
      }                                                                       \
    }                                                                         \
  }

// Define reduce operations for different types
#define SUM_OP(a, b) ((a) + (b))
#define PROD_OP(a, b) ((a) * (b))

// Sum operations
DEFINE_REDUCE_OP_KERNEL(sum, int8_t, 0, SUM_OP)
DEFINE_REDUCE_OP_KERNEL(sum, uint8_t, 0, SUM_OP)
DEFINE_REDUCE_OP_KERNEL(sum, int16_t, 0, SUM_OP)
DEFINE_REDUCE_OP_KERNEL(sum, uint16_t, 0, SUM_OP)
DEFINE_REDUCE_OP_KERNEL(sum, int32_t, 0, SUM_OP)
DEFINE_REDUCE_OP_KERNEL(sum, int64_t, 0, SUM_OP)
DEFINE_REDUCE_OP_KERNEL(sum, intnat, 0, SUM_OP)
DEFINE_REDUCE_OP_KERNEL(sum, float, 0.0f, SUM_OP)
DEFINE_REDUCE_OP_KERNEL(sum, double, 0.0, SUM_OP)
// float16 and quantized sum
DEFINE_REDUCE_OP_KERNEL(sum, float16_t, (float16_t)0.0f, SUM_OP)
DEFINE_REDUCE_OP_KERNEL(sum, qint8_t, 0, SUM_OP)
DEFINE_REDUCE_OP_KERNEL(sum, quint8_t, 0, SUM_OP)

// Max operations
DEFINE_REDUCE_OP_KERNEL(max, int8_t, INT8_MIN, MAX_OP)
DEFINE_REDUCE_OP_KERNEL(max, uint8_t, 0, MAX_OP)
DEFINE_REDUCE_OP_KERNEL(max, int16_t, INT16_MIN, MAX_OP)
DEFINE_REDUCE_OP_KERNEL(max, uint16_t, 0, MAX_OP)
DEFINE_REDUCE_OP_KERNEL(max, int32_t, INT32_MIN, MAX_OP)
DEFINE_REDUCE_OP_KERNEL(max, int64_t, INT64_MIN, MAX_OP)
DEFINE_REDUCE_OP_KERNEL(max, intnat, INTNAT_MIN, MAX_OP)
DEFINE_REDUCE_OP_KERNEL(max, float, -INFINITY, MAX_NAN_OP)
DEFINE_REDUCE_OP_KERNEL(max, double, -INFINITY, MAX_NAN_OP)
// float16 and quantized max
DEFINE_REDUCE_OP_KERNEL(max, float16_t, (float16_t)(-INFINITY), MAX_NAN_OP)
DEFINE_REDUCE_OP_KERNEL(max, qint8_t, INT8_MIN, MAX_OP)
DEFINE_REDUCE_OP_KERNEL(max, quint8_t, 0, MAX_OP)

// Prod operations
DEFINE_REDUCE_OP_KERNEL(prod, int8_t, 1, PROD_OP)
DEFINE_REDUCE_OP_KERNEL(prod, uint8_t, 1, PROD_OP)
DEFINE_REDUCE_OP_KERNEL(prod, int16_t, 1, PROD_OP)
DEFINE_REDUCE_OP_KERNEL(prod, uint16_t, 1, PROD_OP)
DEFINE_REDUCE_OP_KERNEL(prod, int32_t, 1, PROD_OP)
DEFINE_REDUCE_OP_KERNEL(prod, int64_t, 1, PROD_OP)
DEFINE_REDUCE_OP_KERNEL(prod, intnat, 1, PROD_OP)
DEFINE_REDUCE_OP_KERNEL(prod, float, 1.0f, PROD_OP)
DEFINE_REDUCE_OP_KERNEL(prod, double, 1.0, PROD_OP)
// float16 and quantized prod
DEFINE_REDUCE_OP_KERNEL(prod, float16_t, (float16_t)1.0f, PROD_OP)
DEFINE_REDUCE_OP_KERNEL(prod, qint8_t, 1, PROD_OP)
DEFINE_REDUCE_OP_KERNEL(prod, quint8_t, 1, PROD_OP)

// Helper for where operation with 4 arrays
static inline void iterate_where_inner_dims(
    const ndarray_t *cond, const ndarray_t *x, const ndarray_t *y,
    const ndarray_t *z, long outer_idx,
    void (*op)(void *, void *, void *, void *, long, long, long, long),
    void *cond_data, void *x_data, void *y_data, void *z_data);

// Where operation - conditional selection
#define DEFINE_WHERE_OP_KERNEL(T)                                \
  static void nx_c_where_##T##_kernel(                           \
      void *cond_data, void *x_data, void *y_data, void *z_data, \
      long cond_off, long x_off, long y_off, long z_off) {       \
    uint8_t *cond = (uint8_t *)cond_data;                        \
    T *x = (T *)x_data;                                          \
    T *y = (T *)y_data;                                          \
    T *z = (T *)z_data;                                          \
    z[z_off] = cond[cond_off] ? x[x_off] : y[y_off];             \
  }

#define DEFINE_WHERE_OP(T)                                                   \
  DEFINE_WHERE_OP_KERNEL(T)                                                  \
  static void nx_c_where_##T(const ndarray_t *cond, const ndarray_t *x,      \
                             const ndarray_t *y, ndarray_t *z) {             \
    uint8_t *cond_data = (uint8_t *)cond->data;                              \
    T *x_data = (T *)x->data;                                                \
    T *y_data = (T *)y->data;                                                \
    T *z_data = (T *)z->data;                                                \
    long total = total_elements(x);                                          \
    if (total == 0) return;                                                  \
                                                                             \
    if (is_c_contiguous(cond) && is_c_contiguous(x) && is_c_contiguous(y) && \
        is_c_contiguous(z)) {                                                \
      uint8_t *cond_ptr = cond_data + cond->offset;                          \
      T *x_ptr = x_data + x->offset;                                         \
      T *y_ptr = y_data + y->offset;                                         \
      T *z_ptr = z_data + z->offset;                                         \
      _Pragma("omp parallel for simd") for (long i = 0; i < total; i++) {    \
        z_ptr[i] = cond_ptr[i] ? x_ptr[i] : y_ptr[i];                        \
      }                                                                      \
    } else if (x->ndim > 0) {                                                \
      /* Parallelize over the outermost dimension */                         \
      _Pragma("omp parallel for") for (long i = 0; i < x->shape[0]; i++) {   \
        /* Need special iterate function for 4 arrays */                     \
        iterate_where_inner_dims(cond, x, y, z, i, nx_c_where_##T##_kernel,  \
                                 cond_data, x_data, y_data, z_data);         \
      }                                                                      \
    } else {                                                                 \
      /* Scalar case */                                                      \
      z_data[z->offset] =                                                    \
          cond_data[cond->offset] ? x_data[x->offset] : y_data[y->offset];   \
    }                                                                        \
  }

// Implementation of helper for where operation with 4 arrays
static inline void iterate_where_inner_dims(
    const ndarray_t *cond, const ndarray_t *x, const ndarray_t *y,
    const ndarray_t *z, long outer_idx,
    void (*op)(void *, void *, void *, void *, long, long, long, long),
    void *cond_data, void *x_data, void *y_data, void *z_data) {
  if (x->ndim == 1) {
    long cond_off = cond->offset + outer_idx * cond->strides[0];
    long x_off = x->offset + outer_idx * x->strides[0];
    long y_off = y->offset + outer_idx * y->strides[0];
    long z_off = z->offset + outer_idx * z->strides[0];
    op(cond_data, x_data, y_data, z_data, cond_off, x_off, y_off, z_off);
  } else if (x->ndim == 2) {
    long cond_base = cond->offset + outer_idx * cond->strides[0];
    long x_base = x->offset + outer_idx * x->strides[0];
    long y_base = y->offset + outer_idx * y->strides[0];
    long z_base = z->offset + outer_idx * z->strides[0];

    for (long j = 0; j < x->shape[1]; j++) {
      long cond_off = cond_base + j * cond->strides[1];
      long x_off = x_base + j * x->strides[1];
      long y_off = y_base + j * y->strides[1];
      long z_off = z_base + j * z->strides[1];
      op(cond_data, x_data, y_data, z_data, cond_off, x_off, y_off, z_off);
    }
  } else {
    // General case for higher dimensions
    long indices[16] = {0};
    long total_inner = 1;
    for (int d = 1; d < x->ndim; d++) total_inner *= x->shape[d];

    for (long i = 0; i < total_inner; i++) {
      long temp = i;
      long cond_off = cond->offset + outer_idx * cond->strides[0];
      long x_off = x->offset + outer_idx * x->strides[0];
      long y_off = y->offset + outer_idx * y->strides[0];
      long z_off = z->offset + outer_idx * z->strides[0];

      for (int d = x->ndim - 1; d >= 1; d--) {
        indices[d] = temp % x->shape[d];
        cond_off += indices[d] * cond->strides[d];
        x_off += indices[d] * x->strides[d];
        y_off += indices[d] * y->strides[d];
        z_off += indices[d] * z->strides[d];
        temp /= x->shape[d];
      }

      op(cond_data, x_data, y_data, z_data, cond_off, x_off, y_off, z_off);
    }
  }
}

DEFINE_WHERE_OP(int8_t)
DEFINE_WHERE_OP(uint8_t)
DEFINE_WHERE_OP(int16_t)
DEFINE_WHERE_OP(uint16_t)
DEFINE_WHERE_OP(int32_t)
DEFINE_WHERE_OP(int64_t)
DEFINE_WHERE_OP(intnat)
DEFINE_WHERE_OP(float)
DEFINE_WHERE_OP(double)
DEFINE_WHERE_OP(c32_t)
DEFINE_WHERE_OP(c64_t)
// float16 and quantized where operations
DEFINE_WHERE_OP(float16_t)
DEFINE_WHERE_OP(qint8_t)
DEFINE_WHERE_OP(quint8_t)

// Pad operation - adds padding to array edges
#define DEFINE_PAD_OP(T)                                                       \
  static void nx_c_pad_##T(const ndarray_t *x, ndarray_t *z,                   \
                           const int *padding, T pad_value) {                  \
    T *x_data = (T *)x->data;                                                  \
    T *z_data = (T *)z->data;                                                  \
                                                                               \
    /* Initialize output with pad value */                                     \
    long z_total = total_elements(z);                                          \
    _Pragma("omp parallel for simd") for (long i = 0; i < z_total; i++) {      \
      z_data[z->offset + i] = pad_value;                                       \
    }                                                                          \
                                                                               \
    /* Copy input data to the padded region */                                 \
    if (x->ndim == 0) {                                                        \
      /* Scalar case */                                                        \
      z_data[z->offset] = x_data[x->offset];                                   \
    } else if (x->ndim == 1) {                                                 \
      /* 1D optimized case */                                                  \
      int pad_left = padding[0];                                               \
      _Pragma("omp parallel for simd") for (int i = 0; i < x->shape[0]; i++) { \
        z_data[z->offset + (i + pad_left) * z->strides[0]] =                   \
            x_data[x->offset + i * x->strides[0]];                             \
      }                                                                        \
    } else if (x->ndim == 2) {                                                 \
      /* 2D optimized case */                                                  \
      int pad_top = padding[0];                                                \
      int pad_left = padding[2];                                               \
      _Pragma("omp parallel for") for (int i = 0; i < x->shape[0]; i++) {      \
        for (int j = 0; j < x->shape[1]; j++) {                                \
          z_data[z->offset + (i + pad_top) * z->strides[0] +                   \
                 (j + pad_left) * z->strides[1]] =                             \
              x_data[x->offset + i * x->strides[0] + j * x->strides[1]];       \
        }                                                                      \
      }                                                                        \
    } else {                                                                   \
      /* General N-D case */                                                   \
      long indices[16] = {0};                                                  \
      long padded_indices[16];                                                 \
      long x_total = total_elements(x);                                        \
                                                                               \
      for (long idx = 0; idx < x_total; idx++) {                               \
        /* Calculate indices */                                                \
        long temp = idx;                                                       \
        for (int d = x->ndim - 1; d >= 0; d--) {                               \
          indices[d] = temp % x->shape[d];                                     \
          padded_indices[d] = indices[d] + padding[d * 2];                     \
          temp /= x->shape[d];                                                 \
        }                                                                      \
                                                                               \
        /* Calculate offsets */                                                \
        long x_off = x->offset;                                                \
        long z_off = z->offset;                                                \
        for (int d = 0; d < x->ndim; d++) {                                    \
          x_off += indices[d] * x->strides[d];                                 \
          z_off += padded_indices[d] * z->strides[d];                          \
        }                                                                      \
                                                                               \
        z_data[z_off] = x_data[x_off];                                         \
      }                                                                        \
    }                                                                          \
  }

DEFINE_PAD_OP(int32_t)
DEFINE_PAD_OP(int64_t)
DEFINE_PAD_OP(float)
DEFINE_PAD_OP(double)
DEFINE_PAD_OP(c32_t)
DEFINE_PAD_OP(c64_t)
DEFINE_PAD_OP(uint8_t)
// float16 and quantized pad operations
DEFINE_PAD_OP(float16_t)
DEFINE_PAD_OP(qint8_t)
DEFINE_PAD_OP(quint8_t)

// Cat (concatenation) operation
static void nx_c_cat_generic(const ndarray_t **inputs, int num_inputs,
                             ndarray_t *output, int axis, size_t elem_size) {
  if (num_inputs == 0) return;

  // Normalize axis
  if (axis < 0) axis += inputs[0]->ndim;

  // For each input tensor, copy it to the appropriate location in output
  long output_offset_along_axis = 0;

  for (int input_idx = 0; input_idx < num_inputs; input_idx++) {
    const ndarray_t *input = inputs[input_idx];

    // Calculate total number of elements to copy
    long total = total_elements(input);
    if (total == 0) continue;

    // For contiguous tensors along the concatenation axis, use memcpy
    if (axis == input->ndim - 1 && is_c_contiguous(input)) {
      // Fast path for last-axis concatenation of contiguous arrays
      long num_slices = total / input->shape[axis];
      long slice_size = input->shape[axis] * elem_size;

      _Pragma("omp parallel for") for (long slice = 0; slice < num_slices;
                                       slice++) {
        long input_offset = input->offset + slice * input->shape[axis];
        long output_slice_offset =
            slice * output->shape[axis] + output_offset_along_axis;
        long output_offset = output->offset + output_slice_offset;

        memcpy((char *)output->data + output_offset * elem_size,
               (char *)input->data + input_offset * elem_size, slice_size);
      }
    } else {
      // General case: iterate through all elements
      long indices[16] = {0};

      for (long idx = 0; idx < total; idx++) {
        // Calculate multi-dimensional indices
        long temp = idx;
        for (int d = input->ndim - 1; d >= 0; d--) {
          indices[d] = temp % input->shape[d];
          temp /= input->shape[d];
        }

        // Calculate input offset
        long input_offset = input->offset;
        for (int d = 0; d < input->ndim; d++) {
          input_offset += indices[d] * input->strides[d];
        }

        // Calculate output offset (adjust index along concatenation axis)
        long output_offset = output->offset;
        for (int d = 0; d < output->ndim; d++) {
          if (d == axis) {
            output_offset +=
                (indices[d] + output_offset_along_axis) * output->strides[d];
          } else {
            output_offset += indices[d] * output->strides[d];
          }
        }

        // Copy element
        memcpy((char *)output->data + output_offset * elem_size,
               (char *)input->data + input_offset * elem_size, elem_size);
      }
    }

    output_offset_along_axis += input->shape[axis];
  }
}

// OCaml FFI (Foreign Function Interface) Stubs
// ---------------------------------------------

CAMLprim value caml_nx_assign_bc(value *argv, int argn) {
  int ndim = Int_val(argv[0]);
  value v_shape = argv[1], v_src = argv[2], v_src_strides = argv[3],
        v_src_offset = argv[4];
  value v_dst = argv[5], v_dst_strides = argv[6], v_dst_offset = argv[7];

  // Handle 0-dimensional tensors (scalars)
  if (ndim == 0) {
    // Get the element size from the source array's kind.
    int kind = Caml_ba_array_val(v_src)->flags & CAML_BA_KIND_MASK;
    size_t elem_size = get_element_size(kind);
    if (elem_size == 0) caml_failwith("assign: unsupported dtype for copy");

    // For 0-dimensional tensors, just copy the single element
    void *src_data = Caml_ba_data_val(v_src);
    void *dst_data = Caml_ba_data_val(v_dst);
    int src_offset = Int_val(v_src_offset);
    int dst_offset = Int_val(v_dst_offset);

    // Special handling for Int4/UInt4 packed types
    if (kind == NX_BA_INT4 || kind == NX_BA_UINT4) {
      uint8_t value = get_int4_value(src_data, src_offset);
      set_int4_value(dst_data, dst_offset, value);
    } else {
      memcpy((char *)dst_data + dst_offset * elem_size,
             (char *)src_data + src_offset * elem_size, elem_size);
    }

    return Val_unit;
  }

  int shape[ndim > 0 ? ndim : 1], src_strides[ndim > 0 ? ndim : 1],
      dst_strides[ndim > 0 ? ndim : 1];
  for (int i = 0; i < ndim; ++i) shape[i] = Int_val(Field(v_shape, i));
  for (int i = 0; i < ndim; ++i)
    src_strides[i] = Int_val(Field(v_src_strides, i));
  for (int i = 0; i < ndim; ++i)
    dst_strides[i] = Int_val(Field(v_dst_strides, i));

  // Get the element size from the source array's kind.
  int kind = Caml_ba_array_val(v_src)->flags & CAML_BA_KIND_MASK;
  size_t elem_size = get_element_size(kind);
  if (elem_size == 0) caml_failwith("assign: unsupported dtype for copy");

  ndarray_t src, dst;
  src.data = Caml_ba_data_val(v_src);
  src.ndim = ndim;
  src.shape = shape;
  src.strides = src_strides;
  src.offset = Int_val(v_src_offset);
  dst.data = Caml_ba_data_val(v_dst);
  dst.ndim = ndim;
  dst.shape = shape;
  dst.strides = dst_strides;
  dst.offset = Int_val(v_dst_offset);

  caml_enter_blocking_section();
  
  // Special handling for Int4/UInt4 packed types
  if (kind == NX_BA_INT4 || kind == NX_BA_UINT4) {
    nx_c_copy_int4(&src, &dst);
  } else {
    nx_c_generic_copy(&src, &dst, elem_size);
  }
  
  caml_leave_blocking_section();

  return Val_unit;
}
NATIVE_WRAPPER_8(assign)

// `copy` is identical to `assign` at the C level.
CAMLprim value caml_nx_copy_bc(value *argv, int argn) {
  return caml_nx_assign_bc(argv, argn);
}
NATIVE_WRAPPER_8(copy)

// Dispatcher for cast operations
typedef void (*cast_op_t)(const ndarray_t *, ndarray_t *);

CAMLprim value caml_nx_cast_bc(value *argv, int argn) {
  int ndim = Int_val(argv[0]);
  value v_shape = argv[1], v_src = argv[2], v_src_strides = argv[3],
        v_src_offset = argv[4];
  value v_dst = argv[5], v_dst_strides = argv[6], v_dst_offset = argv[7];

  // Handle 0-dimensional tensors (scalars)
  if (ndim == 0) {
    int src_kind = Caml_ba_array_val(v_src)->flags & CAML_BA_KIND_MASK;
    int dst_kind = Caml_ba_array_val(v_dst)->flags & CAML_BA_KIND_MASK;

    void *src_data = Caml_ba_data_val(v_src);
    void *dst_data = Caml_ba_data_val(v_dst);
    int src_offset = Int_val(v_src_offset);
    int dst_offset = Int_val(v_dst_offset);

    // Simple scalar cast
    if (src_kind == CAML_BA_UINT8 && dst_kind == CAML_BA_INT32) {
      uint8_t *src = (uint8_t *)src_data;
      int32_t *dst = (int32_t *)dst_data;
      dst[dst_offset] = (int32_t)src[src_offset];
    } else if (src_kind == CAML_BA_FLOAT32 && dst_kind == CAML_BA_FLOAT64) {
      float *src = (float *)src_data;
      double *dst = (double *)dst_data;
      dst[dst_offset] = (double)src[src_offset];
    } else if (src_kind == CAML_BA_FLOAT64 && dst_kind == CAML_BA_FLOAT32) {
      double *src = (double *)src_data;
      float *dst = (float *)dst_data;
      dst[dst_offset] = (float)src[src_offset];
    } else if (src_kind == CAML_BA_INT32 && dst_kind == CAML_BA_FLOAT32) {
      int32_t *src = (int32_t *)src_data;
      float *dst = (float *)dst_data;
      dst[dst_offset] = (float)src[src_offset];
    } else {
      caml_failwith("cast: unsupported dtype conversion for 0d tensor");
    }

    return Val_unit;
  }

  int src_kind = Caml_ba_array_val(v_src)->flags & CAML_BA_KIND_MASK;
  int dst_kind = Caml_ba_array_val(v_dst)->flags & CAML_BA_KIND_MASK;

  int shape[ndim], src_strides[ndim], dst_strides[ndim];
  for (int i = 0; i < ndim; ++i) shape[i] = Int_val(Field(v_shape, i));
  for (int i = 0; i < ndim; ++i)
    src_strides[i] = Int_val(Field(v_src_strides, i));
  for (int i = 0; i < ndim; ++i)
    dst_strides[i] = Int_val(Field(v_dst_strides, i));

  ndarray_t src, dst;
  src.data = Caml_ba_data_val(v_src);
  src.ndim = ndim;
  src.shape = shape;
  src.strides = src_strides;
  src.offset = Int_val(v_src_offset);
  dst.data = Caml_ba_data_val(v_dst);
  dst.ndim = ndim;
  dst.shape = shape;
  dst.strides = dst_strides;
  dst.offset = Int_val(v_dst_offset);

  cast_op_t func = NULL;
  
  // Create a comprehensive cast dispatch
  if (src_kind == CAML_BA_FLOAT32 && dst_kind == CAML_BA_FLOAT64)
    func = nx_c_cast_float_to_double;
  else if (src_kind == CAML_BA_FLOAT64 && dst_kind == CAML_BA_FLOAT32)
    func = nx_c_cast_double_to_float;
  else if (src_kind == CAML_BA_INT32 && dst_kind == CAML_BA_FLOAT32)
    func = nx_c_cast_int32_t_to_float;
  else if (src_kind == CAML_BA_INT64 && dst_kind == CAML_BA_FLOAT32)
    func = nx_c_cast_int64_t_to_float;
  else if (src_kind == CAML_BA_FLOAT32 && dst_kind == CAML_BA_INT32)
    func = nx_c_cast_float_to_int32_t;
  else if (src_kind == CAML_BA_FLOAT32 && dst_kind == CAML_BA_INT64)
    func = nx_c_cast_float_to_int64_t;
  else if (src_kind == CAML_BA_FLOAT64 && dst_kind == CAML_BA_INT32)
    func = nx_c_cast_double_to_int32_t;
  else if (src_kind == CAML_BA_FLOAT64 && dst_kind == CAML_BA_INT64)
    func = nx_c_cast_double_to_int64_t;
  else if (src_kind == CAML_BA_INT32 && dst_kind == CAML_BA_FLOAT64)
    func = nx_c_cast_int32_t_to_double;
  else if (src_kind == CAML_BA_INT64 && dst_kind == CAML_BA_FLOAT64)
    func = nx_c_cast_int64_t_to_double;
  else if (src_kind == CAML_BA_SINT16 && dst_kind == CAML_BA_FLOAT32)
    func = nx_c_cast_int16_t_to_float;
  else if (src_kind == CAML_BA_SINT16 && dst_kind == CAML_BA_FLOAT64)
    func = nx_c_cast_int16_t_to_double;
  else if (src_kind == CAML_BA_SINT16 && dst_kind == CAML_BA_INT32)
    func = nx_c_cast_int16_t_to_int32_t;
  else if (src_kind == CAML_BA_SINT16 && dst_kind == CAML_BA_INT64)
    func = nx_c_cast_int16_t_to_int64_t;
  else if (src_kind == CAML_BA_INT32 && dst_kind == CAML_BA_SINT16)
    func = nx_c_cast_int32_t_to_int16_t;
  else if (src_kind == CAML_BA_INT64 && dst_kind == CAML_BA_SINT16)
    func = nx_c_cast_int64_t_to_int16_t;
  else if (src_kind == CAML_BA_FLOAT32 && dst_kind == CAML_BA_SINT16)
    func = nx_c_cast_float_to_int16_t;
  else if (src_kind == CAML_BA_FLOAT64 && dst_kind == CAML_BA_SINT16)
    func = nx_c_cast_double_to_int16_t;
  else if (src_kind == CAML_BA_UINT8 && dst_kind == CAML_BA_INT32)
    func = nx_c_cast_uint8_t_to_int32_t;
  else if (src_kind == CAML_BA_UINT8 && dst_kind == CAML_BA_FLOAT32)
    func = nx_c_cast_uint8_t_to_float;
  else if (src_kind == CAML_BA_UINT8 && dst_kind == CAML_BA_FLOAT64)
    func = nx_c_cast_uint8_t_to_double;
  else if (src_kind == CAML_BA_INT32 && dst_kind == CAML_BA_UINT8)
    func = nx_c_cast_int32_t_to_uint8_t;
  else if (src_kind == CAML_BA_FLOAT32 && dst_kind == CAML_BA_UINT8)
    func = nx_c_cast_float_to_uint8_t;
  else if (src_kind == CAML_BA_FLOAT64 && dst_kind == CAML_BA_UINT8)
    func = nx_c_cast_double_to_uint8_t;
  // Float16 casts
  else if (src_kind == CAML_BA_FLOAT16 && dst_kind == CAML_BA_FLOAT32)
    func = nx_c_cast_float16_t_to_float;
  else if (src_kind == CAML_BA_FLOAT16 && dst_kind == CAML_BA_FLOAT64)
    func = nx_c_cast_float16_t_to_double;
  else if (src_kind == CAML_BA_FLOAT16 && dst_kind == CAML_BA_INT32)
    func = nx_c_cast_float16_t_to_int32_t;
  else if (src_kind == CAML_BA_FLOAT16 && dst_kind == CAML_BA_INT64)
    func = nx_c_cast_float16_t_to_int64_t;
  else if (src_kind == CAML_BA_FLOAT32 && dst_kind == CAML_BA_FLOAT16)
    func = nx_c_cast_float_to_float16_t;
  else if (src_kind == CAML_BA_FLOAT64 && dst_kind == CAML_BA_FLOAT16)
    func = nx_c_cast_double_to_float16_t;
  else if (src_kind == CAML_BA_INT32 && dst_kind == CAML_BA_FLOAT16)
    func = nx_c_cast_int32_t_to_float16_t;
  else if (src_kind == CAML_BA_INT64 && dst_kind == CAML_BA_FLOAT16)
    func = nx_c_cast_int64_t_to_float16_t;
  else if (src_kind == CAML_BA_UINT8 && dst_kind == CAML_BA_FLOAT16)
    func = nx_c_cast_uint8_t_to_float16_t;
  else if (src_kind == CAML_BA_FLOAT16 && dst_kind == CAML_BA_UINT8)
    func = nx_c_cast_float16_t_to_uint8_t;
  // Qint8 casts  
  else if (src_kind == NX_BA_QINT8 && dst_kind == CAML_BA_INT32)
    func = nx_c_cast_qint8_t_to_int32_t;
  else if (src_kind == NX_BA_QINT8 && dst_kind == CAML_BA_FLOAT32)
    func = nx_c_cast_qint8_t_to_float;
  else if (src_kind == NX_BA_QINT8 && dst_kind == CAML_BA_FLOAT64)
    func = nx_c_cast_qint8_t_to_double;
  else if (src_kind == CAML_BA_INT32 && dst_kind == NX_BA_QINT8)
    func = nx_c_cast_int32_t_to_qint8_t;
  else if (src_kind == CAML_BA_FLOAT32 && dst_kind == NX_BA_QINT8)
    func = nx_c_cast_float_to_qint8_t;
  else if (src_kind == CAML_BA_FLOAT64 && dst_kind == NX_BA_QINT8)
    func = nx_c_cast_double_to_qint8_t;
  // Quint8 casts
  else if (src_kind == NX_BA_QUINT8 && dst_kind == CAML_BA_INT32)
    func = nx_c_cast_quint8_t_to_int32_t;
  else if (src_kind == NX_BA_QUINT8 && dst_kind == CAML_BA_FLOAT32)
    func = nx_c_cast_quint8_t_to_float;
  else if (src_kind == NX_BA_QUINT8 && dst_kind == CAML_BA_FLOAT64)
    func = nx_c_cast_quint8_t_to_double;
  else if (src_kind == CAML_BA_INT32 && dst_kind == NX_BA_QUINT8)
    func = nx_c_cast_int32_t_to_quint8_t;
  else if (src_kind == CAML_BA_FLOAT32 && dst_kind == NX_BA_QUINT8)
    func = nx_c_cast_float_to_quint8_t;
  else if (src_kind == CAML_BA_FLOAT64 && dst_kind == NX_BA_QUINT8)
    func = nx_c_cast_double_to_quint8_t;

  if (func) {
    caml_enter_blocking_section();
    func(&src, &dst);
    caml_leave_blocking_section();
  } else {
    caml_failwith("cast: unsupported dtype conversion");
  }

  return Val_unit;
}
NATIVE_WRAPPER_8(cast)

// Table and dispatcher for unary operations

typedef void (*unary_op_t)(const ndarray_t *, ndarray_t *);

typedef struct {
  unary_op_t i8, u8, i16, u16, i32, i64, inat, f16, f32, f64, c32, c64, qi8, qu8;
  unary_op_t bool_, bf16, fp8_e4m3, fp8_e5m2, c16;
  // Note: int4/uint4 need special handling due to packing
} unary_op_table;

static value dispatch_unary(value *argv, const unary_op_table *table,
                            const char *op_name) {
  int ndim = Int_val(argv[0]);
  value v_shape = argv[1], v_x = argv[2], v_xstrides = argv[3],
        v_xoffset = argv[4];
  value v_z = argv[5], v_zstrides = argv[6], v_zoffset = argv[7];

  // Sanity check ndim to prevent stack overflow
  if (ndim < 0 || ndim > 32) {
    caml_failwith("dispatch_unary: ndim must be between 0 and 32");
  }

  // Handle 0-dimensional tensors (scalars)
  if (ndim == 0) {
    struct caml_ba_array *ba = Caml_ba_array_val(v_x);
    int kind = ba->flags & CAML_BA_KIND_MASK;

    void *x_data = Caml_ba_data_val(v_x);
    void *z_data = Caml_ba_data_val(v_z);
    int x_offset = Int_val(v_xoffset);
    int z_offset = Int_val(v_zoffset);

    unary_op_t func = NULL;
    switch (kind) {
      case CAML_BA_SINT8:
        func = table->i8;
        break;
      case CAML_BA_UINT8:
        func = table->u8;
        break;
      case CAML_BA_SINT16:
        func = table->i16;
        break;
      case CAML_BA_UINT16:
        func = table->u16;
        break;
      case CAML_BA_INT32:
        func = table->i32;
        break;
      case CAML_BA_INT64:
        func = table->i64;
        break;
      case CAML_BA_CAML_INT:
      case CAML_BA_NATIVE_INT:
        func = table->inat;
        break;
      case CAML_BA_FLOAT16:
        func = table->f16;
        break;
      case CAML_BA_FLOAT32:
        func = table->f32;
        break;
      case CAML_BA_FLOAT64:
        func = table->f64;
        break;
      case CAML_BA_COMPLEX32:
        func = table->c32;
        break;
      case CAML_BA_COMPLEX64:
        func = table->c64;
        break;
      case NX_BA_QINT8:
        func = table->qi8;
        break;
      case NX_BA_QUINT8:
        func = table->qu8;
        break;
      case NX_BA_BOOL:
        func = table->bool_;
        break;
      case NX_BA_BFLOAT16:
        func = table->bf16;
        break;
      case NX_BA_FP8_E4M3:
        func = table->fp8_e4m3;
        break;
      case NX_BA_FP8_E5M2:
        func = table->fp8_e5m2;
        break;
      case NX_BA_COMPLEX16:
        func = table->c16;
        break;
      // Note: NX_BA_INT4 and NX_BA_UINT4 need special handling
    }

    if (func) {
      ndarray_t x_arr = {x_data, 0, NULL, NULL, x_offset};
      ndarray_t z_arr = {z_data, 0, NULL, NULL, z_offset};
      func(&x_arr, &z_arr);
    } else {
      static char err_buf[100];
      snprintf(err_buf, sizeof(err_buf), "%s: unsupported dtype", op_name);
      caml_failwith(err_buf);
    }

    return Val_unit;
  }

  int shape[ndim], x_strides[ndim], z_strides[ndim];
  for (int i = 0; i < ndim; ++i) shape[i] = Int_val(Field(v_shape, i));
  for (int i = 0; i < ndim; ++i) x_strides[i] = Int_val(Field(v_xstrides, i));
  for (int i = 0; i < ndim; ++i) z_strides[i] = Int_val(Field(v_zstrides, i));

  struct caml_ba_array *ba = Caml_ba_array_val(v_x);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  ndarray_t x, z;
  x.data = ba->data;
  x.ndim = ndim;
  x.shape = shape;
  x.strides = x_strides;
  x.offset = Int_val(v_xoffset);
  z.data = Caml_ba_data_val(v_z);
  z.ndim = ndim;
  z.shape = shape;
  z.strides = z_strides;
  z.offset = Int_val(v_zoffset);

  unary_op_t func = NULL;
  switch (kind) {
    case CAML_BA_SINT8:
      func = table->i8;
      break;
    case CAML_BA_UINT8:
      func = table->u8;
      break;
    case CAML_BA_SINT16:
      func = table->i16;
      break;
    case CAML_BA_UINT16:
      func = table->u16;
      break;
    case CAML_BA_INT32:
      func = table->i32;
      break;
    case CAML_BA_INT64:
      func = table->i64;
      break;
    case CAML_BA_CAML_INT:
      func = table->inat;
      break;
    case CAML_BA_NATIVE_INT:
      func = table->inat;
      break;
    case CAML_BA_FLOAT16:
      func = table->f16;
      break;
    case CAML_BA_FLOAT32:
      func = table->f32;
      break;
    case CAML_BA_FLOAT64:
      func = table->f64;
      break;
    case CAML_BA_COMPLEX32:
      func = table->c32;
      break;
    case CAML_BA_COMPLEX64:
      func = table->c64;
      break;
    case NX_BA_QINT8:
      func = table->qi8;
      break;
    case NX_BA_QUINT8:
      func = table->qu8;
      break;
      case NX_BA_BOOL:
        func = table->bool_;
        break;
      case NX_BA_BFLOAT16:
        func = table->bf16;
        break;
      case NX_BA_FP8_E4M3:
        func = table->fp8_e4m3;
        break;
      case NX_BA_FP8_E5M2:
        func = table->fp8_e5m2;
        break;
      case NX_BA_COMPLEX16:
        func = table->c16;
        break;
      // Note: NX_BA_INT4 and NX_BA_UINT4 need special handling
  }

  if (func) {
    caml_enter_blocking_section();
    func(&x, &z);
    caml_leave_blocking_section();
  } else {
    static char err_buf[100];
    snprintf(err_buf, sizeof(err_buf), "%s: unsupported dtype", op_name);
    caml_failwith(err_buf);
  }
  return Val_unit;
}

#define UNARY_STUB(name, ...)                                 \
  CAMLprim value caml_nx_##name##_bc(value *argv, int argn) { \
    static const unary_op_table table = {__VA_ARGS__};        \
    return dispatch_unary(argv, &table, #name);               \
  }

UNARY_STUB(neg, .f16 = nx_c_neg_float16_t, .f32 = nx_c_neg_float,
           .f64 = nx_c_neg_double, .i8 = nx_c_neg_int8_t,
           .i16 = nx_c_neg_int16_t, .i32 = nx_c_neg_int32_t,
           .i64 = nx_c_neg_int64_t, .inat = nx_c_neg_intnat,
           .c32 = nx_c_neg_c32_t, .c64 = nx_c_neg_c64_t,
           .qi8 = nx_c_neg_qint8_t)
NATIVE_WRAPPER_8(neg)

UNARY_STUB(sqrt, .f16 = nx_c_sqrt_float16_t, .f32 = nx_c_sqrt_float,
           .f64 = nx_c_sqrt_double, .c32 = nx_c_sqrt_c32_t,
           .c64 = nx_c_sqrt_c64_t)
NATIVE_WRAPPER_8(sqrt)

UNARY_STUB(sin, .f16 = nx_c_sin_float16_t, .f32 = nx_c_sin_float,
           .f64 = nx_c_sin_double, .c32 = nx_c_sin_c32_t,
           .c64 = nx_c_sin_c64_t)
NATIVE_WRAPPER_8(sin)

UNARY_STUB(exp2, .f16 = nx_c_exp2_float16_t, .f32 = nx_c_exp2_float,
           .f64 = nx_c_exp2_double, .c32 = nx_c_exp2_c32_t,
           .c64 = nx_c_exp2_c64_t)
NATIVE_WRAPPER_8(exp2)

UNARY_STUB(log2, .f16 = nx_c_log2_float16_t, .f32 = nx_c_log2_float,
           .f64 = nx_c_log2_double, .c32 = nx_c_log2_c32_t,
           .c64 = nx_c_log2_c64_t)
NATIVE_WRAPPER_8(log2)

UNARY_STUB(recip, .f16 = nx_c_recip_float16_t, .f32 = nx_c_recip_float,
           .f64 = nx_c_recip_double, .c32 = nx_c_recip_c32_t,
           .c64 = nx_c_recip_c64_t)
NATIVE_WRAPPER_8(recip)

// Table and dispatcher for binary operations

typedef void (*binary_op_t)(const ndarray_t *, const ndarray_t *, ndarray_t *);
typedef struct {
  binary_op_t i8, u8, i16, u16, i32, i64, inat, f16, f32, f64, c32, c64, qi8, qu8;
  binary_op_t bool_, bf16, fp8_e4m3, fp8_e5m2, c16;
  // Note: int4/uint4 need special handling due to packing
} binary_op_table;

static value dispatch_binary(value *argv, const binary_op_table *table,
                             const char *op_name) {
  int ndim = Int_val(argv[0]);
  value v_shape = argv[1], v_x = argv[2], v_xstrides = argv[3],
        v_xoffset = argv[4];
  value v_y = argv[5], v_ystrides = argv[6], v_yoffset = argv[7];
  value v_z = argv[8], v_zstrides = argv[9], v_zoffset = argv[10];

  // Sanity check ndim to prevent stack overflow
  if (ndim < 0 || ndim > 32) {
    caml_failwith("dispatch_binary: ndim must be between 0 and 32");
  }

  // Handle 0-dimensional tensors (scalars)
  if (ndim == 0) {
    struct caml_ba_array *ba = Caml_ba_array_val(v_x);
    int kind = ba->flags & CAML_BA_KIND_MASK;

    void *x_data = Caml_ba_data_val(v_x);
    void *y_data = Caml_ba_data_val(v_y);
    void *z_data = Caml_ba_data_val(v_z);
    int x_offset = Int_val(v_xoffset);
    int y_offset = Int_val(v_yoffset);
    int z_offset = Int_val(v_zoffset);

    binary_op_t func = NULL;
    switch (kind) {
      case CAML_BA_SINT8:
        func = table->i8;
        break;
      case CAML_BA_UINT8:
        func = table->u8;
        break;
      case CAML_BA_SINT16:
        func = table->i16;
        break;
      case CAML_BA_UINT16:
        func = table->u16;
        break;
      case CAML_BA_INT32:
        func = table->i32;
        break;
      case CAML_BA_INT64:
        func = table->i64;
        break;
      case CAML_BA_CAML_INT:
      case CAML_BA_NATIVE_INT:
        func = table->inat;
        break;
      case CAML_BA_FLOAT16:
        func = table->f16;
        break;
      case CAML_BA_FLOAT32:
        func = table->f32;
        break;
      case CAML_BA_FLOAT64:
        func = table->f64;
        break;
      case CAML_BA_COMPLEX32:
        func = table->c32;
        break;
      case CAML_BA_COMPLEX64:
        func = table->c64;
        break;
      case NX_BA_QINT8:
        func = table->qi8;
        break;
      case NX_BA_QUINT8:
        func = table->qu8;
        break;
      case NX_BA_BOOL:
        func = table->bool_;
        break;
      case NX_BA_BFLOAT16:
        func = table->bf16;
        break;
      case NX_BA_FP8_E4M3:
        func = table->fp8_e4m3;
        break;
      case NX_BA_FP8_E5M2:
        func = table->fp8_e5m2;
        break;
      case NX_BA_COMPLEX16:
        func = table->c16;
        break;
      // Note: NX_BA_INT4 and NX_BA_UINT4 need special handling
    }

    if (func) {
      ndarray_t x_arr = {x_data, 0, NULL, NULL, x_offset};
      ndarray_t y_arr = {y_data, 0, NULL, NULL, y_offset};
      ndarray_t z_arr = {z_data, 0, NULL, NULL, z_offset};
      func(&x_arr, &y_arr, &z_arr);
    } else {
      static char err_buf[100];
      snprintf(err_buf, sizeof(err_buf), "%s: unsupported dtype", op_name);
      caml_failwith(err_buf);
    }

    return Val_unit;
  }

  int shape[ndim], x_strides[ndim], y_strides[ndim], z_strides[ndim];
  for (int i = 0; i < ndim; ++i) shape[i] = Int_val(Field(v_shape, i));
  for (int i = 0; i < ndim; ++i) x_strides[i] = Int_val(Field(v_xstrides, i));
  for (int i = 0; i < ndim; ++i) y_strides[i] = Int_val(Field(v_ystrides, i));
  for (int i = 0; i < ndim; ++i) z_strides[i] = Int_val(Field(v_zstrides, i));

  struct caml_ba_array *ba = Caml_ba_array_val(v_x);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  ndarray_t x, y, z;
  x.data = ba->data;
  x.ndim = ndim;
  x.shape = shape;
  x.strides = x_strides;
  x.offset = Int_val(v_xoffset);
  y.data = Caml_ba_data_val(v_y);
  y.ndim = ndim;
  y.shape = shape;
  y.strides = y_strides;
  y.offset = Int_val(v_yoffset);
  z.data = Caml_ba_data_val(v_z);
  z.ndim = ndim;
  z.shape = shape;
  z.strides = z_strides;
  z.offset = Int_val(v_zoffset);

  binary_op_t func = NULL;
  switch (kind) {
    case CAML_BA_SINT8:
      func = table->i8;
      break;
    case CAML_BA_UINT8:
      func = table->u8;
      break;
    case CAML_BA_SINT16:
      func = table->i16;
      break;
    case CAML_BA_UINT16:
      func = table->u16;
      break;
    case CAML_BA_INT32:
      func = table->i32;
      break;
    case CAML_BA_INT64:
      func = table->i64;
      break;
    case CAML_BA_CAML_INT:
    case CAML_BA_NATIVE_INT:
      func = table->inat;
      break;
    case CAML_BA_FLOAT16:
      func = table->f16;
      break;
    case CAML_BA_FLOAT32:
      func = table->f32;
      break;
    case CAML_BA_FLOAT64:
      func = table->f64;
      break;
    case CAML_BA_COMPLEX32:
      func = table->c32;
      break;
    case CAML_BA_COMPLEX64:
      func = table->c64;
      break;
    case NX_BA_QINT8:
      func = table->qi8;
      break;
    case NX_BA_QUINT8:
      func = table->qu8;
      break;
      case NX_BA_BOOL:
        func = table->bool_;
        break;
      case NX_BA_BFLOAT16:
        func = table->bf16;
        break;
      case NX_BA_FP8_E4M3:
        func = table->fp8_e4m3;
        break;
      case NX_BA_FP8_E5M2:
        func = table->fp8_e5m2;
        break;
      case NX_BA_COMPLEX16:
        func = table->c16;
        break;
      // Note: NX_BA_INT4 and NX_BA_UINT4 need special handling
  }

  if (func) {
    caml_enter_blocking_section();
    func(&x, &y, &z);
    caml_leave_blocking_section();
  } else {
    static char err_buf[100];
    snprintf(err_buf, sizeof(err_buf), "%s: unsupported dtype", op_name);
    caml_failwith(err_buf);
  }
  return Val_unit;
}

#define BINARY_STUB(name, ...)                                \
  CAMLprim value caml_nx_##name##_bc(value *argv, int argn) { \
    static const binary_op_table table = {__VA_ARGS__};       \
    return dispatch_binary(argv, &table, #name);              \
  }

// Declarative stub definitions for binary ops
BINARY_STUB(add, .i8 = nx_c_add_int8_t, .u8 = nx_c_add_uint8_t,
            .i16 = nx_c_add_int16_t, .u16 = nx_c_add_uint16_t,
            .i32 = nx_c_add_int32_t, .i64 = nx_c_add_int64_t,
            .inat = nx_c_add_intnat, .f16 = nx_c_add_float16_t,
            .f32 = nx_c_add_float, .f64 = nx_c_add_double,
            .c32 = nx_c_add_c32_t, .c64 = nx_c_add_c64_t,
            .qi8 = nx_c_add_qint8_t, .qu8 = nx_c_add_quint8_t)
NATIVE_WRAPPER_11(add)

BINARY_STUB(sub, .i8 = nx_c_sub_int8_t, .u8 = nx_c_sub_uint8_t,
            .i16 = nx_c_sub_int16_t, .u16 = nx_c_sub_uint16_t,
            .i32 = nx_c_sub_int32_t, .i64 = nx_c_sub_int64_t,
            .inat = nx_c_sub_intnat, .f16 = nx_c_sub_float16_t,
            .f32 = nx_c_sub_float, .f64 = nx_c_sub_double,
            .c32 = nx_c_sub_c32_t, .c64 = nx_c_sub_c64_t,
            .qi8 = nx_c_sub_qint8_t, .qu8 = nx_c_sub_quint8_t)
NATIVE_WRAPPER_11(sub)

BINARY_STUB(mul, .i8 = nx_c_mul_int8_t, .u8 = nx_c_mul_uint8_t,
            .i16 = nx_c_mul_int16_t, .u16 = nx_c_mul_uint16_t,
            .i32 = nx_c_mul_int32_t, .i64 = nx_c_mul_int64_t,
            .inat = nx_c_mul_intnat, .f16 = nx_c_mul_float16_t,
            .f32 = nx_c_mul_float, .f64 = nx_c_mul_double,
            .c32 = nx_c_mul_c32_t, .c64 = nx_c_mul_c64_t,
            .qi8 = nx_c_mul_qint8_t, .qu8 = nx_c_mul_quint8_t)
NATIVE_WRAPPER_11(mul)

BINARY_STUB(fdiv, .f16 = nx_c_fdiv_float16_t, .f32 = nx_c_fdiv_float,
            .f64 = nx_c_fdiv_double, .c32 = nx_c_fdiv_c32_t,
            .c64 = nx_c_fdiv_c64_t)
NATIVE_WRAPPER_11(fdiv)

BINARY_STUB(idiv, .i32 = nx_c_idiv_int32_t, .i64 = nx_c_idiv_int64_t)
NATIVE_WRAPPER_11(idiv)

BINARY_STUB(max, .i8 = nx_c_max_int8_t, .u8 = nx_c_max_uint8_t,
            .i16 = nx_c_max_int16_t, .u16 = nx_c_max_uint16_t,
            .i32 = nx_c_max_int32_t, .i64 = nx_c_max_int64_t,
            .inat = nx_c_max_intnat, .f16 = nx_c_max_float16_t,
            .f32 = nx_c_max_float, .f64 = nx_c_max_double,
            .qi8 = nx_c_max_qint8_t, .qu8 = nx_c_max_quint8_t)
NATIVE_WRAPPER_11(max)

BINARY_STUB(mod, .i8 = nx_c_mod_int8_t, .i16 = nx_c_mod_int16_t,
            .i32 = nx_c_mod_int32_t, .i64 = nx_c_mod_int64_t,
            .inat = nx_c_mod_intnat, .f16 = nx_c_mod_float16_t,
            .f32 = nx_c_mod_float, .f64 = nx_c_mod_double,
            .qi8 = nx_c_mod_qint8_t)
NATIVE_WRAPPER_11(mod)

BINARY_STUB(pow, .f16 = nx_c_pow_float16_t, .f32 = nx_c_pow_float,
            .f64 = nx_c_pow_double)
NATIVE_WRAPPER_11(pow)

BINARY_STUB(xor, .i8 = nx_c_xor_int8_t, .u8 = nx_c_xor_uint8_t,
            .i16 = nx_c_xor_int16_t, .u16 = nx_c_xor_uint16_t,
            .i32 = nx_c_xor_int32_t, .i64 = nx_c_xor_int64_t,
            .inat = nx_c_xor_intnat, .qi8 = nx_c_xor_qint8_t,
            .qu8 = nx_c_xor_quint8_t)
NATIVE_WRAPPER_11(xor)

BINARY_STUB(or, .i8 = nx_c_or_int8_t, .u8 = nx_c_or_uint8_t,
            .i16 = nx_c_or_int16_t, .u16 = nx_c_or_uint16_t,
            .i32 = nx_c_or_int32_t, .i64 = nx_c_or_int64_t,
            .inat = nx_c_or_intnat, .qi8 = nx_c_or_qint8_t,
            .qu8 = nx_c_or_quint8_t)
NATIVE_WRAPPER_11(or)

BINARY_STUB(and, .i8 = nx_c_and_int8_t, .u8 = nx_c_and_uint8_t,
            .i16 = nx_c_and_int16_t, .u16 = nx_c_and_uint16_t,
            .i32 = nx_c_and_int32_t, .i64 = nx_c_and_int64_t,
            .inat = nx_c_and_intnat, .qi8 = nx_c_and_qint8_t,
            .qu8 = nx_c_and_quint8_t)
NATIVE_WRAPPER_11(and)

// Comparison operations
BINARY_STUB(cmplt, .i8 = nx_c_cmplt_int8_t, .u8 = nx_c_cmplt_uint8_t,
            .i16 = nx_c_cmplt_int16_t, .u16 = nx_c_cmplt_uint16_t,
            .i32 = nx_c_cmplt_int32_t, .i64 = nx_c_cmplt_int64_t,
            .inat = nx_c_cmplt_intnat, .f16 = nx_c_cmplt_float16_t,
            .f32 = nx_c_cmplt_float, .f64 = nx_c_cmplt_double,
            .qi8 = nx_c_cmplt_qint8_t, .qu8 = nx_c_cmplt_quint8_t)
NATIVE_WRAPPER_11(cmplt)

BINARY_STUB(cmpne, .i8 = nx_c_cmpne_int8_t, .u8 = nx_c_cmpne_uint8_t,
            .i16 = nx_c_cmpne_int16_t, .u16 = nx_c_cmpne_uint16_t,
            .i32 = nx_c_cmpne_int32_t, .i64 = nx_c_cmpne_int64_t,
            .inat = nx_c_cmpne_intnat, .f16 = nx_c_cmpne_float16_t,
            .f32 = nx_c_cmpne_float, .f64 = nx_c_cmpne_double,
            .c32 = nx_c_cmpne_c32_t, .c64 = nx_c_cmpne_c64_t,
            .qi8 = nx_c_cmpne_qint8_t, .qu8 = nx_c_cmpne_quint8_t)
NATIVE_WRAPPER_11(cmpne)

// Reduce operation dispatch
typedef void (*reduce_op_t)(const ndarray_t *, ndarray_t *, const int *, int);

typedef struct {
  reduce_op_t i8, u8, i16, u16, i32, i64, inat, f16, f32, f64, c32, c64, qi8, qu8;
  reduce_op_t bool_, bf16, fp8_e4m3, fp8_e5m2, c16;
  // Note: int4/uint4 need special handling due to packing
} reduce_op_table;

static value dispatch_reduce(value *argv, const reduce_op_table *table,
                             const char *op_name) {
  int ndim = Int_val(argv[0]);
  value v_shape = argv[1], v_x = argv[2], v_xstrides = argv[3],
        v_xoffset = argv[4];
  value v_z = argv[5], v_zstrides = argv[6], v_zoffset = argv[7];
  value v_axes = argv[8];
  int keepdims = Int_val(argv[9]);

  if (ndim < 1) {
    caml_failwith("dispatch_reduce: ndim must be at least 1");
  }

  int shape[ndim], x_strides[ndim], z_strides[ndim];
  for (int i = 0; i < ndim; ++i) shape[i] = Int_val(Field(v_shape, i));
  for (int i = 0; i < ndim; ++i) x_strides[i] = Int_val(Field(v_xstrides, i));
  for (int i = 0; i < ndim; ++i) z_strides[i] = Int_val(Field(v_zstrides, i));

  // Extract axes
  int num_axes = Wosize_val(v_axes);
  int axes[num_axes];
  for (int i = 0; i < num_axes; i++) {
    axes[i] = Int_val(Field(v_axes, i));
  }

  struct caml_ba_array *ba = Caml_ba_array_val(v_x);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  ndarray_t x, z;
  x.data = ba->data;
  x.ndim = ndim;
  x.shape = shape;
  x.strides = x_strides;
  x.offset = Int_val(v_xoffset);
  z.data = Caml_ba_data_val(v_z);
  z.ndim = keepdims ? ndim : (ndim - num_axes);
  z.shape = shape;  // Will be adjusted by the kernel
  z.strides = z_strides;
  z.offset = Int_val(v_zoffset);

  reduce_op_t func = NULL;
  switch (kind) {
    case CAML_BA_SINT8:
      func = table->i8;
      break;
    case CAML_BA_UINT8:
      func = table->u8;
      break;
    case CAML_BA_SINT16:
      func = table->i16;
      break;
    case CAML_BA_UINT16:
      func = table->u16;
      break;
    case CAML_BA_INT32:
      func = table->i32;
      break;
    case CAML_BA_INT64:
      func = table->i64;
      break;
    case CAML_BA_CAML_INT:
    case CAML_BA_NATIVE_INT:
      func = table->inat;
      break;
    case CAML_BA_FLOAT16:
      func = table->f16;
      break;
    case CAML_BA_FLOAT32:
      func = table->f32;
      break;
    case CAML_BA_FLOAT64:
      func = table->f64;
      break;
    case CAML_BA_COMPLEX32:
      func = table->c32;
      break;
    case CAML_BA_COMPLEX64:
      func = table->c64;
      break;
    case NX_BA_QINT8:
      func = table->qi8;
      break;
    case NX_BA_QUINT8:
      func = table->qu8;
      break;
      case NX_BA_BOOL:
        func = table->bool_;
        break;
      case NX_BA_BFLOAT16:
        func = table->bf16;
        break;
      case NX_BA_FP8_E4M3:
        func = table->fp8_e4m3;
        break;
      case NX_BA_FP8_E5M2:
        func = table->fp8_e5m2;
        break;
      case NX_BA_COMPLEX16:
        func = table->c16;
        break;
      // Note: NX_BA_INT4 and NX_BA_UINT4 need special handling
  }

  if (func) {
    caml_enter_blocking_section();
    func(&x, &z, axes, num_axes);
    caml_leave_blocking_section();
  } else {
    static char err_buf[100];
    snprintf(err_buf, sizeof(err_buf), "%s: unsupported dtype", op_name);
    caml_failwith(err_buf);
  }
  return Val_unit;
}

#define REDUCE_STUB(name, ...)                                       \
  CAMLprim value caml_nx_reduce_##name##_bc(value *argv, int argn) { \
    static const reduce_op_table table = {__VA_ARGS__};              \
    return dispatch_reduce(argv, &table, "reduce_" #name);           \
  }

REDUCE_STUB(sum, .i8 = nx_c_reduce_sum_int8_t_kernel,
            .u8 = nx_c_reduce_sum_uint8_t_kernel,
            .i16 = nx_c_reduce_sum_int16_t_kernel,
            .u16 = nx_c_reduce_sum_uint16_t_kernel,
            .i32 = nx_c_reduce_sum_int32_t_kernel,
            .i64 = nx_c_reduce_sum_int64_t_kernel,
            .inat = nx_c_reduce_sum_intnat_kernel,
            .f16 = nx_c_reduce_sum_float16_t_kernel,
            .f32 = nx_c_reduce_sum_float_kernel,
            .f64 = nx_c_reduce_sum_double_kernel,
            .qi8 = nx_c_reduce_sum_qint8_t_kernel,
            .qu8 = nx_c_reduce_sum_quint8_t_kernel)
REDUCE_NATIVE_WRAPPER_10(sum)

REDUCE_STUB(max, .i8 = nx_c_reduce_max_int8_t_kernel,
            .u8 = nx_c_reduce_max_uint8_t_kernel,
            .i16 = nx_c_reduce_max_int16_t_kernel,
            .u16 = nx_c_reduce_max_uint16_t_kernel,
            .i32 = nx_c_reduce_max_int32_t_kernel,
            .i64 = nx_c_reduce_max_int64_t_kernel,
            .inat = nx_c_reduce_max_intnat_kernel,
            .f16 = nx_c_reduce_max_float16_t_kernel,
            .f32 = nx_c_reduce_max_float_kernel,
            .f64 = nx_c_reduce_max_double_kernel,
            .qi8 = nx_c_reduce_max_qint8_t_kernel,
            .qu8 = nx_c_reduce_max_quint8_t_kernel)
REDUCE_NATIVE_WRAPPER_10(max)

REDUCE_STUB(prod, .i8 = nx_c_reduce_prod_int8_t_kernel,
            .u8 = nx_c_reduce_prod_uint8_t_kernel,
            .i16 = nx_c_reduce_prod_int16_t_kernel,
            .u16 = nx_c_reduce_prod_uint16_t_kernel,
            .i32 = nx_c_reduce_prod_int32_t_kernel,
            .i64 = nx_c_reduce_prod_int64_t_kernel,
            .inat = nx_c_reduce_prod_intnat_kernel,
            .f16 = nx_c_reduce_prod_float16_t_kernel,
            .f32 = nx_c_reduce_prod_float_kernel,
            .f64 = nx_c_reduce_prod_double_kernel,
            .qi8 = nx_c_reduce_prod_qint8_t_kernel,
            .qu8 = nx_c_reduce_prod_quint8_t_kernel)
REDUCE_NATIVE_WRAPPER_10(prod)

// Where operation dispatch
typedef void (*where_op_t)(const ndarray_t *, const ndarray_t *,
                           const ndarray_t *, ndarray_t *);

typedef struct {
  where_op_t i8, u8, i16, u16, i32, i64, inat, f16, f32, f64, c32, c64, qi8, qu8;
  where_op_t bool_, bf16, fp8_e4m3, fp8_e5m2, c16;
  // Note: int4/uint4 need special handling due to packing
} where_op_table;

CAMLprim value caml_nx_where_bc(value *argv, int argn) {
  int ndim = Int_val(argv[0]);
  value v_shape = argv[1];
  value v_cond = argv[2], v_cond_strides = argv[3], v_cond_offset = argv[4];
  value v_x = argv[5], v_xstrides = argv[6], v_xoffset = argv[7];
  value v_y = argv[8], v_ystrides = argv[9], v_yoffset = argv[10];
  value v_z = argv[11], v_zstrides = argv[12], v_zoffset = argv[13];

  if (ndim < 1) {
    caml_failwith("where: ndim must be at least 1");
  }

  int shape[ndim], cond_strides[ndim], x_strides[ndim], y_strides[ndim],
      z_strides[ndim];
  for (int i = 0; i < ndim; ++i) shape[i] = Int_val(Field(v_shape, i));
  for (int i = 0; i < ndim; ++i)
    cond_strides[i] = Int_val(Field(v_cond_strides, i));
  for (int i = 0; i < ndim; ++i) x_strides[i] = Int_val(Field(v_xstrides, i));
  for (int i = 0; i < ndim; ++i) y_strides[i] = Int_val(Field(v_ystrides, i));
  for (int i = 0; i < ndim; ++i) z_strides[i] = Int_val(Field(v_zstrides, i));

  struct caml_ba_array *ba = Caml_ba_array_val(v_x);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  ndarray_t cond, x, y, z;
  cond.data = Caml_ba_data_val(v_cond);
  cond.ndim = ndim;
  cond.shape = shape;
  cond.strides = cond_strides;
  cond.offset = Int_val(v_cond_offset);
  x.data = ba->data;
  x.ndim = ndim;
  x.shape = shape;
  x.strides = x_strides;
  x.offset = Int_val(v_xoffset);
  y.data = Caml_ba_data_val(v_y);
  y.ndim = ndim;
  y.shape = shape;
  y.strides = y_strides;
  y.offset = Int_val(v_yoffset);
  z.data = Caml_ba_data_val(v_z);
  z.ndim = ndim;
  z.shape = shape;
  z.strides = z_strides;
  z.offset = Int_val(v_zoffset);

  where_op_t func = NULL;
  switch (kind) {
    case CAML_BA_SINT8:
      func = nx_c_where_int8_t;
      break;
    case CAML_BA_UINT8:
      func = nx_c_where_uint8_t;
      break;
    case CAML_BA_SINT16:
      func = nx_c_where_int16_t;
      break;
    case CAML_BA_UINT16:
      func = nx_c_where_uint16_t;
      break;
    case CAML_BA_INT32:
      func = nx_c_where_int32_t;
      break;
    case CAML_BA_INT64:
      func = nx_c_where_int64_t;
      break;
    case CAML_BA_CAML_INT:
    case CAML_BA_NATIVE_INT:
      func = nx_c_where_intnat;
      break;
    case CAML_BA_FLOAT16:
      func = nx_c_where_float16_t;
      break;
    case CAML_BA_FLOAT32:
      func = nx_c_where_float;
      break;
    case CAML_BA_FLOAT64:
      func = nx_c_where_double;
      break;
    case CAML_BA_COMPLEX32:
      func = nx_c_where_c32_t;
      break;
    case CAML_BA_COMPLEX64:
      func = nx_c_where_c64_t;
      break;
    case NX_BA_QINT8:
      func = nx_c_where_qint8_t;
      break;
    case NX_BA_QUINT8:
      func = nx_c_where_quint8_t;
      break;
  }

  if (func) {
    caml_enter_blocking_section();
    func(&cond, &x, &y, &z);
    caml_leave_blocking_section();
  } else {
    caml_failwith("where: unsupported dtype");
  }
  return Val_unit;
}

NATIVE_WRAPPER_14(where)

// Pad operation dispatch
typedef void (*pad_op_t)(const ndarray_t *, ndarray_t *, const int *, value);

CAMLprim value caml_nx_pad_bc(value *argv, int argn) {
  int ndim = Int_val(argv[0]);
  value v_input_shape = argv[1], v_x = argv[2], v_xstrides = argv[3],
        v_xoffset = argv[4];
  value v_output_shape = argv[5], v_z = argv[6], v_zstrides = argv[7],
        v_zoffset = argv[8];
  value v_padding = argv[9], v_pad_value = argv[10];

  if (ndim < 1) {
    caml_failwith("pad: ndim must be at least 1");
  }

  int input_shape[ndim], output_shape[ndim], x_strides[ndim], z_strides[ndim];
  int padding[ndim * 2];

  for (int i = 0; i < ndim; ++i) {
    input_shape[i] = Int_val(Field(v_input_shape, i));
    output_shape[i] = Int_val(Field(v_output_shape, i));
    x_strides[i] = Int_val(Field(v_xstrides, i));
    z_strides[i] = Int_val(Field(v_zstrides, i));
  }

  // Extract padding values (pairs of left/right padding for each dimension)
  for (int i = 0; i < ndim * 2; ++i) {
    padding[i] = Int_val(Field(v_padding, i));
  }

  struct caml_ba_array *ba = Caml_ba_array_val(v_x);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  ndarray_t x, z;
  x.data = ba->data;
  x.ndim = ndim;
  x.shape = input_shape;
  x.strides = x_strides;
  x.offset = Int_val(v_xoffset);
  z.data = Caml_ba_data_val(v_z);
  z.ndim = ndim;
  z.shape = output_shape;
  z.strides = z_strides;
  z.offset = Int_val(v_zoffset);

  caml_enter_blocking_section();

  switch (kind) {
    case CAML_BA_INT32:
      nx_c_pad_int32_t(&x, &z, padding, Int32_val(v_pad_value));
      break;
    case CAML_BA_INT64:
      nx_c_pad_int64_t(&x, &z, padding, Int64_val(v_pad_value));
      break;
    case CAML_BA_FLOAT32:
      nx_c_pad_float(&x, &z, padding, Double_val(v_pad_value));
      break;
    case CAML_BA_FLOAT64:
      nx_c_pad_double(&x, &z, padding, Double_val(v_pad_value));
      break;
    case CAML_BA_UINT8:
      nx_c_pad_uint8_t(&x, &z, padding, Int_val(v_pad_value));
      break;
    case CAML_BA_COMPLEX32: {
      float re = (float)Double_field(v_pad_value, 0);
      float im = (float)Double_field(v_pad_value, 1);
      c32_t pad_val = re + im * I;
      nx_c_pad_c32_t(&x, &z, padding, pad_val);
      break;
    }
    case CAML_BA_COMPLEX64: {
      double re = Double_field(v_pad_value, 0);
      double im = Double_field(v_pad_value, 1);
      c64_t pad_val = re + im * I;
      nx_c_pad_c64_t(&x, &z, padding, pad_val);
      break;
    }
    case CAML_BA_FLOAT16:
      nx_c_pad_float16_t(&x, &z, padding, (float16_t)Double_val(v_pad_value));
      break;
    case NX_BA_QINT8:
      nx_c_pad_qint8_t(&x, &z, padding, (qint8_t)Int_val(v_pad_value));
      break;
    case NX_BA_QUINT8:
      nx_c_pad_quint8_t(&x, &z, padding, (quint8_t)Int_val(v_pad_value));
      break;
    default:
      caml_leave_blocking_section();
      caml_failwith("pad: unsupported dtype");
  }

  caml_leave_blocking_section();
  return Val_unit;
}

NATIVE_WRAPPER_11(pad)

// Cat operation dispatch
CAMLprim value caml_nx_cat_bc(value *argv, int argn) {
  value v_inputs = argv[0];
  int axis = Int_val(argv[1]);
  value v_output = argv[2], v_output_strides = argv[3],
        v_output_offset = argv[4];
  value v_output_shape = argv[5];

  int num_inputs = Wosize_val(v_inputs);
  if (num_inputs == 0) return Val_unit;

  // Get first input to determine ndim and dtype
  value first_input = Field(v_inputs, 0);
  value v_first_buffer = Field(first_input, 0);
  value v_first_view = Field(first_input, 1);
  value v_first_shape = Field(v_first_view, 0);
  int ndim = Wosize_val(v_first_shape);

  struct caml_ba_array *ba = Caml_ba_array_val(v_first_buffer);
  int kind = ba->flags & CAML_BA_KIND_MASK;
  size_t elem_size = get_element_size(kind);

  // Allocate array of ndarray_t pointers
  const ndarray_t **inputs = malloc(num_inputs * sizeof(ndarray_t *));
  ndarray_t *input_structs = malloc(num_inputs * sizeof(ndarray_t));

  if (!inputs || !input_structs) {
    free(inputs);
    free(input_structs);
    caml_failwith("cat: failed to allocate memory");
  }

  // Build input ndarray_t structures
  for (int i = 0; i < num_inputs; i++) {
    value input = Field(v_inputs, i);
    value v_buffer = Field(input, 0);
    value v_view = Field(input, 1);
    value v_shape = Field(v_view, 0);
    value v_strides = Field(v_view, 1);
    value v_offset = Field(v_view, 2);

    input_structs[i].data = Caml_ba_data_val(v_buffer);
    input_structs[i].ndim = ndim;
    input_structs[i].shape = malloc(ndim * sizeof(int));
    input_structs[i].strides = malloc(ndim * sizeof(int));

    if (!input_structs[i].shape || !input_structs[i].strides) {
      // Clean up all previously allocated memory
      for (int j = 0; j <= i; j++) {
        free((void *)input_structs[j].shape);
        free((void *)input_structs[j].strides);
      }
      free(input_structs);
      free(inputs);
      caml_failwith("cat: failed to allocate memory for shape/strides");
    }

    input_structs[i].offset = Int_val(v_offset);

    for (int d = 0; d < ndim; d++) {
      ((int *)input_structs[i].shape)[d] = Int_val(Field(v_shape, d));
      ((int *)input_structs[i].strides)[d] = Int_val(Field(v_strides, d));
    }

    inputs[i] = &input_structs[i];
  }

  // Build output ndarray_t
  ndarray_t output;
  output.data = Caml_ba_data_val(v_output);
  output.ndim = ndim;
  output.shape = malloc(ndim * sizeof(int));
  output.strides = malloc(ndim * sizeof(int));

  if (!output.shape || !output.strides) {
    // Clean up all previously allocated memory
    for (int i = 0; i < num_inputs; i++) {
      free((void *)input_structs[i].shape);
      free((void *)input_structs[i].strides);
    }
    free(input_structs);
    free(inputs);
    if (output.shape) free((void *)output.shape);
    if (output.strides) free((void *)output.strides);
    caml_failwith("cat: failed to allocate memory for output shape/strides");
  }

  output.offset = Int_val(v_output_offset);

  for (int d = 0; d < ndim; d++) {
    ((int *)output.shape)[d] = Int_val(Field(v_output_shape, d));
    ((int *)output.strides)[d] = Int_val(Field(v_output_strides, d));
  }

  caml_enter_blocking_section();
  nx_c_cat_generic(inputs, num_inputs, &output, axis, elem_size);
  caml_leave_blocking_section();

  // Free allocated memory
  for (int i = 0; i < num_inputs; i++) {
    free((void *)input_structs[i].shape);
    free((void *)input_structs[i].strides);
  }
  free(input_structs);
  free(inputs);
  free((void *)output.shape);
  free((void *)output.strides);

  return Val_unit;
}

CAMLprim value caml_nx_cat(value v1, value v2, value v3, value v4, value v5,
                           value v6) {
  value argv[6] = {v1, v2, v3, v4, v5, v6};
  return caml_nx_cat_bc(argv, 6);
}

// Threefry random number generator implementation
static const uint32_t KS_PARITY_32 = 0x1BD11BDA;
static const int R_2X32[8] = {13, 15, 26, 6, 17, 29, 16, 24};

static inline uint32_t rotl32(uint32_t x, int n) {
  n &= 31;
  return (x << n) | (x >> (32 - n));
}

static void threefry2x32_20(uint32_t c0, uint32_t c1, uint32_t k0, uint32_t k1,
                            uint32_t *out0, uint32_t *out1) {
  uint32_t x0 = c0, x1 = c1;
  uint32_t keys[3] = {k0, k1, KS_PARITY_32 ^ k0 ^ k1};

  for (int r = 0; r < 20; r++) {
    if (r % 4 == 0) {
      int s_div_4 = r / 4;
      x0 += keys[s_div_4 % 3];
      x1 += keys[(s_div_4 + 1) % 3];
      x1 += s_div_4;
    }
    x0 += x1;
    x1 = rotl32(x1, R_2X32[r % 8]);
    x1 ^= x0;
  }

  int s_div_4_final = 20 / 4;
  x0 += keys[s_div_4_final % 3];
  x1 += keys[(s_div_4_final + 1) % 3];
  x1 += s_div_4_final;

  *out0 = x0;
  *out1 = x1;
}

static void nx_c_threefry(const ndarray_t *data, const ndarray_t *seed,
                          const ndarray_t *out) {
  const uint32_t c1_fixed = 0;
  const uint32_t k1_fixed = 0xCAFEBABE;

  long total_size = 1;
  for (int i = 0; i < out->ndim; i++) {
    total_size *= out->shape[i];
  }

  if (total_size == 0) return;

  // Check if arrays are contiguous
  int data_contig = is_c_contiguous(data);
  int seed_contig = is_c_contiguous(seed);
  int out_contig = is_c_contiguous(out);

  if (data_contig && seed_contig && out_contig) {
    // Fast path for contiguous arrays
    const uint32_t *data_ptr = (const uint32_t *)data->data + data->offset;
    const uint32_t *seed_ptr = (const uint32_t *)seed->data + seed->offset;
    uint32_t *out_ptr = (uint32_t *)out->data + out->offset;

#pragma omp parallel for if (total_size > 10000)
    for (long i = 0; i < total_size; i++) {
      uint32_t res0, res1;
      threefry2x32_20(data_ptr[i], c1_fixed, seed_ptr[i], k1_fixed, &res0,
                      &res1);
      out_ptr[i] = res0;  // Only use first output
    }
  } else {
    // Non-contiguous path using div/mod calculation
    const uint32_t *data_ptr = (const uint32_t *)data->data;
    const uint32_t *seed_ptr = (const uint32_t *)seed->data;
    uint32_t *out_ptr = (uint32_t *)out->data;

#pragma omp parallel for if (total_size > 10000)
    for (long i = 0; i < total_size; i++) {
      // Stack-allocated index arrays
      int data_idx[out->ndim], seed_idx[out->ndim], out_idx[out->ndim];

      // Unravel linear index to multi-dimensional indices
      long temp = i;
      for (int d = out->ndim - 1; d >= 0; d--) {
        out_idx[d] = temp % out->shape[d];
        // For threefry, all shapes are the same
        data_idx[d] = out_idx[d];
        seed_idx[d] = out_idx[d];
        temp /= out->shape[d];
      }

      // Calculate strided offsets
      long data_offset = data->offset;
      long seed_offset = seed->offset;
      long out_offset = out->offset;
      for (int d = 0; d < out->ndim; d++) {
        data_offset += data_idx[d] * data->strides[d];
        seed_offset += seed_idx[d] * seed->strides[d];
        out_offset += out_idx[d] * out->strides[d];
      }

      uint32_t res0, res1;
      threefry2x32_20(data_ptr[data_offset], c1_fixed, seed_ptr[seed_offset],
                      k1_fixed, &res0, &res1);
      out_ptr[out_offset] = res0;
    }
  }
}

// Gather operation implementation
#define DEFINE_GATHER_OP(CTYPE, ITYPE)                                         \
  static void nx_c_gather_##CTYPE(const ndarray_t *data,                       \
                                  const ndarray_t *indices,                    \
                                  const ndarray_t *out, int axis) {            \
    long total_size = 1;                                                       \
    for (int i = 0; i < out->ndim; i++) {                                      \
      total_size *= out->shape[i];                                             \
    }                                                                          \
    if (total_size == 0) return;                                               \
                                                                               \
    const CTYPE *data_ptr = (const CTYPE *)data->data;                         \
    const ITYPE *indices_ptr = (const ITYPE *)indices->data;                   \
    CTYPE *out_ptr = (CTYPE *)out->data;                                       \
                                                                               \
    int data_size_at_axis = data->shape[axis];                                 \
                                                                               \
    _Pragma("omp parallel if(total_size > 10000)") {                           \
      /* Stack allocation for thread-local index arrays */                     \
      int md_idx[out->ndim > 0 ? out->ndim : 1];                               \
      int src_idx[data->ndim > 0 ? data->ndim : 1];                            \
                                                                               \
      _Pragma("omp for") for (long linear_idx = 0; linear_idx < total_size;    \
                              linear_idx++) {                                  \
        /* Unravel linear index to multi-dimensional index */                  \
        long temp = linear_idx;                                                \
        for (int d = out->ndim - 1; d >= 0; d--) {                             \
          md_idx[d] = temp % out->shape[d];                                    \
          temp /= out->shape[d];                                               \
        }                                                                      \
                                                                               \
        /* Get index value from indices tensor */                              \
        long indices_offset = indices->offset;                                 \
        for (int d = 0; d < indices->ndim; d++) {                              \
          indices_offset += md_idx[d] * indices->strides[d];                   \
        }                                                                      \
        int idx_value = (int)indices_ptr[indices_offset];                      \
                                                                               \
        /* Handle negative indices and clamp */                                \
        int normalized_idx =                                                   \
            idx_value < 0 ? idx_value + data_size_at_axis : idx_value;         \
        int clamped_idx =                                                      \
            normalized_idx < 0                                                 \
                ? 0                                                            \
                : (normalized_idx >= data_size_at_axis ? data_size_at_axis - 1 \
                                                       : normalized_idx);      \
                                                                               \
        /* Build source index */                                               \
        for (int d = 0; d < out->ndim; d++) {                                  \
          src_idx[d] = md_idx[d];                                              \
        }                                                                      \
        src_idx[axis] = clamped_idx;                                           \
                                                                               \
        /* Calculate data offset and copy value */                             \
        long data_offset = data->offset;                                       \
        for (int d = 0; d < data->ndim; d++) {                                 \
          data_offset += src_idx[d] * data->strides[d];                        \
        }                                                                      \
                                                                               \
        /* Calculate output offset and write */                                \
        long out_offset = out->offset;                                         \
        for (int d = 0; d < out->ndim; d++) {                                  \
          out_offset += md_idx[d] * out->strides[d];                           \
        }                                                                      \
                                                                               \
        out_ptr[out_offset] = data_ptr[data_offset];                           \
      }                                                                        \
    }                                                                          \
  }

DEFINE_GATHER_OP(int8_t, int32_t)
DEFINE_GATHER_OP(uint8_t, int32_t)
DEFINE_GATHER_OP(int16_t, int32_t)
DEFINE_GATHER_OP(uint16_t, int32_t)
DEFINE_GATHER_OP(int32_t, int32_t)
DEFINE_GATHER_OP(int64_t, int32_t)
DEFINE_GATHER_OP(intnat, int32_t)
DEFINE_GATHER_OP(float16_t, int32_t)
DEFINE_GATHER_OP(float, int32_t)
DEFINE_GATHER_OP(double, int32_t)
DEFINE_GATHER_OP(c32_t, int32_t)
DEFINE_GATHER_OP(c64_t, int32_t)
DEFINE_GATHER_OP(qint8_t, int32_t)
DEFINE_GATHER_OP(quint8_t, int32_t)

// Scatter computation modes
typedef enum { SCATTER_REPLACE = 0, SCATTER_ADD = 1 } scatter_computation_t;

// Scatter operation implementation
#define DEFINE_SCATTER_OP(CTYPE, ITYPE)                                      \
  static void nx_c_scatter_##CTYPE(                                          \
      const ndarray_t *template, const ndarray_t *indices,                   \
      const ndarray_t *updates, const ndarray_t *out, int axis,              \
      scatter_computation_t computation) {                                   \
    /* First copy template to output (already done in OCaml) */              \
    /* Now scatter updates */                                                \
    long updates_size = 1;                                                   \
    for (int i = 0; i < updates->ndim; i++) {                                \
      updates_size *= updates->shape[i];                                     \
    }                                                                        \
    if (updates_size == 0) return;                                           \
    /* Debug: Print scatter info */                                          \
    if (0) { /* Enable for debugging */                                      \
      printf("Scatter: updates_size=%ld, mode=%d, axis=%d\\n", updates_size, \
             computation, axis);                                             \
      printf("  updates shape: ");                                           \
      for (int i = 0; i < updates->ndim; i++) {                              \
        printf("%d ", updates->shape[i]);                                    \
      }                                                                      \
      printf("\\n");                                                         \
      printf("  indices shape: ");                                           \
      for (int i = 0; i < indices->ndim; i++) {                              \
        printf("%d ", indices->shape[i]);                                    \
      }                                                                      \
      printf("\\n");                                                         \
    }                                                                        \
                                                                             \
    const ITYPE *indices_ptr = (const ITYPE *)indices->data;                 \
    const CTYPE *updates_ptr = (const CTYPE *)updates->data;                 \
    CTYPE *out_ptr = (CTYPE *)out->data;                                     \
                                                                             \
    int template_size_at_axis = template->shape[axis];                       \
                                                                             \
    /* Stack allocation for work arrays */                                   \
    int md_idx[updates->ndim > 0 ? updates->ndim : 1];                       \
    int dst_idx[template->ndim > 0 ? template->ndim : 1];                    \
                                                                             \
    /* Process each update sequentially (order matters for scatter) */       \
    for (long linear_idx = 0; linear_idx < updates_size; linear_idx++) {     \
      /* Unravel linear index */                                             \
      long temp = linear_idx;                                                \
      for (int d = updates->ndim - 1; d >= 0; d--) {                         \
        md_idx[d] = temp % updates->shape[d];                                \
        temp /= updates->shape[d];                                           \
      }                                                                      \
                                                                             \
      /* Get index value */                                                  \
      long indices_offset = indices->offset;                                 \
      for (int d = 0; d < indices->ndim; d++) {                              \
        indices_offset += md_idx[d] * indices->strides[d];                   \
      }                                                                      \
      int idx_value = (int)indices_ptr[indices_offset];                      \
                                                                             \
      /* Handle negative indices */                                          \
      int normalized_idx =                                                   \
          idx_value < 0 ? idx_value + template_size_at_axis : idx_value;     \
                                                                             \
      /* Check bounds - skip this update if out of bounds */                 \
      if (normalized_idx < 0 || normalized_idx >= template_size_at_axis) {   \
        continue;                                                            \
      }                                                                      \
                                                                             \
      /* Build destination index */                                          \
      int i_updates = 0;                                                     \
      for (int i_template = 0; i_template < template->ndim; i_template++) {  \
        if (i_template == axis) {                                            \
          dst_idx[i_template] = normalized_idx;                              \
        } else {                                                             \
          dst_idx[i_template] = md_idx[i_updates++];                         \
        }                                                                    \
      }                                                                      \
                                                                             \
      /* Get update value */                                                 \
      long updates_offset = updates->offset;                                 \
      for (int d = 0; d < updates->ndim; d++) {                              \
        updates_offset += md_idx[d] * updates->strides[d];                   \
      }                                                                      \
                                                                             \
      /* Calculate output offset and write */                                \
      long out_offset = out->offset;                                         \
      for (int d = 0; d < out->ndim; d++) {                                  \
        out_offset += dst_idx[d] * out->strides[d];                          \
      }                                                                      \
                                                                             \
      switch (computation) {                                                 \
        case SCATTER_REPLACE:                                                \
          out_ptr[out_offset] = updates_ptr[updates_offset];                 \
          break;                                                             \
        case SCATTER_ADD:                                                    \
          out_ptr[out_offset] += updates_ptr[updates_offset];                \
          break;                                                             \
      }                                                                      \
    }                                                                        \
  }

DEFINE_SCATTER_OP(int8_t, int32_t)
DEFINE_SCATTER_OP(uint8_t, int32_t)
DEFINE_SCATTER_OP(int16_t, int32_t)
DEFINE_SCATTER_OP(uint16_t, int32_t)
DEFINE_SCATTER_OP(int32_t, int32_t)
DEFINE_SCATTER_OP(int64_t, int32_t)
DEFINE_SCATTER_OP(intnat, int32_t)
DEFINE_SCATTER_OP(float16_t, int32_t)
DEFINE_SCATTER_OP(float, int32_t)
DEFINE_SCATTER_OP(double, int32_t)
DEFINE_SCATTER_OP(c32_t, int32_t)
DEFINE_SCATTER_OP(c64_t, int32_t)
DEFINE_SCATTER_OP(qint8_t, int32_t)
DEFINE_SCATTER_OP(quint8_t, int32_t)

// Threefry dispatch
CAMLprim value caml_nx_threefry_bc(value *argv, int argn) {
  int ndim = Int_val(argv[0]);
  value v_shape = argv[1];
  value v_x = argv[2], v_xstrides = argv[3], v_xoffset = argv[4];
  value v_y = argv[5], v_ystrides = argv[6], v_yoffset = argv[7];
  value v_z = argv[8], v_zstrides = argv[9], v_zoffset = argv[10];

  if (ndim < 1) {
    caml_failwith("threefry: ndim must be at least 1");
  }

  int shape[ndim], x_strides[ndim], y_strides[ndim], z_strides[ndim];
  for (int i = 0; i < ndim; ++i) shape[i] = Int_val(Field(v_shape, i));
  for (int i = 0; i < ndim; ++i) x_strides[i] = Int_val(Field(v_xstrides, i));
  for (int i = 0; i < ndim; ++i) y_strides[i] = Int_val(Field(v_ystrides, i));
  for (int i = 0; i < ndim; ++i) z_strides[i] = Int_val(Field(v_zstrides, i));

  struct caml_ba_array *ba_x = Caml_ba_array_val(v_x);
  struct caml_ba_array *ba_y = Caml_ba_array_val(v_y);
  struct caml_ba_array *ba_z = Caml_ba_array_val(v_z);

  ndarray_t x, y, z;
  x.data = ba_x->data;
  x.ndim = ndim;
  x.shape = shape;
  x.strides = x_strides;
  x.offset = Int_val(v_xoffset);

  y.data = ba_y->data;
  y.ndim = ndim;
  y.shape = shape;
  y.strides = y_strides;
  y.offset = Int_val(v_yoffset);

  z.data = ba_z->data;
  z.ndim = ndim;
  z.shape = shape;
  z.strides = z_strides;
  z.offset = Int_val(v_zoffset);

  // Threefry only supports int32
  int kind = ba_x->flags & CAML_BA_KIND_MASK;

  if (kind != CAML_BA_INT32) {
    caml_failwith("threefry: only int32 dtype supported");
  }

  caml_enter_blocking_section();
  nx_c_threefry(&x, &y, &z);
  caml_leave_blocking_section();

  return Val_unit;
}

NATIVE_WRAPPER_11(threefry)

// Gather dispatch
CAMLprim value caml_nx_gather_bc(value *argv, int argn) {
  int ndim = Int_val(argv[0]);
  value v_data_shape = argv[1];
  value v_data = argv[2], v_data_strides = argv[3], v_data_offset = argv[4];
  value v_indices = argv[5], v_indices_strides = argv[6],
        v_indices_offset = argv[7];
  int axis = Int_val(argv[8]);
  value v_z = argv[9], v_zstrides = argv[10], v_zoffset = argv[11];

  if (ndim < 1) {
    caml_failwith("gather: ndim must be at least 1");
  }

  // Get shapes - data and indices can have different shapes
  int data_shape[ndim], indices_shape[ndim], data_strides[ndim],
      indices_strides[ndim], z_strides[ndim];
  for (int i = 0; i < ndim; ++i)
    data_shape[i] = Int_val(Field(v_data_shape, i));

  // For gather, the output shape is the indices shape, get it from z's bigarray
  struct caml_ba_array *ba_z = Caml_ba_array_val(v_z);
  for (int i = 0; i < ndim; ++i) {
    indices_shape[i] = ba_z->dim[i];
  }

  for (int i = 0; i < ndim; ++i)
    data_strides[i] = Int_val(Field(v_data_strides, i));
  for (int i = 0; i < ndim; ++i)
    indices_strides[i] = Int_val(Field(v_indices_strides, i));
  for (int i = 0; i < ndim; ++i) z_strides[i] = Int_val(Field(v_zstrides, i));

  struct caml_ba_array *ba_data = Caml_ba_array_val(v_data);
  struct caml_ba_array *ba_indices = Caml_ba_array_val(v_indices);

  ndarray_t data, indices, z;
  data.data = ba_data->data;
  data.ndim = ndim;
  data.shape = data_shape;
  data.strides = data_strides;
  data.offset = Int_val(v_data_offset);

  indices.data = ba_indices->data;
  indices.ndim = ndim;
  indices.shape = indices_shape;
  indices.strides = indices_strides;
  indices.offset = Int_val(v_indices_offset);

  z.data = ba_z->data;
  z.ndim = ndim;
  z.shape = indices_shape;  // Output has shape of indices
  z.strides = z_strides;
  z.offset = Int_val(v_zoffset);

  int kind = ba_data->flags & CAML_BA_KIND_MASK;

  caml_enter_blocking_section();
  switch (kind) {
    case CAML_BA_SINT8:
      nx_c_gather_int8_t(&data, &indices, &z, axis);
      break;
    case CAML_BA_UINT8:
      nx_c_gather_uint8_t(&data, &indices, &z, axis);
      break;
    case CAML_BA_SINT16:
      nx_c_gather_int16_t(&data, &indices, &z, axis);
      break;
    case CAML_BA_UINT16:
      nx_c_gather_uint16_t(&data, &indices, &z, axis);
      break;
    case CAML_BA_INT32:
      nx_c_gather_int32_t(&data, &indices, &z, axis);
      break;
    case CAML_BA_INT64:
      nx_c_gather_int64_t(&data, &indices, &z, axis);
      break;
    case CAML_BA_CAML_INT:
    case CAML_BA_NATIVE_INT:
      nx_c_gather_intnat(&data, &indices, &z, axis);
      break;
    case CAML_BA_FLOAT16:
      nx_c_gather_float16_t(&data, &indices, &z, axis);
      break;
    case CAML_BA_FLOAT32:
      nx_c_gather_float(&data, &indices, &z, axis);
      break;
    case CAML_BA_FLOAT64:
      nx_c_gather_double(&data, &indices, &z, axis);
      break;
    case CAML_BA_COMPLEX32:
      nx_c_gather_c32_t(&data, &indices, &z, axis);
      break;
    case CAML_BA_COMPLEX64:
      nx_c_gather_c64_t(&data, &indices, &z, axis);
      break;
    case NX_BA_QINT8:
      nx_c_gather_qint8_t(&data, &indices, &z, axis);
      break;
    case NX_BA_QUINT8:
      nx_c_gather_quint8_t(&data, &indices, &z, axis);
      break;
    default:
      caml_leave_blocking_section();
      caml_failwith("gather: unsupported dtype");
  }

  caml_leave_blocking_section();
  return Val_unit;
}

CAMLprim value caml_nx_gather(value v1, value v2, value v3, value v4, value v5,
                              value v6, value v7, value v8, value v9, value v10,
                              value v11, value v12) {
  value argv[12] = {v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12};
  return caml_nx_gather_bc(argv, 12);
}

// Scatter dispatch
CAMLprim value caml_nx_scatter_bc(value *argv, int argn) {
  int template_ndim = Int_val(argv[0]);
  value v_template_shape = argv[1];
  int indices_ndim = Int_val(argv[2]);
  value v_indices_shape = argv[3];
  value v_template = argv[4], v_template_strides = argv[5],
        v_template_offset = argv[6];
  value v_indices = argv[7], v_indices_strides = argv[8],
        v_indices_offset = argv[9];
  value v_updates = argv[10], v_updates_strides = argv[11],
        v_updates_offset = argv[12];
  int axis = Int_val(argv[13]);
  value v_output = argv[14], v_output_strides = argv[15],
        v_output_offset = argv[16];
  int computation_mode = Int_val(argv[17]);

  if (template_ndim < 1 || indices_ndim < 1) {
    caml_failwith("scatter: ndim must be at least 1");
  }

  // Get shapes for template and updates/indices
  int template_shape[template_ndim], indices_shape[indices_ndim];
  for (int i = 0; i < template_ndim; ++i)
    template_shape[i] = Int_val(Field(v_template_shape, i));
  for (int i = 0; i < indices_ndim; ++i)
    indices_shape[i] = Int_val(Field(v_indices_shape, i));

  int template_strides[template_ndim], indices_strides[indices_ndim],
      updates_strides[indices_ndim], output_strides[template_ndim];
  for (int i = 0; i < template_ndim; ++i)
    template_strides[i] = Int_val(Field(v_template_strides, i));
  for (int i = 0; i < indices_ndim; ++i)
    indices_strides[i] = Int_val(Field(v_indices_strides, i));
  for (int i = 0; i < indices_ndim; ++i)
    updates_strides[i] = Int_val(Field(v_updates_strides, i));
  for (int i = 0; i < template_ndim; ++i)
    output_strides[i] = Int_val(Field(v_output_strides, i));

  struct caml_ba_array *ba_template = Caml_ba_array_val(v_template);
  struct caml_ba_array *ba_indices = Caml_ba_array_val(v_indices);
  struct caml_ba_array *ba_updates = Caml_ba_array_val(v_updates);
  struct caml_ba_array *ba_output = Caml_ba_array_val(v_output);

  ndarray_t template_arr, indices_arr, updates_arr, output_arr;
  template_arr.data = ba_template->data;
  template_arr.ndim = template_ndim;
  template_arr.shape = template_shape;
  template_arr.strides = template_strides;
  template_arr.offset = Int_val(v_template_offset);

  indices_arr.data = ba_indices->data;
  indices_arr.ndim = indices_ndim;
  indices_arr.shape = indices_shape;
  indices_arr.strides = indices_strides;
  indices_arr.offset = Int_val(v_indices_offset);

  updates_arr.data = ba_updates->data;
  updates_arr.ndim = indices_ndim;    // Updates have same shape as indices
  updates_arr.shape = indices_shape;  // Same shape as indices
  updates_arr.strides = updates_strides;
  updates_arr.offset = Int_val(v_updates_offset);

  output_arr.data = ba_output->data;
  output_arr.ndim = template_ndim;
  output_arr.shape = template_shape;  // Output has shape of template
  output_arr.strides = output_strides;
  output_arr.offset = Int_val(v_output_offset);

  struct caml_ba_array *ba = Caml_ba_array_val(v_template);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  caml_enter_blocking_section();
  switch (kind) {
    case CAML_BA_SINT8:
      nx_c_scatter_int8_t(&template_arr, &indices_arr, &updates_arr,
                          &output_arr, axis,
                          (scatter_computation_t)computation_mode);
      break;
    case CAML_BA_UINT8:
      nx_c_scatter_uint8_t(&template_arr, &indices_arr, &updates_arr,
                           &output_arr, axis,
                           (scatter_computation_t)computation_mode);
      break;
    case CAML_BA_SINT16:
      nx_c_scatter_int16_t(&template_arr, &indices_arr, &updates_arr,
                           &output_arr, axis,
                           (scatter_computation_t)computation_mode);
      break;
    case CAML_BA_UINT16:
      nx_c_scatter_uint16_t(&template_arr, &indices_arr, &updates_arr,
                            &output_arr, axis,
                            (scatter_computation_t)computation_mode);
      break;
    case CAML_BA_INT32:
      nx_c_scatter_int32_t(&template_arr, &indices_arr, &updates_arr,
                           &output_arr, axis,
                           (scatter_computation_t)computation_mode);
      break;
    case CAML_BA_INT64:
      nx_c_scatter_int64_t(&template_arr, &indices_arr, &updates_arr,
                           &output_arr, axis,
                           (scatter_computation_t)computation_mode);
      break;
    case CAML_BA_CAML_INT:
    case CAML_BA_NATIVE_INT:
      nx_c_scatter_intnat(&template_arr, &indices_arr, &updates_arr,
                          &output_arr, axis,
                          (scatter_computation_t)computation_mode);
      break;
    case CAML_BA_FLOAT16:
      nx_c_scatter_float16_t(&template_arr, &indices_arr, &updates_arr,
                             &output_arr, axis,
                             (scatter_computation_t)computation_mode);
      break;
    case CAML_BA_FLOAT32:
      nx_c_scatter_float(&template_arr, &indices_arr, &updates_arr, &output_arr,
                         axis, (scatter_computation_t)computation_mode);
      break;
    case CAML_BA_FLOAT64:
      nx_c_scatter_double(&template_arr, &indices_arr, &updates_arr,
                          &output_arr, axis,
                          (scatter_computation_t)computation_mode);
      break;
    case CAML_BA_COMPLEX32:
      nx_c_scatter_c32_t(&template_arr, &indices_arr, &updates_arr, &output_arr,
                         axis, (scatter_computation_t)computation_mode);
      break;
    case CAML_BA_COMPLEX64:
      nx_c_scatter_c64_t(&template_arr, &indices_arr, &updates_arr, &output_arr,
                         axis, (scatter_computation_t)computation_mode);
      break;
    case NX_BA_QINT8:
      nx_c_scatter_qint8_t(&template_arr, &indices_arr, &updates_arr,
                           &output_arr, axis,
                           (scatter_computation_t)computation_mode);
      break;
    case NX_BA_QUINT8:
      nx_c_scatter_quint8_t(&template_arr, &indices_arr, &updates_arr,
                            &output_arr, axis,
                            (scatter_computation_t)computation_mode);
      break;
    default:
      caml_leave_blocking_section();
      caml_failwith("scatter: unsupported dtype");
  }

  caml_leave_blocking_section();
  return Val_unit;
}

CAMLprim value caml_nx_scatter(value v1, value v2, value v3, value v4, value v5,
                               value v6, value v7, value v8, value v9,
                               value v10, value v11, value v12, value v13,
                               value v14, value v15, value v16, value v17,
                               value v18) {
  value argv[18] = {v1,  v2,  v3,  v4,  v5,  v6,  v7,  v8,  v9,
                    v10, v11, v12, v13, v14, v15, v16, v17, v18};
  return caml_nx_scatter_bc(argv, 18);
}

// Tiled Matrix Multiplication

// Define cache-friendly block sizes. These can be tuned for specific
// architectures. Good starting values are often related to L1/L2 cache sizes.
#define GEMM_BLOCK_M 64
#define GEMM_BLOCK_N 64
#define GEMM_BLOCK_K 64

// Helper to compute min of two integers
static inline int min(int a, int b) { return a < b ? a : b; }

// Generic, tiled, parallel matmul implementation.
// This supports N-D tensors, batching, broadcasting, and arbitrary strides.
#define DEFINE_MATMUL_OP(T, name)                                              \
  static void nx_c_matmul_##name(const ndarray_t *a, const ndarray_t *b,       \
                                 ndarray_t *c) {                               \
    /* Get matrix dimensions */                                                \
    const int ndim_c = c->ndim;                                                \
    const int M = c->shape[ndim_c - 2];                                        \
    const int N = c->shape[ndim_c - 1];                                        \
    const int K = a->shape[a->ndim - 1];                                       \
                                                                               \
    /* Get strides for the matrix dimensions (last two) */                     \
    const long a_row_stride = a->strides[a->ndim - 2];                         \
    const long a_col_stride = a->strides[a->ndim - 1];                         \
    const long b_row_stride = b->strides[b->ndim - 2];                         \
    const long b_col_stride = b->strides[b->ndim - 1];                         \
    const long c_row_stride = c->strides[c->ndim - 2];                         \
    const long c_col_stride = c->strides[c->ndim - 1];                         \
                                                                               \
    /* Calculate total number of batches */                                    \
    long batch_count = 1;                                                      \
    const int num_batch_dims = ndim_c - 2;                                     \
    for (int i = 0; i < num_batch_dims; i++) {                                 \
      batch_count *= c->shape[i];                                              \
    }                                                                          \
                                                                               \
    if (batch_count == 0 || M == 0 || N == 0) return;                          \
                                                                               \
    /* Pre-calculate strides for decoding linear batch index */                \
    long c_batch_decoding_strides[num_batch_dims > 0 ? num_batch_dims : 1];    \
    if (num_batch_dims > 0) {                                                  \
      c_batch_decoding_strides[num_batch_dims - 1] = 1;                        \
      for (int i = num_batch_dims - 2; i >= 0; i--) {                          \
        c_batch_decoding_strides[i] =                                          \
            c_batch_decoding_strides[i + 1] * c->shape[i + 1];                 \
      }                                                                        \
    }                                                                          \
                                                                               \
    /* Parallelize over batches, which are independent */                      \
    _Pragma("omp parallel for") for (long batch_idx = 0;                       \
                                     batch_idx < batch_count; batch_idx++) {   \
      long a_batch_offset = a->offset;                                         \
      long b_batch_offset = b->offset;                                         \
      long c_batch_offset = c->offset;                                         \
                                                                               \
      if (num_batch_dims > 0) {                                                \
        long temp_idx = batch_idx;                                             \
        for (int i = 0; i < num_batch_dims; i++) {                             \
          int dim_idx = temp_idx / c_batch_decoding_strides[i];                \
          temp_idx %= c_batch_decoding_strides[i];                             \
                                                                               \
          /* Add stride if the dimension exists in the input tensor */         \
          /* This handles broadcasting, e.g., a.shape=[M,K], b.shape=[B,K,N]   \
           */                                                                  \
          if (i < a->ndim - 2 && a->shape[i] > 1) {                            \
            a_batch_offset += dim_idx * a->strides[i];                         \
          }                                                                    \
          if (i < b->ndim - 2 && b->shape[i] > 1) {                            \
            b_batch_offset += dim_idx * b->strides[i];                         \
          }                                                                    \
          c_batch_offset += dim_idx * c->strides[i];                           \
        }                                                                      \
      }                                                                        \
                                                                               \
      const T *a_mat = (const T *)a->data + a_batch_offset;                    \
      const T *b_mat = (const T *)b->data + b_batch_offset;                    \
      T *c_mat = (T *)c->data + c_batch_offset;                                \
                                                                               \
      /* Tiled matrix multiplication for C = A * B */                          \
      for (int i0 = 0; i0 < M; i0 += GEMM_BLOCK_M) {                           \
        for (int j0 = 0; j0 < N; j0 += GEMM_BLOCK_N) {                         \
          /* Zero the C block before accumulating */                           \
          for (int i = i0; i < min(i0 + GEMM_BLOCK_M, M); i++) {               \
            for (int j = j0; j < min(j0 + GEMM_BLOCK_N, N); j++) {             \
              c_mat[i * c_row_stride + j * c_col_stride] = 0;                  \
            }                                                                  \
          }                                                                    \
          /* Accumulate into C block */                                        \
          for (int k0 = 0; k0 < K; k0 += GEMM_BLOCK_K) {                       \
            for (int i = i0; i < min(i0 + GEMM_BLOCK_M, M); i++) {             \
              for (int k = k0; k < min(k0 + GEMM_BLOCK_K, K); k++) {           \
                /* Hoist A[i,k] for better locality */                         \
                const T a_ik = a_mat[i * a_row_stride + k * a_col_stride];     \
                /* Innermost loop over j for SIMD-friendliness */              \
                _Pragma("omp simd") for (int j = j0;                           \
                                         j < min(j0 + GEMM_BLOCK_N, N); j++) { \
                  c_mat[i * c_row_stride + j * c_col_stride] +=                \
                      a_ik * b_mat[k * b_row_stride + j * b_col_stride];       \
                }                                                              \
              }                                                                \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

// Instantiate the generic function for float and double
DEFINE_MATMUL_OP(float, float)
DEFINE_MATMUL_OP(double, double)

// Matmul dispatch
CAMLprim value caml_nx_matmul_bc(value *argv, int argn) {
  value v_a = argv[0], v_ashape = argv[1], v_astrides = argv[2],
        v_aoffset = argv[3];
  value v_b = argv[4], v_bshape = argv[5], v_bstrides = argv[6],
        v_boffset = argv[7];
  value v_c = argv[8], v_cshape = argv[9], v_cstrides = argv[10],
        v_coffset = argv[11];

  struct caml_ba_array *ba_a = Caml_ba_array_val(v_a);
  struct caml_ba_array *ba_b = Caml_ba_array_val(v_b);
  struct caml_ba_array *ba_c = Caml_ba_array_val(v_c);
  int kind = ba_a->flags & CAML_BA_KIND_MASK;

  // Get actual dimensions from shape arrays
  int ndim_a = Wosize_val(v_ashape);
  int ndim_b = Wosize_val(v_bshape);
  int ndim_c = Wosize_val(v_cshape);

  // Limit ndim to prevent stack overflow
  if (ndim_a > 32 || ndim_b > 32 || ndim_c > 32) {
    caml_failwith("matmul: tensor rank too high (max 32)");
  }

  // Stack allocate shape and stride arrays
  int shape_a[ndim_a], strides_a[ndim_a];
  int shape_b[ndim_b], strides_b[ndim_b];
  int shape_c[ndim_c], strides_c[ndim_c];

  for (int i = 0; i < ndim_a; i++) {
    shape_a[i] = Int_val(Field(v_ashape, i));
    strides_a[i] = Int_val(Field(v_astrides, i));
  }
  for (int i = 0; i < ndim_b; i++) {
    shape_b[i] = Int_val(Field(v_bshape, i));
    strides_b[i] = Int_val(Field(v_bstrides, i));
  }
  for (int i = 0; i < ndim_c; i++) {
    shape_c[i] = Int_val(Field(v_cshape, i));
    strides_c[i] = Int_val(Field(v_cstrides, i));
  }

  ndarray_t a, b, c;
  a.data = ba_a->data;
  a.ndim = ndim_a;
  a.shape = shape_a;
  a.strides = strides_a;
  a.offset = Int_val(v_aoffset);

  b.data = ba_b->data;
  b.ndim = ndim_b;
  b.shape = shape_b;
  b.strides = strides_b;
  b.offset = Int_val(v_boffset);

  c.data = ba_c->data;
  c.ndim = ndim_c;
  c.shape = shape_c;
  c.strides = strides_c;
  c.offset = Int_val(v_coffset);

  // Validate dimensions before entering blocking section
  if (ndim_a < 2 || ndim_b < 2 || ndim_c < 2) {
    caml_failwith("matmul: all inputs must have at least 2 dimensions");
  }

  // Check inner dimensions match
  if (shape_a[ndim_a - 1] != shape_b[ndim_b - 2]) {
    caml_failwith("matmul: inner dimensions must match");
  }

  caml_enter_blocking_section();
  switch (kind) {
    case CAML_BA_FLOAT32:
      nx_c_matmul_float(&a, &b, &c);
      break;
    case CAML_BA_FLOAT64:
      nx_c_matmul_double(&a, &b, &c);
      break;
    default:
      caml_leave_blocking_section();
      caml_failwith("matmul: unsupported dtype");
  }
  caml_leave_blocking_section();

  return Val_unit;
}

NATIVE_WRAPPER_12(matmul)

// Helper function to compute strides for decoding a linear index.
static inline void compute_decoding_strides(int ndim, const int *shape,
                                            long *decoding_strides) {
  if (ndim <= 0) return;
  decoding_strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; --i) {
    decoding_strides[i] = decoding_strides[i + 1] * shape[i + 1];
  }
}

// Helper function to decode a linear index into a multi-dimensional index.
static inline void decode_linear_index(long linear_idx, int ndim,
                                       const long *decoding_strides,
                                       int *multi_dim_idx) {
  for (int i = 0; i < ndim; ++i) {
    multi_dim_idx[i] = linear_idx / decoding_strides[i];
    linear_idx %= decoding_strides[i];
  }
}

// Generic, parallel function to zero an ndarray, respecting strides.
static void nx_c_zero_generic(ndarray_t *z, size_t elem_size) {
  long total = total_elements(z);
  if (total == 0) return;

  if (is_c_contiguous(z)) {
    memset((char *)z->data + z->offset * elem_size, 0, total * elem_size);
  } else {
    // Non-contiguous case: Use a dedicated, safe iterator.
    nd_iterator_t it;
    // We only iterate over one array, z. Pass it as x and z args.
    nd_iterator_init(&it, z, NULL, z);

    char *z_data = (char *)z->data;

    // Process all elements sequentially for now.
    // A parallel version would require dividing the work differently.
    do {
      long x_off_unused, z_off;
      nd_iterator_get_offsets(&it, &x_off_unused, NULL, &z_off);
      memset(z_data + z_off * elem_size, 0, elem_size);
    } while (nd_iterator_next(&it));

    nd_iterator_destroy(&it);
  }
}

// Unfold (im2col) implementation
#define DEFINE_UNFOLD_OP(T)                                                    \
  static void nx_c_unfold_##T(const ndarray_t *input, ndarray_t *output,       \
                              const int *output_spatial_shape,                 \
                              const int *kernel_shape, const int *strides,     \
                              const int *padding_lower, const int *dilation,   \
                              int num_spatial_dims) {                          \
    T *input_data = (T *)input->data;                                          \
    T *output_data = (T *)output->data;                                        \
                                                                               \
    const int num_batch_dims = input->ndim - num_spatial_dims - 1;             \
    if (num_batch_dims < 0) return;                                            \
                                                                               \
    const int channels = input->shape[num_batch_dims];                         \
    long patch_size_per_channel = 1;                                           \
    for (int i = 0; i < num_spatial_dims; ++i) {                               \
      patch_size_per_channel *= kernel_shape[i];                               \
    }                                                                          \
                                                                               \
    long batch_size = 1;                                                       \
    for (int i = 0; i < num_batch_dims; ++i) {                                 \
      batch_size *= input->shape[i];                                           \
    }                                                                          \
                                                                               \
    long num_patches = 1;                                                      \
    for (int i = 0; i < num_spatial_dims; ++i) {                               \
      num_patches *= output_spatial_shape[i];                                  \
    }                                                                          \
                                                                               \
    long total_work_items = batch_size * num_patches;                          \
    if (total_work_items == 0) return;                                         \
                                                                               \
    /* Precompute strides for decoding linear indices */                       \
    long batch_decoding_strides[num_batch_dims > 0 ? num_batch_dims : 1];      \
    long patch_decoding_strides[num_spatial_dims > 0 ? num_spatial_dims : 1];  \
    long kernel_decoding_strides[num_spatial_dims > 0 ? num_spatial_dims : 1]; \
                                                                               \
    compute_decoding_strides(num_batch_dims, input->shape,                     \
                             batch_decoding_strides);                          \
    compute_decoding_strides(num_spatial_dims, output_spatial_shape,           \
                             patch_decoding_strides);                          \
    compute_decoding_strides(num_spatial_dims, kernel_shape,                   \
                             kernel_decoding_strides);                         \
                                                                               \
    _Pragma("omp parallel for") for (long work_idx = 0;                        \
                                     work_idx < total_work_items;              \
                                     ++work_idx) {                             \
      long batch_linear_idx = work_idx / num_patches;                          \
      long patch_linear_idx = work_idx % num_patches;                          \
                                                                               \
      int batch_indices[num_batch_dims > 0 ? num_batch_dims : 1];              \
      int patch_indices[num_spatial_dims > 0 ? num_spatial_dims : 1];          \
      decode_linear_index(batch_linear_idx, num_batch_dims,                    \
                          batch_decoding_strides, batch_indices);              \
      decode_linear_index(patch_linear_idx, num_spatial_dims,                  \
                          patch_decoding_strides, patch_indices);              \
                                                                               \
      long input_base_offset = input->offset;                                  \
      long output_base_offset = output->offset;                                \
      for (int i = 0; i < num_batch_dims; ++i) {                               \
        input_base_offset += batch_indices[i] * input->strides[i];             \
        output_base_offset += batch_indices[i] * output->strides[i];           \
      }                                                                        \
                                                                               \
      /* Loop over all elements in a single patch/column */                    \
      for (int c = 0; c < channels; ++c) {                                     \
        for (long k_linear_idx = 0; k_linear_idx < patch_size_per_channel;     \
             ++k_linear_idx) {                                                 \
          int kernel_indices[num_spatial_dims > 0 ? num_spatial_dims : 1];     \
          decode_linear_index(k_linear_idx, num_spatial_dims,                  \
                              kernel_decoding_strides, kernel_indices);        \
                                                                               \
          long current_input_offset =                                          \
              input_base_offset + c * input->strides[num_batch_dims];          \
          int in_bounds = 1;                                                   \
          for (int i = 0; i < num_spatial_dims; ++i) {                         \
            long pos = (long)patch_indices[i] * strides[i] +                   \
                       (long)kernel_indices[i] * dilation[i] -                 \
                       padding_lower[i];                                       \
            if (pos < 0 || pos >= input->shape[num_batch_dims + 1 + i]) {      \
              in_bounds = 0;                                                   \
              break;                                                           \
            }                                                                  \
            current_input_offset +=                                            \
                pos * input->strides[num_batch_dims + 1 + i];                  \
          }                                                                    \
                                                                               \
          long output_row_major_offset =                                       \
              output_base_offset +                                             \
              (c * patch_size_per_channel + k_linear_idx) *                    \
                  output->strides[output->ndim - 2] +                          \
              patch_linear_idx * output->strides[output->ndim - 1];            \
                                                                               \
          if (in_bounds) {                                                     \
            output_data[output_row_major_offset] =                             \
                input_data[current_input_offset];                              \
          } else {                                                             \
            output_data[output_row_major_offset] = (T)0;                       \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

// Fold (col2im) implementation
#define DEFINE_FOLD_OP(T)                                                      \
  static void nx_c_fold_##T(const ndarray_t *input_cols, ndarray_t *output,    \
                            const int *output_spatial_shape,                   \
                            const int *kernel_shape, const int *strides,       \
                            const int *padding_lower, const int *dilation,     \
                            int num_spatial_dims) {                            \
    T *input_data = (T *)input_cols->data;                                     \
    T *output_data = (T *)output->data;                                        \
    size_t elem_size = sizeof(T);                                              \
                                                                               \
    nx_c_zero_generic(output, elem_size);                                      \
                                                                               \
    const int num_batch_dims = output->ndim - num_spatial_dims - 1;            \
    if (num_batch_dims < 0) return;                                            \
                                                                               \
    const int channels = output->shape[num_batch_dims];                        \
    long patch_size_per_channel = 1;                                           \
    for (int i = 0; i < num_spatial_dims; ++i) {                               \
      patch_size_per_channel *= kernel_shape[i];                               \
    }                                                                          \
                                                                               \
    long batch_size = 1;                                                       \
    for (int i = 0; i < num_batch_dims; ++i) {                                 \
      batch_size *= output->shape[i];                                          \
    }                                                                          \
                                                                               \
    long num_patches = input_cols->shape[input_cols->ndim - 1];                \
    long total_work_items = batch_size * num_patches;                          \
    if (total_work_items == 0) return;                                         \
                                                                               \
    /* Precompute strides for decoding linear indices */                       \
    long batch_decoding_strides[num_batch_dims > 0 ? num_batch_dims : 1];      \
    long patch_decoding_strides[num_spatial_dims > 0 ? num_spatial_dims : 1];  \
    long kernel_decoding_strides[num_spatial_dims > 0 ? num_spatial_dims : 1]; \
                                                                               \
    compute_decoding_strides(num_batch_dims, output->shape,                    \
                             batch_decoding_strides);                          \
    compute_decoding_strides(num_spatial_dims, output_spatial_shape,           \
                             patch_decoding_strides);                          \
    compute_decoding_strides(num_spatial_dims, kernel_shape,                   \
                             kernel_decoding_strides);                         \
                                                                               \
    /* Optimization: if strides are >= kernel shape, no overlap occurs. */     \
    int no_overlap = 1;                                                        \
    for (int i = 0; i < num_spatial_dims; ++i) {                               \
      if (strides[i] < kernel_shape[i]) {                                      \
        no_overlap = 0;                                                        \
        break;                                                                 \
      }                                                                        \
    }                                                                          \
                                                                               \
    _Pragma("omp parallel for") for (long work_idx = 0;                        \
                                     work_idx < total_work_items;              \
                                     ++work_idx) {                             \
      long batch_linear_idx = work_idx / num_patches;                          \
      long patch_linear_idx = work_idx % num_patches;                          \
                                                                               \
      int batch_indices[num_batch_dims > 0 ? num_batch_dims : 1];              \
      int patch_indices[num_spatial_dims > 0 ? num_spatial_dims : 1];          \
      decode_linear_index(batch_linear_idx, num_batch_dims,                    \
                          batch_decoding_strides, batch_indices);              \
      decode_linear_index(patch_linear_idx, num_spatial_dims,                  \
                          patch_decoding_strides, patch_indices);              \
                                                                               \
      long input_base_offset = input_cols->offset;                             \
      long output_base_offset = output->offset;                                \
      for (int i = 0; i < num_batch_dims; ++i) {                               \
        input_base_offset += batch_indices[i] * input_cols->strides[i];        \
        output_base_offset += batch_indices[i] * output->strides[i];           \
      }                                                                        \
                                                                               \
      for (int c = 0; c < channels; ++c) {                                     \
        for (long k_linear_idx = 0; k_linear_idx < patch_size_per_channel;     \
             ++k_linear_idx) {                                                 \
          int kernel_indices[num_spatial_dims > 0 ? num_spatial_dims : 1];     \
          decode_linear_index(k_linear_idx, num_spatial_dims,                  \
                              kernel_decoding_strides, kernel_indices);        \
                                                                               \
          long current_output_offset =                                         \
              output_base_offset + c * output->strides[num_batch_dims];        \
          int in_bounds = 1;                                                   \
          for (int i = 0; i < num_spatial_dims; ++i) {                         \
            long pos = (long)patch_indices[i] * strides[i] +                   \
                       (long)kernel_indices[i] * dilation[i] -                 \
                       padding_lower[i];                                       \
            if (pos < 0 || pos >= output->shape[num_batch_dims + 1 + i]) {     \
              in_bounds = 0;                                                   \
              break;                                                           \
            }                                                                  \
            current_output_offset +=                                           \
                pos * output->strides[num_batch_dims + 1 + i];                 \
          }                                                                    \
                                                                               \
          if (in_bounds) {                                                     \
            long input_row_major_offset =                                      \
                input_base_offset +                                            \
                patch_linear_idx * input_cols->strides[input_cols->ndim - 1] + \
                (c * patch_size_per_channel + k_linear_idx) *                  \
                    input_cols->strides[input_cols->ndim - 2];                 \
            T val = input_data[input_row_major_offset];                        \
            if (no_overlap) {                                                  \
              output_data[current_output_offset] = val;                        \
            } else {                                                           \
              _Pragma("omp atomic update")                                     \
                  output_data[current_output_offset] += val;                   \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

DEFINE_UNFOLD_OP(int8_t)
DEFINE_UNFOLD_OP(uint8_t)
DEFINE_UNFOLD_OP(int16_t)
DEFINE_UNFOLD_OP(uint16_t)
DEFINE_UNFOLD_OP(int32_t)
DEFINE_UNFOLD_OP(int64_t)
DEFINE_UNFOLD_OP(intnat)
DEFINE_UNFOLD_OP(float16_t)
DEFINE_UNFOLD_OP(float)
DEFINE_UNFOLD_OP(double)
DEFINE_UNFOLD_OP(c32_t)
DEFINE_UNFOLD_OP(c64_t)
DEFINE_UNFOLD_OP(qint8_t)
DEFINE_UNFOLD_OP(quint8_t)

DEFINE_FOLD_OP(int8_t)
DEFINE_FOLD_OP(uint8_t)
DEFINE_FOLD_OP(int16_t)
DEFINE_FOLD_OP(uint16_t)
DEFINE_FOLD_OP(int32_t)
DEFINE_FOLD_OP(int64_t)
DEFINE_FOLD_OP(intnat)
DEFINE_FOLD_OP(float16_t)
DEFINE_FOLD_OP(float)
DEFINE_FOLD_OP(double)
DEFINE_FOLD_OP(c32_t)
DEFINE_FOLD_OP(c64_t)
DEFINE_FOLD_OP(qint8_t)
DEFINE_FOLD_OP(quint8_t)

// Unfold dispatch
CAMLprim value caml_nx_unfold_bc(value *argv, int argn) {
  value v_ndim = argv[0], v_shape = argv[1];
  value v_input = argv[2], v_input_strides = argv[3], v_input_offset = argv[4];
  value v_output_ndim = argv[5], v_output_shape = argv[6];
  value v_output = argv[7], v_output_strides = argv[8],
        v_output_offset = argv[9];
  value v_output_spatial_shape = argv[10], v_kernel_shape = argv[11],
        v_strides = argv[12], v_padding_lower = argv[13], v_dilation = argv[14];

  struct caml_ba_array *ba_input = Caml_ba_array_val(v_input);
  struct caml_ba_array *ba_output = Caml_ba_array_val(v_output);
  int kind = ba_input->flags & CAML_BA_KIND_MASK;
  int ndim = Int_val(v_ndim);

  if (ndim < 1) {
    caml_failwith("unfold: input must have at least 1 dimension");
  }

  // Get num_spatial_dims from the kernel_shape array length
  int num_spatial_dims = Wosize_val(v_kernel_shape);

  // Validate input dimensions
  if (ndim < num_spatial_dims + 1) {
    caml_failwith("unfold: input dimensions incompatible with kernel size");
  }

  // Additional validation: need at least channels + spatial dims
  if (ndim <= num_spatial_dims) {
    caml_failwith(
        "unfold: input must have at least 1 channel dimension before spatial "
        "dims");
  }

  // Stack allocate arrays with minimum size of 1 to avoid zero-length VLAs
  int input_shape[ndim], input_strides[ndim];
  int output_ndim = Int_val(v_output_ndim);
  int output_shape[output_ndim], output_strides[output_ndim];
  int output_spatial_shape[num_spatial_dims > 0 ? num_spatial_dims : 1];
  int kernel_shape[num_spatial_dims > 0 ? num_spatial_dims : 1],
      strides[num_spatial_dims > 0 ? num_spatial_dims : 1],
      padding_lower[num_spatial_dims > 0 ? num_spatial_dims : 1],
      dilation[num_spatial_dims > 0 ? num_spatial_dims : 1];

  for (int i = 0; i < ndim; i++) {
    input_shape[i] = Int_val(Field(v_shape, i));
    input_strides[i] = Int_val(Field(v_input_strides, i));
  }
  for (int i = 0; i < output_ndim; i++) {
    output_shape[i] = Int_val(Field(v_output_shape, i));
    output_strides[i] = Int_val(Field(v_output_strides, i));
  }
  for (int i = 0; i < num_spatial_dims; i++) {
    output_spatial_shape[i] = Int_val(Field(v_output_spatial_shape, i));
    kernel_shape[i] = Int_val(Field(v_kernel_shape, i));
    strides[i] = Int_val(Field(v_strides, i));
    padding_lower[i] = Int_val(Field(v_padding_lower, i));
    dilation[i] = Int_val(Field(v_dilation, i));
  }

  ndarray_t input, output;
  input.data = ba_input->data;
  input.ndim = ndim;
  input.shape = input_shape;
  input.strides = input_strides;
  input.offset = Int_val(v_input_offset);

  output.data = ba_output->data;
  output.ndim = output_ndim;
  output.shape = output_shape;
  output.strides = output_strides;
  output.offset = Int_val(v_output_offset);

  caml_enter_blocking_section();
  switch (kind) {
    case CAML_BA_SINT8:
      nx_c_unfold_int8_t(&input, &output, output_spatial_shape, kernel_shape,
                         strides, padding_lower, dilation, num_spatial_dims);
      break;
    case CAML_BA_UINT8:
      nx_c_unfold_uint8_t(&input, &output, output_spatial_shape, kernel_shape,
                          strides, padding_lower, dilation, num_spatial_dims);
      break;
    case CAML_BA_SINT16:
      nx_c_unfold_int16_t(&input, &output, output_spatial_shape, kernel_shape,
                          strides, padding_lower, dilation, num_spatial_dims);
      break;
    case CAML_BA_UINT16:
      nx_c_unfold_uint16_t(&input, &output, output_spatial_shape, kernel_shape,
                           strides, padding_lower, dilation, num_spatial_dims);
      break;
    case CAML_BA_INT32:
      nx_c_unfold_int32_t(&input, &output, output_spatial_shape, kernel_shape,
                          strides, padding_lower, dilation, num_spatial_dims);
      break;
    case CAML_BA_INT64:
      nx_c_unfold_int64_t(&input, &output, output_spatial_shape, kernel_shape,
                          strides, padding_lower, dilation, num_spatial_dims);
      break;
    case CAML_BA_CAML_INT:
    case CAML_BA_NATIVE_INT:
      nx_c_unfold_intnat(&input, &output, output_spatial_shape, kernel_shape,
                         strides, padding_lower, dilation, num_spatial_dims);
      break;
    case CAML_BA_FLOAT16:
      nx_c_unfold_float16_t(&input, &output, output_spatial_shape, kernel_shape,
                            strides, padding_lower, dilation, num_spatial_dims);
      break;
    case CAML_BA_FLOAT32:
      nx_c_unfold_float(&input, &output, output_spatial_shape, kernel_shape,
                        strides, padding_lower, dilation, num_spatial_dims);
      break;
    case CAML_BA_FLOAT64:
      nx_c_unfold_double(&input, &output, output_spatial_shape, kernel_shape,
                         strides, padding_lower, dilation, num_spatial_dims);
      break;
    case CAML_BA_COMPLEX32:
      nx_c_unfold_c32_t(&input, &output, output_spatial_shape, kernel_shape,
                        strides, padding_lower, dilation, num_spatial_dims);
      break;
    case CAML_BA_COMPLEX64:
      nx_c_unfold_c64_t(&input, &output, output_spatial_shape, kernel_shape,
                        strides, padding_lower, dilation, num_spatial_dims);
      break;
    case NX_BA_QINT8:
      nx_c_unfold_qint8_t(&input, &output, output_spatial_shape, kernel_shape,
                          strides, padding_lower, dilation, num_spatial_dims);
      break;
    case NX_BA_QUINT8:
      nx_c_unfold_quint8_t(&input, &output, output_spatial_shape, kernel_shape,
                           strides, padding_lower, dilation, num_spatial_dims);
      break;
    default:
      caml_leave_blocking_section();
      caml_failwith("unfold: unsupported dtype");
  }
  caml_leave_blocking_section();

  return Val_unit;
}

// Need custom wrapper for unfold (13 args)
#define NATIVE_WRAPPER_13(name)                                               \
  CAMLprim value caml_nx_##name(                                              \
      value arg1, value arg2, value arg3, value arg4, value arg5, value arg6, \
      value arg7, value arg8, value arg9, value arg10, value arg11,           \
      value arg12, value arg13) {                                             \
    value argv[13] = {arg1, arg2, arg3,  arg4,  arg5,  arg6, arg7,            \
                      arg8, arg9, arg10, arg11, arg12, arg13};                \
    return caml_nx_##name##_bc(argv, 13);                                     \
  }

NATIVE_WRAPPER_15(unfold)

// Fold dispatch
CAMLprim value caml_nx_fold_bc(value *argv, int argn) {
  value v_ndim = argv[0], v_shape = argv[1];
  value v_input_cols = argv[2], v_input_cols_strides = argv[3],
        v_input_cols_offset = argv[4];
  value v_output_ndim = argv[5], v_output_shape = argv[6];
  value v_output = argv[7], v_output_strides = argv[8],
        v_output_offset = argv[9];
  value v_output_spatial_shape = argv[10], v_kernel_shape = argv[11],
        v_strides = argv[12], v_padding_lower = argv[13], v_dilation = argv[14];

  struct caml_ba_array *ba_input = Caml_ba_array_val(v_input_cols);
  struct caml_ba_array *ba_output = Caml_ba_array_val(v_output);
  int kind = ba_input->flags & CAML_BA_KIND_MASK;
  int input_ndim = Int_val(v_ndim);
  int output_ndim = Int_val(v_output_ndim);

  if (input_ndim < 1) {
    caml_failwith("fold: input must have at least 1 dimension");
  }
  if (output_ndim < 1) {
    caml_failwith("fold: output must have at least 1 dimension");
  }

  // Get num_spatial_dims from the kernel_shape array length
  int num_spatial_dims = Wosize_val(v_kernel_shape);

  // Calculate expected dimensions accounting for batch
  // Input should have shape [...batch, channels*kernel_elements, num_blocks]
  // Output should have shape [...batch, channels, ...spatial]
  int batch_dims = input_ndim - 2;  // Everything except last 2 dims

  // Validate dimensions
  if (output_ndim != batch_dims + num_spatial_dims + 1) {
    caml_failwith(
        "fold: output dimensions don't match expected batch + channels + "
        "spatial dimensions");
  }
  if (input_ndim < 2) {
    caml_failwith("fold: input must have at least 2 dimensions");
  }

  // Stack allocate arrays with minimum size of 1 to avoid zero-length VLAs
  int input_shape[input_ndim], input_strides[input_ndim];
  int output_shape[output_ndim], output_strides[output_ndim];
  int output_spatial_shape[num_spatial_dims > 0 ? num_spatial_dims : 1];
  int kernel_shape[num_spatial_dims > 0 ? num_spatial_dims : 1],
      strides[num_spatial_dims > 0 ? num_spatial_dims : 1],
      padding_lower[num_spatial_dims > 0 ? num_spatial_dims : 1],
      dilation[num_spatial_dims > 0 ? num_spatial_dims : 1];

  for (int i = 0; i < input_ndim; i++) {
    input_shape[i] = Int_val(Field(v_shape, i));
    input_strides[i] = Int_val(Field(v_input_cols_strides, i));
  }
  for (int i = 0; i < output_ndim; i++) {
    output_shape[i] = Int_val(Field(v_output_shape, i));
    output_strides[i] = Int_val(Field(v_output_strides, i));
  }
  for (int i = 0; i < num_spatial_dims; i++) {
    output_spatial_shape[i] = Int_val(Field(v_output_spatial_shape, i));
    kernel_shape[i] = Int_val(Field(v_kernel_shape, i));
    strides[i] = Int_val(Field(v_strides, i));
    padding_lower[i] = Int_val(Field(v_padding_lower, i));
    dilation[i] = Int_val(Field(v_dilation, i));
  }

  ndarray_t input_cols, output;
  input_cols.data = ba_input->data;
  input_cols.ndim = input_ndim;
  input_cols.shape = input_shape;
  input_cols.strides = input_strides;
  input_cols.offset = Int_val(v_input_cols_offset);

  output.data = ba_output->data;
  output.ndim = output_ndim;
  output.shape = output_shape;
  output.strides = output_strides;
  output.offset = Int_val(v_output_offset);

  caml_enter_blocking_section();
  switch (kind) {
    case CAML_BA_SINT8:
      nx_c_fold_int8_t(&input_cols, &output, output_spatial_shape, kernel_shape,
                       strides, padding_lower, dilation, num_spatial_dims);
      break;
    case CAML_BA_UINT8:
      nx_c_fold_uint8_t(&input_cols, &output, output_spatial_shape,
                        kernel_shape, strides, padding_lower, dilation,
                        num_spatial_dims);
      break;
    case CAML_BA_SINT16:
      nx_c_fold_int16_t(&input_cols, &output, output_spatial_shape, kernel_shape,
                        strides, padding_lower, dilation, num_spatial_dims);
      break;
    case CAML_BA_UINT16:
      nx_c_fold_uint16_t(&input_cols, &output, output_spatial_shape,
                         kernel_shape, strides, padding_lower, dilation,
                         num_spatial_dims);
      break;
    case CAML_BA_INT32:
      nx_c_fold_int32_t(&input_cols, &output, output_spatial_shape,
                        kernel_shape, strides, padding_lower, dilation,
                        num_spatial_dims);
      break;
    case CAML_BA_INT64:
      nx_c_fold_int64_t(&input_cols, &output, output_spatial_shape,
                        kernel_shape, strides, padding_lower, dilation,
                        num_spatial_dims);
      break;
    case CAML_BA_CAML_INT:
    case CAML_BA_NATIVE_INT:
      nx_c_fold_intnat(&input_cols, &output, output_spatial_shape, kernel_shape,
                       strides, padding_lower, dilation, num_spatial_dims);
      break;
    case CAML_BA_FLOAT16:
      nx_c_fold_float16_t(&input_cols, &output, output_spatial_shape,
                          kernel_shape, strides, padding_lower, dilation,
                          num_spatial_dims);
      break;
    case CAML_BA_FLOAT32:
      nx_c_fold_float(&input_cols, &output, output_spatial_shape, kernel_shape,
                      strides, padding_lower, dilation, num_spatial_dims);
      break;
    case CAML_BA_FLOAT64:
      nx_c_fold_double(&input_cols, &output, output_spatial_shape, kernel_shape,
                       strides, padding_lower, dilation, num_spatial_dims);
      break;
    case CAML_BA_COMPLEX32:
      nx_c_fold_c32_t(&input_cols, &output, output_spatial_shape, kernel_shape,
                      strides, padding_lower, dilation, num_spatial_dims);
      break;
    case CAML_BA_COMPLEX64:
      nx_c_fold_c64_t(&input_cols, &output, output_spatial_shape, kernel_shape,
                      strides, padding_lower, dilation, num_spatial_dims);
      break;
    case NX_BA_QINT8:
      nx_c_fold_qint8_t(&input_cols, &output, output_spatial_shape, kernel_shape,
                        strides, padding_lower, dilation, num_spatial_dims);
      break;
    case NX_BA_QUINT8:
      nx_c_fold_quint8_t(&input_cols, &output, output_spatial_shape,
                         kernel_shape, strides, padding_lower, dilation,
                         num_spatial_dims);
      break;
    default:
      caml_leave_blocking_section();
      caml_failwith("fold: unsupported dtype");
  }
  caml_leave_blocking_section();

  return Val_unit;
}

NATIVE_WRAPPER_15(fold)
