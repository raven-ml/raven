// Reduction operations for nx C backend

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/threads.h>
#include <complex.h>
#include <math.h>
#include <stdlib.h>

#include "nx_c_shared.h"

// Type definitions for reduction operations
typedef void (*reduce_op_t)(const ndarray_t *, ndarray_t *, const int *, int,
                            bool);

// Dispatch table for each type
typedef struct {
  reduce_op_t i8, u8, i16, u16, i32, i64, inat;
  reduce_op_t f16, f32, f64;
  reduce_op_t c32, c64;
  reduce_op_t bf16, bool_, i4, u4, f8e4m3, f8e5m2, c16, qi8, qu8;
} reduce_op_table;

// Forward declarations for optimized fallback symbols generated later
// (needed so we can call them from fast paths before their definitions)
// Forward decls for wrapper functions that call the generic (macro-generated)
// fallback implementations. Definitions are placed after the generic impls.
static void nx_c_reduce_sum_f32_generic_wrap(const ndarray_t *, ndarray_t *,
                                             const int *, int, bool);
static void nx_c_reduce_sum_f64_generic_wrap(const ndarray_t *, ndarray_t *,
                                             const int *, int, bool);
static void nx_c_reduce_max_f32_generic_wrap(const ndarray_t *, ndarray_t *,
                                             const int *, int, bool);
static void nx_c_reduce_max_f64_generic_wrap(const ndarray_t *, ndarray_t *,
                                             const int *, int, bool);

// Helper utilities for optimized paths
static inline bool normalize_and_sort_axes(int ndim, const int *axes,
                                           int num_axes, int *out_axes) {
  if (!axes || !out_axes || num_axes <= 0) return false;
  for (int i = 0; i < num_axes; ++i) {
    int axis = axes[i];
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) return false;
    out_axes[i] = axis;
  }
  for (int i = 0; i < num_axes - 1; ++i) {
    for (int j = i + 1; j < num_axes; ++j) {
      if (out_axes[j] < out_axes[i]) {
        int tmp = out_axes[i];
        out_axes[i] = out_axes[j];
        out_axes[j] = tmp;
      }
    }
  }
  for (int i = 1; i < num_axes; ++i) {
    if (out_axes[i] == out_axes[i - 1]) return false;
  }
  return true;
}

static inline bool axes_are_trailing(int ndim, int num_axes,
                                     const int *sorted_axes) {
  for (int i = 0; i < num_axes; ++i) {
    if (sorted_axes[i] != ndim - num_axes + i) return false;
  }
  return true;
}

static inline long product_of_axes(const ndarray_t *input,
                                   const int *sorted_axes, int num_axes) {
  long prod = 1;
  for (int i = 0; i < num_axes; ++i) {
    long dim = input->shape[sorted_axes[i]];
    if (dim == 0) return 0;
    prod *= dim;
  }
  return prod;
}

// Helper functions
static int cmp_int(const void *a, const void *b) {
  return *(const int *)a - *(const int *)b;
}

static long get_offset(const ndarray_t *nd, const int *coord) {
  long off = 0;
  for (int i = 0; i < nd->ndim; ++i) {
    off += (long)coord[i] * nd->strides[i];
  }
  return off;
}

static void get_coord_from_idx(long idx, const ndarray_t *nd, int *coord) {
  for (int i = nd->ndim - 1; i >= 0; --i) {
    coord[i] = idx % nd->shape[i];
    idx /= nd->shape[i];
  }
}

// Macro to generate reduction implementation for standard types
#define REDUCE_OP_IMPL(name, T, suffix, IDENTITY, HAS_IDENTITY, OP)        \
  static void nx_c_##name##_##suffix(const ndarray_t *input,               \
                                     ndarray_t *output, const int *axes,   \
                                     int num_axes, bool keepdims) {        \
    if (!input || !output) {                                               \
      caml_failwith("nx_c_" #name "_" #suffix ": null pointer");           \
    }                                                                      \
    bool *is_reduced = (bool *)calloc(input->ndim, sizeof(bool));          \
    if (!is_reduced) caml_failwith("allocation failed");                   \
    for (int i = 0; i < num_axes; ++i) {                                   \
      if (axes[i] < 0 || axes[i] >= input->ndim) {                         \
        free(is_reduced);                                                  \
        caml_failwith("invalid axis");                                     \
      }                                                                    \
      is_reduced[axes[i]] = true;                                          \
    }                                                                      \
    int num_kept = input->ndim - num_axes;                                 \
    int *kept_axes = (int *)malloc(num_kept * sizeof(int));                \
    if (!kept_axes) {                                                      \
      free(is_reduced);                                                    \
      caml_failwith("allocation failed");                                  \
    }                                                                      \
    int kk = 0;                                                            \
    for (int i = 0; i < input->ndim; ++i) {                                \
      if (!is_reduced[i]) kept_axes[kk++] = i;                             \
    }                                                                      \
    long reduce_prod = 1;                                                  \
    bool zero_size = false;                                                \
    for (int i = 0; i < num_axes; ++i) {                                   \
      long ss = input->shape[axes[i]];                                     \
      if (ss == 0) zero_size = true;                                       \
      reduce_prod *= ss;                                                   \
    }                                                                      \
    if (zero_size) reduce_prod = 0;                                        \
    if (reduce_prod == 0 && !HAS_IDENTITY) {                               \
      free(is_reduced);                                                    \
      free(kept_axes);                                                     \
      caml_failwith("zero-size array to reduction operation " #name        \
                    " which has no identity");                             \
    }                                                                      \
    long total_out = total_elements_safe(output);                          \
    if (total_out == 0) {                                                  \
      free(is_reduced);                                                    \
      free(kept_axes);                                                     \
      return;                                                              \
    }                                                                      \
    _Pragma("omp parallel for if(total_out > 1000)") for (long idx = 0;    \
                                                          idx < total_out; \
                                                          ++idx) {         \
      int local_out_coord[MAX_NDIM];                                       \
      int local_in_coord[MAX_NDIM];                                        \
      int local_reduced_coord[MAX_NDIM];                                   \
      get_coord_from_idx(idx, output, local_out_coord);                    \
      memset(local_in_coord, 0, input->ndim * sizeof(int));                \
      if (keepdims) {                                                      \
        for (int d = 0; d < input->ndim; ++d) {                            \
          if (!is_reduced[d]) local_in_coord[d] = local_out_coord[d];      \
        }                                                                  \
      } else {                                                             \
        for (int ii = 0; ii < num_kept; ++ii) {                            \
          local_in_coord[kept_axes[ii]] = local_out_coord[ii];             \
        }                                                                  \
      }                                                                    \
      T acc;                                                               \
      if (reduce_prod == 0) {                                              \
        acc = IDENTITY;                                                    \
      } else {                                                             \
        bool first = true;                                                 \
        memset(local_reduced_coord, 0, num_axes * sizeof(int));            \
        bool inner_done = false;                                           \
        while (!inner_done) {                                              \
          for (int j = 0; j < num_axes; ++j) {                             \
            local_in_coord[axes[j]] = local_reduced_coord[j];              \
          }                                                                \
          long in_off = input->offset + get_offset(input, local_in_coord); \
          T val = ((T *)input->data)[in_off];                              \
          if (first) {                                                     \
            acc = val;                                                     \
            first = false;                                                 \
          } else {                                                         \
            acc = OP(acc, val);                                            \
          }                                                                \
          inner_done = true;                                               \
          for (int j = num_axes - 1; j >= 0; --j) {                        \
            local_reduced_coord[j]++;                                      \
            if (local_reduced_coord[j] < input->shape[axes[j]]) {          \
              inner_done = false;                                          \
              break;                                                       \
            }                                                              \
            local_reduced_coord[j] = 0;                                    \
          }                                                                \
        }                                                                  \
      }                                                                    \
      long out_off = output->offset + get_offset(output, local_out_coord); \
      ((T *)output->data)[out_off] = acc;                                  \
    }                                                                      \
    free(is_reduced);                                                      \
    free(kept_axes);                                                       \
  }

// Macro to generate both for a type
#define REDUCE_OP_FOR_TYPE(name, T, suffix, IDENTITY, HAS_IDENTITY, OP) \
  REDUCE_OP_IMPL(name, T, suffix, IDENTITY, HAS_IDENTITY, OP)

// Low-precision reduce impl
#define LOW_PREC_REDUCE_OP_IMPL(name, T, suffix, IDENTITY_FLOAT, HAS_IDENTITY, \
                                OP_FLOAT, TO_FLOAT, FROM_FLOAT)                \
  static void nx_c_##name##_##suffix(const ndarray_t *input,                   \
                                     ndarray_t *output, const int *axes,       \
                                     int num_axes, bool keepdims) {            \
    if (!input || !output) {                                                   \
      caml_failwith("nx_c_" #name "_" #suffix ": null pointer");               \
    }                                                                          \
    bool *is_reduced = (bool *)calloc(input->ndim, sizeof(bool));              \
    if (!is_reduced) caml_failwith("allocation failed");                       \
    for (int i = 0; i < num_axes; ++i) {                                       \
      if (axes[i] < 0 || axes[i] >= input->ndim) {                             \
        free(is_reduced);                                                      \
        caml_failwith("invalid axis");                                         \
      }                                                                        \
      is_reduced[axes[i]] = true;                                              \
    }                                                                          \
    int num_kept = input->ndim - num_axes;                                     \
    int *kept_axes = (int *)malloc(num_kept * sizeof(int));                    \
    if (!kept_axes) {                                                          \
      free(is_reduced);                                                        \
      caml_failwith("allocation failed");                                      \
    }                                                                          \
    int kk = 0;                                                                \
    for (int i = 0; i < input->ndim; ++i) {                                    \
      if (!is_reduced[i]) kept_axes[kk++] = i;                                 \
    }                                                                          \
    long reduce_prod = 1;                                                      \
    bool zero_size = false;                                                    \
    for (int i = 0; i < num_axes; ++i) {                                       \
      long ss = input->shape[axes[i]];                                         \
      if (ss == 0) zero_size = true;                                           \
      reduce_prod *= ss;                                                       \
    }                                                                          \
    if (zero_size) reduce_prod = 0;                                            \
    if (reduce_prod == 0 && !HAS_IDENTITY) {                                   \
      free(is_reduced);                                                        \
      free(kept_axes);                                                         \
      caml_failwith("zero-size array to reduction operation " #name            \
                    " which has no identity");                                 \
    }                                                                          \
    long total_out = total_elements_safe(output);                              \
    if (total_out == 0) {                                                      \
      free(is_reduced);                                                        \
      free(kept_axes);                                                         \
      return;                                                                  \
    }                                                                          \
    _Pragma("omp parallel for if(total_out > 1000)") for (long idx = 0;        \
                                                          idx < total_out;     \
                                                          ++idx) {             \
      int local_out_coord[MAX_NDIM];                                           \
      int local_in_coord[MAX_NDIM];                                            \
      int local_reduced_coord[MAX_NDIM];                                       \
      get_coord_from_idx(idx, output, local_out_coord);                        \
      memset(local_in_coord, 0, input->ndim * sizeof(int));                    \
      if (keepdims) {                                                          \
        for (int d = 0; d < input->ndim; ++d) {                                \
          if (!is_reduced[d]) local_in_coord[d] = local_out_coord[d];          \
        }                                                                      \
      } else {                                                                 \
        for (int ii = 0; ii < num_kept; ++ii) {                                \
          local_in_coord[kept_axes[ii]] = local_out_coord[ii];                 \
        }                                                                      \
      }                                                                        \
      float acc;                                                               \
      if (reduce_prod == 0) {                                                  \
        acc = IDENTITY_FLOAT;                                                  \
      } else {                                                                 \
        bool first = true;                                                     \
        memset(local_reduced_coord, 0, num_axes * sizeof(int));                \
        bool inner_done = false;                                               \
        while (!inner_done) {                                                  \
          for (int j = 0; j < num_axes; ++j) {                                 \
            local_in_coord[axes[j]] = local_reduced_coord[j];                  \
          }                                                                    \
          long in_off = input->offset + get_offset(input, local_in_coord);     \
          float val = TO_FLOAT(((T *)input->data)[in_off]);                    \
          if (first) {                                                         \
            acc = val;                                                         \
            first = false;                                                     \
          } else {                                                             \
            acc = OP_FLOAT(acc, val);                                          \
          }                                                                    \
          inner_done = true;                                                   \
          for (int j = num_axes - 1; j >= 0; --j) {                            \
            local_reduced_coord[j]++;                                          \
            if (local_reduced_coord[j] < input->shape[axes[j]]) {              \
              inner_done = false;                                              \
              break;                                                           \
            }                                                                  \
            local_reduced_coord[j] = 0;                                        \
          }                                                                    \
        }                                                                      \
      }                                                                        \
      long out_off = output->offset + get_offset(output, local_out_coord);     \
      ((T *)output->data)[out_off] = FROM_FLOAT(acc);                          \
    }                                                                          \
    free(is_reduced);                                                          \
    free(kept_axes);                                                           \
  }

// Complex16 reduce impl
#define COMPLEX16_REDUCE_IMPL(name, IDENTITY, HAS_IDENTITY, OP)                \
  static void nx_c_##name##_c16(const ndarray_t *input, ndarray_t *output,     \
                                const int *axes, int num_axes,                 \
                                bool keepdims) {                               \
    if (!input || !output) {                                                   \
      caml_failwith("nx_c_" #name "_c16: null pointer");                       \
    }                                                                          \
    bool *is_reduced = (bool *)calloc(input->ndim, sizeof(bool));              \
    if (!is_reduced) caml_failwith("allocation failed");                       \
    for (int i = 0; i < num_axes; ++i) {                                       \
      if (axes[i] < 0 || axes[i] >= input->ndim) {                             \
        free(is_reduced);                                                      \
        caml_failwith("invalid axis");                                         \
      }                                                                        \
      is_reduced[axes[i]] = true;                                              \
    }                                                                          \
    int num_kept = input->ndim - num_axes;                                     \
    int *kept_axes = (int *)malloc(num_kept * sizeof(int));                    \
    if (!kept_axes) {                                                          \
      free(is_reduced);                                                        \
      caml_failwith("allocation failed");                                      \
    }                                                                          \
    int kk = 0;                                                                \
    for (int i = 0; i < input->ndim; ++i) {                                    \
      if (!is_reduced[i]) kept_axes[kk++] = i;                                 \
    }                                                                          \
    long reduce_prod = 1;                                                      \
    bool zero_size = false;                                                    \
    for (int i = 0; i < num_axes; ++i) {                                       \
      long ss = input->shape[axes[i]];                                         \
      if (ss == 0) zero_size = true;                                           \
      reduce_prod *= ss;                                                       \
    }                                                                          \
    if (zero_size) reduce_prod = 0;                                            \
    if (reduce_prod == 0 && !HAS_IDENTITY) {                                   \
      free(is_reduced);                                                        \
      free(kept_axes);                                                         \
      caml_failwith("zero-size array to reduction operation " #name            \
                    " which has no identity");                                 \
    }                                                                          \
    long total_out = total_elements_safe(output);                              \
    if (total_out == 0) {                                                      \
      free(is_reduced);                                                        \
      free(kept_axes);                                                         \
      return;                                                                  \
    }                                                                          \
    _Pragma("omp parallel for if(total_out > 1000)") for (long idx = 0;        \
                                                          idx < total_out;     \
                                                          ++idx) {             \
      int *local_out_coord = (int *)calloc(output->ndim, sizeof(int));         \
      if (!local_out_coord) caml_failwith("allocation failed");                \
      int *local_in_coord = (int *)calloc(input->ndim, sizeof(int));           \
      if (!local_in_coord) {                                                   \
        free(local_out_coord);                                                 \
        caml_failwith("allocation failed");                                    \
      }                                                                        \
      int *local_reduced_coord = (int *)calloc(num_axes, sizeof(int));         \
      if (!local_reduced_coord) {                                              \
        free(local_out_coord);                                                 \
        free(local_in_coord);                                                  \
        caml_failwith("allocation failed");                                    \
      }                                                                        \
      get_coord_from_idx(idx, output, local_out_coord);                        \
      memset(local_in_coord, 0, input->ndim * sizeof(int));                    \
      if (keepdims) {                                                          \
        for (int d = 0; d < input->ndim; ++d) {                                \
          if (!is_reduced[d]) local_in_coord[d] = local_out_coord[d];          \
        }                                                                      \
      } else {                                                                 \
        for (int ii = 0; ii < num_kept; ++ii) {                                \
          local_in_coord[kept_axes[ii]] = local_out_coord[ii];                 \
        }                                                                      \
      }                                                                        \
      complex32 acc;                                                           \
      if (reduce_prod == 0) {                                                  \
        acc = IDENTITY;                                                        \
      } else {                                                                 \
        bool first = true;                                                     \
        memset(local_reduced_coord, 0, num_axes * sizeof(int));                \
        bool inner_done = false;                                               \
        while (!inner_done) {                                                  \
          for (int j = 0; j < num_axes; ++j) {                                 \
            local_in_coord[axes[j]] = local_reduced_coord[j];                  \
          }                                                                    \
          long in_off = input->offset + get_offset(input, local_in_coord);     \
          caml_ba_complex16 cval = ((caml_ba_complex16 *)input->data)[in_off]; \
          complex32 val = complex16_to_complex32(cval);                        \
          if (first) {                                                         \
            acc = val;                                                         \
            first = false;                                                     \
          } else {                                                             \
            acc = OP(acc, val);                                                \
          }                                                                    \
          inner_done = true;                                                   \
          for (int j = num_axes - 1; j >= 0; --j) {                            \
            local_reduced_coord[j]++;                                          \
            if (local_reduced_coord[j] < input->shape[axes[j]]) {              \
              inner_done = false;                                              \
              break;                                                           \
            }                                                                  \
            local_reduced_coord[j] = 0;                                        \
          }                                                                    \
        }                                                                      \
      }                                                                        \
      long out_off = output->offset + get_offset(output, local_out_coord);     \
      ((caml_ba_complex16 *)output->data)[out_off] =                           \
          complex32_to_complex16(acc);                                         \
      free(local_out_coord);                                                   \
      free(local_in_coord);                                                    \
      free(local_reduced_coord);                                               \
    }                                                                          \
    free(is_reduced);                                                          \
    free(kept_axes);                                                           \
  }

// Int4/Uint4 reduce impl
#define INT4_REDUCE_IMPL(name, signedness, suffix, IDENTITY, HAS_IDENTITY, OP, \
                         CLAMP)                                                \
  static void nx_c_##name##_##suffix(const ndarray_t *input,                   \
                                     ndarray_t *output, const int *axes,       \
                                     int num_axes, bool keepdims) {            \
    if (!input || !output) {                                                   \
      caml_failwith("nx_c_" #name "_" #suffix ": null pointer");               \
    }                                                                          \
    bool *is_reduced = (bool *)calloc(input->ndim, sizeof(bool));              \
    if (!is_reduced) caml_failwith("allocation failed");                       \
    for (int i = 0; i < num_axes; ++i) {                                       \
      if (axes[i] < 0 || axes[i] >= input->ndim) {                             \
        free(is_reduced);                                                      \
        caml_failwith("invalid axis");                                         \
      }                                                                        \
      is_reduced[axes[i]] = true;                                              \
    }                                                                          \
    int num_kept = input->ndim - num_axes;                                     \
    int *kept_axes = (int *)malloc(num_kept * sizeof(int));                    \
    if (!kept_axes) {                                                          \
      free(is_reduced);                                                        \
      caml_failwith("allocation failed");                                      \
    }                                                                          \
    int kk = 0;                                                                \
    for (int i = 0; i < input->ndim; ++i) {                                    \
      if (!is_reduced[i]) kept_axes[kk++] = i;                                 \
    }                                                                          \
    long reduce_prod = 1;                                                      \
    bool zero_size = false;                                                    \
    for (int i = 0; i < num_axes; ++i) {                                       \
      long ss = input->shape[axes[i]];                                         \
      if (ss == 0) zero_size = true;                                           \
      reduce_prod *= ss;                                                       \
    }                                                                          \
    if (zero_size) reduce_prod = 0;                                            \
    if (reduce_prod == 0 && !HAS_IDENTITY) {                                   \
      free(is_reduced);                                                        \
      free(kept_axes);                                                         \
      caml_failwith("zero-size array to reduction operation " #name            \
                    " which has no identity");                                 \
    }                                                                          \
    long total_out = total_elements_safe(output);                              \
    if (total_out == 0) {                                                      \
      free(is_reduced);                                                        \
      free(kept_axes);                                                         \
      return;                                                                  \
    }                                                                          \
    _Pragma("omp parallel for if(total_out > 1000)") for (long idx = 0;        \
                                                          idx < total_out;     \
                                                          ++idx) {             \
      int *local_out_coord = (int *)calloc(output->ndim, sizeof(int));         \
      if (!local_out_coord) caml_failwith("allocation failed");                \
      int *local_in_coord = (int *)calloc(input->ndim, sizeof(int));           \
      if (!local_in_coord) {                                                   \
        free(local_out_coord);                                                 \
        caml_failwith("allocation failed");                                    \
      }                                                                        \
      int *local_reduced_coord = (int *)calloc(num_axes, sizeof(int));         \
      if (!local_reduced_coord) {                                              \
        free(local_out_coord);                                                 \
        free(local_in_coord);                                                  \
        caml_failwith("allocation failed");                                    \
      }                                                                        \
      get_coord_from_idx(idx, output, local_out_coord);                        \
      memset(local_in_coord, 0, input->ndim * sizeof(int));                    \
      if (keepdims) {                                                          \
        for (int d = 0; d < input->ndim; ++d) {                                \
          if (!is_reduced[d]) local_in_coord[d] = local_out_coord[d];          \
        }                                                                      \
      } else {                                                                 \
        for (int ii = 0; ii < num_kept; ++ii) {                                \
          local_in_coord[kept_axes[ii]] = local_out_coord[ii];                 \
        }                                                                      \
      }                                                                        \
      int acc;                                                                 \
      if (reduce_prod == 0) {                                                  \
        acc = IDENTITY;                                                        \
      } else {                                                                 \
        bool first = true;                                                     \
        memset(local_reduced_coord, 0, num_axes * sizeof(int));                \
        bool inner_done = false;                                               \
        while (!inner_done) {                                                  \
          for (int j = 0; j < num_axes; ++j) {                                 \
            local_in_coord[axes[j]] = local_reduced_coord[j];                  \
          }                                                                    \
          long in_off = input->offset + get_offset(input, local_in_coord);     \
          long byte_off = in_off / 2;                                          \
          int nib_off = in_off % 2;                                            \
          uint8_t *in_data = (uint8_t *)input->data;                           \
          int val =                                                            \
              nib_off ? (signedness ? (int8_t)(in_data[byte_off] >> 4)         \
                                    : (in_data[byte_off] >> 4) & 0x0F)         \
                      : (signedness                                            \
                             ? (int8_t)((in_data[byte_off] & 0x0F) << 4) >> 4  \
                             : in_data[byte_off] & 0x0F);                      \
          if (first) {                                                         \
            acc = val;                                                         \
            first = false;                                                     \
          } else {                                                             \
            acc = OP(acc, val);                                                \
          }                                                                    \
          inner_done = true;                                                   \
          for (int j = num_axes - 1; j >= 0; --j) {                            \
            local_reduced_coord[j]++;                                          \
            if (local_reduced_coord[j] < input->shape[axes[j]]) {              \
              inner_done = false;                                              \
              break;                                                           \
            }                                                                  \
            local_reduced_coord[j] = 0;                                        \
          }                                                                    \
        }                                                                      \
      }                                                                        \
      long out_off = output->offset + get_offset(output, local_out_coord);     \
      long out_byte_off = out_off / 2;                                         \
      int out_nib_off = out_off % 2;                                           \
      int res = CLAMP(acc);                                                    \
      uint8_t nib = (uint8_t)res & 0x0F;                                       \
      uint8_t *out_data = (uint8_t *)output->data;                             \
      if (out_nib_off) {                                                       \
        out_data[out_byte_off] = (out_data[out_byte_off] & 0x0F) | (nib << 4); \
      } else {                                                                 \
        out_data[out_byte_off] = (out_data[out_byte_off] & 0xF0) | nib;        \
      }                                                                        \
      free(local_out_coord);                                                   \
      free(local_in_coord);                                                    \
      free(local_reduced_coord);                                               \
    }                                                                          \
    free(is_reduced);                                                          \
    free(kept_axes);                                                           \
  }

// Macro to build dispatch table
#define BUILD_DISPATCH_TABLE(name)                                             \
  static const reduce_op_table name##_table = {.i8 = nx_c_##name##_i8,         \
                                               .u8 = nx_c_##name##_u8,         \
                                               .i16 = nx_c_##name##_i16,       \
                                               .u16 = nx_c_##name##_u16,       \
                                               .i32 = nx_c_##name##_i32,       \
                                               .i64 = nx_c_##name##_i64,       \
                                               .inat = nx_c_##name##_inat,     \
                                               .f16 = nx_c_##name##_f16,       \
                                               .f32 = nx_c_##name##_f32,       \
                                               .f64 = nx_c_##name##_f64,       \
                                               .c32 = nx_c_##name##_c32,       \
                                               .c64 = nx_c_##name##_c64,       \
                                               .bf16 = nx_c_##name##_bf16,     \
                                               .bool_ = nx_c_##name##_bool_,   \
                                               .i4 = nx_c_##name##_i4,         \
                                               .u4 = nx_c_##name##_u4,         \
                                               .f8e4m3 = nx_c_##name##_f8e4m3, \
                                               .f8e5m2 = nx_c_##name##_f8e5m2, \
                                               .c16 = nx_c_##name##_c16,       \
                                               .qi8 = nx_c_##name##_qi8,       \
                                               .qu8 = nx_c_##name##_qu8};

// Generate for reduce_sum
#define SUM_OP(acc, val) ((acc) + (val))
#define SUM_IDENTITY(T) ((T)0)
#define SUM_HAS_IDENTITY 1
#define SUM_OP_FLOAT(acc, val) ((acc) + (val))
#define SUM_IDENTITY_FLOAT 0.0f
#define SUM_COMPLEX_IDENTITY (0)
#define SUM_COMPLEX_OP(acc, val) COMPLEX_ADD(acc, val)

REDUCE_OP_FOR_TYPE(reduce_sum, int8_t, i8, SUM_IDENTITY(int8_t),
                   SUM_HAS_IDENTITY, SUM_OP)
REDUCE_OP_FOR_TYPE(reduce_sum, uint8_t, u8, SUM_IDENTITY(uint8_t),
                   SUM_HAS_IDENTITY, SUM_OP)
REDUCE_OP_FOR_TYPE(reduce_sum, int16_t, i16, SUM_IDENTITY(int16_t),
                   SUM_HAS_IDENTITY, SUM_OP)
REDUCE_OP_FOR_TYPE(reduce_sum, uint16_t, u16, SUM_IDENTITY(uint16_t),
                   SUM_HAS_IDENTITY, SUM_OP)
REDUCE_OP_FOR_TYPE(reduce_sum, int32_t, i32, SUM_IDENTITY(int32_t),
                   SUM_HAS_IDENTITY, SUM_OP)
REDUCE_OP_FOR_TYPE(reduce_sum, int64_t, i64, SUM_IDENTITY(int64_t),
                   SUM_HAS_IDENTITY, SUM_OP)
REDUCE_OP_FOR_TYPE(reduce_sum, intnat, inat, SUM_IDENTITY(intnat),
                   SUM_HAS_IDENTITY, SUM_OP)
// Optimized last-dim contiguous fast paths for f32/f64 reduce_sum
static void nx_c_reduce_sum_f32(const ndarray_t *input, ndarray_t *output,
                                const int *axes, int num_axes, bool keepdims) {
  int last = input->ndim - 1;
  if (num_axes == 1 && axes[0] == last && is_contiguous(input) &&
      is_contiguous(output) && input->ndim >= 1) {
    long K = input->shape[last];
    long total = total_elements_safe(input);
    long M = (K == 0) ? 0 : (total / K);
    float *in = (float *)input->data;
    float *out = (float *)output->data;
    long in_off = input->offset;
    long out_off = output->offset;
    _Pragma("omp parallel for if(M > 1024)")
    for (long r = 0; r < M; ++r) {
      const float *row = in + in_off + r * K;
      float acc = 0.0f;
      for (long p = 0; p < K; ++p) acc += row[p];
      out[out_off + r] = acc;
    }
    return;
  }

  if (num_axes == input->ndim && is_contiguous(input) &&
      total_elements_safe(output) == 1) {
    long total = total_elements_safe(input);
    float *in = (float *)input->data + input->offset;
    float result = 0.0f;
    _Pragma("omp parallel for reduction(+:result) if(total > 4096)")
    for (long i = 0; i < total; ++i) {
      result += in[i];
    }
    float *out = (float *)output->data + output->offset;
    out[0] = result;
    return;
  }

  if (is_contiguous(input) && is_contiguous(output) && num_axes > 0) {
    int sorted_axes[MAX_NDIM];
    if (normalize_and_sort_axes(input->ndim, axes, num_axes, sorted_axes) &&
        axes_are_trailing(input->ndim, num_axes, sorted_axes)) {
      long total = total_elements_safe(input);
      long K = product_of_axes(input, sorted_axes, num_axes);
      float *out = (float *)output->data + output->offset;
      long out_total = total_elements_safe(output);
      if (K == 0 || total == 0 || out_total == 0) {
        for (long i = 0; i < out_total; ++i) out[i] = 0.0f;
        return;
      }
      long M = total / K;
      float *in = (float *)input->data + input->offset;
      _Pragma("omp parallel for if(M > 1024)")
      for (long m = 0; m < M; ++m) {
        const float *chunk = in + (m * K);
        float acc = 0.0f;
        for (long p = 0; p < K; ++p) acc += chunk[p];
        out[m] = acc;
      }
      return;
    }
  }
  // Fallback to generic implementation
  nx_c_reduce_sum_f32_generic_wrap(input, output, axes, num_axes, keepdims);
}

static void nx_c_reduce_sum_f64(const ndarray_t *input, ndarray_t *output,
                                const int *axes, int num_axes, bool keepdims) {
  int last = input->ndim - 1;
  if (num_axes == 1 && axes[0] == last && is_contiguous(input) &&
      is_contiguous(output) && input->ndim >= 1) {
    long K = input->shape[last];
    long total = total_elements_safe(input);
    long M = (K == 0) ? 0 : (total / K);
    double *in = (double *)input->data;
    double *out = (double *)output->data;
    long in_off = input->offset;
    long out_off = output->offset;
    _Pragma("omp parallel for if(M > 1024)")
    for (long r = 0; r < M; ++r) {
      const double *row = in + in_off + r * K;
      double acc = 0.0;
      for (long p = 0; p < K; ++p) acc += row[p];
      out[out_off + r] = acc;
    }
    return;
  }

  if (num_axes == input->ndim && is_contiguous(input) &&
      total_elements_safe(output) == 1) {
    long total = total_elements_safe(input);
    double *in = (double *)input->data + input->offset;
    double result = 0.0;
    _Pragma("omp parallel for reduction(+:result) if(total > 4096)")
    for (long i = 0; i < total; ++i) {
      result += in[i];
    }
    double *out = (double *)output->data + output->offset;
    out[0] = result;
    return;
  }

  if (is_contiguous(input) && is_contiguous(output) && num_axes > 0) {
    int sorted_axes[MAX_NDIM];
    if (normalize_and_sort_axes(input->ndim, axes, num_axes, sorted_axes) &&
        axes_are_trailing(input->ndim, num_axes, sorted_axes)) {
      long total = total_elements_safe(input);
      long K = product_of_axes(input, sorted_axes, num_axes);
      double *out = (double *)output->data + output->offset;
      long out_total = total_elements_safe(output);
      if (K == 0 || total == 0 || out_total == 0) {
        for (long i = 0; i < out_total; ++i) out[i] = 0.0;
        return;
      }
      long M = total / K;
      double *in = (double *)input->data + input->offset;
      _Pragma("omp parallel for if(M > 1024)")
      for (long m = 0; m < M; ++m) {
        const double *chunk = in + (m * K);
        double acc = 0.0;
        for (long p = 0; p < K; ++p) acc += chunk[p];
        out[m] = acc;
      }
      return;
    }
  }
  // Fallback to generic implementation
  nx_c_reduce_sum_f64_generic_wrap(input, output, axes, num_axes, keepdims);
}

// Provide generic versions under alternate names to call in fallback
#define nx_c_reduce_sum_f32_generic nx_c_reduce_sum_f32_fallback
#define nx_c_reduce_sum_f64_generic nx_c_reduce_sum_f64_fallback

REDUCE_OP_FOR_TYPE(reduce_sum, float, f32_fallback, SUM_IDENTITY(float),
                   SUM_HAS_IDENTITY, SUM_OP)
REDUCE_OP_FOR_TYPE(reduce_sum, double, f64_fallback, SUM_IDENTITY(double),
                   SUM_HAS_IDENTITY, SUM_OP)
REDUCE_OP_FOR_TYPE(reduce_sum, complex32, c32, SUM_COMPLEX_IDENTITY,
                   SUM_HAS_IDENTITY, SUM_COMPLEX_OP)
REDUCE_OP_FOR_TYPE(reduce_sum, complex64, c64, SUM_COMPLEX_IDENTITY,
                   SUM_HAS_IDENTITY, SUM_COMPLEX_OP)
REDUCE_OP_FOR_TYPE(reduce_sum, caml_ba_bool, bool_, SUM_IDENTITY(caml_ba_bool),
                   SUM_HAS_IDENTITY, SUM_OP)
REDUCE_OP_FOR_TYPE(reduce_sum, caml_ba_qint8, qi8, SUM_IDENTITY(caml_ba_qint8),
                   SUM_HAS_IDENTITY, SUM_OP)
REDUCE_OP_FOR_TYPE(reduce_sum, caml_ba_quint8, qu8,
                   SUM_IDENTITY(caml_ba_quint8), SUM_HAS_IDENTITY, SUM_OP)

LOW_PREC_REDUCE_OP_IMPL(reduce_sum, uint16_t, f16, SUM_IDENTITY_FLOAT,
                        SUM_HAS_IDENTITY, SUM_OP_FLOAT, half_to_float,
                        float_to_half)
LOW_PREC_REDUCE_OP_IMPL(reduce_sum, caml_ba_bfloat16, bf16, SUM_IDENTITY_FLOAT,
                        SUM_HAS_IDENTITY, SUM_OP_FLOAT, bfloat16_to_float,
                        float_to_bfloat16)
LOW_PREC_REDUCE_OP_IMPL(reduce_sum, caml_ba_fp8_e4m3, f8e4m3,
                        SUM_IDENTITY_FLOAT, SUM_HAS_IDENTITY, SUM_OP_FLOAT,
                        fp8_e4m3_to_float, float_to_fp8_e4m3)
LOW_PREC_REDUCE_OP_IMPL(reduce_sum, caml_ba_fp8_e5m2, f8e5m2,
                        SUM_IDENTITY_FLOAT, SUM_HAS_IDENTITY, SUM_OP_FLOAT,
                        fp8_e5m2_to_float, float_to_fp8_e5m2)

COMPLEX16_REDUCE_IMPL(reduce_sum, SUM_COMPLEX_IDENTITY, SUM_HAS_IDENTITY,
                      SUM_COMPLEX_OP)
  INT4_REDUCE_IMPL(reduce_sum, 1, i4, 0, SUM_HAS_IDENTITY, SUM_OP, CLAMP_I4)
  INT4_REDUCE_IMPL(reduce_sum, 0, u4, 0, SUM_HAS_IDENTITY, SUM_OP, CLAMP_U4)
// Define wrappers now that generic functions exist
static void nx_c_reduce_sum_f32_generic_wrap(const ndarray_t *input,
                                             ndarray_t *output,
                                             const int *axes, int num_axes,
                                             bool keepdims) {
  nx_c_reduce_sum_f32_fallback(input, output, axes, num_axes, keepdims);
}
static void nx_c_reduce_sum_f64_generic_wrap(const ndarray_t *input,
                                             ndarray_t *output,
                                             const int *axes, int num_axes,
                                             bool keepdims) {
  nx_c_reduce_sum_f64_fallback(input, output, axes, num_axes, keepdims);
}
// Build dispatch table (bind f32/f64 to optimized versions)
static const reduce_op_table reduce_sum_table = {
    .i8 = nx_c_reduce_sum_i8,
    .u8 = nx_c_reduce_sum_u8,
    .i16 = nx_c_reduce_sum_i16,
    .u16 = nx_c_reduce_sum_u16,
    .i32 = nx_c_reduce_sum_i32,
    .i64 = nx_c_reduce_sum_i64,
    .inat = nx_c_reduce_sum_inat,
    .f16 = nx_c_reduce_sum_f16,
    .f32 = nx_c_reduce_sum_f32,
    .f64 = nx_c_reduce_sum_f64,
    .c32 = nx_c_reduce_sum_c32,
    .c64 = nx_c_reduce_sum_c64,
    .bf16 = nx_c_reduce_sum_bf16,
    .bool_ = nx_c_reduce_sum_bool_,
    .i4 = nx_c_reduce_sum_i4,
    .u4 = nx_c_reduce_sum_u4,
    .f8e4m3 = nx_c_reduce_sum_f8e4m3,
    .f8e5m2 = nx_c_reduce_sum_f8e5m2,
    .c16 = nx_c_reduce_sum_c16,
    .qi8 = nx_c_reduce_sum_qi8,
    .qu8 = nx_c_reduce_sum_qu8};

// Generate for reduce_max
#define MAX_OP(acc, val) ((acc) > (val) ? (acc) : (val))
#define MAX_IDENTITY(T) ((T)0)  // unused
#define MAX_HAS_IDENTITY 0
/* NaN propagation: if either operand is NaN, return NaN */
#define MAX_OP_FLOAT(acc, val) \
  (isnan(acc) || isnan(val) ? NAN : ((acc) > (val) ? (acc) : (val)))
#define MAX_IDENTITY_FLOAT 0.0f   // unused
#define MAX_COMPLEX_IDENTITY (0)  // unused
#define MAX_COMPLEX_OP(acc, val) complex_max(acc, val)
#define MAX_COMPLEX64_OP(acc, val) complex64_max(acc, val)

// Optimized last-dim contiguous fast paths for f32/f64 reduce_max
static void nx_c_reduce_max_f32(const ndarray_t *input, ndarray_t *output,
                                const int *axes, int num_axes, bool keepdims) {
  int last = input->ndim - 1;
  if (num_axes == 1 && axes[0] == last && is_contiguous(input) &&
      is_contiguous(output) && input->ndim >= 1) {
    long K = input->shape[last];
    long total = total_elements_safe(input);
    long M = (K == 0) ? 0 : (total / K);
    float *in = (float *)input->data;
    float *out = (float *)output->data;
    long in_off = input->offset;
    long out_off = output->offset;
    _Pragma("omp parallel for if(M > 1024)")
    for (long r = 0; r < M; ++r) {
      const float *row = in + in_off + r * K;
      float acc = -INFINITY;
      for (long p = 0; p < K; ++p) {
        float v = row[p];
        acc = (isnan(acc) || isnan(v)) ? NAN : (v > acc ? v : acc);
      }
      out[out_off + r] = acc;
    }
    return;
  }
  // Fallback to generic implementation
  nx_c_reduce_max_f32_generic_wrap(input, output, axes, num_axes, keepdims);
}

static void nx_c_reduce_max_f64(const ndarray_t *input, ndarray_t *output,
                                const int *axes, int num_axes, bool keepdims) {
  int last = input->ndim - 1;
  if (num_axes == 1 && axes[0] == last && is_contiguous(input) &&
      is_contiguous(output) && input->ndim >= 1) {
    long K = input->shape[last];
    long total = total_elements_safe(input);
    long M = (K == 0) ? 0 : (total / K);
    double *in = (double *)input->data;
    double *out = (double *)output->data;
    long in_off = input->offset;
    long out_off = output->offset;
    _Pragma("omp parallel for if(M > 1024)")
    for (long r = 0; r < M; ++r) {
      const double *row = in + in_off + r * K;
      double acc = -INFINITY;
      for (long p = 0; p < K; ++p) {
        double v = row[p];
        acc = (isnan(acc) || isnan(v)) ? NAN : (v > acc ? v : acc);
      }
      out[out_off + r] = acc;
    }
    return;
  }
  // Fallback to generic implementation
  nx_c_reduce_max_f64_generic_wrap(input, output, axes, num_axes, keepdims);
}

// Provide generic versions under alternate names to call in fallback
#define nx_c_reduce_max_f32_generic nx_c_reduce_max_f32_fallback
#define nx_c_reduce_max_f64_generic nx_c_reduce_max_f64_fallback

REDUCE_OP_FOR_TYPE(reduce_max, int8_t, i8, MAX_IDENTITY(int8_t),
                   MAX_HAS_IDENTITY, MAX_OP)
REDUCE_OP_FOR_TYPE(reduce_max, uint8_t, u8, MAX_IDENTITY(uint8_t),
                   MAX_HAS_IDENTITY, MAX_OP)
REDUCE_OP_FOR_TYPE(reduce_max, int16_t, i16, MAX_IDENTITY(int16_t),
                   MAX_HAS_IDENTITY, MAX_OP)
REDUCE_OP_FOR_TYPE(reduce_max, uint16_t, u16, MAX_IDENTITY(uint16_t),
                   MAX_HAS_IDENTITY, MAX_OP)
REDUCE_OP_FOR_TYPE(reduce_max, int32_t, i32, MAX_IDENTITY(int32_t),
                   MAX_HAS_IDENTITY, MAX_OP)
REDUCE_OP_FOR_TYPE(reduce_max, int64_t, i64, MAX_IDENTITY(int64_t),
                   MAX_HAS_IDENTITY, MAX_OP)
REDUCE_OP_FOR_TYPE(reduce_max, intnat, inat, MAX_IDENTITY(intnat),
                   MAX_HAS_IDENTITY, MAX_OP)
REDUCE_OP_FOR_TYPE(reduce_max, float, f32_fallback, MAX_IDENTITY(float),
                   MAX_HAS_IDENTITY, MAX_OP_FLOAT)
REDUCE_OP_FOR_TYPE(reduce_max, double, f64_fallback, MAX_IDENTITY(double),
                   MAX_HAS_IDENTITY, MAX_OP_FLOAT)
REDUCE_OP_FOR_TYPE(reduce_max, complex32, c32, MAX_COMPLEX_IDENTITY,
                   MAX_HAS_IDENTITY, MAX_COMPLEX_OP)
REDUCE_OP_FOR_TYPE(reduce_max, complex64, c64, MAX_COMPLEX_IDENTITY,
                   MAX_HAS_IDENTITY, MAX_COMPLEX64_OP)
REDUCE_OP_FOR_TYPE(reduce_max, caml_ba_bool, bool_, MAX_IDENTITY(caml_ba_bool),
                   MAX_HAS_IDENTITY, MAX_OP)
REDUCE_OP_FOR_TYPE(reduce_max, caml_ba_qint8, qi8, MAX_IDENTITY(caml_ba_qint8),
                   MAX_HAS_IDENTITY, MAX_OP)
REDUCE_OP_FOR_TYPE(reduce_max, caml_ba_quint8, qu8,
                   MAX_IDENTITY(caml_ba_quint8), MAX_HAS_IDENTITY, MAX_OP)

LOW_PREC_REDUCE_OP_IMPL(reduce_max, uint16_t, f16, MAX_IDENTITY_FLOAT,
                        MAX_HAS_IDENTITY, MAX_OP_FLOAT, half_to_float,
                        float_to_half)
LOW_PREC_REDUCE_OP_IMPL(reduce_max, caml_ba_bfloat16, bf16, MAX_IDENTITY_FLOAT,
                        MAX_HAS_IDENTITY, MAX_OP_FLOAT, bfloat16_to_float,
                        float_to_bfloat16)
LOW_PREC_REDUCE_OP_IMPL(reduce_max, caml_ba_fp8_e4m3, f8e4m3,
                        MAX_IDENTITY_FLOAT, MAX_HAS_IDENTITY, MAX_OP_FLOAT,
                        fp8_e4m3_to_float, float_to_fp8_e4m3)
LOW_PREC_REDUCE_OP_IMPL(reduce_max, caml_ba_fp8_e5m2, f8e5m2,
                        MAX_IDENTITY_FLOAT, MAX_HAS_IDENTITY, MAX_OP_FLOAT,
                        fp8_e5m2_to_float, float_to_fp8_e5m2)

COMPLEX16_REDUCE_IMPL(reduce_max, MAX_COMPLEX_IDENTITY, MAX_HAS_IDENTITY,
                      complex_max)
INT4_REDUCE_IMPL(reduce_max, 1, i4, 0, MAX_HAS_IDENTITY, MAX_OP, CLAMP_I4)
INT4_REDUCE_IMPL(reduce_max, 0, u4, 0, MAX_HAS_IDENTITY, MAX_OP, CLAMP_U4)
// Define wrappers now that generic functions exist
static void nx_c_reduce_max_f32_generic_wrap(const ndarray_t *input,
                                             ndarray_t *output,
                                             const int *axes, int num_axes,
                                             bool keepdims) {
  nx_c_reduce_max_f32_fallback(input, output, axes, num_axes, keepdims);
}
static void nx_c_reduce_max_f64_generic_wrap(const ndarray_t *input,
                                             ndarray_t *output,
                                             const int *axes, int num_axes,
                                             bool keepdims) {
  nx_c_reduce_max_f64_fallback(input, output, axes, num_axes, keepdims);
}
// Build dispatch table (bind f32/f64 to optimized versions)
static const reduce_op_table reduce_max_table = {
    .i8 = nx_c_reduce_max_i8,
    .u8 = nx_c_reduce_max_u8,
    .i16 = nx_c_reduce_max_i16,
    .u16 = nx_c_reduce_max_u16,
    .i32 = nx_c_reduce_max_i32,
    .i64 = nx_c_reduce_max_i64,
    .inat = nx_c_reduce_max_inat,
    .f16 = nx_c_reduce_max_f16,
    .f32 = nx_c_reduce_max_f32,
    .f64 = nx_c_reduce_max_f64,
    .c32 = nx_c_reduce_max_c32,
    .c64 = nx_c_reduce_max_c64,
    .bf16 = nx_c_reduce_max_bf16,
    .bool_ = nx_c_reduce_max_bool_,
    .i4 = nx_c_reduce_max_i4,
    .u4 = nx_c_reduce_max_u4,
    .f8e4m3 = nx_c_reduce_max_f8e4m3,
    .f8e5m2 = nx_c_reduce_max_f8e5m2,
    .c16 = nx_c_reduce_max_c16,
    .qi8 = nx_c_reduce_max_qi8,
    .qu8 = nx_c_reduce_max_qu8};

// Generate for reduce_prod
#define PROD_OP(acc, val) ((acc) * (val))
#define PROD_IDENTITY(T) ((T)1)
#define PROD_HAS_IDENTITY 1
#define PROD_OP_FLOAT(acc, val) ((acc) * (val))
#define PROD_IDENTITY_FLOAT 1.0f
#define PROD_COMPLEX_IDENTITY (1)
#define PROD_COMPLEX_OP(acc, val) COMPLEX_MUL(acc, val)

REDUCE_OP_FOR_TYPE(reduce_prod, int8_t, i8, PROD_IDENTITY(int8_t),
                   PROD_HAS_IDENTITY, PROD_OP)
REDUCE_OP_FOR_TYPE(reduce_prod, uint8_t, u8, PROD_IDENTITY(uint8_t),
                   PROD_HAS_IDENTITY, PROD_OP)
REDUCE_OP_FOR_TYPE(reduce_prod, int16_t, i16, PROD_IDENTITY(int16_t),
                   PROD_HAS_IDENTITY, PROD_OP)
REDUCE_OP_FOR_TYPE(reduce_prod, uint16_t, u16, PROD_IDENTITY(uint16_t),
                   PROD_HAS_IDENTITY, PROD_OP)
REDUCE_OP_FOR_TYPE(reduce_prod, int32_t, i32, PROD_IDENTITY(int32_t),
                   PROD_HAS_IDENTITY, PROD_OP)
REDUCE_OP_FOR_TYPE(reduce_prod, int64_t, i64, PROD_IDENTITY(int64_t),
                   PROD_HAS_IDENTITY, PROD_OP)
REDUCE_OP_FOR_TYPE(reduce_prod, intnat, inat, PROD_IDENTITY(intnat),
                   PROD_HAS_IDENTITY, PROD_OP)
REDUCE_OP_FOR_TYPE(reduce_prod, float, f32, PROD_IDENTITY(float),
                   PROD_HAS_IDENTITY, PROD_OP)
REDUCE_OP_FOR_TYPE(reduce_prod, double, f64, PROD_IDENTITY(double),
                   PROD_HAS_IDENTITY, PROD_OP)
REDUCE_OP_FOR_TYPE(reduce_prod, complex32, c32, PROD_COMPLEX_IDENTITY,
                   PROD_HAS_IDENTITY, PROD_COMPLEX_OP)
REDUCE_OP_FOR_TYPE(reduce_prod, complex64, c64, PROD_COMPLEX_IDENTITY,
                   PROD_HAS_IDENTITY, PROD_COMPLEX_OP)
REDUCE_OP_FOR_TYPE(reduce_prod, caml_ba_bool, bool_,
                   PROD_IDENTITY(caml_ba_bool), PROD_HAS_IDENTITY, PROD_OP)
REDUCE_OP_FOR_TYPE(reduce_prod, caml_ba_qint8, qi8,
                   PROD_IDENTITY(caml_ba_qint8), PROD_HAS_IDENTITY, PROD_OP)
REDUCE_OP_FOR_TYPE(reduce_prod, caml_ba_quint8, qu8,
                   PROD_IDENTITY(caml_ba_quint8), PROD_HAS_IDENTITY, PROD_OP)

LOW_PREC_REDUCE_OP_IMPL(reduce_prod, uint16_t, f16, PROD_IDENTITY_FLOAT,
                        PROD_HAS_IDENTITY, PROD_OP_FLOAT, half_to_float,
                        float_to_half)
LOW_PREC_REDUCE_OP_IMPL(reduce_prod, caml_ba_bfloat16, bf16,
                        PROD_IDENTITY_FLOAT, PROD_HAS_IDENTITY, PROD_OP_FLOAT,
                        bfloat16_to_float, float_to_bfloat16)
LOW_PREC_REDUCE_OP_IMPL(reduce_prod, caml_ba_fp8_e4m3, f8e4m3,
                        PROD_IDENTITY_FLOAT, PROD_HAS_IDENTITY, PROD_OP_FLOAT,
                        fp8_e4m3_to_float, float_to_fp8_e4m3)
LOW_PREC_REDUCE_OP_IMPL(reduce_prod, caml_ba_fp8_e5m2, f8e5m2,
                        PROD_IDENTITY_FLOAT, PROD_HAS_IDENTITY, PROD_OP_FLOAT,
                        fp8_e5m2_to_float, float_to_fp8_e5m2)

COMPLEX16_REDUCE_IMPL(reduce_prod, PROD_COMPLEX_IDENTITY, PROD_HAS_IDENTITY,
                      COMPLEX_MUL)
INT4_REDUCE_IMPL(reduce_prod, 1, i4, 1, PROD_HAS_IDENTITY, PROD_OP, CLAMP_I4)
INT4_REDUCE_IMPL(reduce_prod, 0, u4, 1, PROD_HAS_IDENTITY, PROD_OP, CLAMP_U4)
BUILD_DISPATCH_TABLE(reduce_prod)

// Generic dispatch function for reduction operations
static void dispatch_reduce_op(value v_input, value v_output, int *axes,
                               int num_axes, bool keepdims,
                               const reduce_op_table *table,
                               const char *op_name) {
  ndarray_t input = extract_ndarray(v_input);
  ndarray_t output = extract_ndarray(v_output);

  // Validate axes before entering blocking section
  // (Cannot call caml_failwith from within blocking section)
  for (int i = 0; i < num_axes; ++i) {
    if (axes[i] < 0 || axes[i] >= input.ndim) {
      char msg[256];
      snprintf(msg, sizeof(msg), "%s: axis %d out of bounds for tensor of rank %d",
               op_name, axes[i], input.ndim);
      cleanup_ndarray(&input);
      cleanup_ndarray(&output);
      caml_failwith(msg);
    }
  }

  // Sort axes for consistency
  qsort(axes, num_axes, sizeof(int), cmp_int);

  // Check dtypes match
  value v_input_data = Field(v_input, FFI_TENSOR_DATA);
  value v_output_data = Field(v_output, FFI_TENSOR_DATA);
  struct caml_ba_array *ba_input = Caml_ba_array_val(v_input_data);
  struct caml_ba_array *ba_output = Caml_ba_array_val(v_output_data);
  int kind_input = nx_ba_get_kind(ba_input);
  int kind_output = nx_ba_get_kind(ba_output);
  if (kind_input != kind_output) {
    cleanup_ndarray(&input);
    cleanup_ndarray(&output);
    caml_failwith("dtype mismatch");
  }

  // Select operation based on dtype
  reduce_op_t op = NULL;
  switch (kind_input) {
    case CAML_BA_SINT8:
      op = table->i8;
      break;
    case CAML_BA_UINT8:
      op = table->u8;
      break;
    case CAML_BA_SINT16:
      op = table->i16;
      break;
    case CAML_BA_UINT16:
      op = table->u16;
      break;
    case CAML_BA_INT32:
      op = table->i32;
      break;
    case CAML_BA_INT64:
      op = table->i64;
      break;
    case CAML_BA_CAML_INT:
    case CAML_BA_NATIVE_INT:
      op = table->inat;
      break;
    case CAML_BA_FLOAT16:
      op = table->f16;
      break;
    case CAML_BA_FLOAT32:
      op = table->f32;
      break;
    case CAML_BA_FLOAT64:
      op = table->f64;
      break;
    case CAML_BA_COMPLEX32:
      op = table->c32;
      break;
    case CAML_BA_COMPLEX64:
      op = table->c64;
      break;
    case NX_BA_BFLOAT16:
      op = table->bf16;
      break;
    case NX_BA_BOOL:
      op = table->bool_;
      break;
    case NX_BA_INT4:
      op = table->i4;
      break;
    case NX_BA_UINT4:
      op = table->u4;
      break;
    case NX_BA_FP8_E4M3:
      op = table->f8e4m3;
      break;
    case NX_BA_FP8_E5M2:
      op = table->f8e5m2;
      break;
    case NX_BA_COMPLEX16:
      op = table->c16;
      break;
    case NX_BA_QINT8:
      op = table->qi8;
      break;
    case NX_BA_QUINT8:
      op = table->qu8;
      break;
    default:
      cleanup_ndarray(&input);
      cleanup_ndarray(&output);
      caml_failwith("dispatch_reduce_op: unsupported dtype");
  }

  if (!op) {
    char msg[256];
    snprintf(msg, sizeof(msg), "%s: operation not supported for dtype",
             op_name);
    cleanup_ndarray(&input);
    cleanup_ndarray(&output);
    caml_failwith(msg);
  }

  // Enter blocking section for potentially long computation
  caml_enter_blocking_section();
  op(&input, &output, axes, num_axes, keepdims);
  caml_leave_blocking_section();

  // Clean up if heap allocated
  cleanup_ndarray(&input);
  cleanup_ndarray(&output);
}

// ============================================================================
// OCaml FFI Stubs
// ============================================================================

// Macro to define FFI stub for each operation
#define DEFINE_FFI_STUB(name)                                                  \
  CAMLprim value caml_nx_##name(value v_input, value v_output, value v_axes,   \
                                value v_keepdims) {                            \
    CAMLparam4(v_input, v_output, v_axes, v_keepdims);                         \
    int num_axes = Wosize_val(v_axes);                                         \
    int *axes = (int *)malloc(num_axes * sizeof(int));                         \
    if (!axes) caml_failwith("allocation failed");                             \
    for (int i = 0; i < num_axes; ++i) {                                       \
      axes[i] = Int_val(Field(v_axes, i));                                     \
    }                                                                          \
    bool keepdims = Bool_val(v_keepdims);                                      \
    dispatch_reduce_op(v_input, v_output, axes, num_axes, keepdims,            \
                       &name##_table, #name);                                  \
    free(axes);                                                                \
    CAMLreturn(Val_unit);                                                      \
  }

DEFINE_FFI_STUB(reduce_sum)
DEFINE_FFI_STUB(reduce_max)
DEFINE_FFI_STUB(reduce_prod)
