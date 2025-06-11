/* nx_cblas_ops_impl.c - CBLAS operations implementation */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

/* =========================== Configuration =========================== */

#define MAX_STACK_DIMS 8 /* Most tensors have <= 8 dimensions */
#define PARALLEL_THRESHOLD \
  10000 /* Use parallel ops for arrays larger than this */
#define VECTORIZATION_THRESHOLD 64 /* Min size for vectorization */
#define CACHE_LINE_SIZE 64         /* Typical cache line size */
#define L1_CACHE_SIZE 32768        /* 32KB L1 cache */

/* =========================== Type Definitions =========================== */

typedef struct {
  void *data;
  int ndim;
  int *shape;
  int *strides;
  int offset;
} strided_array_t;

/* =========================== Helper Functions =========================== */

void nx_cblas_init_strided_array(strided_array_t *arr, void *data, int ndim,
                                 const int *shape, const int *strides,
                                 int offset) {
  arr->data = data;
  arr->ndim = ndim;
  arr->offset = offset;
  arr->shape = (int *)malloc(ndim * sizeof(int));
  arr->strides = (int *)malloc(ndim * sizeof(int));

  for (int i = 0; i < ndim; i++) {
    arr->shape[i] = shape[i];
    arr->strides[i] = strides[i];
  }
}

void nx_cblas_free_strided_array(strided_array_t *arr) {
  free(arr->shape);
  free(arr->strides);
}

static inline int is_contiguous(const strided_array_t *arr) {
  int expected_stride = 1;
  for (int i = arr->ndim - 1; i >= 0; i--) {
    if (arr->strides[i] != expected_stride) return 0;
    expected_stride *= arr->shape[i];
  }
  return 1;  /* Allow non-zero offsets as we handle them properly */
}

static inline int total_elements(const strided_array_t *arr) {
  int total = 1;
  for (int i = 0; i < arr->ndim; i++) {
    total *= arr->shape[i];
  }
  return total;
}

/* =========================== Iteration =========================== */

/* Iterator state for efficient strided traversal */
typedef struct {
  int ndim;
  int shape[MAX_STACK_DIMS];
  int strides_x[MAX_STACK_DIMS];
  int strides_y[MAX_STACK_DIMS];
  int strides_z[MAX_STACK_DIMS];
  int indices[MAX_STACK_DIMS];
  int offset_x;
  int offset_y;
  int offset_z;
} iterator_state_t;

static inline void init_binary_iterator(iterator_state_t *it,
                                        const strided_array_t *x,
                                        const strided_array_t *y,
                                        const strided_array_t *z) {
  it->ndim = x->ndim;
  it->offset_x = x->offset;
  it->offset_y = y->offset;
  it->offset_z = z->offset;

  for (int i = 0; i < x->ndim; i++) {
    it->shape[i] = x->shape[i];
    it->strides_x[i] = x->strides[i];
    it->strides_y[i] = y->strides[i];
    it->strides_z[i] = z->strides[i];
    it->indices[i] = 0;
  }
}

static inline void init_unary_iterator(iterator_state_t *it,
                                       const strided_array_t *x,
                                       const strided_array_t *z) {
  it->ndim = x->ndim;
  it->offset_x = x->offset;
  it->offset_z = z->offset;

  for (int i = 0; i < x->ndim; i++) {
    it->shape[i] = x->shape[i];
    it->strides_x[i] = x->strides[i];
    it->strides_z[i] = z->strides[i];
    it->indices[i] = 0;
  }
}

/* =========================== Binary Operations =========================== */

/* Generic binary operation for float32 */
#define BINARY_OP_F32(name, OP)                                                       \
  void nx_cblas_##name##_f32(const strided_array_t *x,                                \
                             const strided_array_t *y, strided_array_t *z) {          \
    float *x_data = (float *)x->data;                                                 \
    float *y_data = (float *)y->data;                                                 \
    float *z_data = (float *)z->data;                                                 \
    int total = total_elements(x);                                                    \
                                                                                      \
    /* Fast path: contiguous arrays */                                                \
    if (is_contiguous(x) && is_contiguous(y) && is_contiguous(z)) {                   \
      /* Vectorized loop */                                                           \
      if (total > PARALLEL_THRESHOLD * 4) {                                           \
        /* Parallel execution for very large arrays */                                \
        _Pragma(                                                                      \
            "omp parallel for simd aligned(x_data, y_data, z_data: 32)") for (int i = \
                                                                                  0;  \
                                                                              i <     \
                                                                              total;  \
                                                                              i++) {  \
          z_data[i] = OP(x_data[i], y_data[i]);                                       \
        }                                                                             \
      } else {                                                                        \
        /* SIMD only for medium arrays */                                             \
        _Pragma(                                                                      \
            "omp simd aligned(x_data, y_data, z_data: 32)") for (int i = 0;           \
                                                                 i < total;           \
                                                                 i++) {               \
          z_data[i] = OP(x_data[i], y_data[i]);                                       \
        }                                                                             \
      }                                                                               \
    } /* strided path */                                                              \
    else if (x->ndim <= MAX_STACK_DIMS) {                                             \
      iterator_state_t it;                                                            \
      init_binary_iterator(&it, x, y, z);                                             \
                                                                                      \
      /* Generate nested loops based on dimensionality */                             \
      switch (it.ndim) {                                                              \
        case 1: {                                                                     \
          for (int i0 = 0; i0 < it.shape[0]; i0++) {                                  \
            int idx_x = it.offset_x + i0 * it.strides_x[0];                           \
            int idx_y = it.offset_y + i0 * it.strides_y[0];                           \
            int idx_z = it.offset_z + i0 * it.strides_z[0];                           \
            z_data[idx_z] = OP(x_data[idx_x], y_data[idx_y]);                         \
          }                                                                           \
          break;                                                                      \
        }                                                                             \
        case 2: {                                                                     \
          for (int i0 = 0; i0 < it.shape[0]; i0++) {                                  \
            int base_x = it.offset_x + i0 * it.strides_x[0];                          \
            int base_y = it.offset_y + i0 * it.strides_y[0];                          \
            int base_z = it.offset_z + i0 * it.strides_z[0];                          \
            for (int i1 = 0; i1 < it.shape[1]; i1++) {                                \
              int idx_x = base_x + i1 * it.strides_x[1];                              \
              int idx_y = base_y + i1 * it.strides_y[1];                              \
              int idx_z = base_z + i1 * it.strides_z[1];                              \
              z_data[idx_z] = OP(x_data[idx_x], y_data[idx_y]);                       \
            }                                                                         \
          }                                                                           \
          break;                                                                      \
        }                                                                             \
        default: {                                                                    \
          /* General case for higher dimensions */                                    \
          int indices[MAX_STACK_DIMS] = {0};                                          \
          for (int i = 0; i < total; i++) {                                           \
            /* Compute strided indices */                                             \
            int idx_x = it.offset_x;                                                  \
            int idx_y = it.offset_y;                                                  \
            int idx_z = it.offset_z;                                                  \
            for (int d = 0; d < it.ndim; d++) {                                       \
              idx_x += indices[d] * it.strides_x[d];                                  \
              idx_y += indices[d] * it.strides_y[d];                                  \
              idx_z += indices[d] * it.strides_z[d];                                  \
            }                                                                         \
            z_data[idx_z] = OP(x_data[idx_x], y_data[idx_y]);                         \
                                                                                      \
            /* Increment indices */                                                   \
            for (int d = it.ndim - 1; d >= 0; d--) {                                  \
              if (++indices[d] < it.shape[d]) break;                                  \
              indices[d] = 0;                                                         \
            }                                                                         \
          }                                                                           \
        }                                                                             \
      }                                                                               \
    } /* Fallback for very high dimensional arrays */                                 \
    else {                                                                            \
      int *indices = (int *)calloc(x->ndim, sizeof(int));                             \
      for (int i = 0; i < total; i++) {                                               \
        int idx_x = x->offset;                                                        \
        int idx_y = y->offset;                                                        \
        int idx_z = z->offset;                                                        \
        for (int d = 0; d < x->ndim; d++) {                                           \
          idx_x += indices[d] * x->strides[d];                                        \
          idx_y += indices[d] * y->strides[d];                                        \
          idx_z += indices[d] * z->strides[d];                                        \
        }                                                                             \
        z_data[idx_z] = OP(x_data[idx_x], y_data[idx_y]);                             \
                                                                                      \
        for (int d = x->ndim - 1; d >= 0; d--) {                                      \
          if (++indices[d] < x->shape[d]) break;                                      \
          indices[d] = 0;                                                             \
        }                                                                             \
      }                                                                               \
      free(indices);                                                                  \
    }                                                                                 \
  }

/* Operation macros */
#define ADD(x, y) ((x) + (y))
#define SUB(x, y) ((x) - (y))
#define MUL(x, y) ((x) * (y))
#define DIV(x, y) ((x) / (y))
#undef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define NEG(x) (-(x))
#define RECIP(x) (1.0f / (x))

/* Define binary operations for float32 */
/* Special handling for add - uses BLAS */
void nx_cblas_add_f32(const strided_array_t *x, const strided_array_t *y,
                      strided_array_t *z) {
  float *x_data = (float *)x->data;
  float *y_data = (float *)y->data;
  float *z_data = (float *)z->data;
  int total = total_elements(x);

  if (is_contiguous(x) && is_contiguous(y) && is_contiguous(z)) {
    /* Offset-adjusted pointers for contiguous arrays */
    float *x_ptr = x_data + x->offset;
    float *y_ptr = y_data + y->offset;
    float *z_ptr = z_data + z->offset;
    
    /* Simple loop for now */
    for (int i = 0; i < total; i++) {
      z_ptr[i] = x_ptr[i] + y_ptr[i];
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    iterator_state_t it;
    init_binary_iterator(&it, x, y, z);

    switch (it.ndim) {
      case 1: {
        for (int i0 = 0; i0 < it.shape[0]; i0++) {
          int idx_x = it.offset_x + i0 * it.strides_x[0];
          int idx_y = it.offset_y + i0 * it.strides_y[0];
          int idx_z = it.offset_z + i0 * it.strides_z[0];
          z_data[idx_z] = x_data[idx_x] + y_data[idx_y];
        }
        break;
      }
      case 2: {
        for (int i0 = 0; i0 < it.shape[0]; i0++) {
          int base_x = it.offset_x + i0 * it.strides_x[0];
          int base_y = it.offset_y + i0 * it.strides_y[0];
          int base_z = it.offset_z + i0 * it.strides_z[0];
          for (int i1 = 0; i1 < it.shape[1]; i1++) {
            int idx_x = base_x + i1 * it.strides_x[1];
            int idx_y = base_y + i1 * it.strides_y[1];
            int idx_z = base_z + i1 * it.strides_z[1];
            z_data[idx_z] = x_data[idx_x] + y_data[idx_y];
          }
        }
        break;
      }
      default: {
        int indices[MAX_STACK_DIMS] = {0};
        for (int i = 0; i < total; i++) {
          int idx_x = it.offset_x;
          int idx_y = it.offset_y;
          int idx_z = it.offset_z;
          for (int d = 0; d < it.ndim; d++) {
            idx_x += indices[d] * it.strides_x[d];
            idx_y += indices[d] * it.strides_y[d];
            idx_z += indices[d] * it.strides_z[d];
          }
          z_data[idx_z] = x_data[idx_x] + y_data[idx_y];

          for (int d = it.ndim - 1; d >= 0; d--) {
            if (++indices[d] < it.shape[d]) break;
            indices[d] = 0;
          }
        }
      }
    }
  } else {
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx_x = x->offset;
      int idx_y = y->offset;
      int idx_z = z->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx_x += indices[d] * x->strides[d];
        idx_y += indices[d] * y->strides[d];
        idx_z += indices[d] * z->strides[d];
      }
      z_data[idx_z] = x_data[idx_x] + y_data[idx_y];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }
}

/* Similar for sub */
void nx_cblas_sub_f32(const strided_array_t *x, const strided_array_t *y,
                      strided_array_t *z) {
  float *x_data = (float *)x->data;
  float *y_data = (float *)y->data;
  float *z_data = (float *)z->data;
  int total = total_elements(x);

  if (is_contiguous(x) && is_contiguous(y) && is_contiguous(z)) {
    /* Offset-adjusted pointers for contiguous arrays */
    float *x_ptr = x_data + x->offset;
    float *y_ptr = y_data + y->offset;
    float *z_ptr = z_data + z->offset;
    
    /* Simple loop for now */
    for (int i = 0; i < total; i++) {
      z_ptr[i] = x_ptr[i] - y_ptr[i];
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    iterator_state_t it;
    init_binary_iterator(&it, x, y, z);

    switch (it.ndim) {
      case 1: {
        for (int i0 = 0; i0 < it.shape[0]; i0++) {
          int idx_x = it.offset_x + i0 * it.strides_x[0];
          int idx_y = it.offset_y + i0 * it.strides_y[0];
          int idx_z = it.offset_z + i0 * it.strides_z[0];
          z_data[idx_z] = x_data[idx_x] - y_data[idx_y];
        }
        break;
      }
      case 2: {
        for (int i0 = 0; i0 < it.shape[0]; i0++) {
          int base_x = it.offset_x + i0 * it.strides_x[0];
          int base_y = it.offset_y + i0 * it.strides_y[0];
          int base_z = it.offset_z + i0 * it.strides_z[0];
          for (int i1 = 0; i1 < it.shape[1]; i1++) {
            int idx_x = base_x + i1 * it.strides_x[1];
            int idx_y = base_y + i1 * it.strides_y[1];
            int idx_z = base_z + i1 * it.strides_z[1];
            z_data[idx_z] = x_data[idx_x] - y_data[idx_y];
          }
        }
        break;
      }
      default: {
        int indices[MAX_STACK_DIMS] = {0};
        for (int i = 0; i < total; i++) {
          int idx_x = it.offset_x;
          int idx_y = it.offset_y;
          int idx_z = it.offset_z;
          for (int d = 0; d < it.ndim; d++) {
            idx_x += indices[d] * it.strides_x[d];
            idx_y += indices[d] * it.strides_y[d];
            idx_z += indices[d] * it.strides_z[d];
          }
          z_data[idx_z] = x_data[idx_x] - y_data[idx_y];

          for (int d = it.ndim - 1; d >= 0; d--) {
            if (++indices[d] < it.shape[d]) break;
            indices[d] = 0;
          }
        }
      }
    }
  } else {
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx_x = x->offset;
      int idx_y = y->offset;
      int idx_z = z->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx_x += indices[d] * x->strides[d];
        idx_y += indices[d] * y->strides[d];
        idx_z += indices[d] * z->strides[d];
      }
      z_data[idx_z] = x_data[idx_x] - y_data[idx_y];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }
}

BINARY_OP_F32(mul, MUL)
BINARY_OP_F32(div, DIV)
BINARY_OP_F32(max, MAX)

/* Special handling for pow and mod - no SIMD */
void nx_cblas_pow_f32(const strided_array_t *x, const strided_array_t *y,
                      strided_array_t *z) {
  float *x_data = (float *)x->data;
  float *y_data = (float *)y->data;
  float *z_data = (float *)z->data;
  int total = total_elements(x);

  if (is_contiguous(x) && is_contiguous(y) && is_contiguous(z)) {
    for (int i = 0; i < total; i++) {
      z_data[i] = powf(x_data[i], y_data[i]);
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    iterator_state_t it;
    init_binary_iterator(&it, x, y, z);

    int indices[MAX_STACK_DIMS] = {0};
    for (int i = 0; i < total; i++) {
      int idx_x = it.offset_x;
      int idx_y = it.offset_y;
      int idx_z = it.offset_z;
      for (int d = 0; d < it.ndim; d++) {
        idx_x += indices[d] * it.strides_x[d];
        idx_y += indices[d] * it.strides_y[d];
        idx_z += indices[d] * it.strides_z[d];
      }
      z_data[idx_z] = powf(x_data[idx_x], y_data[idx_y]);

      for (int d = it.ndim - 1; d >= 0; d--) {
        if (++indices[d] < it.shape[d]) break;
        indices[d] = 0;
      }
    }
  } else {
    /* Fallback for high dimensional arrays */
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx_x = x->offset;
      int idx_y = y->offset;
      int idx_z = z->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx_x += indices[d] * x->strides[d];
        idx_y += indices[d] * y->strides[d];
        idx_z += indices[d] * z->strides[d];
      }
      z_data[idx_z] = powf(x_data[idx_x], y_data[idx_y]);

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }
}

void nx_cblas_mod_f32(const strided_array_t *x, const strided_array_t *y,
                      strided_array_t *z) {
  float *x_data = (float *)x->data;
  float *y_data = (float *)y->data;
  float *z_data = (float *)z->data;
  int total = total_elements(x);

  if (is_contiguous(x) && is_contiguous(y) && is_contiguous(z)) {
    for (int i = 0; i < total; i++) {
      z_data[i] = fmodf(x_data[i], y_data[i]);
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    iterator_state_t it;
    init_binary_iterator(&it, x, y, z);

    int indices[MAX_STACK_DIMS] = {0};
    for (int i = 0; i < total; i++) {
      int idx_x = it.offset_x;
      int idx_y = it.offset_y;
      int idx_z = it.offset_z;
      for (int d = 0; d < it.ndim; d++) {
        idx_x += indices[d] * it.strides_x[d];
        idx_y += indices[d] * it.strides_y[d];
        idx_z += indices[d] * it.strides_z[d];
      }
      z_data[idx_z] = fmodf(x_data[idx_x], y_data[idx_y]);

      for (int d = it.ndim - 1; d >= 0; d--) {
        if (++indices[d] < it.shape[d]) break;
        indices[d] = 0;
      }
    }
  } else {
    /* Fallback for high dimensional arrays */
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx_x = x->offset;
      int idx_y = y->offset;
      int idx_z = z->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx_x += indices[d] * x->strides[d];
        idx_y += indices[d] * y->strides[d];
        idx_z += indices[d] * z->strides[d];
      }
      z_data[idx_z] = fmodf(x_data[idx_x], y_data[idx_y]);

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }
}

/* =========================== Unary Operations =========================== */

/* Generic unary operation for float32 */
#define UNARY_OP_F32(name, OP)                                                 \
  void nx_cblas_##name##_f32(const strided_array_t *x, strided_array_t *z) {   \
    float *x_data = (float *)x->data;                                          \
    float *z_data = (float *)z->data;                                          \
    int total = total_elements(x);                                             \
                                                                               \
    /* Fast path: contiguous arrays */                                         \
    if (is_contiguous(x) && is_contiguous(z)) {                                \
      /* Vectorized loop */                                                    \
      if (total > PARALLEL_THRESHOLD * 4) {                                    \
        /* Parallel execution for very large arrays */                         \
        _Pragma(                                                               \
            "omp parallel for simd aligned(x_data, z_data: 32)") for (int i =  \
                                                                          0;   \
                                                                      i <      \
                                                                      total;   \
                                                                      i++) {   \
          z_data[i] = OP(x_data[i]);                                           \
        }                                                                      \
      } else {                                                                 \
        /* SIMD only for medium arrays */                                      \
        _Pragma("omp simd aligned(x_data, z_data: 32)") for (int i = 0;        \
                                                             i < total; i++) { \
          z_data[i] = OP(x_data[i]);                                           \
        }                                                                      \
      }                                                                        \
    } /* strided path */                                                       \
    else if (x->ndim <= MAX_STACK_DIMS) {                                      \
      iterator_state_t it;                                                     \
      init_unary_iterator(&it, x, z);                                          \
                                                                               \
      /* Generate nested loops based on dimensionality */                      \
      switch (it.ndim) {                                                       \
        case 1: {                                                              \
          for (int i0 = 0; i0 < it.shape[0]; i0++) {                           \
            int idx_x = it.offset_x + i0 * it.strides_x[0];                    \
            int idx_z = it.offset_z + i0 * it.strides_z[0];                    \
            z_data[idx_z] = OP(x_data[idx_x]);                                 \
          }                                                                    \
          break;                                                               \
        }                                                                      \
        case 2: {                                                              \
          for (int i0 = 0; i0 < it.shape[0]; i0++) {                           \
            int base_x = it.offset_x + i0 * it.strides_x[0];                   \
            int base_z = it.offset_z + i0 * it.strides_z[0];                   \
            for (int i1 = 0; i1 < it.shape[1]; i1++) {                         \
              int idx_x = base_x + i1 * it.strides_x[1];                       \
              int idx_z = base_z + i1 * it.strides_z[1];                       \
              z_data[idx_z] = OP(x_data[idx_x]);                               \
            }                                                                  \
          }                                                                    \
          break;                                                               \
        }                                                                      \
        default: {                                                             \
          /* General case for higher dimensions */                             \
          int indices[MAX_STACK_DIMS] = {0};                                   \
          for (int i = 0; i < total; i++) {                                    \
            /* Compute strided indices */                                      \
            int idx_x = it.offset_x;                                           \
            int idx_z = it.offset_z;                                           \
            for (int d = 0; d < it.ndim; d++) {                                \
              idx_x += indices[d] * it.strides_x[d];                           \
              idx_z += indices[d] * it.strides_z[d];                           \
            }                                                                  \
            z_data[idx_z] = OP(x_data[idx_x]);                                 \
                                                                               \
            /* Increment indices */                                            \
            for (int d = it.ndim - 1; d >= 0; d--) {                           \
              if (++indices[d] < it.shape[d]) break;                           \
              indices[d] = 0;                                                  \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    } /* Fallback for very high dimensional arrays */                          \
    else {                                                                     \
      int *indices = (int *)calloc(x->ndim, sizeof(int));                      \
      for (int i = 0; i < total; i++) {                                        \
        int idx_x = x->offset;                                                 \
        int idx_z = z->offset;                                                 \
        for (int d = 0; d < x->ndim; d++) {                                    \
          idx_x += indices[d] * x->strides[d];                                 \
          idx_z += indices[d] * z->strides[d];                                 \
        }                                                                      \
        z_data[idx_z] = OP(x_data[idx_x]);                                     \
                                                                               \
        for (int d = x->ndim - 1; d >= 0; d--) {                               \
          if (++indices[d] < x->shape[d]) break;                               \
          indices[d] = 0;                                                      \
        }                                                                      \
      }                                                                        \
      free(indices);                                                           \
    }                                                                          \
  }

/* Define unary operations */
/* Special handling for neg - uses BLAS */
void nx_cblas_neg_f32(const strided_array_t *x, strided_array_t *z) {
  float *x_data = (float *)x->data;
  float *z_data = (float *)z->data;
  int total = total_elements(x);

  if (is_contiguous(x) && is_contiguous(z)) {
    if (total > PARALLEL_THRESHOLD) {
      /* Use BLAS for large arrays */
      cblas_scopy(total, x_data, 1, z_data, 1);
      cblas_sscal(total, -1.0f, z_data, 1);
    } else if (total > PARALLEL_THRESHOLD * 4) {
/* Parallel execution for very large arrays */
#pragma omp parallel for simd aligned(x_data, z_data : 32)
      for (int i = 0; i < total; i++) {
        z_data[i] = -x_data[i];
      }
    } else {
/* SIMD only for medium arrays */
#pragma omp simd aligned(x_data, z_data : 32)
      for (int i = 0; i < total; i++) {
        z_data[i] = -x_data[i];
      }
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    iterator_state_t it;
    init_unary_iterator(&it, x, z);

    switch (it.ndim) {
      case 1: {
        for (int i0 = 0; i0 < it.shape[0]; i0++) {
          int idx_x = it.offset_x + i0 * it.strides_x[0];
          int idx_z = it.offset_z + i0 * it.strides_z[0];
          z_data[idx_z] = -x_data[idx_x];
        }
        break;
      }
      case 2: {
        for (int i0 = 0; i0 < it.shape[0]; i0++) {
          int base_x = it.offset_x + i0 * it.strides_x[0];
          int base_z = it.offset_z + i0 * it.strides_z[0];
          for (int i1 = 0; i1 < it.shape[1]; i1++) {
            int idx_x = base_x + i1 * it.strides_x[1];
            int idx_z = base_z + i1 * it.strides_z[1];
            z_data[idx_z] = -x_data[idx_x];
          }
        }
        break;
      }
      default: {
        int indices[MAX_STACK_DIMS] = {0};
        for (int i = 0; i < total; i++) {
          int idx_x = it.offset_x;
          int idx_z = it.offset_z;
          for (int d = 0; d < it.ndim; d++) {
            idx_x += indices[d] * it.strides_x[d];
            idx_z += indices[d] * it.strides_z[d];
          }
          z_data[idx_z] = -x_data[idx_x];

          for (int d = it.ndim - 1; d >= 0; d--) {
            if (++indices[d] < it.shape[d]) break;
            indices[d] = 0;
          }
        }
      }
    }
  } else {
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx_x = x->offset;
      int idx_z = z->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx_x += indices[d] * x->strides[d];
        idx_z += indices[d] * z->strides[d];
      }
      z_data[idx_z] = -x_data[idx_x];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }
}

UNARY_OP_F32(sqrt, sqrtf)
UNARY_OP_F32(sin, sinf)
UNARY_OP_F32(exp2, exp2f)
UNARY_OP_F32(log2, log2f)
UNARY_OP_F32(recip, RECIP)

/* =========================== Copy =========================== */

void nx_cblas_copy_f32(const strided_array_t *x, strided_array_t *z) {
  int total = total_elements(x);
  float *x_data = (float *)x->data;
  float *z_data = (float *)z->data;

  if (is_contiguous(x) && is_contiguous(z)) {
    cblas_scopy(total, x_data, 1, z_data, 1);
  } else if (x->ndim <= MAX_STACK_DIMS) {
    iterator_state_t it;
    init_unary_iterator(&it, x, z);

    /* Special case for common patterns */
    if (it.ndim == 2 && it.strides_x[1] == 1 && it.strides_z[1] == 1) {
      /* Row-wise contiguous copy */
      for (int i = 0; i < it.shape[0]; i++) {
        cblas_scopy(it.shape[1], x_data + it.offset_x + i * it.strides_x[0], 1,
                    z_data + it.offset_z + i * it.strides_z[0], 1);
      }
    } else {
      /* General strided copy */
      int indices[MAX_STACK_DIMS] = {0};
      for (int i = 0; i < total; i++) {
        int idx_x = it.offset_x;
        int idx_z = it.offset_z;
        for (int d = 0; d < it.ndim; d++) {
          idx_x += indices[d] * it.strides_x[d];
          idx_z += indices[d] * it.strides_z[d];
        }
        z_data[idx_z] = x_data[idx_x];

        for (int d = it.ndim - 1; d >= 0; d--) {
          if (++indices[d] < it.shape[d]) break;
          indices[d] = 0;
        }
      }
    }
  } else {
    /* Fallback for high dimensional arrays */
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx_x = x->offset;
      int idx_z = z->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx_x += indices[d] * x->strides[d];
        idx_z += indices[d] * z->strides[d];
      }
      z_data[idx_z] = x_data[idx_x];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }
}

/* =========================== Matrix Multiplication ===========================
 */

static int is_matrix_mul(const strided_array_t *x, const strided_array_t *y,
                         const strided_array_t *z) {
  if (x->ndim < 2 || y->ndim < 2) return 0;
  if (x->shape[x->ndim - 1] != y->shape[y->ndim - 2]) return 0;
  if (z->shape[z->ndim - 2] != x->shape[x->ndim - 2]) return 0;
  if (z->shape[z->ndim - 1] != y->shape[y->ndim - 1]) return 0;

  /* Check if the last two dimensions are contiguous */
  if (x->strides[x->ndim - 1] != 1) return 0;
  if (y->strides[y->ndim - 1] != 1) return 0;
  if (z->strides[z->ndim - 1] != 1) return 0;

  if (x->strides[x->ndim - 2] != x->shape[x->ndim - 1]) return 0;
  if (y->strides[y->ndim - 2] != y->shape[y->ndim - 1]) return 0;
  if (z->strides[z->ndim - 2] != z->shape[z->ndim - 1]) return 0;

  return 1;
}

/* Matrix multiplication dispatch removed - OCaml handles the logic */

/* =========================== Comparison Operations ===========================
 */

void nx_cblas_cmplt_f32(const strided_array_t *x, const strided_array_t *y,
                        strided_array_t *z) {
  int total = total_elements(x);
  float *x_data = (float *)x->data;
  float *y_data = (float *)y->data;
  uint8_t *z_data = (uint8_t *)z->data;

  if (is_contiguous(x) && is_contiguous(y) && is_contiguous(z)) {
#pragma omp simd aligned(x_data, y_data : 32)
    for (int i = 0; i < total; i++) {
      z_data[i] = (x_data[i] < y_data[i]) ? 1 : 0;
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    iterator_state_t it;
    init_binary_iterator(&it, x, y, z);

    int indices[MAX_STACK_DIMS] = {0};
    for (int i = 0; i < total; i++) {
      int idx_x = it.offset_x;
      int idx_y = it.offset_y;
      int idx_z = it.offset_z;
      for (int d = 0; d < it.ndim; d++) {
        idx_x += indices[d] * it.strides_x[d];
        idx_y += indices[d] * it.strides_y[d];
        idx_z += indices[d] * it.strides_z[d];
      }
      z_data[idx_z] = (x_data[idx_x] < y_data[idx_y]) ? 1 : 0;

      for (int d = it.ndim - 1; d >= 0; d--) {
        if (++indices[d] < it.shape[d]) break;
        indices[d] = 0;
      }
    }
  } else {
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx_x = x->offset;
      int idx_y = y->offset;
      int idx_z = z->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx_x += indices[d] * x->strides[d];
        idx_y += indices[d] * y->strides[d];
        idx_z += indices[d] * z->strides[d];
      }
      z_data[idx_z] = (x_data[idx_x] < y_data[idx_y]) ? 1 : 0;

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }
}

void nx_cblas_cmpne_f32(const strided_array_t *x, const strided_array_t *y,
                        strided_array_t *z) {
  int total = total_elements(x);
  float *x_data = (float *)x->data;
  float *y_data = (float *)y->data;
  uint8_t *z_data = (uint8_t *)z->data;

  if (is_contiguous(x) && is_contiguous(y) && is_contiguous(z)) {
#pragma omp simd aligned(x_data, y_data : 32)
    for (int i = 0; i < total; i++) {
      z_data[i] = (x_data[i] != y_data[i]) ? 1 : 0;
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    iterator_state_t it;
    init_binary_iterator(&it, x, y, z);

    int indices[MAX_STACK_DIMS] = {0};
    for (int i = 0; i < total; i++) {
      int idx_x = it.offset_x;
      int idx_y = it.offset_y;
      int idx_z = it.offset_z;
      for (int d = 0; d < it.ndim; d++) {
        idx_x += indices[d] * it.strides_x[d];
        idx_y += indices[d] * it.strides_y[d];
        idx_z += indices[d] * it.strides_z[d];
      }
      z_data[idx_z] = (x_data[idx_x] != y_data[idx_y]) ? 1 : 0;

      for (int d = it.ndim - 1; d >= 0; d--) {
        if (++indices[d] < it.shape[d]) break;
        indices[d] = 0;
      }
    }
  } else {
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx_x = x->offset;
      int idx_y = y->offset;
      int idx_z = z->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx_x += indices[d] * x->strides[d];
        idx_y += indices[d] * y->strides[d];
        idx_z += indices[d] * z->strides[d];
      }
      z_data[idx_z] = (x_data[idx_x] != y_data[idx_y]) ? 1 : 0;

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }
}

/* =========================== Reduction Operations ===========================
 */

void nx_cblas_reduce_sum_f32(const strided_array_t *x, void *result) {
  int total = total_elements(x);
  float *x_data = (float *)x->data;
  float sum = 0.0f;

  if (is_contiguous(x)) {
    /* Use BLAS for large arrays */
    if (total > PARALLEL_THRESHOLD) {
      sum = cblas_sasum(total, x_data, 1);
    } else {
/* Vectorized sum for smaller arrays */
#pragma omp simd reduction(+ : sum)
      for (int i = 0; i < total; i++) {
        sum += x_data[i];
      }
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    /* strided sum */
    int indices[MAX_STACK_DIMS] = {0};
    for (int i = 0; i < total; i++) {
      int idx = x->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx += indices[d] * x->strides[d];
      }
      sum += x_data[idx];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
  } else {
    /* Fallback */
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx = x->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx += indices[d] * x->strides[d];
      }
      sum += x_data[idx];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }

  *((float *)result) = sum;
}

void nx_cblas_reduce_max_f32(const strided_array_t *x, void *result) {
  int total = total_elements(x);
  float *x_data = (float *)x->data;

  if (total == 0) {
    *((float *)result) = -INFINITY;
    return;
  }

  float max_val = x_data[x->offset];

  if (is_contiguous(x)) {
    if (total > PARALLEL_THRESHOLD) {
      /* Use BLAS isamax for large arrays */
      int max_idx = cblas_isamax(total, x_data, 1);
      max_val = x_data[max_idx];
    } else {
/* Vectorized max for smaller arrays */
#pragma omp simd reduction(max : max_val)
      for (int i = 0; i < total; i++) {
        if (x_data[i] > max_val) max_val = x_data[i];
      }
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    /* strided max */
    int indices[MAX_STACK_DIMS] = {0};
    for (int i = 0; i < total; i++) {
      int idx = x->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx += indices[d] * x->strides[d];
      }
      if (x_data[idx] > max_val) max_val = x_data[idx];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
  } else {
    /* Fallback */
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx = x->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx += indices[d] * x->strides[d];
      }
      if (x_data[idx] > max_val) max_val = x_data[idx];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }

  *((float *)result) = max_val;
}

void nx_cblas_reduce_min_f32(const strided_array_t *x, void *result) {
  int total = total_elements(x);
  float *x_data = (float *)x->data;

  if (total == 0) {
    *((float *)result) = INFINITY;
    return;
  }

  float min_val = x_data[x->offset];

  if (is_contiguous(x)) {
    /* Vectorized min */
    if (total > PARALLEL_THRESHOLD * 4) {
#pragma omp parallel for simd reduction(min : min_val)
      for (int i = 0; i < total; i++) {
        if (x_data[i] < min_val) min_val = x_data[i];
      }
    } else {
#pragma omp simd reduction(min : min_val)
      for (int i = 0; i < total; i++) {
        if (x_data[i] < min_val) min_val = x_data[i];
      }
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    /* strided min */
    int indices[MAX_STACK_DIMS] = {0};
    for (int i = 0; i < total; i++) {
      int idx = x->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx += indices[d] * x->strides[d];
      }
      if (x_data[idx] < min_val) min_val = x_data[idx];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
  } else {
    /* Fallback */
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx = x->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx += indices[d] * x->strides[d];
      }
      if (x_data[idx] < min_val) min_val = x_data[idx];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }

  *((float *)result) = min_val;
}

/* =========================== Float64 Operations =========================== */

/* Generic binary operation for float64 */
#define BINARY_OP_F64(name, OP)                                                       \
  void nx_cblas_##name##_f64(const strided_array_t *x,                                \
                             const strided_array_t *y, strided_array_t *z) {          \
    double *x_data = (double *)x->data;                                               \
    double *y_data = (double *)y->data;                                               \
    double *z_data = (double *)z->data;                                               \
    int total = total_elements(x);                                                    \
                                                                                      \
    /* Fast path: contiguous arrays */                                                \
    if (is_contiguous(x) && is_contiguous(y) && is_contiguous(z)) {                   \
      /* Vectorized loop */                                                           \
      if (total > PARALLEL_THRESHOLD * 4) {                                           \
        /* Parallel execution for very large arrays */                                \
        _Pragma(                                                                      \
            "omp parallel for simd aligned(x_data, y_data, z_data: 32)") for (int i = \
                                                                                  0;  \
                                                                              i <     \
                                                                              total;  \
                                                                              i++) {  \
          z_data[i] = OP(x_data[i], y_data[i]);                                       \
        }                                                                             \
      } else {                                                                        \
        /* SIMD only for medium arrays */                                             \
        _Pragma(                                                                      \
            "omp simd aligned(x_data, y_data, z_data: 32)") for (int i = 0;           \
                                                                 i < total;           \
                                                                 i++) {               \
          z_data[i] = OP(x_data[i], y_data[i]);                                       \
        }                                                                             \
      }                                                                               \
    } /* strided path */                                                              \
    else if (x->ndim <= MAX_STACK_DIMS) {                                             \
      iterator_state_t it;                                                            \
      init_binary_iterator(&it, x, y, z);                                             \
                                                                                      \
      /* Generate nested loops based on dimensionality */                             \
      switch (it.ndim) {                                                              \
        case 1: {                                                                     \
          for (int i0 = 0; i0 < it.shape[0]; i0++) {                                  \
            int idx_x = it.offset_x + i0 * it.strides_x[0];                           \
            int idx_y = it.offset_y + i0 * it.strides_y[0];                           \
            int idx_z = it.offset_z + i0 * it.strides_z[0];                           \
            z_data[idx_z] = OP(x_data[idx_x], y_data[idx_y]);                         \
          }                                                                           \
          break;                                                                      \
        }                                                                             \
        case 2: {                                                                     \
          for (int i0 = 0; i0 < it.shape[0]; i0++) {                                  \
            int base_x = it.offset_x + i0 * it.strides_x[0];                          \
            int base_y = it.offset_y + i0 * it.strides_y[0];                          \
            int base_z = it.offset_z + i0 * it.strides_z[0];                          \
            for (int i1 = 0; i1 < it.shape[1]; i1++) {                                \
              int idx_x = base_x + i1 * it.strides_x[1];                              \
              int idx_y = base_y + i1 * it.strides_y[1];                              \
              int idx_z = base_z + i1 * it.strides_z[1];                              \
              z_data[idx_z] = OP(x_data[idx_x], y_data[idx_y]);                       \
            }                                                                         \
          }                                                                           \
          break;                                                                      \
        }                                                                             \
        default: {                                                                    \
          /* General case for higher dimensions */                                    \
          int indices[MAX_STACK_DIMS] = {0};                                          \
          for (int i = 0; i < total; i++) {                                           \
            /* Compute strided indices */                                             \
            int idx_x = it.offset_x;                                                  \
            int idx_y = it.offset_y;                                                  \
            int idx_z = it.offset_z;                                                  \
            for (int d = 0; d < it.ndim; d++) {                                       \
              idx_x += indices[d] * it.strides_x[d];                                  \
              idx_y += indices[d] * it.strides_y[d];                                  \
              idx_z += indices[d] * it.strides_z[d];                                  \
            }                                                                         \
            z_data[idx_z] = OP(x_data[idx_x], y_data[idx_y]);                         \
                                                                                      \
            /* Increment indices */                                                   \
            for (int d = it.ndim - 1; d >= 0; d--) {                                  \
              if (++indices[d] < it.shape[d]) break;                                  \
              indices[d] = 0;                                                         \
            }                                                                         \
          }                                                                           \
        }                                                                             \
      }                                                                               \
    } /* Fallback for very high dimensional arrays */                                 \
    else {                                                                            \
      int *indices = (int *)calloc(x->ndim, sizeof(int));                             \
      for (int i = 0; i < total; i++) {                                               \
        int idx_x = x->offset;                                                        \
        int idx_y = y->offset;                                                        \
        int idx_z = z->offset;                                                        \
        for (int d = 0; d < x->ndim; d++) {                                           \
          idx_x += indices[d] * x->strides[d];                                        \
          idx_y += indices[d] * y->strides[d];                                        \
          idx_z += indices[d] * z->strides[d];                                        \
        }                                                                             \
        z_data[idx_z] = OP(x_data[idx_x], y_data[idx_y]);                             \
                                                                                      \
        for (int d = x->ndim - 1; d >= 0; d--) {                                      \
          if (++indices[d] < x->shape[d]) break;                                      \
          indices[d] = 0;                                                             \
        }                                                                             \
      }                                                                               \
      free(indices);                                                                  \
    }                                                                                 \
  }

/* Double precision operation macros */
#define ADD_F64(x, y) ((x) + (y))
#define SUB_F64(x, y) ((x) - (y))
#define MUL_F64(x, y) ((x) * (y))
#define DIV_F64(x, y) ((x) / (y))
#define MAX_F64(x, y) ((x) > (y) ? (x) : (y))
#define NEG_F64(x) (-(x))
#define RECIP_F64(x) (1.0 / (x))

/* Define binary operations for float64 */
/* Special handling for add - uses BLAS */
void nx_cblas_add_f64(const strided_array_t *x, const strided_array_t *y,
                      strided_array_t *z) {
  double *x_data = (double *)x->data;
  double *y_data = (double *)y->data;
  double *z_data = (double *)z->data;
  int total = total_elements(x);

  if (is_contiguous(x) && is_contiguous(y) && is_contiguous(z)) {
    if (total > PARALLEL_THRESHOLD) {
      /* Use BLAS for large arrays */
      cblas_dcopy(total, y_data, 1, z_data, 1);
      cblas_daxpy(total, 1.0, x_data, 1, z_data, 1);
    } else if (total > PARALLEL_THRESHOLD * 4) {
/* Parallel execution for very large arrays */
#pragma omp parallel for simd aligned(x_data, y_data, z_data : 32)
      for (int i = 0; i < total; i++) {
        z_data[i] = x_data[i] + y_data[i];
      }
    } else {
/* SIMD only for medium arrays */
#pragma omp simd aligned(x_data, y_data, z_data : 32)
      for (int i = 0; i < total; i++) {
        z_data[i] = x_data[i] + y_data[i];
      }
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    iterator_state_t it;
    init_binary_iterator(&it, x, y, z);

    switch (it.ndim) {
      case 1: {
        for (int i0 = 0; i0 < it.shape[0]; i0++) {
          int idx_x = it.offset_x + i0 * it.strides_x[0];
          int idx_y = it.offset_y + i0 * it.strides_y[0];
          int idx_z = it.offset_z + i0 * it.strides_z[0];
          z_data[idx_z] = x_data[idx_x] + y_data[idx_y];
        }
        break;
      }
      case 2: {
        for (int i0 = 0; i0 < it.shape[0]; i0++) {
          int base_x = it.offset_x + i0 * it.strides_x[0];
          int base_y = it.offset_y + i0 * it.strides_y[0];
          int base_z = it.offset_z + i0 * it.strides_z[0];
          for (int i1 = 0; i1 < it.shape[1]; i1++) {
            int idx_x = base_x + i1 * it.strides_x[1];
            int idx_y = base_y + i1 * it.strides_y[1];
            int idx_z = base_z + i1 * it.strides_z[1];
            z_data[idx_z] = x_data[idx_x] + y_data[idx_y];
          }
        }
        break;
      }
      default: {
        int indices[MAX_STACK_DIMS] = {0};
        for (int i = 0; i < total; i++) {
          int idx_x = it.offset_x;
          int idx_y = it.offset_y;
          int idx_z = it.offset_z;
          for (int d = 0; d < it.ndim; d++) {
            idx_x += indices[d] * it.strides_x[d];
            idx_y += indices[d] * it.strides_y[d];
            idx_z += indices[d] * it.strides_z[d];
          }
          z_data[idx_z] = x_data[idx_x] + y_data[idx_y];

          for (int d = it.ndim - 1; d >= 0; d--) {
            if (++indices[d] < it.shape[d]) break;
            indices[d] = 0;
          }
        }
      }
    }
  } else {
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx_x = x->offset;
      int idx_y = y->offset;
      int idx_z = z->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx_x += indices[d] * x->strides[d];
        idx_y += indices[d] * y->strides[d];
        idx_z += indices[d] * z->strides[d];
      }
      z_data[idx_z] = x_data[idx_x] + y_data[idx_y];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }
}

/* Similar for sub */
void nx_cblas_sub_f64(const strided_array_t *x, const strided_array_t *y,
                      strided_array_t *z) {
  double *x_data = (double *)x->data;
  double *y_data = (double *)y->data;
  double *z_data = (double *)z->data;
  int total = total_elements(x);

  if (is_contiguous(x) && is_contiguous(y) && is_contiguous(z)) {
    if (total > PARALLEL_THRESHOLD) {
      /* Use BLAS for large arrays */
      cblas_dcopy(total, x_data, 1, z_data, 1);
      cblas_daxpy(total, -1.0, y_data, 1, z_data, 1);
    } else if (total > PARALLEL_THRESHOLD * 4) {
/* Parallel execution for very large arrays */
#pragma omp parallel for simd aligned(x_data, y_data, z_data : 32)
      for (int i = 0; i < total; i++) {
        z_data[i] = x_data[i] - y_data[i];
      }
    } else {
/* SIMD only for medium arrays */
#pragma omp simd aligned(x_data, y_data, z_data : 32)
      for (int i = 0; i < total; i++) {
        z_data[i] = x_data[i] - y_data[i];
      }
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    iterator_state_t it;
    init_binary_iterator(&it, x, y, z);

    switch (it.ndim) {
      case 1: {
        for (int i0 = 0; i0 < it.shape[0]; i0++) {
          int idx_x = it.offset_x + i0 * it.strides_x[0];
          int idx_y = it.offset_y + i0 * it.strides_y[0];
          int idx_z = it.offset_z + i0 * it.strides_z[0];
          z_data[idx_z] = x_data[idx_x] - y_data[idx_y];
        }
        break;
      }
      case 2: {
        for (int i0 = 0; i0 < it.shape[0]; i0++) {
          int base_x = it.offset_x + i0 * it.strides_x[0];
          int base_y = it.offset_y + i0 * it.strides_y[0];
          int base_z = it.offset_z + i0 * it.strides_z[0];
          for (int i1 = 0; i1 < it.shape[1]; i1++) {
            int idx_x = base_x + i1 * it.strides_x[1];
            int idx_y = base_y + i1 * it.strides_y[1];
            int idx_z = base_z + i1 * it.strides_z[1];
            z_data[idx_z] = x_data[idx_x] - y_data[idx_y];
          }
        }
        break;
      }
      default: {
        int indices[MAX_STACK_DIMS] = {0};
        for (int i = 0; i < total; i++) {
          int idx_x = it.offset_x;
          int idx_y = it.offset_y;
          int idx_z = it.offset_z;
          for (int d = 0; d < it.ndim; d++) {
            idx_x += indices[d] * it.strides_x[d];
            idx_y += indices[d] * it.strides_y[d];
            idx_z += indices[d] * it.strides_z[d];
          }
          z_data[idx_z] = x_data[idx_x] - y_data[idx_y];

          for (int d = it.ndim - 1; d >= 0; d--) {
            if (++indices[d] < it.shape[d]) break;
            indices[d] = 0;
          }
        }
      }
    }
  } else {
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx_x = x->offset;
      int idx_y = y->offset;
      int idx_z = z->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx_x += indices[d] * x->strides[d];
        idx_y += indices[d] * y->strides[d];
        idx_z += indices[d] * z->strides[d];
      }
      z_data[idx_z] = x_data[idx_x] - y_data[idx_y];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }
}

BINARY_OP_F64(div, DIV_F64)
BINARY_OP_F64(max, MAX_F64)

/* matrix multiplication for float64 */
void nx_cblas_mul_f64(const strided_array_t *x, const strided_array_t *y,
                      strided_array_t *z) {
  if (is_matrix_mul(x, y, z)) {
    int batch_size = 1;
    for (int i = 0; i < x->ndim - 2; i++) {
      batch_size *= x->shape[i];
    }

    int m = x->shape[x->ndim - 2];
    int k = x->shape[x->ndim - 1];
    int n = y->shape[y->ndim - 1];

    double *a_data = (double *)x->data + x->offset;
    double *b_data = (double *)y->data + y->offset;
    double *c_data = (double *)z->data + z->offset;

    /* Calculate batch strides */
    int a_batch_stride = (x->ndim > 2) ? x->strides[x->ndim - 3] : 0;
    int b_batch_stride = (y->ndim > 2) ? y->strides[y->ndim - 3] : 0;
    int c_batch_stride = (z->ndim > 2) ? z->strides[z->ndim - 3] : 0;

    if (batch_size == 1) {
      /* Single matrix multiplication */
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0,
                  a_data, k, b_data, n, 0.0, c_data, n);
    } else {
/* Batched matrix multiplication */
#pragma omp parallel for if (batch_size > 4)
      for (int b = 0; b < batch_size; b++) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0,
                    a_data + b * a_batch_stride, k, b_data + b * b_batch_stride,
                    n, 0.0, c_data + b * c_batch_stride, n);
      }
    }
  } else {
    /* Element-wise multiplication - use the same logic as the macro */
    double *x_data = (double *)x->data;
    double *y_data = (double *)y->data;
    double *z_data = (double *)z->data;
    int total = total_elements(x);

    if (is_contiguous(x) && is_contiguous(y) && is_contiguous(z)) {
      if (total > PARALLEL_THRESHOLD * 4) {
#pragma omp parallel for simd aligned(x_data, y_data, z_data : 32)
        for (int i = 0; i < total; i++) {
          z_data[i] = x_data[i] * y_data[i];
        }
      } else {
#pragma omp simd aligned(x_data, y_data, z_data : 32)
        for (int i = 0; i < total; i++) {
          z_data[i] = x_data[i] * y_data[i];
        }
      }
    } else if (x->ndim <= MAX_STACK_DIMS) {
      iterator_state_t it;
      init_binary_iterator(&it, x, y, z);

      int indices[MAX_STACK_DIMS] = {0};
      for (int i = 0; i < total; i++) {
        int idx_x = it.offset_x;
        int idx_y = it.offset_y;
        int idx_z = it.offset_z;
        for (int d = 0; d < it.ndim; d++) {
          idx_x += indices[d] * it.strides_x[d];
          idx_y += indices[d] * it.strides_y[d];
          idx_z += indices[d] * it.strides_z[d];
        }
        z_data[idx_z] = x_data[idx_x] * y_data[idx_y];

        for (int d = it.ndim - 1; d >= 0; d--) {
          if (++indices[d] < it.shape[d]) break;
          indices[d] = 0;
        }
      }
    } else {
      int *indices = (int *)calloc(x->ndim, sizeof(int));
      for (int i = 0; i < total; i++) {
        int idx_x = x->offset;
        int idx_y = y->offset;
        int idx_z = z->offset;
        for (int d = 0; d < x->ndim; d++) {
          idx_x += indices[d] * x->strides[d];
          idx_y += indices[d] * y->strides[d];
          idx_z += indices[d] * z->strides[d];
        }
        z_data[idx_z] = x_data[idx_x] * y_data[idx_y];

        for (int d = x->ndim - 1; d >= 0; d--) {
          if (++indices[d] < x->shape[d]) break;
          indices[d] = 0;
        }
      }
      free(indices);
    }
  }
}

/* Special handling for pow and mod - no SIMD */
void nx_cblas_pow_f64(const strided_array_t *x, const strided_array_t *y,
                      strided_array_t *z) {
  double *x_data = (double *)x->data;
  double *y_data = (double *)y->data;
  double *z_data = (double *)z->data;
  int total = total_elements(x);

  if (is_contiguous(x) && is_contiguous(y) && is_contiguous(z)) {
    for (int i = 0; i < total; i++) {
      z_data[i] = pow(x_data[i], y_data[i]);
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    iterator_state_t it;
    init_binary_iterator(&it, x, y, z);

    int indices[MAX_STACK_DIMS] = {0};
    for (int i = 0; i < total; i++) {
      int idx_x = it.offset_x;
      int idx_y = it.offset_y;
      int idx_z = it.offset_z;
      for (int d = 0; d < it.ndim; d++) {
        idx_x += indices[d] * it.strides_x[d];
        idx_y += indices[d] * it.strides_y[d];
        idx_z += indices[d] * it.strides_z[d];
      }
      z_data[idx_z] = pow(x_data[idx_x], y_data[idx_y]);

      for (int d = it.ndim - 1; d >= 0; d--) {
        if (++indices[d] < it.shape[d]) break;
        indices[d] = 0;
      }
    }
  } else {
    /* Fallback for high dimensional arrays */
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx_x = x->offset;
      int idx_y = y->offset;
      int idx_z = z->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx_x += indices[d] * x->strides[d];
        idx_y += indices[d] * y->strides[d];
        idx_z += indices[d] * z->strides[d];
      }
      z_data[idx_z] = pow(x_data[idx_x], y_data[idx_y]);

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }
}

void nx_cblas_mod_f64(const strided_array_t *x, const strided_array_t *y,
                      strided_array_t *z) {
  double *x_data = (double *)x->data;
  double *y_data = (double *)y->data;
  double *z_data = (double *)z->data;
  int total = total_elements(x);

  if (is_contiguous(x) && is_contiguous(y) && is_contiguous(z)) {
    for (int i = 0; i < total; i++) {
      z_data[i] = fmod(x_data[i], y_data[i]);
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    iterator_state_t it;
    init_binary_iterator(&it, x, y, z);

    int indices[MAX_STACK_DIMS] = {0};
    for (int i = 0; i < total; i++) {
      int idx_x = it.offset_x;
      int idx_y = it.offset_y;
      int idx_z = it.offset_z;
      for (int d = 0; d < it.ndim; d++) {
        idx_x += indices[d] * it.strides_x[d];
        idx_y += indices[d] * it.strides_y[d];
        idx_z += indices[d] * it.strides_z[d];
      }
      z_data[idx_z] = fmod(x_data[idx_x], y_data[idx_y]);

      for (int d = it.ndim - 1; d >= 0; d--) {
        if (++indices[d] < it.shape[d]) break;
        indices[d] = 0;
      }
    }
  } else {
    /* Fallback for high dimensional arrays */
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx_x = x->offset;
      int idx_y = y->offset;
      int idx_z = z->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx_x += indices[d] * x->strides[d];
        idx_y += indices[d] * y->strides[d];
        idx_z += indices[d] * z->strides[d];
      }
      z_data[idx_z] = fmod(x_data[idx_x], y_data[idx_y]);

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }
}

/* Generic unary operation for float64 */
#define UNARY_OP_F64(name, OP)                                                 \
  void nx_cblas_##name##_f64(const strided_array_t *x, strided_array_t *z) {   \
    double *x_data = (double *)x->data;                                        \
    double *z_data = (double *)z->data;                                        \
    int total = total_elements(x);                                             \
                                                                               \
    /* Fast path: contiguous arrays */                                         \
    if (is_contiguous(x) && is_contiguous(z)) {                                \
      /* Vectorized loop */                                                    \
      if (total > PARALLEL_THRESHOLD * 4) {                                    \
        /* Parallel execution for very large arrays */                         \
        _Pragma(                                                               \
            "omp parallel for simd aligned(x_data, z_data: 32)") for (int i =  \
                                                                          0;   \
                                                                      i <      \
                                                                      total;   \
                                                                      i++) {   \
          z_data[i] = OP(x_data[i]);                                           \
        }                                                                      \
      } else {                                                                 \
        /* SIMD only for medium arrays */                                      \
        _Pragma("omp simd aligned(x_data, z_data: 32)") for (int i = 0;        \
                                                             i < total; i++) { \
          z_data[i] = OP(x_data[i]);                                           \
        }                                                                      \
      }                                                                        \
    } /* strided path */                                                       \
    else if (x->ndim <= MAX_STACK_DIMS) {                                      \
      iterator_state_t it;                                                     \
      init_unary_iterator(&it, x, z);                                          \
                                                                               \
      /* Generate nested loops based on dimensionality */                      \
      switch (it.ndim) {                                                       \
        case 1: {                                                              \
          for (int i0 = 0; i0 < it.shape[0]; i0++) {                           \
            int idx_x = it.offset_x + i0 * it.strides_x[0];                    \
            int idx_z = it.offset_z + i0 * it.strides_z[0];                    \
            z_data[idx_z] = OP(x_data[idx_x]);                                 \
          }                                                                    \
          break;                                                               \
        }                                                                      \
        case 2: {                                                              \
          for (int i0 = 0; i0 < it.shape[0]; i0++) {                           \
            int base_x = it.offset_x + i0 * it.strides_x[0];                   \
            int base_z = it.offset_z + i0 * it.strides_z[0];                   \
            for (int i1 = 0; i1 < it.shape[1]; i1++) {                         \
              int idx_x = base_x + i1 * it.strides_x[1];                       \
              int idx_z = base_z + i1 * it.strides_z[1];                       \
              z_data[idx_z] = OP(x_data[idx_x]);                               \
            }                                                                  \
          }                                                                    \
          break;                                                               \
        }                                                                      \
        default: {                                                             \
          /* General case for higher dimensions */                             \
          int indices[MAX_STACK_DIMS] = {0};                                   \
          for (int i = 0; i < total; i++) {                                    \
            /* Compute strided indices */                                      \
            int idx_x = it.offset_x;                                           \
            int idx_z = it.offset_z;                                           \
            for (int d = 0; d < it.ndim; d++) {                                \
              idx_x += indices[d] * it.strides_x[d];                           \
              idx_z += indices[d] * it.strides_z[d];                           \
            }                                                                  \
            z_data[idx_z] = OP(x_data[idx_x]);                                 \
                                                                               \
            /* Increment indices */                                            \
            for (int d = it.ndim - 1; d >= 0; d--) {                           \
              if (++indices[d] < it.shape[d]) break;                           \
              indices[d] = 0;                                                  \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    } /* Fallback for very high dimensional arrays */                          \
    else {                                                                     \
      int *indices = (int *)calloc(x->ndim, sizeof(int));                      \
      for (int i = 0; i < total; i++) {                                        \
        int idx_x = x->offset;                                                 \
        int idx_z = z->offset;                                                 \
        for (int d = 0; d < x->ndim; d++) {                                    \
          idx_x += indices[d] * x->strides[d];                                 \
          idx_z += indices[d] * z->strides[d];                                 \
        }                                                                      \
        z_data[idx_z] = OP(x_data[idx_x]);                                     \
                                                                               \
        for (int d = x->ndim - 1; d >= 0; d--) {                               \
          if (++indices[d] < x->shape[d]) break;                               \
          indices[d] = 0;                                                      \
        }                                                                      \
      }                                                                        \
      free(indices);                                                           \
    }                                                                          \
  }

/* Define unary operations */
/* Special handling for neg - uses BLAS */
void nx_cblas_neg_f64(const strided_array_t *x, strided_array_t *z) {
  double *x_data = (double *)x->data;
  double *z_data = (double *)z->data;
  int total = total_elements(x);

  if (is_contiguous(x) && is_contiguous(z)) {
    if (total > PARALLEL_THRESHOLD) {
      /* Use BLAS for large arrays */
      cblas_dcopy(total, x_data, 1, z_data, 1);
      cblas_dscal(total, -1.0, z_data, 1);
    } else if (total > PARALLEL_THRESHOLD * 4) {
/* Parallel execution for very large arrays */
#pragma omp parallel for simd aligned(x_data, z_data : 32)
      for (int i = 0; i < total; i++) {
        z_data[i] = -x_data[i];
      }
    } else {
/* SIMD only for medium arrays */
#pragma omp simd aligned(x_data, z_data : 32)
      for (int i = 0; i < total; i++) {
        z_data[i] = -x_data[i];
      }
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    iterator_state_t it;
    init_unary_iterator(&it, x, z);

    switch (it.ndim) {
      case 1: {
        for (int i0 = 0; i0 < it.shape[0]; i0++) {
          int idx_x = it.offset_x + i0 * it.strides_x[0];
          int idx_z = it.offset_z + i0 * it.strides_z[0];
          z_data[idx_z] = -x_data[idx_x];
        }
        break;
      }
      case 2: {
        for (int i0 = 0; i0 < it.shape[0]; i0++) {
          int base_x = it.offset_x + i0 * it.strides_x[0];
          int base_z = it.offset_z + i0 * it.strides_z[0];
          for (int i1 = 0; i1 < it.shape[1]; i1++) {
            int idx_x = base_x + i1 * it.strides_x[1];
            int idx_z = base_z + i1 * it.strides_z[1];
            z_data[idx_z] = -x_data[idx_x];
          }
        }
        break;
      }
      default: {
        int indices[MAX_STACK_DIMS] = {0};
        for (int i = 0; i < total; i++) {
          int idx_x = it.offset_x;
          int idx_z = it.offset_z;
          for (int d = 0; d < it.ndim; d++) {
            idx_x += indices[d] * it.strides_x[d];
            idx_z += indices[d] * it.strides_z[d];
          }
          z_data[idx_z] = -x_data[idx_x];

          for (int d = it.ndim - 1; d >= 0; d--) {
            if (++indices[d] < it.shape[d]) break;
            indices[d] = 0;
          }
        }
      }
    }
  } else {
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx_x = x->offset;
      int idx_z = z->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx_x += indices[d] * x->strides[d];
        idx_z += indices[d] * z->strides[d];
      }
      z_data[idx_z] = -x_data[idx_x];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }
}

UNARY_OP_F64(sqrt, sqrt)
UNARY_OP_F64(sin, sin)
UNARY_OP_F64(exp2, exp2)
UNARY_OP_F64(log2, log2)
UNARY_OP_F64(recip, RECIP_F64)

/* copy for float64 */
void nx_cblas_copy_f64(const strided_array_t *x, strided_array_t *z) {
  int total = total_elements(x);
  double *x_data = (double *)x->data;
  double *z_data = (double *)z->data;

  if (is_contiguous(x) && is_contiguous(z)) {
    cblas_dcopy(total, x_data, 1, z_data, 1);
  } else if (x->ndim <= MAX_STACK_DIMS) {
    iterator_state_t it;
    init_unary_iterator(&it, x, z);

    /* Special case for common patterns */
    if (it.ndim == 2 && it.strides_x[1] == 1 && it.strides_z[1] == 1) {
      /* Row-wise contiguous copy */
      for (int i = 0; i < it.shape[0]; i++) {
        cblas_dcopy(it.shape[1], x_data + it.offset_x + i * it.strides_x[0], 1,
                    z_data + it.offset_z + i * it.strides_z[0], 1);
      }
    } else {
      /* General strided copy */
      int indices[MAX_STACK_DIMS] = {0};
      for (int i = 0; i < total; i++) {
        int idx_x = it.offset_x;
        int idx_z = it.offset_z;
        for (int d = 0; d < it.ndim; d++) {
          idx_x += indices[d] * it.strides_x[d];
          idx_z += indices[d] * it.strides_z[d];
        }
        z_data[idx_z] = x_data[idx_x];

        for (int d = it.ndim - 1; d >= 0; d--) {
          if (++indices[d] < it.shape[d]) break;
          indices[d] = 0;
        }
      }
    }
  } else {
    /* Fallback for high dimensional arrays */
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx_x = x->offset;
      int idx_z = z->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx_x += indices[d] * x->strides[d];
        idx_z += indices[d] * z->strides[d];
      }
      z_data[idx_z] = x_data[idx_x];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }
}

/* Comparison operations for float64 */
void nx_cblas_cmplt_f64(const strided_array_t *x, const strided_array_t *y,
                        strided_array_t *z) {
  int total = total_elements(x);
  double *x_data = (double *)x->data;
  double *y_data = (double *)y->data;
  uint8_t *z_data = (uint8_t *)z->data;

  if (is_contiguous(x) && is_contiguous(y) && is_contiguous(z)) {
#pragma omp simd aligned(x_data, y_data : 32)
    for (int i = 0; i < total; i++) {
      z_data[i] = (x_data[i] < y_data[i]) ? 1 : 0;
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    iterator_state_t it;
    init_binary_iterator(&it, x, y, z);

    int indices[MAX_STACK_DIMS] = {0};
    for (int i = 0; i < total; i++) {
      int idx_x = it.offset_x;
      int idx_y = it.offset_y;
      int idx_z = it.offset_z;
      for (int d = 0; d < it.ndim; d++) {
        idx_x += indices[d] * it.strides_x[d];
        idx_y += indices[d] * it.strides_y[d];
        idx_z += indices[d] * it.strides_z[d];
      }
      z_data[idx_z] = (x_data[idx_x] < y_data[idx_y]) ? 1 : 0;

      for (int d = it.ndim - 1; d >= 0; d--) {
        if (++indices[d] < it.shape[d]) break;
        indices[d] = 0;
      }
    }
  } else {
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx_x = x->offset;
      int idx_y = y->offset;
      int idx_z = z->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx_x += indices[d] * x->strides[d];
        idx_y += indices[d] * y->strides[d];
        idx_z += indices[d] * z->strides[d];
      }
      z_data[idx_z] = (x_data[idx_x] < y_data[idx_y]) ? 1 : 0;

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }
}

void nx_cblas_cmpne_f64(const strided_array_t *x, const strided_array_t *y,
                        strided_array_t *z) {
  int total = total_elements(x);
  double *x_data = (double *)x->data;
  double *y_data = (double *)y->data;
  uint8_t *z_data = (uint8_t *)z->data;

  if (is_contiguous(x) && is_contiguous(y) && is_contiguous(z)) {
#pragma omp simd aligned(x_data, y_data : 32)
    for (int i = 0; i < total; i++) {
      z_data[i] = (x_data[i] != y_data[i]) ? 1 : 0;
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    iterator_state_t it;
    init_binary_iterator(&it, x, y, z);

    int indices[MAX_STACK_DIMS] = {0};
    for (int i = 0; i < total; i++) {
      int idx_x = it.offset_x;
      int idx_y = it.offset_y;
      int idx_z = it.offset_z;
      for (int d = 0; d < it.ndim; d++) {
        idx_x += indices[d] * it.strides_x[d];
        idx_y += indices[d] * it.strides_y[d];
        idx_z += indices[d] * it.strides_z[d];
      }
      z_data[idx_z] = (x_data[idx_x] != y_data[idx_y]) ? 1 : 0;

      for (int d = it.ndim - 1; d >= 0; d--) {
        if (++indices[d] < it.shape[d]) break;
        indices[d] = 0;
      }
    }
  } else {
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx_x = x->offset;
      int idx_y = y->offset;
      int idx_z = z->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx_x += indices[d] * x->strides[d];
        idx_y += indices[d] * y->strides[d];
        idx_z += indices[d] * z->strides[d];
      }
      z_data[idx_z] = (x_data[idx_x] != y_data[idx_y]) ? 1 : 0;

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }
}

/* Reduction operations for float64 */
void nx_cblas_reduce_sum_f64(const strided_array_t *x, void *result) {
  int total = total_elements(x);
  double *x_data = (double *)x->data;
  double sum = 0.0;

  if (is_contiguous(x)) {
    /* Use BLAS for large arrays */
    if (total > PARALLEL_THRESHOLD) {
      sum = cblas_dasum(total, x_data, 1);
    } else {
/* Vectorized sum for smaller arrays */
#pragma omp simd reduction(+ : sum)
      for (int i = 0; i < total; i++) {
        sum += x_data[i];
      }
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    /* strided sum */
    int indices[MAX_STACK_DIMS] = {0};
    for (int i = 0; i < total; i++) {
      int idx = x->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx += indices[d] * x->strides[d];
      }
      sum += x_data[idx];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
  } else {
    /* Fallback */
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx = x->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx += indices[d] * x->strides[d];
      }
      sum += x_data[idx];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }

  *((double *)result) = sum;
}

void nx_cblas_reduce_max_f64(const strided_array_t *x, void *result) {
  int total = total_elements(x);
  double *x_data = (double *)x->data;

  if (total == 0) {
    *((double *)result) = -INFINITY;
    return;
  }

  double max_val = x_data[x->offset];

  if (is_contiguous(x)) {
    if (total > PARALLEL_THRESHOLD) {
      /* Use BLAS idamax for large arrays */
      int max_idx = cblas_idamax(total, x_data, 1);
      max_val = x_data[max_idx];
    } else {
/* Vectorized max for smaller arrays */
#pragma omp simd reduction(max : max_val)
      for (int i = 0; i < total; i++) {
        if (x_data[i] > max_val) max_val = x_data[i];
      }
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    /* strided max */
    int indices[MAX_STACK_DIMS] = {0};
    for (int i = 0; i < total; i++) {
      int idx = x->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx += indices[d] * x->strides[d];
      }
      if (x_data[idx] > max_val) max_val = x_data[idx];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
  } else {
    /* Fallback */
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx = x->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx += indices[d] * x->strides[d];
      }
      if (x_data[idx] > max_val) max_val = x_data[idx];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }

  *((double *)result) = max_val;
}

void nx_cblas_reduce_min_f64(const strided_array_t *x, void *result) {
  int total = total_elements(x);
  double *x_data = (double *)x->data;

  if (total == 0) {
    *((double *)result) = INFINITY;
    return;
  }

  double min_val = x_data[x->offset];

  if (is_contiguous(x)) {
    /* Vectorized min */
    if (total > PARALLEL_THRESHOLD * 4) {
#pragma omp parallel for simd reduction(min : min_val)
      for (int i = 0; i < total; i++) {
        if (x_data[i] < min_val) min_val = x_data[i];
      }
    } else {
#pragma omp simd reduction(min : min_val)
      for (int i = 0; i < total; i++) {
        if (x_data[i] < min_val) min_val = x_data[i];
      }
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    /* strided min */
    int indices[MAX_STACK_DIMS] = {0};
    for (int i = 0; i < total; i++) {
      int idx = x->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx += indices[d] * x->strides[d];
      }
      if (x_data[idx] < min_val) min_val = x_data[idx];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
  } else {
    /* Fallback */
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx = x->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx += indices[d] * x->strides[d];
      }
      if (x_data[idx] < min_val) min_val = x_data[idx];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }

  *((double *)result) = min_val;
}

/* =========================== Integer Operations =========================== */

/* Integer division for int32 */
void nx_cblas_idiv_i32(const strided_array_t *x, const strided_array_t *y,
                       strided_array_t *z) {
  int32_t *x_data = (int32_t *)x->data;
  int32_t *y_data = (int32_t *)y->data;
  int32_t *z_data = (int32_t *)z->data;
  int total = total_elements(x);

  if (is_contiguous(x) && is_contiguous(y) && is_contiguous(z)) {
    if (total > PARALLEL_THRESHOLD * 4) {
#pragma omp parallel for simd
      for (int i = 0; i < total; i++) {
        z_data[i] = x_data[i] / y_data[i];
      }
    } else {
#pragma omp simd
      for (int i = 0; i < total; i++) {
        z_data[i] = x_data[i] / y_data[i];
      }
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    iterator_state_t it;
    init_binary_iterator(&it, x, y, z);

    int indices[MAX_STACK_DIMS] = {0};
    for (int i = 0; i < total; i++) {
      int idx_x = it.offset_x;
      int idx_y = it.offset_y;
      int idx_z = it.offset_z;
      for (int d = 0; d < it.ndim; d++) {
        idx_x += indices[d] * it.strides_x[d];
        idx_y += indices[d] * it.strides_y[d];
        idx_z += indices[d] * it.strides_z[d];
      }
      z_data[idx_z] = x_data[idx_x] / y_data[idx_y];

      for (int d = it.ndim - 1; d >= 0; d--) {
        if (++indices[d] < it.shape[d]) break;
        indices[d] = 0;
      }
    }
  } else {
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx_x = x->offset;
      int idx_y = y->offset;
      int idx_z = z->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx_x += indices[d] * x->strides[d];
        idx_y += indices[d] * y->strides[d];
        idx_z += indices[d] * z->strides[d];
      }
      z_data[idx_z] = x_data[idx_x] / y_data[idx_y];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }
}

/* Integer division for int64 */
void nx_cblas_idiv_i64(const strided_array_t *x, const strided_array_t *y,
                       strided_array_t *z) {
  int64_t *x_data = (int64_t *)x->data;
  int64_t *y_data = (int64_t *)y->data;
  int64_t *z_data = (int64_t *)z->data;
  int total = total_elements(x);

  if (is_contiguous(x) && is_contiguous(y) && is_contiguous(z)) {
    if (total > PARALLEL_THRESHOLD * 4) {
#pragma omp parallel for simd
      for (int i = 0; i < total; i++) {
        z_data[i] = x_data[i] / y_data[i];
      }
    } else {
#pragma omp simd
      for (int i = 0; i < total; i++) {
        z_data[i] = x_data[i] / y_data[i];
      }
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    iterator_state_t it;
    init_binary_iterator(&it, x, y, z);

    int indices[MAX_STACK_DIMS] = {0};
    for (int i = 0; i < total; i++) {
      int idx_x = it.offset_x;
      int idx_y = it.offset_y;
      int idx_z = it.offset_z;
      for (int d = 0; d < it.ndim; d++) {
        idx_x += indices[d] * it.strides_x[d];
        idx_y += indices[d] * it.strides_y[d];
        idx_z += indices[d] * it.strides_z[d];
      }
      z_data[idx_z] = x_data[idx_x] / y_data[idx_y];

      for (int d = it.ndim - 1; d >= 0; d--) {
        if (++indices[d] < it.shape[d]) break;
        indices[d] = 0;
      }
    }
  } else {
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx_x = x->offset;
      int idx_y = y->offset;
      int idx_z = z->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx_x += indices[d] * x->strides[d];
        idx_y += indices[d] * y->strides[d];
        idx_z += indices[d] * z->strides[d];
      }
      z_data[idx_z] = x_data[idx_x] / y_data[idx_y];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }
}

/* =========================== Bitwise Operations =========================== */

/* Macro for bitwise operations */
#define BITWISE_OP(name, OP, TYPE)                                         \
  void nx_cblas_##name##_##TYPE(const strided_array_t *x,                  \
                                const strided_array_t *y,                  \
                                strided_array_t *z) {                      \
    TYPE##_t *x_data = (TYPE##_t *)x->data;                                \
    TYPE##_t *y_data = (TYPE##_t *)y->data;                                \
    TYPE##_t *z_data = (TYPE##_t *)z->data;                                \
    int total = total_elements(x);                                         \
                                                                           \
    if (is_contiguous(x) && is_contiguous(y) && is_contiguous(z)) {        \
      if (total > PARALLEL_THRESHOLD * 4) {                                \
        _Pragma("omp parallel for simd") for (int i = 0; i < total; i++) { \
          z_data[i] = x_data[i] OP y_data[i];                              \
        }                                                                  \
      } else {                                                             \
        _Pragma("omp simd") for (int i = 0; i < total; i++) {              \
          z_data[i] = x_data[i] OP y_data[i];                              \
        }                                                                  \
      }                                                                    \
    } else if (x->ndim <= MAX_STACK_DIMS) {                                \
      iterator_state_t it;                                                 \
      init_binary_iterator(&it, x, y, z);                                  \
                                                                           \
      int indices[MAX_STACK_DIMS] = {0};                                   \
      for (int i = 0; i < total; i++) {                                    \
        int idx_x = it.offset_x;                                           \
        int idx_y = it.offset_y;                                           \
        int idx_z = it.offset_z;                                           \
        for (int d = 0; d < it.ndim; d++) {                                \
          idx_x += indices[d] * it.strides_x[d];                           \
          idx_y += indices[d] * it.strides_y[d];                           \
          idx_z += indices[d] * it.strides_z[d];                           \
        }                                                                  \
        z_data[idx_z] = x_data[idx_x] OP y_data[idx_y];                    \
                                                                           \
        for (int d = it.ndim - 1; d >= 0; d--) {                           \
          if (++indices[d] < it.shape[d]) break;                           \
          indices[d] = 0;                                                  \
        }                                                                  \
      }                                                                    \
    } else {                                                               \
      int *indices = (int *)calloc(x->ndim, sizeof(int));                  \
      for (int i = 0; i < total; i++) {                                    \
        int idx_x = x->offset;                                             \
        int idx_y = y->offset;                                             \
        int idx_z = z->offset;                                             \
        for (int d = 0; d < x->ndim; d++) {                                \
          idx_x += indices[d] * x->strides[d];                             \
          idx_y += indices[d] * y->strides[d];                             \
          idx_z += indices[d] * z->strides[d];                             \
        }                                                                  \
        z_data[idx_z] = x_data[idx_x] OP y_data[idx_y];                    \
                                                                           \
        for (int d = x->ndim - 1; d >= 0; d--) {                           \
          if (++indices[d] < x->shape[d]) break;                           \
          indices[d] = 0;                                                  \
        }                                                                  \
      }                                                                    \
      free(indices);                                                       \
    }                                                                      \
  }

/* Define bitwise operations for different integer types */
BITWISE_OP(xor, ^, int32)
BITWISE_OP(xor, ^, int64)
BITWISE_OP(xor, ^, uint8)
BITWISE_OP(xor, ^, uint16)

BITWISE_OP(or, |, int32)
BITWISE_OP(or, |, int64)
BITWISE_OP(or, |, uint8)
BITWISE_OP(or, |, uint16)

BITWISE_OP(and, &, int32)
BITWISE_OP(and, &, int64)
BITWISE_OP(and, &, uint8)
BITWISE_OP(and, &, uint16)

/* =========================== Ternary Operation (WHERE)
 * =========================== */

/* WHERE operation for float32 */
void nx_cblas_where_f32(const strided_array_t *cond, const strided_array_t *x,
                        const strided_array_t *y, strided_array_t *z) {
  uint8_t *cond_data = (uint8_t *)cond->data;
  float *x_data = (float *)x->data;
  float *y_data = (float *)y->data;
  float *z_data = (float *)z->data;
  int total = total_elements(x);

  if (is_contiguous(cond) && is_contiguous(x) && is_contiguous(y) &&
      is_contiguous(z)) {
    if (total > PARALLEL_THRESHOLD * 4) {
#pragma omp parallel for simd aligned(cond_data, x_data, y_data, z_data : 32)
      for (int i = 0; i < total; i++) {
        z_data[i] = cond_data[i] ? x_data[i] : y_data[i];
      }
    } else {
#pragma omp simd aligned(cond_data, x_data, y_data, z_data : 32)
      for (int i = 0; i < total; i++) {
        z_data[i] = cond_data[i] ? x_data[i] : y_data[i];
      }
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    /* Initialize iterator for ternary operation */
    int ndim = x->ndim;
    int indices[MAX_STACK_DIMS] = {0};

    for (int i = 0; i < total; i++) {
      int idx_cond = cond->offset;
      int idx_x = x->offset;
      int idx_y = y->offset;
      int idx_z = z->offset;

      for (int d = 0; d < ndim; d++) {
        idx_cond += indices[d] * cond->strides[d];
        idx_x += indices[d] * x->strides[d];
        idx_y += indices[d] * y->strides[d];
        idx_z += indices[d] * z->strides[d];
      }

      z_data[idx_z] = cond_data[idx_cond] ? x_data[idx_x] : y_data[idx_y];

      for (int d = ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
  } else {
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx_cond = cond->offset;
      int idx_x = x->offset;
      int idx_y = y->offset;
      int idx_z = z->offset;

      for (int d = 0; d < x->ndim; d++) {
        idx_cond += indices[d] * cond->strides[d];
        idx_x += indices[d] * x->strides[d];
        idx_y += indices[d] * y->strides[d];
        idx_z += indices[d] * z->strides[d];
      }

      z_data[idx_z] = cond_data[idx_cond] ? x_data[idx_x] : y_data[idx_y];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }
}

/* WHERE operation for float64 */
void nx_cblas_where_f64(const strided_array_t *cond, const strided_array_t *x,
                        const strided_array_t *y, strided_array_t *z) {
  uint8_t *cond_data = (uint8_t *)cond->data;
  double *x_data = (double *)x->data;
  double *y_data = (double *)y->data;
  double *z_data = (double *)z->data;
  int total = total_elements(x);

  if (is_contiguous(cond) && is_contiguous(x) && is_contiguous(y) &&
      is_contiguous(z)) {
    if (total > PARALLEL_THRESHOLD * 4) {
#pragma omp parallel for simd aligned(cond_data, x_data, y_data, z_data : 32)
      for (int i = 0; i < total; i++) {
        z_data[i] = cond_data[i] ? x_data[i] : y_data[i];
      }
    } else {
#pragma omp simd aligned(cond_data, x_data, y_data, z_data : 32)
      for (int i = 0; i < total; i++) {
        z_data[i] = cond_data[i] ? x_data[i] : y_data[i];
      }
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    /* Initialize iterator for ternary operation */
    int ndim = x->ndim;
    int indices[MAX_STACK_DIMS] = {0};

    for (int i = 0; i < total; i++) {
      int idx_cond = cond->offset;
      int idx_x = x->offset;
      int idx_y = y->offset;
      int idx_z = z->offset;

      for (int d = 0; d < ndim; d++) {
        idx_cond += indices[d] * cond->strides[d];
        idx_x += indices[d] * x->strides[d];
        idx_y += indices[d] * y->strides[d];
        idx_z += indices[d] * z->strides[d];
      }

      z_data[idx_z] = cond_data[idx_cond] ? x_data[idx_x] : y_data[idx_y];

      for (int d = ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
  } else {
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx_cond = cond->offset;
      int idx_x = x->offset;
      int idx_y = y->offset;
      int idx_z = z->offset;

      for (int d = 0; d < x->ndim; d++) {
        idx_cond += indices[d] * cond->strides[d];
        idx_x += indices[d] * x->strides[d];
        idx_y += indices[d] * y->strides[d];
        idx_z += indices[d] * z->strides[d];
      }

      z_data[idx_z] = cond_data[idx_cond] ? x_data[idx_x] : y_data[idx_y];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }
}

/* =========================== Reduction Product =========================== */

void nx_cblas_reduce_prod_f32(const strided_array_t *x, void *result) {
  int total = total_elements(x);
  float *x_data = (float *)x->data;
  float prod = 1.0f;

  if (is_contiguous(x)) {
/* Vectorized product for arrays */
#pragma omp simd reduction(* : prod)
    for (int i = 0; i < total; i++) {
      prod *= x_data[i];
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    /* strided product */
    int indices[MAX_STACK_DIMS] = {0};
    for (int i = 0; i < total; i++) {
      int idx = x->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx += indices[d] * x->strides[d];
      }
      prod *= x_data[idx];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
  } else {
    /* Fallback */
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx = x->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx += indices[d] * x->strides[d];
      }
      prod *= x_data[idx];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }

  *((float *)result) = prod;
}

void nx_cblas_reduce_prod_f64(const strided_array_t *x, void *result) {
  int total = total_elements(x);
  double *x_data = (double *)x->data;
  double prod = 1.0;

  if (is_contiguous(x)) {
/* Vectorized product for arrays */
#pragma omp simd reduction(* : prod)
    for (int i = 0; i < total; i++) {
      prod *= x_data[i];
    }
  } else if (x->ndim <= MAX_STACK_DIMS) {
    /* strided product */
    int indices[MAX_STACK_DIMS] = {0};
    for (int i = 0; i < total; i++) {
      int idx = x->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx += indices[d] * x->strides[d];
      }
      prod *= x_data[idx];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
  } else {
    /* Fallback */
    int *indices = (int *)calloc(x->ndim, sizeof(int));
    for (int i = 0; i < total; i++) {
      int idx = x->offset;
      for (int d = 0; d < x->ndim; d++) {
        idx += indices[d] * x->strides[d];
      }
      prod *= x_data[idx];

      for (int d = x->ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
    free(indices);
  }

  *((double *)result) = prod;
}

/* =========================== Generic Copy =========================== */

void nx_cblas_copy_generic(const strided_array_t *x, strided_array_t *z,
                           size_t elem_size) {
  int total = total_elements(x);

  if (is_contiguous(x) && is_contiguous(z)) {
    memcpy(z->data, x->data, total * elem_size);
  } else {
    /* Use stack allocation for common case */
    if (x->ndim <= MAX_STACK_DIMS) {
      int indices[MAX_STACK_DIMS] = {0};

      for (int i = 0; i < total; i++) {
        int idx_x = x->offset;
        int idx_z = z->offset;
        for (int d = 0; d < x->ndim; d++) {
          idx_x += indices[d] * x->strides[d];
          idx_z += indices[d] * z->strides[d];
        }
        memcpy((char *)z->data + idx_z * elem_size,
               (char *)x->data + idx_x * elem_size, elem_size);

        for (int d = x->ndim - 1; d >= 0; d--) {
          if (++indices[d] < x->shape[d]) break;
          indices[d] = 0;
        }
      }
    } else {
      /* Fallback for high dimensional arrays */
      int *indices = (int *)calloc(x->ndim, sizeof(int));

      for (int i = 0; i < total; i++) {
        int idx_x = x->offset;
        int idx_z = z->offset;
        for (int d = 0; d < x->ndim; d++) {
          idx_x += indices[d] * x->strides[d];
          idx_z += indices[d] * z->strides[d];
        }
        memcpy((char *)z->data + idx_z * elem_size,
               (char *)x->data + idx_x * elem_size, elem_size);

        for (int d = x->ndim - 1; d >= 0; d--) {
          if (++indices[d] < x->shape[d]) break;
          indices[d] = 0;
        }
      }

      free(indices);
    }
  }
}

/* =========================== Pad Operation =========================== */

/* PAD operation for float32 */
void nx_cblas_pad_f32(const strided_array_t *x, strided_array_t *z,
                      const int *pad_config, float fill_value) {
  float *x_data = (float *)x->data;
  float *z_data = (float *)z->data;

  int ndim = x->ndim;

  /* First fill the entire output with the fill value */
  int total_z = total_elements(z);
#pragma omp simd
  for (int i = 0; i < total_z; i++) {
    z_data[i] = fill_value;
  }

  /* Then copy the input data to the appropriate positions */
  int total_x = total_elements(x);

  if (ndim <= MAX_STACK_DIMS) {
    int indices[MAX_STACK_DIMS] = {0};

    for (int i = 0; i < total_x; i++) {
      /* Calculate source index in x */
      int idx_x = x->offset;
      for (int d = 0; d < ndim; d++) {
        idx_x += indices[d] * x->strides[d];
      }

      /* Calculate destination index in z, accounting for padding */
      int idx_z = z->offset;
      for (int d = 0; d < ndim; d++) {
        idx_z += (indices[d] + pad_config[d * 2]) * z->strides[d];
      }

      z_data[idx_z] = x_data[idx_x];

      /* Increment indices */
      for (int d = ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
  } else {
    int *indices = (int *)calloc(ndim, sizeof(int));

    for (int i = 0; i < total_x; i++) {
      /* Calculate source index in x */
      int idx_x = x->offset;
      for (int d = 0; d < ndim; d++) {
        idx_x += indices[d] * x->strides[d];
      }

      /* Calculate destination index in z, accounting for padding */
      int idx_z = z->offset;
      for (int d = 0; d < ndim; d++) {
        idx_z += (indices[d] + pad_config[d * 2]) * z->strides[d];
      }

      z_data[idx_z] = x_data[idx_x];

      /* Increment indices */
      for (int d = ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }

    free(indices);
  }
}

/* PAD operation for float64 */
void nx_cblas_pad_f64(const strided_array_t *x, strided_array_t *z,
                      const int *pad_config, double fill_value) {
  double *x_data = (double *)x->data;
  double *z_data = (double *)z->data;

  int ndim = x->ndim;

  /* First fill the entire output with the fill value */
  int total_z = total_elements(z);
#pragma omp simd
  for (int i = 0; i < total_z; i++) {
    z_data[i] = fill_value;
  }

  /* Then copy the input data to the appropriate positions */
  int total_x = total_elements(x);

  if (ndim <= MAX_STACK_DIMS) {
    int indices[MAX_STACK_DIMS] = {0};

    for (int i = 0; i < total_x; i++) {
      /* Calculate source index in x */
      int idx_x = x->offset;
      for (int d = 0; d < ndim; d++) {
        idx_x += indices[d] * x->strides[d];
      }

      /* Calculate destination index in z, accounting for padding */
      int idx_z = z->offset;
      for (int d = 0; d < ndim; d++) {
        idx_z += (indices[d] + pad_config[d * 2]) * z->strides[d];
      }

      z_data[idx_z] = x_data[idx_x];

      /* Increment indices */
      for (int d = ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }
  } else {
    int *indices = (int *)calloc(ndim, sizeof(int));

    for (int i = 0; i < total_x; i++) {
      /* Calculate source index in x */
      int idx_x = x->offset;
      for (int d = 0; d < ndim; d++) {
        idx_x += indices[d] * x->strides[d];
      }

      /* Calculate destination index in z, accounting for padding */
      int idx_z = z->offset;
      for (int d = 0; d < ndim; d++) {
        idx_z += (indices[d] + pad_config[d * 2]) * z->strides[d];
      }

      z_data[idx_z] = x_data[idx_x];

      /* Increment indices */
      for (int d = ndim - 1; d >= 0; d--) {
        if (++indices[d] < x->shape[d]) break;
        indices[d] = 0;
      }
    }

    free(indices);
  }
}

/* =========================== Threefry Operation =========================== */

/* Threefry-2x32 random number generator */
void nx_cblas_threefry(const strided_array_t *key,
                       const strided_array_t *counter, strided_array_t *z) {
  int32_t *key_data = (int32_t *)key->data;
  int32_t *counter_data = (int32_t *)counter->data;
  int32_t *z_data = (int32_t *)z->data;

  /* Threefry-2x32 rotation constants */
  const uint32_t R[8][2] = {{13, 15}, {26, 6}, {17, 29}, {16, 14},
                            {13, 15}, {26, 6}, {17, 29}, {16, 14}};

  int total = total_elements(z);

  /* For each output element, generate a random number */
  for (int i = 0; i < total; i++) {
    /* Get key values (broadcast if necessary) */
    uint32_t k0 = (uint32_t)key_data[i % total_elements(key)];
    uint32_t k1 = (uint32_t)key_data[(i + 1) % total_elements(key)];
    uint32_t k2 = 0x1BD11BDA; /* Threefry constant */

    /* Get counter values */
    uint32_t x0 = (uint32_t)counter_data[i % total_elements(counter)];
    uint32_t x1 = (uint32_t)counter_data[(i + 1) % total_elements(counter)];

    /* Initialize */
    k2 = k0 ^ k1 ^ k2;
    x0 += k0;
    x1 += k1;

    /* 4 rounds of Threefry */
    for (int round = 0; round < 4; round++) {
      /* Mix */
      x0 += x1;
      x1 = (x1 << R[round][0]) | (x1 >> (32 - R[round][0]));
      x1 ^= x0;

      x0 += x1;
      x1 = (x1 << R[round][1]) | (x1 >> (32 - R[round][1]));
      x1 ^= x0;

      /* Key injection */
      if (round == 1) {
        x0 += k1;
        x1 += k2 + 1;
      } else if (round == 3) {
        x0 += k2;
        x1 += k0 + 2;
      }
    }

    /* Store result */
    z_data[i] = (int32_t)x0;
  }
}

/* =========================== Gather Operation =========================== */

/* GATHER operation for float32 */
void nx_cblas_gather_f32(const strided_array_t *data,
                         const strided_array_t *indices, int axis,
                         strided_array_t *z) {
  float *data_ptr = (float *)data->data;
  int32_t *indices_ptr = (int32_t *)indices->data;
  float *z_data = (float *)z->data;

  int ndim = data->ndim;
  int total = total_elements(indices);

  if (ndim <= MAX_STACK_DIMS) {
    int idx[MAX_STACK_DIMS] = {0};

    for (int i = 0; i < total; i++) {
      /* Get the index value for the gather axis */
      int gather_idx = indices_ptr[i];

      /* Bounds check */
      if (gather_idx < 0 || gather_idx >= data->shape[axis]) {
        z_data[i] = 0.0f; /* Out of bounds */
        continue;
      }

      /* Calculate source position in data */
      int src_idx = data->offset;
      for (int d = 0; d < ndim; d++) {
        if (d == axis) {
          src_idx += gather_idx * data->strides[d];
        } else {
          src_idx += idx[d] * data->strides[d];
        }
      }

      /* Copy value */
      z_data[i] = data_ptr[src_idx];

      /* Increment indices */
      for (int d = ndim - 1; d >= 0; d--) {
        if (d == axis) continue;
        if (++idx[d] < indices->shape[d]) break;
        idx[d] = 0;
      }
    }
  } else {
    /* High-dimensional fallback */
    int *idx = (int *)calloc(ndim, sizeof(int));

    for (int i = 0; i < total; i++) {
      /* Get the index value for the gather axis */
      int gather_idx = indices_ptr[i];

      /* Bounds check */
      if (gather_idx < 0 || gather_idx >= data->shape[axis]) {
        z_data[i] = 0.0f; /* Out of bounds */
        continue;
      }

      /* Calculate source position in data */
      int src_idx = data->offset;
      for (int d = 0; d < ndim; d++) {
        if (d == axis) {
          src_idx += gather_idx * data->strides[d];
        } else {
          src_idx += idx[d] * data->strides[d];
        }
      }

      /* Copy value */
      z_data[i] = data_ptr[src_idx];

      /* Increment indices */
      for (int d = ndim - 1; d >= 0; d--) {
        if (d == axis) continue;
        if (++idx[d] < indices->shape[d]) break;
        idx[d] = 0;
      }
    }

    free(idx);
  }
}

/* GATHER operation for float64 */
void nx_cblas_gather_f64(const strided_array_t *data,
                         const strided_array_t *indices, int axis,
                         strided_array_t *z) {
  double *data_ptr = (double *)data->data;
  int32_t *indices_ptr = (int32_t *)indices->data;
  double *z_data = (double *)z->data;

  int ndim = data->ndim;
  int total = total_elements(indices);

  if (ndim <= MAX_STACK_DIMS) {
    int idx[MAX_STACK_DIMS] = {0};

    for (int i = 0; i < total; i++) {
      /* Get the index value for the gather axis */
      int gather_idx = indices_ptr[i];

      /* Bounds check */
      if (gather_idx < 0 || gather_idx >= data->shape[axis]) {
        z_data[i] = 0.0; /* Out of bounds */
        continue;
      }

      /* Calculate source position in data */
      int src_idx = data->offset;
      for (int d = 0; d < ndim; d++) {
        if (d == axis) {
          src_idx += gather_idx * data->strides[d];
        } else {
          src_idx += idx[d] * data->strides[d];
        }
      }

      /* Copy value */
      z_data[i] = data_ptr[src_idx];

      /* Increment indices */
      for (int d = ndim - 1; d >= 0; d--) {
        if (d == axis) continue;
        if (++idx[d] < indices->shape[d]) break;
        idx[d] = 0;
      }
    }
  } else {
    /* High-dimensional fallback */
    int *idx = (int *)calloc(ndim, sizeof(int));

    for (int i = 0; i < total; i++) {
      /* Get the index value for the gather axis */
      int gather_idx = indices_ptr[i];

      /* Bounds check */
      if (gather_idx < 0 || gather_idx >= data->shape[axis]) {
        z_data[i] = 0.0; /* Out of bounds */
        continue;
      }

      /* Calculate source position in data */
      int src_idx = data->offset;
      for (int d = 0; d < ndim; d++) {
        if (d == axis) {
          src_idx += gather_idx * data->strides[d];
        } else {
          src_idx += idx[d] * data->strides[d];
        }
      }

      /* Copy value */
      z_data[i] = data_ptr[src_idx];

      /* Increment indices */
      for (int d = ndim - 1; d >= 0; d--) {
        if (d == axis) continue;
        if (++idx[d] < indices->shape[d]) break;
        idx[d] = 0;
      }
    }

    free(idx);
  }
}

/* =========================== Scatter Operation =========================== */

/* SCATTER operation for float32 */
void nx_cblas_scatter_f32(const strided_array_t *data_template,
                          const strided_array_t *indices,
                          const strided_array_t *updates, int axis,
                          strided_array_t *z) {
  float *template_ptr = (float *)data_template->data;
  int32_t *indices_ptr = (int32_t *)indices->data;
  float *updates_ptr = (float *)updates->data;
  float *z_data = (float *)z->data;

  int ndim = data_template->ndim;
  int total_template = total_elements(data_template);
  int total_updates = total_elements(updates);

  /* First, copy template to output */
  if (is_contiguous(data_template) && is_contiguous(z)) {
    memcpy(z_data, template_ptr, total_template * sizeof(float));
  } else {
    /* Element-by-element copy */
    if (ndim <= MAX_STACK_DIMS) {
      int idx[MAX_STACK_DIMS] = {0};
      for (int i = 0; i < total_template; i++) {
        int src_idx = data_template->offset;
        int dst_idx = z->offset;
        for (int d = 0; d < ndim; d++) {
          src_idx += idx[d] * data_template->strides[d];
          dst_idx += idx[d] * z->strides[d];
        }
        z_data[dst_idx] = template_ptr[src_idx];

        for (int d = ndim - 1; d >= 0; d--) {
          if (++idx[d] < data_template->shape[d]) break;
          idx[d] = 0;
        }
      }
    } else {
      int *idx = (int *)calloc(ndim, sizeof(int));
      for (int i = 0; i < total_template; i++) {
        int src_idx = data_template->offset;
        int dst_idx = z->offset;
        for (int d = 0; d < ndim; d++) {
          src_idx += idx[d] * data_template->strides[d];
          dst_idx += idx[d] * z->strides[d];
        }
        z_data[dst_idx] = template_ptr[src_idx];

        for (int d = ndim - 1; d >= 0; d--) {
          if (++idx[d] < data_template->shape[d]) break;
          idx[d] = 0;
        }
      }
      free(idx);
    }
  }

  /* Then scatter updates */
  if (ndim <= MAX_STACK_DIMS) {
    int idx[MAX_STACK_DIMS] = {0};

    for (int i = 0; i < total_updates; i++) {
      /* Get the scatter index */
      int scatter_idx = indices_ptr[i];

      /* Bounds check */
      if (scatter_idx < 0 || scatter_idx >= z->shape[axis]) {
        continue; /* Skip out of bounds */
      }

      /* Calculate destination position */
      int dst_idx = z->offset;
      for (int d = 0; d < ndim; d++) {
        if (d == axis) {
          dst_idx += scatter_idx * z->strides[d];
        } else {
          dst_idx += idx[d] * z->strides[d];
        }
      }

      /* Update value */
      z_data[dst_idx] = updates_ptr[i];

      /* Increment indices */
      for (int d = ndim - 1; d >= 0; d--) {
        if (d == axis) continue;
        if (++idx[d] < updates->shape[d]) break;
        idx[d] = 0;
      }
    }
  } else {
    /* High-dimensional fallback */
    int *idx = (int *)calloc(ndim, sizeof(int));

    for (int i = 0; i < total_updates; i++) {
      /* Get the scatter index */
      int scatter_idx = indices_ptr[i];

      /* Bounds check */
      if (scatter_idx < 0 || scatter_idx >= z->shape[axis]) {
        continue; /* Skip out of bounds */
      }

      /* Calculate destination position */
      int dst_idx = z->offset;
      for (int d = 0; d < ndim; d++) {
        if (d == axis) {
          dst_idx += scatter_idx * z->strides[d];
        } else {
          dst_idx += idx[d] * z->strides[d];
        }
      }

      /* Update value */
      z_data[dst_idx] = updates_ptr[i];

      /* Increment indices */
      for (int d = ndim - 1; d >= 0; d--) {
        if (d == axis) continue;
        if (++idx[d] < updates->shape[d]) break;
        idx[d] = 0;
      }
    }

    free(idx);
  }
}

/* SCATTER operation for float64 */
void nx_cblas_scatter_f64(const strided_array_t *data_template,
                          const strided_array_t *indices,
                          const strided_array_t *updates, int axis,
                          strided_array_t *z) {
  double *template_ptr = (double *)data_template->data;
  int32_t *indices_ptr = (int32_t *)indices->data;
  double *updates_ptr = (double *)updates->data;
  double *z_data = (double *)z->data;

  int ndim = data_template->ndim;
  int total_template = total_elements(data_template);
  int total_updates = total_elements(updates);

  /* First, copy template to output */
  if (is_contiguous(data_template) && is_contiguous(z)) {
    memcpy(z_data, template_ptr, total_template * sizeof(double));
  } else {
    /* Element-by-element copy */
    if (ndim <= MAX_STACK_DIMS) {
      int idx[MAX_STACK_DIMS] = {0};
      for (int i = 0; i < total_template; i++) {
        int src_idx = data_template->offset;
        int dst_idx = z->offset;
        for (int d = 0; d < ndim; d++) {
          src_idx += idx[d] * data_template->strides[d];
          dst_idx += idx[d] * z->strides[d];
        }
        z_data[dst_idx] = template_ptr[src_idx];

        for (int d = ndim - 1; d >= 0; d--) {
          if (++idx[d] < data_template->shape[d]) break;
          idx[d] = 0;
        }
      }
    } else {
      int *idx = (int *)calloc(ndim, sizeof(int));
      for (int i = 0; i < total_template; i++) {
        int src_idx = data_template->offset;
        int dst_idx = z->offset;
        for (int d = 0; d < ndim; d++) {
          src_idx += idx[d] * data_template->strides[d];
          dst_idx += idx[d] * z->strides[d];
        }
        z_data[dst_idx] = template_ptr[src_idx];

        for (int d = ndim - 1; d >= 0; d--) {
          if (++idx[d] < data_template->shape[d]) break;
          idx[d] = 0;
        }
      }
      free(idx);
    }
  }

  /* Then scatter updates */
  if (ndim <= MAX_STACK_DIMS) {
    int idx[MAX_STACK_DIMS] = {0};

    for (int i = 0; i < total_updates; i++) {
      /* Get the scatter index */
      int scatter_idx = indices_ptr[i];

      /* Bounds check */
      if (scatter_idx < 0 || scatter_idx >= z->shape[axis]) {
        continue; /* Skip out of bounds */
      }

      /* Calculate destination position */
      int dst_idx = z->offset;
      for (int d = 0; d < ndim; d++) {
        if (d == axis) {
          dst_idx += scatter_idx * z->strides[d];
        } else {
          dst_idx += idx[d] * z->strides[d];
        }
      }

      /* Update value */
      z_data[dst_idx] = updates_ptr[i];

      /* Increment indices */
      for (int d = ndim - 1; d >= 0; d--) {
        if (d == axis) continue;
        if (++idx[d] < updates->shape[d]) break;
        idx[d] = 0;
      }
    }
  } else {
    /* High-dimensional fallback */
    int *idx = (int *)calloc(ndim, sizeof(int));

    for (int i = 0; i < total_updates; i++) {
      /* Get the scatter index */
      int scatter_idx = indices_ptr[i];

      /* Bounds check */
      if (scatter_idx < 0 || scatter_idx >= z->shape[axis]) {
        continue; /* Skip out of bounds */
      }

      /* Calculate destination position */
      int dst_idx = z->offset;
      for (int d = 0; d < ndim; d++) {
        if (d == axis) {
          dst_idx += scatter_idx * z->strides[d];
        } else {
          dst_idx += idx[d] * z->strides[d];
        }
      }

      /* Update value */
      z_data[dst_idx] = updates_ptr[i];

      /* Increment indices */
      for (int d = ndim - 1; d >= 0; d--) {
        if (d == axis) continue;
        if (++idx[d] < updates->shape[d]) break;
        idx[d] = 0;
      }
    }

    free(idx);
  }
}

/* =========================== Cast Operation =========================== */

/* Cast float32 to float64 */
void nx_cblas_cast_f32_to_f64(const strided_array_t *x, strided_array_t *z) {
  float *x_data = (float *)x->data;
  double *z_data = (double *)z->data;
  int total = total_elements(x);

  if (is_contiguous(x) && is_contiguous(z)) {
#pragma omp simd
    for (int i = 0; i < total; i++) {
      z_data[i] = (double)x_data[i];
    }
  } else {
    if (x->ndim <= MAX_STACK_DIMS) {
      int indices[MAX_STACK_DIMS] = {0};
      for (int i = 0; i < total; i++) {
        int idx_x = x->offset;
        int idx_z = z->offset;
        for (int d = 0; d < x->ndim; d++) {
          idx_x += indices[d] * x->strides[d];
          idx_z += indices[d] * z->strides[d];
        }
        z_data[idx_z] = (double)x_data[idx_x];

        for (int d = x->ndim - 1; d >= 0; d--) {
          if (++indices[d] < x->shape[d]) break;
          indices[d] = 0;
        }
      }
    } else {
      int *indices = (int *)calloc(x->ndim, sizeof(int));
      for (int i = 0; i < total; i++) {
        int idx_x = x->offset;
        int idx_z = z->offset;
        for (int d = 0; d < x->ndim; d++) {
          idx_x += indices[d] * x->strides[d];
          idx_z += indices[d] * z->strides[d];
        }
        z_data[idx_z] = (double)x_data[idx_x];

        for (int d = x->ndim - 1; d >= 0; d--) {
          if (++indices[d] < x->shape[d]) break;
          indices[d] = 0;
        }
      }
      free(indices);
    }
  }
}

/* Cast float64 to float32 */
void nx_cblas_cast_f64_to_f32(const strided_array_t *x, strided_array_t *z) {
  double *x_data = (double *)x->data;
  float *z_data = (float *)z->data;
  int total = total_elements(x);

  if (is_contiguous(x) && is_contiguous(z)) {
#pragma omp simd
    for (int i = 0; i < total; i++) {
      z_data[i] = (float)x_data[i];
    }
  } else {
    if (x->ndim <= MAX_STACK_DIMS) {
      int indices[MAX_STACK_DIMS] = {0};
      for (int i = 0; i < total; i++) {
        int idx_x = x->offset;
        int idx_z = z->offset;
        for (int d = 0; d < x->ndim; d++) {
          idx_x += indices[d] * x->strides[d];
          idx_z += indices[d] * z->strides[d];
        }
        z_data[idx_z] = (float)x_data[idx_x];

        for (int d = x->ndim - 1; d >= 0; d--) {
          if (++indices[d] < x->shape[d]) break;
          indices[d] = 0;
        }
      }
    } else {
      int *indices = (int *)calloc(x->ndim, sizeof(int));
      for (int i = 0; i < total; i++) {
        int idx_x = x->offset;
        int idx_z = z->offset;
        for (int d = 0; d < x->ndim; d++) {
          idx_x += indices[d] * x->strides[d];
          idx_z += indices[d] * z->strides[d];
        }
        z_data[idx_z] = (float)x_data[idx_x];

        for (int d = x->ndim - 1; d >= 0; d--) {
          if (++indices[d] < x->shape[d]) break;
          indices[d] = 0;
        }
      }
      free(indices);
    }
  }
}

/* ========================== Matrix Multiplication ========================== */

/* Float32 matrix multiplication */
void nx_cblas_matmul_f32(const strided_array_t *a, const strided_array_t *b, strided_array_t *c) {
  float *a_data = (float *)a->data;
  float *b_data = (float *)b->data;
  float *c_data = (float *)c->data;
  
  int ndim_a = a->ndim;
  int ndim_b = b->ndim;
  int ndim_c = c->ndim;
  
  // Extract matrix dimensions
  int m = a->shape[ndim_a - 2];
  int k = a->shape[ndim_a - 1];
  int n = b->shape[ndim_b - 1];
  
  // Calculate batch size
  int batch_size = 1;
  for (int i = 0; i < ndim_c - 2; i++) {
    batch_size *= c->shape[i];
  }
  
  // Perform batched matrix multiplication using CBLAS
  for (int batch = 0; batch < batch_size; batch++) {
    // Calculate batch offsets
    int batch_offset_a = batch * m * k;
    int batch_offset_b = batch * k * n;
    int batch_offset_c = batch * m * n;
    
    // Use CBLAS sgemm for matrix multiplication
    // C = alpha * A * B + beta * C
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                1.0f,  // alpha
                a_data + batch_offset_a, k,  // A, lda
                b_data + batch_offset_b, n,  // B, ldb
                0.0f,  // beta
                c_data + batch_offset_c, n); // C, ldc
  }
}

/* Float64 matrix multiplication */
void nx_cblas_matmul_f64(const strided_array_t *a, const strided_array_t *b, strided_array_t *c) {
  double *a_data = (double *)a->data;
  double *b_data = (double *)b->data;
  double *c_data = (double *)c->data;
  
  int ndim_a = a->ndim;
  int ndim_b = b->ndim;
  int ndim_c = c->ndim;
  
  // Extract matrix dimensions
  int m = a->shape[ndim_a - 2];
  int k = a->shape[ndim_a - 1];
  int n = b->shape[ndim_b - 1];
  
  // Calculate batch size
  int batch_size = 1;
  for (int i = 0; i < ndim_c - 2; i++) {
    batch_size *= c->shape[i];
  }
  
  // Perform batched matrix multiplication using CBLAS
  for (int batch = 0; batch < batch_size; batch++) {
    // Calculate batch offsets
    int batch_offset_a = batch * m * k;
    int batch_offset_b = batch * k * n;
    int batch_offset_c = batch * m * n;
    
    // Use CBLAS dgemm for matrix multiplication
    // C = alpha * A * B + beta * C
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                1.0,   // alpha
                a_data + batch_offset_a, k,  // A, lda
                b_data + batch_offset_b, n,  // B, ldb
                0.0,   // beta
                c_data + batch_offset_c, n); // C, ldc
  }
}

/* ========================== Unfold (im2col) ========================== */

/* Float32 unfold */
void nx_cblas_unfold_f32(const strided_array_t *input, strided_array_t *output,
                         int n_spatial, const int *kernel_size, const int *stride,
                         const int *dilation, const int *padding) {
  float *in_data = (float *)input->data;
  float *out_data = (float *)output->data;
  
  int batch_size = input->shape[0];
  int channels = input->shape[1];
  
  // Only handle 2D case for now
  if (n_spatial != 2) {
    return; // TODO: Add support for other dimensions
  }
  
  int h_in = input->shape[2];
  int w_in = input->shape[3];
  
  int kh = kernel_size[0];
  int kw = kernel_size[1];
  int sh = stride[0];
  int sw = stride[1];
  int dh = dilation[0];
  int dw = dilation[1];
  
  // Calculate output dimensions
  int h_out = (h_in - (kh - 1) * dh - 1) / sh + 1;
  int w_out = (w_in - (kw - 1) * dw - 1) / sw + 1;
  
  int patch_size = channels * kh * kw;
  int num_patches = h_out * w_out;
  
  // Extract patches
  for (int b = 0; b < batch_size; b++) {
    for (int oh = 0; oh < h_out; oh++) {
      for (int ow = 0; ow < w_out; ow++) {
        int patch_idx = oh * w_out + ow;
        
        for (int c = 0; c < channels; c++) {
          for (int kh_idx = 0; kh_idx < kh; kh_idx++) {
            for (int kw_idx = 0; kw_idx < kw; kw_idx++) {
              int h = oh * sh + kh_idx * dh;
              int w = ow * sw + kw_idx * dw;
              
              int in_idx = ((b * channels + c) * h_in + h) * w_in + w;
              int out_idx = (b * patch_size + (c * kh + kh_idx) * kw + kw_idx) * num_patches + patch_idx;
              
              if (h >= 0 && h < h_in && w >= 0 && w < w_in) {
                out_data[out_idx] = in_data[in_idx];
              } else {
                out_data[out_idx] = 0.0f;
              }
            }
          }
        }
      }
    }
  }
}

/* Float64 unfold */
void nx_cblas_unfold_f64(const strided_array_t *input, strided_array_t *output,
                         int n_spatial, const int *kernel_size, const int *stride,
                         const int *dilation, const int *padding) {
  double *in_data = (double *)input->data;
  double *out_data = (double *)output->data;
  
  int batch_size = input->shape[0];
  int channels = input->shape[1];
  
  // Only handle 2D case for now
  if (n_spatial != 2) {
    return; // TODO: Add support for other dimensions
  }
  
  int h_in = input->shape[2];
  int w_in = input->shape[3];
  
  int kh = kernel_size[0];
  int kw = kernel_size[1];
  int sh = stride[0];
  int sw = stride[1];
  int dh = dilation[0];
  int dw = dilation[1];
  
  // Calculate output dimensions
  int h_out = (h_in - (kh - 1) * dh - 1) / sh + 1;
  int w_out = (w_in - (kw - 1) * dw - 1) / sw + 1;
  
  int patch_size = channels * kh * kw;
  int num_patches = h_out * w_out;
  
  // Extract patches
  for (int b = 0; b < batch_size; b++) {
    for (int oh = 0; oh < h_out; oh++) {
      for (int ow = 0; ow < w_out; ow++) {
        int patch_idx = oh * w_out + ow;
        
        for (int c = 0; c < channels; c++) {
          for (int kh_idx = 0; kh_idx < kh; kh_idx++) {
            for (int kw_idx = 0; kw_idx < kw; kw_idx++) {
              int h = oh * sh + kh_idx * dh;
              int w = ow * sw + kw_idx * dw;
              
              int in_idx = ((b * channels + c) * h_in + h) * w_in + w;
              int out_idx = (b * patch_size + (c * kh + kh_idx) * kw + kw_idx) * num_patches + patch_idx;
              
              if (h >= 0 && h < h_in && w >= 0 && w < w_in) {
                out_data[out_idx] = in_data[in_idx];
              } else {
                out_data[out_idx] = 0.0;
              }
            }
          }
        }
      }
    }
  }
}

/* ========================== Fold (col2im) ========================== */

/* Float32 fold */
void nx_cblas_fold_f32(const strided_array_t *input, strided_array_t *output,
                       int n_spatial, const int *output_size, const int *kernel_size,
                       const int *stride, const int *dilation, const int *padding) {
  float *in_data = (float *)input->data;
  float *out_data = (float *)output->data;
  
  int batch_size = input->shape[0];
  int patch_size = input->shape[1];
  int num_patches = input->shape[2];
  
  // Only handle 2D case for now
  if (n_spatial != 2) {
    return; // TODO: Add support for other dimensions
  }
  
  int kernel_elements = kernel_size[0] * kernel_size[1];
  int channels = patch_size / kernel_elements;
  
  int h_out = output_size[0];
  int w_out = output_size[1];
  
  int kh = kernel_size[0];
  int kw = kernel_size[1];
  int sh = stride[0];
  int sw = stride[1];
  int dh = dilation[0];
  int dw = dilation[1];
  
  // Initialize output to zero
  int out_total = batch_size * channels * h_out * w_out;
  for (int i = 0; i < out_total; i++) {
    out_data[i] = 0.0f;
  }
  
  // Calculate patch grid dimensions
  int effective_kh = 1 + (kh - 1) * dh;
  int effective_kw = 1 + (kw - 1) * dw;
  int h_patches = (h_out - effective_kh) / sh + 1;
  int w_patches = (w_out - effective_kw) / sw + 1;
  
  // Accumulate patches
  for (int b = 0; b < batch_size; b++) {
    for (int ph = 0; ph < h_patches; ph++) {
      for (int pw = 0; pw < w_patches; pw++) {
        int patch_idx = ph * w_patches + pw;
        
        for (int c = 0; c < channels; c++) {
          for (int kh_idx = 0; kh_idx < kh; kh_idx++) {
            for (int kw_idx = 0; kw_idx < kw; kw_idx++) {
              int h = ph * sh + kh_idx * dh;
              int w = pw * sw + kw_idx * dw;
              
              if (h >= 0 && h < h_out && w >= 0 && w < w_out) {
                int in_idx = (b * patch_size + (c * kh + kh_idx) * kw + kw_idx) * num_patches + patch_idx;
                int out_idx = ((b * channels + c) * h_out + h) * w_out + w;
                
                out_data[out_idx] += in_data[in_idx];
              }
            }
          }
        }
      }
    }
  }
}

/* Float64 fold */
void nx_cblas_fold_f64(const strided_array_t *input, strided_array_t *output,
                       int n_spatial, const int *output_size, const int *kernel_size,
                       const int *stride, const int *dilation, const int *padding) {
  double *in_data = (double *)input->data;
  double *out_data = (double *)output->data;
  
  int batch_size = input->shape[0];
  int patch_size = input->shape[1];
  int num_patches = input->shape[2];
  
  // Only handle 2D case for now
  if (n_spatial != 2) {
    return; // TODO: Add support for other dimensions
  }
  
  int kernel_elements = kernel_size[0] * kernel_size[1];
  int channels = patch_size / kernel_elements;
  
  int h_out = output_size[0];
  int w_out = output_size[1];
  
  int kh = kernel_size[0];
  int kw = kernel_size[1];
  int sh = stride[0];
  int sw = stride[1];
  int dh = dilation[0];
  int dw = dilation[1];
  
  // Initialize output to zero
  int out_total = batch_size * channels * h_out * w_out;
  for (int i = 0; i < out_total; i++) {
    out_data[i] = 0.0;
  }
  
  // Calculate patch grid dimensions
  int effective_kh = 1 + (kh - 1) * dh;
  int effective_kw = 1 + (kw - 1) * dw;
  int h_patches = (h_out - effective_kh) / sh + 1;
  int w_patches = (w_out - effective_kw) / sw + 1;
  
  // Accumulate patches
  for (int b = 0; b < batch_size; b++) {
    for (int ph = 0; ph < h_patches; ph++) {
      for (int pw = 0; pw < w_patches; pw++) {
        int patch_idx = ph * w_patches + pw;
        
        for (int c = 0; c < channels; c++) {
          for (int kh_idx = 0; kh_idx < kh; kh_idx++) {
            for (int kw_idx = 0; kw_idx < kw; kw_idx++) {
              int h = ph * sh + kh_idx * dh;
              int w = pw * sw + kw_idx * dw;
              
              if (h >= 0 && h < h_out && w >= 0 && w < w_out) {
                int in_idx = (b * patch_size + (c * kh + kh_idx) * kw + kw_idx) * num_patches + patch_idx;
                int out_idx = ((b * channels + c) * h_out + h) * w_out + w;
                
                out_data[out_idx] += in_data[in_idx];
              }
            }
          }
        }
      }
    }
  }
}