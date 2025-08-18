// nx_c_strides.h
#pragma once
#include "nx_c_shared.h"

// ---- Shape / stride access (OCaml arrays of ints) ----
static inline int nx_ndim(value v_shape) { return Wosize_val(v_shape); }
static inline int nx_shape_at(value v_shape, int d) {
  return Int_val(Field(v_shape, d));
}
static inline int nx_stride_at(value v_strides, int d) {
  return Int_val(Field(v_strides, d));
}

// Product of batch dims (..., m, n)
static inline int nx_batch_size(value v_shape) {
  int ndim = nx_ndim(v_shape);
  int bs = 1;
  for (int d = 0; d < ndim - 2; ++d) bs *= nx_shape_at(v_shape, d);
  return bs;
}

// Decompose a flat batch index b across all batch dims and compute element
// offset (in elements)
static inline size_t nx_batch_offset_elems(int b, value v_shape,
                                           value v_strides) {
  int ndim = nx_ndim(v_shape);
  size_t off = 0;
  // walk batch dims from last to first (excluding the last two matrix dims)
  for (int d = ndim - 3; d >= 0; --d) {
    int dim = nx_shape_at(v_shape, d);
    int idx = b % dim;
    b /= dim;
    off += (size_t)idx * (size_t)nx_stride_at(v_strides, d);
  }
  return off;
}

// Typed pack/unpack between an arbitrary-strided matrix view and a contiguous
// buffer. The view's last two dims are rows (ndim-2) and cols (ndim-1).
static inline void nx_pack_f32(float *dst, const float *base, int rows,
                               int cols, int srow, int scol) {
  for (int i = 0; i < rows; ++i) {
    const float *src_row = base + (size_t)i * srow;
    float *dst_row = dst + (size_t)i * cols;
    for (int j = 0; j < cols; ++j) dst_row[j] = src_row[(size_t)j * scol];
  }
}
static inline void nx_unpack_f32(float *base, const float *src, int rows,
                                 int cols, int srow, int scol) {
  for (int i = 0; i < rows; ++i) {
    float *dst_row = base + (size_t)i * srow;
    const float *src_row = src + (size_t)i * cols;
    for (int j = 0; j < cols; ++j) dst_row[(size_t)j * scol] = src_row[j];
  }
}
static inline void nx_pack_f64(double *dst, const double *base, int rows,
                               int cols, int srow, int scol) {
  for (int i = 0; i < rows; ++i) {
    const double *src_row = base + (size_t)i * srow;
    double *dst_row = dst + (size_t)i * cols;
    for (int j = 0; j < cols; ++j) dst_row[j] = src_row[(size_t)j * scol];
  }
}
static inline void nx_unpack_f64(double *base, const double *src, int rows,
                                 int cols, int srow, int scol) {
  for (int i = 0; i < rows; ++i) {
    double *dst_row = base + (size_t)i * srow;
    const double *src_row = src + (size_t)i * cols;
    for (int j = 0; j < cols; ++j) dst_row[(size_t)j * scol] = src_row[j];
  }
}
static inline void nx_pack_c32(c32_t *dst, const c32_t *base, int rows,
                               int cols, int srow, int scol) {
  for (int i = 0; i < rows; ++i) {
    const c32_t *src_row = base + (size_t)i * srow;
    c32_t *dst_row = dst + (size_t)i * cols;
    for (int j = 0; j < cols; ++j) dst_row[j] = src_row[(size_t)j * scol];
  }
}
static inline void nx_unpack_c32(c32_t *base, const c32_t *src, int rows,
                                 int cols, int srow, int scol) {
  for (int i = 0; i < rows; ++i) {
    c32_t *dst_row = base + (size_t)i * srow;
    const c32_t *src_row = src + (size_t)i * cols;
    for (int j = 0; j < cols; ++j) dst_row[(size_t)j * scol] = src_row[j];
  }
}
static inline void nx_pack_c64(c64_t *dst, const c64_t *base, int rows,
                               int cols, int srow, int scol) {
  for (int i = 0; i < rows; ++i) {
    const c64_t *src_row = base + (size_t)i * srow;
    c64_t *dst_row = dst + (size_t)i * cols;
    for (int j = 0; j < cols; ++j) dst_row[j] = src_row[(size_t)j * scol];
  }
}
static inline void nx_unpack_c64(c64_t *base, const c64_t *src, int rows,
                                 int cols, int srow, int scol) {
  for (int i = 0; i < rows; ++i) {
    c64_t *dst_row = base + (size_t)i * srow;
    const c64_t *src_row = src + (size_t)i * cols;
    for (int j = 0; j < cols; ++j) dst_row[(size_t)j * scol] = src_row[j];
  }
}

// Dtype-scaled eps for tolerances
static inline float nx_eps32(void) { return 1.1920929e-7f; }
static inline double nx_eps64(void) { return 2.220446049250313e-16; }