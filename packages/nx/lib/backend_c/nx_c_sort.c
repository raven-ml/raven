/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

// Sort/search kernels for nx C backend: argmax/argmin/sort/argsort.

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "nx_c_shared.h"

#if defined(_OPENMP)
#define NX_ARG_PAR_THRESHOLD 4096
#define NX_SORT_PAR_THRESHOLD 64
#else
#define NX_ARG_PAR_THRESHOLD LONG_MAX
#define NX_SORT_PAR_THRESHOLD LONG_MAX
#endif

typedef struct {
  const ndarray_t* x;
  int kind;
  long x_base;
  long x_stride_axis;
  bool descending;
} slice_sort_ctx_t;

static inline int compare_u64(uint64_t a, uint64_t b) {
  return (a > b) - (a < b);
}

static inline int compare_i64(int64_t a, int64_t b) {
  return (a > b) - (a < b);
}

static inline int compare_float_total(double a, double b) {
  if (a < b) return -1;
  if (a > b) return 1;
  if (a == b) return 0;
  bool a_nan = isnan(a);
  bool b_nan = isnan(b);
  if (a_nan && b_nan) return 0;
  if (a_nan) return -1;
  if (b_nan) return 1;
  return 0;
}

static inline bool is_nan_at(int kind, const void* data, long off) {
  switch (kind) {
    case CAML_BA_FLOAT16:
      return isnan(half_to_float(((const uint16_t*)data)[off]));
    case CAML_BA_FLOAT32:
      return isnan(((const float*)data)[off]);
    case CAML_BA_FLOAT64:
      return isnan(((const double*)data)[off]);
    case NX_BA_BFLOAT16:
      return isnan(bfloat16_to_float(((const caml_ba_bfloat16*)data)[off]));
    case NX_BA_FP8_E4M3:
      return isnan(fp8_e4m3_to_float(((const caml_ba_fp8_e4m3*)data)[off]));
    case NX_BA_FP8_E5M2:
      return isnan(fp8_e5m2_to_float(((const caml_ba_fp8_e5m2*)data)[off]));
    default:
      return false;
  }
}

static inline int compare_values_at(int kind, const void* data, long a_off,
                                    long b_off) {
  switch (kind) {
    case CAML_BA_SINT8: {
      int8_t a = ((const int8_t*)data)[a_off];
      int8_t b = ((const int8_t*)data)[b_off];
      return (a > b) - (a < b);
    }
    case CAML_BA_UINT8: {
      uint8_t a = ((const uint8_t*)data)[a_off];
      uint8_t b = ((const uint8_t*)data)[b_off];
      return (a > b) - (a < b);
    }
    case CAML_BA_SINT16: {
      int16_t a = ((const int16_t*)data)[a_off];
      int16_t b = ((const int16_t*)data)[b_off];
      return (a > b) - (a < b);
    }
    case CAML_BA_UINT16: {
      uint16_t a = ((const uint16_t*)data)[a_off];
      uint16_t b = ((const uint16_t*)data)[b_off];
      return (a > b) - (a < b);
    }
    case CAML_BA_INT32: {
      int32_t a = ((const int32_t*)data)[a_off];
      int32_t b = ((const int32_t*)data)[b_off];
      return (a > b) - (a < b);
    }
    case CAML_BA_INT64: {
      int64_t a = ((const int64_t*)data)[a_off];
      int64_t b = ((const int64_t*)data)[b_off];
      return compare_i64(a, b);
    }
    case NX_BA_UINT32: {
      caml_ba_uint32 a = ((const caml_ba_uint32*)data)[a_off];
      caml_ba_uint32 b = ((const caml_ba_uint32*)data)[b_off];
      return (a > b) - (a < b);
    }
    case NX_BA_UINT64: {
      caml_ba_uint64 a = ((const caml_ba_uint64*)data)[a_off];
      caml_ba_uint64 b = ((const caml_ba_uint64*)data)[b_off];
      return compare_u64(a, b);
    }
    case CAML_BA_CAML_INT:
    case CAML_BA_NATIVE_INT: {
      intnat a = ((const intnat*)data)[a_off];
      intnat b = ((const intnat*)data)[b_off];
      return (a > b) - (a < b);
    }
    case CAML_BA_FLOAT16: {
      float a = half_to_float(((const uint16_t*)data)[a_off]);
      float b = half_to_float(((const uint16_t*)data)[b_off]);
      return compare_float_total(a, b);
    }
    case CAML_BA_FLOAT32: {
      float a = ((const float*)data)[a_off];
      float b = ((const float*)data)[b_off];
      return compare_float_total(a, b);
    }
    case CAML_BA_FLOAT64: {
      double a = ((const double*)data)[a_off];
      double b = ((const double*)data)[b_off];
      return compare_float_total(a, b);
    }
    case CAML_BA_COMPLEX32: {
      complex32 a = ((const complex32*)data)[a_off];
      complex32 b = ((const complex32*)data)[b_off];
      int c = compare_float_total(crealf(a), crealf(b));
      if (c != 0) return c;
      return compare_float_total(cimagf(a), cimagf(b));
    }
    case CAML_BA_COMPLEX64: {
      complex64 a = ((const complex64*)data)[a_off];
      complex64 b = ((const complex64*)data)[b_off];
      int c = compare_float_total(creal(a), creal(b));
      if (c != 0) return c;
      return compare_float_total(cimag(a), cimag(b));
    }
    case NX_BA_BFLOAT16: {
      float a = bfloat16_to_float(((const caml_ba_bfloat16*)data)[a_off]);
      float b = bfloat16_to_float(((const caml_ba_bfloat16*)data)[b_off]);
      return compare_float_total(a, b);
    }
    case NX_BA_BOOL: {
      caml_ba_bool a = ((const caml_ba_bool*)data)[a_off];
      caml_ba_bool b = ((const caml_ba_bool*)data)[b_off];
      return (a > b) - (a < b);
    }
    case NX_BA_INT4: {
      int a = int4_get((const uint8_t*)data, a_off, true);
      int b = int4_get((const uint8_t*)data, b_off, true);
      return (a > b) - (a < b);
    }
    case NX_BA_UINT4: {
      int a = int4_get((const uint8_t*)data, a_off, false);
      int b = int4_get((const uint8_t*)data, b_off, false);
      return (a > b) - (a < b);
    }
    case NX_BA_FP8_E4M3: {
      float a = fp8_e4m3_to_float(((const caml_ba_fp8_e4m3*)data)[a_off]);
      float b = fp8_e4m3_to_float(((const caml_ba_fp8_e4m3*)data)[b_off]);
      return compare_float_total(a, b);
    }
    case NX_BA_FP8_E5M2: {
      float a = fp8_e5m2_to_float(((const caml_ba_fp8_e5m2*)data)[a_off]);
      float b = fp8_e5m2_to_float(((const caml_ba_fp8_e5m2*)data)[b_off]);
      return compare_float_total(a, b);
    }
    default:
      return 0;
  }
}

static inline void copy_value_at(int kind, const void* src_data, long src_off,
                                 void* dst_data, long dst_off) {
  switch (kind) {
    case CAML_BA_SINT8:
      ((int8_t*)dst_data)[dst_off] = ((const int8_t*)src_data)[src_off];
      return;
    case CAML_BA_UINT8:
      ((uint8_t*)dst_data)[dst_off] = ((const uint8_t*)src_data)[src_off];
      return;
    case CAML_BA_SINT16:
      ((int16_t*)dst_data)[dst_off] = ((const int16_t*)src_data)[src_off];
      return;
    case CAML_BA_UINT16:
      ((uint16_t*)dst_data)[dst_off] = ((const uint16_t*)src_data)[src_off];
      return;
    case CAML_BA_INT32:
      ((int32_t*)dst_data)[dst_off] = ((const int32_t*)src_data)[src_off];
      return;
    case CAML_BA_INT64:
      ((int64_t*)dst_data)[dst_off] = ((const int64_t*)src_data)[src_off];
      return;
    case NX_BA_UINT32:
      ((caml_ba_uint32*)dst_data)[dst_off] =
          ((const caml_ba_uint32*)src_data)[src_off];
      return;
    case NX_BA_UINT64:
      ((caml_ba_uint64*)dst_data)[dst_off] =
          ((const caml_ba_uint64*)src_data)[src_off];
      return;
    case CAML_BA_CAML_INT:
    case CAML_BA_NATIVE_INT:
      ((intnat*)dst_data)[dst_off] = ((const intnat*)src_data)[src_off];
      return;
    case CAML_BA_FLOAT16:
      ((uint16_t*)dst_data)[dst_off] = ((const uint16_t*)src_data)[src_off];
      return;
    case CAML_BA_FLOAT32:
      ((float*)dst_data)[dst_off] = ((const float*)src_data)[src_off];
      return;
    case CAML_BA_FLOAT64:
      ((double*)dst_data)[dst_off] = ((const double*)src_data)[src_off];
      return;
    case CAML_BA_COMPLEX32:
      ((complex32*)dst_data)[dst_off] = ((const complex32*)src_data)[src_off];
      return;
    case CAML_BA_COMPLEX64:
      ((complex64*)dst_data)[dst_off] = ((const complex64*)src_data)[src_off];
      return;
    case NX_BA_BFLOAT16:
      ((caml_ba_bfloat16*)dst_data)[dst_off] =
          ((const caml_ba_bfloat16*)src_data)[src_off];
      return;
    case NX_BA_BOOL:
      ((caml_ba_bool*)dst_data)[dst_off] =
          ((const caml_ba_bool*)src_data)[src_off];
      return;
    case NX_BA_INT4: {
      int v = int4_get((const uint8_t*)src_data, src_off, true);
      int4_set((uint8_t*)dst_data, dst_off, v, true);
      return;
    }
    case NX_BA_UINT4: {
      int v = int4_get((const uint8_t*)src_data, src_off, false);
      int4_set((uint8_t*)dst_data, dst_off, v, false);
      return;
    }
    case NX_BA_FP8_E4M3:
      ((caml_ba_fp8_e4m3*)dst_data)[dst_off] =
          ((const caml_ba_fp8_e4m3*)src_data)[src_off];
      return;
    case NX_BA_FP8_E5M2:
      ((caml_ba_fp8_e5m2*)dst_data)[dst_off] =
          ((const caml_ba_fp8_e5m2*)src_data)[src_off];
      return;
    default:
      return;
  }
}

static inline bool kind_supported_for_sort(int kind) {
  switch (kind) {
    case CAML_BA_SINT8:
    case CAML_BA_UINT8:
    case CAML_BA_SINT16:
    case CAML_BA_UINT16:
    case CAML_BA_INT32:
    case CAML_BA_INT64:
    case NX_BA_UINT32:
    case NX_BA_UINT64:
    case CAML_BA_CAML_INT:
    case CAML_BA_NATIVE_INT:
    case CAML_BA_FLOAT16:
    case CAML_BA_FLOAT32:
    case CAML_BA_FLOAT64:
    case CAML_BA_COMPLEX32:
    case CAML_BA_COMPLEX64:
    case NX_BA_BFLOAT16:
    case NX_BA_BOOL:
    case NX_BA_INT4:
    case NX_BA_UINT4:
    case NX_BA_FP8_E4M3:
    case NX_BA_FP8_E5M2:
      return true;
    default:
      return false;
  }
}

static inline long product_range(const int* shape, int start, int end) {
  long p = 1;
  for (int i = start; i < end; ++i) p *= shape[i];
  return p;
}

static inline long compute_base_offset_same_rank(const ndarray_t* arr,
                                                 const ndarray_t* ref, int axis,
                                                 long outer_idx,
                                                 long inner_idx) {
  long off = arr->offset;
  long tmp_outer = outer_idx;
  for (int d = axis - 1; d >= 0; --d) {
    int coord = (int)(tmp_outer % ref->shape[d]);
    tmp_outer /= ref->shape[d];
    off += (long)coord * arr->strides[d];
  }

  long tmp_inner = inner_idx;
  for (int d = ref->ndim - 1; d > axis; --d) {
    int coord = (int)(tmp_inner % ref->shape[d]);
    tmp_inner /= ref->shape[d];
    off += (long)coord * arr->strides[d];
  }
  return off;
}

static inline long compute_out_offset_arg(const ndarray_t* x,
                                          const ndarray_t* out, int axis,
                                          long outer_idx, long inner_idx,
                                          bool keepdims) {
  long off = out->offset;
  long tmp_outer = outer_idx;
  for (int d = axis - 1; d >= 0; --d) {
    int coord = (int)(tmp_outer % x->shape[d]);
    tmp_outer /= x->shape[d];
    off += (long)coord * out->strides[d];
  }

  long tmp_inner = inner_idx;
  for (int d = x->ndim - 1; d > axis; --d) {
    int coord = (int)(tmp_inner % x->shape[d]);
    tmp_inner /= x->shape[d];
    int out_d = keepdims ? d : (d - 1);
    off += (long)coord * out->strides[out_d];
  }
  return off;
}

static inline int compare_slice_indices(const slice_sort_ctx_t* ctx, int ia,
                                        int ib) {
  long a_off = ctx->x_base + (long)ia * ctx->x_stride_axis;
  long b_off = ctx->x_base + (long)ib * ctx->x_stride_axis;

  bool a_nan = is_nan_at(ctx->kind, ctx->x->data, a_off);
  bool b_nan = is_nan_at(ctx->kind, ctx->x->data, b_off);

  int cmp = 0;
  if (a_nan && b_nan) {
    cmp = 0;
  } else if (a_nan) {
    cmp = 1;
  } else if (b_nan) {
    cmp = -1;
  } else {
    cmp = compare_values_at(ctx->kind, ctx->x->data, a_off, b_off);
    if (ctx->descending) cmp = -cmp;
  }
  if (cmp != 0) return cmp;
  return (ia > ib) - (ia < ib);
}

static void stable_mergesort_indices(int* idx, int* tmp, int n,
                                     const slice_sort_ctx_t* ctx) {
  if (n <= 1) return;

  int* src = idx;
  int* dst = tmp;
  for (int width = 1; width < n; width <<= 1) {
    for (int left = 0; left < n; left += (width << 1)) {
      int mid = left + width;
      int right = left + (width << 1);
      if (mid > n) mid = n;
      if (right > n) right = n;

      int i = left;
      int j = mid;
      int k = left;
      while (i < mid && j < right) {
        if (compare_slice_indices(ctx, src[i], src[j]) <= 0) {
          dst[k++] = src[i++];
        } else {
          dst[k++] = src[j++];
        }
      }
      while (i < mid) dst[k++] = src[i++];
      while (j < right) dst[k++] = src[j++];
    }
    int* swap = src;
    src = dst;
    dst = swap;
  }

  if (src != idx) memcpy(idx, src, (size_t)n * sizeof(int));
}

static void arg_reduce_impl(const ndarray_t* x, ndarray_t* out, int kind,
                            int axis, bool keepdims, bool is_max) {
  long axis_size = x->shape[axis];
  long outer = product_range(x->shape, 0, axis);
  long inner = product_range(x->shape, axis + 1, x->ndim);
  long groups = outer * inner;
  if (groups == 0 || axis_size == 0) return;

  bool fast_path = is_contiguous(x) && is_contiguous(out);
  if (fast_path) {
    int32_t* restrict out_data = (int32_t*)out->data;
    const void* x_data = x->data;
    long x_stride_axis = inner;

    _Pragma(
        "omp parallel for if(groups >= NX_ARG_PAR_THRESHOLD)") for (long g = 0;
                                                                    g < groups;
                                                                    ++g) {
      long outer_idx = g / inner;
      long inner_idx = g - (outer_idx * inner);
      long base = x->offset + (outer_idx * axis_size * inner) + inner_idx;

      int best_idx = 0;
      long best_off = base;
      for (long k = 1; k < axis_size; ++k) {
        long off = base + (k * x_stride_axis);
        int cmp = compare_values_at(kind, x_data, off, best_off);
        if ((is_max && cmp > 0) || (!is_max && cmp < 0)) {
          best_idx = (int)k;
          best_off = off;
        }
      }
      out_data[out->offset + g] = (int32_t)best_idx;
    }
    return;
  }

  for (long outer_idx = 0; outer_idx < outer; ++outer_idx) {
    for (long inner_idx = 0; inner_idx < inner; ++inner_idx) {
      long x_base =
          compute_base_offset_same_rank(x, x, axis, outer_idx, inner_idx);
      long out_off =
          compute_out_offset_arg(x, out, axis, outer_idx, inner_idx, keepdims);
      long x_stride_axis = x->strides[axis];

      int best_idx = 0;
      long best_off = x_base;
      for (long k = 1; k < axis_size; ++k) {
        long off = x_base + (k * x_stride_axis);
        int cmp = compare_values_at(kind, x->data, off, best_off);
        if ((is_max && cmp > 0) || (!is_max && cmp < 0)) {
          best_idx = (int)k;
          best_off = off;
        }
      }
      ((int32_t*)out->data)[out_off] = (int32_t)best_idx;
    }
  }
}

static int sort_impl(const ndarray_t* x, ndarray_t* out, int kind, int axis,
                     bool descending, bool write_indices) {
  long axis_size = x->shape[axis];
  long outer = product_range(x->shape, 0, axis);
  long inner = product_range(x->shape, axis + 1, x->ndim);
  long groups = outer * inner;
  if (groups == 0 || axis_size == 0) return 0;
  if (axis_size > INT_MAX) return -2;

  int nthreads = 1;
#if defined(_OPENMP)
  if (groups >= NX_SORT_PAR_THRESHOLD) nthreads = omp_get_max_threads();
#endif

  int** idx_bufs = (int**)malloc((size_t)nthreads * sizeof(int*));
  int** tmp_bufs = (int**)malloc((size_t)nthreads * sizeof(int*));
  if (!idx_bufs || !tmp_bufs) {
    free(idx_bufs);
    free(tmp_bufs);
    return -1;
  }
  for (int t = 0; t < nthreads; ++t) {
    idx_bufs[t] = (int*)malloc((size_t)axis_size * sizeof(int));
    tmp_bufs[t] = (int*)malloc((size_t)axis_size * sizeof(int));
    if (!idx_bufs[t] || !tmp_bufs[t]) {
      for (int j = 0; j <= t; ++j) {
        free(idx_bufs[j]);
        free(tmp_bufs[j]);
      }
      free(idx_bufs);
      free(tmp_bufs);
      return -1;
    }
  }

  bool fast_path = is_contiguous(x) && is_contiguous(out);

  _Pragma(
      "omp parallel for if(groups >= NX_SORT_PAR_THRESHOLD)") for (long g = 0;
                                                                   g < groups;
                                                                   ++g) {
    int tid = 0;
#if defined(_OPENMP)
    tid = omp_get_thread_num();
#endif
    int* idx = idx_bufs[tid];
    int* tmp = tmp_bufs[tid];
    for (int i = 0; i < axis_size; ++i) idx[i] = i;

    long outer_idx = g / inner;
    long inner_idx = g - (outer_idx * inner);

    long x_base, out_base;
    long x_stride_axis, out_stride_axis;
    if (fast_path) {
      x_base = x->offset + (outer_idx * axis_size * inner) + inner_idx;
      out_base = out->offset + (outer_idx * axis_size * inner) + inner_idx;
      x_stride_axis = inner;
      out_stride_axis = inner;
    } else {
      x_base = compute_base_offset_same_rank(x, x, axis, outer_idx, inner_idx);
      out_base =
          compute_base_offset_same_rank(out, x, axis, outer_idx, inner_idx);
      x_stride_axis = x->strides[axis];
      out_stride_axis = out->strides[axis];
    }

    slice_sort_ctx_t ctx = {.x = x,
                            .kind = kind,
                            .x_base = x_base,
                            .x_stride_axis = x_stride_axis,
                            .descending = descending};
    stable_mergesort_indices(idx, tmp, (int)axis_size, &ctx);

    if (write_indices) {
      int32_t* out_data = (int32_t*)out->data;
      for (long k = 0; k < axis_size; ++k) {
        long dst_off = out_base + (k * out_stride_axis);
        out_data[dst_off] = (int32_t)idx[k];
      }
    } else {
      for (long k = 0; k < axis_size; ++k) {
        long src_off = x_base + ((long)idx[k] * x_stride_axis);
        long dst_off = out_base + (k * out_stride_axis);
        copy_value_at(kind, x->data, src_off, out->data, dst_off);
      }
    }
  }

  for (int t = 0; t < nthreads; ++t) {
    free(idx_bufs[t]);
    free(tmp_bufs[t]);
  }
  free(idx_bufs);
  free(tmp_bufs);
  return 0;
}

static const char* validate_axis(const ndarray_t* x, int axis, const char* op) {
  if (axis < 0 || axis >= x->ndim) {
    if (strcmp(op, "argmax") == 0) return "argmax: axis out of bounds";
    if (strcmp(op, "argmin") == 0) return "argmin: axis out of bounds";
    if (strcmp(op, "sort") == 0) return "sort: axis out of bounds";
    if (strcmp(op, "argsort") == 0) return "argsort: axis out of bounds";
    return "axis out of bounds";
  }
  return NULL;
}

static const char* validate_same_shape(const ndarray_t* a, const ndarray_t* b,
                                       const char* op) {
  if (a->ndim != b->ndim) {
    if (strcmp(op, "sort") == 0) return "sort: shape mismatch";
    if (strcmp(op, "argsort") == 0) return "argsort: shape mismatch";
    return "shape mismatch";
  }
  for (int i = 0; i < a->ndim; ++i) {
    if (a->shape[i] != b->shape[i]) {
      if (strcmp(op, "sort") == 0) return "sort: shape mismatch";
      if (strcmp(op, "argsort") == 0) return "argsort: shape mismatch";
      return "shape mismatch";
    }
  }
  return NULL;
}

static const char* validate_arg_output(const ndarray_t* x, const ndarray_t* out,
                                       int axis, bool keepdims,
                                       const char* op) {
  if (keepdims) {
    if (out->ndim != x->ndim) {
      if (strcmp(op, "argmax") == 0) return "argmax: shape mismatch";
      if (strcmp(op, "argmin") == 0) return "argmin: shape mismatch";
      return "shape mismatch";
    }
    for (int d = 0; d < x->ndim; ++d) {
      int expected = (d == axis) ? 1 : x->shape[d];
      if (out->shape[d] != expected) {
        if (strcmp(op, "argmax") == 0) return "argmax: shape mismatch";
        if (strcmp(op, "argmin") == 0) return "argmin: shape mismatch";
        return "shape mismatch";
      }
    }
  } else {
    if (out->ndim != x->ndim - 1) {
      if (strcmp(op, "argmax") == 0) return "argmax: shape mismatch";
      if (strcmp(op, "argmin") == 0) return "argmin: shape mismatch";
      return "shape mismatch";
    }
    for (int d = 0; d < x->ndim; ++d) {
      if (d < axis) {
        if (out->shape[d] != x->shape[d]) {
          if (strcmp(op, "argmax") == 0) return "argmax: shape mismatch";
          if (strcmp(op, "argmin") == 0) return "argmin: shape mismatch";
          return "shape mismatch";
        }
      } else if (d > axis) {
        if (out->shape[d - 1] != x->shape[d]) {
          if (strcmp(op, "argmax") == 0) return "argmax: shape mismatch";
          if (strcmp(op, "argmin") == 0) return "argmin: shape mismatch";
          return "shape mismatch";
        }
      }
    }
  }
  return NULL;
}

CAMLprim value caml_nx_argmax(value v_x, value v_out, value v_axis,
                              value v_keepdims) {
  CAMLparam4(v_x, v_out, v_axis, v_keepdims);

  ndarray_t x = extract_ndarray(v_x);
  ndarray_t out = extract_ndarray(v_out);
  const char* err = NULL;
  int axis = Int_val(v_axis);
  bool keepdims = Bool_val(v_keepdims);

  int kind = nx_buffer_get_kind(Caml_ba_array_val(Field(v_x, FFI_TENSOR_DATA)));
  int out_kind =
      nx_buffer_get_kind(Caml_ba_array_val(Field(v_out, FFI_TENSOR_DATA)));

  err = validate_axis(&x, axis, "argmax");
  if (err) goto fail;
  err = validate_arg_output(&x, &out, axis, keepdims, "argmax");
  if (err) goto fail;
  if (out_kind != CAML_BA_INT32) {
    err = "argmax: output must be int32";
    goto fail;
  }
  if (!kind_supported_for_sort(kind)) {
    err = "argmax: unsupported dtype";
    goto fail;
  }
  if (x.shape[axis] > INT_MAX || x.shape[axis] > INT32_MAX) {
    err = "argmax: axis too large";
    goto fail;
  }

  caml_enter_blocking_section();
  arg_reduce_impl(&x, &out, kind, axis, keepdims, true);
  caml_leave_blocking_section();

  cleanup_ndarray(&x);
  cleanup_ndarray(&out);
  CAMLreturn(Val_unit);

fail:
  cleanup_ndarray(&x);
  cleanup_ndarray(&out);
  caml_failwith(err);
}

CAMLprim value caml_nx_argmin(value v_x, value v_out, value v_axis,
                              value v_keepdims) {
  CAMLparam4(v_x, v_out, v_axis, v_keepdims);

  ndarray_t x = extract_ndarray(v_x);
  ndarray_t out = extract_ndarray(v_out);
  const char* err = NULL;
  int axis = Int_val(v_axis);
  bool keepdims = Bool_val(v_keepdims);

  int kind = nx_buffer_get_kind(Caml_ba_array_val(Field(v_x, FFI_TENSOR_DATA)));
  int out_kind =
      nx_buffer_get_kind(Caml_ba_array_val(Field(v_out, FFI_TENSOR_DATA)));

  err = validate_axis(&x, axis, "argmin");
  if (err) goto fail;
  err = validate_arg_output(&x, &out, axis, keepdims, "argmin");
  if (err) goto fail;
  if (out_kind != CAML_BA_INT32) {
    err = "argmin: output must be int32";
    goto fail;
  }
  if (!kind_supported_for_sort(kind)) {
    err = "argmin: unsupported dtype";
    goto fail;
  }
  if (x.shape[axis] > INT_MAX || x.shape[axis] > INT32_MAX) {
    err = "argmin: axis too large";
    goto fail;
  }

  caml_enter_blocking_section();
  arg_reduce_impl(&x, &out, kind, axis, keepdims, false);
  caml_leave_blocking_section();

  cleanup_ndarray(&x);
  cleanup_ndarray(&out);
  CAMLreturn(Val_unit);

fail:
  cleanup_ndarray(&x);
  cleanup_ndarray(&out);
  caml_failwith(err);
}

CAMLprim value caml_nx_sort(value v_x, value v_out, value v_axis,
                            value v_descending) {
  CAMLparam4(v_x, v_out, v_axis, v_descending);

  ndarray_t x = extract_ndarray(v_x);
  ndarray_t out = extract_ndarray(v_out);
  const char* err = NULL;
  int status = 0;
  int axis = Int_val(v_axis);
  bool descending = Bool_val(v_descending);

  int kind = nx_buffer_get_kind(Caml_ba_array_val(Field(v_x, FFI_TENSOR_DATA)));
  int out_kind =
      nx_buffer_get_kind(Caml_ba_array_val(Field(v_out, FFI_TENSOR_DATA)));

  err = validate_axis(&x, axis, "sort");
  if (err) goto fail;
  err = validate_same_shape(&x, &out, "sort");
  if (err) goto fail;
  if (kind != out_kind) {
    err = "sort: dtype mismatch";
    goto fail;
  }
  if (!kind_supported_for_sort(kind)) {
    err = "sort: unsupported dtype";
    goto fail;
  }
  if (x.shape[axis] > INT_MAX) {
    err = "sort: axis too large";
    goto fail;
  }

  caml_enter_blocking_section();
  status = sort_impl(&x, &out, kind, axis, descending, false);
  caml_leave_blocking_section();
  if (status == -1) {
    err = "sort: allocation failed";
    goto fail;
  }
  if (status == -2) {
    err = "sort: axis too large";
    goto fail;
  }

  cleanup_ndarray(&x);
  cleanup_ndarray(&out);
  CAMLreturn(Val_unit);

fail:
  cleanup_ndarray(&x);
  cleanup_ndarray(&out);
  caml_failwith(err);
}

CAMLprim value caml_nx_argsort(value v_x, value v_out, value v_axis,
                               value v_descending) {
  CAMLparam4(v_x, v_out, v_axis, v_descending);

  ndarray_t x = extract_ndarray(v_x);
  ndarray_t out = extract_ndarray(v_out);
  const char* err = NULL;
  int status = 0;
  int axis = Int_val(v_axis);
  bool descending = Bool_val(v_descending);

  int kind = nx_buffer_get_kind(Caml_ba_array_val(Field(v_x, FFI_TENSOR_DATA)));
  int out_kind =
      nx_buffer_get_kind(Caml_ba_array_val(Field(v_out, FFI_TENSOR_DATA)));

  err = validate_axis(&x, axis, "argsort");
  if (err) goto fail;
  err = validate_same_shape(&x, &out, "argsort");
  if (err) goto fail;
  if (out_kind != CAML_BA_INT32) {
    err = "argsort: output must be int32";
    goto fail;
  }
  if (!kind_supported_for_sort(kind)) {
    err = "argsort: unsupported dtype";
    goto fail;
  }
  if (x.shape[axis] > INT_MAX || x.shape[axis] > INT32_MAX) {
    err = "argsort: axis too large";
    goto fail;
  }

  caml_enter_blocking_section();
  status = sort_impl(&x, &out, kind, axis, descending, true);
  caml_leave_blocking_section();
  if (status == -1) {
    err = "argsort: allocation failed";
    goto fail;
  }
  if (status == -2) {
    err = "argsort: axis too large";
    goto fail;
  }

  cleanup_ndarray(&x);
  cleanup_ndarray(&out);
  CAMLreturn(Val_unit);

fail:
  cleanup_ndarray(&x);
  cleanup_ndarray(&out);
  caml_failwith(err);
}
