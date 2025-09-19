#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/threads.h>
#include <complex.h>

#include "nx_c_shared.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper to validate and normalize axes
static int validate_and_normalize_axes(int *axes, int num_axes, int ndim) {
  // Normalize negative axes and validate range
  for (int i = 0; i < num_axes; i++) {
    if (axes[i] < 0) axes[i] += ndim;
    if (axes[i] < 0 || axes[i] >= ndim) {
      return 0;  // Invalid axis
    }
    // Check for duplicates
    for (int j = 0; j < i; j++) {
      if (axes[i] == axes[j]) {
        return 0;  // Duplicate axis
      }
    }
  }
  return 1;  // Valid
}

// Helper function to compute twiddle factors
static inline void compute_twiddle(double angle, double *cos_val,
                                   double *sin_val) {
  *cos_val = cos(angle);
  *sin_val = sin(angle);
}

// Bit reversal for FFT
static inline size_t bit_reverse(size_t x, int log2n) {
  size_t result = 0;
  for (int i = 0; i < log2n; i++) {
    result = (result << 1) | (x & 1);
    x >>= 1;
  }
  return result;
}

// Macro for generating 1D DFT for precision types (double/float)
#define DFT_1D_PREC(PREC, SUFFIX, COS_FUNC, SIN_FUNC)                          \
  static void dft_##SUFFIX(PREC *data_re, PREC *data_im, size_t n, int stride, \
                           size_t offset, bool inverse) {                      \
    if (n == 0) return;                                                        \
    PREC *temp_re = (PREC *)malloc(n * sizeof(PREC));                          \
    PREC *temp_im = (PREC *)malloc(n * sizeof(PREC));                          \
    if (!temp_re || !temp_im) {                                                \
      free(temp_re);                                                           \
      free(temp_im);                                                           \
      return;                                                                  \
    }                                                                          \
    PREC sign = inverse ? (PREC)1.0 : (PREC) - 1.0;                            \
    PREC two_pi_n = (PREC)2.0 * (PREC)M_PI / n;                                \
    for (size_t i = 0; i < n; i++) {                                           \
      temp_re[i] = data_re[offset + i * stride];                               \
      temp_im[i] = data_im[offset + i * stride];                               \
    }                                                                          \
    for (size_t k = 0; k < n; k++) {                                           \
      PREC sum_re = (PREC)0.0;                                                 \
      PREC sum_im = (PREC)0.0;                                                 \
      PREC angle_step = sign * two_pi_n * (PREC)k;                             \
      for (size_t j = 0; j < n; j++) {                                         \
        PREC angle = angle_step * (PREC)j;                                     \
        PREC w_re = COS_FUNC(angle);                                           \
        PREC w_im = SIN_FUNC(angle);                                           \
        sum_re += temp_re[j] * w_re - temp_im[j] * w_im;                       \
        sum_im += temp_re[j] * w_im + temp_im[j] * w_re;                       \
      }                                                                        \
      /* Normalization is handled by the frontend, not here */                 \
      data_re[offset + k * stride] = sum_re;                                   \
      data_im[offset + k * stride] = sum_im;                                   \
    }                                                                          \
    free(temp_re);                                                             \
    free(temp_im);                                                             \
  }

// Macro for generating 1D FFT for precision types (double/float)
#define FFT_1D_PREC(PREC, SUFFIX, COS_FUNC, SIN_FUNC)                    \
  static void fft_1d_##SUFFIX(PREC *data_re, PREC *data_im, size_t n,    \
                              int stride, size_t offset, bool inverse) { \
    if (n == 0) return;                                                  \
    if (n <= 1) return;                                                  \
    int log2n = 0;                                                       \
    size_t temp = n;                                                     \
    bool is_power_of_2 = true;                                           \
    while (temp > 1) {                                                   \
      if (temp & 1) {                                                    \
        is_power_of_2 = false;                                           \
        break;                                                           \
      }                                                                  \
      temp >>= 1;                                                        \
      log2n++;                                                           \
    }                                                                    \
    if (!is_power_of_2) {                                                \
      dft_##SUFFIX(data_re, data_im, n, stride, offset, inverse);        \
      return;                                                            \
    }                                                                    \
    for (size_t i = 0; i < n; i++) {                                     \
      size_t j = bit_reverse(i, log2n);                                  \
      if (i < j) {                                                       \
        size_t idx_i = offset + i * stride;                              \
        size_t idx_j = offset + j * stride;                              \
        PREC temp_re = data_re[idx_i];                                   \
        PREC temp_im = data_im[idx_i];                                   \
        data_re[idx_i] = data_re[idx_j];                                 \
        data_im[idx_i] = data_im[idx_j];                                 \
        data_re[idx_j] = temp_re;                                        \
        data_im[idx_j] = temp_im;                                        \
      }                                                                  \
    }                                                                    \
    PREC sign = inverse ? (PREC)1.0 : (PREC) - 1.0;                      \
    for (size_t stage_size = 2; stage_size <= n; stage_size *= 2) {      \
      size_t half_stage = stage_size / 2;                                \
      PREC angle_step = sign * (PREC)2.0 * (PREC)M_PI / stage_size;      \
      for (size_t k = 0; k < n; k += stage_size) {                       \
        for (size_t j = 0; j < half_stage; j++) {                        \
          PREC angle = angle_step * (PREC)j;                             \
          PREC twiddle_re = COS_FUNC(angle);                             \
          PREC twiddle_im = SIN_FUNC(angle);                             \
          size_t idx1 = offset + (k + j) * stride;                       \
          size_t idx2 = offset + (k + j + half_stage) * stride;          \
          PREC a_re = data_re[idx1];                                     \
          PREC a_im = data_im[idx1];                                     \
          PREC b_re = data_re[idx2];                                     \
          PREC b_im = data_im[idx2];                                     \
          PREC b_twiddle_re = b_re * twiddle_re - b_im * twiddle_im;     \
          PREC b_twiddle_im = b_re * twiddle_im + b_im * twiddle_re;     \
          data_re[idx1] = a_re + b_twiddle_re;                           \
          data_im[idx1] = a_im + b_twiddle_im;                           \
          data_re[idx2] = a_re - b_twiddle_re;                           \
          data_im[idx2] = a_im - b_twiddle_im;                           \
        }                                                                \
      }                                                                  \
    }                                                                    \
    /* Normalization is handled by the frontend, not here */             \
  }

// Macro for generating multi-dimensional FFT for complex types
#define FFT_MULTI_PREC(PREC, SUFFIX)                                       \
  static void fft_multi_##SUFFIX(ndarray_t *data, int *axes, int num_axes, \
                                 bool inverse) {                           \
    int ndim = data->ndim;                                                 \
    const int *shape = data->shape;                                        \
    long strides_buf[MAX_NDIM];                                            \
    for (int i = 0; i < ndim; i++) {                                       \
      strides_buf[i] = data->strides[i];                                   \
    }                                                                      \
    const long *strides = strides_buf;                                     \
    if (num_axes == 0) return;                                             \
    size_t max_n = 0;                                                      \
    for (int ax_idx = 0; ax_idx < num_axes; ax_idx++) {                    \
      int axis = axes[ax_idx];                                             \
      if (axis < 0) axis += ndim;                                          \
      size_t n = shape[axis];                                              \
      if (n > max_n) max_n = n;                                            \
    }                                                                      \
    PREC *temp_re = (PREC *)malloc(max_n * sizeof(PREC));                  \
    PREC *temp_im = (PREC *)malloc(max_n * sizeof(PREC));                  \
    if (!temp_re || !temp_im) {                                            \
      free(temp_re);                                                       \
      free(temp_im);                                                       \
      return;                                                              \
    }                                                                      \
    for (int ax_idx = 0; ax_idx < num_axes; ax_idx++) {                    \
      int axis = axes[ax_idx];                                             \
      if (axis < 0) axis += ndim;                                          \
      size_t n = shape[axis];                                              \
      if (n <= 1) continue;                                                \
      long axis_stride = strides[axis];                                    \
      size_t total_elements = total_elements_safe(data);                   \
      size_t num_ffts = total_elements / n;                                \
      for (size_t fft_idx = 0; fft_idx < num_ffts; fft_idx++) {            \
        long offset = 0;                                                   \
        long temp = (long)fft_idx;                                         \
        for (int d = ndim - 1; d >= 0; d--) {                              \
          if (d != axis) {                                                 \
            long coord = temp % shape[d];                                  \
            temp /= shape[d];                                              \
            offset += coord * strides[d];                                  \
          }                                                                \
        }                                                                  \
        /* Extract complex data properly */                                \
        PREC _Complex *complex_data = (PREC _Complex *)data->data;         \
        for (size_t i = 0; i < n; i++) {                                   \
          long idx = data->offset + offset + (long)i * axis_stride;        \
          PREC _Complex val = complex_data[idx];                           \
          temp_re[i] = __real__(val);                                      \
          temp_im[i] = __imag__(val);                                      \
        }                                                                  \
        fft_1d_##SUFFIX(temp_re, temp_im, n, 1, 0, inverse);               \
        /* Store complex data properly */                                  \
        for (size_t i = 0; i < n; i++) {                                   \
          long idx = data->offset + offset + (long)i * axis_stride;        \
          complex_data[idx] = temp_re[i] + temp_im[i] * I;                 \
        }                                                                  \
      }                                                                    \
    }                                                                      \
    free(temp_re);                                                         \
    free(temp_im);                                                         \
  }

// Instantiate for complex64 (double precision)
DFT_1D_PREC(double, complex64, cos, sin)
FFT_1D_PREC(double, complex64, cos, sin)
FFT_MULTI_PREC(double, complex64)

// Instantiate for complex32 (single precision)
DFT_1D_PREC(float, complex32, cosf, sinf)
FFT_1D_PREC(float, complex32, cosf, sinf)
FFT_MULTI_PREC(float, complex32)

// Macro for low-precision DFT/FFT (converts to float for computation)
#define LOW_PREC_DFT_KERNEL(T, SUFFIX, TO_FLOAT, FROM_FLOAT)             \
  static void dft_##SUFFIX(T *data, size_t n, int stride, size_t offset, \
                           bool inverse) {                               \
    if (n == 0) return;                                                  \
    float *temp_re = (float *)malloc(n * sizeof(float));                 \
    float *temp_im = (float *)malloc(n * sizeof(float));                 \
    if (!temp_re || !temp_im) {                                          \
      free(temp_re);                                                     \
      free(temp_im);                                                     \
      return;                                                            \
    }                                                                    \
    caml_ba_complex16 *d = (caml_ba_complex16 *)data;                    \
    for (size_t i = 0; i < n; i++) {                                     \
      long idx = offset + (long)i * stride;                              \
      temp_re[i] = TO_FLOAT(d[idx].re);                                  \
      temp_im[i] = TO_FLOAT(d[idx].im);                                  \
    }                                                                    \
    dft_complex32(temp_re, temp_im, n, 1, 0, inverse);                   \
    for (size_t i = 0; i < n; i++) {                                     \
      long idx = offset + (long)i * stride;                              \
      d[idx].re = FROM_FLOAT(temp_re[i]);                                \
      d[idx].im = FROM_FLOAT(temp_im[i]);                                \
    }                                                                    \
    free(temp_re);                                                       \
    free(temp_im);                                                       \
  }

#define LOW_PREC_FFT_1D_KERNEL(T, SUFFIX, TO_FLOAT, FROM_FLOAT)             \
  static void fft_1d_##SUFFIX(T *data, size_t n, int stride, size_t offset, \
                              bool inverse) {                               \
    if (n == 0) return;                                                     \
    if (n <= 1) return;                                                     \
    int log2n = 0;                                                          \
    size_t temp_n = n;                                                      \
    bool is_power_of_2 = true;                                              \
    while (temp_n > 1) {                                                    \
      if (temp_n & 1) is_power_of_2 = false;                                \
      temp_n >>= 1;                                                         \
      log2n++;                                                              \
    }                                                                       \
    float *temp_re = (float *)malloc(n * sizeof(float));                    \
    float *temp_im = (float *)malloc(n * sizeof(float));                    \
    if (!temp_re || !temp_im) {                                             \
      free(temp_re);                                                        \
      free(temp_im);                                                        \
      return;                                                               \
    }                                                                       \
    caml_ba_complex16 *d = (caml_ba_complex16 *)data;                       \
    for (size_t i = 0; i < n; i++) {                                        \
      long idx = offset + (long)i * stride;                                 \
      temp_re[i] = TO_FLOAT(d[idx].re);                                     \
      temp_im[i] = TO_FLOAT(d[idx].im);                                     \
    }                                                                       \
    if (!is_power_of_2) {                                                   \
      dft_complex32(temp_re, temp_im, n, 1, 0, inverse);                    \
    } else {                                                                \
      fft_1d_complex32(temp_re, temp_im, n, 1, 0, inverse);                 \
    }                                                                       \
    for (size_t i = 0; i < n; i++) {                                        \
      long idx = offset + (long)i * stride;                                 \
      d[idx].re = FROM_FLOAT(temp_re[i]);                                   \
      d[idx].im = FROM_FLOAT(temp_im[i]);                                   \
    }                                                                       \
    free(temp_re);                                                          \
    free(temp_im);                                                          \
  }

#define LOW_PREC_FFT_MULTI_KERNEL(SUFFIX)                                  \
  static void fft_multi_##SUFFIX(ndarray_t *data, int *axes, int num_axes, \
                                 bool inverse) {                           \
    int ndim = data->ndim;                                                 \
    const int *shape = data->shape;                                        \
    long strides_buf[MAX_NDIM];                                            \
    for (int i = 0; i < ndim; i++) {                                       \
      strides_buf[i] = data->strides[i];                                   \
    }                                                                      \
    const long *strides = strides_buf;                                     \
    if (num_axes == 0) return;                                             \
    size_t max_n = 0;                                                      \
    for (int ax_idx = 0; ax_idx < num_axes; ax_idx++) {                    \
      int axis = axes[ax_idx];                                             \
      if (axis < 0) axis += ndim;                                          \
      size_t n = shape[axis];                                              \
      if (n > max_n) max_n = n;                                            \
    }                                                                      \
    if (max_n == 0) return;                                                \
    for (int ax_idx = 0; ax_idx < num_axes; ax_idx++) {                    \
      int axis = axes[ax_idx];                                             \
      if (axis < 0) axis += ndim;                                          \
      size_t n = shape[axis];                                              \
      if (n <= 1) continue;                                                \
      long axis_stride = strides[axis];                                    \
      size_t total_elements = total_elements_safe(data);                   \
      size_t num_ffts = total_elements / n;                                \
      for (size_t fft_idx = 0; fft_idx < num_ffts; fft_idx++) {            \
        long offset = 0;                                                   \
        long temp = (long)fft_idx;                                         \
        for (int d = ndim - 1; d >= 0; d--) {                              \
          if (d != axis) {                                                 \
            long coord = temp % shape[d];                                  \
            temp /= shape[d];                                              \
            offset += coord * strides[d];                                  \
          }                                                                \
        }                                                                  \
        fft_1d_##SUFFIX(data->data, n, axis_stride, data->offset + offset, \
                        inverse);                                          \
      }                                                                    \
    }                                                                      \
  }

// For complex16 (two uint16_t, float16 re/im)
LOW_PREC_DFT_KERNEL(caml_ba_complex16, complex16, half_to_float, float_to_half)
LOW_PREC_FFT_1D_KERNEL(caml_ba_complex16, complex16, half_to_float,
                       float_to_half)
LOW_PREC_FFT_MULTI_KERNEL(complex16)

// Macro for RFFT multi (real to complex)
#define RFFT_MULTI_PREC(REAL_PREC, COMP_PREC, REAL_SUFFIX, COMP_SUFFIX)        \
  static void rfft_multi_##REAL_SUFFIX(                                        \
      const ndarray_t *input, ndarray_t *output, int *axes, int num_axes) {    \
    if (num_axes == 0) return;                                                 \
    int ndim = input->ndim;                                                    \
    int rfft_axis = axes[num_axes - 1];                                        \
    size_t n = input->shape[rfft_axis];                                        \
    if (n == 0) return;                                                        \
    size_t n_out = output->shape[rfft_axis];                                   \
    if (n_out != n / 2 + 1) {                                                  \
      /* Shape check failed - output won't be valid */                         \
      return;                                                                  \
    }                                                                          \
    size_t total_elements = total_elements_safe(input);                        \
    if (num_axes == 1) {                                                       \
      size_t num_ffts = total_elements / n;                                    \
      COMP_PREC *temp_re = (COMP_PREC *)malloc(n * sizeof(COMP_PREC));         \
      COMP_PREC *temp_im = (COMP_PREC *)malloc(n * sizeof(COMP_PREC));         \
      if (!temp_re || !temp_im) caml_failwith("rfft: allocation failed");      \
      for (size_t fft_idx = 0; fft_idx < num_ffts; fft_idx++) {                \
        long offset_in = 0, offset_out = 0;                                    \
        long temp = (long)fft_idx;                                             \
        for (int d = ndim - 1; d >= 0; d--) {                                  \
          if (d != rfft_axis) {                                                \
            long coord = temp % input->shape[d];                               \
            temp /= input->shape[d];                                           \
            offset_in += coord * input->strides[d];                            \
            offset_out += coord * output->strides[d];                          \
          }                                                                    \
        }                                                                      \
        for (size_t i = 0; i < n; i++) {                                       \
          long idx =                                                           \
              input->offset + offset_in + (long)i * input->strides[rfft_axis]; \
          temp_re[i] = ((REAL_PREC *)input->data)[idx];                        \
          temp_im[i] = (COMP_PREC)0.0;                                         \
        }                                                                      \
        fft_1d_##COMP_SUFFIX(temp_re, temp_im, n, 1, 0, false);                \
        for (size_t i = 0; i < n_out; i++) {                                   \
          long idx = output->offset + offset_out +                             \
                     (long)i * output->strides[rfft_axis];                     \
          COMP_PREC _Complex *complex_out =                                    \
              (COMP_PREC _Complex *)output->data;                              \
          complex_out[idx] = temp_re[i] + temp_im[i] * I;                      \
        }                                                                      \
      }                                                                        \
      free(temp_re);                                                           \
      free(temp_im);                                                           \
    } else {                                                                   \
      size_t inter_size = total_elements_safe(output);                         \
      COMP_PREC *inter_data =                                                  \
          (COMP_PREC *)calloc(2 * inter_size, sizeof(COMP_PREC));              \
      if (!inter_data) caml_failwith("rfft: allocation failed");               \
      long inter_strides[MAX_NDIM];                                            \
      inter_strides[ndim - 1] = 2;  /* 2 COMP_PREC per complex number */       \
      for (int d = ndim - 2; d >= 0; d--) {                                    \
        inter_strides[d] = inter_strides[d + 1] * output->shape[d + 1];        \
      }                                                                        \
      size_t num_1d = inter_size / n_out;                                      \
      COMP_PREC *temp_re = (COMP_PREC *)malloc(n * sizeof(COMP_PREC));         \
      COMP_PREC *temp_im = (COMP_PREC *)malloc(n * sizeof(COMP_PREC));         \
      if (!temp_re || !temp_im) {                                              \
        free(inter_data);                                                      \
        caml_failwith("rfft: allocation failed");                              \
      }                                                                        \
      for (size_t idx = 0; idx < num_1d; idx++) {                              \
        long offset_in = 0, offset_inter = 0;                                  \
        long temp = (long)idx;                                                 \
        for (int d = ndim - 1; d >= 0; d--) {                                  \
          if (d != rfft_axis) {                                                \
            long coord = temp % input->shape[d];                               \
            temp /= input->shape[d];                                           \
            offset_in += coord * input->strides[d];                            \
            offset_inter += coord * inter_strides[d];                          \
          }                                                                    \
        }                                                                      \
        for (size_t i = 0; i < n; i++) {                                       \
          long idx =                                                           \
              input->offset + offset_in + (long)i * input->strides[rfft_axis]; \
          temp_re[i] = ((REAL_PREC *)input->data)[idx];                        \
          temp_im[i] = (COMP_PREC)0.0;                                         \
        }                                                                      \
        fft_1d_##COMP_SUFFIX(temp_re, temp_im, n, 1, 0, false);                \
        for (size_t i = 0; i < n_out; i++) {                                   \
          long idx_out = offset_inter + (long)i * inter_strides[rfft_axis];    \
          inter_data[idx_out] = temp_re[i];                                    \
          inter_data[idx_out + 1] = temp_im[i];                                \
        }                                                                      \
      }                                                                        \
      free(temp_re);                                                           \
      free(temp_im);                                                           \
      int inter_strides_int[MAX_NDIM];                                         \
      for (int i = 0; i < ndim; i++) {                                         \
        /* Convert strides from COMP_PREC elements to complex elements */      \
        inter_strides_int[i] = (int)(inter_strides[i] / 2);                    \
      }                                                                        \
      ndarray_t inter_array = {.data = inter_data,                             \
                               .shape = output->shape,                         \
                               .strides = inter_strides_int,                   \
                               .ndim = ndim,                                   \
                               .offset = 0};                                   \
      fft_multi_##COMP_SUFFIX(&inter_array, axes, num_axes - 1, false);        \
      nd_single_iterator_t it;                                                 \
      nd_iterator_init(&it, &inter_array);                                     \
      do {                                                                     \
        long inter_off, out_off;                                               \
        nd_iterator_get_offset(&it, &inter_off);                               \
        out_off = 0;                                                           \
        for (int d = 0; d < ndim; d++) {                                       \
          out_off += it.coords[d] * output->strides[d];                        \
        }                                                                      \
        COMP_PREC _Complex *complex_out = (COMP_PREC _Complex *)output->data;  \
        /* inter_off is in complex units, need to convert to COMP_PREC units */\
        long data_idx = inter_off * 2;                                         \
        complex_out[output->offset + out_off] =                                \
            inter_data[data_idx] + inter_data[data_idx + 1] * I;               \
      } while (nd_single_iterator_next(&it));                                  \
      nd_single_iterator_destroy(&it);                                         \
      free(inter_data);                                                        \
    }                                                                          \
  }

// Macro for IRFFT multi (complex to real)
#define IRFFT_MULTI_PREC(COMP_PREC, REAL_PREC, COMP_SUFFIX, REAL_SUFFIX)       \
  static void irfft_multi_##COMP_SUFFIX(const ndarray_t *input,                \
                                        ndarray_t *output, int *axes,          \
                                        int num_axes, int last_size) {         \
    if (num_axes == 0) return;                                                 \
    int ndim = input->ndim;                                                    \
    int irfft_axis = axes[num_axes - 1];                                       \
    size_t n_in = input->shape[irfft_axis];                                    \
    size_t n_out = (size_t)last_size;                                          \
    /* DEBUG: Check sizes */                                                   \
    if (0)                                                                     \
      printf("IRFFT DEBUG: n_in=%zu, n_out=%zu, last_size=%d\\n", n_in, n_out, \
             last_size);                                                       \
    if (n_out == 0) return;                                                    \
    size_t freq_len = n_out / 2 + 1;                                           \
    if (freq_len > n_out) freq_len = n_out;                                    \
    size_t copy_len = n_in < freq_len ? n_in : freq_len;                       \
    size_t total_elements = total_elements_safe(input);                        \
    if (num_axes == 1) {                                                       \
      size_t num_ffts = total_elements / n_in;                                 \
      COMP_PREC *temp_re = (COMP_PREC *)malloc(n_out * sizeof(COMP_PREC));     \
      COMP_PREC *temp_im = (COMP_PREC *)malloc(n_out * sizeof(COMP_PREC));     \
      if (!temp_re || !temp_im) caml_failwith("irfft: allocation failed");     \
      for (size_t fft_idx = 0; fft_idx < num_ffts; fft_idx++) {                \
        long offset_in = 0, offset_out = 0;                                    \
        long temp = (long)fft_idx;                                             \
        for (int d = ndim - 1; d >= 0; d--) {                                  \
          if (d != irfft_axis) {                                               \
            long coord = temp % input->shape[d];                               \
            temp /= input->shape[d];                                           \
            offset_in += coord * input->strides[d];                            \
            offset_out += coord * output->strides[d];                          \
          }                                                                    \
        }                                                                      \
        for (size_t i = 0; i < copy_len; i++) {                                \
          long idx = input->offset + offset_in +                               \
                     (long)i * input->strides[irfft_axis];                     \
          COMP_PREC _Complex *complex_in = (COMP_PREC _Complex *)input->data;  \
          COMP_PREC _Complex val = complex_in[idx];                            \
          temp_re[i] = __real__(val);                                          \
          temp_im[i] = __imag__(val);                                          \
        }                                                                      \
        for (size_t i = copy_len; i < freq_len; i++) {                         \
          temp_re[i] = (COMP_PREC)0.0;                                         \
          temp_im[i] = (COMP_PREC)0.0;                                         \
        }                                                                      \
        for (size_t i = freq_len; i < n_out; i++) {                            \
          size_t mirror_idx = n_out - i;                                       \
          if (mirror_idx < freq_len) {                                         \
            temp_re[i] = temp_re[mirror_idx];                                  \
            temp_im[i] = -temp_im[mirror_idx];                                 \
          } else {                                                             \
            temp_re[i] = (COMP_PREC)0.0;                                       \
            temp_im[i] = (COMP_PREC)0.0;                                       \
          }                                                                    \
        }                                                                      \
        temp_im[0] = (COMP_PREC)0.0;                                           \
        if ((n_out % 2) == 0 && freq_len > n_out / 2)                          \
          temp_im[n_out / 2] = (COMP_PREC)0.0;                                 \
        fft_1d_##COMP_SUFFIX(temp_re, temp_im, n_out, 1, 0, true);             \
        for (size_t i = 0; i < n_out; i++) {                                   \
          long idx = output->offset + offset_out +                             \
                     (long)i * output->strides[irfft_axis];                    \
          ((REAL_PREC *)output->data)[idx] = temp_re[i];                       \
        }                                                                      \
      }                                                                        \
      free(temp_re);                                                           \
      free(temp_im);                                                           \
    } else {                                                                   \
      size_t inter_size = total_elements;                                      \
      COMP_PREC *inter_data =                                                  \
          (COMP_PREC *)malloc(2 * inter_size * sizeof(COMP_PREC));             \
      if (!inter_data) caml_failwith("irfft: allocation failed");              \
      nd_single_iterator_t copy_it;                                            \
      nd_iterator_init(&copy_it, input);                                       \
      do {                                                                     \
        long in_off, inter_off;                                                \
        nd_iterator_get_offset(&copy_it, &in_off);                             \
        inter_off = 0;                                                         \
        for (int d = 0; d < ndim; d++) {                                       \
          inter_off += copy_it.coords[d] * input->strides[d];                  \
        }                                                                      \
        COMP_PREC _Complex *complex_in = (COMP_PREC _Complex *)input->data;    \
        COMP_PREC _Complex val = complex_in[input->offset + in_off];           \
        inter_data[2 * inter_off] = __real__(val);                             \
        inter_data[2 * inter_off + 1] = __imag__(val);                         \
      } while (nd_single_iterator_next(&copy_it));                             \
      nd_single_iterator_destroy(&copy_it);                                    \
      ndarray_t inter_array = {.data = inter_data,                             \
                               .shape = input->shape,                          \
                               .strides = input->strides,                      \
                               .ndim = ndim,                                   \
                               .offset = 0};                                   \
      fft_multi_##COMP_SUFFIX(&inter_array, axes, num_axes - 1, true);         \
      size_t num_1d = inter_size / n_in;                                       \
      COMP_PREC *temp_re = (COMP_PREC *)malloc(n_out * sizeof(COMP_PREC));     \
      COMP_PREC *temp_im = (COMP_PREC *)malloc(n_out * sizeof(COMP_PREC));     \
      if (!temp_re || !temp_im) {                                              \
        free(inter_data);                                                      \
        caml_failwith("irfft: allocation failed");                             \
      }                                                                        \
      for (size_t idx = 0; idx < num_1d; idx++) {                              \
        long offset_inter = 0, offset_out = 0;                                 \
        long temp = (long)idx;                                                 \
        for (int d = ndim - 1; d >= 0; d--) {                                  \
          if (d != irfft_axis) {                                               \
            long coord = temp % output->shape[d];                              \
            temp /= output->shape[d];                                          \
            offset_inter += coord * input->strides[d];                         \
            offset_out += coord * output->strides[d];                          \
          }                                                                    \
        }                                                                      \
        for (size_t i = 0; i < copy_len; i++) {                                \
          long idx_in = offset_inter + (long)i * input->strides[irfft_axis];   \
          temp_re[i] = inter_data[2 * idx_in];                                 \
          temp_im[i] = inter_data[2 * idx_in + 1];                             \
        }                                                                      \
        for (size_t i = copy_len; i < freq_len; i++) {                         \
          temp_re[i] = (COMP_PREC)0.0;                                         \
          temp_im[i] = (COMP_PREC)0.0;                                         \
        }                                                                      \
        for (size_t i = freq_len; i < n_out; i++) {                            \
          size_t mirror_idx = n_out - i;                                       \
          if (mirror_idx < freq_len) {                                         \
            temp_re[i] = temp_re[mirror_idx];                                  \
            temp_im[i] = -temp_im[mirror_idx];                                 \
          } else {                                                             \
            temp_re[i] = (COMP_PREC)0.0;                                       \
            temp_im[i] = (COMP_PREC)0.0;                                       \
          }                                                                    \
        }                                                                      \
        temp_im[0] = (COMP_PREC)0.0;                                           \
        if ((n_out % 2) == 0 && freq_len > n_out / 2)                          \
          temp_im[n_out / 2] = (COMP_PREC)0.0;                                 \
        fft_1d_##COMP_SUFFIX(temp_re, temp_im, n_out, 1, 0, true);             \
        for (size_t i = 0; i < n_out; i++) {                                   \
          long idx = output->offset + offset_out +                             \
                     (long)i * output->strides[irfft_axis];                    \
          ((REAL_PREC *)output->data)[idx] = temp_re[i];                       \
        }                                                                      \
      }                                                                        \
      free(temp_re);                                                           \
      free(temp_im);                                                           \
      free(inter_data);                                                        \
    }                                                                          \
  }

// Instantiate for float64 to complex64
RFFT_MULTI_PREC(double, double, float64, complex64)
IRFFT_MULTI_PREC(double, double, complex64, float64)

// Instantiate for float32 to complex32
RFFT_MULTI_PREC(float, float, float32, complex32)
IRFFT_MULTI_PREC(float, float, complex32, float32)

// Macro for low-precision RFFT (converts to float32 for computation, output to
// low prec complex if applicable)
#define LOW_PREC_RFFT(REAL_T, REAL_SUFFIX, TO_FLOAT, FROM_FLOAT, COMP_SUFFIX)  \
  static void rfft_multi_##REAL_SUFFIX(                                        \
      const ndarray_t *input, ndarray_t *output, int *axes, int num_axes) {    \
    if (num_axes == 0) return;                                                 \
    int ndim = input->ndim;                                                    \
    int rfft_axis = axes[num_axes - 1];                                        \
    size_t n = input->shape[rfft_axis];                                        \
    if (n == 0) return;                                                        \
    size_t n_out = output->shape[rfft_axis];                                   \
    if (n_out != n / 2 + 1) {                                                  \
      /* Shape check failed - output won't be valid */                         \
      return;                                                                  \
    }                                                                          \
    size_t total_elements = total_elements_safe(input);                        \
    if (num_axes == 1) {                                                       \
      size_t num_ffts = total_elements / n;                                    \
      float *temp_re = (float *)malloc(n * sizeof(float));                     \
      float *temp_im = (float *)malloc(n * sizeof(float));                     \
      if (!temp_re || !temp_im) caml_failwith("rfft: allocation failed");      \
      for (size_t fft_idx = 0; fft_idx < num_ffts; fft_idx++) {                \
        long offset_in = 0, offset_out = 0;                                    \
        long temp = (long)fft_idx;                                             \
        for (int d = ndim - 1; d >= 0; d--) {                                  \
          if (d != rfft_axis) {                                                \
            long coord = temp % input->shape[d];                               \
            temp /= input->shape[d];                                           \
            offset_in += coord * input->strides[d];                            \
            offset_out += coord * output->strides[d];                          \
          }                                                                    \
        }                                                                      \
        for (size_t i = 0; i < n; i++) {                                       \
          long idx =                                                           \
              input->offset + offset_in + (long)i * input->strides[rfft_axis]; \
          temp_re[i] = TO_FLOAT(((REAL_T *)input->data)[idx]);                 \
          temp_im[i] = 0.0f;                                                   \
        }                                                                      \
        fft_1d_complex32(temp_re, temp_im, n, 1, 0, false);                    \
        for (size_t i = 0; i < n_out; i++) {                                   \
          long idx = output->offset + offset_out +                             \
                     (long)i * output->strides[rfft_axis];                     \
          if (0) {                                                             \
          } /* to avoid unused warning */                                      \
          else if (0) { /* complex16 case handled separately */                \
            ((caml_ba_complex16 *)output->data)[idx].re =                      \
                FROM_FLOAT(temp_re[i]);                                        \
            ((caml_ba_complex16 *)output->data)[idx].im =                      \
                FROM_FLOAT(temp_im[i]);                                        \
          } else {                                                             \
            ((float *)output->data)[2 * idx] = temp_re[i];                     \
            ((float *)output->data)[2 * idx + 1] = temp_im[i];                 \
          }                                                                    \
        }                                                                      \
      }                                                                        \
      free(temp_re);                                                           \
      free(temp_im);                                                           \
    } else {                                                                   \
      size_t inter_size = total_elements_safe(output);                         \
      float *inter_data = (float *)calloc(2 * inter_size, sizeof(float));      \
      if (!inter_data) caml_failwith("rfft: allocation failed");               \
      long inter_strides[MAX_NDIM];                                            \
      inter_strides[ndim - 1] = 2;                                             \
      for (int d = ndim - 2; d >= 0; d--) {                                    \
        inter_strides[d] = inter_strides[d + 1] * output->shape[d + 1];        \
      }                                                                        \
      size_t num_1d = inter_size / n_out;                                      \
      float *temp_re = (float *)malloc(n * sizeof(float));                     \
      float *temp_im = (float *)malloc(n * sizeof(float));                     \
      if (!temp_re || !temp_im) {                                              \
        free(inter_data);                                                      \
        caml_failwith("rfft: allocation failed");                              \
      }                                                                        \
      for (size_t idx = 0; idx < num_1d; idx++) {                              \
        long offset_in = 0, offset_inter = 0;                                  \
        long temp = (long)idx;                                                 \
        for (int d = ndim - 1; d >= 0; d--) {                                  \
          if (d != rfft_axis) {                                                \
            long coord = temp % input->shape[d];                               \
            temp /= input->shape[d];                                           \
            offset_in += coord * input->strides[d];                            \
            offset_inter += coord * inter_strides[d];                          \
          }                                                                    \
        }                                                                      \
        for (size_t i = 0; i < n; i++) {                                       \
          long idx =                                                           \
              input->offset + offset_in + (long)i * input->strides[rfft_axis]; \
          temp_re[i] = TO_FLOAT(((REAL_T *)input->data)[idx]);                 \
          temp_im[i] = 0.0f;                                                   \
        }                                                                      \
        fft_1d_complex32(temp_re, temp_im, n, 1, 0, false);                    \
        for (size_t i = 0; i < n_out; i++) {                                   \
          long idx_out = offset_inter + (long)i * inter_strides[rfft_axis];    \
          inter_data[idx_out] = temp_re[i];                                    \
          inter_data[idx_out + 1] = temp_im[i];                                \
        }                                                                      \
      }                                                                        \
      free(temp_re);                                                           \
      free(temp_im);                                                           \
      int inter_strides_int[MAX_NDIM];                                         \
      for (int i = 0; i < ndim; i++) {                                         \
        /* Convert strides from COMP_PREC elements to complex elements */      \
        inter_strides_int[i] = (int)(inter_strides[i] / 2);                    \
      }                                                                        \
      ndarray_t inter_array = {.data = inter_data,                             \
                               .shape = output->shape,                         \
                               .strides = inter_strides_int,                   \
                               .ndim = ndim,                                   \
                               .offset = 0};                                   \
      fft_multi_complex32(&inter_array, axes, num_axes - 1, false);            \
      nd_single_iterator_t it;                                                 \
      nd_iterator_init(&it, &inter_array);                                     \
      do {                                                                     \
        long inter_off, out_off;                                               \
        nd_iterator_get_offset(&it, &inter_off);                               \
        out_off = 0;                                                           \
        for (int d = 0; d < ndim; d++) {                                       \
          out_off += it.coords[d] * output->strides[d];                        \
        }                                                                      \
        if (0) {                                                               \
        } else if (0) { /* complex16 case handled separately */                \
          ((caml_ba_complex16 *)output->data)[output->offset + out_off].re =   \
              FROM_FLOAT(inter_data[inter_off]);                               \
          ((caml_ba_complex16 *)output->data)[output->offset + out_off].im =   \
              FROM_FLOAT(inter_data[inter_off + 1]);                           \
        } else {                                                               \
          ((float *)output->data)[2 * (output->offset + out_off)] =            \
              inter_data[inter_off];                                           \
          ((float *)output->data)[2 * (output->offset + out_off) + 1] =        \
              inter_data[inter_off + 1];                                       \
        }                                                                      \
      } while (nd_single_iterator_next(&it));                                  \
      nd_single_iterator_destroy(&it);                                         \
      free(inter_data);                                                        \
    }                                                                          \
  }

#define LOW_PREC_IRFFT(COMP_T, COMP_SUFFIX, TO_FLOAT, FROM_FLOAT, REAL_SUFFIX) \
  static void irfft_multi_##REAL_SUFFIX(const ndarray_t *input,                \
                                        ndarray_t *output, int *axes,          \
                                        int num_axes, int last_size) {         \
    if (num_axes == 0) return;                                                 \
    int ndim = input->ndim;                                                    \
    int irfft_axis = axes[num_axes - 1];                                       \
    size_t n_in = input->shape[irfft_axis];                                    \
    size_t n_out = (size_t)last_size;                                          \
    if (n_out == 0) return;                                                    \
    size_t total_elements = total_elements_safe(input);                        \
    if (num_axes == 1) {                                                       \
      size_t num_ffts = total_elements / n_in;                                 \
      float *temp_re = (float *)malloc(n_out * sizeof(float));                 \
      float *temp_im = (float *)malloc(n_out * sizeof(float));                 \
      if (!temp_re || !temp_im) caml_failwith("irfft: allocation failed");     \
      for (size_t fft_idx = 0; fft_idx < num_ffts; fft_idx++) {                \
        long offset_in = 0, offset_out = 0;                                    \
        long temp = (long)fft_idx;                                             \
        for (int d = ndim - 1; d >= 0; d--) {                                  \
          if (d != irfft_axis) {                                               \
            long coord = temp % input->shape[d];                               \
            temp /= input->shape[d];                                           \
            offset_in += coord * input->strides[d];                            \
            offset_out += coord * output->strides[d];                          \
          }                                                                    \
        }                                                                      \
        for (size_t i = 0; i < n_in; i++) {                                    \
          long idx = input->offset + offset_in +                               \
                     (long)i * input->strides[irfft_axis];                     \
          if (0) {                                                             \
          } /* complex16 case is handled separately */ else if (0) {           \
            temp_re[i] = TO_FLOAT(((caml_ba_complex16 *)input->data)[idx].re); \
            temp_im[i] = TO_FLOAT(((caml_ba_complex16 *)input->data)[idx].im); \
          } else {                                                             \
            temp_re[i] = ((float *)input->data)[2 * idx];                      \
            temp_im[i] = ((float *)input->data)[2 * idx + 1];                  \
          }                                                                    \
        }                                                                      \
        for (size_t i = n_in; i < n_out; i++) {                                \
          size_t mirror_idx = n_out - i;                                       \
          if (mirror_idx > 0 && mirror_idx < n_in) {                           \
            temp_re[i] = temp_re[mirror_idx];                                  \
            temp_im[i] = -temp_im[mirror_idx];                                 \
          } else {                                                             \
            temp_re[i] = 0.0f;                                                 \
            temp_im[i] = 0.0f;                                                 \
          }                                                                    \
        }                                                                      \
        temp_im[0] = 0.0f;                                                     \
        if ((n_out % 2) == 0 && n_in > n_out / 2) temp_im[n_out / 2] = 0.0f;   \
        fft_1d_complex32(temp_re, temp_im, n_out, 1, 0, true);                 \
        for (size_t i = 0; i < n_out; i++) {                                   \
          long idx = output->offset + offset_out +                             \
                     (long)i * output->strides[irfft_axis];                    \
          ((COMP_T *)output->data)[idx] = FROM_FLOAT(temp_re[i]);              \
        }                                                                      \
      }                                                                        \
      free(temp_re);                                                           \
      free(temp_im);                                                           \
    } else {                                                                   \
      size_t inter_size = total_elements;                                      \
      float *inter_data = (float *)malloc(2 * inter_size * sizeof(float));     \
      if (!inter_data) caml_failwith("irfft: allocation failed");              \
      nd_single_iterator_t copy_it;                                            \
      nd_iterator_init(&copy_it, input);                                       \
      do {                                                                     \
        long in_off, inter_off = 0;                                            \
        nd_iterator_get_offset(&copy_it, &in_off);                             \
        for (int d = 0; d < ndim; d++) {                                       \
          inter_off += copy_it.coords[d] * input->strides[d];                  \
        }                                                                      \
        if (0) {                                                               \
        } else if (0) { /* complex16 case handled separately */                \
          inter_data[inter_off * 2] = TO_FLOAT(                                \
              ((caml_ba_complex16 *)input->data)[input->offset + in_off].re);  \
          inter_data[inter_off * 2 + 1] = TO_FLOAT(                            \
              ((caml_ba_complex16 *)input->data)[input->offset + in_off].im);  \
        } else {                                                               \
          inter_data[inter_off * 2] =                                          \
              ((float *)input->data)[2 * (input->offset + in_off)];            \
          inter_data[inter_off * 2 + 1] =                                      \
              ((float *)input->data)[2 * (input->offset + in_off) + 1];        \
        }                                                                      \
      } while (nd_single_iterator_next(&copy_it));                             \
      nd_single_iterator_destroy(&copy_it);                                    \
      ndarray_t inter_array = {.data = inter_data,                             \
                               .shape = input->shape,                          \
                               .strides = input->strides,                      \
                               .ndim = ndim,                                   \
                               .offset = 0};                                   \
      fft_multi_complex32(&inter_array, axes, num_axes - 1, true);             \
      size_t num_1d = inter_size / n_in;                                       \
      float *temp_re = (float *)malloc(n_out * sizeof(float));                 \
      float *temp_im = (float *)malloc(n_out * sizeof(float));                 \
      if (!temp_re || !temp_im) {                                              \
        free(inter_data);                                                      \
        caml_failwith("irfft: allocation failed");                             \
      }                                                                        \
      for (size_t idx = 0; idx < num_1d; idx++) {                              \
        long offset_inter = 0, offset_out = 0;                                 \
        long temp = (long)idx;                                                 \
        for (int d = ndim - 1; d >= 0; d--) {                                  \
          if (d != irfft_axis) {                                               \
            long coord = temp % output->shape[d];                              \
            temp /= output->shape[d];                                          \
            offset_inter += coord * input->strides[d];                         \
            offset_out += coord * output->strides[d];                          \
          }                                                                    \
        }                                                                      \
        for (size_t i = 0; i < n_in; i++) {                                    \
          long idx_in = offset_inter + (long)i * input->strides[irfft_axis];   \
          temp_re[i] = inter_data[2 * idx_in];                                 \
          temp_im[i] = inter_data[2 * idx_in + 1];                             \
        }                                                                      \
        for (size_t i = n_in; i < n_out; i++) {                                \
          size_t mirror_idx = n_out - i;                                       \
          if (mirror_idx > 0 && mirror_idx < n_in) {                           \
            temp_re[i] = temp_re[mirror_idx];                                  \
            temp_im[i] = -temp_im[mirror_idx];                                 \
          } else {                                                             \
            temp_re[i] = 0.0f;                                                 \
            temp_im[i] = 0.0f;                                                 \
          }                                                                    \
        }                                                                      \
        temp_im[0] = 0.0f;                                                     \
        if ((n_out % 2) == 0 && n_in > n_out / 2) temp_im[n_out / 2] = 0.0f;   \
        fft_1d_complex32(temp_re, temp_im, n_out, 1, 0, true);                 \
        for (size_t i = 0; i < n_out; i++) {                                   \
          long idx = output->offset + offset_out +                             \
                     (long)i * output->strides[irfft_axis];                    \
          ((COMP_T *)output->data)[idx] = FROM_FLOAT(temp_re[i]);              \
        }                                                                      \
      }                                                                        \
      free(temp_re);                                                           \
      free(temp_im);                                                           \
      free(inter_data);                                                        \
    }                                                                          \
  }

// For float16 (uint16_t) to complex16
LOW_PREC_RFFT(uint16_t, float16, half_to_float, float_to_half, complex16)
LOW_PREC_IRFFT(uint16_t, complex16, half_to_float, float_to_half, float16)

// For bfloat16 (uint16_t) to complex32
LOW_PREC_RFFT(uint16_t, bfloat16, bfloat16_to_float, float_to_bfloat16,
              complex32)
LOW_PREC_IRFFT(uint16_t, complex32, , float_to_bfloat16,
               bfloat16) /* TO_FLOAT not needed for input complex32 */

// For float8_e4m3 (uint8_t) to complex32
LOW_PREC_RFFT(uint8_t, float8_e4m3, fp8_e4m3_to_float, float_to_fp8_e4m3,
              complex32)
LOW_PREC_IRFFT(uint8_t, complex32, , float_to_fp8_e4m3, float8_e4m3)

// For float8_e5m2 (uint8_t) to complex32
LOW_PREC_RFFT(uint8_t, float8_e5m2, fp8_e5m2_to_float, float_to_fp8_e5m2,
              complex32)
LOW_PREC_IRFFT(uint8_t, complex32, , float_to_fp8_e5m2, float8_e5m2)

// Helper to copy ndarray data respecting strides (for complex64)
static void copy_ndarray_complex64(const ndarray_t *src, ndarray_t *dst) {
  size_t total = total_elements_safe(src);
  if (total == 0) return;
  if (is_contiguous(src) && is_contiguous(dst)) {
    memcpy((complex64 *)dst->data + dst->offset,
           (complex64 *)src->data + src->offset, total * sizeof(complex64));
  } else {
    nd_copy_iterator_t it;
    nd_copy_iterator_init(&it, src, dst);
    do {
      long src_off, dst_off;
      nd_copy_iterator_get_offsets(&it, &src_off, &dst_off);
      complex64 *src_data = (complex64 *)src->data;
      complex64 *dst_data = (complex64 *)dst->data;
      dst_data[dst->offset + dst_off] = src_data[src->offset + src_off];
    } while (nd_copy_iterator_next(&it));
    nd_copy_iterator_destroy(&it);
  }
}

// Helper to copy ndarray data respecting strides (for complex32)
static void copy_ndarray_complex32(const ndarray_t *src, ndarray_t *dst) {
  size_t total = total_elements_safe(src);
  if (total == 0) return;
  if (is_contiguous(src) && is_contiguous(dst)) {
    memcpy((complex32 *)dst->data + dst->offset,
           (complex32 *)src->data + src->offset, total * sizeof(complex32));
  } else {
    nd_copy_iterator_t it;
    nd_copy_iterator_init(&it, src, dst);
    do {
      long src_off, dst_off;
      nd_copy_iterator_get_offsets(&it, &src_off, &dst_off);
      complex32 *src_data = (complex32 *)src->data;
      complex32 *dst_data = (complex32 *)dst->data;
      dst_data[dst->offset + dst_off] = src_data[src->offset + src_off];
    } while (nd_copy_iterator_next(&it));
    nd_copy_iterator_destroy(&it);
  }
}

// Helper to copy ndarray data for complex16
static void copy_ndarray_complex16(const ndarray_t *src, ndarray_t *dst) {
  size_t total = total_elements_safe(src);
  if (total == 0) return;
  if (is_contiguous(src) && is_contiguous(dst)) {
    memcpy((caml_ba_complex16 *)dst->data + dst->offset,
           (caml_ba_complex16 *)src->data + src->offset,
           total * sizeof(caml_ba_complex16));
  } else {
    nd_copy_iterator_t it;
    nd_copy_iterator_init(&it, src, dst);
    do {
      long src_off, dst_off;
      nd_copy_iterator_get_offsets(&it, &src_off, &dst_off);
      ((caml_ba_complex16 *)dst->data)[dst->offset + dst_off] =
          ((caml_ba_complex16 *)src->data)[src->offset + src_off];
    } while (nd_copy_iterator_next(&it));
    nd_copy_iterator_destroy(&it);
  }
}

// Helper to extract axes from OCaml list
static int *extract_axes(value v_axes, int *num_axes) {
  *num_axes = 0;
  value curr = v_axes;
  while (curr != Val_emptylist) {
    (*num_axes)++;
    curr = Field(curr, 1);
  }
  int *axes = (int *)caml_stat_alloc(*num_axes * sizeof(int));
  curr = v_axes;
  for (int i = 0; i < *num_axes; i++) {
    axes[i] = Int_val(Field(curr, 0));
    curr = Field(curr, 1);
  }
  return axes;
}

// ============================================================================
// OCaml FFI Stubs
// ============================================================================

// Stub for complex FFT/ IFFT
CAMLprim value caml_nx_op_fft(value v_input, value v_output, value v_axes,
                              value v_inverse) {
  CAMLparam4(v_input, v_output, v_axes, v_inverse);
  ndarray_t input = extract_ndarray(v_input);
  ndarray_t output = extract_ndarray(v_output);
  value v_in_data = Field(v_input, FFI_TENSOR_DATA);
  value v_out_data = Field(v_output, FFI_TENSOR_DATA);
  struct caml_ba_array *ba_in = Caml_ba_array_val(v_in_data);
  struct caml_ba_array *ba_out = Caml_ba_array_val(v_out_data);
  int kind = ba_in->flags & CAML_BA_KIND_MASK;
  if (kind != (ba_out->flags & CAML_BA_KIND_MASK))
    caml_failwith("fft: dtype mismatch");
  if (input.ndim != output.ndim) caml_failwith("fft: ndim mismatch");
  for (int i = 0; i < input.ndim; i++) {
    if (input.shape[i] != output.shape[i]) caml_failwith("fft: shape mismatch");
  }
  int num_axes;
  int *axes = extract_axes(v_axes, &num_axes);
  if (num_axes > 0 &&
      !validate_and_normalize_axes(axes, num_axes, input.ndim)) {
    caml_stat_free(axes);
    caml_failwith("fft: invalid or duplicate axes");
  }
  bool inverse = Bool_val(v_inverse);
  void (*fft_multi)(ndarray_t *, int *, int, bool) = NULL;
  void (*copy_fn)(const ndarray_t *, ndarray_t *) = NULL;
  switch (kind) {
    case CAML_BA_COMPLEX64:
      fft_multi = fft_multi_complex64;
      copy_fn = copy_ndarray_complex64;
      break;
    case CAML_BA_COMPLEX32:
      fft_multi = fft_multi_complex32;
      copy_fn = copy_ndarray_complex32;
      break;
    case NX_BA_COMPLEX16:
      fft_multi = fft_multi_complex16;
      copy_fn = copy_ndarray_complex16;
      break;
    default:
      caml_stat_free(axes);
      caml_failwith("fft: unsupported dtype");
  }
  caml_enter_blocking_section();
  copy_fn(&input, &output);
  fft_multi(&output, axes, num_axes, inverse);
  caml_leave_blocking_section();
  caml_stat_free(axes);
  cleanup_ndarray(&input);
  cleanup_ndarray(&output);
  CAMLreturn(Val_unit);
}

// Stub for RFFT
CAMLprim value caml_nx_op_rfft(value v_input, value v_output, value v_axes) {
  CAMLparam3(v_input, v_output, v_axes);
  ndarray_t input = extract_ndarray(v_input);
  ndarray_t output = extract_ndarray(v_output);
  value v_in_data = Field(v_input, FFI_TENSOR_DATA);
  value v_out_data = Field(v_output, FFI_TENSOR_DATA);
  struct caml_ba_array *ba_in = Caml_ba_array_val(v_in_data);
  struct caml_ba_array *ba_out = Caml_ba_array_val(v_out_data);
  int kind_in = ba_in->flags & CAML_BA_KIND_MASK;
  int kind_out = ba_out->flags & CAML_BA_KIND_MASK;
  if (input.ndim != output.ndim) caml_failwith("rfft: ndim mismatch");
  int num_axes;
  int *axes = extract_axes(v_axes, &num_axes);
  if (num_axes > 0 &&
      !validate_and_normalize_axes(axes, num_axes, input.ndim)) {
    caml_stat_free(axes);
    caml_failwith("rfft: invalid or duplicate axes");
  }
  void (*rfft_multi)(const ndarray_t *, ndarray_t *, int *, int) = NULL;
  switch (kind_in) {
    case CAML_BA_FLOAT64:
      if (kind_out != CAML_BA_COMPLEX64)
        caml_failwith("rfft: output dtype mismatch");
      rfft_multi = rfft_multi_float64;
      break;
    case CAML_BA_FLOAT32:
      if (kind_out != CAML_BA_COMPLEX32)
        caml_failwith("rfft: output dtype mismatch");
      rfft_multi = rfft_multi_float32;
      break;
    case CAML_BA_FLOAT16:
      if (kind_out != NX_BA_COMPLEX16)
        caml_failwith("rfft: output dtype mismatch");
      rfft_multi = rfft_multi_float16;
      break;
    case NX_BA_BFLOAT16:
      if (kind_out != CAML_BA_COMPLEX32)
        caml_failwith("rfft: output dtype mismatch");
      rfft_multi = rfft_multi_bfloat16;
      break;
    case NX_BA_FP8_E4M3:
      if (kind_out != CAML_BA_COMPLEX32)
        caml_failwith("rfft: output dtype mismatch");
      rfft_multi = rfft_multi_float8_e4m3;
      break;
    case NX_BA_FP8_E5M2:
      if (kind_out != CAML_BA_COMPLEX32)
        caml_failwith("rfft: output dtype mismatch");
      rfft_multi = rfft_multi_float8_e5m2;
      break;
    default:
      caml_stat_free(axes);
      caml_failwith("rfft: unsupported dtype");
  }

  // Validate output shape before entering blocking section
  if (num_axes > 0) {
    int rfft_axis = axes[num_axes - 1];
    size_t n = input.shape[rfft_axis];
    size_t n_out = output.shape[rfft_axis];
    if (n_out != n / 2 + 1) {
      caml_stat_free(axes);
      cleanup_ndarray(&input);
      cleanup_ndarray(&output);
      caml_failwith("rfft: invalid output shape");
    }
    // Check other axes have matching shapes
    for (int i = 0; i < input.ndim; i++) {
      if (i != rfft_axis && input.shape[i] != output.shape[i]) {
        caml_stat_free(axes);
        cleanup_ndarray(&input);
        cleanup_ndarray(&output);
        caml_failwith("rfft: shape mismatch on non-transform axis");
      }
    }
  }

  caml_enter_blocking_section();
  rfft_multi(&input, &output, axes, num_axes);
  caml_leave_blocking_section();
  caml_stat_free(axes);
  cleanup_ndarray(&input);
  cleanup_ndarray(&output);
  CAMLreturn(Val_unit);
}

// Stub for IRFFT
CAMLprim value caml_nx_op_irfft(value v_input, value v_output, value v_axes,
                                value v_last_dim_size) {
  CAMLparam4(v_input, v_output, v_axes, v_last_dim_size);
  ndarray_t input = extract_ndarray(v_input);
  ndarray_t output = extract_ndarray(v_output);
  value v_in_data = Field(v_input, FFI_TENSOR_DATA);
  value v_out_data = Field(v_output, FFI_TENSOR_DATA);
  struct caml_ba_array *ba_in = Caml_ba_array_val(v_in_data);
  struct caml_ba_array *ba_out = Caml_ba_array_val(v_out_data);
  int kind_in = ba_in->flags & CAML_BA_KIND_MASK;
  int kind_out = ba_out->flags & CAML_BA_KIND_MASK;
  if (input.ndim != output.ndim) caml_failwith("irfft: ndim mismatch");
  int num_axes;
  int *axes = extract_axes(v_axes, &num_axes);
  if (num_axes > 0 &&
      !validate_and_normalize_axes(axes, num_axes, input.ndim)) {
    caml_stat_free(axes);
    caml_failwith("irfft: invalid or duplicate axes");
  }
  int last_size = Int_val(v_last_dim_size);
  void (*irfft_multi)(const ndarray_t *, ndarray_t *, int *, int, int) = NULL;
  switch (kind_in) {
    case CAML_BA_COMPLEX64:
      if (kind_out != CAML_BA_FLOAT64)
        caml_failwith("irfft: output dtype mismatch");
      irfft_multi = irfft_multi_complex64;
      break;
    case CAML_BA_COMPLEX32:
      if (kind_out != CAML_BA_FLOAT32)
        caml_failwith("irfft: output dtype mismatch");
      irfft_multi = irfft_multi_complex32;
      break;
    case NX_BA_COMPLEX16:
      if (kind_out != CAML_BA_FLOAT16)
        caml_failwith("irfft: output dtype mismatch");
      irfft_multi = irfft_multi_float16;
      break;
    default:
      if (kind_in == CAML_BA_COMPLEX32) {
        switch (kind_out) {
          case NX_BA_BFLOAT16:
            irfft_multi = irfft_multi_complex32;
            break;
          case NX_BA_FP8_E4M3:
            irfft_multi = irfft_multi_complex32;
            break;
          case NX_BA_FP8_E5M2:
            irfft_multi = irfft_multi_complex32;
            break;
          default:
            caml_stat_free(axes);
            caml_failwith("irfft: unsupported dtype");
        }
      } else {
        caml_stat_free(axes);
        caml_failwith("irfft: unsupported dtype");
      }
  }
  caml_enter_blocking_section();
  irfft_multi(&input, &output, axes, num_axes, last_size);
  caml_leave_blocking_section();
  caml_stat_free(axes);
  cleanup_ndarray(&input);
  cleanup_ndarray(&output);
  CAMLreturn(Val_unit);
}
