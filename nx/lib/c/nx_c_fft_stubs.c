#include "nx_c_shared.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

// DFT implementation for arbitrary sizes with optimized twiddles
static void dft_complex64(double *data_re, double *data_im, size_t n,
                          int stride, size_t offset, bool inverse) {
  if (n == 0) return;
  
  double *temp_re = (double *)malloc(n * sizeof(double));
  double *temp_im = (double *)malloc(n * sizeof(double));
  double *twiddle_cos = (double *)malloc(n * n * sizeof(double));  // Precompute
  double *twiddle_sin = (double *)malloc(n * n * sizeof(double));

  if (!temp_re || !temp_im || !twiddle_cos || !twiddle_sin) {
    // Cleanup and return
    free(temp_re); free(temp_im); free(twiddle_cos); free(twiddle_sin);
    return;
  }

  double sign = inverse ? 1.0 : -1.0;
  double two_pi_n = 2.0 * M_PI / n;

  // Precompute all twiddles
  for (size_t k = 0; k < n; k++) {
    for (size_t j = 0; j < n; j++) {
      double angle = sign * two_pi_n * k * j;
      twiddle_cos[k * n + j] = cos(angle);
      twiddle_sin[k * n + j] = sin(angle);
    }
  }

  // Copy input
  for (size_t i = 0; i < n; i++) {
    temp_re[i] = data_re[offset + i * stride];
    temp_im[i] = data_im[offset + i * stride];
  }

  // Compute DFT with unrolling (partial, for performance)
  for (size_t k = 0; k < n; k++) {
    double sum_re = 0.0;
    double sum_im = 0.0;
    size_t base = k * n;
    
    // Process in groups of 4 for better performance
    size_t j;
    for (j = 0; j + 3 < n; j += 4) {
      sum_re += temp_re[j] * twiddle_cos[base + j] - temp_im[j] * twiddle_sin[base + j] +
                temp_re[j+1] * twiddle_cos[base + j+1] - temp_im[j+1] * twiddle_sin[base + j+1] +
                temp_re[j+2] * twiddle_cos[base + j+2] - temp_im[j+2] * twiddle_sin[base + j+2] +
                temp_re[j+3] * twiddle_cos[base + j+3] - temp_im[j+3] * twiddle_sin[base + j+3];
      sum_im += temp_re[j] * twiddle_sin[base + j] + temp_im[j] * twiddle_cos[base + j] +
                temp_re[j+1] * twiddle_sin[base + j+1] + temp_im[j+1] * twiddle_cos[base + j+1] +
                temp_re[j+2] * twiddle_sin[base + j+2] + temp_im[j+2] * twiddle_cos[base + j+2] +
                temp_re[j+3] * twiddle_sin[base + j+3] + temp_im[j+3] * twiddle_cos[base + j+3];
    }
    // Handle remainder
    for (; j < n; j++) {
      sum_re += temp_re[j] * twiddle_cos[base + j] - temp_im[j] * twiddle_sin[base + j];
      sum_im += temp_re[j] * twiddle_sin[base + j] + temp_im[j] * twiddle_cos[base + j];
    }

    data_re[offset + k * stride] = sum_re;
    data_im[offset + k * stride] = sum_im;
  }

  free(temp_re); free(temp_im); free(twiddle_cos); free(twiddle_sin);
}

// 1D FFT implementation using Cooley-Tukey algorithm
static void fft_1d_complex64(double *data_re, double *data_im, size_t n,
                             int stride, size_t offset, bool inverse) {
  if (n == 0) return;  // Fix n=0 crash
  if (n <= 1) return;

  // Check if n is power of 2
  int log2n = 0;
  size_t temp = n;
  bool is_power_of_2 = true;
  while (temp > 1) {
    if (temp & 1) {
      is_power_of_2 = false;
      break;
    }
    temp >>= 1;
    log2n++;
  }

  if (!is_power_of_2) {
    // Not a power of 2, use DFT
    dft_complex64(data_re, data_im, n, stride, offset, inverse);
    return;
  }

  // Bit reversal permutation
  for (size_t i = 0; i < n; i++) {
    size_t j = bit_reverse(i, log2n);
    if (i < j) {
      // Swap elements
      size_t idx_i = offset + i * stride;
      size_t idx_j = offset + j * stride;

      double temp_re = data_re[idx_i];
      double temp_im = data_im[idx_i];
      data_re[idx_i] = data_re[idx_j];
      data_im[idx_i] = data_im[idx_j];
      data_re[idx_j] = temp_re;
      data_im[idx_j] = temp_im;
    }
  }

  // FFT computation
  double sign = inverse ? 1.0 : -1.0;

  for (size_t stage_size = 2; stage_size <= n; stage_size *= 2) {
    size_t half_stage = stage_size / 2;
    double angle_step = sign * 2.0 * M_PI / stage_size;

    for (size_t k = 0; k < n; k += stage_size) {
      for (size_t j = 0; j < half_stage; j++) {
        double angle = angle_step * j;
        double twiddle_re, twiddle_im;
        compute_twiddle(angle, &twiddle_re, &twiddle_im);

        size_t idx1 = offset + (k + j) * stride;
        size_t idx2 = offset + (k + j + half_stage) * stride;

        double a_re = data_re[idx1];
        double a_im = data_im[idx1];
        double b_re = data_re[idx2];
        double b_im = data_im[idx2];

        // Complex multiplication: b * twiddle
        double b_twiddle_re = b_re * twiddle_re - b_im * twiddle_im;
        double b_twiddle_im = b_re * twiddle_im + b_im * twiddle_re;

        // Butterfly operation
        data_re[idx1] = a_re + b_twiddle_re;
        data_im[idx1] = a_im + b_twiddle_im;
        data_re[idx2] = a_re - b_twiddle_re;
        data_im[idx2] = a_im - b_twiddle_im;
      }
    }
  }

  // Do NOT apply scaling here - it will be done after all transforms
}

// Multi-dimensional FFT for complex64
static void fft_multi_complex64(ndarray_t *data, int *axes, int num_axes,
                                bool inverse) {
  int ndim = data->ndim;
  const int *shape = data->shape;

  // Find maximum axis size for temp allocation
  size_t max_n = 0;
  for (int ax_idx = 0; ax_idx < num_axes; ax_idx++) {
    int axis = axes[ax_idx];
    if (axis < 0) axis += ndim;
    size_t n = shape[axis];
    if (n > max_n) max_n = n;
  }

  // Allocate temporary arrays once for all axes
  double *temp_re = (double *)malloc(max_n * sizeof(double));
  double *temp_im = (double *)malloc(max_n * sizeof(double));

  if (!temp_re || !temp_im) {
    free(temp_re);
    free(temp_im);
    return;
  }

  // Perform FFT along each specified axis
  for (int ax_idx = 0; ax_idx < num_axes; ax_idx++) {
    int axis = axes[ax_idx];
    if (axis < 0) axis += ndim;

    size_t n = shape[axis];
    if (n <= 1) continue;

    // Calculate stride for this axis
    size_t axis_stride = 1;
    for (int d = axis + 1; d < ndim; d++) {
      axis_stride *= shape[d];
    }

    // Iterate over all other dimensions
    size_t total_elements = 1;
    for (int d = 0; d < ndim; d++) {
      total_elements *= shape[d];
    }

    size_t num_ffts = total_elements / n;

    // Process each 1D FFT along the axis
    for (size_t fft_idx = 0; fft_idx < num_ffts; fft_idx++) {
      // Calculate starting offset for this FFT
      size_t offset = 0;
      size_t temp_idx = fft_idx;

      for (int d = ndim - 1; d >= 0; d--) {
        if (d != axis) {
          size_t dim_size = (d > axis) ? shape[d] : shape[d];
          size_t coord = temp_idx % dim_size;
          temp_idx /= dim_size;

          size_t stride_d = 1;
          for (int dd = d + 1; dd < ndim; dd++) {
            stride_d *= shape[dd];
          }
          offset += coord * stride_d;
        }
      }

      // Extract data along the axis
      // Note: data is stored as interleaved complex values
      double *complex_data = (double *)data->data;
      for (size_t i = 0; i < n; i++) {
        size_t idx = offset + i * axis_stride;
        temp_re[i] = complex_data[2 * idx];
        temp_im[i] = complex_data[2 * idx + 1];
      }

      // Perform 1D FFT
      fft_1d_complex64(temp_re, temp_im, n, 1, 0, inverse);

      // Write back the result
      for (size_t i = 0; i < n; i++) {
        size_t idx = offset + i * axis_stride;
        complex_data[2 * idx] = temp_re[i];
        complex_data[2 * idx + 1] = temp_im[i];
      }
    }
  }

  free(temp_re);
  free(temp_im);

  // Note: Scaling is handled by the frontend based on norm parameter
}

// Forward declarations
CAMLprim value caml_nx_fft_complex64(value v_ndim, value v_shape, value v_input,
                                     value v_input_strides,
                                     value v_input_offset, value v_output,
                                     value v_output_strides,
                                     value v_output_offset, value v_axes,
                                     value v_num_axes, value v_inverse);

CAMLprim value caml_nx_fft_complex32(value v_ndim, value v_shape, value v_input,
                                     value v_input_strides,
                                     value v_input_offset, value v_output,
                                     value v_output_strides,
                                     value v_output_offset, value v_axes,
                                     value v_num_axes, value v_inverse);

CAMLprim value caml_nx_rfft_float64(value v_ndim, value v_shape, value v_input,
                                    value v_input_strides, value v_input_offset,
                                    value v_output, value v_output_strides,
                                    value v_output_offset, value v_axes,
                                    value v_num_axes);

CAMLprim value caml_nx_irfft_complex64(value v_ndim, value v_shape,
                                       value v_input, value v_input_strides,
                                       value v_input_offset, value v_output,
                                       value v_output_strides,
                                       value v_output_offset, value v_axes,
                                       value v_num_axes, value v_last_size);

CAMLprim value caml_nx_rfft_float32(value v_ndim, value v_shape, value v_input,
                                    value v_input_strides, value v_input_offset,
                                    value v_output, value v_output_strides,
                                    value v_output_offset, value v_axes,
                                    value v_num_axes);

CAMLprim value caml_nx_irfft_complex32(value v_ndim, value v_shape,
                                       value v_input, value v_input_strides,
                                       value v_input_offset, value v_output,
                                       value v_output_strides,
                                       value v_output_offset, value v_axes,
                                       value v_num_axes, value v_last_size);

// FFT dispatch for complex64
CAMLprim value caml_nx_fft_complex64_bc(value *argv, int argn) {
  (void)argn;
  value v_ndim = argv[0], v_shape = argv[1];
  value v_input = argv[2], v_input_strides = argv[3], v_input_offset = argv[4];
  value v_output = argv[5], v_output_strides = argv[6],
        v_output_offset = argv[7];
  value v_axes = argv[8], v_num_axes = argv[9], v_inverse = argv[10];

  return caml_nx_fft_complex64(v_ndim, v_shape, v_input, v_input_strides,
                               v_input_offset, v_output, v_output_strides,
                               v_output_offset, v_axes, v_num_axes, v_inverse);
}

CAMLprim value caml_nx_fft_complex64(value v_ndim, value v_shape, value v_input,
                                     value v_input_strides,
                                     value v_input_offset, value v_output,
                                     value v_output_strides,
                                     value v_output_offset, value v_axes,
                                     value v_num_axes, value v_inverse) {
  // Parse input
  int ndim = Int_val(v_ndim);
  int num_axes = Int_val(v_num_axes);
  bool inverse = Bool_val(v_inverse);

  int shape[ndim > 0 ? ndim : 1];
  int input_strides[ndim > 0 ? ndim : 1];
  int output_strides[ndim > 0 ? ndim : 1];
  int axes[num_axes > 0 ? num_axes : 1];

  for (int i = 0; i < ndim; ++i) {
    shape[i] = Int_val(Field(v_shape, i));
    input_strides[i] = Int_val(Field(v_input_strides, i));
    output_strides[i] = Int_val(Field(v_output_strides, i));
  }

  for (int i = 0; i < num_axes; ++i) {
    axes[i] = Int_val(Field(v_axes, i));
  }

  // Get arrays - input and output can be the same
  ndarray_t input = {
      .data = Caml_ba_data_val(v_input) + Long_val(v_input_offset),
      .shape = shape,
      .strides = input_strides,
      .ndim = ndim};

  ndarray_t output = {
      .data = Caml_ba_data_val(v_output) + Long_val(v_output_offset),
      .shape = shape,
      .strides = output_strides,
      .ndim = ndim};

  // Copy input to output if different buffers
  if (input.data != output.data) {
    size_t total_size = 1;
    for (int i = 0; i < ndim; i++) {
      total_size *= shape[i];
    }
    memcpy(output.data, input.data, total_size * 2 * sizeof(double));
  }

  caml_enter_blocking_section();
  fft_multi_complex64(&output, axes, num_axes, inverse);
  caml_leave_blocking_section();

  return Val_unit;
}

// DFT implementation for arbitrary sizes (single precision) with optimized twiddles
static void dft_complex32(float *data_re, float *data_im, size_t n, int stride,
                          size_t offset, bool inverse) {
  if (n == 0) return;
  
  float *temp_re = (float *)malloc(n * sizeof(float));
  float *temp_im = (float *)malloc(n * sizeof(float));
  float *twiddle_cos = (float *)malloc(n * n * sizeof(float));  // Precompute
  float *twiddle_sin = (float *)malloc(n * n * sizeof(float));

  if (!temp_re || !temp_im || !twiddle_cos || !twiddle_sin) {
    // Cleanup and return
    free(temp_re); free(temp_im); free(twiddle_cos); free(twiddle_sin);
    return;
  }

  float sign = inverse ? 1.0f : -1.0f;
  float two_pi_n = 2.0f * (float)M_PI / n;

  // Precompute all twiddles
  for (size_t k = 0; k < n; k++) {
    for (size_t j = 0; j < n; j++) {
      float angle = sign * two_pi_n * k * j;
      twiddle_cos[k * n + j] = cosf(angle);
      twiddle_sin[k * n + j] = sinf(angle);
    }
  }

  // Copy input
  for (size_t i = 0; i < n; i++) {
    temp_re[i] = data_re[offset + i * stride];
    temp_im[i] = data_im[offset + i * stride];
  }

  // Compute DFT with unrolling (partial, for performance)
  for (size_t k = 0; k < n; k++) {
    float sum_re = 0.0f;
    float sum_im = 0.0f;
    size_t base = k * n;
    
    // Process in groups of 4 for better performance
    size_t j;
    for (j = 0; j + 3 < n; j += 4) {
      sum_re += temp_re[j] * twiddle_cos[base + j] - temp_im[j] * twiddle_sin[base + j] +
                temp_re[j+1] * twiddle_cos[base + j+1] - temp_im[j+1] * twiddle_sin[base + j+1] +
                temp_re[j+2] * twiddle_cos[base + j+2] - temp_im[j+2] * twiddle_sin[base + j+2] +
                temp_re[j+3] * twiddle_cos[base + j+3] - temp_im[j+3] * twiddle_sin[base + j+3];
      sum_im += temp_re[j] * twiddle_sin[base + j] + temp_im[j] * twiddle_cos[base + j] +
                temp_re[j+1] * twiddle_sin[base + j+1] + temp_im[j+1] * twiddle_cos[base + j+1] +
                temp_re[j+2] * twiddle_sin[base + j+2] + temp_im[j+2] * twiddle_cos[base + j+2] +
                temp_re[j+3] * twiddle_sin[base + j+3] + temp_im[j+3] * twiddle_cos[base + j+3];
    }
    // Handle remainder
    for (; j < n; j++) {
      sum_re += temp_re[j] * twiddle_cos[base + j] - temp_im[j] * twiddle_sin[base + j];
      sum_im += temp_re[j] * twiddle_sin[base + j] + temp_im[j] * twiddle_cos[base + j];
    }

    data_re[offset + k * stride] = sum_re;
    data_im[offset + k * stride] = sum_im;
  }

  free(temp_re); free(temp_im); free(twiddle_cos); free(twiddle_sin);
}

// FFT for complex32 (single precision)
static void fft_1d_complex32(float *data_re, float *data_im, size_t n,
                             int stride, size_t offset, bool inverse) {
  if (n == 0) return;  // Fix n=0 crash
  if (n <= 1) return;

  // Check if n is power of 2
  int log2n = 0;
  size_t temp = n;
  bool is_power_of_2 = true;
  while (temp > 1) {
    if (temp & 1) {
      is_power_of_2 = false;
      break;
    }
    temp >>= 1;
    log2n++;
  }

  if (!is_power_of_2) {
    // Not a power of 2, use DFT
    dft_complex32(data_re, data_im, n, stride, offset, inverse);
    return;
  }

  // Bit reversal permutation
  for (size_t i = 0; i < n; i++) {
    size_t j = bit_reverse(i, log2n);
    if (i < j) {
      size_t idx_i = offset + i * stride;
      size_t idx_j = offset + j * stride;

      float temp_re = data_re[idx_i];
      float temp_im = data_im[idx_i];
      data_re[idx_i] = data_re[idx_j];
      data_im[idx_i] = data_im[idx_j];
      data_re[idx_j] = temp_re;
      data_im[idx_j] = temp_im;
    }
  }

  // FFT computation
  float sign = inverse ? 1.0f : -1.0f;

  for (size_t stage_size = 2; stage_size <= n; stage_size *= 2) {
    size_t half_stage = stage_size / 2;
    float angle_step = sign * 2.0f * (float)M_PI / stage_size;

    for (size_t k = 0; k < n; k += stage_size) {
      for (size_t j = 0; j < half_stage; j++) {
        float angle = angle_step * j;
        float twiddle_re = cosf(angle);
        float twiddle_im = sinf(angle);

        size_t idx1 = offset + (k + j) * stride;
        size_t idx2 = offset + (k + j + half_stage) * stride;

        float a_re = data_re[idx1];
        float a_im = data_im[idx1];
        float b_re = data_re[idx2];
        float b_im = data_im[idx2];

        float b_twiddle_re = b_re * twiddle_re - b_im * twiddle_im;
        float b_twiddle_im = b_re * twiddle_im + b_im * twiddle_re;

        data_re[idx1] = a_re + b_twiddle_re;
        data_im[idx1] = a_im + b_twiddle_im;
        data_re[idx2] = a_re - b_twiddle_re;
        data_im[idx2] = a_im - b_twiddle_im;
      }
    }
  }

  // Note: scaling is now done after all axis transforms in fft_multi_complex32
}

// Multi-dimensional FFT for complex32
static void fft_multi_complex32(ndarray_t *data, int *axes, int num_axes,
                                bool inverse) {
  int ndim = data->ndim;
  const int *shape = data->shape;

  // Find maximum axis size for temp allocation
  size_t max_n = 0;
  for (int ax_idx = 0; ax_idx < num_axes; ax_idx++) {
    int axis = axes[ax_idx];
    if (axis < 0) axis += ndim;
    size_t n = shape[axis];
    if (n > max_n) max_n = n;
  }

  // Allocate temporary arrays once for all axes
  float *temp_re = (float *)malloc(max_n * sizeof(float));
  float *temp_im = (float *)malloc(max_n * sizeof(float));

  if (!temp_re || !temp_im) {
    free(temp_re);
    free(temp_im);
    return;
  }

  for (int ax_idx = 0; ax_idx < num_axes; ax_idx++) {
    int axis = axes[ax_idx];
    if (axis < 0) axis += ndim;

    size_t n = shape[axis];
    if (n <= 1) continue;

    size_t axis_stride = 1;
    for (int d = axis + 1; d < ndim; d++) {
      axis_stride *= shape[d];
    }

    size_t total_elements = 1;
    for (int d = 0; d < ndim; d++) {
      total_elements *= shape[d];
    }

    size_t num_ffts = total_elements / n;

    for (size_t fft_idx = 0; fft_idx < num_ffts; fft_idx++) {
      size_t offset = 0;
      size_t temp_idx = fft_idx;

      for (int d = ndim - 1; d >= 0; d--) {
        if (d != axis) {
          size_t dim_size = (d > axis) ? shape[d] : shape[d];
          size_t coord = temp_idx % dim_size;
          temp_idx /= dim_size;

          size_t stride_d = 1;
          for (int dd = d + 1; dd < ndim; dd++) {
            stride_d *= shape[dd];
          }
          offset += coord * stride_d;
        }
      }

      for (size_t i = 0; i < n; i++) {
        size_t idx = offset + i * axis_stride;
        temp_re[i] = ((float *)data->data)[2 * idx];
        temp_im[i] = ((float *)data->data)[2 * idx + 1];
      }

      fft_1d_complex32(temp_re, temp_im, n, 1, 0, inverse);

      for (size_t i = 0; i < n; i++) {
        size_t idx = offset + i * axis_stride;
        ((float *)data->data)[2 * idx] = temp_re[i];
        ((float *)data->data)[2 * idx + 1] = temp_im[i];
      }
    }
  }

  free(temp_re);
  free(temp_im);

  // Note: Scaling is handled by the frontend based on norm parameter
}

// FFT dispatch for complex32
CAMLprim value caml_nx_fft_complex32_bc(value *argv, int argn) {
  (void)argn;
  value v_ndim = argv[0], v_shape = argv[1];
  value v_input = argv[2], v_input_strides = argv[3], v_input_offset = argv[4];
  value v_output = argv[5], v_output_strides = argv[6],
        v_output_offset = argv[7];
  value v_axes = argv[8], v_num_axes = argv[9], v_inverse = argv[10];

  return caml_nx_fft_complex32(v_ndim, v_shape, v_input, v_input_strides,
                               v_input_offset, v_output, v_output_strides,
                               v_output_offset, v_axes, v_num_axes, v_inverse);
}

CAMLprim value caml_nx_fft_complex32(value v_ndim, value v_shape, value v_input,
                                     value v_input_strides,
                                     value v_input_offset, value v_output,
                                     value v_output_strides,
                                     value v_output_offset, value v_axes,
                                     value v_num_axes, value v_inverse) {
  int ndim = Int_val(v_ndim);
  int num_axes = Int_val(v_num_axes);
  bool inverse = Bool_val(v_inverse);

  int shape[ndim > 0 ? ndim : 1];
  int input_strides[ndim > 0 ? ndim : 1];
  int output_strides[ndim > 0 ? ndim : 1];
  int axes[num_axes > 0 ? num_axes : 1];

  for (int i = 0; i < ndim; ++i) {
    shape[i] = Int_val(Field(v_shape, i));
    input_strides[i] = Int_val(Field(v_input_strides, i));
    output_strides[i] = Int_val(Field(v_output_strides, i));
  }

  for (int i = 0; i < num_axes; ++i) {
    axes[i] = Int_val(Field(v_axes, i));
  }

  ndarray_t input = {
      .data = Caml_ba_data_val(v_input) + Long_val(v_input_offset),
      .shape = shape,
      .strides = input_strides,
      .ndim = ndim};

  ndarray_t output = {
      .data = Caml_ba_data_val(v_output) + Long_val(v_output_offset),
      .shape = shape,
      .strides = output_strides,
      .ndim = ndim};

  if (input.data != output.data) {
    size_t total_size = 1;
    for (int i = 0; i < ndim; i++) {
      total_size *= shape[i];
    }
    memcpy(output.data, input.data, total_size * 2 * sizeof(float));
  }

  caml_enter_blocking_section();
  fft_multi_complex32(&output, axes, num_axes, inverse);
  caml_leave_blocking_section();

  return Val_unit;
}

// RFFT dispatch for float64
CAMLprim value caml_nx_rfft_float64_bc(value *argv, int argn) {
  (void)argn;
  value v_ndim = argv[0], v_shape = argv[1];
  value v_input = argv[2], v_input_strides = argv[3], v_input_offset = argv[4];
  value v_output = argv[5], v_output_strides = argv[6],
        v_output_offset = argv[7];
  value v_axes = argv[8], v_num_axes = argv[9];

  return caml_nx_rfft_float64(v_ndim, v_shape, v_input, v_input_strides,
                              v_input_offset, v_output, v_output_strides,
                              v_output_offset, v_axes, v_num_axes);
}

CAMLprim value caml_nx_rfft_float64(value v_ndim, value v_shape, value v_input,
                                    value v_input_strides, value v_input_offset,
                                    value v_output, value v_output_strides,
                                    value v_output_offset, value v_axes,
                                    value v_num_axes) {
  int ndim = Int_val(v_ndim);
  int num_axes = Int_val(v_num_axes);

  int shape[ndim > 0 ? ndim : 1];
  int input_strides[ndim > 0 ? ndim : 1];
  int output_strides[ndim > 0 ? ndim : 1];
  int axes[num_axes > 0 ? num_axes : 1];

  for (int i = 0; i < ndim; ++i) {
    shape[i] = Int_val(Field(v_shape, i));
    input_strides[i] = Int_val(Field(v_input_strides, i));
    output_strides[i] = Int_val(Field(v_output_strides, i));
  }

  for (int i = 0; i < num_axes; ++i) {
    axes[i] = Int_val(Field(v_axes, i));
    if (axes[i] < 0) axes[i] += ndim;  // Normalize negative axes consistently
  }

  if (num_axes == 0) return Val_unit;  // Early exit for no axes

  // RFFT axis is the last one in axes list
  int rfft_axis = axes[num_axes - 1];

  // Output shape: only RFFT axis changes to n/2 + 1
  int output_shape[ndim > 0 ? ndim : 1];
  memcpy(output_shape, shape, ndim * sizeof(int));
  output_shape[rfft_axis] = (shape[rfft_axis] / 2) + 1;

  double *input_data = (double *)Caml_ba_data_val(v_input) + Long_val(v_input_offset);
  double *output_data = (double *)Caml_ba_data_val(v_output) + Long_val(v_output_offset);

  caml_enter_blocking_section();

  if (num_axes == 1) {
    // Single axis case (optimized, no extra complex buffer)
    size_t n = shape[rfft_axis];
    if (n == 0) { caml_leave_blocking_section(); return Val_unit; }  // Fix n=0 crash

    size_t n_out = output_shape[rfft_axis];
    size_t num_ffts = 1;
    for (int d = 0; d < ndim; d++) if (d != rfft_axis) num_ffts *= shape[d];

    double *temp_re = (double *)malloc(n * sizeof(double));  // Alloc outside loop
    double *temp_im = (double *)malloc(n * sizeof(double));
    if (!temp_re || !temp_im) {
      free(temp_re); free(temp_im);
      caml_leave_blocking_section();
      caml_failwith("rfft: memory allocation failed");
    }

    size_t axis_stride_in = 1, axis_stride_out = 1;
    for (int d = rfft_axis + 1; d < ndim; d++) {
      axis_stride_in *= shape[d];
      axis_stride_out *= output_shape[d];
    }

    for (size_t fft_idx = 0; fft_idx < num_ffts; fft_idx++) {
      size_t offset_in = 0, offset_out = 0, temp = fft_idx;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != rfft_axis) {
          size_t coord = temp % shape[d];
          temp /= shape[d];
          offset_in += coord * (size_t)input_strides[d];
          offset_out += coord * (size_t)output_strides[d];
        }
      }

      for (size_t i = 0; i < n; i++) {
        size_t idx = offset_in + i * axis_stride_in * input_strides[rfft_axis];
        temp_re[i] = input_data[idx];
        temp_im[i] = 0.0;
      }

      fft_1d_complex64(temp_re, temp_im, n, 1, 0, false);  // Forward FFT

      for (size_t i = 0; i < n_out; i++) {
        size_t idx = offset_out + i * axis_stride_out * output_strides[rfft_axis];
        output_data[2 * idx] = temp_re[i];
        output_data[2 * idx + 1] = temp_im[i];
      }
    }

    free(temp_re);
    free(temp_im);
  } else {
    // Multi-dimensional: first RFFT on last axis, then CFFT on previous
    // Alloc large complex buffer for intermediate (size based on output_shape)
    size_t inter_size = 1;
    for (int d = 0; d < ndim; d++) inter_size *= output_shape[d];
    double *inter_data = (double *)calloc(2 * inter_size, sizeof(double));  // Zero-init for imag
    if (!inter_data) {
      caml_leave_blocking_section();
      caml_failwith("rfft: memory allocation failed");
    }

    // Strides for intermediate (contiguous complex)
    int inter_strides[ndim > 0 ? ndim : 1];
    inter_strides[ndim - 1] = 2;  // Interleaved real/imag
    for (int d = ndim - 2; d >= 0; d--) {
      inter_strides[d] = inter_strides[d + 1] * output_shape[d + 1];
    }

    // Do 1D RFFT on last axis
    size_t n = shape[rfft_axis];
    size_t n_out = output_shape[rfft_axis];
    size_t num_1d = inter_size / n_out;

    double *temp_re = (double *)malloc(n * sizeof(double));
    double *temp_im = (double *)malloc(n * sizeof(double));
    if (!temp_re || !temp_im) {
      free(inter_data); free(temp_re); free(temp_im);
      caml_leave_blocking_section();
      caml_failwith("rfft: memory allocation failed");
    }

    for (size_t idx = 0; idx < num_1d; idx++) {
      size_t offset_in = 0, offset_inter = 0, temp = idx;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != rfft_axis) {
          size_t coord = temp % shape[d];
          temp /= shape[d];
          offset_in += coord * input_strides[d];
          offset_inter += coord * inter_strides[d];
        }
      }

      for (size_t i = 0; i < n; i++) {
        temp_re[i] = input_data[offset_in + i * input_strides[rfft_axis]];
        temp_im[i] = 0.0;
      }

      fft_1d_complex64(temp_re, temp_im, n, 1, 0, false);

      for (size_t i = 0; i < n_out; i++) {
        size_t idx_out = offset_inter + i * inter_strides[rfft_axis];
        inter_data[idx_out] = temp_re[i];
        inter_data[idx_out + 1] = temp_im[i];
      }
    }

    free(temp_re);
    free(temp_im);

    // Now CFFT on previous axes
    ndarray_t inter_array = {.data = inter_data, .shape = output_shape, .strides = inter_strides, .ndim = ndim};
    int prev_axes[num_axes - 1];
    for (int i = 0; i < num_axes - 1; i++) prev_axes[i] = axes[i];

    if (num_axes > 1) {
      fft_multi_complex64(&inter_array, prev_axes, num_axes - 1, false);
    }

    // Copy to final output (complex)
    for (size_t i = 0; i < inter_size; i++) {
      size_t offset_out = 0, temp = i;
      for (int d = ndim - 1; d >= 0; d--) {
        size_t coord = temp % output_shape[d];
        temp /= output_shape[d];
        offset_out += coord * output_strides[d];
      }
      output_data[2 * offset_out] = inter_data[2 * i];
      output_data[2 * offset_out + 1] = inter_data[2 * i + 1];
    }

    free(inter_data);
  }

  caml_leave_blocking_section();
  return Val_unit;
}

// IRFFT dispatch for complex64
CAMLprim value caml_nx_irfft_complex64_bc(value *argv, int argn) {
  (void)argn;
  value v_ndim = argv[0], v_shape = argv[1];
  value v_input = argv[2], v_input_strides = argv[3], v_input_offset = argv[4];
  value v_output = argv[5], v_output_strides = argv[6],
        v_output_offset = argv[7];
  value v_axes = argv[8], v_num_axes = argv[9], v_last_size = argv[10];

  return caml_nx_irfft_complex64(
      v_ndim, v_shape, v_input, v_input_strides, v_input_offset, v_output,
      v_output_strides, v_output_offset, v_axes, v_num_axes, v_last_size);
}

CAMLprim value caml_nx_irfft_complex64(value v_ndim, value v_shape,
                                       value v_input, value v_input_strides,
                                       value v_input_offset, value v_output,
                                       value v_output_strides,
                                       value v_output_offset, value v_axes,
                                       value v_num_axes, value v_last_size) {
  int ndim = Int_val(v_ndim);
  int num_axes = Int_val(v_num_axes);
  int last_size = Int_val(v_last_size);

  int shape[ndim > 0 ? ndim : 1];
  int input_strides[ndim > 0 ? ndim : 1];
  int output_strides[ndim > 0 ? ndim : 1];
  int axes[num_axes > 0 ? num_axes : 1];

  for (int i = 0; i < ndim; ++i) {
    shape[i] = Int_val(Field(v_shape, i));
    input_strides[i] = Int_val(Field(v_input_strides, i));
    output_strides[i] = Int_val(Field(v_output_strides, i));
  }

  for (int i = 0; i < num_axes; ++i) {
    axes[i] = Int_val(Field(v_axes, i));
    if (axes[i] < 0) axes[i] += ndim;  // Normalize negative axes
  }

  if (num_axes == 0) return Val_unit;  // Early exit

  // IRFFT axis is the last one in axes list
  int irfft_axis = axes[num_axes - 1];

  // Build output shape
  int output_shape[ndim > 0 ? ndim : 1];
  memcpy(output_shape, shape, ndim * sizeof(int));
  output_shape[irfft_axis] = last_size;

  double *input_data = (double *)Caml_ba_data_val(v_input) + Long_val(v_input_offset);
  double *output_data = (double *)Caml_ba_data_val(v_output) + Long_val(v_output_offset);

  caml_enter_blocking_section();

  if (num_axes == 1) {
    // Single axis case - optimized
    size_t n_in = shape[irfft_axis];
    size_t n_out = output_shape[irfft_axis];
    if (n_out == 0) { caml_leave_blocking_section(); return Val_unit; }

    size_t num_ffts = 1;
    for (int d = 0; d < ndim; d++) if (d != irfft_axis) num_ffts *= shape[d];

    double *temp_re = (double *)malloc(n_out * sizeof(double));
    double *temp_im = (double *)malloc(n_out * sizeof(double));
    if (!temp_re || !temp_im) {
      free(temp_re); free(temp_im);
      caml_leave_blocking_section();
      caml_failwith("irfft: memory allocation failed");
    }

    size_t axis_stride_in = 1, axis_stride_out = 1;
    for (int d = irfft_axis + 1; d < ndim; d++) {
      axis_stride_in *= shape[d];
      axis_stride_out *= output_shape[d];
    }

    for (size_t fft_idx = 0; fft_idx < num_ffts; fft_idx++) {
      size_t offset_in = 0, offset_out = 0, temp = fft_idx;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != irfft_axis) {
          size_t coord = temp % shape[d];
          temp /= shape[d];
          offset_in += coord * (size_t)input_strides[d];
          offset_out += coord * (size_t)output_strides[d];
        }
      }

      // Copy input and reconstruct Hermitian symmetry
      for (size_t i = 0; i < n_in; i++) {
        size_t idx = offset_in + i * axis_stride_in * input_strides[irfft_axis];
        temp_re[i] = input_data[2 * idx];
        temp_im[i] = input_data[2 * idx + 1];
      }

      // Fill the symmetric part
      for (size_t i = n_in; i < n_out; i++) {
        size_t mirror_idx = n_out - i;
        if (mirror_idx > 0 && mirror_idx < n_in) {
          temp_re[i] = temp_re[mirror_idx];
          temp_im[i] = -temp_im[mirror_idx];  // Complex conjugate
        } else {
          temp_re[i] = 0.0;
          temp_im[i] = 0.0;
        }
      }

      fft_1d_complex64(temp_re, temp_im, n_out, 1, 0, true);  // Inverse FFT

      // Copy real part to output
      for (size_t i = 0; i < n_out; i++) {
        size_t idx = offset_out + i * axis_stride_out * output_strides[irfft_axis];
        output_data[idx] = temp_re[i];
      }
    }

    free(temp_re);
    free(temp_im);
  } else {
    // Multi-dimensional: first CIFFT on previous axes, then IRFFT on last
    // Create intermediate complex buffer with IRFFT axis at half+1 size
    int inter_shape[ndim > 0 ? ndim : 1];
    memcpy(inter_shape, output_shape, ndim * sizeof(int));
    inter_shape[irfft_axis] = shape[irfft_axis];  // Use input shape for complex part
    
    size_t inter_size = 1;
    for (int d = 0; d < ndim; d++) inter_size *= inter_shape[d];
    
    double *inter_data = (double *)calloc(2 * inter_size, sizeof(double));
    if (!inter_data) {
      caml_leave_blocking_section();
      caml_failwith("irfft: memory allocation failed");
    }

    // Copy input to intermediate buffer
    for (size_t i = 0; i < inter_size; i++) {
      size_t offset = 0, temp = i;
      for (int d = ndim - 1; d >= 0; d--) {
        size_t coord = temp % inter_shape[d];
        temp /= inter_shape[d];
        offset += coord * input_strides[d];
      }
      inter_data[2 * i] = input_data[2 * offset];
      inter_data[2 * i + 1] = input_data[2 * offset + 1];
    }

    // Strides for intermediate (contiguous complex)
    int inter_strides[ndim > 0 ? ndim : 1];
    inter_strides[ndim - 1] = 2;
    for (int d = ndim - 2; d >= 0; d--) {
      inter_strides[d] = inter_strides[d + 1] * inter_shape[d + 1];
    }

    // Do CIFFT on previous axes first
    if (num_axes > 1) {
      ndarray_t inter_array = {.data = inter_data, .shape = inter_shape, .strides = inter_strides, .ndim = ndim};
      int prev_axes[num_axes - 1];
      for (int i = 0; i < num_axes - 1; i++) prev_axes[i] = axes[i];
      
      fft_multi_complex64(&inter_array, prev_axes, num_axes - 1, true);
    }

    // Now do IRFFT on last axis
    size_t n_in = shape[irfft_axis];
    size_t n_out = output_shape[irfft_axis];
    size_t num_1d = inter_size / n_in;

    double *temp_re = (double *)malloc(n_out * sizeof(double));
    double *temp_im = (double *)malloc(n_out * sizeof(double));
    if (!temp_re || !temp_im) {
      free(inter_data); free(temp_re); free(temp_im);
      caml_leave_blocking_section();
      caml_failwith("irfft: memory allocation failed");
    }

    for (size_t idx = 0; idx < num_1d; idx++) {
      size_t offset_inter = 0, offset_out = 0, temp = idx;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != irfft_axis) {
          size_t coord = temp % output_shape[d];
          temp /= output_shape[d];
          offset_inter += coord * inter_strides[d];
          offset_out += coord * output_strides[d];
        }
      }

      // Extract complex data
      for (size_t i = 0; i < n_in; i++) {
        size_t idx_in = offset_inter + i * inter_strides[irfft_axis];
        temp_re[i] = inter_data[idx_in];
        temp_im[i] = inter_data[idx_in + 1];
      }

      // Reconstruct Hermitian symmetry
      for (size_t i = n_in; i < n_out; i++) {
        size_t mirror_idx = n_out - i;
        if (mirror_idx > 0 && mirror_idx < n_in) {
          temp_re[i] = temp_re[mirror_idx];
          temp_im[i] = -temp_im[mirror_idx];
        } else {
          temp_re[i] = 0.0;
          temp_im[i] = 0.0;
        }
      }

      fft_1d_complex64(temp_re, temp_im, n_out, 1, 0, true);

      // Copy real part to output
      for (size_t i = 0; i < n_out; i++) {
        output_data[offset_out + i * output_strides[irfft_axis]] = temp_re[i];
      }
    }

    free(temp_re);
    free(temp_im);
    free(inter_data);
  }

  caml_leave_blocking_section();
  return Val_unit;
}

// RFFT dispatch for float32
CAMLprim value caml_nx_rfft_float32_bc(value *argv, int argn) {
  (void)argn;
  value v_ndim = argv[0], v_shape = argv[1];
  value v_input = argv[2], v_input_strides = argv[3], v_input_offset = argv[4];
  value v_output = argv[5], v_output_strides = argv[6],
        v_output_offset = argv[7];
  value v_axes = argv[8], v_num_axes = argv[9];

  return caml_nx_rfft_float32(v_ndim, v_shape, v_input, v_input_strides,
                              v_input_offset, v_output, v_output_strides,
                              v_output_offset, v_axes, v_num_axes);
}

CAMLprim value caml_nx_rfft_float32(value v_ndim, value v_shape, value v_input,
                                    value v_input_strides, value v_input_offset,
                                    value v_output, value v_output_strides,
                                    value v_output_offset, value v_axes,
                                    value v_num_axes) {
  int ndim = Int_val(v_ndim);
  int num_axes = Int_val(v_num_axes);

  int shape[ndim > 0 ? ndim : 1];
  int input_strides[ndim > 0 ? ndim : 1];
  int output_strides[ndim > 0 ? ndim : 1];
  int axes[num_axes > 0 ? num_axes : 1];

  for (int i = 0; i < ndim; ++i) {
    shape[i] = Int_val(Field(v_shape, i));
    input_strides[i] = Int_val(Field(v_input_strides, i));
    output_strides[i] = Int_val(Field(v_output_strides, i));
  }

  for (int i = 0; i < num_axes; ++i) {
    axes[i] = Int_val(Field(v_axes, i));
    if (axes[i] < 0) axes[i] += ndim;  // Normalize negative axes consistently
  }

  if (num_axes == 0) return Val_unit;  // Early exit for no axes

  // RFFT axis is the last one in axes list
  int rfft_axis = axes[num_axes - 1];

  // Output shape: only RFFT axis changes to n/2 + 1
  int output_shape[ndim > 0 ? ndim : 1];
  memcpy(output_shape, shape, ndim * sizeof(int));
  output_shape[rfft_axis] = (shape[rfft_axis] / 2) + 1;

  float *input_data = (float *)Caml_ba_data_val(v_input) + Long_val(v_input_offset);
  float *output_data = (float *)Caml_ba_data_val(v_output) + Long_val(v_output_offset);

  caml_enter_blocking_section();

  if (num_axes == 1) {
    // Single axis case (optimized, no extra complex buffer)
    size_t n = shape[rfft_axis];
    if (n == 0) { caml_leave_blocking_section(); return Val_unit; }  // Fix n=0 crash

    size_t n_out = output_shape[rfft_axis];
    size_t num_ffts = 1;
    for (int d = 0; d < ndim; d++) if (d != rfft_axis) num_ffts *= shape[d];

    float *temp_re = (float *)malloc(n * sizeof(float));  // Alloc outside loop
    float *temp_im = (float *)malloc(n * sizeof(float));
    if (!temp_re || !temp_im) {
      free(temp_re); free(temp_im);
      caml_leave_blocking_section();
      caml_failwith("rfft: memory allocation failed");
    }

    size_t axis_stride_in = 1, axis_stride_out = 1;
    for (int d = rfft_axis + 1; d < ndim; d++) {
      axis_stride_in *= shape[d];
      axis_stride_out *= output_shape[d];
    }

    for (size_t fft_idx = 0; fft_idx < num_ffts; fft_idx++) {
      size_t offset_in = 0, offset_out = 0, temp = fft_idx;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != rfft_axis) {
          size_t coord = temp % shape[d];
          temp /= shape[d];
          offset_in += coord * (size_t)input_strides[d];
          offset_out += coord * (size_t)output_strides[d];
        }
      }

      for (size_t i = 0; i < n; i++) {
        size_t idx = offset_in + i * axis_stride_in * input_strides[rfft_axis];
        temp_re[i] = input_data[idx];
        temp_im[i] = 0.0f;
      }

      fft_1d_complex32(temp_re, temp_im, n, 1, 0, false);  // Forward FFT

      for (size_t i = 0; i < n_out; i++) {
        size_t idx = offset_out + i * axis_stride_out * output_strides[rfft_axis];
        output_data[2 * idx] = temp_re[i];
        output_data[2 * idx + 1] = temp_im[i];
      }
    }

    free(temp_re);
    free(temp_im);
  } else {
    // Multi-dimensional: first RFFT on last axis, then CFFT on previous
    // Alloc large complex buffer for intermediate (size based on output_shape)
    size_t inter_size = 1;
    for (int d = 0; d < ndim; d++) inter_size *= output_shape[d];
    float *inter_data = (float *)calloc(2 * inter_size, sizeof(float));  // Zero-init for imag
    if (!inter_data) {
      caml_leave_blocking_section();
      caml_failwith("rfft: memory allocation failed");
    }

    // Strides for intermediate (contiguous complex)
    int inter_strides[ndim > 0 ? ndim : 1];
    inter_strides[ndim - 1] = 2;  // Interleaved real/imag
    for (int d = ndim - 2; d >= 0; d--) {
      inter_strides[d] = inter_strides[d + 1] * output_shape[d + 1];
    }

    // Do 1D RFFT on last axis
    size_t n = shape[rfft_axis];
    size_t n_out = output_shape[rfft_axis];
    size_t num_1d = inter_size / n_out;

    float *temp_re = (float *)malloc(n * sizeof(float));
    float *temp_im = (float *)malloc(n * sizeof(float));
    if (!temp_re || !temp_im) {
      free(inter_data); free(temp_re); free(temp_im);
      caml_leave_blocking_section();
      caml_failwith("rfft: memory allocation failed");
    }

    for (size_t idx = 0; idx < num_1d; idx++) {
      size_t offset_in = 0, offset_inter = 0, temp = idx;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != rfft_axis) {
          size_t coord = temp % shape[d];
          temp /= shape[d];
          offset_in += coord * input_strides[d];
          offset_inter += coord * inter_strides[d];
        }
      }

      for (size_t i = 0; i < n; i++) {
        temp_re[i] = input_data[offset_in + i * input_strides[rfft_axis]];
        temp_im[i] = 0.0f;
      }

      fft_1d_complex32(temp_re, temp_im, n, 1, 0, false);

      for (size_t i = 0; i < n_out; i++) {
        size_t idx_out = offset_inter + i * inter_strides[rfft_axis];
        inter_data[idx_out] = temp_re[i];
        inter_data[idx_out + 1] = temp_im[i];
      }
    }

    free(temp_re);
    free(temp_im);

    // Now CFFT on previous axes
    ndarray_t inter_array = {.data = inter_data, .shape = output_shape, .strides = inter_strides, .ndim = ndim};
    int prev_axes[num_axes - 1];
    for (int i = 0; i < num_axes - 1; i++) prev_axes[i] = axes[i];

    if (num_axes > 1) {
      fft_multi_complex32(&inter_array, prev_axes, num_axes - 1, false);
    }

    // Copy to final output (complex)
    for (size_t i = 0; i < inter_size; i++) {
      size_t offset_out = 0, temp = i;
      for (int d = ndim - 1; d >= 0; d--) {
        size_t coord = temp % output_shape[d];
        temp /= output_shape[d];
        offset_out += coord * output_strides[d];
      }
      output_data[2 * offset_out] = inter_data[2 * i];
      output_data[2 * offset_out + 1] = inter_data[2 * i + 1];
    }

    free(inter_data);
  }

  caml_leave_blocking_section();
  return Val_unit;
}

// IRFFT dispatch for complex32
CAMLprim value caml_nx_irfft_complex32_bc(value *argv, int argn) {
  (void)argn;
  value v_ndim = argv[0], v_shape = argv[1];
  value v_input = argv[2], v_input_strides = argv[3], v_input_offset = argv[4];
  value v_output = argv[5], v_output_strides = argv[6],
        v_output_offset = argv[7];
  value v_axes = argv[8], v_num_axes = argv[9], v_last_size = argv[10];

  return caml_nx_irfft_complex32(
      v_ndim, v_shape, v_input, v_input_strides, v_input_offset, v_output,
      v_output_strides, v_output_offset, v_axes, v_num_axes, v_last_size);
}

CAMLprim value caml_nx_irfft_complex32(value v_ndim, value v_shape,
                                       value v_input, value v_input_strides,
                                       value v_input_offset, value v_output,
                                       value v_output_strides,
                                       value v_output_offset, value v_axes,
                                       value v_num_axes, value v_last_size) {
  int ndim = Int_val(v_ndim);
  int num_axes = Int_val(v_num_axes);
  int last_size = Int_val(v_last_size);

  int shape[ndim > 0 ? ndim : 1];
  int input_strides[ndim > 0 ? ndim : 1];
  int output_strides[ndim > 0 ? ndim : 1];
  int axes[num_axes > 0 ? num_axes : 1];

  for (int i = 0; i < ndim; ++i) {
    shape[i] = Int_val(Field(v_shape, i));
    input_strides[i] = Int_val(Field(v_input_strides, i));
    output_strides[i] = Int_val(Field(v_output_strides, i));
  }

  for (int i = 0; i < num_axes; ++i) {
    axes[i] = Int_val(Field(v_axes, i));
    if (axes[i] < 0) axes[i] += ndim;  // Normalize negative axes
  }

  if (num_axes == 0) return Val_unit;  // Early exit

  // IRFFT axis is the last one in axes list
  int irfft_axis = axes[num_axes - 1];

  // Build output shape
  int output_shape[ndim > 0 ? ndim : 1];
  memcpy(output_shape, shape, ndim * sizeof(int));
  output_shape[irfft_axis] = last_size;

  float *input_data = (float *)Caml_ba_data_val(v_input) + Long_val(v_input_offset);
  float *output_data = (float *)Caml_ba_data_val(v_output) + Long_val(v_output_offset);

  caml_enter_blocking_section();

  if (num_axes == 1) {
    // Single axis case - optimized
    size_t n_in = shape[irfft_axis];
    size_t n_out = output_shape[irfft_axis];
    if (n_out == 0) { caml_leave_blocking_section(); return Val_unit; }

    size_t num_ffts = 1;
    for (int d = 0; d < ndim; d++) if (d != irfft_axis) num_ffts *= shape[d];

    float *temp_re = (float *)malloc(n_out * sizeof(float));
    float *temp_im = (float *)malloc(n_out * sizeof(float));
    if (!temp_re || !temp_im) {
      free(temp_re); free(temp_im);
      caml_leave_blocking_section();
      caml_failwith("irfft: memory allocation failed");
    }

    size_t axis_stride_in = 1, axis_stride_out = 1;
    for (int d = irfft_axis + 1; d < ndim; d++) {
      axis_stride_in *= shape[d];
      axis_stride_out *= output_shape[d];
    }

    for (size_t fft_idx = 0; fft_idx < num_ffts; fft_idx++) {
      size_t offset_in = 0, offset_out = 0, temp = fft_idx;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != irfft_axis) {
          size_t coord = temp % shape[d];
          temp /= shape[d];
          offset_in += coord * (size_t)input_strides[d];
          offset_out += coord * (size_t)output_strides[d];
        }
      }

      // Copy input and reconstruct Hermitian symmetry
      for (size_t i = 0; i < n_in; i++) {
        size_t idx = offset_in + i * axis_stride_in * input_strides[irfft_axis];
        temp_re[i] = input_data[2 * idx];
        temp_im[i] = input_data[2 * idx + 1];
      }

      // Fill the symmetric part
      for (size_t i = n_in; i < n_out; i++) {
        size_t mirror_idx = n_out - i;
        if (mirror_idx > 0 && mirror_idx < n_in) {
          temp_re[i] = temp_re[mirror_idx];
          temp_im[i] = -temp_im[mirror_idx];  // Complex conjugate
        } else {
          temp_re[i] = 0.0f;
          temp_im[i] = 0.0f;
        }
      }

      fft_1d_complex32(temp_re, temp_im, n_out, 1, 0, true);  // Inverse FFT

      // Copy real part to output
      for (size_t i = 0; i < n_out; i++) {
        size_t idx = offset_out + i * axis_stride_out * output_strides[irfft_axis];
        output_data[idx] = temp_re[i];
      }
    }

    free(temp_re);
    free(temp_im);
  } else {
    // Multi-dimensional: first CIFFT on previous axes, then IRFFT on last
    // Create intermediate complex buffer with IRFFT axis at half+1 size
    int inter_shape[ndim > 0 ? ndim : 1];
    memcpy(inter_shape, output_shape, ndim * sizeof(int));
    inter_shape[irfft_axis] = shape[irfft_axis];  // Use input shape for complex part
    
    size_t inter_size = 1;
    for (int d = 0; d < ndim; d++) inter_size *= inter_shape[d];
    
    float *inter_data = (float *)calloc(2 * inter_size, sizeof(float));
    if (!inter_data) {
      caml_leave_blocking_section();
      caml_failwith("irfft: memory allocation failed");
    }

    // Copy input to intermediate buffer
    for (size_t i = 0; i < inter_size; i++) {
      size_t offset = 0, temp = i;
      for (int d = ndim - 1; d >= 0; d--) {
        size_t coord = temp % inter_shape[d];
        temp /= inter_shape[d];
        offset += coord * input_strides[d];
      }
      inter_data[2 * i] = input_data[2 * offset];
      inter_data[2 * i + 1] = input_data[2 * offset + 1];
    }

    // Strides for intermediate (contiguous complex)
    int inter_strides[ndim > 0 ? ndim : 1];
    inter_strides[ndim - 1] = 2;
    for (int d = ndim - 2; d >= 0; d--) {
      inter_strides[d] = inter_strides[d + 1] * inter_shape[d + 1];
    }

    // Do CIFFT on previous axes first
    if (num_axes > 1) {
      ndarray_t inter_array = {.data = inter_data, .shape = inter_shape, .strides = inter_strides, .ndim = ndim};
      int prev_axes[num_axes - 1];
      for (int i = 0; i < num_axes - 1; i++) prev_axes[i] = axes[i];
      
      fft_multi_complex32(&inter_array, prev_axes, num_axes - 1, true);
    }

    // Now do IRFFT on last axis
    size_t n_in = shape[irfft_axis];
    size_t n_out = output_shape[irfft_axis];
    size_t num_1d = inter_size / n_in;

    float *temp_re = (float *)malloc(n_out * sizeof(float));
    float *temp_im = (float *)malloc(n_out * sizeof(float));
    if (!temp_re || !temp_im) {
      free(inter_data); free(temp_re); free(temp_im);
      caml_leave_blocking_section();
      caml_failwith("irfft: memory allocation failed");
    }

    for (size_t idx = 0; idx < num_1d; idx++) {
      size_t offset_inter = 0, offset_out = 0, temp = idx;
      for (int d = ndim - 1; d >= 0; d--) {
        if (d != irfft_axis) {
          size_t coord = temp % output_shape[d];
          temp /= output_shape[d];
          offset_inter += coord * inter_strides[d];
          offset_out += coord * output_strides[d];
        }
      }

      // Extract complex data
      for (size_t i = 0; i < n_in; i++) {
        size_t idx_in = offset_inter + i * inter_strides[irfft_axis];
        temp_re[i] = inter_data[idx_in];
        temp_im[i] = inter_data[idx_in + 1];
      }

      // Reconstruct Hermitian symmetry
      for (size_t i = n_in; i < n_out; i++) {
        size_t mirror_idx = n_out - i;
        if (mirror_idx > 0 && mirror_idx < n_in) {
          temp_re[i] = temp_re[mirror_idx];
          temp_im[i] = -temp_im[mirror_idx];
        } else {
          temp_re[i] = 0.0f;
          temp_im[i] = 0.0f;
        }
      }

      fft_1d_complex32(temp_re, temp_im, n_out, 1, 0, true);

      // Copy real part to output
      for (size_t i = 0; i < n_out; i++) {
        output_data[offset_out + i * output_strides[irfft_axis]] = temp_re[i];
      }
    }

    free(temp_re);
    free(temp_im);
    free(inter_data);
  }

  caml_leave_blocking_section();
  return Val_unit;
}
