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

// DFT implementation for arbitrary sizes
static void dft_complex64(double *data_re, double *data_im, size_t n,
                          int stride, size_t offset, bool inverse) {
  double *temp_re = (double *)malloc(n * sizeof(double));
  double *temp_im = (double *)malloc(n * sizeof(double));

  if (!temp_re || !temp_im) {
    free(temp_re);
    free(temp_im);
    return;
  }

  double sign = inverse ? 1.0 : -1.0;
  double two_pi_n = 2.0 * M_PI / n;

  // Copy input data
  for (size_t i = 0; i < n; i++) {
    temp_re[i] = data_re[offset + i * stride];
    temp_im[i] = data_im[offset + i * stride];
  }

  // Compute DFT
  for (size_t k = 0; k < n; k++) {
    double sum_re = 0.0;
    double sum_im = 0.0;

    for (size_t j = 0; j < n; j++) {
      double angle = sign * two_pi_n * k * j;
      double cos_angle = cos(angle);
      double sin_angle = sin(angle);

      sum_re += temp_re[j] * cos_angle - temp_im[j] * sin_angle;
      sum_im += temp_re[j] * sin_angle + temp_im[j] * cos_angle;
    }

    data_re[offset + k * stride] = sum_re;
    data_im[offset + k * stride] = sum_im;
  }

  free(temp_re);
  free(temp_im);
}

// 1D FFT implementation using Cooley-Tukey algorithm
static void fft_1d_complex64(double *data_re, double *data_im, size_t n,
                             int stride, size_t offset, bool inverse) {
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

    // Allocate temporary arrays for real and imaginary parts
    double *temp_re = (double *)malloc(n * sizeof(double));
    double *temp_im = (double *)malloc(n * sizeof(double));

    if (!temp_re || !temp_im) {
      free(temp_re);
      free(temp_im);
      return;
    }

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

    free(temp_re);
    free(temp_im);
  }

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

// DFT implementation for arbitrary sizes (single precision)
static void dft_complex32(float *data_re, float *data_im, size_t n, int stride,
                          size_t offset, bool inverse) {
  float *temp_re = (float *)malloc(n * sizeof(float));
  float *temp_im = (float *)malloc(n * sizeof(float));

  if (!temp_re || !temp_im) {
    free(temp_re);
    free(temp_im);
    return;
  }

  float sign = inverse ? 1.0f : -1.0f;
  float two_pi_n = 2.0f * (float)M_PI / n;

  // Copy input data
  for (size_t i = 0; i < n; i++) {
    temp_re[i] = data_re[offset + i * stride];
    temp_im[i] = data_im[offset + i * stride];
  }

  // Compute DFT
  for (size_t k = 0; k < n; k++) {
    float sum_re = 0.0f;
    float sum_im = 0.0f;

    for (size_t j = 0; j < n; j++) {
      float angle = sign * two_pi_n * k * j;
      float cos_angle = cosf(angle);
      float sin_angle = sinf(angle);

      sum_re += temp_re[j] * cos_angle - temp_im[j] * sin_angle;
      sum_im += temp_re[j] * sin_angle + temp_im[j] * cos_angle;
    }

    data_re[offset + k * stride] = sum_re;
    data_im[offset + k * stride] = sum_im;
  }

  free(temp_re);
  free(temp_im);
}

// FFT for complex32 (single precision)
static void fft_1d_complex32(float *data_re, float *data_im, size_t n,
                             int stride, size_t offset, bool inverse) {
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

    float *temp_re = (float *)malloc(n * sizeof(float));
    float *temp_im = (float *)malloc(n * sizeof(float));

    if (!temp_re || !temp_im) {
      free(temp_re);
      free(temp_im);
      return;
    }

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

    free(temp_re);
    free(temp_im);
  }

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

  int input_shape[ndim > 0 ? ndim : 1];
  int input_strides[ndim > 0 ? ndim : 1];
  int output_strides[ndim > 0 ? ndim : 1];
  int axes[num_axes > 0 ? num_axes : 1];

  for (int i = 0; i < ndim; ++i) {
    input_shape[i] = Int_val(Field(v_shape, i));
    input_strides[i] = Int_val(Field(v_input_strides, i));
    output_strides[i] = Int_val(Field(v_output_strides, i));
  }

  for (int i = 0; i < num_axes; ++i) {
    axes[i] = Int_val(Field(v_axes, i));
  }

  // Handle multi-dimensional RFFT
  // For multi-dimensional RFFT, we perform complex FFT on all axes except the
  // last, then RFFT on the last axis

  // Determine which axis will get the RFFT (last in the axes list)
  int rfft_axis = axes[num_axes - 1];
  if (rfft_axis < 0) rfft_axis += ndim;

  // Calculate output shape - only the RFFT axis changes size
  int output_shape[ndim > 0 ? ndim : 1];
  memcpy(output_shape, input_shape, ndim * sizeof(int));
  output_shape[rfft_axis] = (input_shape[rfft_axis] / 2) + 1;

  // If we have multiple axes, we need to do complex FFTs first
  if (num_axes > 1) {
    // Create a complex copy of the real input
    size_t total_size = 1;
    for (int d = 0; d < ndim; d++) {
      total_size *= input_shape[d];
    }

    double *complex_data = (double *)malloc(total_size * 2 * sizeof(double));
    if (!complex_data) {
      caml_failwith("rfft: memory allocation failed");
    }

    // Copy real data to complex format with proper striding
    double *real_input =
        (double *)Caml_ba_data_val(v_input) + Long_val(v_input_offset);
    for (size_t i = 0; i < total_size; i++) {
      // Calculate the multi-dimensional index
      size_t temp_idx = i;
      size_t offset = 0;
      for (int d = ndim - 1; d >= 0; d--) {
        size_t coord = temp_idx % input_shape[d];
        temp_idx /= input_shape[d];
        offset += coord * input_strides[d];
      }
      complex_data[2 * i] = real_input[offset];
      complex_data[2 * i + 1] = 0.0;
    }

    // Create strides for complex data (contiguous)
    int complex_strides[ndim > 0 ? ndim : 1];
    complex_strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; d--) {
      complex_strides[d] = complex_strides[d + 1] * input_shape[d + 1];
    }

    // Create ndarray for complex data
    ndarray_t complex_array = {.data = complex_data,
                               .shape = input_shape,
                               .strides = complex_strides,
                               .ndim = ndim};

    // Perform complex FFT on all axes except the last
    int complex_axes[num_axes - 1];
    for (int i = 0; i < num_axes - 1; i++) {
      complex_axes[i] = axes[i];
    }

    caml_enter_blocking_section();
    if (num_axes > 1) {
      fft_multi_complex64(&complex_array, complex_axes, num_axes - 1, false);
    }

    // Now perform RFFT on the last axis
    size_t n = input_shape[rfft_axis];
    size_t n_out = (n / 2) + 1;

    // Calculate number of 1D RFFTs to perform
    size_t num_rffts = total_size / n;

    // Allocate temporary arrays for single RFFT
    double *temp_re = (double *)malloc(n * sizeof(double));
    double *temp_im = (double *)malloc(n * sizeof(double));

    if (!temp_re || !temp_im) {
      free(complex_data);
      free(temp_re);
      free(temp_im);
      caml_leave_blocking_section();
      caml_failwith("rfft: memory allocation failed");
    }

    // Get output pointer
    double *output_data =
        (double *)Caml_ba_data_val(v_output) + Long_val(v_output_offset);

    // Process each 1D RFFT along the RFFT axis
    for (size_t rfft_idx = 0; rfft_idx < num_rffts; rfft_idx++) {
      // Calculate base offset for this RFFT
      size_t base_offset_in = 0;
      size_t base_offset_out = 0;
      size_t temp_idx = rfft_idx;

      for (int d = ndim - 1; d >= 0; d--) {
        if (d != rfft_axis) {
          size_t dim_size = input_shape[d];
          size_t coord = temp_idx % dim_size;
          temp_idx /= dim_size;

          base_offset_in += coord * complex_strides[d];

          // Calculate output offset with modified shape
          size_t out_stride = 1;
          for (int dd = d + 1; dd < ndim; dd++) {
            out_stride *= (dd == rfft_axis) ? n_out : input_shape[dd];
          }
          base_offset_out += coord * out_stride;
        }
      }

      // Copy complex data to temp arrays
      for (size_t i = 0; i < n; i++) {
        size_t idx = base_offset_in + i * complex_strides[rfft_axis];
        temp_re[i] = complex_data[2 * idx];
        temp_im[i] = complex_data[2 * idx + 1];
      }

      // Always perform FFT on the last axis
      fft_1d_complex64(temp_re, temp_im, n, 1, 0, false);

      // Copy only the non-redundant part to output
      size_t out_stride = 1;
      for (int d = rfft_axis + 1; d < ndim; d++) {
        out_stride *= (d == rfft_axis) ? n_out : input_shape[d];
      }

      for (size_t i = 0; i < n_out; i++) {
        size_t idx = base_offset_out + i * out_stride;
        output_data[2 * idx] = temp_re[i];
        output_data[2 * idx + 1] = temp_im[i];
      }
    }

    caml_leave_blocking_section();

    free(complex_data);
    free(temp_re);
    free(temp_im);

    return Val_unit;
  }

  // Single axis RFFT (original code)
  int axis = axes[0];
  if (axis < 0) axis += ndim;

  size_t n = input_shape[axis];
  size_t n_out = (n / 2) + 1;

  // Calculate total number of FFTs to perform
  size_t total_elements = 1;
  for (int d = 0; d < ndim; d++) {
    total_elements *= input_shape[d];
  }
  size_t num_ffts = total_elements / n;

  ndarray_t input = {
      .data = Caml_ba_data_val(v_input) + Long_val(v_input_offset),
      .shape = input_shape,
      .strides = input_strides,
      .ndim = ndim};

  // Output shape has n_out elements on the transform axis
  int output_shape2[ndim > 0 ? ndim : 1];
  memcpy(output_shape2, input_shape, ndim * sizeof(int));
  output_shape2[axis] = n_out;

  ndarray_t output = {
      .data = Caml_ba_data_val(v_output) + Long_val(v_output_offset),
      .shape = output_shape2,
      .strides = output_strides,
      .ndim = ndim};

  // Allocate temporary arrays
  double *temp_re = (double *)malloc(n * sizeof(double));
  double *temp_im = (double *)malloc(n * sizeof(double));

  if (!temp_re || !temp_im) {
    free(temp_re);
    free(temp_im);
    caml_failwith("rfft: memory allocation failed");
  }

  caml_enter_blocking_section();

  // Calculate stride for the FFT axis
  size_t axis_stride_in = 1;
  size_t axis_stride_out = 1;
  for (int d = axis + 1; d < ndim; d++) {
    axis_stride_in *= input_shape[d];
    axis_stride_out *= output_shape2[d];
  }

  // Process each 1D RFFT
  for (size_t fft_idx = 0; fft_idx < num_ffts; fft_idx++) {
    // Calculate starting offset
    size_t offset_in = 0;
    size_t offset_out = 0;
    size_t temp_idx = fft_idx;

    for (int d = ndim - 1; d >= 0; d--) {
      if (d != axis) {
        size_t dim_size = input_shape[d];
        size_t coord = temp_idx % dim_size;
        temp_idx /= dim_size;

        size_t stride_in = 1;
        size_t stride_out = 1;
        for (int dd = d + 1; dd < ndim; dd++) {
          stride_in *= input_shape[dd];
          stride_out *= output_shape2[dd];
        }
        offset_in += coord * stride_in;
        offset_out += coord * stride_out;
      }
    }

    // Copy real input to temp arrays
    for (size_t i = 0; i < n; i++) {
      temp_re[i] = ((double *)input.data)[offset_in + i * axis_stride_in];
      temp_im[i] = 0.0;
    }

    // Perform FFT
    fft_1d_complex64(temp_re, temp_im, n, 1, 0, false);

    // Copy only the non-redundant part to output
    for (size_t i = 0; i < n_out; i++) {
      size_t idx = offset_out + i * axis_stride_out;
      ((double *)output.data)[2 * idx] = temp_re[i];
      ((double *)output.data)[2 * idx + 1] = temp_im[i];
    }
  }

  caml_leave_blocking_section();

  free(temp_re);
  free(temp_im);

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

  int input_shape[ndim > 0 ? ndim : 1];
  int input_strides[ndim > 0 ? ndim : 1];
  int output_strides[ndim > 0 ? ndim : 1];
  int axes[num_axes > 0 ? num_axes : 1];

  for (int i = 0; i < ndim; ++i) {
    input_shape[i] = Int_val(Field(v_shape, i));
    input_strides[i] = Int_val(Field(v_input_strides, i));
    output_strides[i] = Int_val(Field(v_output_strides, i));
  }

  for (int i = 0; i < num_axes; ++i) {
    axes[i] = Int_val(Field(v_axes, i));
  }

  // Handle multi-dimensional IRFFT
  // For multi-dimensional IRFFT, we perform IRFFT on the last axis first,
  // then complex IFFT on all other axes

  // Determine which axis will get the IRFFT (last in the axes list)
  int irfft_axis = axes[num_axes - 1];
  if (irfft_axis < 0) irfft_axis += ndim;

  // Build output shape
  int output_shape[ndim > 0 ? ndim : 1];
  memcpy(output_shape, input_shape, ndim * sizeof(int));
  output_shape[irfft_axis] = last_size;

  if (num_axes > 1) {
    // Multi-dimensional case
    // First, allocate complex buffer for intermediate result
    size_t total_complex_size = 1;
    for (int d = 0; d < ndim; d++) {
      total_complex_size *= output_shape[d];
    }

    double *complex_data =
        (double *)malloc(total_complex_size * 2 * sizeof(double));
    if (!complex_data) {
      caml_failwith("irfft: memory allocation failed");
    }

    // Initialize to zero
    memset(complex_data, 0, total_complex_size * 2 * sizeof(double));

    // Create strides for complex data (contiguous)
    int complex_strides[ndim > 0 ? ndim : 1];
    complex_strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; d--) {
      complex_strides[d] = complex_strides[d + 1] * output_shape[d + 1];
    }

    caml_enter_blocking_section();

    // Perform IRFFT on the last axis
    size_t n_in = input_shape[irfft_axis];
    size_t n_out = output_shape[irfft_axis];
    size_t num_irffts = total_complex_size / n_out;

    // Allocate temporary arrays for single IRFFT
    double *temp_re = (double *)malloc(n_out * sizeof(double));
    double *temp_im = (double *)malloc(n_out * sizeof(double));

    if (!temp_re || !temp_im) {
      free(complex_data);
      free(temp_re);
      free(temp_im);
      caml_leave_blocking_section();
      caml_failwith("irfft: memory allocation failed");
    }

    double *input_data =
        (double *)Caml_ba_data_val(v_input) + Long_val(v_input_offset);

    // Process each 1D IRFFT along the IRFFT axis
    for (size_t irfft_idx = 0; irfft_idx < num_irffts; irfft_idx++) {
      // Calculate base offset
      size_t base_offset_in = 0;
      size_t base_offset_out = 0;
      size_t temp_idx = irfft_idx;

      for (int d = ndim - 1; d >= 0; d--) {
        if (d != irfft_axis) {
          size_t dim_size = output_shape[d];
          size_t coord = temp_idx % dim_size;
          temp_idx /= dim_size;

          // Input offset calculation
          size_t in_stride = 1;
          for (int dd = d + 1; dd < ndim; dd++) {
            in_stride *= input_shape[dd];
          }
          base_offset_in += coord * in_stride;

          // Output offset
          base_offset_out += coord * complex_strides[d];
        }
      }

      // Copy input Hermitian data
      size_t in_stride = 1;
      for (int d = irfft_axis + 1; d < ndim; d++) {
        in_stride *= input_shape[d];
      }

      for (size_t i = 0; i < n_in; i++) {
        size_t idx = base_offset_in + i * in_stride;
        temp_re[i] = input_data[2 * idx];
        temp_im[i] = input_data[2 * idx + 1];
      }

      // Reconstruct full Hermitian symmetry
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

      // Perform IFFT
      fft_1d_complex64(temp_re, temp_im, n_out, 1, 0, true);

      // Copy to complex buffer
      for (size_t i = 0; i < n_out; i++) {
        size_t idx = base_offset_out + i * complex_strides[irfft_axis];
        complex_data[2 * idx] = temp_re[i];
        complex_data[2 * idx + 1] = temp_im[i];
      }
    }

    free(temp_re);
    free(temp_im);

    // Now perform complex IFFTs on remaining axes
    if (num_axes > 1) {
      ndarray_t complex_array = {.data = complex_data,
                                 .shape = output_shape,
                                 .strides = complex_strides,
                                 .ndim = ndim};

      int complex_axes[num_axes - 1];
      for (int i = 0; i < num_axes - 1; i++) {
        complex_axes[i] = axes[i];
        if (complex_axes[i] < 0) complex_axes[i] += ndim;
      }

      fft_multi_complex64(&complex_array, complex_axes, num_axes - 1, true);
    }

    // Copy real parts to output
    double *output_data =
        (double *)Caml_ba_data_val(v_output) + Long_val(v_output_offset);
    for (size_t i = 0; i < total_complex_size; i++) {
      // Calculate multi-dimensional index and output offset
      size_t temp_idx = i;
      size_t offset = 0;
      for (int d = ndim - 1; d >= 0; d--) {
        size_t coord = temp_idx % output_shape[d];
        temp_idx /= output_shape[d];
        offset += coord * output_strides[d];
      }
      output_data[offset] = complex_data[2 * i];
    }

    caml_leave_blocking_section();

    free(complex_data);
    return Val_unit;
  }

  int axis = axes[0];
  if (axis < 0) axis += ndim;

  size_t n_in = input_shape[axis];
  size_t n = last_size;

  // Calculate total number of IFFTs to perform
  size_t total_elements = 1;
  for (int d = 0; d < ndim; d++) {
    if (d == axis) {
      total_elements *= n;
    } else {
      total_elements *= input_shape[d];
    }
  }
  size_t num_ffts = total_elements / n;

  ndarray_t input = {
      .data = Caml_ba_data_val(v_input) + Long_val(v_input_offset),
      .shape = input_shape,
      .strides = input_strides,
      .ndim = ndim};

  // Output shape has n elements on the transform axis
  int output_shape_1d[ndim > 0 ? ndim : 1];
  memcpy(output_shape_1d, input_shape, ndim * sizeof(int));
  output_shape_1d[axis] = n;

  ndarray_t output = {
      .data = Caml_ba_data_val(v_output) + Long_val(v_output_offset),
      .shape = output_shape_1d,
      .strides = output_strides,
      .ndim = ndim};

  // Allocate temporary arrays
  double *temp_re = (double *)malloc(n * sizeof(double));
  double *temp_im = (double *)malloc(n * sizeof(double));

  if (!temp_re || !temp_im) {
    free(temp_re);
    free(temp_im);
    caml_failwith("irfft: memory allocation failed");
  }

  caml_enter_blocking_section();

  // Calculate stride for the FFT axis
  size_t axis_stride_in = 1;
  size_t axis_stride_out = 1;
  for (int d = axis + 1; d < ndim; d++) {
    axis_stride_in *= input_shape[d];
    axis_stride_out *= output_shape_1d[d];
  }

  // Process each 1D IRFFT
  for (size_t fft_idx = 0; fft_idx < num_ffts; fft_idx++) {
    // Calculate starting offset
    size_t offset_in = 0;
    size_t offset_out = 0;
    size_t temp_idx = fft_idx;

    for (int d = ndim - 1; d >= 0; d--) {
      if (d != axis) {
        size_t dim_size = input_shape[d];
        size_t coord = temp_idx % dim_size;
        temp_idx /= dim_size;

        size_t stride_in = 1;
        size_t stride_out = 1;
        for (int dd = d + 1; dd < ndim; dd++) {
          stride_in *= input_shape[dd];
          stride_out *= output_shape_1d[dd];
        }
        offset_in += coord * stride_in;
        offset_out += coord * stride_out;
      }
    }

    // Copy input and reconstruct Hermitian symmetry
    for (size_t i = 0; i < n_in; i++) {
      size_t idx = offset_in + i * axis_stride_in;
      temp_re[i] = ((double *)input.data)[2 * idx];
      temp_im[i] = ((double *)input.data)[2 * idx + 1];
    }

    // Fill the symmetric part
    for (size_t i = n_in; i < n; i++) {
      size_t mirror_idx = n - i;
      if (mirror_idx > 0 && mirror_idx < n_in) {
        temp_re[i] = temp_re[mirror_idx];
        temp_im[i] = -temp_im[mirror_idx];  // Complex conjugate
      } else {
        temp_re[i] = 0.0;
        temp_im[i] = 0.0;
      }
    }

    // Perform inverse FFT
    fft_1d_complex64(temp_re, temp_im, n, 1, 0, true);

    // Copy real part to output
    for (size_t i = 0; i < n; i++) {
      ((double *)output.data)[offset_out + i * axis_stride_out] = temp_re[i];
    }
  }

  caml_leave_blocking_section();

  free(temp_re);
  free(temp_im);

  return Val_unit;
}
