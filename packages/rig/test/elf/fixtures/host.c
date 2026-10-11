/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Host code as the host loader takes it: a call to a function another
   object defines, a counter in .bss, constants in .rodata, a helper the
   kernels call and math functions the loader resolves by name. */

void ext(int);
static int counter;
static const int table[4] = {1, 2, 3, 4};
void f(int i) { ext(i); counter += table[i & 3]; }

float expf(float);
float logf(float);
float tanhf(float);

static const float coeffs[8] = {0.5f,  0.044715f, 0.7978846f, 1.0f,
                                -1.0f, 2.0f,      0.25f,      3.0f};

__attribute__((noinline)) static float gelu1(float x) {
  return coeffs[0] * x *
         (coeffs[3] + tanhf(coeffs[2] * (x + coeffs[1] * x * x * x)));
}

void gelu(void **b, const long long *v) {
  float *out = b[0];
  const float *in = b[1];
  for (long long i = 0; i < v[0]; i++) out[i] = gelu1(in[i]);
}

void softmax(void **b, const long long *v) {
  float *out = b[0];
  const float *in = b[1];
  long long rows = v[0], cols = v[1];
  for (long long r = 0; r < rows; r++) {
    const float *x = in + r * cols;
    float *y = out + r * cols;
    float m = x[0], s = 0.0f;
    for (long long c = 1; c < cols; c++) m = x[c] > m ? x[c] : m;
    for (long long c = 0; c < cols; c++) s += (y[c] = expf(x[c] - m));
    for (long long c = 0; c < cols; c++) y[c] /= s;
  }
}

void log_softmax(void **b, const long long *v) {
  float *out = b[0];
  const float *in = b[1];
  long long rows = v[0], cols = v[1];
  for (long long r = 0; r < rows; r++) {
    const float *x = in + r * cols;
    float m = x[0], s = 0.0f;
    for (long long c = 1; c < cols; c++) m = x[c] > m ? x[c] : m;
    for (long long c = 0; c < cols; c++) s += expf(x[c] - m);
    float l = logf(s) + m;
    for (long long c = 0; c < cols; c++) out[r * cols + c] = x[c] - l;
  }
}

void matmul(void **b, const long long *v) {
  float *c = b[0];
  const float *a = b[1], *bt = b[2];
  long long m = v[0], n = v[1], k = v[2];
  for (long long i = 0; i < m; i++)
    for (long long j = 0; j < n; j++) {
      float acc = 0.0f;
      for (long long p = 0; p < k; p++) acc += a[i * k + p] * bt[j * k + p];
      c[i * n + j] = acc;
    }
}

void poly(void **b, const long long *v) {
  float *out = b[0];
  const float *in = b[1];
  for (long long i = 0; i < v[0]; i++) {
    float x = in[i], y = coeffs[7];
    for (int d = 6; d >= 0; d--) y = y * x + coeffs[d];
    out[i] = y;
  }
}

void scale_shift(void **b, const long long *v) {
  float *out = b[0];
  const float *in = b[1];
  for (long long i = 0; i < v[0]; i++)
    out[i] = in[i] * coeffs[(i & 3) + 4] + coeffs[i & 3];
}
