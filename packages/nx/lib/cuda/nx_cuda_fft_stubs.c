/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The tables of the transforms' plans (fft.ml, kernels.h's Fourier
   transforms): roots of unity, Bluestein's chirp and kernel, Rader's
   kernel and logarithms, returned as bigarrays of their bytes, complex
   numbers of two floats or two doubles.

   Each is computed in double-double arithmetic from IEEE double
   additions, multiplications and divisions alone, compiled without
   contraction (dune), so its bits are the same on every host: no libm
   function enters a table. A root of unity is its angle's sine and cosine
   by their Taylor series; a long table's roots are products of two of
   them; a kernel is a double-double FFT of a 7-smooth length. Each value
   is within a few 2^-100 relative of the exact one, then rounded once to
   the plan's precision. The larger tables are computed with the runtime
   released. */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>

/* Double-double arithmetic: a value hi + lo, |lo| <= ulp(hi) / 2 */

typedef struct {
  double hi, lo;
} dd;

static dd two_sum(double a, double b) {
  double s = a + b, bb = s - a;
  dd r = {s, (a - (s - bb)) + (b - bb)};
  return r;
}

static dd fast_two_sum(double a, double b) {
  double s = a + b;
  dd r = {s, b - (s - a)};
  return r;
}

/* Dekker's exact product, without a fused multiply-add. */
static dd two_prod(double a, double b) {
  const double split = 134217729.0; /* 2^27 + 1 */
  double p = a * b;
  double ta = split * a, ah = ta - (ta - a), al = a - ah;
  double tb = split * b, bh = tb - (tb - b), bl = b - bh;
  dd r = {p, ((ah * bh - p) + ah * bl + al * bh) + al * bl};
  return r;
}

static dd dd_add(dd a, dd b) {
  dd s = two_sum(a.hi, b.hi), t = two_sum(a.lo, b.lo);
  s = fast_two_sum(s.hi, s.lo + t.hi);
  return fast_two_sum(s.hi, s.lo + t.lo);
}

static dd dd_neg(dd a) {
  dd r = {-a.hi, -a.lo};
  return r;
}

static dd dd_mul(dd a, dd b) {
  dd p = two_prod(a.hi, b.hi);
  return fast_two_sum(p.hi, p.lo + (a.hi * b.lo + a.lo * b.hi));
}

/* a / d for a double d. */
static dd dd_div(dd a, double d) {
  double q1 = a.hi / d;
  dd p = two_prod(q1, d);
  double q2 = (((a.hi - p.hi) - p.lo) + a.lo) / d;
  return fast_two_sum(q1, q2);
}

static dd dd_of(double x) {
  dd r = {x, 0};
  return r;
}

typedef struct {
  dd r, i;
} cdd;

static cdd cmul(cdd a, cdd b) {
  cdd c = {dd_add(dd_mul(a.r, b.r), dd_neg(dd_mul(a.i, b.i))),
           dd_add(dd_mul(a.r, b.i), dd_mul(a.i, b.r))};
  return c;
}

static cdd cadd(cdd a, cdd b) {
  cdd c = {dd_add(a.r, b.r), dd_add(a.i, b.i)};
  return c;
}

static cdd csub(cdd a, cdd b) {
  cdd c = {dd_add(a.r, dd_neg(b.r)), dd_add(a.i, dd_neg(b.i))};
  return c;
}

static cdd cconj(cdd a) {
  cdd c = {a.r, dd_neg(a.i)};
  return c;
}

/* Roots of unity */

/* sin t and cos t for 0 <= t <= π/4, by their series: past 30 terms,
   t^k / k! is below 2^-110. */
static void sincos_dd(dd t, dd *s, dd *c) {
  dd term = dd_of(1), sum_s = dd_of(0), sum_c = dd_of(0);
  for (int k = 0; k < 30; k++) {
    switch (k % 4) {
    case 0: sum_c = dd_add(sum_c, term); break;
    case 1: sum_s = dd_add(sum_s, term); break;
    case 2: sum_c = dd_add(sum_c, dd_neg(term)); break;
    default: sum_s = dd_add(sum_s, dd_neg(term)); break;
    }
    term = dd_div(dd_mul(term, t), (double)(k + 1));
  }
  *s = sum_s;
  *c = sum_c;
}

/* e^(-2πi k/n), n < 2^51: the angle within its quarter turn taken from
   the nearer end, then turned by whole quarters, so that it is exact at
   the multiples of n / 8. */
static cdd omega(uint64_t k, uint64_t n) {
  static const dd half_pi = {1.5707963267948966, 6.123233995736766e-17};
  k %= n;
  uint64_t k4 = 4 * k, q = k4 / n, r = k4 - q * n;
  int far = 2 * r > n;
  dd s, c, cr, sr;
  dd t = dd_mul(half_pi, dd_div(dd_of((double)(far ? n - r : r)), (double)n));
  sincos_dd(t, &s, &c);
  if (far) {
    dd x = s;
    s = c;
    c = x;
  }
  /* cos and sin of q π/2 + t */
  switch (q) {
  case 0: cr = c, sr = s; break;
  case 1: cr = dd_neg(s), sr = c; break;
  case 2: cr = dd_neg(c), sr = dd_neg(s); break;
  default: cr = s, sr = dd_neg(c); break;
  }
  cdd w = {cr, dd_neg(sr)};
  return w;
}

/* ω_n^k, k < n, by two tables: lo[k mod 2^bits] hi[k >> bits]. */
typedef struct {
  int bits;
  cdd *lo, *hi;
} roots2;

static int roots2_make(roots2 *t, uint64_t n) {
  int b = 0;
  while (((uint64_t)1 << b) < n) b++;
  t->bits = (b + 1) / 2;
  uint64_t nl = (uint64_t)1 << t->bits, nh = (n + nl - 1) / nl;
  t->lo = malloc(nl * sizeof *t->lo);
  t->hi = malloc(nh * sizeof *t->hi);
  if (t->lo == NULL || t->hi == NULL) {
    free(t->lo);
    free(t->hi);
    return 0;
  }
  for (uint64_t i = 0; i < nl; i++) t->lo[i] = omega(i, n);
  for (uint64_t i = 0; i < nh; i++) t->hi[i] = omega(i * nl, n);
  return 1;
}

static cdd roots2_at(const roots2 *t, uint64_t k) {
  return cmul(t->lo[k & (((uint64_t)1 << t->bits) - 1)], t->hi[k >> t->bits]);
}

static void roots2_free(roots2 *t) {
  free(t->lo);
  free(t->hi);
}

/* The forward DFT of a 7-smooth length */

static int64_t least_factor(int64_t n) {
  for (int64_t p = 2; p <= 7; p++)
    if (n % p == 0) return p;
  return n;
}

/* out[k] = Σ_j in[j is] w^(jk) for n points, w = W[wstep] the n-th root
   of a table of N roots: decimation in time by the least prime. */
static void dd_rec(const cdd *in, int64_t n, int64_t is, cdd *out, const cdd *W, int64_t N,
                   int64_t wstep) {
  if (n == 1) {
    out[0] = in[0];
    return;
  }
  int64_t p = least_factor(n), m = n / p;
  cdd t[7];
  for (int64_t s = 0; s < p; s++) dd_rec(in + s * is, m, is * p, out + s * m, W, N, wstep * p);
  for (int64_t k = 0; k < m; k++) {
    t[0] = out[k];
    for (int64_t s = 1; s < p; s++) t[s] = cmul(out[s * m + k], W[s * k * wstep % N]);
    if (p == 2) {
      out[k] = cadd(t[0], t[1]);
      out[k + m] = csub(t[0], t[1]);
      continue;
    }
    for (int64_t q = 0; q < p; q++) {
      cdd acc = t[0];
      for (int64_t s = 1; s < p; s++) acc = cadd(acc, cmul(t[s], W[s * q * m * wstep % N]));
      out[k + m * q] = acc;
    }
  }
}

/* The forward DFT of [a]'s n points into [out]; 0 where memory ran out. */
static int dd_fft(const cdd *a, int64_t n, cdd *out) {
  cdd *W = malloc((size_t)n * sizeof *W);
  roots2 t;
  if (W == NULL || !roots2_make(&t, (uint64_t)n)) {
    free(W);
    return 0;
  }
  for (int64_t k = 0; k < n; k++) W[k] = roots2_at(&t, (uint64_t)k);
  roots2_free(&t);
  dd_rec(a, n, 1, out, W, n, 1);
  free(W);
  return 1;
}

/* Rounding */

/* [x] rounded to a float: hi rounded, but where hi lies halfway between
   two floats, the one lo points to. */
static float round_float(dd x) {
  float f = (float)x.hi;
  double d = x.hi - (double)f;
  if (d != 0 && x.lo != 0) {
    double g = (double)f + 2 * d;
    if ((double)(float)g == g && (d > 0) == (x.lo > 0)) f = (float)g;
  }
  return f;
}

typedef unsigned char byte;

/* [n] complex numbers divided by [scale], rounded to floats or doubles, in
   a buffer of their bytes, at least 16. */
static byte *rounded(const cdd *v, int64_t n, int f64, double scale) {
  size_t bytes = (size_t)n * (f64 ? 16 : 8);
  byte *b = malloc(bytes < 16 ? 16 : bytes);
  if (b == NULL) return NULL;
  for (int64_t i = 0; i < n; i++) {
    dd re = scale == 1 ? v[i].r : dd_div(v[i].r, scale);
    dd im = scale == 1 ? v[i].i : dd_div(v[i].i, scale);
    if (f64) {
      double *d = (double *)b;
      d[2 * i] = re.hi;
      d[2 * i + 1] = im.hi;
    } else {
      float *f = (float *)b;
      f[2 * i] = round_float(re);
      f[2 * i + 1] = round_float(im);
    }
  }
  return b;
}

/* The bigarray of [bytes] bytes at [b], which it frees. */
static value bigbytes(byte *b, int64_t bytes) {
  return caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT | CAML_BA_MANAGED, 1, b,
                            (intnat)bytes);
}

/* Tables */

/* ω_n^(start + i step) for i < count. */
CAMLprim value nx_cuda_fft_roots(value vf64, value vn, value vstart, value vstep,
                                 value vcount) {
  int f64 = Bool_val(vf64);
  uint64_t n = Long_val(vn), start = Long_val(vstart), step = Long_val(vstep);
  int64_t count = Long_val(vcount);
  cdd *v = malloc((size_t)(count > 0 ? count : 1) * sizeof *v);
  if (v == NULL) caml_raise_out_of_memory();
  for (int64_t i = 0; i < count; i++) v[i] = omega(start + (uint64_t)i * step, n);
  byte *b = rounded(v, count, f64, 1);
  free(v);
  if (b == NULL) caml_raise_out_of_memory();
  return bigbytes(b, count * (f64 ? 16 : 8));
}

/* Bluestein's chirp: w_j = e^(-πi j²/n) = ω_2n^(j² mod 2n), j < n, into
   [w]; 0 where memory ran out. j² mod 2n grows by 2j + 1 a step, so no
   product passes 64 bits. */
static int chirp_of(cdd *w, int64_t n) {
  uint64_t q = 0, m = 2 * (uint64_t)n;
  roots2 t;
  if (!roots2_make(&t, m)) return 0;
  for (int64_t j = 0; j < n; j++) {
    w[j] = roots2_at(&t, q);
    q = (q + 2 * (uint64_t)j + 1) % m;
  }
  roots2_free(&t);
  return 1;
}

CAMLprim value nx_cuda_fft_chirp(value vf64, value vn) {
  int f64 = Bool_val(vf64);
  int64_t n = Long_val(vn);
  cdd *w = malloc((size_t)n * sizeof *w);
  byte *b = NULL;
  if (w == NULL) caml_raise_out_of_memory();
  caml_enter_blocking_section();
  if (chirp_of(w, n)) b = rounded(w, n, f64, 1);
  caml_leave_blocking_section();
  free(w);
  if (b == NULL) caml_raise_out_of_memory();
  return bigbytes(b, n * (f64 ? 16 : 8));
}

/* Bluestein's kernel for n points convolved at m >= 2n - 1: the transform
   B of b, b_j = b_(m-j) = conj w_j for j < n and 0 between, divided by
   m. */
CAMLprim value nx_cuda_fft_bluestein(value vf64, value vn, value vm) {
  int f64 = Bool_val(vf64);
  int64_t n = Long_val(vn), m = Long_val(vm);
  cdd *a = calloc((size_t)m, sizeof *a), *out = malloc((size_t)m * sizeof *out);
  byte *b = NULL;
  int ok = a != NULL && out != NULL;
  caml_enter_blocking_section();
  if (ok) ok = chirp_of(out, n);
  if (ok) {
    for (int64_t j = 0; j < n; j++) {
      a[j] = cconj(out[j]);
      if (j) a[m - j] = a[j];
    }
    ok = dd_fft(a, m, out);
  }
  if (ok) b = rounded(out, m, f64, (double)m);
  caml_leave_blocking_section();
  free(a);
  free(out);
  if (b == NULL) caml_raise_out_of_memory();
  return bigbytes(b, m * (f64 ? 16 : 8));
}

static int64_t powmod(int64_t b, int64_t e, int64_t n) {
  int64_t r = 1, x = b % n;
  for (; e; e >>= 1, x = x * x % n)
    if (e & 1) r = r * x % n;
  return r;
}

/* Rader's tables for a prime n < 2^31 whose m = n - 1 is 7-smooth, g the
   least primitive root: the transform of b_s = ω_n^(g^-s), s < m, divided
   by m, and the discrete logarithms lg[g^s mod n] = s as int32. */
CAMLprim value nx_cuda_fft_rader(value vf64, value vn) {
  CAMLparam2(vf64, vn);
  CAMLlocal3(pair, kernel, logs);
  int f64 = Bool_val(vf64);
  int64_t n = Long_val(vn), m = n - 1, g = 2;
  cdd *a = malloc((size_t)m * sizeof *a), *out = malloc((size_t)m * sizeof *out);
  int32_t *lg = calloc((size_t)n, sizeof *lg);
  byte *b = NULL;
  int ok = a != NULL && out != NULL && lg != NULL;
  caml_enter_blocking_section();
  if (ok) {
    for (;; g++) {
      int primitive = 1;
      for (int64_t q = 2; q <= 7; q++)
        if (m % q == 0 && powmod(g, m / q, n) == 1) primitive = 0;
      if (primitive) break;
    }
    int64_t gi = powmod(g, m - 1, n), p = 1, pi = 1; /* g^s, g^-s */
    for (int64_t s = 0; s < m; s++) {
      lg[p] = (int32_t)s;
      a[s] = omega((uint64_t)pi, (uint64_t)n);
      p = p * g % n;
      pi = pi * gi % n;
    }
    ok = dd_fft(a, m, out);
  }
  if (ok) {
    b = rounded(out, m, f64, (double)m);
    ok = b != NULL;
  }
  caml_leave_blocking_section();
  free(a);
  free(out);
  if (!ok) {
    free(lg);
    caml_raise_out_of_memory();
  }
  kernel = bigbytes(b, m * (f64 ? 16 : 8));
  logs = bigbytes((byte *)lg, n * 4);
  pair = caml_alloc_tuple(2);
  Store_field(pair, 0, kernel);
  Store_field(pair, 1, logs);
  CAMLreturn(pair);
}
