/*---------------------------------------------------------------------------
   Copyright (C) 2010-2019 Max-Planck-Society (pocketfft). BSD-3-Clause (see
   dev/nx2/LICENSE-pocketfft).
   Copyright (c) 2026 The Raven authors. All rights reserved.

   SPDX-License-Identifier: BSD-3-Clause AND ISC
  ---------------------------------------------------------------------------*/

/* Fourier transforms: Nx_kernel.S's c2c, r2c and c2r.

   A transform runs along one axis at a time, over lines: the points along
   that axis at one index of the others. A line of n points has a plan,
   which depends on n alone:

     a prime factor        Bluestein's convolution, by two transforms of
     > BLUESTEIN_PRIME     m >= 2n - 1 points
     n <= DIRECT_POINTS    passes of radix 2, 3, 4, 5 and odd primes
     or n prime            (pocketfft's), L lines a vector, one a lane
     else                  Bailey's four steps: n = n1 n2, columns of n1
                           points by passes, a twiddle, rows of n2

   A pass computes L lines at once, a lane each, in vectors of L
   numbers: lines side by side, or L points of one line, or, for the four
   steps, L columns then L rows of one line. Which lines share a vector
   changes no bit: a line's arithmetic is a function of n alone. The
   inverse is the conjugate of the forward transform of the conjugate,
   read and written so. Real lines of even n >= HALVED_POINTS go through
   complex ones of n / 2 points, others through complex ones of n.

     x ──gather──► lanes ──passes──► lanes ──scatter──► y
                     (L × L tiles transposed where points lie side by side)

   Plans are kept for the next call in a cache of PLANS, shared by every
   thread. A job's units are lines where there are enough for its
   threads; else each line's steps are jobs of their own. */

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "cpu.h"

/* The most points a line computes by passes alone. Above, the four steps
   give a single line every lane: a line of 256 points took 0.7 us by them
   against 4.9 by passes, which fill one lane of eight. A batch fills the
   lanes either way, and 64 lines of 256 took 26 us by the four steps
   against 19 by passes (M1, one thread); they take the four steps all the
   same, since a line's arithmetic depends on n alone. */
#define DIRECT_POINTS 64

/* The most passes of a plan: one a factor of n < 2^64. */
#define MAX_PASSES 64

/* The points a unit of a job of elementwise steps covers: enough units
   for every thread's chunks. At 4096, the split of an r2c of 2^16 points
   had 5 units and took 39 us on the job's threads against 28 at 1024;
   256 and 512 were no faster from 2^13 to 2^20 points (M1). */
#define CHUNK 1024

/* The plans kept from call to call, and the most bytes they hold. A
   plan costs a sine and a cosine a point to make: a first transform of
   1024 points took 27 us against 1.9 after it, of 2^20 points 9.3 ms
   against 4.6 (M1, one thread). A transform holds a plan an axis, and
   r2c or c2r one more, so 16 keep those of several transforms used in
   turn; a call scans them twice, under a lock. A four-step plan holds 8
   bytes a point in single precision, Bluestein's 40 to 72: 128 MiB keeps
   the plan of a line of 2^24 complex64 points. */
#define PLANS 16
#define PLAN_BYTES (128 * 1024 * 1024)

/* How a plan computes a line; ANY asks for the plan of n's choice. */
enum { PASSES, FOUR, BLUESTEIN, ANY };

/* A plan of complex lines, or of real ones. */
enum { COMPLEX, REAL };

/* The steps of a job. */
enum {
  LINES,
  COLUMNS,
  ROWS,
  CHIRP,
  CONVOLVE,
  UNCHIRP,
  SPLIT,
  MERGE,
  COMPLETE,
  HALVE
};

/* Lines

   A set of lines of points: point j of line l has its real part at
   element at(l) + j·step from [base], counted in the precision's reals,
   and its imaginary part [io] further, at(l) the position of line l in C
   order of the [rank] extents [dim] stepping [stride]. A [real] line has
   no imaginary part: it reads +0 and writes none. A [conj] line reads and
   writes conjugates. */
typedef struct {
  uint8_t *base;
  int64_t step, io;
  int real, conj, rank;
  int64_t dim[NX_MAX_RANK], stride[NX_MAX_RANK];
} lines;

static int64_t line_at(const lines *x, int64_t l) {
  int64_t at = 0;
  for (int i = x->rank - 1; i >= 0; i--) {
    at += l % x->dim[i] * x->stride[i];
    l /= x->dim[i];
  }
  return at;
}

/* Whether the [lanes] lines at [o] lie side by side, complex numbers
   apart. */
static int side_by_side(const lines *x, const int64_t *o, int lanes) {
  if (x->real || x->io != 1) return 0;
  for (int v = 1; v < lanes; v++)
    if (o[v] != o[0] + 2 * v) return 0;
  return 1;
}

/* Whether lines [l, l + lanes) lie side by side, on [x]'s last axis. */
static int lines_side_by_side(const lines *x, int64_t l, int lanes) {
  int r = x->rank - 1;
  return !x->real && x->io == 1 && r >= 0 && x->stride[r] == 2 &&
         l % x->dim[r] + lanes <= x->dim[r];
}

static int64_t numel(const nx_array *a) {
  int64_t n = 1;
  for (int i = 0; i < a->rank; i++) n *= a->dim[i];
  return n;
}

/* The axes of [a] but [axis] into [order], by decreasing stride: the last,
   the lines' innermost, steps least. */
static void others(const nx_array *a, int axis, int *order) {
  int k = 0;
  for (int i = 0; i < a->rank; i++) {
    if (i == axis) continue;
    int64_t s = llabs(a->dim[a->rank + i]);
    int j = k++;
    for (; j > 0 && llabs(a->dim[a->rank + order[j - 1]]) < s; j--)
      order[j] = order[j - 1];
    order[j] = i;
  }
}

/* The lines of [a] along [axis], its other axes in [order], [w] reals an
   element. */
static lines lines_of(const nx_array *a, int axis, const int *order, int w) {
  lines x = {.base = a->base + a->offset * (a->bits / 8),
             .step = w * a->dim[a->rank + axis],
             .io = 1,
             .rank = a->rank - 1};
  for (int i = 0; i < x.rank; i++) {
    x.dim[i] = a->dim[order[i]];
    x.stride[i] = w * a->dim[a->rank + order[i]];
  }
  return x;
}

/* Sets every element of [a] to +0. */
static void zero(const nx_array *a) {
  int64_t n = numel(a), w = a->bits / 8;
  for (int64_t e = 0; e < n; e++) {
    int64_t at = a->offset, r = e;
    for (int i = a->rank - 1; i >= 0; i--) {
      at += r % a->dim[i] * a->dim[a->rank + i];
      r /= a->dim[i];
    }
    memset(a->base + at * w, 0, (size_t)w);
  }
}

/* Unit [u] of a job of [per] chunks a line, of [n] points each: its line,
   and its points [*a, *b). */
static int64_t chunk(int64_t u, int64_t per, int64_t n, int64_t *a,
                     int64_t *b) {
  *a = u % per * CHUNK;
  *b = *a + CHUNK < n ? *a + CHUNK : n;
  return u / per;
}

/* The cost (job.c) of transforms over [bytes] of lines of [n] points by
   passes or the four steps, or by Bluestein's convolution of [m] points
   (m 0 otherwise). One thread took 0.9 to 1.5 log₂ n times a copy of the
   lines from 2^10 to 2^20 points; Bluestein's computes two transforms of
   m points a line, and took 1.2 times that for lines of 1009 points
   (M1). */
static double cost_of(int64_t bytes, int64_t n, int64_t m) {
  double w = m > 0 ? 2 * (double)m / (double)n * log2((double)m)
                   : log2((double)n);
  return (double)bytes * (w > 1 ? w : 1);
}

/* The cost of the steps between real lines and complex ones of half as
   many points, in copies of their bytes: r2c and c2r took 3 to 10 times
   such a copy's time more than c2c of half as many points, from 2^12 to
   2^20 points (M1, one thread). */
#define HALF_COST 6

/* The fewest points of an even real line that goes through complex lines
   of half as many. Below, the split and merge steps left c2r's error up
   to 1.25 times the expansion's (8 and 16 points), and the line goes
   through complex lines of n points, as an odd n's does, for 1.4 to 2.5
   times the time from 8 to 256 points (M1, one thread). From 512 points,
   c2r's error is 0.95 to 0.99 times the expansion's. */
#define HALVED_POINTS 512

static int halved(int64_t n) { return n % 2 == 0 && n >= HALVED_POINTS; }

/* Numbers */

/* e^(-2πi m / n), 0 <= m < n, within about an ulp of double: θ = 2π m / n
   is (π/2) (q + r / n) in integers, its angle within the quarter taken
   from the nearer end, at most π/4, and turned by q quarters exactly. */
static void unit(int64_t m, int64_t n, double *re, double *im) {
  int64_t q = 4 * m / n, r = 4 * m - q * n;
  int far = 2 * r > n;
  double a = M_PI_2 * (double)(far ? n - r : r) / (double)n;
  double c = cos(a), s = sin(a);
  if (far) {
    double t = c;
    c = s;
    s = t;
  }
  double x[4][2] = {{c, s}, {-s, c}, {-c, -s}, {s, -c}};
  *re = x[q][0];
  *im = -x[q][1];
}

/* The points of Bluestein's convolution for n points: the least power
   of two at least 2n - 1. At pocketfft's choice, the least product of 2,
   3, 5, 7 and 11, the error exceeded the expansion's, whose convolution
   runs at a power of two: float64's up to 1.5 times (c2r of 32771
   points: 6.0 against 4.1 in units of u √N ‖x‖₂; of 2062: 4.7 against
   3.4, and 3.9 with the kernel and the chirp in double-double), and
   float32's c2r of even n by 2 to 5% (514 to 8198 points). At a power of
   two the error is 0.4 to 0.95 times the expansion's, for 1.3 to 2.2
   times the time from 257 to 131101 points (M1, one thread). */
static int64_t blue_points(int64_t n) {
  int64_t m = 1;
  while (m < 2 * n - 1) m *= 2;
  return m;
}

/* The passes pocketfft's factorisation gives n. */
static int passes_of(int64_t n) {
  int k = 0;
  for (; n % 4 == 0; n /= 4) k++;
  for (int64_t d = 2; d * d <= n; d++)
    for (; n % d == 0; n /= d) k++;
  return k + (n > 1);
}

static int prime(int64_t n) {
  if (n < 2) return 0;
  for (int64_t d = 2; d * d <= n; d++)
    if (n % d == 0) return 0;
  return 1;
}

/* Whether a line of n points goes by Bluestein: where a prime factor of
   n exceeds BLUESTEIN_PRIME, which a pass of that radix computes in time
   proportional to it. Against passes on lines of p, 16 p and 1024 p
   points, Bluestein's took 0.98 to 1.38 the time at p = 199, 0.62 to
   0.87 at 251, 0.80 to 1.43 at 307, 0.49 to 0.87 at 401 and 0.33 to
   0.58 at 503 (M1, one thread, loaded). */
#define BLUESTEIN_PRIME 250

static int by_bluestein(int64_t n) {
  int64_t top = 1;
  for (int64_t d = 2; d * d <= n; d++)
    for (; n % d == 0; n /= d) top = d;
  return (n > 1 ? n : top) > BLUESTEIN_PRIME;
}

/* The two precisions */

/* Fills [c] with the n complex values of the chirp e^(-πi j² / n) and [k]
   with the 2m reals of Bluestein's kernel for n points: the transform of
   the conjugate chirp wrapped at m, over m, computed in double. Answers 0
   if memory runs out, -1 if the transform exceeds 2 √(2n - 1) in
   magnitude, which the error bound of nx_cpu.mli takes, else 1. */
static int chirp_kernel(int64_t n, int64_t m, double *c, double *k);

/* A plan of n points of [kind] (COMPLEX or REAL) in the precision of the
   real dtype [rdt], kept or made: NULL if memory runs out. plan_put
   gives it back, and frees it if it is not kept. */
static void *plan_get(int64_t n, int rdt, int kind);
static void plan_put(void *p, int rdt, int kind);

#define CAT2(a, b) a##_##b
#define CAT(a, b) CAT2(a, b)
#define F(name) CAT(name, SFX)

#define T float
#define L 8
#define SFX f32
#define RDT NX_FLOAT32
#include "fft_body.h"
#undef T
#undef L
#undef SFX
#undef RDT

#define T double
#define L 4
#define SFX f64
#define RDT NX_FLOAT64
#include "fft_body.h"
#undef T
#undef L
#undef SFX
#undef RDT

static int chirp_kernel(int64_t n, int64_t m, double *c, double *k) {
  memset(k, 0, sizeof(double) * 2 * m);
  /* e^(-πi j²/n) = ω_2n^(j² mod 2n), j² stepped by 2j + 1. */
  int64_t q = 0;
  for (int64_t j = 0; j < n; j++) {
    unit(q, 2 * n, &c[2 * j], &c[2 * j + 1]);
    k[2 * j] = c[2 * j];
    k[2 * j + 1] = -c[2 * j + 1];
    if (j > 0) {
      k[2 * (m - j)] = c[2 * j];
      k[2 * (m - j) + 1] = -c[2 * j + 1];
    }
    q += 2 * j + 1;
    if (q >= 2 * n) q -= 2 * n;
  }
  plan_f64 *p = plan_make_f64(m, ANY);
  lines one = rows_f64(k, 1, m);
  int ok = p && serial_f64(p, &one, &one, 0, 1);
  plan_free_f64(p);
  double peak = 0;
  for (int64_t j = 0; j < m; j++)
    peak = fmax(peak, hypot(k[2 * j], k[2 * j + 1]));
  for (int64_t j = 0; j < 2 * m; j++) k[j] /= (double)m;
  return !ok ? 0 : peak > 2 * sqrt(2.0 * (double)n - 1) ? -1 : 1;
}

/* The cache */

typedef struct {
  int64_t n, bytes;
  int rdt, kind, users;
  uint64_t used;
  void *plan;
} kept;

static kept cache[PLANS];
static uint64_t ticks;
static pthread_mutex_t cache_lock = PTHREAD_MUTEX_INITIALIZER;

static void plan_free(int rdt, int kind, void *p) {
  if (rdt == NX_FLOAT32) {
    if (kind == REAL) real_free_f32(p);
    else plan_free_f32(p);
  } else {
    if (kind == REAL) real_free_f64(p);
    else plan_free_f64(p);
  }
}

/* A new plan, and the bytes it holds in [*bytes]; NULL if memory runs
   out. */
static void *plan_new(int64_t n, int rdt, int kind, int64_t *bytes) {
  if (rdt == NX_FLOAT32 && kind == REAL) {
    real_plan_f32 *r = real_make_f32(n);
    *bytes = r ? r->bytes : 0;
    return r;
  }
  if (rdt == NX_FLOAT32) {
    plan_f32 *p = plan_make_f32(n, ANY);
    *bytes = p ? p->bytes : 0;
    return p;
  }
  if (kind == REAL) {
    real_plan_f64 *r = real_make_f64(n);
    *bytes = r ? r->bytes : 0;
    return r;
  }
  plan_f64 *p = plan_make_f64(n, ANY);
  *bytes = p ? p->bytes : 0;
  return p;
}

/* The kept plan of n points, a user more, or NULL; under cache_lock. */
static void *find(int64_t n, int rdt, int kind) {
  for (int i = 0; i < PLANS; i++) {
    kept *k = &cache[i];
    if (k->plan && k->n == n && k->rdt == rdt && k->kind == kind) {
      k->users++;
      k->used = ++ticks;
      return k->plan;
    }
  }
  return NULL;
}

static void *plan_get(int64_t n, int rdt, int kind) {
  pthread_mutex_lock(&cache_lock);
  void *p = find(n, rdt, kind);
  pthread_mutex_unlock(&cache_lock);
  if (p != NULL) return p;
  int64_t bytes;
  p = plan_new(n, rdt, kind, &bytes);
  if (p == NULL) return NULL;
  /* A call that missed too may have kept its plan meanwhile: that one is
     used and this one freed. Else this one is kept in the free slot, or
     in place of the least recently used plan no call holds, while the
     plans' bytes stay within PLAN_BYTES. */
  pthread_mutex_lock(&cache_lock);
  void *there = find(n, rdt, kind);
  if (there != NULL) {
    pthread_mutex_unlock(&cache_lock);
    plan_free(rdt, kind, p);
    return there;
  }
  int64_t total = bytes;
  int slot = -1;
  for (int i = 0; i < PLANS; i++) {
    kept *k = &cache[i];
    total += k->plan ? k->bytes : 0;
    if (k->plan && k->users > 0) continue;
    if (slot < 0 || !k->plan ||
        (cache[slot].plan && k->used < cache[slot].used))
      slot = i;
  }
  if (slot >= 0 && cache[slot].plan) total -= cache[slot].bytes;
  if (slot < 0 || total > PLAN_BYTES) {
    pthread_mutex_unlock(&cache_lock);
    return p;
  }
  kept old = cache[slot];
  cache[slot] = (kept){.n = n,
                       .bytes = bytes,
                       .rdt = rdt,
                       .kind = kind,
                       .users = 1,
                       .used = ++ticks,
                       .plan = p};
  pthread_mutex_unlock(&cache_lock);
  if (old.plan) plan_free(old.rdt, old.kind, old.plan);
  return p;
}

static void plan_put(void *p, int rdt, int kind) {
  pthread_mutex_lock(&cache_lock);
  for (int i = 0; i < PLANS; i++)
    if (cache[i].plan == p) {
      cache[i].users--;
      pthread_mutex_unlock(&cache_lock);
      return;
    }
  pthread_mutex_unlock(&cache_lock);
  plan_free(rdt, kind, p);
}

/* The entries */

enum { C2C, R2C, C2R };

/* Runs the transform [t], [inverse] for c2c's Inverse, of x into dst
   along the OCaml int array [vaxes], whose shapes nx_cpu.ml checked, on
   [threads] threads as cpu.h states. The caller roots its values. */
static value transform(int t, int inverse, value vaxes, value dst, value x,
                       int threads) {
  int axes[NX_MAX_RANK], na = (int)Wosize_val(vaxes);
  for (int i = 0; i < na; i++) axes[i] = Int_val(Field(vaxes, i));
  nx_operand in[2] = {{x, nx_array_dtype(x), 0},
                      {dst, nx_array_dtype(dst), 1}};
  nx_array a[2];
  int e = nx_read(2, in, a);
  if (e) return Val_int(e);
  int single = (t == C2R ? a[0].dtype : a[1].dtype) == NX_COMPLEX64, ok;
  if (t == C2C)
    ok = single ? c2c_f32(&a[0], &a[1], axes, na, inverse, threads)
                : c2c_f64(&a[0], &a[1], axes, na, inverse, threads);
  else if (t == R2C)
    ok = single ? r2c_f32(&a[0], &a[1], axes, na, threads)
                : r2c_f64(&a[0], &a[1], axes, na, threads);
  else
    ok = single ? c2r_f32(&a[0], &a[1], axes, na, threads)
                : c2r_f64(&a[0], &a[1], axes, na, threads);
  nx_done(2, a);
  if (!ok) caml_raise_out_of_memory();
  return Val_int(NX_OK);
}

value nx_cpu_c2c(value threads, value direction, value axes, value dst,
                 value x) {
  CAMLparam3(axes, dst, x);
  CAMLreturn(transform(C2C, Int_val(direction), axes, dst, x,
                       Int_val(threads)));
}

value nx_cpu_r2c(value threads, value axes, value dst, value x) {
  CAMLparam3(axes, dst, x);
  CAMLreturn(transform(R2C, 0, axes, dst, x, Int_val(threads)));
}

value nx_cpu_c2r(value threads, value axes, value dst, value x) {
  CAMLparam3(axes, dst, x);
  CAMLreturn(transform(C2R, 0, axes, dst, x, Int_val(threads)));
}
