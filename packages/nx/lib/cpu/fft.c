/*---------------------------------------------------------------------------
   Copyright (C) 2010-2019 Max-Planck-Society (pocketfft). BSD-3-Clause (see
   packages/nx/LICENSE-pocketfft).
   Copyright (c) 2026 The Raven authors. All rights reserved.

   SPDX-License-Identifier: BSD-3-Clause AND ISC
  ---------------------------------------------------------------------------*/

/* Fourier transforms: Nx_kernel.S's c2c, r2c and c2r.

   A transform runs along one axis at a time, over lines: the points along
   that axis at one index of the others. A line of n points has a plan,
   which depends on n and its precision alone:

     a prime factor        Bluestein's convolution, by two transforms of
     > BLUESTEIN_PRIME     m >= 2n - 1 points
     n <= DIRECT_POINTS    passes of radix 2, 3, 4, 5 and odd primes
     or n prime            (pocketfft's), L lines a vector, one a lane
     else                  Bailey's four steps: n = n1 n2, columns of n1
                           points by passes, a twiddle, rows of n2; for
                           8 | n, n2 = 8 up to ROWS8_BYTES

   A pass computes L lines at once, a lane each, in vectors of L
   numbers: lines side by side, or L points of one line, or, for the four
   steps, L columns then L rows of one line. Which lines share a vector
   changes no bit: a line's arithmetic is a function of n and its
   precision alone. The inverse is the conjugate of the forward transform
   of the conjugate, read and written so. Real lines of even n >=
   HALVED_POINTS go through complex ones of n / 2 points, others through
   complex ones of n.

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
#include "nx_kinds.h"

/* Stores of 32 bytes past the caches (streams), where the target has
   them (cpu.h's NX_STREAMS): on x86-64 for a result over a core's L2 on
   one thread and over the largest cache on several, on arm64 for a result
   over half of a core's L2 on any threads (measured on the M1 alone). */
#if defined(__x86_64__)
#define STREAM_BYTES nx_cpu_l2
#define STREAM_SEVERAL nx_cpu_llc
#else
#define STREAM_BYTES (nx_cpu_l2 / 2)
#define STREAM_SEVERAL (nx_cpu_l2 / 2)
#endif

/* The most points a line computes by passes alone. Above, the four steps
   give a single line every lane: a line of 256 points took 0.7 us by them
   against 4.9 by passes, which fill one lane of eight. A batch fills the
   lanes either way, and 64 lines of 256 took 26 us by the four steps
   against 19 by passes (M1, one thread); they take the four steps all the
   same, since a line's arithmetic depends on n alone. */
#define DIRECT_POINTS 64

/* The most bytes of a line whose four steps take rows of 8, n2 = 8,
   each L of them twiddled, transposed, transformed and written in
   registers: one sweep where rows of about √n points take a twiddle and
   transpose into a grid, their passes and a scatter. Lines of 2^10
   complex64 points took 1.25 us against 1.58 (M1, one thread). Those
   are the lines job.c gives one thread: a longer line alone splits its
   steps across threads, and n / 8 × 8 has too few columns to split, so
   that 2^11 points took 3.7 us on the M1's threads against 2.5. */
#define ROWS8_BYTES 8192

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

/* How a plan computes a line; ANY asks for the plan of n's choice, REAL
   for a real line's: where n is halved, the plan of n / 2 points with
   the twiddles between them and the real line's bins, else ANY's. */
enum { PASSES, FOUR, BLUESTEIN, ANY, REAL };

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
   writes conjugates. A [stream] line, a result's, is written where its
   points lie apart by stores past the caches, where they gain (streams,
   transform). */
typedef struct {
  uint8_t *base;
  int64_t step, io;
  int real, conj, stream, rank;
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
   the lines' innermost, steps least. Returns their count. */
static int others(const nx_array *a, int axis, int *order) {
  int k = 0;
  for (int i = 0; i < a->rank; i++) {
    if (i == axis) continue;
    int64_t s = llabs(a->dim[a->rank + i]);
    int j = k++;
    for (; j > 0 && llabs(a->dim[a->rank + order[j - 1]]) < s; j--)
      order[j] = order[j - 1];
    order[j] = i;
  }
  return k;
}

/* The lines of [a] along [axis], its other axes ordered as [by]'s by
   their strides (others), [w] reals an element. */
static lines lines_of(const nx_array *a, int axis, const nx_array *by, int w) {
  int order[NX_MAX_RANK];
  lines x = {.base = a->base + a->offset * (a->bits / 8),
             .step = w * a->dim[a->rank + axis],
             .io = 1,
             .rank = others(by, axis, order)};
  for (int i = 0; i < x.rank; i++) {
    x.dim[i] = a->dim[order[i]];
    x.stride[i] = w * a->dim[a->rank + order[i]];
  }
  return x;
}

/* Whether a result of [bytes] that [threads] threads write streams past
   the caches: past STREAM_BYTES on one thread and STREAM_SEVERAL on
   several, which a smaller result stays in for its reader. The four
   steps' rows write a line's points in pieces of 64 bytes a row apart,
   and each piece took a read of the cache line it fills: non-temporal
   stores, which take none, cut a line of 2^20 points from 4.2 to 3.1 ms
   and fft2 of 1024 × 1024 from 3.7 to 3.2 on one Arrow Lake core. On
   six, the 8 MiB line, which the 24 MiB L3 holds, took 1.46 times as
   long streamed and fft2 1.14, while a line of 2^22 points, 32 MiB, took
   6.05 ms against 6.67 and fft2 of 2048 × 2048 5.40 against 5.69. On the
   M1 they gain on any threads: 2^20 complex64 points 4.0 to 3.3 ms on
   one core and 0.65 to 0.56 on its threads, 1024 lines of 1024 side by
   side 3.1 to 1.5 and 0.52 to 0.26, 2^18 points 0.61 to 0.58 on one
   core; below half of a core's L2 (1.5 MiB), lines of 2^16 points (0.5
   to 1 MiB) lost up to 5% on its threads. */
static int streams(int64_t bytes, int threads) {
  return NX_STREAMS && bytes > (threads == 1 ? STREAM_BYTES : STREAM_SEVERAL);
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

/* The transforms' costs (cpu.h, Calls and jobs; M1 alone, one thread,
   2026-10-14).

   The steps between real lines and complex ones of half as many points,
   in copies of their bytes: r2c and c2r took 3 to 10 times such a copy's
   time more than c2c of half as many points, from 2^12 to 2^20 points. */
#define HALF_COST 6

/* A plan's cost, in lines of its points (plan_cost). */
#define PLAN_LINES 4

/* The fewest points of an even real line that goes through complex lines
   of half as many, within nx_cpu.mli's [e(n/2) + 8]. Measured against a
   complex transform of the n completed points, by radix-2 passes or by
   Bluestein's at a power of two: below, the split and merge steps left
   c2r's error up to 1.25 times that transform's (8 and 16 points), so the
   line goes through complex lines of n points, as an odd n's does, for 1.4
   to 2.5 times the time from 8 to 256 points (M1, one thread). From 512
   points, c2r's error is 0.95 to 0.99 times that transform's. */
#define HALVED_POINTS 512

static int halved(int64_t n) { return n % 2 == 0 && n >= HALVED_POINTS; }

/* Numbers */

/* π/2 as a double-double: rounded to double, then the rest rounded. */
#define HALF_PI_HI 0x1.921fb54442d18p0
#define HALF_PI_LO 0x1.1a62633145c07p-54

/* cos θ and sin θ, θ = (π/2) t / n, 0 <= t <= n / 2, so that θ <= π/4.
   θ is a + e in double-double, to about 2^-104 relatively; sin a and cos a
   are nx_kinds.h's polynomials, their leading terms' rounding errors and
   e's first-order terms added: sin θ = a + a³S(a²) + e (1 - a²/2) and
   cos θ = 1 - a²/2 + a⁴C(a²) - a e, a² exact as z + ez.

   A transform multiplies by many twiddles in turn, so the error of |w|²
   compounds across its passes where each value's alone does not. Where
   the pair's |w|² - 1, computed exactly (hc - 1 and its sum with hs are
   exact by Sterbenz's lemma), exceeds half an ulp of 1, the value whose
   estimated rounding error pushes |w|² that way most moves one ulp back;
   at θ = π/4 it moves cos, so that c ≠ s there. With values rounded as
   computed, float64's c2r of 8 points erred 6% more than a radix-2
   transform of its completed points did.

   Only +, -, ×, ÷, fma and nextafter round, so every machine gives the
   same bits, where macOS's and glibc's cos and sin, and with them the
   transforms, differed in the last bit. Against MPFR, over every θ of
   every n up to 12000 and of n = 2062, 65538, 131072, 262202 and 2^20 to
   2^22: values within 0.77 ulp, |w|² within half an ulp of 1, where cos
   and sin of θ rounded to double were within 2.4 ulp of the values. */
static void quarter(int64_t t, int64_t n, double *c, double *s) {
  double yh = (double)t / (double)n;
  double yl = nx_fmad(-yh, (double)n, (double)t) / (double)n;
  double ah = HALF_PI_HI * yh;
  double al = nx_fmad(HALF_PI_HI, yh, -ah) +
              nx_fmad(HALF_PI_HI, yl, HALF_PI_LO * yh);
  double a = ah + al, e = al - (a - ah);
  double z = a * a, ez = nx_fmad(a, a, -z);
  double hz = 0.5 * z, w = 1.0 - hz;
  double ts = nx_fmad(a * z, nx_sincos_p_f64(z, 0), e * (1.0 - hz));
  double tc = ((1.0 - w) - hz) + nx_fmad(z * z, nx_sincos_p_f64(z, 1),
                                         -nx_fmad(0.5, ez, a * e));
  double sv = a + ts, cv = w + tc;
  double hc = cv * cv, hs = sv * sv;
  double dn = ((hc - 1.0) + hs) +
              (nx_fmad(cv, cv, -hc) + nx_fmad(sv, sv, -hs));
  if (fabs(dn) > 0x1p-53) {
    /* The final additions' rounding errors, exact - rounded (Fast2Sum),
       which miss ts's and tc's own. */
    double es = ts - (sv - a), ec = tc - (cv - w);
    double ps = -dn * sv * es, pc = -dn * cv * ec;
    double *v = ps > pc ? &sv : &cv;
    *v = nextafter(*v, dn > 0 ? 0.0 : 2.0);
  }
  *s = sv;
  *c = cv;
}

/* e^(-2πi m / n), 0 <= m < n: θ = 2π m / n is (π/2) (q + r / n) in
   integers, its angle within the quarter taken from the nearer end, at
   most π/4 (quarter), and turned by q quarters exactly. */
static void unit(int64_t m, int64_t n, double *re, double *im) {
  int64_t q = 4 * m / n, r = 4 * m - q * n;
  int far = 2 * r > n;
  double c, s;
  quarter(far ? n - r : r, n, &c, &s);
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
   of two at least 2n - 1. Measured against the same convolution at a
   power of two by radix-2 passes, pocketfft's choice, the least product
   of 2, 3, 5, 7 and 11, erred more: float64's up to 1.5 times (c2r of
   32771 points: 6.0 against 4.1 in units of u √N ‖x‖₂; of 2062: 4.7
   against 3.4, and 3.9 with the kernel and the chirp in double-double),
   and float32's c2r of even n by 2 to 5% (514 to 8198 points). At a power
   of two the error is 0.4 to 0.95 times that reference's, for 1.3 to 2.2
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

/* The time a pass of a line of the prime radix [p] takes, in copies of
   the line's bytes: log₂ p for 2 to 5, p / 2 for the others, whose pass
   adds about p / 2 products a point. A line of 241 points took 11 copies
   times log₂ 241, of 241 · 1024 points 5.4 times log₂ of them, of 7⁵ 1.1
   times (M1 alone, one thread). */
static double radix_cost(int64_t p) {
  return p <= 5 ? log2((double)p) : (double)p / 2;
}

/* The time a transform of lines of n points takes, in copies of the
   lines' bytes: its passes', or where a prime factor exceeds
   BLUESTEIN_PRIME (by_bluestein) Bluestein's two transforms of m points
   a line, which took 1.2 times their passes' for lines of 1009 points,
   and its pointwise steps, a pass each over m, m and n points. */
static double line_cost(int64_t n) {
  if (n < 2) return 1;
  int twos = __builtin_ctzll((unsigned long long)n);
  int64_t k = n >> twos, top = twos > 0 ? 2 : 1;
  double w = twos;
  for (int64_t d = 3; d * d <= k; d += 2)
    for (; k % d == 0; k /= d) {
      w += radix_cost(d);
      top = d;
    }
  if (k > 1) {
    w += radix_cost(k);
    top = k;
  }
  if (top <= BLUESTEIN_PRIME) return w > 1 ? w : 1;
  double m = (double)blue_points(n);
  return (2 * m * log2(m) + 2 * m + (double)n) / (double)n;
}

/* The bytes a pass moves for each point of a line: the point read and
   written, a complex number of two reals of [w] bytes. */
static double point_bytes(int w) { return 4.0 * w; }

/* The cost of the passes of [count] lines of [n] points, [w] a real's
   bytes (line_cost): a four-step line's column and row steps', each
   lines of one of its factors. */
static double passes_cost(int64_t n, double count, int w) {
  return count * point_bytes(w) * (double)n * line_cost(n);
}

/* The cost of [count] lines of [n] points, [w] a real's bytes, the call's
   and each transform job's: their points' passes and each line's set-up,
   371 measured for each of 16384 lines of 2 complex64 points (lines,
   NX_CPU_LINE). */
static double lines_cost(int64_t n, double count, int w) {
  return passes_cost(n, count, w) + count * NX_CPU_LINE;
}

/* The cost of a pointwise step over [points] complex points, [w] a real's
   bytes, one pass: a Bluestein line's chirp, convolution and unchirp. */
static double pointwise_cost(double points, int w) {
  return point_bytes(w) * points;
}

/* The cost of the steps between [points] real points, [w] a real's bytes,
   and their complex lines of half as many (streaming). */
static double half_cost(double points, int w) {
  return HALF_COST * point_bytes(w) * points;
}

/* The two precisions */

/* Fills [c] with the n complex values of the chirp e^(-πi j² / n) and [k]
   with the 2m reals of Bluestein's kernel for n points: the transform of
   the conjugate chirp wrapped at m, over m, computed in double. Answers 0
   if memory runs out, -1 if the transform exceeds 2 √(2n - 1) in
   magnitude, which the error bound of nx_cpu.mli takes, else 1. */
static int chirp_kernel(int64_t n, int64_t m, double *c, double *k);

/* The plan of a line of n points of [kind] (ANY or REAL) in the
   precision of the real dtype [rdt], kept or made: NULL if memory runs
   out. plan_put gives it back, and frees it if it is not kept. */
static void *plan_get(int64_t n, int rdt, int kind);
static void plan_put(void *p, int rdt);

#define CAT2(a, b) a##_##b
#define CAT(a, b) CAT2(a, b)
#define F(name) CAT(name, SFX)

/* The bytes of a vector of L lanes: the 32 of x86-64-v3's, also on
   targets of 16-byte vectors where clang builds, but their 16 where gcc
   builds, which computes a 32-byte vector there lane by lane through
   memory. Which lanes share a vector changes no bit. Lines of complex64
   / complex128 points, µs, M1, one core, by 32 and 16 bytes:

                     gcc 32        gcc 16        clang 32      clang 16
     1024            3.36 / 5.44   1.59 / 4.00   1.34 / 3.08   1.47 / 3.68
     2^16            311 / 535     163 / 498     145 / 312     157 / 489
     2^20 (ms)       9.12 / 19.2   4.79 / 10.6   4.35 / 9.47   4.81 / 10.7
     64 × 1024       216 / 360     102 / 263     88 / 201      93 / 240
     side by side    214 / 385     117 / 525     98 / 347      115 / 485

   Under gcc, 16 bytes but for complex128 lines side by side, whose two
   lanes read 32 bytes of a 128-byte cache line. So the width is the FFT's
   own: under clang it takes 32 bytes where NX_VECTOR_BYTES is 16. */
#if defined(__AVX__) || defined(__clang__) || !defined(__GNUC__)
#define VEC_BYTES 32
#else
#define VEC_BYTES 16
#endif

/* The alignment of a streamed piece (F(streams)). On x86-64 a whole
   64-byte line, which no other unit's store shares: its streams combine a
   line's stores in a buffer, and a line written in parts goes to memory
   in parts, as tiles 65 elements wide made a transposed copy twice as
   slow. arm64's stnp has no such buffer: a piece lies on its own size,
   and the M1's lines are 128 bytes, so its measured gains already come
   from parts of lines. */
#if defined(__x86_64__)
#define STREAM_PIECE 64
#else
#define STREAM_PIECE (2 * VEC_BYTES)
#endif

#define T float
#define L (VEC_BYTES / 4)
#define SFX f32
#define RDT NX_FLOAT32
#include "fft_body.h"
#undef T
#undef L
#undef SFX
#undef RDT

#define T double
#define L (VEC_BYTES / 8)
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
  /* The peak's square against the bound's: hypot differs by an ulp between
     libcs, and a plan's choice must be the same on every target. */
  double peak = 0;
  for (int64_t j = 0; j < m; j++) {
    double re = k[2 * j], im = k[2 * j + 1];
    peak = fmax(peak, re * re + im * im);
  }
  for (int64_t j = 0; j < 2 * m; j++) k[j] /= (double)m;
  return !ok ? 0 : peak > 4 * (2.0 * (double)n - 1) ? -1 : 1;
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

static void plan_free(int rdt, void *p) {
  if (rdt == NX_FLOAT32) plan_free_f32(p);
  else plan_free_f64(p);
}

/* A new plan, and the bytes it holds in [*bytes]; NULL if memory runs
   out. */
static void *plan_new(int64_t n, int rdt, int kind, int64_t *bytes) {
  if (rdt == NX_FLOAT32) {
    plan_f32 *p = plan_make_f32(n, kind);
    *bytes = p ? p->bytes : 0;
    return p;
  }
  plan_f64 *p = plan_make_f64(n, kind);
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
    plan_free(rdt, p);
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
  if (old.plan) plan_free(old.rdt, old.plan);
  return p;
}

static void plan_put(void *p, int rdt) {
  pthread_mutex_lock(&cache_lock);
  for (int i = 0; i < PLANS; i++)
    if (cache[i].plan == p) {
      cache[i].users--;
      pthread_mutex_unlock(&cache_lock);
      return;
    }
  pthread_mutex_unlock(&cache_lock);
  plan_free(rdt, p);
}

/* The entries */

enum { C2C, R2C, C2R };

/* What making a plan of n points takes (cpu.h, Calls), in lines of its
   points: making one took 1 to 2.3 times a line's transform from 2^16 to
   2^20 points, Bluestein's included (M1 alone). [w] is a real's
   bytes. */
static double plan_cost(int64_t n, int w) {
  return PLAN_LINES * point_bytes(w) * (double)n * line_cost(n);
}

/* The cost (cpu.h, Calls and jobs) of the transform [t] of x into y
   along the [na] [axes]: each axis's lines (lines_cost), as F(transform)
   takes them, and the plans it makes on a cache's miss where they could
   take it to NX_CPU_HOLD_BYTES; a real transform's steps between real and
   complex lines (half_cost); and a real transform of no point's zeros,
   element by element (zero), 3 ns a complex64 bin (M1 alone). A line of n
   points costs at most n: a call held at that bound skips the factoring,
   which took 8 ns of a 16-point line's 240 (M1). */
static double call_cost(int t, const nx_array *x, const nx_array *y,
                        const int *axes, int na) {
  const nx_array *r = t == R2C ? x : y, *c = t == C2R ? x : y;
  int w = c->bits / 16;
  double points = (double)numel(r), bytes = point_bytes(w) * points;
  double price = t == C2C ? 0 : half_cost(points, w);
  if (t == R2C && x->dim[axes[na - 1]] == 0)
    return price + NX_CPU_LINE * (double)numel(y);
  double most = price;
  for (int i = 0; i < na; i++) {
    double n = (double)r->dim[axes[i]];
    if (n > 0) most += NX_CPU_LINE * points / n;
    most += (bytes + PLAN_LINES * point_bytes(w) * n) * n;
  }
  if (most < NX_CPU_RELEASE_BYTES) return most;
  double plans = 0;
  for (int i = 0; i < na; i++) {
    int64_t n = r->dim[axes[i]];
    if (n > 0) price += lines_cost(n, points / (double)n, w);
    plans += plan_cost(n, w);
  }
  return price + plans < NX_CPU_HOLD_BYTES ? price : price + plans;
}

/* The transform [t] of [x] into [y], with the domain lock released where
   the call's price reaches NX_CPU_RELEASE_BYTES: 0 if memory runs out. */
static int run(int t, int inverse, const nx_array *x, const nx_array *y,
               const int *axes, int na) {
  int single = (t == C2R ? x->dtype : y->dtype) == NX_COMPLEX64, ok;
  int released = nx_cpu_release(call_cost(t, x, y, axes, na));
  if (t == C2C)
    ok = single ? c2c_f32(x, y, axes, na, inverse)
                : c2c_f64(x, y, axes, na, inverse);
  else if (t == R2C)
    ok = single ? r2c_f32(x, y, axes, na)
                : r2c_f64(x, y, axes, na);
  else
    ok = single ? c2r_f32(x, y, axes, na)
                : c2r_f64(x, y, axes, na);
  nx_cpu_reacquire(released);
  return ok;
}

/* Runs the transform [t], [inverse] for c2c's Inverse, of x into dst
   along the OCaml int array [vaxes], whose shapes nx_cpu.ml checked. The
   caller roots its values. */
static value transform(int t, int inverse, value vaxes, value dst, value x) {
  int axes[NX_MAX_RANK], na = (int)Wosize_val(vaxes);
  for (int i = 0; i < na; i++) axes[i] = Int_val(Field(vaxes, i));
  nx_operand in[2] = {{x, nx_array_dtype(x), 0},
                      {dst, nx_array_dtype(dst), 1}};
  nx_array a[2];
  int e = nx_read(2, in, a);
  if (e) return Val_int(e);
  int ok = run(t, inverse, &a[0], &a[1], axes, na);
  nx_done(2, a);
  if (!ok) caml_raise_out_of_memory();
  return Val_int(NX_OK);
}

value nx_cpu_c2c(value direction, value axes, value dst, value x) {
  CAMLparam3(axes, dst, x);
  CAMLreturn(transform(C2C, Int_val(direction), axes, dst, x));
}

value nx_cpu_r2c(value axes, value dst, value x) {
  CAMLparam3(axes, dst, x);
  CAMLreturn(transform(R2C, 0, axes, dst, x));
}

value nx_cpu_c2r(value axes, value dst, value x) {
  CAMLparam3(axes, dst, x);
  CAMLreturn(transform(C2R, 0, axes, dst, x));
}
