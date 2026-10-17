/*---------------------------------------------------------------------------
   Copyright (C) 2010-2019 Max-Planck-Society (pocketfft). BSD-3-Clause (see
   dev/nx2/LICENSE-pocketfft).
   Copyright (c) 2026 The Raven authors. All rights reserved.

   SPDX-License-Identifier: BSD-3-Clause AND ISC
  ---------------------------------------------------------------------------*/

/* fft.c's transforms in one precision: T float or double, its complex
   numbers' parts. Included once per precision, with T, L (the lanes of a
   vector), SFX, RDT (the real dtype's code) and the shuffles of fft.c for
   L lanes defined. */

#define VEC CAT(vec, SFX)
#define CV CAT(cv, SFX)
#define PLAN CAT(plan, SFX)

/* L complex numbers, one a lane: the elements every pass computes on, L
   lines side by side. A VEC is aligned as the base targets' 16-byte
   vectors, which hold it in two or more: at its own 32 bytes, gcc notes on
   x86-64 that passing it by value changed ABI in gcc 4.6, an ABI only AVX
   uses and no call here crosses. */
typedef T VEC __attribute__((vector_size(L * sizeof(T)), aligned(16)));
typedef struct {
  VEC re, im;
} CV;

/* Shuffles: the real and imaginary parts of 2L interleaved values, the
   inverse, the lanes reversed, and the transpose of L vectors as an L × L
   tile. */
#define SH __builtin_shufflevector

/* Unrolls the small loop that follows over vectors of an array: gcc -O2
   left such loops rolled and their arrays in memory, and the rows of 8 of
   a line of 1024 points took 0.54 µs against 0.26 by clang (Arrow Lake). */
#define UNROLL _Pragma("GCC unroll 8")
#if L == 8
#define DEAL_RE(a, b) SH(a, b, 0, 2, 4, 6, 8, 10, 12, 14)
#define DEAL_IM(a, b) SH(a, b, 1, 3, 5, 7, 9, 11, 13, 15)
#define ZIP_LO(a, b) SH(a, b, 0, 8, 1, 9, 2, 10, 3, 11)
#define ZIP_HI(a, b) SH(a, b, 4, 12, 5, 13, 6, 14, 7, 15)
#define REV(a) SH(a, a, 7, 6, 5, 4, 3, 2, 1, 0)

static inline void F(transpose)(VEC *r) {
  VEC a[8], b[8];
  UNROLL
  for (int i = 0; i < 8; i += 2) {
    a[i] = SH(r[i], r[i + 1], 0, 8, 1, 9, 4, 12, 5, 13);
    a[i + 1] = SH(r[i], r[i + 1], 2, 10, 3, 11, 6, 14, 7, 15);
  }
  UNROLL
  for (int i = 0; i < 8; i += 4)
    UNROLL
    for (int k = 0; k < 2; k++) {
      b[i + 2 * k] = SH(a[i + k], a[i + k + 2], 0, 1, 8, 9, 4, 5, 12, 13);
      b[i + 2 * k + 1] = SH(a[i + k], a[i + k + 2], 2, 3, 10, 11, 6, 7, 14, 15);
    }
  UNROLL
  for (int i = 0; i < 4; i++) {
    r[i] = SH(b[i], b[i + 4], 0, 1, 2, 3, 8, 9, 10, 11);
    r[i + 4] = SH(b[i], b[i + 4], 4, 5, 6, 7, 12, 13, 14, 15);
  }
}
#elif L == 2
#define DEAL_RE(a, b) SH(a, b, 0, 2)
#define DEAL_IM(a, b) SH(a, b, 1, 3)
#define ZIP_LO(a, b) SH(a, b, 0, 2)
#define ZIP_HI(a, b) SH(a, b, 1, 3)
#define REV(a) SH(a, a, 1, 0)

static inline void F(transpose)(VEC *r) {
  VEC a = SH(r[0], r[1], 0, 2), b = SH(r[0], r[1], 1, 3);
  r[0] = a;
  r[1] = b;
}
#else
#define DEAL_RE(a, b) SH(a, b, 0, 2, 4, 6)
#define DEAL_IM(a, b) SH(a, b, 1, 3, 5, 7)
#define ZIP_LO(a, b) SH(a, b, 0, 4, 1, 5)
#define ZIP_HI(a, b) SH(a, b, 2, 6, 3, 7)
#define REV(a) SH(a, a, 3, 2, 1, 0)

static inline void F(transpose)(VEC *r) {
  VEC a[4];
  UNROLL
  for (int i = 0; i < 4; i += 2) {
    a[i] = SH(r[i], r[i + 1], 0, 4, 2, 6);
    a[i + 1] = SH(r[i], r[i + 1], 1, 5, 3, 7);
  }
  UNROLL
  for (int k = 0; k < 2; k++) {
    r[k] = SH(a[k], a[k + 2], 0, 1, 4, 5);
    r[k + 2] = SH(a[k], a[k + 2], 2, 3, 6, 7);
  }
}
#endif
#define TRANSPOSE F(transpose)

static inline CV F(add)(CV a, CV b) { return (CV){a.re + b.re, a.im + b.im}; }
static inline CV F(sub)(CV a, CV b) { return (CV){a.re - b.re, a.im - b.im}; }
static inline CV F(scale)(CV a, T s) { return (CV){a.re * s, a.im * s}; }

/* a + i b and a - i b. */
static inline CV F(plus_i)(CV a, CV b) {
  return (CV){a.re - b.im, a.im + b.re};
}
static inline CV F(minus_i)(CV a, CV b) {
  return (CV){a.re + b.im, a.im - b.re};
}

/* a·w for one complex w, and lane by lane. */
static inline CV F(mul)(CV a, T wr, T wi) {
  return (CV){a.re * wr - a.im * wi, a.re * wi + a.im * wr};
}
static inline CV F(mulv)(CV a, CV w) {
  return (CV){a.re * w.re - a.im * w.im, a.re * w.im + a.im * w.re};
}

/* Plans */

/* How a plan computes a line of n points (fft.c): by passes, by the four
   steps, or by Bluestein's convolution. */
typedef struct PLAN {
  int kind;
  int64_t n, bytes; /* bytes: what the plan holds, counted as it is made */
  /* PASSES: the radices, first applied first, and each pass's twiddles,
     (ip - 1) × ido of them, (x, i) at [2 (x ido + i)], ω^((x + 1) l1 i);
     cs[k], a generic pass's ω_ip^m. */
  int passes, radix[MAX_PASSES];
  T *tw[MAX_PASSES], *cs[MAX_PASSES];
  /* FOUR: n = n1 n2, the columns' and the rows' plans, and the twiddles
     between: entry g n1 + k1, lane v, is ω^(k1 (g L + v)). */
  int64_t n1, n2;
  struct PLAN *p1, *p2;
  CV *tw4;
  /* BLUESTEIN: the convolution's length m and plan; the chirp
     e^(-πi j² / n), j < n; the kernel, the transform of the conjugate
     chirp wrapped at m, over m. */
  int64_t m;
  struct PLAN *pm;
  T *chirp, *kernel;
  /* REAL, a halved real line of 2n points: ω_2n^k for k < n. */
  T *w;
} PLAN;

static void F(plan_free)(PLAN *p) {
  if (p == NULL) return;
  for (int k = 0; k < p->passes; k++) {
    free(p->tw[k]);
    free(p->cs[k]);
  }
  F(plan_free)(p->p1);
  F(plan_free)(p->p2);
  free(p->tw4);
  F(plan_free)(p->pm);
  free(p->chirp);
  free(p->kernel);
  free(p->w);
  free(p);
}

/* [bytes] for [p] to keep, on a vector's alignment, counted in its
   bytes; NULL if memory runs out. */
static void *F(hold)(PLAN *p, int64_t bytes) {
  int64_t v = sizeof(CV), size = (bytes + v - 1) / v * v;
  p->bytes += size;
  return aligned_alloc(v, (size_t)(size > 0 ? size : v));
}

/* Fills [w] with the n complex values unit(m, n) at m = k j, k < count. */
static void F(units)(T *w, int64_t count, int64_t j, int64_t n) {
  int64_t m = 0;
  for (int64_t k = 0; k < count; k++) {
    double re, im;
    unit(m, n, &re, &im);
    w[2 * k] = (T)re;
    w[2 * k + 1] = (T)im;
    m += j;
    if (m >= n) m -= n;
  }
}

static PLAN *F(plan_make)(int64_t n, int kind);

/* The passes of [p]: pocketfft's factorisation, fours first, a two
   moved to the front, then the odd factors in increasing order. Radix 8
   (pocketfft's C++ pass8) took 10 to 20% longer: its eight inputs of L
   lanes spill registers. Within the rows of 8 (within8) it bought
   nothing: columns of 128 points by passes 2, 8, 8 took 1.22 us a line
   of 1024 points against 1.20 by 2, 4, 4, 4 (M1, one thread). F(eight)
   computes eight points by this factorisation's passes, 2 then 4. */
static int F(passes_make)(PLAN *p) {
  int64_t n = p->n, rest = n;
  int *f = p->radix;
  while (rest % 4 == 0) {
    f[p->passes++] = 4;
    rest /= 4;
  }
  if (rest % 2 == 0) {
    rest /= 2;
    f[p->passes] = p->passes > 0 ? f[0] : 2;
    f[0] = 2;
    p->passes++;
  }
  for (int64_t d = 3; rest > 1; d += 2) {
    if (d * d > rest) d = rest;
    while (rest % d == 0) {
      f[p->passes++] = (int)d;
      rest /= d;
    }
  }
  int64_t l1 = 1;
  for (int k = 0; k < p->passes; k++) {
    int ip = f[k];
    int64_t ido = n / (l1 * ip);
    p->tw[k] = F(hold)(p, sizeof(T) * 2 * (ip - 1) * ido);
    if (p->tw[k] == NULL) return 0;
    for (int x = 1; x < ip; x++)
      F(units)(p->tw[k] + 2 * (x - 1) * ido, ido, x * l1, n);
    if (ip > 5) {
      p->cs[k] = F(hold)(p, sizeof(T) * 2 * ip);
      if (p->cs[k] == NULL) return 0;
      F(units)(p->cs[k], ip, 1, ip);
    }
    l1 *= ip;
  }
  return 1;
}

/* The four steps' plans and twiddles: n1 = n / 8 where 8 | n and the
   line holds at most ROWS8_BYTES, else n1 the largest divisor of n up to
   its square root of the fewest passes of n1 and n / n1 (1024 points as
   16 × 64 took 11% less time than as 32 × 32 across 64 lines). ω^(k1 j2)
   is the product, in double, of unit's values at its exponent's low and
   high halves. */
static int F(four_make)(PLAN *p) {
  int64_t n = p->n, n1 = 1;
  if (n % 8 == 0 && n * 2 * (int64_t)sizeof(T) <= ROWS8_BYTES) n1 = n / 8;
  else
    for (int64_t d = 2; d * d <= n; d++)
      if (n % d == 0 && passes_of(d) + passes_of(n / d) <=
                            passes_of(n1) + passes_of(n / n1))
        n1 = d;
  int64_t n2 = n / n1, groups = (n2 + L - 1) / L;
  p->n1 = n1;
  p->n2 = n2;
  p->p1 = F(plan_make)(n1, PASSES);
  p->p2 = F(plan_make)(n2, PASSES);
  p->tw4 = F(hold)(p, sizeof(CV) * groups * n1);
  int64_t half = 1;
  while (half * half < n) half *= 2;
  double *lo = malloc(sizeof(double) * 2 * half),
         *hi = malloc(sizeof(double) * 2 * (n / half + 1));
  int ok = p->p1 && p->p2 && p->tw4 && lo && hi;
  if (ok) {
    p->bytes += p->p1->bytes + p->p2->bytes;
    for (int64_t i = 0; i < half; i++) unit(i, n, &lo[2 * i], &lo[2 * i + 1]);
    for (int64_t i = 0; i <= n / half; i++)
      unit(i * half, n, &hi[2 * i], &hi[2 * i + 1]);
    for (int64_t g = 0; g < groups; g++)
      for (int64_t k1 = 0; k1 < n1; k1++) {
        CV *w = &p->tw4[g * n1 + k1];
        for (int v = 0; v < L; v++) {
          int64_t j2 = g * L + v, m = j2 < n2 ? k1 * j2 % n : 0;
          const double *a = lo + 2 * (m % half), *b = hi + 2 * (m / half);
          w->re[v] = (T)(a[0] * b[0] - a[1] * b[1]);
          w->im[v] = (T)(a[0] * b[1] + a[1] * b[0]);
        }
      }
  }
  free(lo);
  free(hi);
  return ok;
}

/* Bluestein's plan, its kernel computed in double: 1, or 0 if memory
   runs out, -1 if the kernel is too large (chirp_kernel). */
static int F(blue_make)(PLAN *p) {
  int64_t n = p->n, m = blue_points(n);
  p->m = m;
  p->pm = F(plan_make)(m, ANY);
  p->chirp = F(hold)(p, sizeof(T) * 2 * n);
  p->kernel = F(hold)(p, sizeof(T) * 2 * m);
  double *c = malloc(sizeof(double) * 2 * n), *k = malloc(sizeof(double) * 2 * m);
  int ok = p->pm && p->chirp && p->kernel && c && k;
  if (ok) ok = chirp_kernel(n, m, c, k);
  if (ok == 1) {
    p->bytes += p->pm->bytes;
    for (int64_t j = 0; j < 2 * n; j++) p->chirp[j] = (T)c[j];
    for (int64_t j = 0; j < 2 * m; j++) p->kernel[j] = (T)k[j];
  }
  free(c);
  free(k);
  return ok;
}

/* The plan of a line of n points of [kind] PASSES, ANY or REAL (fft.c).
   ANY's is by Bluestein where by_bluestein takes it and its kernel is
   small enough, else by passes up to DIRECT_POINTS or for a prime n, and
   by the four steps above. NULL if memory runs out. */
static PLAN *F(plan_make)(int64_t n, int kind) {
  if (kind == REAL && halved(n)) {
    PLAN *p = F(plan_make)(n / 2, ANY);
    if (p == NULL) return NULL;
    p->w = F(hold)(p, sizeof(T) * n);
    if (p->w != NULL) {
      F(units)(p->w, n / 2, 1, n);
      return p;
    }
    F(plan_free)(p);
    return NULL;
  }
  if (kind == REAL) kind = ANY;
  PLAN *p = calloc(1, sizeof(PLAN));
  if (p == NULL) return NULL;
  p->n = n;
  p->bytes = sizeof(PLAN);
  if (kind == ANY && by_bluestein(n)) {
    p->kind = BLUESTEIN;
    int ok = F(blue_make)(p);
    if (ok == 1) return p;
    F(plan_free)(p);
    if (ok == 0) return NULL;
    p = calloc(1, sizeof(PLAN));
    if (p == NULL) return NULL;
    p->n = n;
    p->bytes = sizeof(PLAN);
  }
  p->kind = kind == PASSES || n <= DIRECT_POINTS || prime(n) ? PASSES : FOUR;
  if (p->kind == PASSES ? F(passes_make)(p) : F(four_make)(p)) return p;
  F(plan_free)(p);
  return NULL;
}

/* Passes, after pocketfft's forward ones: pass k of radix ip takes l1
   transforms of ido · ip points to ip l1 of ido. CC(i, u, k) is input
   point i + ido (u + ip k); CH(i, k, u) output point i + ido (k + l1 u).
   Each lane is a line of its own. */

#define CC(i, u, k) cc[(i) + ido * ((u) + ip * (k))]
#define CH(i, k, u) ch[(i) + ido * ((k) + l1 * (u))]
#define TW(x, i) tw[2 * ((x) * ido + (i))], tw[2 * ((x) * ido + (i)) + 1]

/* Butterflies: radix ip's outputs y, untwiddled, of its inputs c, [s]
   apart. Radix 3 and 5 keep butterflies of their own: the odd radix's
   with a constant ip took 7 to 13% longer on lines of 1000, 2187, 15625
   and 59049 points (M1, one thread). */

static inline void F(two)(const CV *c, int64_t s, CV *y) {
  y[0] = F(add)(c[0], c[s]);
  y[1] = F(sub)(c[0], c[s]);
}

static inline void F(three)(const CV *c, int64_t s, CV *y) {
  const T c1 = -0.5, s1 = -0.86602540378443864676;
  CV t0 = c[0], t1 = F(add)(c[s], c[2 * s]), t2 = F(sub)(c[s], c[2 * s]);
  CV a = F(add)(t0, F(scale)(t1, c1)), b = F(scale)(t2, s1);
  y[0] = F(add)(t0, t1);
  y[1] = F(plus_i)(a, b);
  y[2] = F(minus_i)(a, b);
}

static inline void F(four)(const CV *c, int64_t s, CV *y) {
  CV t1 = F(sub)(c[0], c[2 * s]), t2 = F(add)(c[0], c[2 * s]),
     t3 = F(add)(c[s], c[3 * s]), t4 = F(sub)(c[s], c[3 * s]);
  y[0] = F(add)(t2, t3);
  y[1] = F(minus_i)(t1, t4);
  y[2] = F(sub)(t2, t3);
  y[3] = F(plus_i)(t1, t4);
}

static inline void F(five)(const CV *c, int64_t s, CV *y) {
  const T c1 = 0.3090169943749474241, s1 = -0.95105651629515357212,
          c2 = -0.8090169943749474241, s2 = -0.58778525229247312917;
  CV t0 = c[0], t1 = F(add)(c[s], c[4 * s]), t4 = F(sub)(c[s], c[4 * s]),
     t2 = F(add)(c[2 * s], c[3 * s]), t3 = F(sub)(c[2 * s], c[3 * s]);
  CV a1 = F(add)(F(add)(t0, F(scale)(t1, c1)), F(scale)(t2, c2)),
     b1 = F(add)(F(scale)(t4, s1), F(scale)(t3, s2)),
     a2 = F(add)(F(add)(t0, F(scale)(t1, c2)), F(scale)(t2, c1)),
     b2 = F(sub)(F(scale)(t4, s2), F(scale)(t3, s1));
  y[0] = F(add)(F(add)(t0, t1), t2);
  y[1] = F(plus_i)(a1, b1);
  y[2] = F(plus_i)(a2, b2);
  y[3] = F(minus_i)(a2, b2);
  y[4] = F(minus_i)(a1, b1);
}

/* An odd radix, its outputs [t] apart: output u of the pair sums
   t+ = c_j + c_(ip-j) and differences t- is a + i b and output ip - u is
   a - i b, with a = c_0 + Σ Re ω^(uj) t+ and b = Σ Im ω^(uj) t-, ω^m at
   [cs]. */
static inline void F(odd)(int ip, const CV *c, int64_t s, CV *y, int64_t t,
                          const T *cs) {
  int h = (ip + 1) / 2;
  CV y0 = c[0];
  for (int j = 1; j < h; j++)
    y0 = F(add)(y0, F(add)(c[j * s], c[(ip - j) * s]));
  y[0] = y0;
  for (int u = 1; u < h; u++) {
    CV a = c[0], b = {0};
    int m = 0;
    for (int j = 1; j < h; j++) {
      m += u;
      if (m >= ip) m -= ip;
      CV p = c[j * s], q = c[(ip - j) * s];
      a = F(add)(a, F(scale)(F(add)(p, q), cs[2 * m]));
      b = F(add)(b, F(scale)(F(sub)(p, q), cs[2 * m + 1]));
    }
    y[u * t] = F(plus_i)(a, b);
    y[(ip - u) * t] = F(minus_i)(a, b);
  }
}

/* Pass k of radix ip: each point's butterfly, then its outputs but the
   first turned by their twiddles, which are 1 at i = 0. It is
   instantiated once per radix below, with a constant ip up to 5, where
   the outputs stay in registers; the odd radix, whose ip has no bound,
   writes them in place in ch, where they turn. */
static inline __attribute__((always_inline)) void F(pass)(
    int ip, int64_t ido, int64_t l1, const CV *restrict cc, CV *restrict ch,
    const T *tw, const T *cs) {
  int64_t t = ip > 5 ? l1 * ido : 1;
  for (int64_t k = 0; k < l1; k++)
    for (int64_t i = 0; i < ido; i++) {
      CV r[5], *y = ip > 5 ? &CH(i, k, 0) : r;
      const CV *c = &CC(i, 0, k);
      switch (ip) {
        case 2: F(two)(c, ido, y); break;
        case 3: F(three)(c, ido, y); break;
        case 4: F(four)(c, ido, y); break;
        case 5: F(five)(c, ido, y); break;
        default: F(odd)(ip, c, ido, y, t, cs);
      }
      if (i == 0 && ip > 5) continue;
      CH(i, k, 0) = y[0];
      UNROLL
      for (int u = 1; u < ip; u++)
        CH(i, k, u) = i == 0 ? y[u * t] : F(mul)(y[u * t], TW(u - 1, i));
    }
}

#undef CC
#undef CH
#undef TW

/* The passes of each radix, each a function of its own: inlined into
   run_passes together, the outputs stayed on the stack (a 960-byte
   frame), and lines of 64 and 255 points took 2 to 4% longer (M1, one
   thread). */
static __attribute__((noinline)) void F(pass2)(int64_t ido, int64_t l1,
                                               const CV *restrict cc,
                                               CV *restrict ch,
                                               const T *tw) {
  F(pass)(2, ido, l1, cc, ch, tw, NULL);
}

static __attribute__((noinline)) void F(pass3)(int64_t ido, int64_t l1,
                                               const CV *restrict cc,
                                               CV *restrict ch,
                                               const T *tw) {
  F(pass)(3, ido, l1, cc, ch, tw, NULL);
}

static __attribute__((noinline)) void F(pass4)(int64_t ido, int64_t l1,
                                               const CV *restrict cc,
                                               CV *restrict ch,
                                               const T *tw) {
  F(pass)(4, ido, l1, cc, ch, tw, NULL);
}

static __attribute__((noinline)) void F(pass5)(int64_t ido, int64_t l1,
                                               const CV *restrict cc,
                                               CV *restrict ch,
                                               const T *tw) {
  F(pass)(5, ido, l1, cc, ch, tw, NULL);
}

static __attribute__((noinline)) void F(passg)(int ip, int64_t ido,
                                               int64_t l1,
                                               const CV *restrict cc,
                                               CV *restrict ch, const T *tw,
                                               const T *cs) {
  F(pass)(ip, ido, l1, cc, ch, tw, cs);
}

/* Transforms the lanes at [a], [p]'s n points, by its passes, with [b]
   of as many: the buffer that holds the result. */
static CV *F(run_passes)(const PLAN *p, CV *a, CV *b) {
  int64_t l1 = 1;
  for (int k = 0; k < p->passes; k++) {
    int ip = p->radix[k];
    int64_t ido = p->n / (l1 * ip);
    const T *tw = p->tw[k];
    switch (ip) {
      case 2: F(pass2)(ido, l1, a, b, tw); break;
      case 3: F(pass3)(ido, l1, a, b, tw); break;
      case 4: F(pass4)(ido, l1, a, b, tw); break;
      case 5: F(pass5)(ido, l1, a, b, tw); break;
      default: F(passg)(ip, ido, l1, a, b, tw, p->cs[k]);
    }
    CV *t = a;
    a = b;
    b = t;
    l1 *= ip;
  }
  return a;
}

/* Lines to lanes and back */

/* Lanes [0, lanes) of [d], [n] points of lines [o] of x, [st] reals
   apart: lanes past [lanes] are +0. Three ways, by how the points lie:
   lanes side by side, points side by side (L × L tiles, transposed), or
   any. */
static void F(gather)(CV *restrict d, const lines *x, const int64_t *o,
                      int lanes, int64_t n, int64_t st) {
  const T *b = (const T *)x->base;
  int64_t io = x->io;
  int conj = x->conj;
  if (lanes == L && side_by_side(x, o, L)) {
    for (int64_t j = 0; j < n; j++) {
      VEC lo, hi;
      memcpy(&lo, b + o[0] + j * st, sizeof lo);
      memcpy(&hi, b + o[0] + j * st + L, sizeof hi);
      VEC im = DEAL_IM(lo, hi);
      d[j] = (CV){DEAL_RE(lo, hi), conj ? -im : im};
    }
    return;
  }
  int64_t j = 0;
  if (lanes == L && st == 2 && io == 1 && !x->real)
    for (; j + L <= n; j += L) {
      VEC re[L], im[L];
      UNROLL
      for (int v = 0; v < L; v++) {
        VEC lo, hi;
        memcpy(&lo, b + o[v] + 2 * j, sizeof lo);
        memcpy(&hi, b + o[v] + 2 * j + L, sizeof hi);
        re[v] = DEAL_RE(lo, hi);
        im[v] = DEAL_IM(lo, hi);
        if (conj) im[v] = -im[v];
      }
      TRANSPOSE(re);
      TRANSPOSE(im);
      UNROLL
      for (int u = 0; u < L; u++) d[j + u] = (CV){re[u], im[u]};
    }
  for (; j < n; j++)
    for (int v = 0; v < L; v++) {
      const T *p = b + o[v] + j * st;
      T im = v < lanes && !x->real ? p[io] : 0;
      d[j].re[v] = v < lanes ? p[0] : 0;
      d[j].im[v] = conj ? -im : im;
    }
}

/* Stores the two vectors at [v], 2 L reals, at [d] on 32 bytes, past the
   caches (nx_stream32). By pointer: a vector of 32 bytes passed by value
   has another ABI on x86-64 without AVX. Always inlined, so that the
   vectors go from registers to the stores. */
static inline __attribute__((always_inline)) void F(nt)(T *d, const VEC *v) {
  for (int i = 0; i < 2 * VEC_BYTES; i += 32)
    nx_stream32((uint8_t *)d + i, (const uint8_t *)v + i);
}

/* Whether y's pieces of 2 L reals, the first at [d] and the others [st]
   reals apart, are stored past the caches: where y streams, its points
   lie apart, and the pieces lie on STREAM_PIECE bytes. */
static int F(streams)(const lines *y, const T *d, int64_t st) {
  return y->stream && st > 2 * L &&
         st * (int64_t)sizeof(T) % STREAM_PIECE == 0 &&
         (uintptr_t)d % STREAM_PIECE == 0;
}

/* Writes the lanes of [s] at [d] as L complex numbers side by side,
   their conjugates where [conj], past the caches where [nt]. */
static inline void F(zip)(T *d, CV s, int conj, int nt) {
  VEC im = conj ? -s.im : s.im;
  VEC v[2] = {ZIP_LO(s.re, im), ZIP_HI(s.re, im)};
  if (nt) {
    F(nt)(d, v);
    return;
  }
  memcpy(d, v, sizeof v);
}

/* Writes lanes [0, lanes) of [s], points [0, n) of lines [o] of y, [st]
   reals apart: lanes side by side by stores past the caches where y
   streams and its points lie apart. */
static void F(scatter)(const lines *y, const int64_t *o, int lanes,
                       int64_t n, int64_t st, const CV *restrict s) {
  T *b = (T *)y->base;
  int64_t io = y->io;
  int conj = y->conj;
  if (lanes == L && side_by_side(y, o, L)) {
    int nt = F(streams)(y, b + o[0], st);
    for (int64_t j = 0; j < n; j++) F(zip)(b + o[0] + j * st, s[j], conj, nt);
    if (nt) nx_stream_drain();
    return;
  }
  int64_t j = 0;
  if (lanes == L && st == 2 && io == 1 && !y->real)
    for (; j + L <= n; j += L) {
      VEC re[L], im[L];
      UNROLL
      for (int u = 0; u < L; u++) {
        re[u] = s[j + u].re;
        im[u] = conj ? -s[j + u].im : s[j + u].im;
      }
      TRANSPOSE(re);
      TRANSPOSE(im);
      UNROLL
      for (int v = 0; v < L; v++)
        F(zip)(b + o[v] + 2 * j, (CV){re[v], im[v]}, 0, 0);
    }
  for (; j < n; j++)
    for (int v = 0; v < lanes; v++) {
      T *p = b + o[v] + j * st;
      p[0] = s[j].re[v];
      if (!y->real) p[io] = conj ? -s[j].im[v] : s[j].im[v];
    }
}

/* The lines of [lo, hi) by passes, L at a time. */
static int F(direct)(const PLAN *p, const lines *x, const lines *y,
                     int64_t lo, int64_t hi) {
  int64_t n = p->n;
  CV *a = nx_cpu_take(sizeof(CV) * n), *b = nx_cpu_take(sizeof(CV) * n);
  int ok = a != NULL && b != NULL;
  for (int64_t l = lo; ok && l < hi; l += L) {
    int lanes = hi - l < L ? (int)(hi - l) : L;
    int64_t o[L], q[L];
    for (int v = 0; v < L; v++) {
      o[v] = line_at(x, l + (v < lanes ? v : 0));
      q[v] = line_at(y, l + (v < lanes ? v : 0));
    }
    F(gather)(a, x, o, lanes, n, x->step);
    F(scatter)(y, q, lanes, n, y->step, F(run_passes)(p, a, b));
  }
  nx_cpu_give(a);
  nx_cpu_give(b);
  return ok;
}

/* The four steps of a line, Bailey's: its points as n1 rows of n2,
   j = j1 n2 + j2; each column transformed and twiddled by ω^(k1 j2); each
   row transformed; point k1 + n1 k2 of the result is row k1's point k2.
   A unit is a line, its lanes L columns, then L rows, or L lines side by
   side, a line a lane. Each line's arithmetic is the same either way.

   Between the two steps the columns lie in a grid in scratch, in the
   rows' lanes, which their passes run on in place: L rows of a line a
   vector, or a row of L lines. */

/* A unit of the four steps: lane v's line of x and of y at xo[v] and
   yo[v], one line's for every lane where the unit is one. Its grid is s,
   rows of [width] vectors. */
typedef struct {
  const PLAN *p;
  const lines *x, *y;
  int side;              /* lines side by side */
  int64_t columns, rows; /* its column and row steps */
  CV *s;
  int64_t width;
  int64_t xo[L], yo[L];
} F(unit);

/* The vectors of a unit's grid: blocks of L rows, each as
   wide as n2 rounded up to L, or n1 rows of L lines. */
static int64_t F(grid)(const PLAN *p, int side) {
  return side ? p->n : (p->n1 + L - 1) / L * ((p->n2 + L - 1) / L * L);
}

/* The unit of line [l], or of lines [l, l + L) side by side where
   [side], its grid at [s]. */
static void F(unit_at)(F(unit) * u, const PLAN *p, const lines *x,
                       const lines *y, int64_t l, int side, CV *s) {
  int64_t blocks = (p->n1 + L - 1) / L, groups = (p->n2 + L - 1) / L;
  *u = (F(unit)){.p = p,
                 .x = x,
                 .y = y,
                 .side = side,
                 .columns = side ? p->n2 : groups,
                 .rows = side ? p->n1 : blocks,
                 .s = s,
                 .width = side ? p->n2 : groups * L};
  for (int v = 0; v < L; v++) {
    int64_t k = l + (side ? v : 0);
    u->xo[v] = line_at(x, k);
    u->yo[v] = line_at(y, k);
  }
}

/* L consecutive points of a column group's transform [r], lanes [0,
   lanes) of them twiddled by [tw] and the rest +0, transposed into [t]:
   t[v] holds the L points' lane v. */
static inline void F(twist)(const CV *r, const CV *tw, int lanes, CV *t) {
  VEC re[L], im[L];
  UNROLL
  for (int u = 0; u < L; u++) {
    CV z = u < lanes ? F(mulv)(r[u], tw[u]) : (CV){0};
    re[u] = z.re;
    im[u] = z.im;
  }
  TRANSPOSE(re);
  TRANSPOSE(im);
  UNROLL
  for (int v = 0; v < L; v++) t[v] = (CV){re[v], im[v]};
}

/* Column step [c] of [u]: column c of its lines, or its line's columns
   [c L, c L + L), from x into the grid, through [a] and [b]. */
static void F(column)(const F(unit) * u, int64_t c, CV *a, CV *b) {
  const PLAN *p = u->p;
  const lines *x = u->x;
  int64_t n1 = p->n1, n2 = p->n2, o[L];
  int lanes = u->side || n2 - c * L >= L ? L : (int)(n2 - c * L);
  for (int v = 0; v < L; v++)
    o[v] = u->xo[v] + (u->side ? c : c * L + (v < lanes ? v : 0)) * x->step;
  F(gather)(a, x, o, lanes, n1, n2 * x->step);
  CV *r = F(run_passes)(p->p1, a, b);
  /* Lane v's column is j2 = c L + v, or c for every lane side by side. */
  const CV *tw = p->tw4 + (u->side ? c / L : c) * n1;
  if (u->side) {
    for (int64_t k1 = 0; k1 < n1; k1++)
      u->s[k1 * n2 + c] = F(mul)(r[k1], tw[k1].re[c % L], tw[k1].im[c % L]);
    return;
  }
  /* L rows of the L columns at a time, twiddled and transposed into the
     grid's block. */
  for (int64_t h = 0; h < n1; h += L)
    F(twist)(r + h, tw + h, n1 - h < L ? (int)(n1 - h) : L,
             u->s + h / L * u->width + c * L);
}

/* The column groups of a line of n1 × 8 points. */
#define G8 (8 / L)

/* The transform of the eight points c into z by the arithmetic of the
   rows' plan p2 of the four steps of n1 × 8 points: its pass of radix 2,
   the differences turned by its twiddles ω_8^i, then its pass of radix 4
   on each half (passes_make). Always inlined, its loops unrolled, so that
   gcc too keeps the points in registers: it kept pass2's loop of four,
   then a call of eight, and with them the points in memory. */
static inline __attribute__((always_inline)) void F(eight)(const PLAN *p2,
                                                          const CV *c, CV *z) {
  const T *w = p2->tw[0];
  CV a[4], d[4];
  UNROLL
  for (int i = 0; i < 4; i++) {
    CV y[2];
    F(two)(c + i, 4, y);
    a[i] = y[0];
    d[i] = i == 0 ? y[1] : F(mul)(y[1], w[2 * i], w[2 * i + 1]);
  }
  CV ya[4], yd[4];
  F(four)(a, 1, ya);
  F(four)(d, 1, yd);
  UNROLL
  for (int u = 0; u < 4; u++) {
    z[2 * u] = ya[u];
    z[2 * u + 1] = yd[u];
  }
}

/* Rows [h, h + lanes) of a line of n1 × 8 points into y at [to], from
   its columns' transforms [r]: twiddled, transposed, transformed and
   written side by side at [out] where it is not NULL, conjugated where
   [conj], past the caches where [nt]. Inlined with lanes = L, its tests
   vanish from the line's every block but the last. */
static inline __attribute__((always_inline)) void F(rows8)(
    const PLAN *p, const CV *const *r, int64_t h, int lanes, const lines *y,
    int64_t to, T *out, int conj, int nt) {
  int64_t n1 = p->n1;
  CV c[8], z[8];
  UNROLL
  for (int g = 0; g < G8; g++)
    F(twist)(r[g] + h, p->tw4 + g * n1 + h, lanes, c + g * L);
  F(eight)(p->p2, c, z);
  if (out != NULL && lanes == L) {
    UNROLL
    for (int k2 = 0; k2 < 8; k2++)
      F(zip)(out + 2 * (h + n1 * k2), z[k2], conj, nt);
    return;
  }
  /* scatter reads a copy, w: with z's address taken, the compiler kept z
     in memory on the zip path too, and the line took 10% longer. */
  CV w[8];
  int64_t q[L];
  for (int k2 = 0; k2 < 8; k2++) w[k2] = z[k2];
  for (int v = 0; v < L; v++) q[v] = to + (h + (v < lanes ? v : 0)) * y->step;
  F(scatter)(y, q, lanes, 8, n1 * y->step, w);
}

/* Line [l] of n1 × 8 points by the four steps, on this thread: the
   columns by passes, then the rows L at a time in registers (rows8),
   with no grid. */
static int F(within8)(const PLAN *p, const lines *x, const lines *y,
                      int64_t l) {
  int64_t n1 = p->n1;
  CV *buf = nx_cpu_take(sizeof(CV) * 2 * G8 * n1);
  if (buf == NULL) return 0;
  const CV *r[G8];
  int64_t at = line_at(x, l), to = line_at(y, l), o[L];
  for (int g = 0; g < G8; g++) {
    for (int v = 0; v < L; v++) o[v] = at + (g * L + v) * x->step;
    CV *a = buf + 2 * g * n1;
    F(gather)(a, x, o, L, n1, 8 * x->step);
    r[g] = F(run_passes)(p->p1, a, a + n1);
  }
  /* The stores might alias y: its fields are read once, here. A block's
     L rows are L points of one line, side by side where the line's
     points are adjacent complex numbers (side_by_side asks it of lines). */
  int conj = y->conj;
  T *out = y->step == 2 && y->io == 1 && !y->real ? (T *)y->base + to : NULL;
  int nt = out != NULL && F(streams)(y, out, 2 * n1);
  int64_t h = 0;
  for (; h + L <= n1; h += L) F(rows8)(p, r, h, L, y, to, out, conj, nt);
  if (h < n1) F(rows8)(p, r, h, (int)(n1 - h), y, to, out, conj, nt);
  if (nt) nx_stream_drain();
  nx_cpu_give(buf);
  return 1;
}

/* Row step [r] of [u]: row r of its lines, or its line's rows [r L,
   r L + L), from the grid into y, through [b]. Rows of 8 of
   lines side by side go by F(eight) and to y from registers, their
   streams drained by the caller: through b and scatter, fft2 of 1024 ×
   1024 took 3% longer (Arrow Lake, M1). */
static void F(row)(const F(unit) * u, int64_t r, CV *b) {
  const PLAN *p = u->p;
  const lines *y = u->y;
  int64_t n1 = p->n1, n2 = p->n2, o[L];
  int lanes = u->side || n1 - r * L >= L ? L : (int)(n1 - r * L);
  for (int v = 0; v < L; v++)
    o[v] = u->yo[v] + (u->side ? r : r * L + (v < lanes ? v : 0)) * y->step;
  CV *rows = u->s + r * u->width;
  if (n2 == 8 && u->side) {
    CV z[8];
    F(eight)(p->p2, rows, z);
    T *d = (T *)y->base + o[0];
    int64_t st = n1 * y->step;
    int nt = F(streams)(y, d, st);
    for (int k2 = 0; k2 < 8; k2++) F(zip)(d + k2 * st, z[k2], y->conj, nt);
    return;
  }
  F(scatter)(y, o, lanes, n2, n1 * y->step, F(run_passes)(p->p2, rows, b));
}

/* The points of a step's lanes buffers a and b. */
static int64_t F(four_lanes)(const PLAN *p) {
  return p->n1 > p->n2 ? p->n1 : p->n2;
}

/* Line [l], or lines [l, l + L) side by side where [side], by the four
   steps, on this thread. */
static int F(steps)(const PLAN *p, const lines *x, const lines *y,
                    int64_t l, int side) {
  int64_t lanes = F(four_lanes)(p);
  CV *s = nx_cpu_take(sizeof(CV) * F(grid)(p, side)),
     *a = nx_cpu_take(sizeof(CV) * lanes), *b = nx_cpu_take(sizeof(CV) * lanes);
  int ok = s != NULL && a != NULL && b != NULL;
  F(unit) u;
  F(unit_at)(&u, p, x, y, l, side, s);
  for (int64_t c = 0; ok && c < u.columns; c++) F(column)(&u, c, a, b);
  for (int64_t r = 0; ok && r < u.rows; r++) F(row)(&u, r, b);
  nx_stream_drain();
  nx_cpu_give(s);
  nx_cpu_give(a);
  nx_cpu_give(b);
  return ok;
}

/* Points one at a time */

static inline void F(load)(const lines *x, int64_t o, int64_t j, T *re,
                           T *im) {
  const T *p = (const T *)x->base + o + j * x->step;
  *re = p[0];
  *im = x->real ? 0 : x->conj ? -p[x->io] : p[x->io];
}

static inline void F(store)(const lines *y, int64_t o, int64_t j, T re,
                            T im) {
  T *p = (T *)y->base + o + j * y->step;
  p[0] = re;
  if (!y->real) p[y->io] = y->conj ? -im : im;
}

/* Bluestein's convolution (pocketfft's fftblue): the transform of x at
   k is c_k Σ_j (x_j c_j) conj c_(k - j), c the chirp, since
   jk = (j² + k² - (k - j)²) / 2: a cyclic convolution of m >= 2n - 1
   points, by the forward transform twice and the conjugates. A row
   holds a line's m points, interleaved. */

/* Points [lo, hi) of the row [r] of the line at [o] of x: x_j c_j, +0
   from n on. */
static void F(chirped)(const PLAN *p, const lines *x, int64_t o, T *r,
                       int64_t lo, int64_t hi) {
  const T *c = p->chirp;
  for (int64_t j = lo; j < hi; j++) {
    T re = 0, im = 0;
    if (j < p->n) {
      F(load)(x, o, j, &re, &im);
      T t = re * c[2 * j] - im * c[2 * j + 1];
      im = re * c[2 * j + 1] + im * c[2 * j];
      re = t;
    }
    r[2 * j] = re;
    r[2 * j + 1] = im;
  }
}

/* Points [lo, hi) of the row [r]: conj (r_k K_k), K the kernel. */
static void F(convolve)(const PLAN *p, T *r, int64_t lo, int64_t hi) {
  const T *K = p->kernel;
  for (int64_t k = lo; k < hi; k++) {
    T re = r[2 * k] * K[2 * k] - r[2 * k + 1] * K[2 * k + 1];
    T im = r[2 * k] * K[2 * k + 1] + r[2 * k + 1] * K[2 * k];
    r[2 * k] = re;
    r[2 * k + 1] = -im;
  }
}

/* Points [lo, hi) of the line at [o] of y: conj (r_k) c_k. */
static void F(unchirped)(const PLAN *p, const T *r, const lines *y, int64_t o,
                         int64_t lo, int64_t hi) {
  const T *c = p->chirp;
  for (int64_t k = lo; k < hi; k++) {
    T re = r[2 * k], im = -r[2 * k + 1];
    F(store)(y, o, k, re * c[2 * k] - im * c[2 * k + 1],
             re * c[2 * k + 1] + im * c[2 * k]);
  }
}

/* The rows of [count] lines of m points at [r], as lines. */
static lines F(rows)(T *r, int64_t count, int64_t m) {
  return (lines){.base = (uint8_t *)r,
                 .step = 2,
                 .io = 1,
                 .rank = 1,
                 .dim = {count},
                 .stride = {2 * m}};
}

static int F(serial)(const PLAN *p, const lines *x, const lines *y,
                     int64_t lo, int64_t hi);

/* Lines [l, l + c), c <= L, by Bluestein's convolution, on this
   thread. */
static int F(blue)(const PLAN *p, const lines *x, const lines *y, int64_t l,
                   int64_t c) {
  int64_t m = p->m;
  T *r = nx_cpu_take(sizeof(T) * 2 * m * c);
  if (r == NULL) return 0;
  for (int64_t v = 0; v < c; v++)
    F(chirped)(p, x, line_at(x, l + v), r + 2 * m * v, 0, m);
  lines rows = F(rows)(r, c, m);
  int ok = F(serial)(p->pm, &rows, &rows, 0, c);
  for (int64_t v = 0; ok && v < c; v++) F(convolve)(p, r + 2 * m * v, 0, m);
  ok = ok && F(serial)(p->pm, &rows, &rows, 0, c);
  for (int64_t v = 0; ok && v < c; v++)
    F(unchirped)(p, r + 2 * m * v, y, line_at(y, l + v), 0, p->n);
  nx_cpu_give(r);
  return ok;
}

/* Lines [lo, hi) of x by [p] into y, on this thread. The four steps run
   L lines at once where their points are side by side, else line by
   line. */
static int F(serial)(const PLAN *p, const lines *x, const lines *y,
                     int64_t lo, int64_t hi) {
  switch (p->kind) {
    case PASSES: return F(direct)(p, x, y, lo, hi);
    case FOUR:
      for (int64_t l = lo; l < hi;) {
        int side = hi - l >= L && lines_side_by_side(x, l, L) &&
                   lines_side_by_side(y, l, L);
        int ok = !side && p->n2 == 8 ? F(within8)(p, x, y, l)
                                     : F(steps)(p, x, y, l, side);
        if (!ok) return 0;
        l += side ? L : 1;
      }
      return 1;
    default:
      for (int64_t l = lo; l < hi; l += L)
        if (!F(blue)(p, x, y, l, hi - l < L ? hi - l : L)) return 0;
      return 1;
  }
}

/* Real lines, through complex ones of half as many points (pocketfft's
   r2c by the half-length transform): for a halved n = 2h, z_j =
   x_2j + i x_(2j+1) has the transform Z, and X_k = E_k - i ω^k D_k with
   E_k = (Z_k + conj Z_(h-k)) / 2 and D_k = (Z_k - conj Z_(h-k)) / 2,
   bins k and h - k computed together. A real line's plan [rp] is REAL's:
   of its h complex points with ω^k where n is halved, else of its n. */

/* Whether [x]'s complex points lie side by side, interleaved, read and
   written as they are. */
static int F(dense)(const lines *x) {
  return x->step == 2 && x->io == 1 && !x->real && !x->conj;
}

/* L points from [p], interleaved, in lanes; and back. */
static inline CV F(load_lanes)(const T *p) {
  VEC lo, hi;
  memcpy(&lo, p, sizeof lo);
  memcpy(&hi, p + L, sizeof hi);
  return (CV){DEAL_RE(lo, hi), DEAL_IM(lo, hi)};
}

static inline void F(store_lanes)(T *p, CV z) {
  VEC lo = ZIP_LO(z.re, z.im), hi = ZIP_HI(z.re, z.im);
  memcpy(p, &lo, sizeof lo);
  memcpy(p + L, &hi, sizeof hi);
}

/* Whether pairs [k, k + L), past 0 and below [hi), compute in lanes:
   their bins h - k - L < j <= h - k lie apart from bins [k, k + L), so
   no pair reads a bin another writes. */
static int F(pair_lanes)(int64_t k, int64_t hi, int64_t h) {
  return k > 0 && k + L <= hi && 2 * (k + L) - 1 <= h;
}

/* Pairs [lo, hi) of k <= h / 2 of the line at [o] of y, Z in its first
   h bins, into X. Where y's points lie side by side, L pairs compute a
   vector, bins h - k in reversed lanes, each pair's arithmetic as one
   pair's. */
static void F(split)(const PLAN *rp, const lines *y, int64_t o,
                     int64_t lo, int64_t hi) {
  int64_t h = rp->n;
  const T *w = rp->w;
  int dense = F(dense)(y);
  T *p = (T *)y->base + o;
  for (int64_t k = lo; k < hi; k++) {
    if (dense && F(pair_lanes)(k, hi, h)) {
      int64_t m = h - k - (L - 1);
      CV a = F(load_lanes)(p + 2 * k), b = F(load_lanes)(p + 2 * m),
         wk = F(load_lanes)(w + 2 * k);
      VEC br = REV(b.re), bi = REV(b.im);
      VEC er = (a.re + br) * (T)0.5, ei = (a.im - bi) * (T)0.5;
      VEC dr = (a.re - br) * (T)0.5, di = (a.im + bi) * (T)0.5;
      VEC fr = -(wk.re * di + wk.im * dr), fi = wk.re * dr - wk.im * di;
      F(store_lanes)(p + 2 * m, (CV){REV(er + fr), REV(-(ei + fi))});
      F(store_lanes)(p + 2 * k, (CV){er - fr, ei - fi});
      k += L - 1;
      continue;
    }
    T ar, ai, br, bi;
    F(load)(y, o, k, &ar, &ai);
    if (k == 0) {
      F(store)(y, o, h, ar - ai, 0);
      F(store)(y, o, 0, ar + ai, 0);
      continue;
    }
    F(load)(y, o, h - k, &br, &bi);
    T er = (ar + br) * (T)0.5, ei = (ai - bi) * (T)0.5;
    T dr = (ar - br) * (T)0.5, di = (ai + bi) * (T)0.5;
    /* f = i ω^k d */
    T fr = -(w[2 * k] * di + w[2 * k + 1] * dr);
    T fi = w[2 * k] * dr - w[2 * k + 1] * di;
    F(store)(y, o, h - k, er + fr, -(ei + fi));
    F(store)(y, o, k, er - fr, ei - fi);
  }
}

/* The inverse: Z_k = S_k + i conj ω^k D_k with S_k = X_k + conj X_(h-k)
   and D_k = X_k - conj X_(h-k), the imaginary parts of X_0 and X_h
   taken as +0. Pairs [lo, hi) of k <= h / 2 of the line at [o] of x into
   the row [z] of h points, L pairs a vector as split's. */
static void F(merge)(const PLAN *rp, const lines *x, int64_t o, T *z,
                     int64_t lo, int64_t hi) {
  int64_t h = rp->n;
  const T *w = rp->w;
  int dense = F(dense)(x);
  const T *p = (const T *)x->base + o;
  for (int64_t k = lo; k < hi; k++) {
    if (dense && F(pair_lanes)(k, hi, h)) {
      int64_t m = h - k - (L - 1);
      CV a = F(load_lanes)(p + 2 * k), b = F(load_lanes)(p + 2 * m),
         wk = F(load_lanes)(w + 2 * k);
      VEC br = REV(b.re), bi = REV(b.im);
      VEC sr = a.re + br, si = a.im - bi, dr = a.re - br, di = a.im + bi;
      VEC gr = wk.im * dr - wk.re * di, gi = wk.re * dr + wk.im * di;
      F(store_lanes)(z + 2 * m, (CV){REV(sr - gr), REV(-(si - gi))});
      F(store_lanes)(z + 2 * k, (CV){sr + gr, si + gi});
      k += L - 1;
      continue;
    }
    T ar, ai, br, bi;
    F(load)(x, o, k, &ar, &ai);
    F(load)(x, o, h - k, &br, &bi);
    if (k == 0) {
      z[0] = ar + br;
      z[1] = ar - br;
      continue;
    }
    T sr = ar + br, si = ai - bi, dr = ar - br, di = ai + bi;
    /* g = i conj(ω^k) d */
    T gr = w[2 * k + 1] * dr - w[2 * k] * di;
    T gi = w[2 * k] * dr + w[2 * k + 1] * di;
    z[2 * (h - k)] = sr - gr;
    z[2 * (h - k) + 1] = -(si - gi);
    z[2 * k] = sr + gr;
    z[2 * k + 1] = si + gi;
  }
}

/* A line not halved: points [lo, hi) of the row [z] of n points, X
   completed by conj X_(n-k) past n / 2, the imaginary parts of X_0 and,
   for an even n, of X_(n/2) taken as +0. */
static void F(complete)(const PLAN *rp, const lines *x, int64_t o, T *z,
                        int64_t lo, int64_t hi) {
  int64_t n = rp->n;
  for (int64_t k = lo; k < hi; k++) {
    T re, im;
    F(load)(x, o, k <= n / 2 ? k : n - k, &re, &im);
    z[2 * k] = re;
    z[2 * k + 1] = k == 0 || 2 * k == n ? 0 : k <= n / 2 ? im : -im;
  }
}

/* A line not halved: bins [lo, hi) of the row [z] into the line at [o]
   of y. */
static void F(halve)(const T *z, const lines *y, int64_t o, int64_t lo,
                     int64_t hi) {
  for (int64_t k = lo; k < hi; k++) F(store)(y, o, k, z[2 * k], z[2 * k + 1]);
}

/* Jobs */

static int F(transform)(const PLAN *p, const lines *x, const lines *y,
                        int64_t count, int threads);

/* A job of one step over its units: lines, a line's column or row steps,
   or chunks of a line's or of every line's points. */
typedef struct {
  int step;
  const PLAN *p;
  const lines *x, *y;
  int64_t count; /* the lines */
  int64_t per;   /* lines a unit, or chunks a line */
  int64_t l;     /* the line whose steps the job runs */
  F(unit) * u;   /* that line's four steps */
  T *r;          /* its Bluestein row, or every line's complex row */
  int failed;
} F(job);

static void F(body)(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  F(job) *j = ctx;
  const PLAN *p = j->p;
  int ok = 1;
  /* The plan's points: a halved real line's h, another line's n. */
  int64_t n = p->n;
  switch (j->step) {
    case LINES: {
      int64_t end = hi * j->per < j->count ? hi * j->per : j->count;
      ok = F(serial)(p, j->x, j->y, lo * j->per, end);
      break;
    }
    case COLUMNS:
    case ROWS: {
      int64_t lanes = F(four_lanes)(p);
      CV *a = nx_cpu_take(sizeof(CV) * lanes),
         *b = nx_cpu_take(sizeof(CV) * lanes);
      ok = a != NULL && b != NULL;
      for (int64_t u = lo; ok && u < hi; u++)
        if (j->step == COLUMNS) F(column)(j->u, u, a, b);
        else F(row)(j->u, u, b);
      nx_cpu_give(a);
      nx_cpu_give(b);
      break;
    }
    default:
      for (int64_t u = lo; u < hi; u++) {
        int64_t a, b;
        switch (j->step) {
          case CHIRP:
            chunk(u, j->per, p->m, &a, &b);
            F(chirped)(p, j->x, line_at(j->x, j->l), j->r, a, b);
            break;
          case CONVOLVE:
            chunk(u, j->per, p->m, &a, &b);
            F(convolve)(p, j->r, a, b);
            break;
          case UNCHIRP:
            chunk(u, j->per, n, &a, &b);
            F(unchirped)(p, j->r, j->y, line_at(j->y, j->l), a, b);
            break;
          case SPLIT: {
            int64_t l = chunk(u, j->per, n / 2 + 1, &a, &b);
            F(split)(p, j->y, line_at(j->y, l), a, b);
            break;
          }
          case MERGE: {
            int64_t l = chunk(u, j->per, n / 2 + 1, &a, &b);
            F(merge)(p, j->x, line_at(j->x, l), j->r + 2 * n * l, a, b);
            break;
          }
          case COMPLETE: {
            int64_t l = chunk(u, j->per, n, &a, &b);
            F(complete)(p, j->x, line_at(j->x, l), j->r + 2 * n * l, a, b);
            break;
          }
          case HALVE: {
            int64_t l = chunk(u, j->per, n / 2 + 1, &a, &b);
            F(halve)(j->r + 2 * n * l, j->y, line_at(j->y, l), a, b);
          }
        }
      }
  }
  if (!ok) __atomic_store_n(&j->failed, 1, __ATOMIC_RELAXED);
}

/* Runs [j]'s step over [units] that touch [bytes] and cost [cost] as
   job.c prices them, on one thread where [threads] is 1. */
static int F(run)(F(job) * j, int64_t units, int64_t bytes, double cost,
                  int threads) {
  if (units == 0) return 1;
  if (threads == 1) F(body)(0, units, 0, j);
  else nx_cpu_job(units, bytes, cost, F(body), j);
  return !j->failed;
}

/* Line j->l's four steps, each a job over j->u: its column steps, then
   its row steps. */
static int F(four_steps)(F(job) * j, int64_t bytes, double cost,
                         int threads) {
  j->step = COLUMNS;
  if (!F(run)(j, j->u->columns, bytes, cost, threads)) return 0;
  j->step = ROWS;
  return F(run)(j, j->u->rows, bytes, cost, threads);
}

/* Line j->l by Bluestein's convolution, each step a job, in the shared
   row j->r. */
static int F(blue_steps)(F(job) * j, int64_t bytes, double cost,
                         int threads) {
  const PLAN *p = j->p;
  lines rows = F(rows)(j->r, 1, p->m);
  int64_t chunks = (p->m + CHUNK - 1) / CHUNK;
  j->per = chunks;
  j->step = CHIRP;
  if (!F(run)(j, chunks, bytes, cost, threads)) return 0;
  if (!F(transform)(p->pm, &rows, &rows, 1, threads)) return 0;
  j->step = CONVOLVE;
  if (!F(run)(j, chunks, bytes, cost, threads)) return 0;
  if (!F(transform)(p->pm, &rows, &rows, 1, threads)) return 0;
  j->step = UNCHIRP;
  j->per = (p->n + CHUNK - 1) / CHUNK;
  return F(run)(j, j->per, bytes, cost, threads);
}

/* [count] lines of x by [p] into y. Lines are the units where there are
   enough for the threads; else each line's steps are jobs of their own,
   which share a grid, or Bluestein's row, in scratch the call holds. */
static int F(transform)(const PLAN *p, const lines *x, const lines *y,
                        int64_t count, int threads) {
  int64_t n = p->n, bytes = 4 * (int64_t)sizeof(T) * n * count;
  double cost = cost_of(bytes, n, p->kind == BLUESTEIN ? p->m : 0);
  /* A result streams where its lines' bytes gain on the threads that
     write them (streams). */
  int t = threads == 1 ? 1 : nx_cpu_threads(bytes, cost);
  int64_t out = (y->real ? 1 : 2) * (int64_t)sizeof(T) * n * count;
  lines result;
  if (y->stream) {
    result = *y;
    result.stream = streams(out, t);
    y = &result;
  }
  F(job) j = {.step = LINES, .p = p, .x = x, .y = y, .count = count};
  j.per = p->kind == FOUR ? 1 : L;
  /* Lines of four steps side by side go L to a unit where each thread
     still has one, so that a unit writes whole lines (F(streams)). */
  if (p->kind == FOUR && lines_side_by_side(y, 0, L) && count / L >= t)
    j.per = L;
  int64_t units = (count + j.per - 1) / j.per;
  if (threads == 1 || p->kind == PASSES ||
      units >= nx_cpu_threads(bytes, cost))
    return F(run)(&j, units, bytes, cost, threads);
  int four = p->kind == FOUR;
  int64_t shared = four ? (int64_t)sizeof(CV) * F(grid)(p, 0)
                        : (int64_t)sizeof(T) * 2 * p->m;
  void *buffer = nx_cpu_take(shared);
  if (buffer == NULL) return 0;
  F(unit) u;
  j.u = &u;
  j.r = buffer;
  int ok = 1;
  for (j.l = 0; j.l < count && ok; j.l++) {
    if (four) F(unit_at)(&u, p, x, y, j.l, 0, buffer);
    ok = four ? F(four_steps)(&j, bytes / count, cost / count, threads)
              : F(blue_steps)(&j, bytes / count, cost / count, threads);
  }
  nx_cpu_give(buffer);
  return ok;
}

/* [count] real lines of [n] points of x into y's bins by [rp], as r2c
   states. */
static int F(r2c_lines)(const PLAN *rp, int64_t n, const lines *x,
                        const lines *y, int64_t count, int threads) {
  int64_t h = n / 2, bytes = 3 * (int64_t)sizeof(T) * n * count;
  F(job) j = {.p = rp, .x = x, .y = y, .count = count};
  if (halved(n)) {
    lines pairs = *x;
    pairs.step = 2 * x->step;
    pairs.io = x->step;
    pairs.real = 0;
    if (!F(transform)(rp, &pairs, y, count, threads)) return 0;
    j.step = SPLIT;
    j.per = (h / 2 + 1 + CHUNK - 1) / CHUNK;
    return F(run)(&j, count * j.per, bytes, HALF_COST * bytes, threads);
  }
  T *r = malloc(sizeof(T) * 2 * n * count);
  if (r == NULL) return 0;
  lines rows = F(rows)(r, count, n);
  int ok = F(transform)(rp, x, &rows, count, threads);
  j.step = HALVE;
  j.r = r;
  j.per = (h + 1 + CHUNK - 1) / CHUNK;
  ok = ok && F(run)(&j, count * j.per, bytes, bytes, threads);
  free(r);
  return ok;
}

/* [count] lines of x's bins into y's real lines of [n] points by [rp],
   as c2r states. */
static int F(c2r_lines)(const PLAN *rp, int64_t n, const lines *x,
                        const lines *y, int64_t count, int threads) {
  int64_t h = n / 2, bytes = 3 * (int64_t)sizeof(T) * n * count;
  int half = halved(n);
  int64_t points = half ? h : n;
  T *r = malloc(sizeof(T) * 2 * (points > 0 ? points : 1) * count);
  if (r == NULL) return 0;
  F(job) j = {.p = rp, .x = x, .y = y, .count = count, .r = r};
  j.step = half ? MERGE : COMPLETE;
  j.per = ((half ? h / 2 + 1 : n) + CHUNK - 1) / CHUNK;
  int ok = F(run)(&j, count * j.per, bytes, (half ? HALF_COST : 1) * bytes,
                  threads);
  lines rows = F(rows)(r, count, points), out = *y;
  rows.conj = out.conj = 1;
  if (half) {
    out.step = 2 * y->step;
    out.io = y->step;
    out.real = 0;
  }
  ok = ok && F(transform)(rp, &rows, &out, count, threads);
  free(r);
  return ok;
}

/* The entries' work per precision */

/* The transform [inverse] of x into y along [axes], one after another,
   the first from x, the others in place in y. */
static int F(c2c)(const nx_array *x, const nx_array *y, const int *axes,
                  int na, int inverse, int threads) {
  for (int i = 0; i < na; i++) {
    int a = axes[i];
    int64_t n = y->dim[a], count = numel(y) / (n > 0 ? n : 1);
    if (n == 0 || count == 0) return 1;
    lines xs = lines_of(i == 0 ? x : y, a, y, 2), ys = lines_of(y, a, y, 2);
    xs.conj = ys.conj = inverse;
    ys.stream = 1;
    PLAN *p = plan_get(n, RDT, ANY);
    if (p == NULL) return 0;
    int ok = F(transform)(p, &xs, &ys, count, threads);
    plan_put(p, RDT);
    if (!ok) return 0;
  }
  return 1;
}

/* x's real lines along the last of [axes] into y's bins, then the
   forward transform along each of the others. */
static int F(r2c)(const nx_array *x, const nx_array *y, const int *axes,
                  int na, int threads) {
  int l = axes[na - 1];
  int64_t n = x->dim[l], count = numel(x) / (n > 0 ? n : 1);
  if (numel(y) == 0) return 1;
  if (n == 0) {
    zero(y);
    return 1;
  }
  lines xs = lines_of(x, l, y, 1), ys = lines_of(y, l, y, 2);
  xs.real = 1;
  ys.stream = 1;
  PLAN *rp = plan_get(n, RDT, REAL);
  if (rp == NULL) return 0;
  int ok = F(r2c_lines)(rp, n, &xs, &ys, count, threads);
  plan_put(rp, RDT);
  return ok && F(c2c)(y, y, axes, na - 1, 0, threads);
}

/* The inverse transform of x along each of [axes] but the last, into a
   copy, then x's bins along the last into y's real lines. */
static int F(c2r)(const nx_array *x, const nx_array *y, const int *axes,
                  int na, int threads) {
  int l = axes[na - 1];
  int64_t n = y->dim[l], count = numel(y) / (n > 0 ? n : 1);
  if (count == 0 || n == 0) return 1;
  nx_array copy = *x;
  T *c = NULL;
  if (na > 1) {
    c = malloc(sizeof(T) * 2 * numel(x));
    if (c == NULL) return 0;
    copy.base = (uint8_t *)c;
    copy.offset = 0;
    int64_t step = 1;
    for (int i = x->rank - 1; i >= 0; i--) {
      copy.dim[x->rank + i] = step;
      step *= x->dim[i];
    }
    if (!F(c2c)(x, &copy, axes, na - 1, 1, threads)) {
      free(c);
      return 0;
    }
  }
  lines xs = lines_of(&copy, l, y, 2), ys = lines_of(y, l, y, 1);
  ys.real = 1;
  ys.stream = 1;
  PLAN *rp = plan_get(n, RDT, REAL);
  int ok = rp != NULL && F(c2r_lines)(rp, n, &xs, &ys, count, threads);
  if (rp != NULL) plan_put(rp, RDT);
  free(c);
  return ok;
}

#undef SH
#undef UNROLL
#undef DEAL_RE
#undef DEAL_IM
#undef ZIP_LO
#undef ZIP_HI
#undef REV
#undef TRANSPOSE
#undef G8
#undef VEC
#undef CV
#undef PLAN
