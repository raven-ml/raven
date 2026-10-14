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
#define REAL_PLAN CAT(real_plan, SFX)

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
#if L == 8
#define DEAL_RE(a, b) SH(a, b, 0, 2, 4, 6, 8, 10, 12, 14)
#define DEAL_IM(a, b) SH(a, b, 1, 3, 5, 7, 9, 11, 13, 15)
#define ZIP_LO(a, b) SH(a, b, 0, 8, 1, 9, 2, 10, 3, 11)
#define ZIP_HI(a, b) SH(a, b, 4, 12, 5, 13, 6, 14, 7, 15)
#define REV(a) SH(a, a, 7, 6, 5, 4, 3, 2, 1, 0)

static inline void F(transpose)(VEC *r) {
  VEC a[8], b[8];
  for (int i = 0; i < 8; i += 2) {
    a[i] = SH(r[i], r[i + 1], 0, 8, 1, 9, 4, 12, 5, 13);
    a[i + 1] = SH(r[i], r[i + 1], 2, 10, 3, 11, 6, 14, 7, 15);
  }
  for (int i = 0; i < 8; i += 4)
    for (int k = 0; k < 2; k++) {
      b[i + 2 * k] = SH(a[i + k], a[i + k + 2], 0, 1, 8, 9, 4, 5, 12, 13);
      b[i + 2 * k + 1] = SH(a[i + k], a[i + k + 2], 2, 3, 10, 11, 6, 7, 14, 15);
    }
  for (int i = 0; i < 4; i++) {
    r[i] = SH(b[i], b[i + 4], 0, 1, 2, 3, 8, 9, 10, 11);
    r[i + 4] = SH(b[i], b[i + 4], 4, 5, 6, 7, 12, 13, 14, 15);
  }
}
#else
#define DEAL_RE(a, b) SH(a, b, 0, 2, 4, 6)
#define DEAL_IM(a, b) SH(a, b, 1, 3, 5, 7)
#define ZIP_LO(a, b) SH(a, b, 0, 4, 1, 5)
#define ZIP_HI(a, b) SH(a, b, 2, 6, 3, 7)
#define REV(a) SH(a, a, 3, 2, 1, 0)

static inline void F(transpose)(VEC *r) {
  VEC a[4];
  for (int i = 0; i < 4; i += 2) {
    a[i] = SH(r[i], r[i + 1], 0, 4, 2, 6);
    a[i + 1] = SH(r[i], r[i + 1], 1, 5, 3, 7);
  }
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
} PLAN;

/* A real line of n points: its half-length plan and ω^k for k < n / 2
   where n is halved, else its complex plan of n points. */
typedef struct {
  int64_t n, bytes;
  PLAN *complex;
  T *w;
} REAL_PLAN;

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
   lanes spill registers. */
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

/* The four steps' plans and twiddles: n1 the largest divisor of n up to
   its square root of the fewest passes of n1 and n / n1 (1024 points as
   16 × 64 took 11% less time than as 32 × 32 across 64 lines). ω^(k1 j2)
   is the product, in double, of unit's values at its exponent's low and
   high halves. */
static int F(four_make)(PLAN *p) {
  int64_t n = p->n, n1 = 1;
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

/* The plan of a line of n points, [kind] PASSES or ANY: for ANY, by
   Bluestein where by_bluestein takes it and its kernel is small enough,
   else by passes up to DIRECT_POINTS or for a prime n, and by the four
   steps above. NULL if memory runs out. */
static PLAN *F(plan_make)(int64_t n, int kind) {
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

static void F(real_free)(REAL_PLAN *r) {
  if (r == NULL) return;
  F(plan_free)(r->complex);
  free(r->w);
  free(r);
}

static REAL_PLAN *F(real_make)(int64_t n) {
  REAL_PLAN *r = calloc(1, sizeof(REAL_PLAN));
  if (r == NULL) return NULL;
  int64_t h = halved(n) ? n / 2 : 0;
  r->n = n;
  r->complex = F(plan_make)(halved(n) ? h : n, ANY);
  r->w = malloc(sizeof(T) * 2 * (h > 0 ? h : 1));
  if (r->complex == NULL || r->w == NULL) {
    F(real_free)(r);
    return NULL;
  }
  r->bytes = sizeof(REAL_PLAN) + r->complex->bytes + sizeof(T) * 2 * h;
  F(units)(r->w, h, 1, n);
  return r;
}

/* Passes, after pocketfft's forward ones: pass k of radix ip takes l1
   transforms of ido · ip points to ip l1 of ido. CC(i, u, k) is input
   point i + ido (u + ip k); CH(i, k, u) output point i + ido (k + l1 u).
   Each lane is a line of its own. Radix 3 and 5 keep passes of their
   own: passg inlined with a constant ip took 7 to 13% longer on lines of
   1000, 2187, 15625 and 59049 points (M1, one thread). */

#define CC(i, u, k) cc[(i) + ido * ((u) + ip * (k))]
#define CH(i, k, u) ch[(i) + ido * ((k) + l1 * (u))]
#define TW(x, i) tw[2 * ((x) * ido + (i))], tw[2 * ((x) * ido + (i)) + 1]

static void F(pass2)(int64_t ido, int64_t l1, const CV *restrict cc,
                     CV *restrict ch, const T *tw) {
  const int ip = 2;
  for (int64_t k = 0; k < l1; k++)
    for (int64_t i = 0; i < ido; i++) {
      CV a = CC(i, 0, k), b = CC(i, 1, k);
      CH(i, k, 0) = F(add)(a, b);
      CH(i, k, 1) = i == 0 ? F(sub)(a, b) : F(mul)(F(sub)(a, b), TW(0, i));
    }
}

static void F(pass3)(int64_t ido, int64_t l1, const CV *restrict cc,
                     CV *restrict ch, const T *tw) {
  const int ip = 3;
  const T c = -0.5, s = -0.86602540378443864676;
  for (int64_t k = 0; k < l1; k++)
    for (int64_t i = 0; i < ido; i++) {
      CV t0 = CC(i, 0, k), t1 = F(add)(CC(i, 1, k), CC(i, 2, k)),
         t2 = F(sub)(CC(i, 1, k), CC(i, 2, k));
      CV a = F(add)(t0, F(scale)(t1, c)), b = F(scale)(t2, s);
      CH(i, k, 0) = F(add)(t0, t1);
      if (i == 0) {
        CH(i, k, 1) = F(plus_i)(a, b);
        CH(i, k, 2) = F(minus_i)(a, b);
      } else {
        CH(i, k, 1) = F(mul)(F(plus_i)(a, b), TW(0, i));
        CH(i, k, 2) = F(mul)(F(minus_i)(a, b), TW(1, i));
      }
    }
}

static void F(pass4)(int64_t ido, int64_t l1, const CV *restrict cc,
                     CV *restrict ch, const T *tw) {
  const int ip = 4;
  for (int64_t k = 0; k < l1; k++)
    for (int64_t i = 0; i < ido; i++) {
      CV c0 = CC(i, 0, k), c1 = CC(i, 1, k), c2 = CC(i, 2, k), c3 = CC(i, 3, k);
      CV t1 = F(sub)(c0, c2), t2 = F(add)(c0, c2), t3 = F(add)(c1, c3),
         t4 = F(sub)(c1, c3);
      CH(i, k, 0) = F(add)(t2, t3);
      if (i == 0) {
        CH(i, k, 1) = F(minus_i)(t1, t4);
        CH(i, k, 2) = F(sub)(t2, t3);
        CH(i, k, 3) = F(plus_i)(t1, t4);
      } else {
        CH(i, k, 1) = F(mul)(F(minus_i)(t1, t4), TW(0, i));
        CH(i, k, 2) = F(mul)(F(sub)(t2, t3), TW(1, i));
        CH(i, k, 3) = F(mul)(F(plus_i)(t1, t4), TW(2, i));
      }
    }
}

static void F(pass5)(int64_t ido, int64_t l1, const CV *restrict cc,
                     CV *restrict ch, const T *tw) {
  const int ip = 5;
  const T c1 = 0.3090169943749474241, s1 = -0.95105651629515357212,
          c2 = -0.8090169943749474241, s2 = -0.58778525229247312917;
  for (int64_t k = 0; k < l1; k++)
    for (int64_t i = 0; i < ido; i++) {
      CV t0 = CC(i, 0, k), t1 = F(add)(CC(i, 1, k), CC(i, 4, k)),
         t4 = F(sub)(CC(i, 1, k), CC(i, 4, k)),
         t2 = F(add)(CC(i, 2, k), CC(i, 3, k)),
         t3 = F(sub)(CC(i, 2, k), CC(i, 3, k));
      CV a1 = F(add)(F(add)(t0, F(scale)(t1, c1)), F(scale)(t2, c2)),
         b1 = F(add)(F(scale)(t4, s1), F(scale)(t3, s2)),
         a2 = F(add)(F(add)(t0, F(scale)(t1, c2)), F(scale)(t2, c1)),
         b2 = F(sub)(F(scale)(t4, s2), F(scale)(t3, s1));
      CV y[5] = {F(add)(F(add)(t0, t1), t2), F(plus_i)(a1, b1),
                 F(plus_i)(a2, b2), F(minus_i)(a2, b2), F(minus_i)(a1, b1)};
      CH(i, k, 0) = y[0];
      for (int u = 1; u < 5; u++)
        CH(i, k, u) = i == 0 ? y[u] : F(mul)(y[u], TW(u - 1, i));
    }
}

/* An odd radix: output u of the pair sums t+ = c_j + c_(ip-j) and
   differences t- is a + i b and output ip - u is a - i b, with
   a = c_0 + Σ Re ω^(uj) t+ and b = Σ Im ω^(uj) t-. */
static void F(passg)(int ip, int64_t ido, int64_t l1, const CV *restrict cc,
                     CV *restrict ch, const T *tw, const T *cs) {
  int h = (ip + 1) / 2;
  for (int64_t k = 0; k < l1; k++)
    for (int64_t i = 0; i < ido; i++) {
      CV c0 = CC(i, 0, k), y0 = c0;
      for (int j = 1; j < h; j++)
        y0 = F(add)(y0, F(add)(CC(i, j, k), CC(i, ip - j, k)));
      CH(i, k, 0) = y0;
      for (int u = 1; u < h; u++) {
        CV a = c0, b = {0};
        int m = 0;
        for (int j = 1; j < h; j++) {
          m += u;
          if (m >= ip) m -= ip;
          CV p = CC(i, j, k), q = CC(i, ip - j, k);
          a = F(add)(a, F(scale)(F(add)(p, q), cs[2 * m]));
          b = F(add)(b, F(scale)(F(sub)(p, q), cs[2 * m + 1]));
        }
        CV yu = F(plus_i)(a, b), yv = F(minus_i)(a, b);
        if (i > 0) {
          yu = F(mul)(yu, TW(u - 1, i));
          yv = F(mul)(yv, TW(ip - u - 1, i));
        }
        CH(i, k, u) = yu;
        CH(i, k, ip - u) = yv;
      }
    }
}

#undef CC
#undef CH
#undef TW

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

#if NT_STORES
/* scatter's lanes side by side, points [st] reals apart from [d], by
   x86's non-temporal stores (stream_of): [d] on 32 bytes. */
static void F(stream)(T *d, int64_t st, int64_t n, int conj,
                      const CV *restrict s) {
  for (int64_t j = 0; j < n; j++) {
    VEC re = s[j].re, im = conj ? -s[j].im : s[j].im;
    VEC lo = ZIP_LO(re, im), hi = ZIP_HI(re, im);
#if L == 8
    _mm256_stream_ps(d + j * st, (__m256)lo);
    _mm256_stream_ps(d + j * st + L, (__m256)hi);
#else
    _mm256_stream_pd(d + j * st, (__m256d)lo);
    _mm256_stream_pd(d + j * st + L, (__m256d)hi);
#endif
  }
  _mm_sfence();
}
#endif

/* Writes lanes [0, lanes) of [s], points [0, n) of lines [o] of y, [st]
   reals apart: lanes side by side by stores past the caches where y
   streams and its points lie apart. */
static void F(scatter)(const lines *y, const int64_t *o, int lanes,
                       int64_t n, int64_t st, const CV *restrict s) {
  T *b = (T *)y->base;
  int64_t io = y->io;
  int conj = y->conj;
  if (lanes == L && side_by_side(y, o, L)) {
#if NT_STORES
    T *d = b + o[0];
    if (y->stream && st > 2 * L && st % L == 0 && (uintptr_t)d % 32 == 0) {
      F(stream)(d, st, n, conj, s);
      return;
    }
#endif
    for (int64_t j = 0; j < n; j++) {
      VEC re = s[j].re, im = conj ? -s[j].im : s[j].im;
      VEC lo = ZIP_LO(re, im), hi = ZIP_HI(re, im);
      memcpy(b + o[0] + j * st, &lo, sizeof lo);
      memcpy(b + o[0] + j * st + L, &hi, sizeof hi);
    }
    return;
  }
  int64_t j = 0;
  if (lanes == L && st == 2 && io == 1 && !y->real)
    for (; j + L <= n; j += L) {
      VEC re[L], im[L];
      for (int u = 0; u < L; u++) {
        re[u] = s[j + u].re;
        im[u] = conj ? -s[j + u].im : s[j + u].im;
      }
      TRANSPOSE(re);
      TRANSPOSE(im);
      for (int v = 0; v < L; v++) {
        VEC lo = ZIP_LO(re[v], im[v]), hi = ZIP_HI(re[v], im[v]);
        memcpy(b + o[v] + 2 * j, &lo, sizeof lo);
        memcpy(b + o[v] + 2 * j + L, &hi, sizeof hi);
      }
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
   Within a line, the lanes are L columns, then L rows: between, the
   columns' L × L tiles transpose into s, block h holding rows h L to
   h L + L - 1. Across lines, the lanes are L lines and s holds them
   row after row. Each line's arithmetic is the same either way. */

/* The points of the lanes buffers a and b of the steps of a line. */
static int64_t F(four_lanes)(const PLAN *p) {
  int64_t big = p->n1 > p->n2 ? p->n1 : p->n2;
  return (big + L - 1) / L * L;
}

/* Step one of line [l]: column group [g] into s, through [a] and [b]. */
static void F(column)(const PLAN *p, const lines *x, int64_t l, int64_t g,
                      CV *s, CV *a, CV *b) {
  int64_t n1 = p->n1, n2 = p->n2, rows = (n1 + L - 1) / L * L;
  int64_t width = (n2 + L - 1) / L * L;
  int lanes = n2 - g * L < L ? (int)(n2 - g * L) : L;
  int64_t o[L], at = line_at(x, l);
  for (int v = 0; v < L; v++)
    o[v] = at + (g * L + (v < lanes ? v : 0)) * x->step;
  F(gather)(a, x, o, lanes, n1, n2 * x->step);
  CV *r = F(run_passes)(p->p1, a, b);
  const CV *tw = p->tw4 + g * n1;
  for (int64_t h = 0; h < rows; h += L) {
    VEC re[L], im[L];
    for (int u = 0; u < L; u++) {
      CV z = h + u < n1 ? F(mulv)(r[h + u], tw[h + u]) : (CV){0};
      re[u] = z.re;
      im[u] = z.im;
    }
    TRANSPOSE(re);
    TRANSPOSE(im);
    CV *t = s + h / L * width + g * L;
    for (int u = 0; u < L; u++) t[u] = (CV){re[u], im[u]};
  }
}

/* Step two of line [l]: row block [h] of s into y, through [b]. */
static void F(row)(const PLAN *p, const lines *y, int64_t l, int64_t h,
                   CV *s, CV *b) {
  int64_t n1 = p->n1, n2 = p->n2, width = (n2 + L - 1) / L * L;
  int lanes = n1 - h * L < L ? (int)(n1 - h * L) : L;
  int64_t q[L], at = line_at(y, l);
  for (int v = 0; v < L; v++)
    q[v] = at + (h * L + (v < lanes ? v : 0)) * y->step;
  F(scatter)(y, q, lanes, n2, n1 * y->step,
             F(run_passes)(p->p2, s + h * width, b));
}

/* Line [l] by the four steps, on this thread. */
static int F(within)(const PLAN *p, const lines *x, const lines *y,
                     int64_t l) {
  int64_t width = (p->n2 + L - 1) / L * L, blocks = (p->n1 + L - 1) / L;
  int64_t lanes = F(four_lanes)(p);
  CV *s = nx_cpu_take(sizeof(CV) * blocks * width),
     *a = nx_cpu_take(sizeof(CV) * lanes), *b = nx_cpu_take(sizeof(CV) * lanes);
  int ok = s != NULL && a != NULL && b != NULL;
  for (int64_t g = 0; ok && g < width / L; g++) F(column)(p, x, l, g, s, a, b);
  for (int64_t h = 0; ok && h < blocks; h++) F(row)(p, y, l, h, s, b);
  nx_cpu_give(s);
  nx_cpu_give(a);
  nx_cpu_give(b);
  return ok;
}

/* Lines [l, l + L), side by side, by the four steps. */
static int F(across)(const PLAN *p, const lines *x, const lines *y,
                     int64_t l) {
  int64_t n1 = p->n1, n2 = p->n2, big = n1 > n2 ? n1 : n2;
  CV *a = nx_cpu_take(sizeof(CV) * big), *b = nx_cpu_take(sizeof(CV) * big),
     *s = nx_cpu_take(sizeof(CV) * p->n);
  if (a == NULL || b == NULL || s == NULL) {
    nx_cpu_give(a);
    nx_cpu_give(b);
    nx_cpu_give(s);
    return 0;
  }
  int64_t o[L], q[L];
  for (int v = 0; v < L; v++) {
    o[v] = line_at(x, l + v);
    q[v] = line_at(y, l + v);
  }
  for (int64_t j2 = 0; j2 < n2; j2++) {
    int64_t c[L];
    for (int v = 0; v < L; v++) c[v] = o[v] + j2 * x->step;
    F(gather)(a, x, c, L, n1, n2 * x->step);
    CV *r = F(run_passes)(p->p1, a, b);
    const CV *tw = p->tw4 + j2 / L * n1;
    for (int64_t k1 = 0; k1 < n1; k1++) {
      CV t;
      for (int v = 0; v < L; v++) {
        t.re[v] = tw[k1].re[j2 % L];
        t.im[v] = tw[k1].im[j2 % L];
      }
      s[k1 * n2 + j2] = F(mulv)(r[k1], t);
    }
  }
  for (int64_t k1 = 0; k1 < n1; k1++) {
    int64_t c[L];
    for (int v = 0; v < L; v++) c[v] = q[v] + k1 * y->step;
    F(scatter)(y, c, L, n2, n1 * y->step, F(run_passes)(p->p2, s + k1 * n2, b));
  }
  nx_cpu_give(a);
  nx_cpu_give(b);
  nx_cpu_give(s);
  return 1;
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
      for (int64_t l = lo; l < hi;)
        if (hi - l >= L && lines_side_by_side(x, l, L) &&
            lines_side_by_side(y, l, L)) {
          if (!F(across)(p, x, y, l)) return 0;
          l += L;
        } else {
          if (!F(within)(p, x, y, l)) return 0;
          l++;
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
   bins k and h - k computed together. */

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
static void F(split)(const REAL_PLAN *rp, const lines *y, int64_t o,
                     int64_t lo, int64_t hi) {
  int64_t h = rp->n / 2;
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
static void F(merge)(const REAL_PLAN *rp, const lines *x, int64_t o, T *z,
                     int64_t lo, int64_t hi) {
  int64_t h = rp->n / 2;
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
static void F(complete)(const REAL_PLAN *rp, const lines *x, int64_t o, T *z,
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

/* A job of one step over its units: lines, a line's column groups or row
   blocks, or chunks of a line's or of every line's points. */
typedef struct {
  int step;
  const PLAN *p;
  const REAL_PLAN *rp;
  const lines *x, *y;
  int64_t count; /* the lines */
  int64_t per;   /* lines a unit, or chunks a line */
  int64_t l;     /* the line whose steps the job runs */
  CV *s;         /* that line's four-step grid */
  T *r;          /* its Bluestein row, or every line's complex row */
  int failed;
} F(job);

static void F(body)(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  F(job) *j = ctx;
  const PLAN *p = j->p;
  const REAL_PLAN *rp = j->rp;
  int ok = 1;
  int64_t n = rp ? rp->n : p->n, h = n / 2;
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
        if (j->step == COLUMNS) F(column)(p, j->x, j->l, u, j->s, a, b);
        else F(row)(p, j->y, j->l, u, j->s, b);
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
            int64_t l = chunk(u, j->per, h / 2 + 1, &a, &b);
            F(split)(rp, j->y, line_at(j->y, l), a, b);
            break;
          }
          case MERGE: {
            int64_t l = chunk(u, j->per, h / 2 + 1, &a, &b);
            F(merge)(rp, j->x, line_at(j->x, l), j->r + 2 * h * l, a, b);
            break;
          }
          case COMPLETE: {
            int64_t l = chunk(u, j->per, n, &a, &b);
            F(complete)(rp, j->x, line_at(j->x, l), j->r + 2 * n * l, a, b);
            break;
          }
          case HALVE: {
            int64_t l = chunk(u, j->per, h + 1, &a, &b);
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

/* Line j->l's four steps, each a job: its column groups into the shared
   [s], then its row blocks. */
static int F(four_steps)(F(job) * j, int64_t bytes, double cost,
                         int threads) {
  const PLAN *p = j->p;
  j->step = COLUMNS;
  if (!F(run)(j, (p->n2 + L - 1) / L, bytes, cost, threads)) return 0;
  j->step = ROWS;
  return F(run)(j, (p->n1 + L - 1) / L, bytes, cost, threads);
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
   which share a buffer the call allocates. */
static int F(transform)(const PLAN *p, const lines *x, const lines *y,
                        int64_t count, int threads) {
  int64_t n = p->n, bytes = 4 * (int64_t)sizeof(T) * n * count;
  double cost = cost_of(bytes, n, p->kind == BLUESTEIN ? p->m : 0);
  /* A result streams from one thread alone (stream_of). */
  lines cached;
  if (y->stream && threads != 1 && nx_cpu_threads(bytes, cost) != 1) {
    cached = *y;
    cached.stream = 0;
    y = &cached;
  }
  F(job) j = {.step = LINES, .p = p, .x = x, .y = y, .count = count};
  j.per = p->kind == FOUR ? 1 : L;
  int64_t units = (count + j.per - 1) / j.per;
  if (threads == 1 || p->kind == PASSES ||
      units >= nx_cpu_threads(bytes, cost))
    return F(run)(&j, units, bytes, cost, threads);
  int four = p->kind == FOUR;
  int64_t shared = four ? (int64_t)sizeof(CV) * (p->n1 + L - 1) / L * L *
                              ((p->n2 + L - 1) / L * L)
                        : (int64_t)sizeof(T) * 2 * p->m;
  void *buffer = nx_cpu_take(shared);
  if (buffer == NULL) return 0;
  j.s = buffer;
  j.r = buffer;
  int ok = 1;
  for (j.l = 0; j.l < count && ok; j.l++)
    ok = four ? F(four_steps)(&j, bytes / count, cost / count, threads)
              : F(blue_steps)(&j, bytes / count, cost / count, threads);
  nx_cpu_give(buffer);
  return ok;
}

/* [count] real lines of x into y's bins, as r2c states. */
static int F(r2c_lines)(const REAL_PLAN *rp, const lines *x, const lines *y,
                        int64_t count, int threads) {
  int64_t n = rp->n, h = n / 2, bytes = 3 * (int64_t)sizeof(T) * n * count;
  F(job) j = {.rp = rp, .x = x, .y = y, .count = count};
  if (halved(n)) {
    lines pairs = *x;
    pairs.step = 2 * x->step;
    pairs.io = x->step;
    pairs.real = 0;
    if (!F(transform)(rp->complex, &pairs, y, count, threads)) return 0;
    j.step = SPLIT;
    j.per = (h / 2 + 1 + CHUNK - 1) / CHUNK;
    return F(run)(&j, count * j.per, bytes, HALF_COST * bytes, threads);
  }
  T *r = malloc(sizeof(T) * 2 * n * count);
  if (r == NULL) return 0;
  lines rows = F(rows)(r, count, n);
  int ok = F(transform)(rp->complex, x, &rows, count, threads);
  j.step = HALVE;
  j.r = r;
  j.per = (h + 1 + CHUNK - 1) / CHUNK;
  ok = ok && F(run)(&j, count * j.per, bytes, bytes, threads);
  free(r);
  return ok;
}

/* [count] lines of x's bins into y's real lines, as c2r states. */
static int F(c2r_lines)(const REAL_PLAN *rp, const lines *x, const lines *y,
                        int64_t count, int threads) {
  int64_t n = rp->n, h = n / 2, bytes = 3 * (int64_t)sizeof(T) * n * count;
  int half = halved(n);
  int64_t points = half ? h : n;
  T *r = malloc(sizeof(T) * 2 * (points > 0 ? points : 1) * count);
  if (r == NULL) return 0;
  F(job) j = {.rp = rp, .x = x, .y = y, .count = count, .r = r};
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
  ok = ok && F(transform)(rp->complex, &rows, &out, count, threads);
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
    int order[NX_MAX_RANK];
    others(y, a, order);
    lines xs = lines_of(i == 0 ? x : y, a, order, 2),
          ys = lines_of(y, a, order, 2);
    xs.conj = ys.conj = inverse;
    ys.stream = stream_of(y);
    PLAN *p = plan_get(n, RDT, COMPLEX);
    if (p == NULL) return 0;
    int ok = F(transform)(p, &xs, &ys, count, threads);
    plan_put(p, RDT, COMPLEX);
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
  int order[NX_MAX_RANK];
  others(y, l, order);
  lines xs = lines_of(x, l, order, 1), ys = lines_of(y, l, order, 2);
  xs.real = 1;
  ys.stream = stream_of(y);
  REAL_PLAN *rp = plan_get(n, RDT, REAL);
  if (rp == NULL) return 0;
  int ok = F(r2c_lines)(rp, &xs, &ys, count, threads);
  plan_put(rp, RDT, REAL);
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
  int order[NX_MAX_RANK];
  others(y, l, order);
  lines xs = lines_of(&copy, l, order, 2), ys = lines_of(y, l, order, 1);
  ys.real = 1;
  ys.stream = stream_of(y);
  REAL_PLAN *rp = plan_get(n, RDT, REAL);
  int ok = rp != NULL && F(c2r_lines)(rp, &xs, &ys, count, threads);
  if (rp != NULL) plan_put(rp, RDT, REAL);
  free(c);
  return ok;
}

#undef SH
#undef DEAL_RE
#undef DEAL_IM
#undef ZIP_LO
#undef ZIP_HI
#undef REV
#undef TRANSPOSE
#undef VEC
#undef CV
#undef PLAN
#undef REAL_PLAN
