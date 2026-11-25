/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_cpu.mli's orders of a contraction's sum, written out as its Lane order
   and Order sections state them: a reference for nx.cpu's bits, sharing no
   code with its kernels. Built with -ffp-contract=off, so that no product
   fuses into a sum the text keeps apart, and no fast-math. */

#include <stdint.h>
#include <string.h>

#include <caml/alloc.h>
#include <caml/mlvalues.h>

/* float64's fused multiply-add is the instruction's on every path. The
   instruction computes a b + c exactly and rounds it once (IEEE 754-2019
   §5.4.1, fusedMultiplyAdd), in every range, subnormal results included, so
   its bits are those of an exact product and sum rounded once: Exact.fma's.
   A C library's fma is no such reference, as mingw-w64's rounds twice at
   subnormal results; a target without the instruction is refused, by the test
   of nx_kinds.h, which nx.cpu's kinds need for the same reason. Its
   FP_FAST_FMA clause is not here: that macro comes from <math.h>, which this
   file does not include, and the targets it names glibc's POWER for are named
   below. */
#if !defined(__FMA__) && !defined(__ARM_FEATURE_FMA) &&                       \
    !(defined(__FP_FAST_FMA) && defined(__FP_FAST_FMAF)) &&                   \
    !defined(__powerpc64__) && !defined(__s390x__) &&                         \
    !(defined(__riscv_flen) && __riscv_flen >= 64) &&                         \
    !defined(__loongarch_double_float)
#error "the order's reference needs a fused multiply-add instruction"
#endif

static inline uint64_t bits_of(double x) {
  uint64_t b;
  memcpy(&b, &x, sizeof b);
  return b;
}

static inline double of_bits(uint64_t b) {
  double x;
  memcpy(&x, &b, sizeof x);
  return x;
}

static inline double round32(double x) { return (double)(float)x; }

/* float32's fused multiply-add: a b, exact in a double for float32 a and b,
   plus c, rounded to odd in a double, then to float32, which rounds once
   since 53 >= 2 24 + 2. The sum's error is TwoSum's. */
static inline double fma32(double a, double b, double c) {
  double p = a * b;
  double s = c + p;
  double z = s - c;
  double e = (c - (s - z)) + (p - z);
  uint64_t u = bits_of(s);
  if (e == 0. || !(s - s == 0.) || (u & 1)) return round32(s);
  /* s is not zero where e is not: a sum that rounds to zero is exact. One
     step toward e, away from zero where e has s's sign. */
  int away = (e > 0.) == (s > 0.);
  return round32(of_bits(away ? u + 1 : u - 1));
}

static inline double fma64(double a, double b, double c) {
  return __builtin_fma(a, b, c);
}

static inline double add32(double x, double y) { return round32(x + y); }

/* fma and addition of float32 where [f32], else of float64: each caller
   passes a constant, so that each order compiles once for each. */
static inline double fma_of(int f32, double a, double b, double c) {
  return f32 ? fma32(a, b, c) : fma64(a, b, c);
}

static inline double add_of(int f32, double x, double y) {
  return f32 ? add32(x, y) : x + y;
}

/* Chain order: blocks of 256 terms from 0, each a chain from +0 in
   increasing number, each block's sum added into the sum of those before
   it. A chain waits on each of its fused multiply-adds, so up to CHAINS
   outputs of one depth chain side by side, each in its own order. */
#define CHAIN_TERMS 256
#define CHAINS 8

static inline void chains(int f32, const double *const *a,
                          const double *const *b, int n, intnat k,
                          double *sum) {
  double s[CHAINS];
  for (int g = 0; g < n; g++) sum[g] = 0.;
  for (intnat lo = 0; lo < k; lo += CHAIN_TERMS) {
    intnat hi = lo + CHAIN_TERMS < k ? lo + CHAIN_TERMS : k;
    for (int g = 0; g < n; g++) s[g] = 0.;
    for (intnat i = lo; i < hi; i++)
      for (int g = 0; g < n; g++) s[g] = fma_of(f32, a[g][i], b[g][i], s[g]);
    for (int g = 0; g < n; g++)
      sum[g] = lo == 0 ? s[g] : add_of(f32, sum[g], s[g]);
  }
}

/* Lane order: block x's terms in 16 lanes by number modulo 16, each from
   +0 in increasing number; lane i takes lane i + 8, then i + 4, i + 2 and
   lane 0 lane 1. */
#define BLOCK_TERMS 1024
#define LANES 16

static inline double block(int f32, const double *a, const double *b,
                           intnat k, intnat x) {
  double l[LANES];
  for (int q = 0; q < LANES; q++) l[q] = 0.;
  intnat lo = x * BLOCK_TERMS;
  intnat hi = lo + BLOCK_TERMS < k ? lo + BLOCK_TERMS : k;
  for (intnat i = lo; i < hi; i++)
    l[i % LANES] = fma_of(f32, a[i], b[i], l[i % LANES]);
  for (int w = LANES / 2; w >= 1; w /= 2)
    for (int i = 0; i < w; i++) l[i] = add_of(f32, l[i], l[i + w]);
  return l[0];
}

/* The blocks from [x] on, [n] of them, by a binary tree whose left part
   holds the largest power of two below [n]. */
static double tree(int f32, const double *a, const double *b, intnat k,
                   intnat x, intnat n) {
  if (n == 1) return block(f32, a, b, k, x);
  intnat h = 1;
  while (2 * h < n) h *= 2;
  return add_of(f32, tree(f32, a, b, k, x, h),
                tree(f32, a, b, k, x + h, n - h));
}

static intnat length(value v) { return (intnat)(Wosize_val(v) / Double_wosize); }

/* Output i's sum of [va]'s and [vb]'s i-th float arrays' products into
   [vout]'s element i, then [vinit]'s added where it has an element i.
   [vflags]: 1 float32, else float64; 2 chain order, else lane order. With
   no term, init as it is, or +0. */
static void ordered(int f32, int chain_order, value va, value vb,
                    value vinit, value vout) {
  intnat n = (intnat)Wosize_val(va);
  int has_init = length(vinit) > 0;
  for (intnat o = 0; o < n;) {
    const double *a[CHAINS], *b[CHAINS];
    double sum[CHAINS];
    intnat k = length(Field(va, o));
    int g = 0;
    /* Outputs side by side share their depth. */
    while (g < CHAINS && o + g < n && length(Field(va, o + g)) == k) {
      a[g] = (const double *)Field(va, o + g);
      b[g] = (const double *)Field(vb, o + g);
      g++;
    }
    if (chain_order)
      chains(f32, a, b, g, k, sum);
    else
      for (int j = 0; j < g; j++)
        sum[j] = k == 0 ? 0.
                        : tree(f32, a[j], b[j], k, 0,
                               (k + BLOCK_TERMS - 1) / BLOCK_TERMS);
    for (int j = 0; j < g; j++) {
      double y = sum[j];
      if (has_init) {
        double i = Double_flat_field(vinit, o + j);
        y = k == 0 ? i : chain_order ? add_of(f32, y, i) : add_of(f32, i, y);
      }
      Store_double_flat_field(vout, o + j, y);
    }
    o += g;
  }
}

value nx_kernels_support_ordered(value va, value vb, value vinit,
                                 value vflags, value vout) {
  intnat flags = Long_val(vflags);
  int chain_order = (flags & 2) != 0;
  if (flags & 1)
    ordered(1, chain_order, va, vb, vinit, vout);
  else
    ordered(0, chain_order, va, vb, vinit, vout);
  return Val_unit;
}
