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

typedef struct {
  const double *a, *b;
  intnat k;
  int f32;
} terms;

static inline double fma_of(const terms *t, double a, double b, double c) {
  return t->f32 ? fma32(a, b, c) : fma64(a, b, c);
}

static inline double add_of(const terms *t, double x, double y) {
  return t->f32 ? add32(x, y) : x + y;
}

/* Chain order: blocks of 256 terms from 0, each a chain from +0 in
   increasing number, each block's sum added into the sum of those before
   it. */
#define CHAIN_TERMS 256

static double chain(const terms *t) {
  double sum = 0.;
  for (intnat lo = 0; lo < t->k; lo += CHAIN_TERMS) {
    intnat hi = lo + CHAIN_TERMS < t->k ? lo + CHAIN_TERMS : t->k;
    double s = 0.;
    for (intnat i = lo; i < hi; i++) s = fma_of(t, t->a[i], t->b[i], s);
    sum = lo == 0 ? s : add_of(t, sum, s);
  }
  return sum;
}

/* Lane order: block x's terms in 16 lanes by number modulo 16, each from
   +0 in increasing number; lane i takes lane i + 8, then i + 4, i + 2 and
   lane 0 lane 1. */
#define BLOCK_TERMS 1024
#define LANES 16

static double block(const terms *t, intnat x) {
  double l[LANES];
  for (int q = 0; q < LANES; q++) l[q] = 0.;
  intnat lo = x * BLOCK_TERMS;
  intnat hi = lo + BLOCK_TERMS < t->k ? lo + BLOCK_TERMS : t->k;
  for (intnat i = lo; i < hi; i++)
    l[i % LANES] = fma_of(t, t->a[i], t->b[i], l[i % LANES]);
  for (int w = LANES / 2; w >= 1; w /= 2)
    for (int i = 0; i < w; i++) l[i] = add_of(t, l[i], l[i + w]);
  return l[0];
}

/* The blocks from [x] on, [n] of them, by a binary tree whose left part
   holds the largest power of two below [n]. */
static double tree(const terms *t, intnat x, intnat n) {
  if (n == 1) return block(t, x);
  intnat h = 1;
  while (2 * h < n) h *= 2;
  return add_of(t, tree(t, x, h), tree(t, x + h, n - h));
}

/* [flags]: 1 float32, else float64; 2 chain order, else lane order; 4 init
   [init] is added last. With no term, init as it is, or +0. */
double nx_kernels_support_ordered(value va, value vb, intnat flags,
                                  double init) {
  terms t = {(const double *)va, (const double *)vb,
             (intnat)(Wosize_val(va) / Double_wosize), (int)(flags & 1)};
  int has_init = (flags & 4) != 0;
  if (t.k == 0) return has_init ? init : 0.;
  if (flags & 2) {
    double sum = chain(&t);
    return has_init ? add_of(&t, sum, init) : sum;
  }
  double sum = tree(&t, 0, (t.k + BLOCK_TERMS - 1) / BLOCK_TERMS);
  return has_init ? add_of(&t, init, sum) : sum;
}

value nx_kernels_support_ordered_byte(value va, value vb, value vflags,
                                      value vinit) {
  return caml_copy_double(nx_kernels_support_ordered(
      va, vb, Long_val(vflags), Double_val(vinit)));
}
