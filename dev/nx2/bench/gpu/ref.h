/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The host reference of the GPU suites and benches: what a GPU result
   must be, from the operands' memory on the host. Every backend checks
   against it, so a bound is stated once. */

#ifndef NX_REF_H
#define NX_REF_H

#include <stdint.h>

/* An operand of three axes: element (z, r, c) of the dtype [dtype]
   (nx_dtype.h's code) at [base] plus z, r and c times [s], in elements. */
typedef struct {
  const void *base;
  int dtype;
  int64_t s[3];
} nx_ref_view;

/* A check's outcome. [wrong] outputs differ from an answer that is exact:
   an integer accumulator's sum, or the NaN or infinity IEEE arithmetic
   gives a float sum with a NaN or an infinity among its terms; and a NaN
   where the float sum is finite, which has no error to bound. Of the other
   float outputs, [worst] is the largest ratio of an output's error to its
   allowance, and an output is within its bound iff its ratio is at most 1.
   [at] is the first wrong output, else the worst one, as z·m·n + i·n + j,
   or -1 if every output is exact. */
typedef struct {
  double worst;
  int64_t wrong, at;
} nx_ref_result;

/* Checks [samples] outputs (z, i, j) of y = init + a·b over [k] (every
   output if there are fewer, else outputs drawn by a fixed hash of their
   count), y and init [batch] × [m] × [n], a [batch] × [m] × [k], b [batch]
   × [n] × [k]. [init] may be NULL.

   For the float accumulator [acc], the allowance is the bound γ(k + 1, 2u)
   (|init| + Σ|a||b|), u [acc]'s unit roundoff, plus half a unit in the
   last place of y's dtype, against the exact sum. [flush] states that the
   device's arithmetic reads a subnormal operand as zero and writes a
   subnormal result as zero: the allowance gains 2^-126·(1 + Σ(1 + |a| +
   |b|)). A NaN or an infinity among the terms asks for IEEE's answer (any
   NaN for NaN), and an exact sum that rounds past y's range its infinity.

   An integer or bool y of a float [acc] lies between the casts of the
   ends of the sums the bound allows. For an integer [acc], y holds the sum
   wrapped to [acc], then cast from [acc] to y's dtype, as nx casts:
   widened by [acc]'s sign and wrapped, rounded once to a float, or
   nonzero for bool. */
nx_ref_result nx_ref_contract(const nx_ref_view *a, const nx_ref_view *b,
                              const nx_ref_view *init, const nx_ref_view *y,
                              int64_t batch, int64_t m, int64_t n, int64_t k,
                              int acc, int flush, int64_t samples);

#endif
