/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_random.c — counter-based PRNG: threefry2x32-20.

   Threefry maps a (key, counter) pair of two uint32 words to two output words.
   The frontend lays key/counter/output out as int32 tensors whose last axis has
   extent 2 (the 2-word vector); every position before that axis is an
   independent vector. So this is a thin custom driver, not a map ride: it
   iterates the prefix nest (all axes but the last) and processes each vector
   through the pair kernel, reading the two words along the last-axis stride so
   any layout works. The vectors are independent, so it parallelizes freely over
   them (COMPUTE class — 20 rounds per vector is arithmetic-bound), race-free by
   construction (disjoint output vectors).

   The kernel is the verified threefry2x32-20 (Random123 constants: the
   0x1BD11BDA key-schedule parity word, the eight rotation constants, one key
   injection every four rounds). It computes in uint32 (modular, matching the
   Int32-wrapping reference); the int32 storage is a pure reinterpretation.

   Like every kernel-family file, this TU includes neither caml/fail.h nor
   caml/threads.h: it raises and hands off the runtime lock only through the
   engine (nx_c_raise, nx_c_parallel_run). */

#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_c_engine.h"

#define NX_C_ERR_THREEFRY_SHAPE "threefry: last axis must have extent 2"

static inline uint32_t nx_c_rotl32(uint32_t x, int r) {
  return (uint32_t)((x << r) | (x >> (32 - r)));
}

/* threefry2x32-20: 20 rounds, one rotation per round from the 8-constant
   schedule, a key injection after every 4th round. ks[2] is the Threefry parity
   word. Verified against the Random123 known-answer vectors (see the
   conformance KAT cases). */
static inline void nx_c_threefry2x32(uint32_t k0, uint32_t k1, uint32_t c0,
                                    uint32_t c1, uint32_t *o0, uint32_t *o1) {
  static const int rots[8] = {13, 15, 26, 6, 17, 29, 16, 24};
  uint32_t ks[3] = {k0, k1, (uint32_t)0x1BD11BDA ^ k0 ^ k1};
  uint32_t x0 = c0 + k0, x1 = c1 + k1;
  for (int r = 0; r < 20; r++) {
    x0 += x1;
    x1 = nx_c_rotl32(x1, rots[r % 8]);
    x1 ^= x0;
    if ((r + 1) % 4 == 0) {
      int s = (r + 1) / 4;
      x0 += ks[s % 3];
      x1 += ks[(s + 1) % 3] + (uint32_t)s;
    }
  }
  *o0 = x0;
  *o1 = x1;
}

typedef struct {
  const nx_c_ndarray *key;
  const nx_c_ndarray *ctr;
  const nx_c_ndarray *out;
  int prefix_ndim; /* axes before the size-2 vector axis */
  int64_t key_last, ctr_last, out_last; /* element stride of the vector axis */
} nx_c_threefry_ctx;

static void nx_c_threefry_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  (void)worker;
  const nx_c_threefry_ctx *t = vctx;
  const nx_c_ndarray *key = t->key, *ctr = t->ctr, *out = t->out;
  int pn = t->prefix_ndim;
  const int32_t *kd = key->data, *cd = ctr->data;
  int32_t *od = out->data;
  for (int64_t it = lo; it < hi; it++) {
    int64_t rem = it, kb = key->offset, cb = ctr->offset, ob = out->offset;
    for (int d = pn - 1; d >= 0; d--) {
      int64_t c = rem % key->shape[d];
      rem /= key->shape[d];
      kb += c * key->strides[d];
      cb += c * ctr->strides[d];
      ob += c * out->strides[d];
    }
    uint32_t o0, o1;
    nx_c_threefry2x32((uint32_t)kd[kb], (uint32_t)kd[kb + t->key_last],
                     (uint32_t)cd[cb], (uint32_t)cd[cb + t->ctr_last], &o0, &o1);
    od[ob] = (int32_t)o0;
    od[ob + t->out_last] = (int32_t)o1;
  }
}

CAMLprim value caml_nx_c_threefry(value vout, value vkey, value vctr) {
  CAMLparam3(vout, vkey, vctr);
  nx_c_ndarray out, key, ctr;
  nx_c_status s = nx_c_ndarray_of_value(vout, &out);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vkey, &key);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vctr, &ctr);
  if (s != NX_C_OK) nx_c_raise("threefry", s);
  if (nx_c_dtype_of_value(vout) != NX_C_DTYPE_i32 ||
      nx_c_dtype_of_value(vkey) != NX_C_DTYPE_i32 ||
      nx_c_dtype_of_value(vctr) != NX_C_DTYPE_i32)
    nx_c_raise("threefry", NX_C_ERR_UNSUPPORTED_DTYPE);
  /* The driver odometers over key's shape and applies those coords to ctr/out
     strides, so it must VERIFY (never assume) that all three share a shape with
     a size-2 last axis — backend_intf only promises "compatible shapes", and a
     mismatched prefix would silently under-fill or mis-index. */
  if (key.ndim < 1 || key.ndim != ctr.ndim || key.ndim != out.ndim)
    nx_c_raise_invalid("threefry", NX_C_ERR_THREEFRY_SHAPE);
  int last = key.ndim - 1;
  for (int d = 0; d < key.ndim; d++)
    if (key.shape[d] != ctr.shape[d] || key.shape[d] != out.shape[d])
      nx_c_raise_invalid("threefry", NX_C_ERR_THREEFRY_SHAPE);
  if (key.shape[last] != 2)
    nx_c_raise_invalid("threefry", NX_C_ERR_THREEFRY_SHAPE);

  nx_c_threefry_ctx t = {&key,
                        &ctr,
                        &out,
                        last,
                        key.strides[last],
                        ctr.strides[last],
                        out.strides[last]};
  int64_t vectors = 1;
  for (int d = 0; d < last; d++) vectors *= key.shape[d];
  if (vectors > 0) {
    int nth = nx_c_threads_for(NX_C_COST_COMPUTE, vectors, 1, vectors * 16);
    if (nth > vectors) nth = (int)vectors;
    /* ctx is a stack struct; no per-thread scratch, so free_on_exit is NULL. */
    nx_c_parallel_for(nth, vectors, vectors * 16, nx_c_threefry_body, &t, NULL);
  }
  CAMLreturn(Val_unit);
}
