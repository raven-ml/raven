/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Floors: what the hardware does with the bytes of a row, running no nx.cpu
   code. A copy's floor is memcpy; a cast's is a loop that reads and writes
   the cast's bytes with one instruction per element, an integer's
   truncation or extension. Both run on the pool's threads in slices, as
   nx.cpu's jobs do, with the runtime released. */

#include <stdint.h>
#include <string.h>

#include <caml/bigarray.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>

#include "cpu.h"

/* The loops are compiled for x86-64's v3 too, and the host's best runs, as
   nx.cpu's own loops are (convert_v3.c, rows_v3.c): a floor compiled for
   base alone ran less-f32-1M's bytes at 35 us on kimchi, slower than the
   kernel's 24-30. */
#if defined(__x86_64__) && defined(__linux__)
#define BEST __attribute__((target_clones("avx2", "default")))
#else
#define BEST
#endif

/* Slices of a job per thread: as many as nx.cpu's jobs make. */
#define SLICES 8

typedef struct {
  uint8_t *d;
  const uint8_t *s;
  int64_t n, total;
  int in, out; /* bytes per element, read and written; 0 for memcpy */
} floor_job;

typedef struct {
  uint64_t lo, hi;
} u128;

/* d[i] from s[i]: the low bytes of a wider element, or a narrower one
   extended with zeros. */
#define MOVE(SI, SO)                                               \
  for (int64_t i = 0; i < n; i++) ((SO *)d)[i] = (SO)((const SI *)s)[i]

BEST static void slice(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const floor_job *f = ctx;
  int64_t first = f->n * lo / f->total, n = f->n * hi / f->total - first;
  if (f->in == 0) {
    memcpy(f->d + first, f->s + first, (size_t)n);
    return;
  }
  uint8_t *d = f->d + first * f->out;
  const uint8_t *s = f->s + first * f->in;
  switch (f->in * 100 + f->out) {
    case 401: MOVE(uint32_t, uint8_t); return;
    case 402: MOVE(uint32_t, uint16_t); return;
    case 404: MOVE(uint32_t, uint32_t); return;
    case 408: MOVE(uint32_t, uint64_t); return;
    case 104: MOVE(uint8_t, uint32_t); return;
    case 102: MOVE(uint8_t, uint16_t); return;
    case 201: MOVE(uint16_t, uint8_t); return;
    case 204: MOVE(uint16_t, uint32_t); return;
    case 804: MOVE(uint64_t, uint32_t); return;
    case 802: MOVE(uint64_t, uint16_t); return;
    default: /* 8 to 16 */
      for (int64_t i = 0; i < n; i++)
        ((u128 *)d)[i] = (u128){((const uint64_t *)s)[i], 0};
      return;
  }
}

/* [floor_move threads in out dst src n] moves [n] elements of [in] bytes
   from [src] into [n] of [out] bytes in [dst], or memcpy's [n] bytes when
   [in] is 0, on [threads] of the pool's threads. */
value nx_cpu_bench_floor_move(value threads, value in, value out, value dst,
                              value src, value n) {
  int t = Int_val(threads);
  floor_job f = {Caml_ba_data_val(dst), Caml_ba_data_val(src), Long_val(n),
                 t * SLICES, Int_val(in), Int_val(out)};
  if (t == 1) {
    f.total = 1;
    slice(0, 1, 0, &f);
    return Val_unit;
  }
  caml_enter_blocking_section_no_pending();
  rig_pool_run(t, f.total, f.total, slice, &f);
  caml_leave_blocking_section();
  return Val_unit;
}

/* Streams: [ins] arrays of [n] elements read, their xor stored into [n]
   elements of [outb] bytes, one integer operation per element: the floor
   of an elementwise kind's loop. The inputs are [inb] bytes each, or, for
   where's, a byte then two of [inb] bytes, of which a mask from the byte
   picks one: where-f32-1M ran under an xor's floor on kimchi, and a
   branching select ran scalar on the M1. With no array read it stores a
   constant through nx.cpu's fill row, as a fill does. */
typedef struct {
  uint8_t *d;
  const uint8_t *s[3];
  int64_t n, total;
  int ins, inb, outb, cond;
} stream_job;

#define STREAM(TI, TO)                                                       \
  {                                                                          \
    TO *o = (TO *)d;                                                         \
    const TI *a = (const TI *)s[0], *b = (const TI *)s[1],                   \
             *c = (const TI *)s[2];                                          \
    switch (f->ins) {                                                        \
      case 1: for (int64_t i = 0; i < n; i++) o[i] = (TO)a[i]; break;        \
      case 2: for (int64_t i = 0; i < n; i++) o[i] = (TO)(a[i] ^ b[i]); break; \
      default:                                                               \
        for (int64_t i = 0; i < n; i++) o[i] = (TO)(a[i] ^ b[i] ^ c[i]);     \
    }                                                                        \
  }

BEST static void stream_slice(int64_t lo, int64_t hi, int worker,
                              void *ctx) {
  (void)worker;
  const stream_job *f = ctx;
  int64_t first = f->n * lo / f->total, n = f->n * hi / f->total - first;
  uint8_t *d = f->d + first * f->outb;
  const uint8_t *s[3];
  for (int k = 0; k < 3; k++) s[k] = f->s[k] + first * f->inb;
  if (f->ins == 0) {
    /* A fill's floor is nx.cpu's own fill row on the slice, nothing around
       it: a loop of stores written here ran slower than fill-f32-1M on the
       M1, which no floor may. */
    static const uint8_t one[8] = {1};
    int i = f->outb == 1 ? 0 : f->outb == 2 ? 1 : f->outb == 4 ? 2 : 3;
    nx_cpu_runs->fill[i](n, d, 1, one);
    return;
  }
  if (f->cond) {
    /* A condition byte, then two words: where's operands. */
    const uint8_t *c = f->s[0] + first;
    const uint32_t *a = (const uint32_t *)s[1], *b = (const uint32_t *)s[2];
    uint32_t *o = (uint32_t *)d;
    for (int64_t i = 0; i < n; i++) {
      uint32_t m = 0u - (uint32_t)(c[i] != 0);
      o[i] = (a[i] & m) | (b[i] & ~m);
    }
    return;
  }
  switch (f->inb * 100 + f->outb) {
    case 101: STREAM(uint8_t, uint8_t); return;
    case 401: STREAM(uint32_t, uint8_t); return;
    case 404: STREAM(uint32_t, uint32_t); return;
    default: STREAM(uint64_t, uint64_t); return; /* 8 to 8 */
  }
}

/* [floor_stream threads ins inb outb dst a b c n]: a stream on [threads] of
   the pool's threads; [b] and [c] are read only as [ins] says. */
value nx_cpu_bench_floor_stream(value threads, value ins, value inb,
                                value outb, value dst, value a, value b,
                                value c, value n) {
  int t = Int_val(threads);
  /* [ins] of -1 asks for where's: a condition byte, then two of [inb]. */
  int cond = Int_val(ins) < 0;
  stream_job f = {Caml_ba_data_val(dst),
                  {Caml_ba_data_val(a), Caml_ba_data_val(b),
                   Caml_ba_data_val(c)},
                  Long_val(n),
                  t * SLICES,
                  cond ? 3 : Int_val(ins),
                  Int_val(inb),
                  Int_val(outb),
                  cond};
  if (t == 1) {
    f.total = 1;
    stream_slice(0, 1, 0, &f);
    return Val_unit;
  }
  caml_enter_blocking_section_no_pending();
  rig_pool_run(t, f.total, f.total, stream_slice, &f);
  caml_leave_blocking_section();
  return Val_unit;
}

value nx_cpu_bench_floor_stream_byte(value *argv, int argn) {
  (void)argn;
  return nx_cpu_bench_floor_stream(argv[0], argv[1], argv[2], argv[3],
                                   argv[4], argv[5], argv[6], argv[7],
                                   argv[8]);
}

value nx_cpu_bench_floor_move_byte(value *argv, int argn) {
  (void)argn;
  return nx_cpu_bench_floor_move(argv[0], argv[1], argv[2], argv[3], argv[4],
                                 argv[5]);
}

/* [block_transposed dst src n bits] copies the transpose of [src], [n] x
   [n] elements of [bits] bits, into [dst] with nx_array.h's block copy, on
   the calling thread: the copy kernel the walk runs, without the walk. */
value nx_cpu_bench_block_transposed(value dst, value src, value n,
                                    value bits) {
  int64_t k = Long_val(n);
  nx_copy_box(Caml_ba_data_val(dst), Caml_ba_data_val(src),
              &(nx_box){{1, k, k}, {0, 0}, {{0, k, 1}, {0, 1, k}}},
              Int_val(bits));
  return Val_unit;
}

value nx_cpu_bench_cores(value unit) {
  (void)unit;
  return Val_int(rig_pool_cores());
}

value nx_cpu_bench_performance_cores(value unit) {
  (void)unit;
  return Val_int(rig_pool_performance_cores());
}
