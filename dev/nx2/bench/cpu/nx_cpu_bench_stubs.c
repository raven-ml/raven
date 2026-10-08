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

#include "rig_pool.h"

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

static void slice(int64_t lo, int64_t hi, int worker, void *ctx) {
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

value nx_cpu_bench_floor_move_byte(value *argv, int argn) {
  (void)argn;
  return nx_cpu_bench_floor_move(argv[0], argv[1], argv[2], argv[3], argv[4],
                                 argv[5]);
}

value nx_cpu_bench_cores(value unit) {
  (void)unit;
  return Val_int(rig_pool_cores());
}

value nx_cpu_bench_performance_cores(value unit) {
  (void)unit;
  return Val_int(rig_pool_performance_cores());
}
