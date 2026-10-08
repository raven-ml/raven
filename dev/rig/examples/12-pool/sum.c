/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A job on the host's pool: the sum of the squares of an int64 bigarray,
   cut into chunks that the pool's threads claim. Each thread adds into its
   own partial, indexed by its worker number; the caller adds the partials
   once the job returned. */

#define _GNU_SOURCE

#include <stdint.h>
#include <stdlib.h>

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#include <rig_pool.h>

struct job {
  const int64_t *x;
  int64_t *partials;
};

/* Runs on any thread of the pool: plain C, no OCaml runtime. */
static void squares(int64_t lo, int64_t hi, int worker, void *ctx) {
  struct job *j = ctx;
  int64_t s = 0;
  for (int64_t i = lo; i < hi; i++) s += j->x[i] * j->x[i];
  j->partials[worker] += s;
}

/* Releases the runtime during the job: the workers run while other domains
   collect. The bigarray's data lies outside the OCaml heap and [v_x] keeps
   it alive. */
value caml_rig_example_sum_squares(value v_threads, value v_chunks,
                                   value v_x) {
  CAMLparam3(v_threads, v_chunks, v_x);
  int threads = Int_val(v_threads);
  int64_t chunks = Long_val(v_chunks);
  int64_t n = Caml_ba_array_val(v_x)->dim[0];
  int cores = rig_pool_cores();
  struct job j = {Caml_ba_data_val(v_x), calloc(cores, sizeof(int64_t))};
  if (j.partials == NULL) caml_raise_out_of_memory();
  caml_release_runtime_system();
  rig_pool_run(threads, n, chunks, squares, &j);
  caml_acquire_runtime_system();
  int64_t s = 0;
  for (int w = 0; w < cores; w++) s += j.partials[w];
  free(j.partials);
  CAMLreturn(caml_copy_int64(s));
}

/* Does not release the runtime: it reads two facts computed once. */
value caml_rig_example_cores(value unit) {
  (void)unit;
  value r = caml_alloc_tuple(2);
  Store_field(r, 0, Val_int(rig_pool_cores()));
  Store_field(r, 1, Val_int(rig_pool_performance_cores()));
  return r;
}
