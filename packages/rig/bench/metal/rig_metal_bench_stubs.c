/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The load that rows of waits run under, on macOS; elsewhere nothing here is
   called. Every stub holds the runtime. */

#define _GNU_SOURCE

#define CAML_NAME_SPACE
#include <caml/fail.h>
#include <caml/mlvalues.h>

#ifdef __APPLE__

#include <pthread.h>
#include <pthread/qos.h>

/* Load */

/* Moves the calling thread to the default class, the one of a thread made
   without a class, such as a domain's. */
value rig_metal_bench_default_class(value unit) {
  (void)unit;
  pthread_set_qos_class_self_np(QOS_CLASS_DEFAULT, 0);
  return Val_unit;
}

/* Threads at the default class that spin until [load_stop]: the load a
   wait shares the cores with. */
static int spinning;
static pthread_t spinners[64];
static long nspinners;

static void *spin(void *unused) {
  (void)unused;
  while (__atomic_load_n(&spinning, __ATOMIC_RELAXED)) {
  }
  return NULL;
}

value rig_metal_bench_load_stop(value unit) {
  (void)unit;
  __atomic_store_n(&spinning, 0, __ATOMIC_RELAXED);
  for (long i = 0; i < nspinners; i++) pthread_join(spinners[i], NULL);
  nspinners = 0;
  return Val_unit;
}

/* [v_n] spinning threads, at most 64. */
value rig_metal_bench_load_start(value v_n) {
  long n = Long_val(v_n) < 64 ? Long_val(v_n) : 64;
  __atomic_store_n(&spinning, 1, __ATOMIC_RELAXED);
  for (; nspinners < n; nspinners++)
    if (pthread_create(&spinners[nspinners], NULL, spin, NULL) != 0) {
      rig_metal_bench_load_stop(Val_unit);
      caml_failwith("rig_metal_bench_load_start: no thread");
    }
  return Val_unit;
}

#else

CAMLnoret static void no_metal(void) {
  caml_invalid_argument("rig_metal_bench: Metal exists on macOS only");
}

value rig_metal_bench_default_class(value a) { (void)a, no_metal(); }
value rig_metal_bench_load_start(value a) { (void)a, no_metal(); }
value rig_metal_bench_load_stop(value a) { (void)a, no_metal(); }

#endif
