/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Holds a link's transfers and counts its release entrants, through the
   hook the link calls at each point of its protocol, and shortens a link's
   bounds. */

#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <time.h>

#include "../../../lib/remote/proxy/rig_remote_proxy_stubs.h"

static int64_t now_ns(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec;
}

/* Waits on [cv] for at most [ns] nanoseconds. */
static void wait_ns(pthread_cond_t *cv, pthread_mutex_t *mu, int64_t ns) {
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  int64_t at = (int64_t)t.tv_nsec + ns;
  t.tv_sec += (time_t)(at / 1000000000LL);
  t.tv_nsec = (long)(at % 1000000000LL);
  pthread_cond_timedwait(cv, mu, &t);
}

static pthread_mutex_t hook_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t hook_cv = PTHREAD_COND_INITIALIZER;
static int armed, held, entered, waiters;
static const int64_t patience = 5000000000LL;

static void hook(enum rig_remote_link_point at, int waiting) {
  pthread_mutex_lock(&hook_mu);
  if (armed == (int)at + 1) {
    held = 1;
    pthread_cond_broadcast(&hook_cv);
    int64_t deadline = now_ns() + patience;
    while (armed && now_ns() < deadline)
      wait_ns(&hook_cv, &hook_mu, deadline - now_ns());
  } else if (armed && at == RIG_REMOTE_LINK_RELEASING) {
    entered++;
    waiters += waiting;
    pthread_cond_broadcast(&hook_cv);
  }
  pthread_mutex_unlock(&hook_mu);
}

value rig_link_probe_arm(value point) {
  CAMLparam1(point);
  pthread_mutex_lock(&hook_mu);
  armed = Int_val(point) + 1;
  held = entered = waiters = 0;
  pthread_mutex_unlock(&hook_mu);
  atomic_store_explicit(&rig_remote_link_hook, hook, memory_order_release);
  CAMLreturn(Val_unit);
}

value rig_link_probe_held(value unit) {
  CAMLparam1(unit);
  caml_release_runtime_system();
  pthread_mutex_lock(&hook_mu);
  int64_t deadline = now_ns() + patience;
  while (!held && now_ns() < deadline)
    wait_ns(&hook_cv, &hook_mu, deadline - now_ns());
  int observed = held;
  pthread_mutex_unlock(&hook_mu);
  caml_acquire_runtime_system();
  CAMLreturn(Val_bool(observed));
}

value rig_link_probe_releases(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(result);
  caml_release_runtime_system();
  pthread_mutex_lock(&hook_mu);
  int64_t deadline = now_ns() + patience;
  while (entered < 2 && now_ns() < deadline)
    wait_ns(&hook_cv, &hook_mu, deadline - now_ns());
  int e = entered, w = waiters;
  pthread_mutex_unlock(&hook_mu);
  caml_acquire_runtime_system();
  result = caml_alloc_tuple(2);
  Store_field(result, 0, Val_int(e));
  Store_field(result, 1, Val_int(w));
  CAMLreturn(result);
}

value rig_link_probe_resume(value unit) {
  CAMLparam1(unit);
  pthread_mutex_lock(&hook_mu);
  armed = 0;
  pthread_cond_broadcast(&hook_cv);
  pthread_mutex_unlock(&hook_mu);
  CAMLreturn(Val_unit);
}

value rig_link_probe_bound(value s) {
  CAMLparam1(s);
  atomic_store(&rig_remote_link_bound_s, Int_val(s));
  CAMLreturn(Val_unit);
}
