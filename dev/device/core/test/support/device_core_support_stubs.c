/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Polled: a test driver over host memory whose queue runs only when its
   sleep, or the test, runs it. A submission is queued with its waits, its
   copies and its fills; running the queue runs each submission whose waits
   hold, in order, and stores its value in the word. None of these calls
   blocks but a full queue's submit, which waits for room. */

#define _GNU_SOURCE

#include <signal.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

#include "nx_edge.h"

struct queued {
  uint64_t v;
  int nwaits, nparts;
  struct nx_wait *waits;
  struct nx_part *parts;
};

struct polled {
  _Atomic uint64_t word; /* first, alone in its page */
  pthread_mutex_t mu;
  pthread_cond_t cv;
  int capacity;      /* parts the queue holds */
  int may_block;     /* a full queue's submit waits instead of [room] */
  int fail;          /* the next submit fails */
  int held;          /* parts queued */
  int n, c;
  struct queued *q;
  _Atomic int submits;
  int blocked; /* submits waiting for room */
};

static size_t page(void) {
#ifdef _WIN32
  return 4096;
#else
  return (size_t)sysconf(_SC_PAGESIZE);
#endif
}

static void *aligned(size_t align, size_t n) {
  void *p = NULL;
  return posix_memalign(&p, align, n) == 0 ? p : NULL;
}

#define Polled_val(v) ((struct polled *)Nativeint_val(v))

value device_core_test_polled_new(value v_capacity, value v_may_block) {
  struct polled *p = aligned(page(), page() > sizeof *p ? page() : sizeof *p);
  if (p == NULL) caml_raise_out_of_memory();
  memset(p, 0, sizeof *p);
  pthread_mutex_init(&p->mu, NULL);
  pthread_cond_init(&p->cv, NULL);
  p->capacity = Int_val(v_capacity);
  p->may_block = Bool_val(v_may_block);
  return caml_copy_nativeint((intnat)p);
}

value device_core_test_polled_fail(value v_p) {
  struct polled *p = Polled_val(v_p);
  pthread_mutex_lock(&p->mu);
  p->fail = 1;
  pthread_mutex_unlock(&p->mu);
  return Val_unit;
}

value device_core_test_polled_submits(value v_p) {
  return Val_int(atomic_load(&Polled_val(v_p)->submits));
}

value device_core_test_polled_blocked(value v_p) {
  struct polled *p = Polled_val(v_p);
  pthread_mutex_lock(&p->mu);
  int n = p->blocked;
  pthread_mutex_unlock(&p->mu);
  return Val_int(n);
}

value device_core_test_polled_queued(value v_p) {
  struct polled *p = Polled_val(v_p);
  pthread_mutex_lock(&p->mu);
  int n = p->n;
  pthread_mutex_unlock(&p->mu);
  return Val_int(n);
}

static int waits_hold(struct queued *s) {
  for (int i = 0; i < s->nwaits; i++) {
    uint64_t w = atomic_load((_Atomic uint64_t *)(uintptr_t)s->waits[i].at);
    if (s->waits[i].kind == NX_EQUAL ? w != s->waits[i].value
                                     : w < s->waits[i].value)
      return 0;
  }
  return 1;
}

static void run_one(struct polled *p, struct queued *s) {
  for (int i = 0; i < s->nparts; i++) {
    struct nx_part *part = &s->parts[i];
    if (part->fill != NULL) part->fill(NULL, part->arg, s->v);
    else if (part->copy_bytes != 0)
      memmove((char *)(uintptr_t)part->copy_dst + part->copy_dst_offset,
              (const char *)(uintptr_t)part->copy_src + part->copy_src_offset,
              (size_t)part->copy_bytes);
  }
  p->held -= s->nparts;
  free(s->waits);
  free(s->parts);
  atomic_store_explicit(&p->word, s->v, memory_order_release);
}

/* Runs the queued submissions whose waits hold, in order: answers how
   many ran. */
static int run(struct polled *p) {
  pthread_mutex_lock(&p->mu);
  int k = 0;
  while (k < p->n && waits_hold(&p->q[k])) run_one(p, &p->q[k++]);
  memmove(p->q, p->q + k, (size_t)(p->n - k) * sizeof *p->q);
  p->n -= k;
  pthread_cond_broadcast(&p->cv);
  pthread_mutex_unlock(&p->mu);
  return k;
}

value device_core_test_polled_run(value v_p) {
  return Val_int(run(Polled_val(v_p)));
}

static int polled_room(void *self, const struct nx_part *parts, int n) {
  struct polled *p = self;
  for (int i = 0; i < n; i++)
    if (parts[i].words != NULL) return NX_NEVER;
  if (n > p->capacity) return NX_NEVER;
  if (p->may_block) return NX_FITS;
  pthread_mutex_lock(&p->mu);
  int full = p->held + n > p->capacity;
  pthread_mutex_unlock(&p->mu);
  return full ? NX_LATER : NX_FITS;
}

static int polled_submit(void *self, uint64_t v, const struct nx_wait *waits,
                         int nwaits, const struct nx_part *parts, int nparts,
                         const uint64_t *handles, int nhandles,
                         const char **failure) {
  struct polled *p = self;
  (void)handles;
  (void)nhandles;
  atomic_fetch_add(&p->submits, 1);
  pthread_mutex_lock(&p->mu);
  if (p->fail) {
    p->fail = 0;
    pthread_mutex_unlock(&p->mu);
    *failure = "the submission failed";
    return NX_FAILED;
  }
  while (p->may_block && p->held + nparts > p->capacity) {
    p->blocked++;
    pthread_cond_wait(&p->cv, &p->mu);
    p->blocked--;
  }
  if (p->n == p->c) {
    int c = p->c == 0 ? 8 : 2 * p->c;
    struct queued *q = realloc(p->q, (size_t)c * sizeof *q);
    if (q == NULL) {
      pthread_mutex_unlock(&p->mu);
      *failure = "no memory for the queue";
      return NX_FAILED;
    }
    p->q = q;
    p->c = c;
  }
  struct queued *s = &p->q[p->n++];
  s->v = v;
  s->nwaits = nwaits;
  s->nparts = nparts;
  s->waits = malloc((size_t)(nwaits + 1) * sizeof *waits);
  s->parts = malloc((size_t)(nparts + 1) * sizeof *parts);
  if (nwaits > 0) memcpy(s->waits, waits, (size_t)nwaits * sizeof *waits);
  if (nparts > 0) memcpy(s->parts, parts, (size_t)nparts * sizeof *parts);
  p->held += nparts;
  pthread_mutex_unlock(&p->mu);
  return NX_OK;
}

value device_core_test_polled_room(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&polled_room);
}

value device_core_test_polled_submit(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&polled_submit);
}

value device_core_test_polled_word(value v_p) {
  return Val_long((intnat)atomic_load(&Polled_val(v_p)->word));
}

/* Writes [v] into the word, as a stopped device's driver does. */
value device_core_test_polled_set_word(value v_p, value v) {
  atomic_store(&Polled_val(v_p)->word, (uint64_t)Long_val(v));
  return Val_unit;
}

/* Host memory for regions: on a page, so any of it maps. */
value device_core_test_alloc(value v_n) {
  size_t n = (size_t)Long_val(v_n);
  void *a = aligned(page(), n == 0 ? 1 : n);
  return Val_long((intnat)a);
}

value device_core_test_free(value v_a) {
  free((void *)Long_val(v_a));
  return Val_unit;
}

/* A fill that adds 1 to the 64-bit word its argument points at. */
static int bump(void *queue, void *arg, uint64_t v) {
  (void)queue;
  (void)v;
  _Atomic uint64_t *w = arg;
  atomic_fetch_add(w, 1);
  return 0;
}

value device_core_test_bump(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&bump);
}

/* A fill that stores the second 64-bit word of its argument at the address
   the first holds. */
static int poke(void *queue, void *arg, uint64_t v) {
  (void)queue;
  (void)v;
  _Atomic uint64_t *a = arg;
  _Atomic uint64_t *at = (_Atomic uint64_t *)(uintptr_t)atomic_load(&a[0]);
  atomic_store(at, atomic_load(&a[1]));
  return 0;
}

value device_core_test_poke(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&poke);
}

/* Raises SIGINT in the calling thread: the runtime records it, and the
   thread's next poll point runs its handler. */
value device_core_test_interrupt(value unit) {
  (void)unit;
  raise(SIGINT);
  return Val_unit;
}

value device_core_test_load(value v_addr) {
  return Val_long((intnat)atomic_load((_Atomic uint64_t *)Long_val(v_addr)));
}

value device_core_test_store(value v_addr, value v) {
  atomic_store((_Atomic uint64_t *)Long_val(v_addr), (uint64_t)Long_val(v));
  return Val_unit;
}
