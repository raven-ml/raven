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
#include <malloc.h>
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

#include "device_core.h"
#include "nx_edge.h"

/* Locks, conditions and aligned memory, on Windows and on POSIX. */

#ifdef _WIN32
typedef SRWLOCK lock_t;
typedef CONDITION_VARIABLE cond_t;
static void lock_init(lock_t *l) { InitializeSRWLock(l); }
static void lock(lock_t *l) { AcquireSRWLockExclusive(l); }
static void unlock(lock_t *l) { ReleaseSRWLockExclusive(l); }
static void cond_init(cond_t *c) { InitializeConditionVariable(c); }
static void cond_wait(cond_t *c, lock_t *l) {
  SleepConditionVariableSRW(c, l, INFINITE, 0);
}
static void cond_broadcast(cond_t *c) { WakeAllConditionVariable(c); }
static size_t page(void) { return 4096; }
static void *aligned(size_t align, size_t n) {
  return _aligned_malloc(n, align);
}
static void aligned_free(void *p) { _aligned_free(p); }
#else
typedef pthread_mutex_t lock_t;
typedef pthread_cond_t cond_t;
static void lock_init(lock_t *l) { pthread_mutex_init(l, NULL); }
static void lock(lock_t *l) { pthread_mutex_lock(l); }
static void unlock(lock_t *l) { pthread_mutex_unlock(l); }
static void cond_init(cond_t *c) { pthread_cond_init(c, NULL); }
static void cond_wait(cond_t *c, lock_t *l) { pthread_cond_wait(c, l); }
static void cond_broadcast(cond_t *c) { pthread_cond_broadcast(c); }
static size_t page(void) { return (size_t)sysconf(_SC_PAGESIZE); }
static void *aligned(size_t align, size_t n) {
  void *p = NULL;
  return posix_memalign(&p, align, n) == 0 ? p : NULL;
}
static void aligned_free(void *p) { free(p); }
#endif

struct queued {
  uint64_t v;
  int nwaits, nparts;
  struct nx_wait *waits;
  struct nx_part *parts;
};

#define LAST 8

struct polled {
  _Atomic uint64_t word; /* first, alone in its page */
  lock_t mu;
  cond_t cv;
  int capacity;      /* parts the queue holds */
  int may_block;     /* a full queue's submit waits instead of [room] */
  int fail;          /* the next submit fails */
  int held;          /* parts queued */
  int n, c;
  struct queued *q;
  _Atomic int submits;
  int blocked; /* submits waiting for room */
  int nlast;   /* the waits of the last submit, the first [LAST] of them */
  struct nx_wait last[LAST];
};

#define Polled_val(v) ((struct polled *)Nativeint_val(v))

value device_core_test_polled_new(value v_capacity, value v_may_block) {
  struct polled *p = aligned(page(), page() > sizeof *p ? page() : sizeof *p);
  if (p == NULL) caml_raise_out_of_memory();
  memset(p, 0, sizeof *p);
  lock_init(&p->mu);
  cond_init(&p->cv);
  p->capacity = Int_val(v_capacity);
  p->may_block = Bool_val(v_may_block);
  return caml_copy_nativeint((intnat)p);
}

value device_core_test_polled_fail(value v_p) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  p->fail = 1;
  unlock(&p->mu);
  return Val_unit;
}

value device_core_test_polled_submits(value v_p) {
  return Val_int(atomic_load(&Polled_val(v_p)->submits));
}

value device_core_test_polled_blocked(value v_p) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  int n = p->blocked;
  unlock(&p->mu);
  return Val_int(n);
}

/* The waits of the last submit, as [| kind; at; value; … |]. */
value device_core_test_polled_last_waits(value v_p) {
  CAMLparam1(v_p);
  CAMLlocal1(a);
  struct polled *p = Polled_val(v_p);
  struct nx_wait w[LAST];
  lock(&p->mu);
  int n = p->nlast;
  memcpy(w, p->last, sizeof w);
  unlock(&p->mu);
  a = caml_alloc_tuple(3 * (mlsize_t)n);
  for (int i = 0; i < n; i++) {
    Store_field(a, 3 * i, Val_int(w[i].kind));
    Store_field(a, 3 * i + 1, Val_long((intnat)w[i].at));
    Store_field(a, 3 * i + 2, Val_long((intnat)w[i].value));
  }
  CAMLreturn(a);
}

/* The wait kinds of nx_edge.h. */
value device_core_test_nx_word(value unit) {
  (void)unit;
  return Val_int(NX_WORD);
}

value device_core_test_nx_object(value unit) {
  (void)unit;
  return Val_int(NX_OBJECT);
}

value device_core_test_polled_queued(value v_p) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  int n = p->n;
  unlock(&p->mu);
  return Val_int(n);
}

static int waits_hold(struct queued *s) {
  for (int i = 0; i < s->nwaits; i++) {
    /* An object is a 64-bit counter at its handle, as Polled's word. */
    int kind = s->waits[i].kind;
    uint64_t w = atomic_load((_Atomic uint64_t *)(uintptr_t)s->waits[i].at);
    if (kind == NX_WORD || kind == NX_OBJECT ? w < s->waits[i].value
                                             : w != s->waits[i].value)
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
  lock(&p->mu);
  int k = 0;
  while (k < p->n && waits_hold(&p->q[k])) run_one(p, &p->q[k++]);
  memmove(p->q, p->q + k, (size_t)(p->n - k) * sizeof *p->q);
  p->n -= k;
  cond_broadcast(&p->cv);
  unlock(&p->mu);
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
  lock(&p->mu);
  int full = p->held + n > p->capacity;
  unlock(&p->mu);
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
  lock(&p->mu);
  if (p->fail) {
    p->fail = 0;
    unlock(&p->mu);
    *failure = "the submission failed";
    return NX_FAILED;
  }
  while (p->may_block && p->held + nparts > p->capacity) {
    p->blocked++;
    cond_wait(&p->cv, &p->mu);
    p->blocked--;
  }
  if (p->n == p->c) {
    int c = p->c == 0 ? 8 : 2 * p->c;
    struct queued *q = realloc(p->q, (size_t)c * sizeof *q);
    if (q == NULL) {
      unlock(&p->mu);
      *failure = "no memory for the queue";
      return NX_FAILED;
    }
    p->q = q;
    p->c = c;
  }
  p->nlast = nwaits < LAST ? nwaits : LAST;
  if (p->nlast > 0) memcpy(p->last, waits, (size_t)p->nlast * sizeof *waits);
  struct queued *s = &p->q[p->n++];
  s->v = v;
  s->nwaits = nwaits;
  s->nparts = nparts;
  s->waits = malloc((size_t)(nwaits + 1) * sizeof *waits);
  s->parts = malloc((size_t)(nparts + 1) * sizeof *parts);
  if (nwaits > 0) memcpy(s->waits, waits, (size_t)nwaits * sizeof *waits);
  if (nparts > 0) memcpy(s->parts, parts, (size_t)nparts * sizeof *parts);
  p->held += nparts;
  unlock(&p->mu);
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
  aligned_free((void *)Long_val(v_a));
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

/* The C readers of device_core.h, as a caller in C sees a buffer. */
value device_core_test_reader_host(value v_b) {
  return Val_long((intnat)device_core_buffer_host(v_b));
}

value device_core_test_reader_bytes(value v_b) {
  return Val_long((intnat)device_core_buffer_bytes(v_b));
}

value device_core_test_reader_why(value v_b) {
  CAMLparam1(v_b);
  const char *why = device_core_buffer_why(v_b);
  CAMLreturn(why == NULL ? Val_none : caml_alloc_some(caml_copy_string(why)));
}
