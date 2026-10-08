/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Polled: a test driver over host memory whose queue runs only when its
   sleep, or the test, runs it. A submission is queued with its waits, its
   copies and its fills; running the queue runs each committed submission
   whose waits hold, in order, and stores its value in the word. The driver
   commits on its own once [lag] values are uncommitted, and before a
   submit waits for room. A sleep first runs the Polled devices its first
   submission waits for. None of these calls blocks but a full queue's
   submit, which waits for room. */

#define _GNU_SOURCE

#include <signal.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#ifdef _WIN32
#include <malloc.h>
#include <windows.h>
#else
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#endif

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define RIG_TEST_ASAN
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#define RIG_TEST_ASAN
#endif

#if defined(RIG_TEST_ASAN)
#include <sanitizer/allocator_interface.h>
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#elif defined(__GLIBC__)
#include <malloc.h>
#endif

#include "rig.h"
#include "rig_edge.h"

/* Locks, conditions and aligned memory, on Windows and on POSIX. */

#ifdef _WIN32
typedef SRWLOCK lock_t;
#define LOCK_INITIALIZER SRWLOCK_INIT
typedef CONDITION_VARIABLE cond_t;
static void lock_init(lock_t *l) { InitializeSRWLock(l); }
static void lock(lock_t *l) { AcquireSRWLockExclusive(l); }
static void unlock(lock_t *l) { ReleaseSRWLockExclusive(l); }
static void cond_init(cond_t *c) { InitializeConditionVariable(c); }
static void cond_wait(cond_t *c, lock_t *l) {
  SleepConditionVariableSRW(c, l, INFINITE, 0);
}
static void cond_broadcast(cond_t *c) { WakeAllConditionVariable(c); }
static void nap(void) { Sleep(1); }
static DWORD WINAPI engine_main(void *p);
static int spawn(void *p) {
  HANDLE t = CreateThread(NULL, 0, engine_main, p, 0, NULL);
  if (t == NULL) return 0;
  CloseHandle(t);
  return 1;
}
static size_t page(void) {
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return info.dwPageSize;
}
static void *aligned(size_t align, size_t n) {
  return _aligned_malloc(n, align);
}
static void aligned_free(void *p) { _aligned_free(p); }
#else
typedef pthread_mutex_t lock_t;
#define LOCK_INITIALIZER PTHREAD_MUTEX_INITIALIZER
typedef pthread_cond_t cond_t;
static void lock_init(lock_t *l) { pthread_mutex_init(l, NULL); }
static void lock(lock_t *l) { pthread_mutex_lock(l); }
static void unlock(lock_t *l) { pthread_mutex_unlock(l); }
static void cond_init(cond_t *c) { pthread_cond_init(c, NULL); }
static void cond_wait(cond_t *c, lock_t *l) { pthread_cond_wait(c, l); }
static void cond_broadcast(cond_t *c) { pthread_cond_broadcast(c); }
static void nap(void) {
  struct timespec t = {0, 100000};
  nanosleep(&t, NULL);
}
static void *engine_main(void *p);
static int spawn(void *p) {
  pthread_t t;
  if (pthread_create(&t, NULL, engine_main, p) != 0) return 0;
  pthread_detach(t);
  return 1;
}
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
  struct rig_wait *waits;
  struct rig_part *parts;
};

#define LAST 8
#define LAST_HANDLES 64

struct polled {
  _Atomic uint64_t word; /* first, alone in its page */
  _Atomic uint64_t committed; /* the last value committed */
  uint64_t lag;               /* the uncommitted values that commit */
  lock_t mu;
  cond_t cv;
  cond_t work;       /* signalled by a submit to a device that runs itself */
  int itself;        /* a thread of the driver runs the queue */
  int capacity;      /* parts the queue holds */
  int may_block;     /* a full queue's submit waits instead of [room] */
  int fail;          /* the next submit fails */
  int fail_commit;   /* the next commit of uncommitted values fails */
  int held;          /* parts queued */
  int n, c;
  struct queued *q;
  _Atomic int submits;
  int blocked; /* submits waiting for room */
  int nlast;   /* the waits of the last submit, the first [LAST] of them */
  struct rig_wait last[LAST];
  int nlast_handles; /* its handles, the first [LAST_HANDLES] of them */
  uint64_t last_handles[LAST_HANDLES];
  int nsides; /* the copy_local of each copy run, the first [LAST_HANDLES] */
  int sides[LAST_HANDLES];
  uint64_t received; /* the last value a submit received */
  intnat steps;     /* fallible calls made */
  intnat fail_from; /* the step from which every call fails, or 0 */
  intnat refuse_from, refuse_to; /* the steps from one below the other refuse */
  char why[64]; /* what a hand-over failing from [fail_from] reports */
};

#define Polled_val(v) ((struct polled *)Nativeint_val(v))

/* Every Polled device, never freed: a sleep runs the ones whose words the
   sleeper's work waits for. */
static lock_t all_mu = LOCK_INITIALIZER;
static struct polled **all;
static int nall, call;

static void enrol(struct polled *p) {
  lock(&all_mu);
  if (nall == call) {
    int c = call == 0 ? 16 : 2 * call;
    struct polled **a = realloc(all, (size_t)c * sizeof *a);
    if (a == NULL) {
      unlock(&all_mu);
      caml_raise_out_of_memory();
    }
    all = a;
    call = c;
  }
  all[nall++] = p;
  unlock(&all_mu);
}

static int enrolled(uintptr_t at) {
  lock(&all_mu);
  int found = 0;
  for (int i = 0; i < nall && !found; i++) found = (uintptr_t)all[i] == at;
  unlock(&all_mu);
  return found;
}

value rig_test_polled_new(value v_capacity, value v_may_block,
                          value v_lag) {
  struct polled *p = aligned(page(), page() > sizeof *p ? page() : sizeof *p);
  if (p == NULL) caml_raise_out_of_memory();
  memset(p, 0, sizeof *p);
  lock_init(&p->mu);
  cond_init(&p->cv);
  cond_init(&p->work);
  p->capacity = Int_val(v_capacity);
  p->may_block = Bool_val(v_may_block);
  p->lag = (uint64_t)Long_val(v_lag);
  enrol(p);
  return caml_copy_nativeint((intnat)p);
}

value rig_test_polled_fail(value v_p) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  p->fail = 1;
  unlock(&p->mu);
  return Val_unit;
}

value rig_test_polled_fail_commit(value v_p) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  p->fail_commit = 1;
  unlock(&p->mu);
  return Val_unit;
}

/* Counts a fallible call: whether it fails (STEP_FAIL), refuses
   (STEP_REFUSE) or goes on (0), as [rig_test_polled_fail_at] set. [p]'s
   lock is held. */
#define STEP_REFUSE 1
#define STEP_FAIL 2

static int step(struct polled *p) {
  intnat n = ++p->steps;
  if (p->fail_from > 0 && n >= p->fail_from) return STEP_FAIL;
  return n >= p->refuse_from && n < p->refuse_to ? STEP_REFUSE : 0;
}

value rig_test_polled_step(value v_p) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  int s = step(p);
  unlock(&p->mu);
  return Val_int(s);
}

value rig_test_polled_steps(value v_p) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  intnat n = p->steps;
  unlock(&p->mu);
  return Val_long(n);
}

/* Makes the [v_n]-th fallible call from now on, and every later one, fail
   with [v_why] if [v_k] is negative, and [v_k] calls from the [v_n]-th
   refuse otherwise. */
value rig_test_polled_fail_at(value v_p, value v_n, value v_k, value v_why) {
  struct polled *p = Polled_val(v_p);
  intnat k = Long_val(v_k);
  lock(&p->mu);
  intnat n = p->steps + Long_val(v_n);
  if (k < 0) {
    strncpy(p->why, String_val(v_why), sizeof p->why - 1);
    p->fail_from = n;
  } else {
    p->refuse_from = n;
    p->refuse_to = k > Max_long - n ? Max_long : n + k;
  }
  unlock(&p->mu);
  return Val_unit;
}

value rig_test_polled_submits(value v_p) {
  return Val_int(atomic_load(&Polled_val(v_p)->submits));
}

value rig_test_polled_blocked(value v_p) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  int n = p->blocked;
  unlock(&p->mu);
  return Val_int(n);
}

/* The waits of the last submit, as [| kind; at; value; … |]. */
value rig_test_polled_last_waits(value v_p) {
  CAMLparam1(v_p);
  CAMLlocal1(a);
  struct polled *p = Polled_val(v_p);
  struct rig_wait w[LAST];
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

/* The handles of the last submit. */
value rig_test_polled_last_handles(value v_p) {
  CAMLparam1(v_p);
  CAMLlocal1(a);
  struct polled *p = Polled_val(v_p);
  uint64_t h[LAST_HANDLES];
  lock(&p->mu);
  int n = p->nlast_handles;
  memcpy(h, p->last_handles, sizeof h);
  unlock(&p->mu);
  a = caml_alloc_tuple((mlsize_t)n);
  for (int i = 0; i < n; i++) Store_field(a, i, Val_long((intnat)h[i]));
  CAMLreturn(a);
}

/* The copy_local of each copy part [p] ran, in order. */
value rig_test_polled_copy_sides(value v_p) {
  CAMLparam1(v_p);
  CAMLlocal1(a);
  struct polled *p = Polled_val(v_p);
  int s[LAST_HANDLES];
  lock(&p->mu);
  int n = p->nsides;
  memcpy(s, p->sides, sizeof s);
  unlock(&p->mu);
  a = caml_alloc_tuple((mlsize_t)n);
  for (int i = 0; i < n; i++) Store_field(a, i, Val_int(s[i]));
  CAMLreturn(a);
}

/* rig_edge.h's RIG_LOCAL_NONE, RIG_LOCAL_SRC and RIG_LOCAL_DST. */
value rig_test_rig_local(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(t);
  t = caml_alloc_tuple(3);
  Store_field(t, 0, Val_int(RIG_LOCAL_NONE));
  Store_field(t, 1, Val_int(RIG_LOCAL_SRC));
  Store_field(t, 2, Val_int(RIG_LOCAL_DST));
  CAMLreturn(t);
}

/* The wait kinds of rig_edge.h. */
value rig_test_rig_word(value unit) {
  (void)unit;
  return Val_int(RIG_WORD);
}

value rig_test_rig_object(value unit) {
  (void)unit;
  return Val_int(RIG_OBJECT);
}

value rig_test_polled_queued(value v_p) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  int n = p->n;
  unlock(&p->mu);
  return Val_int(n);
}

static int wait_holds(const struct rig_wait *wait) {
  /* An object is a 64-bit counter at its handle, as Polled's word. */
  uint64_t w = atomic_load((_Atomic uint64_t *)(uintptr_t)wait->at);
  return wait->kind == RIG_WORD || wait->kind == RIG_OBJECT ? w >= wait->value
                                                            : w == wait->value;
}

static int waits_hold(struct queued *s) {
  for (int i = 0; i < s->nwaits; i++)
    if (!wait_holds(&s->waits[i])) return 0;
  return 1;
}

static int committed(struct polled *p, struct queued *s) {
  return s->v <= atomic_load_explicit(&p->committed, memory_order_relaxed);
}

/* Commits [p]'s values up to [v]. [p]'s lock is held. */
static void commit_upto(struct polled *p, uint64_t v) {
  if (v <= atomic_load_explicit(&p->committed, memory_order_relaxed)) return;
  atomic_store_explicit(&p->committed, v, memory_order_release);
  if (p->itself) cond_broadcast(&p->work);
}

static void run_one(struct polled *p, struct queued *s) {
  for (int i = 0; i < s->nparts; i++) {
    struct rig_part *part = &s->parts[i];
    if (part->fill != NULL) part->fill(NULL, part->arg, s->v);
    else if (part->copy_bytes != 0) {
      /* A side that copy_local names holds a host address, as Polled's own
         handles do. */
      memmove((char *)(uintptr_t)part->copy_dst + part->copy_dst_offset,
              (const char *)(uintptr_t)part->copy_src + part->copy_src_offset,
              (size_t)part->copy_bytes);
      if (p->nsides < LAST_HANDLES) p->sides[p->nsides++] = part->copy_local;
    }
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
  while (k < p->n && committed(p, &p->q[k]) && waits_hold(&p->q[k]))
    run_one(p, &p->q[k++]);
  if (k > 0) memmove(p->q, p->q + k, (size_t)(p->n - k) * sizeof *p->q);
  p->n -= k;
  cond_broadcast(&p->cv);
  unlock(&p->mu);
  return k;
}

value rig_test_polled_run(value v_p) {
  return Val_int(run(Polled_val(v_p)));
}

/* How deep a sleep follows Polled devices waiting on one another. */
#define DRIVE_DEPTH 8

/* What [drive] answers when nothing ran for want of a commit. */
#define STUCK (-1)

/* Runs [p]'s queue after the queues of the Polled devices whose words its
   first submission waits for and has not seen, as devices run their own
   work while the host sleeps on another one. Answers how many ran, or
   STUCK if none did while that submission, or the work of one of those
   devices it waits for, is not committed. */
static int drive(struct polled *p, int depth) {
  struct polled *wanted[LAST];
  int n = 0, stuck = 0;
  lock(&p->mu);
  if (p->n > 0) {
    struct queued *s = &p->q[0];
    stuck = !committed(p, s);
    for (int i = 0; i < s->nwaits && n < LAST && depth > 0; i++)
      if (enrolled((uintptr_t)s->waits[i].at) && !wait_holds(&s->waits[i]))
        wanted[n++] = (struct polled *)(uintptr_t)s->waits[i].at;
  }
  unlock(&p->mu);
  for (int i = 0; i < n; i++)
    if (wanted[i] != p && drive(wanted[i], depth - 1) == STUCK) stuck = 1;
  int k = run(p);
  return k == 0 && stuck ? STUCK : k;
}

value rig_test_polled_drive(value v_p) {
  return Val_int(drive(Polled_val(v_p), DRIVE_DEPTH));
}

/* A device that runs itself: its thread runs the queue as work arrives and,
   while the first submission waits on a word that has not moved, again after
   a nap, as a device runs its own work. */
static void engine(struct polled *p) {
  for (;;) {
    lock(&p->mu);
    while (p->n == 0 || !committed(p, &p->q[0])) cond_wait(&p->work, &p->mu);
    unlock(&p->mu);
    if (run(p) == 0) nap();
  }
}

#ifdef _WIN32
static DWORD WINAPI engine_main(void *p) {
  engine(p);
  return 0;
}
#else
static void *engine_main(void *p) {
  engine(p);
  return NULL;
}
#endif

value rig_test_polled_start(value v_p) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  p->itself = 1;
  unlock(&p->mu);
  if (!spawn(p)) caml_failwith("Polled: cannot start the driver's thread");
  return Val_unit;
}

static int polled_room(void *self, const struct rig_part *parts, int n) {
  struct polled *p = self;
  for (int i = 0; i < n; i++)
    if (parts[i].words != NULL) return RIG_NEVER;
  if (n > p->capacity) return RIG_NEVER;
  if (p->may_block) return RIG_FITS;
  lock(&p->mu);
  int full = p->held + n > p->capacity;
  unlock(&p->mu);
  return full ? RIG_LATER : RIG_FITS;
}

static int polled_submit(void *self, uint64_t v, const struct rig_wait *waits,
                         int nwaits, const struct rig_part *parts, int nparts,
                         const uint64_t *handles, int nhandles,
                         const char **failure) {
  struct polled *p = self;
  atomic_fetch_add(&p->submits, 1);
  lock(&p->mu);
  p->received = v;
  if (step(p) == STEP_FAIL) {
    unlock(&p->mu);
    *failure = p->why;
    return RIG_FAILED;
  }
  if (p->fail) {
    p->fail = 0;
    unlock(&p->mu);
    *failure = "the submission failed";
    return RIG_FAILED;
  }
  while (p->may_block && p->held + nparts > p->capacity) {
    commit_upto(p, v - 1);
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
      return RIG_FAILED;
    }
    p->q = q;
    p->c = c;
  }
  p->nlast = nwaits < LAST ? nwaits : LAST;
  if (p->nlast > 0) memcpy(p->last, waits, (size_t)p->nlast * sizeof *waits);
  /* A loop: on x86_64 Linux, a memcpy call here added 15 ns to the bench's
     floors, which carry Polled's own work only. */
  p->nlast_handles = nhandles < LAST_HANDLES ? nhandles : LAST_HANDLES;
  for (int i = 0; i < p->nlast_handles; i++) p->last_handles[i] = handles[i];
  struct rig_wait *ws = malloc((size_t)(nwaits + 1) * sizeof *waits);
  struct rig_part *ps = malloc((size_t)(nparts + 1) * sizeof *parts);
  if (ws == NULL || ps == NULL) {
    free(ws);
    free(ps);
    unlock(&p->mu);
    *failure = "no memory for the submission";
    return RIG_FAILED;
  }
  struct queued *s = &p->q[p->n++];
  s->v = v;
  s->nwaits = nwaits;
  s->nparts = nparts;
  s->waits = ws;
  s->parts = ps;
  if (nwaits > 0) memcpy(s->waits, waits, (size_t)nwaits * sizeof *waits);
  if (nparts > 0) memcpy(s->parts, parts, (size_t)nparts * sizeof *parts);
  p->held += nparts;
  if (v - atomic_load_explicit(&p->committed, memory_order_relaxed) >= p->lag)
    commit_upto(p, v);
  int committed =
      v <= atomic_load_explicit(&p->committed, memory_order_relaxed);
  unlock(&p->mu);
  return committed ? RIG_COMMITTED : RIG_OK;
}

/* A commit of uncommitted values fails as [rig_test_polled_fail_commit]
   asked, or as the hand-over does once [rig_test_polled_fail_at]'s failure
   began, without counting a call. */
static int polled_commit(void *self, uint64_t v, const char **failure) {
  struct polled *p = self;
  if (v <= atomic_load_explicit(&p->committed, memory_order_acquire))
    return RIG_OK;
  lock(&p->mu);
  if (p->fail_commit) {
    p->fail_commit = 0;
    unlock(&p->mu);
    *failure = "the commit failed";
    return RIG_FAILED;
  }
  if (p->fail_from > 0 && p->steps >= p->fail_from) {
    unlock(&p->mu);
    *failure = p->why;
    return RIG_FAILED;
  }
  commit_upto(p, v);
  unlock(&p->mu);
  return RIG_OK;
}

value rig_test_polled_room(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&polled_room);
}

value rig_test_polled_submit(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&polled_submit);
}

value rig_test_polled_commit(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&polled_commit);
}

value rig_test_polled_word(value v_p) {
  return Val_long((intnat)atomic_load(&Polled_val(v_p)->word));
}

/* Stops the device: drops its queued work, which never runs, and writes the
   last value it received into the word, as a stopped driver does. */
value rig_test_polled_stop(value v_p) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  for (int i = 0; i < p->n; i++) {
    free(p->q[i].waits);
    free(p->q[i].parts);
  }
  p->n = 0;
  p->held = 0;
  atomic_store_explicit(&p->word, p->received, memory_order_release);
  cond_broadcast(&p->cv);
  unlock(&p->mu);
  return Val_unit;
}

/* Writes [v] into the word, as a stopped device's driver does. */
value rig_test_polled_set_word(value v_p, value v) {
  atomic_store(&Polled_val(v_p)->word, (uint64_t)Long_val(v));
  return Val_unit;
}

/* Host memory for regions: on a page, so any of it maps. */
value rig_test_alloc(value v_n) {
  size_t n = (size_t)Long_val(v_n);
  void *a = aligned(page(), n == 0 ? 1 : n);
  return Val_long((intnat)a);
}

value rig_test_free(value v_a) {
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

value rig_test_bump(value unit) {
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

value rig_test_poke(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&poke);
}

/* A fill that takes one from the 64-bit word its argument points at, and
   fails if that makes it zero. */
static int countdown(void *queue, void *arg, uint64_t v) {
  (void)queue;
  (void)v;
  return atomic_fetch_sub((_Atomic uint64_t *)arg, 1) == 1;
}

value rig_test_countdown(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&countdown);
}

/* A fill that copies bytes: its argument's three 64-bit words are the
   destination's address, the source's and the number of bytes. */
static int carry(void *queue, void *arg, uint64_t v) {
  (void)queue;
  (void)v;
  _Atomic uint64_t *a = arg;
  memmove((void *)(uintptr_t)atomic_load(&a[0]),
          (const void *)(uintptr_t)atomic_load(&a[1]),
          (size_t)atomic_load(&a[2]));
  return 0;
}

value rig_test_carry(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&carry);
}

/* Raises SIGINT in the calling thread: the runtime records it, and the
   thread's next poll point runs its handler. */
value rig_test_interrupt(value unit) {
  (void)unit;
  raise(SIGINT);
  return Val_unit;
}

value rig_test_move(value v_dst, value v_src, value v_len) {
  memmove((void *)Long_val(v_dst), (const void *)Long_val(v_src),
          (size_t)Long_val(v_len));
  return Val_unit;
}

value rig_test_load(value v_addr) {
  return Val_long((intnat)atomic_load((_Atomic uint64_t *)Long_val(v_addr)));
}

value rig_test_store(value v_addr, value v) {
  atomic_store((_Atomic uint64_t *)Long_val(v_addr), (uint64_t)Long_val(v));
  return Val_unit;
}

/* The C readers of rig.h, as a caller in C sees a buffer. */
value rig_test_reader_host(value v_b) {
  return Val_long((intnat)rig_buffer_host(v_b));
}

value rig_test_reader_bytes(value v_b) {
  return Val_long((intnat)rig_buffer_bytes(v_b));
}

value rig_test_reader_why(value v_b) {
  CAMLparam1(v_b);
  const char *why = rig_buffer_why(v_b);
  CAMLreturn(why == NULL ? Val_none : caml_alloc_some(caml_copy_string(why)));
}

/* rig_buffer_claim, whose answer is the constructor of Reader.answer of
   the same rank, rig_buffer_wait, which raises what the wait raised, and
   rig_buffer_release. */
value rig_test_reader_claim(value v_b, value v_access) {
  return Val_int(rig_buffer_claim(
      v_b, Int_val(v_access) == 0 ? RIG_READ : RIG_READ_WRITE));
}

value rig_test_reader_wait(value v_b, value v_access) {
  return caml_get_value_or_raise(rig_buffer_wait(
      v_b, Int_val(v_access) == 0 ? RIG_READ : RIG_READ_WRITE));
}

value rig_test_reader_release(value v_b) {
  rig_buffer_release(v_b);
  return Val_unit;
}

/* How many holders share [v_ba]'s storage, as its proxy counts them. */
value rig_test_shares(value v_ba) {
  struct caml_ba_proxy *p = Caml_ba_array_val(v_ba)->proxy;
  return Val_long(p == NULL ? 0 : (intnat)atomic_load(&p->refcount));
}

/* Rig's own: the bytes its host heap keeps for reuse. */
extern intnat rig_heap_kept(void);
extern intnat rig_heap_held(void);

value rig_test_host_held(value unit) {
  (void)unit;
  return Val_long(rig_heap_held());
}

value rig_test_host_kept(value unit) {
  (void)unit;
  return Val_long(rig_heap_kept());
}

/* The bytes the C heap holds allocated, or -1 where its allocator does not
   say. */
value rig_test_heap_bytes(value unit) {
  (void)unit;
#if defined(RIG_TEST_ASAN)
  return Val_long((intnat)__sanitizer_get_current_allocated_bytes());
#elif defined(__APPLE__)
  malloc_statistics_t s;
  malloc_zone_statistics(NULL, &s);
  return Val_long((intnat)s.size_in_use);
#elif defined(__GLIBC__)
  struct mallinfo2 m = mallinfo2();
  return Val_long((intnat)(m.uordblks + m.hblkhd));
#else
  return Val_long(-1);
#endif
}
