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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#ifdef _WIN32
#include <winsock2.h>
#include <malloc.h>
#include <windows.h>
#include <psapi.h>
#else
#include <pthread.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#endif
#include <caml/unixsupport.h>
#include <math.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/syscall.h>
#elif defined(__linux__)
#include <errno.h>
#include <sys/resource.h>
#if defined(__GLIBC__)
#include <linux/futex.h>
#include <sys/syscall.h>
#endif
#endif

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define RIG_TEST_TSAN
#endif
#if __has_feature(address_sanitizer)
#define RIG_TEST_SANITIZER
#endif
#endif
#if defined(__SANITIZE_THREAD__)
#define RIG_TEST_TSAN
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(RIG_TEST_TSAN)
#define RIG_TEST_SANITIZER
#endif

#if defined(RIG_TEST_SANITIZER)
/* The sanitizer's allocator serves malloc and counts its own heap; GCC ships
   the runtime but not <sanitizer/allocator_interface.h>, which declares it. */
size_t __sanitizer_get_current_allocated_bytes(void);
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
static void sleep_ms(uint64_t ms) { Sleep((DWORD)ms); }
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
static void sleep_ms(uint64_t ms) {
  struct timespec t = {(time_t)(ms / 1000), (long)(ms % 1000) * 1000000};
  while (nanosleep(&t, &t) != 0) {
  }
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

/* Empty buffers may have address 0; memmove requires nonnull endpoints even
   when no byte moves. Add offsets as integers before reaching this boundary. */
static void move_bytes(uintptr_t dst, uintptr_t src, size_t bytes) {
  if (bytes != 0) memmove((void *)dst, (const void *)src, bytes);
}

/* A queued submission. Its launches' and fills' blocks are its own
   copies, made at the hand-over with the refs' addresses added: each such
   part's block is an offset into [blocks]. */
/* A function the compiler keeps out of its callers. */
#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

struct queued {
  uint64_t v;
  int nwaits, nparts;
  struct rig_wait *waits;
  struct rig_part *parts;
  uint8_t *blocks;
  uint64_t *times; /* where its run's times go, NULL for nowhere */
};

/* The host clock, rig's (Rig.Profile.now), by which Polled times a value's
   run. */
static uint64_t host_ns(void) {
#if defined(_WIN32)
  LARGE_INTEGER count, frequency;
  QueryPerformanceCounter(&count);
  QueryPerformanceFrequency(&frequency);
  uint64_t c = (uint64_t)count.QuadPart, f = (uint64_t)frequency.QuadPart;
  return c / f * 1000000000u + c % f * 1000000000u / f;
#elif defined(__APPLE__)
  return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
#endif
}

/* Polled's functions: host code run once per group of a launch's grid,
   with the group's index, the block and its parameter count. A Polled
   entry's [launch] points to one. */
struct polled_kernel {
  void (*run)(const uint32_t group[3], const struct rig_block *b,
              uint32_t params);
  uint32_t max_threads, max_shared;
};

#define LAST 8
#define LAST_COPIES 64
#define LAST_PARTS 64
#define LAST_AFTER 8
#define LAST_LAUNCHES 8

/* A launch's block as it ran: its [bytes] bytes of header and parameters,
   a copy of its own. */
struct launched {
  uint32_t bytes;
  uint8_t *block;
};

/* A time pair a submit was given, the value it timed, and whether a settle
   of that value returned. */
struct timed {
  uint64_t *times;
  uint64_t v;
  int settled;
};

/* What a device runs work with, from its make to its stop. */
struct polled_run {
  _Atomic uint64_t committed; /* the last value committed */
  uint64_t lag;               /* the uncommitted values that commit */
  int capacity;      /* parts the queue holds */
  intnat grows; /* the most parts the queue grows to hold; 0 for none */
  int may_block;     /* a full queue's submit waits instead of [room] */
  int fail;          /* the next submit fails */
  int fail_commit;   /* the next commit of uncommitted values fails */
  int fail_settle;   /* the next settle fails */
  int held;          /* parts queued */
  int n, c;
  struct queued *q;
  int blocked; /* submits waiting for room */
  int nlast;   /* the waits of the last submit, the first [LAST] of them */
  struct rig_wait last[LAST];
  uint64_t *last_times; /* where the last submit's times go */
  int ntimed, ctimed;   /* each time pair given, with its last value */
  struct timed *timed;
  int nsides; /* the copy.local of each copy run, the first [LAST_COPIES] */
  int sides[LAST_COPIES];
  int device_memory[LAST_COPIES];
  /* The queue and the first [LAST_AFTER] afters of each part of the last
     submit, its first [LAST_PARTS]. */
  int nlayout;
  int layout_queue[LAST_PARTS], layout_nafter[LAST_PARTS];
  int layout_after[LAST_PARTS][LAST_AFTER];
  uint64_t received; /* the last value a submit received */
  /* The blocks of the first LAST_LAUNCHES launches run since they were
     last read, as they ran: fixed, so a device that launches holds no more
     memory as it runs. */
  int nlaunched;
  struct launched launched[LAST_LAUNCHES];
};

/* A device: its word, the one its open was given ([own] until then), what
   reads it, as its [signaled] counts a fallible call, and its run state,
   NULL once stopped. */
struct polled {
  const struct rig_driver *driver; /* first */
  _Atomic uint64_t *word;
  _Atomic uint64_t own;
  lock_t mu;         /* guards the fields below and [run]'s */
  cond_t cv;
  cond_t work;       /* signalled by a submit to a device that runs itself */
  int itself;        /* a thread of the driver runs the queue */
  _Atomic int submits;
  _Atomic int stops; /* calls of its stop */
  intnat steps;     /* fallible calls made */
  intnat fail_from; /* the step from which every call fails, or 0 */
  intnat refuse_from, refuse_to; /* the steps from one below the other refuse */
  char why[64]; /* what a hand-over failing from [fail_from] reports */
  struct polled_run *run;
  struct polled *prev, *next; /* in the list of every device ([all]) */
  int room_gated, room_waiters;     /* [room] blocks while gated */
  int settle_gated, settle_waiters; /* and [settle] */
  _Atomic int freed; /* rig freed its word of its own */
  /* The mappings rig holds of it ([map_peer], [map_host]), by address. */
  struct held_map {
    uintptr_t at;
    size_t bytes;
  } *maps;
  int nmaps, cmaps;
};

/* A Polled device's state, in a custom block that frees it once collected
   ([polled_finalize]). */
#define Polled_val(v) (*(struct polled **)Data_custom_val(v))

static rig_room_fn polled_room;
static rig_submit_fn polled_submit;
static rig_commit_fn polled_commit;
static rig_settle_fn polled_settle;

static const struct rig_driver polled_driver = {polled_room, polled_submit,
                                                polled_commit, polled_settle};

/* What rig must not do, as Polled sees it: read or free twice a word of
   the driver's own after rig freed it, free a word in its array, hand a
   queue a wait whose word rig gave back, or give a time pair to a value
   before it settled the value the pair timed last. Each is recorded, the
   first BROKEN_KEPT in the order seen ([rig_test_polled_broken]); each but
   the last is raised as a Failure where the call that saw it returns to
   OCaml. */
#define BROKE_READ_FREED 0
#define BROKE_FREED_TWICE 1
#define BROKE_GAVE_BACK 2
#define BROKE_FREED_ARRAY 3
#define BROKE_UNSETTLED 4
#define BROKEN_KEPT 64

static lock_t broken_mu = LOCK_INITIALIZER;
static int broken[BROKEN_KEPT], nbroken;

static void broke(int what) {
  lock(&broken_mu);
  if (nbroken < BROKEN_KEPT) broken[nbroken++] = what;
  unlock(&broken_mu);
}

value rig_test_polled_broke(value v_what) {
  broke(Int_val(v_what));
  return Val_unit;
}

/* What was broken, oldest first. */
value rig_test_polled_broken(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(a);
  int kept[BROKEN_KEPT], n;
  lock(&broken_mu);
  n = nbroken;
  memcpy(kept, broken, sizeof kept);
  unlock(&broken_mu);
  a = caml_alloc_tuple((mlsize_t)n);
  for (int i = 0; i < n; i++) Store_field(a, i, Val_int(kept[i]));
  CAMLreturn(a);
}

/* Every Polled device not yet collected, linked through the devices so
   that enrolling allocates nothing: a sleep runs the ones whose words the
   sleeper's work waits for. */
static lock_t all_mu = LOCK_INITIALIZER;
static struct polled *all;

static void enrol(struct polled *p) {
  lock(&all_mu);
  p->prev = NULL;
  p->next = all;
  if (all != NULL) all->prev = p;
  all = p;
  unlock(&all_mu);
}

/* The Polled device whose word is at [at], or NULL: the last one given
   that word, since a word passes from a stopped device to the next. */
static struct polled *enrolled(uintptr_t at) {
  lock(&all_mu);
  struct polled *found = all;
  while (found != NULL && (uintptr_t)found->word != at) found = found->next;
  unlock(&all_mu);
  return found;
}

static void unenrol(struct polled *p) {
  lock(&all_mu);
  if (p->prev != NULL) p->prev->next = p->next;
  else all = p->next;
  if (p->next != NULL) p->next->prev = p->prev;
  unlock(&all_mu);
}

/* Frees [r] with its queued work, which never runs. */
static void free_run(struct polled_run *r) {
  if (r == NULL) return;
  for (int i = 0; i < r->n; i++) {
    free(r->q[i].waits);
    free(r->q[i].parts);
    free(r->q[i].blocks);
  }
  free(r->q);
  free(r->timed);
  for (int i = 0; i < r->nlaunched; i++) free(r->launched[i].block);
  free(r);
}

/* Frees a collected device's state. A device that runs itself keeps it for
   its thread. rig holds the handle of a device it opened, and so the block,
   for the process. */
static void polled_finalize(value v) {
  struct polled *p = Polled_val(v);
  if (p->itself) return;
  unenrol(p);
  free_run(p->run);
  free(p->maps);
#ifndef _WIN32
  pthread_mutex_destroy(&p->mu);
  pthread_cond_destroy(&p->cv);
  pthread_cond_destroy(&p->work);
#endif
  free(p);
}

static struct custom_operations polled_ops = {
    "rig_test.polled",        polled_finalize,
    custom_compare_default,   custom_hash_default,
    custom_serialize_default, custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

value rig_test_polled_new(value v_capacity, value v_may_block,
                          value v_lag) {
  CAMLparam3(v_capacity, v_may_block, v_lag);
  CAMLlocal1(v);
  struct polled *p = calloc(1, sizeof *p);
  struct polled_run *r = calloc(1, sizeof *r);
  if (p == NULL || r == NULL) {
    free(p);
    free(r);
    caml_raise_out_of_memory();
  }
  p->driver = &polled_driver;
  lock_init(&p->mu);
  cond_init(&p->cv);
  cond_init(&p->work);
  r->capacity = Int_val(v_capacity);
  r->may_block = Bool_val(v_may_block);
  r->lag = (uint64_t)Long_val(v_lag);
  p->run = r;
  p->word = &p->own;
  enrol(p);
  v = caml_alloc_custom(&polled_ops, sizeof p, 0, 1);
  Polled_val(v) = p;
  CAMLreturn(v);
}

/* Makes the device's [room] answer RIG_GROW for more parts than its queue
   holds, and its queue grow to hold up to [v_most] parts. */
value rig_test_polled_grows(value v_p, value v_most) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  p->run->grows = Long_val(v_most);
  unlock(&p->mu);
  return Val_unit;
}

/* The parts of the submission whose [room] last answered RIG_GROW on this
   thread, and the device it was for. */
static _Thread_local struct {
  const struct polled *p;
  int parts;
} asked;

/* Grows the device's queue to hold the parts of the submission whose [room]
   answered RIG_GROW on this thread, unless [v_refused] or they are more than
   it grows to: [false] then. */
value rig_test_polled_grow(value v_p, value v_refused) {
  struct polled *p = Polled_val(v_p);
  int parts = asked.p == p ? asked.parts : 0;
  asked.p = NULL;
  lock(&p->mu);
  struct polled_run *r = p->run;
  int grew = r != NULL && !Bool_val(v_refused) && parts <= r->grows;
  if (grew && parts > r->capacity) r->capacity = parts;
  unlock(&p->mu);
  return Val_bool(grew);
}

/* The address of the device's C state, its edge. */
value rig_test_polled_edge(value v_p) {
  return caml_copy_nativeint((intnat)Polled_val(v_p));
}

value rig_test_polled_fail(value v_p) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  if (p->run != NULL) p->run->fail = 1;
  unlock(&p->mu);
  return Val_unit;
}

value rig_test_polled_fail_commit(value v_p) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  if (p->run != NULL) p->run->fail_commit = 1;
  unlock(&p->mu);
  return Val_unit;
}

value rig_test_polled_fail_settle(value v_p) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  if (p->run != NULL) p->run->fail_settle = 1;
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

/* Counts a call of the device's stop. Neither allocates nor polls, so an
   exception that the stop's first allocation raises comes after it. */
value rig_test_polled_stopping(value v_p) {
  atomic_fetch_add(&Polled_val(v_p)->stops, 1);
  return Val_unit;
}

value rig_test_polled_stops(value v_p) {
  return Val_int(atomic_load(&Polled_val(v_p)->stops));
}

value rig_test_polled_blocked(value v_p) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  int n = p->run != NULL ? p->run->blocked : 0;
  unlock(&p->mu);
  return Val_int(n);
}

value rig_test_polled_last_times(value v_p) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  intnat at = p->run != NULL ? (intnat)p->run->last_times : 0;
  unlock(&p->mu);
  return Val_long(at);
}

/* The waits of the last submit, as [| at; value; … |]. */
value rig_test_polled_last_waits(value v_p) {
  CAMLparam1(v_p);
  CAMLlocal1(a);
  struct polled *p = Polled_val(v_p);
  struct rig_wait w[LAST];
  int n = 0;
  lock(&p->mu);
  if (p->run != NULL) {
    n = p->run->nlast;
    memcpy(w, p->run->last, sizeof w);
  }
  unlock(&p->mu);
  a = caml_alloc_tuple(2 * (mlsize_t)n);
  for (int i = 0; i < n; i++) {
    Store_field(a, 2 * i, Val_long((intnat)w[i].at));
    Store_field(a, 2 * i + 1, Val_long((intnat)w[i].value));
  }
  CAMLreturn(a);
}

/* The last submit's parts as [layout] states them: per part, its queue,
   its number of afters, then those afters. */
value rig_test_polled_layout(value v_p) {
  CAMLparam1(v_p);
  CAMLlocal1(a);
  struct polled *p = Polled_val(v_p);
  int words[LAST_PARTS * (2 + LAST_AFTER)], n = 0;
  lock(&p->mu);
  const struct polled_run *r = p->run;
  for (int i = 0; r != NULL && i < r->nlayout; i++) {
    words[n++] = r->layout_queue[i];
    words[n++] = r->layout_nafter[i];
    for (int j = 0; j < r->layout_nafter[i]; j++)
      words[n++] = r->layout_after[i][j];
  }
  unlock(&p->mu);
  a = caml_alloc_tuple((mlsize_t)n);
  for (int i = 0; i < n; i++) Store_field(a, i, Val_int(words[i]));
  CAMLreturn(a);
}

/* The copy facts of each copy part [p] ran, in order. */
value rig_test_polled_copy_facts(value v_p) {
  CAMLparam1(v_p);
  CAMLlocal2(a, facts);
  struct polled *p = Polled_val(v_p);
  int s[LAST_COPIES], m[LAST_COPIES], n = 0;
  lock(&p->mu);
  if (p->run != NULL) {
    n = p->run->nsides;
    memcpy(s, p->run->sides, sizeof s);
    memcpy(m, p->run->device_memory, sizeof m);
  }
  unlock(&p->mu);
  a = caml_alloc_tuple((mlsize_t)n);
  for (int i = 0; i < n; i++) {
    facts = caml_alloc_tuple(2);
    Store_field(facts, 0, Val_int(s[i]));
    Store_field(facts, 1, Val_bool(m[i]));
    Store_field(a, i, facts);
  }
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

value rig_test_polled_queued(value v_p) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  int n = p->run != NULL ? p->run->n : 0;
  unlock(&p->mu);
  return Val_int(n);
}

static int wait_holds(const struct rig_wait *wait) {
  return atomic_load((_Atomic uint64_t *)(uintptr_t)wait->at) >= wait->value;
}

/* What [p]'s queue breaks by reading the word of the wait [wait], or -1:
   a word outside rig's array, which lives for the process, that is a word
   of a Polled device's own that rig freed, or that no mapping [p] holds
   covers. A mapping of another device's word lies at that word's address,
   so a mapping rig gave back is seen only once [p] holds none of that
   word. [p]'s lock is held. */
static int wait_breaks(struct polled *p, const struct rig_wait *wait) {
  uintptr_t at = (uintptr_t)wait->at, words = (uintptr_t)rig_words();
  if (at >= words && at < words + RIG_WORDS_BYTES) return -1;
  struct polled *q = enrolled(at);
  if (q != NULL && atomic_load(&q->freed)) return BROKE_READ_FREED;
  for (int i = 0; i < p->nmaps; i++)
    if (at >= p->maps[i].at && at < p->maps[i].at + p->maps[i].bytes)
      return -1;
  return BROKE_GAVE_BACK;
}

/* Whether [s]'s waits hold, counting in [*broken] the waits that break
   rig's contract, which are still read: Polled's words are its own. */
static int waits_hold(struct polled *p, struct queued *s, int *broken) {
  for (int i = 0; i < s->nwaits; i++) {
    int b = wait_breaks(p, &s->waits[i]);
    if (b >= 0) {
      broke(b);
      (*broken)++;
    }
    if (!wait_holds(&s->waits[i])) return 0;
  }
  return 1;
}

static int committed(struct polled_run *r, struct queued *s) {
  return s->v <= atomic_load_explicit(&r->committed, memory_order_relaxed);
}

/* Commits [p]'s values up to [v]. [p]'s lock is held. */
static void commit_upto(struct polled *p, uint64_t v) {
  struct polled_run *r = p->run;
  if (v <= atomic_load_explicit(&r->committed, memory_order_relaxed)) return;
  atomic_store_explicit(&r->committed, v, memory_order_release);
  if (p->itself) cond_broadcast(&p->work);
}

/* Keeps the [n] bytes of the block [b] as run, among the first
   LAST_LAUNCHES, unless memory lacks for a copy. [p]'s lock is held. */
static void keep_launch(struct polled_run *r, const void *b, uint32_t n) {
  if (r->nlaunched == LAST_LAUNCHES) return;
  uint8_t *block = malloc(n);
  if (block == NULL) return;
  memcpy(block, b, n);
  r->launched[r->nlaunched++] = (struct launched){n, block};
}

/* Runs the launch [part] of [s]: its function once per group, x fastest.
   Kept out of [run_one], whose fills and copies the floors time. */
NOINLINE static void launch(struct polled_run *r, struct queued *s,
                   const struct rig_part *part) {
  const struct rig_block *b =
      (const struct rig_block *)(s->blocks + part->launch.args.block);
  const struct polled_kernel *k = part->launch.launch;
  uint32_t g[3];
  for (g[2] = 0; g[2] < b->groups[2]; g[2]++)
    for (g[1] = 0; g[1] < b->groups[1]; g[1]++)
      for (g[0] = 0; g[0] < b->groups[0]; g[0]++)
        k->run(g, b, part->launch.args.params);
  keep_launch(r, b, (uint32_t)sizeof *b + part->launch.args.params);
}

#if defined(__clang__)
/* Queued fills can be loaded code without Clang's function-type metadata. */
__attribute__((no_sanitize("function")))
#endif
static void run_one(struct polled *p, struct queued *s) {
  struct polled_run *r = p->run;
  if (s->times != NULL) s->times[0] = host_ns();
  for (int i = 0; i < s->nparts; i++) {
    struct rig_part *part = &s->parts[i];
    if (part->kind == RIG_FILL) {
      rig_code_sync();
      part->fill.fn(NULL,
                    s->blocks + part->fill.args.block + sizeof(struct rig_block),
                    s->v);
    } else if (part->kind == RIG_LAUNCH)
      launch(r, s, part);
    else if (part->kind == RIG_COPY) {
      /* A side that copy.local names holds a host address, as Polled's own
         handles do. */
      move_bytes((uintptr_t)part->copy.dst + part->copy.dst_offset,
                 (uintptr_t)part->copy.src + part->copy.src_offset,
                 (size_t)part->copy.bytes);
      if (r->nsides < LAST_COPIES) {
        r->sides[r->nsides] = part->copy.local;
        r->device_memory[r->nsides++] = part->copy.device_memory;
      }
    }
  }
  if (s->times != NULL) s->times[1] = host_ns();
  r->held -= s->nparts;
  free(s->waits);
  free(s->parts);
  if (s->blocks != NULL) free(s->blocks);
  atomic_store_explicit(p->word, s->v, memory_order_release);
}

/* Runs the queued submissions whose waits hold, in order: answers how
   many ran, none once stopped, and counts in [*broken] the waits that broke
   rig's contract ([wait_breaks]). */
static int run(struct polled *p, int *broken) {
  lock(&p->mu);
  struct polled_run *r = p->run;
  int k = 0;
  while (r != NULL && k < r->n && committed(r, &r->q[k]) &&
         waits_hold(p, &r->q[k], broken))
    run_one(p, &r->q[k++]);
  if (k > 0) {
    memmove(r->q, r->q + k, (size_t)(r->n - k) * sizeof *r->q);
    r->n -= k;
  }
  cond_broadcast(&p->cv);
  unlock(&p->mu);
  return k;
}

static void raise_broken(int broken) {
  if (broken > 0) caml_failwith("Polled: a wait read a word rig gave back");
}

value rig_test_polled_run(value v_p) {
  int broken = 0;
  int k = run(Polled_val(v_p), &broken);
  raise_broken(broken);
  return Val_int(k);
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
static int drive(struct polled *p, int depth, int *broken) {
  struct polled *wanted[LAST];
  int n = 0, stuck = 0;
  lock(&p->mu);
  struct polled_run *r = p->run;
  if (r != NULL && r->n > 0) {
    struct queued *s = &r->q[0];
    stuck = !committed(r, s);
    for (int i = 0; i < s->nwaits && n < LAST && depth > 0; i++) {
      struct polled *q = enrolled((uintptr_t)s->waits[i].at);
      if (q != NULL && !wait_holds(&s->waits[i])) wanted[n++] = q;
    }
  }
  unlock(&p->mu);
  for (int i = 0; i < n; i++)
    if (wanted[i] != p && drive(wanted[i], depth - 1, broken) == STUCK)
      stuck = 1;
  int k = run(p, broken);
  return k == 0 && stuck ? STUCK : k;
}

value rig_test_polled_drive(value v_p) {
  int broken = 0;
  int k = drive(Polled_val(v_p), DRIVE_DEPTH, &broken);
  raise_broken(broken);
  return Val_int(k);
}

/* A device that runs itself: its thread runs the queue as work arrives and,
   while the first submission waits on a word that has not moved, again after
   a nap, as a device runs its own work. It ends at the stop. */
static void engine(struct polled *p) {
  for (;;) {
    lock(&p->mu);
    struct polled_run *r;
    while ((r = p->run) != NULL && (r->n == 0 || !committed(r, &r->q[0])))
      cond_wait(&p->work, &p->mu);
    unlock(&p->mu);
    if (r == NULL) return;
    int broken = 0;
    if (run(p, &broken) == 0) nap();
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

/* Polled's limits on a grid, as CUDA's: 2^31 - 1 groups along x, 65535
   along y and z. */
#define MAX_GROUPS_X 0x7fffffffu
#define MAX_GROUPS_YZ 65535u

/* Whether the launch [part], whose block lies in [args], has a grid and
   groups Polled runs. */
static int launches(const struct rig_part *part, const uint8_t *args) {
  const struct rig_block *b =
      (const struct rig_block *)(args + part->launch.args.block);
  const struct polled_kernel *k = part->launch.launch;
  uint64_t threads = 1;
  for (int a = 0; a < 3; a++) {
    if (b->groups[a] == 0 || b->threads[a] == 0) return 0;
    threads *= b->threads[a];
  }
  return b->groups[0] <= MAX_GROUPS_X && b->groups[1] <= MAX_GROUPS_YZ &&
         b->groups[2] <= MAX_GROUPS_YZ && threads <= k->max_threads &&
         b->shared <= k->max_shared;
}

/* Whether [scratch] holds RIG_SCRATCH bytes, aligned to 8, for each of the
   [n] parts, which it then overwrites, as a driver may. */
static int scribble(void *scratch, int n) {
  if (n == 0) return 1;
  if (scratch == NULL || (uintptr_t)scratch % 8 != 0) return 0;
  memset(scratch, 0xa5, (size_t)n * RIG_SCRATCH);
  return 1;
}

static int polled_room(void *self, const struct rig_part *parts, int n,
                       void *scratch, const uint8_t *args) {
  struct polled *p = self;
  struct polled_run *r = p->run;
  if (!scribble(scratch, n)) return RIG_NEVER;
  for (int i = 0; i < n; i++) {
    int kind = parts[i].kind;
    if (kind == RIG_LAUNCH ? !launches(&parts[i], args)
                           : kind != RIG_FILL && kind != RIG_COPY &&
                                 kind != RIG_WORDS)
      return RIG_NEVER;
  }
  if (n > r->capacity) {
    if (r->grows == 0) return RIG_NEVER;
    asked.p = p;
    asked.parts = n;
    return RIG_GROW;
  }
  if (r->may_block) {
    lock(&p->mu);
    p->room_waiters++;
    while (p->room_gated) cond_wait(&p->cv, &p->mu);
    p->room_waiters--;
    unlock(&p->mu);
    return RIG_FITS;
  }
  lock(&p->mu);
  int full = r->held + n > r->capacity;
  unlock(&p->mu);
  return full ? RIG_LATER : RIG_FITS;
}

/* The parameters of a launch or a fill, if [p] is either. */
static const struct rig_args *args_of(const struct rig_part *p) {
  if (p->kind == RIG_LAUNCH) return &p->launch.args;
  if (p->kind == RIG_FILL) return &p->fill.args;
  return NULL;
}

/* Sets [*copy] to the blocks of [parts]' launches and fills up to the end
   of the last one, copied from [args] with each ref's address added, or
   NULL for parts without blocks: the caller may store into [args] once
   the hand-over returns. Whether memory sufficed. */
static int blocks(const struct rig_part *parts, int nparts,
                  const uint8_t *args, const uint64_t *slots, uint8_t **copy) {
  size_t n = 0;
  for (int i = 0; i < nparts; i++) {
    const struct rig_args *a = args_of(&parts[i]);
    if (a == NULL) continue;
    size_t end = a->block + sizeof(struct rig_block) + a->params;
    if (end > n) n = end;
  }
  if (n == 0) return 1;
  uint8_t *b = malloc(n);
  if (b == NULL) return 0;
  memcpy(b, args, n);
  for (int i = 0; i < nparts; i++) {
    const struct rig_args *a = args_of(&parts[i]);
    if (a == NULL) continue;
    rig_params(b + a->block + sizeof(struct rig_block), a, args, slots);
  }
  *copy = b;
  return 1;
}

/* Records that [r]'s value [v] was given the time pair [times], a breach if
   the value the pair timed last has not settled. Is whether it found the
   memory for the record. */
static int give_pair(struct polled_run *r, uint64_t *times, uint64_t v) {
  for (int i = 0; i < r->ntimed; i++) {
    struct timed *t = &r->timed[i];
    if (t->times != times) continue;
    if (!t->settled) broke(BROKE_UNSETTLED);
    *t = (struct timed){times, v, 0};
    return 1;
  }
  if (r->ntimed == r->ctimed) {
    int c = r->ctimed == 0 ? 8 : 2 * r->ctimed;
    struct timed *t = realloc(r->timed, (size_t)c * sizeof *t);
    if (t == NULL) return 0;
    r->timed = t;
    r->ctimed = c;
  }
  r->timed[r->ntimed++] = (struct timed){times, v, 0};
  return 1;
}

static int polled_submit(void *self, uint64_t v, const struct rig_wait *waits,
                         int nwaits, const struct rig_part *parts, int nparts,
                         void *scratch, uint8_t *args,
                         const uint64_t *slots, int nslots, uint64_t *times,
                         const char **failure) {
  (void)nslots;
  struct polled *p = self;
  struct polled_run *r = p->run;
  if (!scribble(scratch, nparts)) {
    *failure = "no scratch for the parts";
    return RIG_FAILED;
  }
  atomic_fetch_add(&p->submits, 1);
  lock(&p->mu);
  r->received = v;
  if (step(p) == STEP_FAIL) {
    unlock(&p->mu);
    *failure = p->why;
    return RIG_FAILED;
  }
  if (r->fail) {
    r->fail = 0;
    unlock(&p->mu);
    *failure = "the submission failed";
    return RIG_FAILED;
  }
  while (r->may_block && r->held + nparts > r->capacity) {
    commit_upto(p, v - 1);
    r->blocked++;
    cond_wait(&p->cv, &p->mu);
    r->blocked--;
  }
  if (r->n == r->c) {
    int c = r->c == 0 ? 8 : 2 * r->c;
    struct queued *q = realloc(r->q, (size_t)c * sizeof *q);
    if (q == NULL) {
      unlock(&p->mu);
      *failure = "no memory for the queue";
      return RIG_FAILED;
    }
    r->q = q;
    r->c = c;
  }
  r->nlast = nwaits < LAST ? nwaits : LAST;
  r->nlayout = nparts < LAST_PARTS ? nparts : LAST_PARTS;
  for (int i = 0; i < r->nlayout; i++) {
    r->layout_queue[i] = parts[i].queue;
    r->layout_nafter[i] =
        parts[i].nafter < LAST_AFTER ? parts[i].nafter : LAST_AFTER;
    for (int j = 0; j < r->layout_nafter[i]; j++)
      r->layout_after[i][j] = parts[i].after[j];
  }
  if (r->nlast > 0) memcpy(r->last, waits, (size_t)r->nlast * sizeof *waits);
  r->last_times = times;
  if (times != NULL && !give_pair(r, times, v)) {
    unlock(&p->mu);
    *failure = "no memory for the time pairs";
    return RIG_FAILED;
  }
  struct rig_wait *ws = malloc((size_t)(nwaits + 1) * sizeof *waits);
  struct rig_part *ps = malloc((size_t)(nparts + 1) * sizeof *parts);
  uint8_t *bs = NULL;
  int ok = ws != NULL && ps != NULL && blocks(parts, nparts, args, slots, &bs);
  if (!ok) {
    free(ws);
    free(ps);
    unlock(&p->mu);
    *failure = "no memory for the submission";
    return RIG_FAILED;
  }
  struct queued *s = &r->q[r->n++];
  s->v = v;
  s->nwaits = nwaits;
  s->nparts = nparts;
  s->waits = ws;
  s->parts = ps;
  s->blocks = bs;
  s->times = times;
  if (nwaits > 0) memcpy(s->waits, waits, (size_t)nwaits * sizeof *waits);
  if (nparts > 0) memcpy(s->parts, parts, (size_t)nparts * sizeof *parts);
  r->held += nparts;
  if (v - atomic_load_explicit(&r->committed, memory_order_relaxed) >= r->lag)
    commit_upto(p, v);
  int committed =
      v <= atomic_load_explicit(&r->committed, memory_order_relaxed);
  unlock(&p->mu);
  return committed ? RIG_COMMITTED : RIG_OK;
}

/* A commit of uncommitted values fails as [rig_test_polled_fail_commit]
   asked, or as the hand-over does once [rig_test_polled_fail_at]'s failure
   began, without counting a call. */
static int polled_commit(void *self, uint64_t v, const char **failure) {
  struct polled *p = self;
  struct polled_run *r = p->run;
  if (v <= atomic_load_explicit(&r->committed, memory_order_acquire))
    return RIG_OK;
  lock(&p->mu);
  if (r->fail_commit) {
    r->fail_commit = 0;
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

/* A settle writes nothing, since the queue's run writes a value's times
   before its word shows it, and records [v] settled, or fails as
   [rig_test_polled_fail_settle] asked. It waits while gated. */
static int polled_settle(void *self, uint64_t v, const char **failure) {
  struct polled *p = self;
  lock(&p->mu);
  p->settle_waiters++;
  while (p->settle_gated) cond_wait(&p->cv, &p->mu);
  p->settle_waiters--;
  struct polled_run *r = p->run;
  int fail = r->fail_settle;
  r->fail_settle = 0;
  for (int i = 0; i < r->ntimed && !fail; i++)
    if (r->timed[i].v == v) r->timed[i].settled = 1;
  unlock(&p->mu);
  if (!fail) return RIG_OK;
  *failure = "the settle failed";
  return RIG_FAILED;
}

/* The host address of the device [v_p]'s word. */
value rig_test_polled_word_at(value v_p) {
  return Val_long((intnat)Polled_val(v_p)->word);
}

value rig_test_polled_word(value v_p) {
  return Val_long((intnat)atomic_load(Polled_val(v_p)->word));
}

/* Makes the 8 bytes at [v_word] the device's word, or, if [v_own], its own
   word with their value, first raising them to [v_start] if they hold less,
   and starts its values after theirs, as an opener's make does with the word
   rig gives it. The device moves to the front of [all], so that a sleep
   finds it by its word. */
value rig_test_polled_bind(value v_p, value v_word, value v_start,
                           value v_own) {
  struct polled *p = Polled_val(v_p);
  _Atomic uint64_t *w = (_Atomic uint64_t *)Nativeint_val(v_word);
  rig_raise(w, (uint64_t)Long_val(v_start));
  uint64_t base = atomic_load(w);
  if (Bool_val(v_own)) {
    atomic_store(&p->own, base);
    w = &p->own;
  }
  unenrol(p);
  lock(&p->mu);
  p->word = w;
  if (p->run != NULL) {
    p->run->received = base;
    atomic_store(&p->run->committed, base);
  }
  unlock(&p->mu);
  enrol(p);
  return Val_unit;
}

/* Stops the device: frees its run state with its queued work, which never
   runs, and writes the last value it received into the word, as a stopped
   driver does. A device that runs itself ends its thread. */
value rig_test_polled_stop(value v_p) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  struct polled_run *r = p->run;
  p->run = NULL;
  if (r != NULL)
    atomic_store_explicit(p->word, r->received, memory_order_release);
  cond_broadcast(&p->cv);
  cond_broadcast(&p->work);
  unlock(&p->mu);
  free_run(r);
  return Val_unit;
}

/* Makes the device's [room] block, if [v_gated], until it is called with
   false. */
value rig_test_polled_gate_room(value v_p, value v_gated) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  p->room_gated = Bool_val(v_gated);
  cond_broadcast(&p->cv);
  unlock(&p->mu);
  return Val_unit;
}

/* Makes the device's [settle] block, if [v_gated], until it is called with
   false. */
value rig_test_polled_gate_settle(value v_p, value v_gated) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  p->settle_gated = Bool_val(v_gated);
  cond_broadcast(&p->cv);
  unlock(&p->mu);
  return Val_unit;
}

value rig_test_polled_settle_waiters(value v_p) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  int n = p->settle_waiters;
  unlock(&p->mu);
  return Val_int(n);
}

value rig_test_polled_room_waiters(value v_p) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  int n = p->room_waiters;
  unlock(&p->mu);
  return Val_int(n);
}

/* Notes that rig freed the device's word: what that broke, or -1 for a
   word outside rig's array freed once. */
value rig_test_polled_free_word(value v_p) {
  struct polled *p = Polled_val(v_p);
  uintptr_t at = (uintptr_t)p->word, words = (uintptr_t)rig_words();
  if (at >= words && at < words + RIG_WORDS_BYTES)
    return Val_int(BROKE_FREED_ARRAY);
  return Val_int(atomic_exchange(&p->freed, 1) == 0 ? -1 : BROKE_FREED_TWICE);
}

/* Notes a mapping rig holds of the [v_n] bytes at [v_at]. */
value rig_test_polled_mapped(value v_p, value v_at, value v_n) {
  struct polled *p = Polled_val(v_p);
  lock(&p->mu);
  if (p->nmaps == p->cmaps) {
    int c = p->cmaps == 0 ? 8 : 2 * p->cmaps;
    struct held_map *m = realloc(p->maps, (size_t)c * sizeof *m);
    if (m == NULL) {
      unlock(&p->mu);
      caml_raise_out_of_memory();
    }
    p->maps = m;
    p->cmaps = c;
  }
  p->maps[p->nmaps].at = (uintptr_t)Long_val(v_at);
  p->maps[p->nmaps++].bytes = (size_t)Long_val(v_n);
  unlock(&p->mu);
  return Val_unit;
}

/* Notes that rig gave back a mapping of the [v_n] bytes at [v_at]. */
value rig_test_polled_unmapped(value v_p, value v_at, value v_n) {
  struct polled *p = Polled_val(v_p);
  uintptr_t at = (uintptr_t)Long_val(v_at);
  size_t n = (size_t)Long_val(v_n);
  lock(&p->mu);
  for (int i = 0; i < p->nmaps; i++)
    if (p->maps[i].at == at && p->maps[i].bytes == n) {
      p->maps[i] = p->maps[--p->nmaps];
      break;
    }
  unlock(&p->mu);
  return Val_unit;
}

/* Writes [v] into the word, as a stopped device's driver does. */
value rig_test_polled_set_word(value v_p, value v) {
  atomic_store(Polled_val(v_p)->word, (uint64_t)Long_val(v));
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

/* A fill whose queue occupancy alone matters. */
static int noop(void *queue, void *arg, uint64_t v) {
  (void)queue;
  (void)arg;
  (void)v;
  return 0;
}

value rig_test_noop(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&noop);
}

/* A fill that adds 1 to the 64-bit word whose address its argument holds. */
static int bump(void *queue, void *arg, uint64_t v) {
  (void)queue;
  (void)v;
  uint64_t at;
  memcpy(&at, arg, sizeof at);
  _Atomic uint64_t *w = (_Atomic uint64_t *)(uintptr_t)at;
  atomic_fetch_add(w, 1);
  return 0;
}

value rig_test_bump(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&bump);
}

/* A fill that sets the second of the two 64-bit words at the address its
   argument holds, then returns once the first is not zero, or after
   [waiting_ticks] ticks of 1 ms. */
static const int waiting_ticks = 30000;

static int waiting(void *queue, void *arg, uint64_t v) {
  (void)queue;
  (void)v;
  uint64_t at;
  memcpy(&at, arg, sizeof at);
  _Atomic uint64_t *w = (_Atomic uint64_t *)(uintptr_t)at;
  atomic_store(&w[1], 1);
  for (int left = waiting_ticks; left > 0 && atomic_load(&w[0]) == 0; left--)
    sleep_ms(1);
  return 0;
}

value rig_test_waiting(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&waiting);
}

/* A fill that stores the second 64-bit word of its argument at the address
   the first holds. */
static int poke(void *queue, void *arg, uint64_t v) {
  (void)queue;
  (void)v;
  uint64_t a[2];
  memcpy(a, arg, sizeof a);
  atomic_store((_Atomic uint64_t *)(uintptr_t)a[0], a[1]);
  return 0;
}

value rig_test_poke(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&poke);
}

/* A fill of 8,192 bytes of argument that stores the 64-bit word at byte
   4,096 at the address the last 8 bytes hold. */
static int far(void *queue, void *arg, uint64_t v) {
  (void)queue;
  (void)v;
  uint64_t x, at;
  memcpy(&x, (const uint8_t *)arg + 4096, sizeof x);
  memcpy(&at, (const uint8_t *)arg + 8184, sizeof at);
  atomic_store((_Atomic uint64_t *)(uintptr_t)at, x);
  return 0;
}

value rig_test_far(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&far);
}

/* The parked thread: one that blocks outside OCaml, holding no runtime lock,
   until it is unparked or [park_ticks] ticks of 10 ms pass. */
static const int park_ticks = 500;
static atomic_int parked, unparked;

value rig_test_park(value unit) {
  (void)unit;
  caml_release_runtime_system();
  atomic_store(&unparked, 0);
  atomic_store(&parked, 1);
  for (int left = park_ticks; left > 0 && !atomic_load(&unparked); left--)
    sleep_ms(10);
  atomic_store(&parked, 0);
  caml_acquire_runtime_system();
  return Val_unit;
}

value rig_test_parked(value unit) {
  (void)unit;
  return Val_bool(atomic_load(&parked));
}

value rig_test_unpark(value unit) {
  (void)unit;
  atomic_store(&unparked, 1);
  return Val_unit;
}

/* Whether a thread of this process blocks in a condition wait, or with
   [mutex] in a mutex's, from the system's view of it: the system call it
   entered is the one such waits use, __psynch_cvwait or __psynch_mutexwait
   on macOS, and with glibc a futex wait on a bitset, which no glibc mutex
   uses, or a plain futex wait. */

#if defined(__APPLE__)

typedef mach_port_t thread_id;

static thread_id self(void) { return pthread_mach_thread_np(pthread_self()); }

static int sees_waits(void) { return 1; }

/* The system call a suspended thread entered: the number its stub put in x16
   (arm64) or rax (x86_64). */
static int in_wait(thread_id t, int mutex) {
  if (thread_suspend(t) != KERN_SUCCESS) return 0;
  thread_basic_info_data_t info;
  mach_msg_type_number_t n = THREAD_BASIC_INFO_COUNT;
  int waiting = thread_info(t, THREAD_BASIC_INFO, (thread_info_t)&info, &n) ==
                    KERN_SUCCESS &&
                info.run_state == TH_STATE_WAITING;
  long call = -1;
#if defined(__arm64__)
  arm_thread_state64_t s;
  n = ARM_THREAD_STATE64_COUNT;
  if (thread_get_state(t, ARM_THREAD_STATE64, (thread_state_t)&s, &n) ==
      KERN_SUCCESS)
    call = (long)s.__x[16];
#elif defined(__x86_64__)
  x86_thread_state64_t s;
  n = x86_THREAD_STATE64_COUNT;
  if (thread_get_state(t, x86_THREAD_STATE64, (thread_state_t)&s, &n) ==
      KERN_SUCCESS)
    call = (long)(s.__rax & 0xffffff);
#endif
  thread_resume(t);
  return waiting && call == (mutex ? SYS_psynch_mutexwait : SYS_psynch_cvwait);
}

#elif defined(__linux__) && defined(__GLIBC__)

typedef pid_t thread_id;

static thread_id self(void) { return (pid_t)syscall(SYS_gettid); }

/* The thread's syscall file: the call it is blocked in and its arguments,
   or "running". */
static FILE *syscall_file(thread_id t) {
  char path[64];
  snprintf(path, sizeof path, "/proc/self/task/%d/syscall", (int)t);
  return fopen(path, "r");
}

static int sees_waits(void) {
  FILE *f = syscall_file(self());
  if (f == NULL) return 0;
  fclose(f);
  return 1;
}

static int in_wait(thread_id t, int mutex) {
  FILE *f = syscall_file(t);
  if (f == NULL) return 0;
  long call = -1;
  unsigned long word, op;
  int got = fscanf(f, "%ld %lx %lx", &call, &word, &op);
  fclose(f);
  return got == 3 && call == SYS_futex &&
         (op & FUTEX_CMD_MASK) == (mutex ? FUTEX_WAIT : FUTEX_WAIT_BITSET);
}

#endif

/* The interrupter: a thread of no OCaml code that raises SIGINT once the
   thread it watches blocks in a condition wait, as a signal from outside the
   process does, and stops when told. The handler runs on the next thread
   that returns to OCaml. */

#if defined(__APPLE__) || (defined(__linux__) && defined(__GLIBC__))

static struct {
  pthread_t thread;
  thread_id watched;
  atomic_int stop;
  atomic_int raised;
} interrupter;

static void *interrupter_main(void *p) {
  (void)p;
  while (!atomic_load(&interrupter.stop)) {
    if (in_wait(interrupter.watched, 0)) {
      atomic_store(&interrupter.raised, 1);
      raise(SIGINT);
      return NULL;
    }
    sleep_ms(1);
  }
  return NULL;
}

value rig_test_sees_waits(value unit) {
  (void)unit;
  return Val_bool(sees_waits());
}

value rig_test_thread_self(value unit) {
  (void)unit;
  return Val_long((intnat)self());
}

value rig_test_waits_for_lock(value v_t) {
  return Val_bool(in_wait((thread_id)Long_val(v_t), 1));
}

value rig_test_waits_in_condition(value v_t) {
  return Val_bool(in_wait((thread_id)Long_val(v_t), 0));
}

value rig_test_interrupter_start(value unit) {
  (void)unit;
  interrupter.watched = self();
  atomic_store(&interrupter.stop, 0);
  atomic_store(&interrupter.raised, 0);
  if (pthread_create(&interrupter.thread, NULL, interrupter_main, NULL) != 0)
    caml_failwith("Rig_support.interrupt_in_wait: no thread");
  return Val_unit;
}

/* Stops the interrupter: whether it raised SIGINT. */
value rig_test_interrupter_stop(value unit) {
  (void)unit;
  atomic_store(&interrupter.stop, 1);
  caml_release_runtime_system();
  pthread_join(interrupter.thread, NULL);
  caml_acquire_runtime_system();
  return Val_bool(atomic_load(&interrupter.raised));
}

#else

value rig_test_sees_waits(value unit) {
  (void)unit;
  return Val_false;
}

value rig_test_thread_self(value unit) {
  (void)unit;
  return Val_long(0);
}

value rig_test_waits_for_lock(value v_t) {
  (void)v_t;
  caml_failwith("Rig_support.waits_for_lock: no view of waits");
}

value rig_test_waits_in_condition(value v_t) {
  (void)v_t;
  caml_failwith("Rig_support.waits_in_condition: no view of waits");
}

value rig_test_interrupter_start(value unit) {
  (void)unit;
  caml_failwith("Rig_support.interrupt_in_wait: no view of waits");
}

value rig_test_interrupter_stop(value unit) {
  (void)unit;
  return Val_false;
}

#endif

/* Holds the runtime: setrlimit does not block. */
value rig_test_set_address_space(value v_n) {
#if defined(__linux__)
  struct rlimit l;
  if (getrlimit(RLIMIT_AS, &l) != 0) return Val_int(errno);
  l.rlim_cur = (rlim_t)Long_val(v_n);
  return Val_int(setrlimit(RLIMIT_AS, &l) == 0 ? 0 : errno);
#else
  (void)v_n;
  return Val_int(-1);
#endif
}

/* A fill that takes one from the 64-bit word whose address its argument
   holds, and fails if that makes it zero. */
static int countdown(void *queue, void *arg, uint64_t v) {
  (void)queue;
  (void)v;
  uint64_t at;
  memcpy(&at, arg, sizeof at);
  return atomic_fetch_sub((_Atomic uint64_t *)(uintptr_t)at, 1) == 1;
}

value rig_test_countdown(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&countdown);
}

/* Stores [c] at [word] with release order: a rail's ready function. */
static void ready(void *word, uint64_t c) {
  atomic_store_explicit((_Atomic uint64_t *)word, c, memory_order_release);
}

value rig_test_ready(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&ready);
}

/* A fill that copies bytes: its argument's three 64-bit words are the
   destination's address, the source's and the number of bytes. */
static int carry(void *queue, void *arg, uint64_t v) {
  (void)queue;
  (void)v;
  uint64_t a[3];
  memcpy(a, arg, sizeof a);
  move_bytes((uintptr_t)a[0], (uintptr_t)a[1], (size_t)a[2]);
  return 0;
}

value rig_test_carry(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&carry);
}

/* Polled's functions */

static uint64_t param64(const struct rig_block *b, uint32_t at) {
  uint64_t w;
  memcpy(&w, b->params + at, sizeof w);
  return w;
}

/* Does nothing. */
static void run_main(const uint32_t g[3], const struct rig_block *b,
                     uint32_t params) {
  (void)g, (void)b, (void)params;
}

/* Stores, as the 64-bit word of its group's index in the grid, x fastest,
   the second parameter word plus that index, into the buffer whose address
   the first word holds. */
static void run_fill(const uint32_t g[3], const struct rig_block *b,
                     uint32_t params) {
  (void)params;
  uint64_t i =
      g[0] + (uint64_t)b->groups[0] * (g[1] + (uint64_t)b->groups[1] * g[2]);
  uint64_t *out = (uint64_t *)(uintptr_t)param64(b, 0);
  out[i] = param64(b, 8) + i;
}

/* Group 0 copies the bytes the third parameter word counts from the
   address the first word holds to the one the second holds. */
static void run_copy(const uint32_t g[3], const struct rig_block *b,
                     uint32_t params) {
  (void)params;
  if (g[0] != 0 || g[1] != 0 || g[2] != 0) return;
  move_bytes((uintptr_t)param64(b, 8), (uintptr_t)param64(b, 0),
             (size_t)param64(b, 16));
}

/* Each allows 1024 threads per group and 48 KiB of shared memory. */
static const struct polled_kernel polled_main = {run_main, 1024, 49152};
static const struct polled_kernel polled_fill = {run_fill, 1024, 49152};
static const struct polled_kernel polled_copy = {run_copy, 1024, 49152};

/* The launch of Polled's function [v_name], or 0 for no such function. */
value rig_test_polled_kernel(value v_name) {
  const char *n = String_val(v_name);
  const struct polled_kernel *k = strcmp(n, "main") == 0   ? &polled_main
                                  : strcmp(n, "fill") == 0 ? &polled_fill
                                  : strcmp(n, "copy") == 0 ? &polled_copy
                                                           : NULL;
  return caml_copy_nativeint((intnat)k);
}

/* The blocks of the first LAST_LAUNCHES launches run since the last call,
   each as it ran, which it forgets and frees. */
value rig_test_polled_launches(value v_p) {
  CAMLparam1(v_p);
  CAMLlocal2(a, s);
  struct polled *p = Polled_val(v_p);
  struct launched *copy = malloc(LAST_LAUNCHES * sizeof *copy);
  if (copy == NULL) caml_raise_out_of_memory();
  int n = 0;
  lock(&p->mu);
  struct polled_run *r = p->run;
  if (r != NULL) {
    n = r->nlaunched;
    memcpy(copy, r->launched, LAST_LAUNCHES * sizeof *copy);
    r->nlaunched = 0;
  }
  unlock(&p->mu);
  a = caml_alloc_tuple((mlsize_t)n);
  for (int i = 0; i < n; i++) {
    s = caml_alloc_initialized_string(copy[i].bytes,
                                      (const char *)copy[i].block);
    Store_field(a, i, s);
  }
  for (int i = 0; i < n; i++) free(copy[i].block);
  free(copy);
  CAMLreturn(a);
}

/* Raises SIGINT in the calling thread: the runtime records it, and the
   thread's next poll point runs its handler. */
value rig_test_interrupt(value unit) {
  (void)unit;
  raise(SIGINT);
  return Val_unit;
}

/* A thread of no OCaml code that raises SIGINT [ms] milliseconds after it
   starts, as a Ctrl-C does, kept until [rig_test_interrupt_join] joins it. */
#ifndef _WIN32
static pthread_t later;
static int later_runs;

static void *interrupt_later(void *p) {
  sleep_ms((uint64_t)(uintptr_t)p);
  raise(SIGINT);
  return NULL;
}
#endif

value rig_test_interrupt_after(value v_ms) {
#ifdef _WIN32
  (void)v_ms;
  caml_failwith("Rig_support.interrupt_after: no thread on Windows");
#else
  if (later_runs) caml_failwith("Rig_support.interrupt_after: one runs");
  if (pthread_create(&later, NULL, interrupt_later,
                     (void *)(uintptr_t)Long_val(v_ms)) != 0)
    caml_failwith("Rig_support.interrupt_after: no thread");
  later_runs = 1;
#endif
  return Val_unit;
}

/* Joins the thread [rig_test_interrupt_after] started, then runs the
   handlers of the signals pending, so its SIGINT lands here. */
value rig_test_interrupt_join(value unit) {
  (void)unit;
#ifndef _WIN32
  if (later_runs) {
    caml_enter_blocking_section_no_pending();
    pthread_join(later, NULL);
    caml_leave_blocking_section();
    later_runs = 0;
  }
#endif
  caml_process_pending_actions();
  return Val_unit;
}

value rig_test_move(value v_dst, value v_src, value v_len) {
  move_bytes((uintptr_t)Long_val(v_dst), (uintptr_t)Long_val(v_src),
             (size_t)Long_val(v_len));
  return Val_unit;
}

value rig_test_load(value v_addr) {
  return Val_long((intnat)atomic_load((_Atomic uint64_t *)Long_val(v_addr)));
}

value rig_test_load64(value v_a, value v_at) {
  char *p = (char *)Caml_ba_data_val(v_a) + Long_val(v_at);
  return caml_copy_int64(
      (int64_t)atomic_load_explicit((_Atomic uint64_t *)p, memory_order_acquire));
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

/* The reason lives in the OCaml heap, which the allocation of its copy may
   move: it is read again after the allocation. */
value rig_test_reader_why(value v_b) {
  CAMLparam1(v_b);
  CAMLlocal1(s);
  const char *why = rig_buffer_why(v_b);
  if (why == NULL) CAMLreturn(Val_none);
  size_t n = strlen(why);
  s = caml_alloc_string(n);
  memcpy((char *)Bytes_val(s), rig_buffer_why(v_b), n);
  CAMLreturn(caml_alloc_some(s));
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

/* rig_buffers_claim of the OCaml arrays [v_bs] and [v_access], whose
   outcome is Reader.outcome: All of the answers, or Refused of the index
   and its answer. */
value rig_test_reader_claim_all(value v_bs, value v_access) {
  CAMLparam2(v_bs, v_access);
  CAMLlocal2(r, answers);
  int n = (int)Wosize_val(v_bs);
  enum rig_access access[n > 0 ? n : 1];
  enum rig_claim answer[n > 0 ? n : 1];
  for (int k = 0; k < n; k++)
    access[k] = Int_val(Field(v_access, k)) == 0 ? RIG_READ : RIG_READ_WRITE;
  int refused = rig_buffers_claim(n, Op_val(v_bs), access, answer);
  if (refused >= 0) {
    r = caml_alloc(2, 1);
    Store_field(r, 0, Val_int(refused));
    Store_field(r, 1, Val_int(answer[refused]));
    CAMLreturn(r);
  }
  answers = caml_alloc(n, 0);
  for (int k = 0; k < n; k++) Store_field(answers, k, Val_int(answer[k]));
  r = caml_alloc(1, 0);
  Store_field(r, 0, answers);
  CAMLreturn(r);
}

value rig_test_reader_release_all(value v_bs) {
  rig_buffers_release((int)Wosize_val(v_bs), Op_val(v_bs));
  return Val_unit;
}

value rig_test_reader_span(value v_b) {
  CAMLparam1(v_b);
  CAMLlocal1(r);
  intnat space, first;
  rig_buffer_span(v_b, &space, &first);
  r = caml_alloc_tuple(2);
  Store_field(r, 0, Val_long(space));
  Store_field(r, 1, Val_long(first));
  CAMLreturn(r);
}

/* How many holders share [v_ba]'s storage, as its proxy counts them. */
value rig_test_shares(value v_ba) {
  struct caml_ba_proxy *p = Caml_ba_array_val(v_ba)->proxy;
  return Val_long(p == NULL ? 0 : (intnat)atomic_load(&p->refcount));
}

/* Rig's own: the bytes host buffers hold in the host's budget. */
extern intnat rig_heap_held(void);

value rig_test_host_held(value unit) {
  (void)unit;
  return Val_long(rig_heap_held());
}

#if defined(__GLIBC__) && !defined(RIG_TEST_SANITIZER)
/* glibc keeps a cache per thread of the chunks it freed last, up to 7 of
   each size from 32 to 1040 bytes by its defaults, and counts them as
   allocated: the count moved by what the cache held, with the order of the
   thread's last frees. Taking more chunks of each size than the cache holds,
   then freeing them, leaves the calling thread's cache full: it then adds the
   same bytes to every count. */
#define CACHED_SIZES 64
#define CACHE_FILL 16

static void fill_cache(void) {
  /* Volatile, so that the compiler keeps each allocation and its free. */
  void *volatile p[CACHE_FILL];
  for (size_t chunk = 32; chunk < 32 + 16 * CACHED_SIZES; chunk += 16) {
    for (int i = 0; i < CACHE_FILL; i++) p[i] = malloc(chunk - sizeof(size_t));
    for (int i = 0; i < CACHE_FILL; i++) free(p[i]);
  }
}
#endif

/* The bytes the C heap holds allocated, or -1 where its allocator does not
   say. */
value rig_test_heap_bytes(value unit) {
  (void)unit;
#if defined(RIG_TEST_SANITIZER)
  return Val_long((intnat)__sanitizer_get_current_allocated_bytes());
#elif defined(__APPLE__)
  malloc_statistics_t s;
  malloc_zone_statistics(NULL, &s);
  return Val_long((intnat)s.size_in_use);
#elif defined(__GLIBC__)
  fill_cache();
  struct mallinfo2 m = mallinfo2();
  return Val_long((intnat)(m.uordblks + m.hblkhd));
#else
  return Val_long(-1);
#endif
}

/* Whether GCC compiled this C with AddressSanitizer. Clang's runtime takes
   its allocator's locks across fork; GCC 14's takes none. */
value rig_test_gcc_asan(value unit) {
  (void)unit;
#if defined(__SANITIZE_ADDRESS__) && !defined(__clang__)
  return Val_true;
#else
  return Val_false;
#endif
}

/* Sockets */

/* Winsock takes the bound as a DWORD of milliseconds, POSIX as a timeval. */
value rig_test_recv_timeout(value fd, value s) {
  double t = Double_val(s);
#ifdef _WIN32
  DWORD bound = (DWORD)ceil(t * 1e3);
  if (setsockopt(Socket_val(fd), SOL_SOCKET, SO_RCVTIMEO, (const char *)&bound,
                 sizeof bound) != 0) {
    caml_win32_maperr(WSAGetLastError());
    caml_uerror("setsockopt", Nothing);
  }
#else
  long long us = (long long)ceil(t * 1e6);
  struct timeval bound = {.tv_sec = (time_t)(us / 1000000),
                          .tv_usec = (suseconds_t)(us % 1000000)};
  if (setsockopt(Int_val(fd), SOL_SOCKET, SO_RCVTIMEO, &bound, sizeof bound) !=
      0)
    caml_uerror("setsockopt", Nothing);
#endif
  return Val_unit;
}

/* Descriptors */

#if !defined(_WIN32)
#include <errno.h>
#include <sys/resource.h>
#endif

/* This process's soft limit of its descriptors: max_int if none, -1 on
   Windows. */
value rig_test_descriptor_limit(value unit) {
  (void)unit;
#if defined(_WIN32)
  return Val_int(-1);
#else
  struct rlimit l;
  if (getrlimit(RLIMIT_NOFILE, &l) != 0) caml_uerror("getrlimit", Nothing);
  if (l.rlim_cur == RLIM_INFINITY || l.rlim_cur > (rlim_t)Max_long)
    return Val_long(Max_long);
  return Val_long((intnat)l.rlim_cur);
#endif
}

/* Sets this process's soft limit of its descriptors to [n]: 0, the errno,
   or -1 on Windows, which has no such limit. */
value rig_test_set_descriptor_limit(value v_n) {
#if defined(_WIN32)
  (void)v_n;
  return Val_int(-1);
#else
  struct rlimit l;
  if (getrlimit(RLIMIT_NOFILE, &l) != 0) return Val_int(errno);
  l.rlim_cur = (rlim_t)Long_val(v_n);
  return Val_int(setrlimit(RLIMIT_NOFILE, &l) == 0 ? 0 : errno);
#endif
}

/* Memory */

/* This process's resident memory in bytes: its working set on Windows. */
value rig_test_resident(value unit) {
  (void)unit;
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS c;
  if (!K32GetProcessMemoryInfo(GetCurrentProcess(), &c, sizeof c))
    caml_failwith("GetProcessMemoryInfo failed");
  return Val_long((intnat)c.WorkingSetSize);
#elif defined(__APPLE__)
  mach_task_basic_info_data_t info;
  mach_msg_type_number_t n = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info,
                &n) != KERN_SUCCESS)
    caml_failwith("task_info failed");
  return Val_long((intnat)info.resident_size);
#elif defined(__linux__)
  long pages = 0, resident = 0;
  FILE *f = fopen("/proc/self/statm", "r");
  if (f == NULL) caml_failwith("opening /proc/self/statm failed");
  int read = fscanf(f, "%ld %ld", &pages, &resident);
  fclose(f);
  if (read != 2) caml_failwith("reading /proc/self/statm failed");
  return Val_long((intnat)resident * sysconf(_SC_PAGESIZE));
#else
  caml_failwith("no resident memory on this system");
#endif
}

#if defined(RIG_TEST_TSAN) && defined(__linux__)
#include <fcntl.h>
#include <sys/syscall.h>

/* ThreadSanitizer's default options in every process that links this file:
   the suppressions of tsan.supp, beside this directory. The executable lies
   under the source root's _build, which names the root. ThreadSanitizer calls
   this before it starts, before the C library has the environment, so it is
   not instrumented, makes its system call itself and copies by hand. Outside
   _build, a process suppresses nothing. */
const char *__tsan_default_options(void);
__attribute__((no_sanitize("thread"))) const char *__tsan_default_options(
    void) {
  static char exe[2048], opts[4096];
  static const char key[] = "suppressions=\"", build[] = "/_build/",
                    file[] = "/packages/rig/test/tsan.supp\"";
  long n = syscall(SYS_readlinkat, AT_FDCWD, "/proc/self/exe", exe,
                   sizeof exe);
  long root = -1;
  for (long i = 0; i + (long)sizeof build - 1 <= n; i++) {
    long k = 0;
    while (build[k] != '\0' && exe[i + k] == build[k]) k++;
    if (build[k] == '\0') root = i;
  }
  if (root < 0) return "";
  long m = 0;
  for (long k = 0; key[k] != '\0'; k++) opts[m++] = key[k];
  for (long k = 0; k < root; k++) opts[m++] = exe[k];
  for (long k = 0; file[k] != '\0'; k++) opts[m++] = file[k];
  opts[m] = '\0';
  return opts;
}
#endif
