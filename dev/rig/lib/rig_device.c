/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Devices: the turn, the hand-over, the commit, loss, spread and the owed
   stop.

   A device's mutex is held only for a few loads and stores, and, for a
   device whose driver never blocks, across its driver's room check and
   hand-over. No stub runs signal handlers: a stub that must wait for a
   mutex, or that waits, releases the domain lock with
   caml_enter_blocking_section_no_pending, so pending handlers run at the
   caller's next poll point, and takes it back after its last unlock. */

#define _GNU_SOURCE

#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/minor_gc.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>
#include <caml/threads.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sched.h>
#include <time.h>
#endif

#include "rig_stubs.h"

/* How long a turn waiter blocks before it returns to OCaml, where pending
   signals run: the still interval of waits. */
#define STILL_MS 200

/* What a submit answers. */
enum {
  SUBMIT_OK,
  SUBMIT_BUSY,
  SUBMIT_NO_ROOM,
  SUBMIT_NEVER,
  SUBMIT_LOST,
  SUBMIT_LOST_AFTER,
  SUBMIT_PRODUCER_LOST,
  SUBMIT_FAILED,
  SUBMIT_NEED_RECORD
};

/* Mutexes and conditions */

#ifdef _WIN32
static void mutex_init(rig_mutex *m) { InitializeSRWLock(m); }
static int mutex_try(rig_mutex *m) { return TryAcquireSRWLockExclusive(m); }
static void mutex_lock(rig_mutex *m) { AcquireSRWLockExclusive(m); }
static void mutex_unlock(rig_mutex *m) { ReleaseSRWLockExclusive(m); }
static void cond_init(rig_cond *c) { InitializeConditionVariable(c); }
static void cond_wait(rig_cond *c, rig_mutex *m) {
  SleepConditionVariableSRW(c, m, INFINITE, 0);
}
static void cond_broadcast(rig_cond *c) { WakeAllConditionVariable(c); }
static void cv_wait_ms(struct rig_device *d, int ms) {
  SleepConditionVariableSRW(&d->cv, &d->mu, (DWORD)ms, 0);
}
#else
static void mutex_init(rig_mutex *m) { pthread_mutex_init(m, NULL); }
static int mutex_try(rig_mutex *m) { return pthread_mutex_trylock(m) == 0; }
static void mutex_lock(rig_mutex *m) { pthread_mutex_lock(m); }
static void mutex_unlock(rig_mutex *m) { pthread_mutex_unlock(m); }
static void cond_init(rig_cond *c) { pthread_cond_init(c, NULL); }
static void cond_wait(rig_cond *c, rig_mutex *m) { pthread_cond_wait(c, m); }
static void cond_broadcast(rig_cond *c) { pthread_cond_broadcast(c); }
/* A clock jump moves one timeout, which only returns a waiter early or late
   to OCaml. */
static void cv_wait_ms(struct rig_device *d, int ms) {
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  t.tv_sec += ms / 1000;
  t.tv_nsec += (long)(ms % 1000) * 1000000;
  if (t.tv_nsec >= 1000000000) {
    t.tv_sec += 1;
    t.tv_nsec -= 1000000000;
  }
  pthread_cond_timedwait(&d->cv, &d->mu, &t);
}
#endif

static void mu_init(struct rig_device *d) {
  mutex_init(&d->mu);
  cond_init(&d->cv);
}
static void cv_broadcast(struct rig_device *d) { cond_broadcast(&d->cv); }
static void mu_lock(struct rig_device *d) { mutex_lock(&d->mu); }
static void mu_unlock(struct rig_device *d) { mutex_unlock(&d->mu); }

void rig_guard_init(struct rig_sub *s) {
  mutex_init(&s->guard);
  cond_init(&s->freed);
}

void rig_guard_destroy(struct rig_sub *s) {
#ifdef _WIN32
  (void)s;
#else
  pthread_mutex_destroy(&s->guard);
  pthread_cond_destroy(&s->freed);
#endif
}

/* Takes the mutex [m] from a stub holding the domain lock: by try-lock,
   and otherwise after releasing the domain lock. Answers whether it
   released it, for [give]. */
static int take_mutex(rig_mutex *m) {
  if (mutex_try(m)) return 0;
  caml_enter_blocking_section_no_pending();
  mutex_lock(m);
  return 1;
}

static int take(struct rig_device *d) { return take_mutex(&d->mu); }

static void give(struct rig_device *d, int released) {
  mu_unlock(d);
  if (released) caml_leave_blocking_section();
}

static void relax(void) {
#ifdef _WIN32
  SwitchToThread();
#else
  sched_yield();
#endif
}

static int64_t now_ms(void) { return (int64_t)(rig_now_ns() / 1000000); }

/* The table of devices */

static _Atomic(struct rig_device *) devices[RIG_DEVICES];
static _Atomic int top;

/* The device of index [i], or NULL. */
static struct rig_device *device_of(int i) {
  if (i <= 0 || i >= RIG_DEVICES) return NULL;
  return atomic_load_explicit(&devices[i], memory_order_acquire);
}

/* The last value [d]'s word showed. */
static uint64_t device_word(struct rig_device *d) {
  _Atomic uint64_t *w = atomic_load_explicit(&d->word, memory_order_acquire);
  if (w != NULL) return atomic_load_explicit(w, memory_order_acquire);
  return atomic_load_explicit(&d->seen, memory_order_acquire);
}

static int state(struct rig_device *d) {
  return atomic_load_explicit(&d->state, memory_order_acquire);
}

static int is_lost(struct rig_device *d) { return state(d) != RIG_LIVE; }

/* Whether [d] is lost and its stop has not returned. */
static int stopping(struct rig_device *d) {
  int s = state(d);
  return s == RIG_LOSING || s == RIG_OWED || s == RIG_STOPPING;
}

/* The word is read before the state: a word the stop raised shows its
   loss. An orphaned device's word is never read: it may be a page the child
   does not map. */
int rig_point_done(uint64_t p) {
  struct rig_device *d = device_of(RIG_INDEX(p));
  if (d == NULL) return 1;
  if (state(d) == RIG_ORPHANED) return 0;
  uint64_t w = device_word(d);
  int s = state(d);
  if (s == RIG_LIVE) return w >= RIG_VALUE(p);
  return s != RIG_ORPHANED &&
         atomic_load_explicit(&d->reached, memory_order_relaxed) >=
             RIG_VALUE(p);
}

value caml_rig_done(value v_p) {
  return Val_bool(rig_point_done((uint64_t)Long_val(v_p)));
}

#define Device_val(v) ((struct rig_device *)Long_val(v))

/* Locks */

/* A lock of rig's OCaml state, with a condition. Every lock is on
   one list, so a forked child can make each anew; none is freed, as a
   device and a module live as long as their process. */
struct rig_lock {
  rig_mutex mu;
  rig_cond cv;
  struct rig_lock *next;
};

static _Atomic(struct rig_lock *) locks;

/* The process's loss facts: the devices whose stop is owed; the first
   failure, 0 for none, a device's index for its loss, -1 for [fail]'s;
   [fail]'s reason; and the forks that made this process, which tell a hold
   made before a fork. */
static _Atomic int owed;
static _Atomic int first;
static _Atomic(char *) failed;
static _Atomic int generation;

#define Lock_val(v) ((struct rig_lock *)Long_val(v))

/* After fork, the locks are made anew, since a thread of the parent may
   have held them. Each device is made anew and its calls in flight
   forgotten. A driver's device the child inherits is orphaned, lost with
   the reason "forked" unless it was lost: freeing would call a driver,
   and a word may be a page the child shares with its parent, or does not
   map. An io device's state is its io library's, which decides what a
   fork does to it: rig leaves it as it was, and runs a stop that a thread
   of the parent claimed or owed. A device the child opens is its own. */
#ifndef _WIN32
static void forked_child(void) {
  static char why[] = "forked";
  for (struct rig_lock *l = atomic_load(&locks); l != NULL; l = l->next) {
    mutex_init(&l->mu);
    cond_init(&l->cv);
  }
  atomic_fetch_add(&generation, 1);
  int n = atomic_load(&top), owing = 0;
  for (int i = 0; i <= n; i++) {
    struct rig_device *d = atomic_load(&devices[i]);
    if (d == NULL) continue;
    mu_init(d);
    d->turn = d->inside = 0;
    if (d->io) {
      if (stopping(d)) atomic_store(&d->state, RIG_OWED);
      owing += state(d) == RIG_OWED;
    } else if (i > 0) {
      char *none = NULL;
      atomic_compare_exchange_strong(&d->lost, &none, why);
      atomic_store(&d->state, RIG_ORPHANED);
    }
  }
  atomic_store(&owed, owing);
}

static pthread_once_t atfork_once = PTHREAD_ONCE_INIT;
static void atfork(void) { pthread_atfork(NULL, NULL, forked_child); }
#endif

/* A new lock. The first is made as the library starts, so the fork
   handler is in place before any thread can hold a lock. */
value caml_rig_lock_new(value unit) {
  (void)unit;
#ifndef _WIN32
  pthread_once(&atfork_once, atfork);
#endif
  struct rig_lock *l = malloc(sizeof *l);
  if (l == NULL) caml_raise_out_of_memory();
  mutex_init(&l->mu);
  cond_init(&l->cv);
  l->next = atomic_load(&locks);
  while (!atomic_compare_exchange_weak(&locks, &l->next, l)) {
  }
  return Val_long((intnat)l);
}

value caml_rig_lock_try(value v_l) {
  return Val_bool(mutex_try(&Lock_val(v_l)->mu));
}

/* Takes the lock [v_l], which a try found held: waits for it without the
   domain lock. */
value caml_rig_lock_take(value v_l) {
  if (take_mutex(&Lock_val(v_l)->mu)) caml_leave_blocking_section();
  return Val_unit;
}

value caml_rig_lock_give(value v_l) {
  mutex_unlock(&Lock_val(v_l)->mu);
  return Val_unit;
}

/* Gives up the lock [v_l], which the caller holds, until a broadcast on
   it or a spurious wake-up, then takes it back; without the domain lock
   meanwhile. */
value caml_rig_lock_wait(value v_l) {
  struct rig_lock *l = Lock_val(v_l);
  caml_enter_blocking_section_no_pending();
  cond_wait(&l->cv, &l->mu);
  caml_leave_blocking_section();
  return Val_unit;
}

value caml_rig_lock_broadcast(value v_l) {
  cond_broadcast(&Lock_val(v_l)->cv);
  return Val_unit;
}

/* A device record of index [v_index] named [v_name], not yet published. */
static struct rig_device *record(value v_index, value v_name) {
  int index = Int_val(v_index);
  struct rig_device *d = calloc(1, sizeof *d);
  char *name = strdup(String_val(v_name));
  if (d == NULL || name == NULL) {
    free(d);
    free(name);
    caml_raise_out_of_memory();
  }
  mu_init(d);
  d->index = index;
  d->name = name;
  return d;
}

/* A driver's device of index [v_index] whose driver blocks ([v_may_block]),
   with its driver's C state [v_edge] and its word's host address (0 behind
   a transport). */
value caml_rig_device_new(value v_index, value v_name, value v_may_block,
                          value v_edge, value v_word) {
  void *self = (void *)Nativeint_val(v_edge);
  if (self == NULL || rig_driver_of(self) == NULL)
    caml_invalid_argument("Rig.open_: the driver's edge names no rig_driver");
  struct rig_device *d = record(v_index, v_name);
  d->may_block = Bool_val(v_may_block);
  d->self = self;
  d->driver = *rig_driver_of(self);
  atomic_init(&d->word, (_Atomic uint64_t *)Nativeint_val(v_word));
  return Val_long((intnat)d);
}

/* An io device of index [v_index]: no queue and no word; its state is its
   io library's. */
value caml_rig_io_new(value v_index, value v_name) {
  struct rig_device *d = record(v_index, v_name);
  d->io = 1;
  return Val_long((intnat)d);
}

static void put(struct rig_device *d) {
  atomic_store(&devices[d->index], d);
  int t = atomic_load(&top);
  while (t < d->index && !atomic_compare_exchange_weak(&top, &t, d->index)) {
  }
}

/* The host's record, of index 0: never lost, with no word, no queue and
   nothing submitted. */
value caml_rig_host_new(value v_name) {
  struct rig_device *d = record(Val_int(0), v_name);
  put(d);
  return Val_long((intnat)d);
}

value caml_rig_word(value v_d) {
  struct rig_device *d = Device_val(v_d);
  if (state(d) == RIG_ORPHANED)
    return Val_long((intnat)atomic_load(&d->seen));
  return Val_long((intnat)device_word(d));
}

/* Records [v] as read from [d]'s word, for a word behind a transport. */
value caml_rig_set_seen(value v_d, value v) {
  struct rig_device *d = Device_val(v_d);
  uint64_t seen = atomic_load(&d->seen), w = (uint64_t)Long_val(v);
  while (seen < w && !atomic_compare_exchange_weak(&d->seen, &seen, w)) {
  }
  return Val_unit;
}

value caml_rig_submitted(value v_d) {
  struct rig_device *d = Device_val(v_d);
  return Val_long((intnat)atomic_load_explicit(&d->submitted,
                                               memory_order_acquire));
}

value caml_rig_committed(value v_d) {
  struct rig_device *d = Device_val(v_d);
  return Val_long((intnat)atomic_load_explicit(&d->committed,
                                               memory_order_acquire));
}

value caml_rig_state(value v_d) { return Val_int(state(Device_val(v_d))); }

value caml_rig_why(value v_d) {
  char *why = atomic_load_explicit(&Device_val(v_d)->lost,
                                   memory_order_acquire);
  return caml_copy_string(why == NULL ? "" : why);
}

/* Loss */

/* Loses [d] with the reason [why], which the record keeps: answers 1 if
   this call lost it, 0 if it was lost already. A loss that [counts] is the
   process's first failure if none came before. Its stop is owed once its
   loser finished spreading it ([finish]). The mutex is held. */
static int lose_locked(struct rig_device *d, char *why, int counts) {
  char *none = NULL;
  if (!atomic_compare_exchange_strong_explicit(
          &d->lost, &none, why, memory_order_acq_rel, memory_order_acquire))
    return 0;
  atomic_store_explicit(&d->reached, device_word(d), memory_order_relaxed);
  atomic_store_explicit(&d->state, RIG_LOSING, memory_order_release);
  int zero = 0;
  if (counts) atomic_compare_exchange_strong(&first, &zero, d->index);
  cv_broadcast(d);
  return 1;
}

/* Claims [d]'s stop if it is owed and [d] runs no counted call: the
   claimant runs it. The mutex is held. */
static int claim_locked(struct rig_device *d) {
  if (state(d) != RIG_OWED || d->inside != 0) return 0;
  atomic_store(&d->state, RIG_STOPPING);
  atomic_fetch_sub(&owed, 1);
  return 1;
}

/* Ends [d]'s spreading: its stop is owed. The loser claims it next
   ([caml_rig_claim]), or, while a counted call is inside, the call that
   leaves last. The count moves under the mutex, so a counted call that
   leaves after it sees it. */
static void finish(struct rig_device *d) {
  mu_lock(d);
  atomic_store(&d->state, RIG_OWED);
  atomic_fetch_add(&owed, 1);
  mu_unlock(d);
}

/* The reason a device lost by [p]'s loss keeps: "[p's name] lost". */
static char *lost_with(struct rig_device *p) {
  size_t n = strlen(p->name);
  char *why = malloc(n + sizeof " lost");
  if (why != NULL) {
    memcpy(why, p->name, n);
    memcpy(why + n, " lost", sizeof " lost");
  }
  return why;
}

/* Loses every device whose unreached work waits in its queue on an
   unreached value of the lost [p], reading both words as they stand, and
   spreads from each. A device is lost once, so each is visited once. The
   domain lock is released and no mutex held. */
static void spread(struct rig_device *p) {
  int n = atomic_load(&top);
  char *why = NULL;
  for (int i = 1; i <= n; i++) {
    struct rig_device *c = device_of(i);
    if (c == NULL || c == p || is_lost(c)) continue;
    if (why == NULL) why = lost_with(p);
    uint64_t pw = device_word(p);
    int won = 0;
    mu_lock(c);
    uint64_t cw = device_word(c);
    for (int j = 0; j < c->nrecord && !won; j++) {
      struct rig_entry *e = &c->record[j];
      if (e->producer == p->index && e->w > pw && e->u > cw)
        won = lose_locked(c, why != NULL ? why : p->name, 1);
    }
    mu_unlock(c);
    if (!won) continue;
    why = NULL; /* [c] keeps it */
    spread(c);
    finish(c);
  }
  free(why);
}

/* Loses [d] for the reason [v_why], a fault of a counted call and a
   failure of the process if [v_fault]: whether this call lost it. [v_why]
   is copied before the runtime is released; [v_d] is an immediate. */
value caml_rig_lose(value v_d, value v_why, value v_fault) {
  struct rig_device *d = Device_val(v_d);
  char *why = strdup(String_val(v_why));
  if (why == NULL) caml_raise_out_of_memory();
  caml_enter_blocking_section_no_pending();
  mu_lock(d);
  int won = lose_locked(d, why, Bool_val(v_fault));
  if (won) d->faulted = Bool_val(v_fault);
  mu_unlock(d);
  if (won) {
    spread(d);
    finish(d);
  } else
    free(why);
  caml_leave_blocking_section();
  return Val_bool(won);
}

/* Whether [v_d] was lost for a fault of a counted call. Read by its stop,
   which runs after the loss, under no lock. */
value caml_rig_faulted(value v_d) {
  return Val_bool(Device_val(v_d)->faulted);
}

/* Publishes the record [v_d], once its OCaml device is in the table: it
   is lost at birth if the process failed. [failed] is read after the
   record is stored, and [caml_rig_fail] stores it before it walks the
   table, so a device that opens during a failure is lost by one or the
   other. */
value caml_rig_publish(value v_d) {
  struct rig_device *d = Device_val(v_d);
  put(d);
  char *why = atomic_load(&failed);
  if (why == NULL) return Val_unit;
  caml_enter_blocking_section_no_pending();
  mu_lock(d);
  int won = lose_locked(d, why, 0);
  mu_unlock(d);
  if (won) finish(d);
  caml_leave_blocking_section();
  return Val_unit;
}

/* Fails the process for the reason [v_why]: loses every device but the
   host, and every device published afterwards. Only the first call acts:
   it answers whether it did. */
value caml_rig_fail(value v_why) {
  char *why = strdup(String_val(v_why));
  if (why == NULL) caml_raise_out_of_memory();
  char *none = NULL;
  if (!atomic_compare_exchange_strong(&failed, &none, why)) {
    free(why);
    return Val_false;
  }
  int zero = 0;
  atomic_compare_exchange_strong(&first, &zero, -1);
  caml_enter_blocking_section_no_pending();
  int n = atomic_load(&top);
  for (int i = 1; i <= n; i++) {
    struct rig_device *d = device_of(i);
    if (d == NULL) continue;
    mu_lock(d);
    int won = lose_locked(d, why, 0);
    mu_unlock(d);
    if (!won) continue;
    spread(d);
    finish(d);
  }
  caml_leave_blocking_section();
  return Val_true;
}

/* [fail]'s reason, if the process failed. */
value caml_rig_failed(value unit) {
  (void)unit;
  char *why = atomic_load(&failed);
  if (why == NULL) return Val_none;
  return caml_alloc_some(caml_copy_string(why));
}

/* The process's first failure: 0 for none, the index of the device whose
   loss it was, or -1 for [fail]'s. */
value caml_rig_first(value unit) {
  (void)unit;
  return Val_int(atomic_load(&first));
}

value caml_rig_generation(value unit) {
  (void)unit;
  return Val_int(atomic_load(&generation));
}

value caml_rig_owed(value unit) {
  (void)unit;
  return Val_int(atomic_load_explicit(&owed, memory_order_relaxed));
}

/* Claims an owed stop whose device runs no counted call: the device's
   index, whose stop the caller runs, or 0 if there is none. */
value caml_rig_claim(value unit) {
  (void)unit;
  if (atomic_load(&owed) == 0) return Val_int(0);
  int n = atomic_load(&top);
  for (int i = 1; i <= n; i++) {
    struct rig_device *d = device_of(i);
    if (d == NULL || state(d) != RIG_OWED) continue;
    int released = take(d);
    int claimed = claim_locked(d);
    give(d, released);
    if (claimed) return Val_int(i);
  }
  return Val_int(0);
}

/* Records that [d]'s stop returned: ENDED for a driver's device, whose
   work may still run, and STOPPED for an io device, whose work is the
   caller's. */
value caml_rig_stopped(value v_d) {
  struct rig_device *d = Device_val(v_d);
  int released = take(d);
  atomic_store(&d->state, d->io ? RIG_STOPPED : RIG_ENDED);
  cv_broadcast(d);
  give(d, released);
  return Val_unit;
}

/* Moves ENDED to STOPPED once [d]'s word reads its submitted value, read
   last through its driver for a word behind a transport: then no work of
   [d] runs. Answers whether this call moved it. */
value caml_rig_upgrade(value v_d) {
  struct rig_device *d = Device_val(v_d);
  if (state(d) != RIG_ENDED) return Val_false;
  if (device_word(d) < atomic_load(&d->submitted)) return Val_false;
  int ended = RIG_ENDED;
  return Val_bool(atomic_compare_exchange_strong(&d->state, &ended,
                                                 RIG_STOPPED));
}

/* Waits with the domain lock released until [d]'s stop returned, or
   [v_ms] milliseconds passed: whether it returned. */
value caml_rig_await_stop(value v_d, value v_ms) {
  struct rig_device *d = Device_val(v_d);
  int64_t until = now_ms() + Long_val(v_ms), left;
  caml_enter_blocking_section_no_pending();
  mu_lock(d);
  while (stopping(d) && (left = until - now_ms()) > 0) cv_wait_ms(d, (int)left);
  int returned = !stopping(d);
  mu_unlock(d);
  caml_leave_blocking_section();
  return Val_bool(returned);
}

/* Counted calls */

/* Counts a call on [d]: 0 once it is counted, 1 if [d] is lost. */
value caml_rig_enter(value v_d) {
  struct rig_device *d = Device_val(v_d);
  int released = take(d);
  int r = 0;
  if (is_lost(d)) r = 1;
  else d->inside++;
  give(d, released);
  return Val_int(r);
}

/* Ends a counted call on [d]. */
value caml_rig_exit(value v_d) {
  struct rig_device *d = Device_val(v_d);
  int released = take(d);
  d->inside--;
  give(d, released);
  return Val_unit;
}

/* Spins on [d]'s word, yielding the processor with the domain lock
   released, until it reads [v_target], [d] is lost, or [v_ms] milliseconds
   passed. Answers the last value read. The spin counts as a call inside
   [d], so [d]'s stop, and the end of its word, wait for it: it reads the
   word without the domain lock, which no minor collection waits for. */
value caml_rig_spin(value v_d, value v_target, value v_ms) {
  struct rig_device *d = Device_val(v_d);
  uint64_t target = (uint64_t)Long_val(v_target), w;
  int64_t until = now_ms() + Long_val(v_ms);
  caml_enter_blocking_section_no_pending();
  mu_lock(d);
  int lost = is_lost(d);
  if (!lost) d->inside++;
  else w = device_word(d);
  mu_unlock(d);
  if (lost) {
    caml_leave_blocking_section();
    return Val_long((intnat)w);
  }
  while ((w = device_word(d)) < target && !is_lost(d) && now_ms() < until)
    relax();
  mu_lock(d);
  d->inside--;
  mu_unlock(d);
  caml_leave_blocking_section();
  return Val_long((intnat)w);
}

/* The end of a stopped device's word. Once nothing reads the driver's
   word, the record moves its readers to [final], which holds the word's
   last value; the driver's word is given back once every domain that
   holds its runtime lock passed a minor collection since, as a reader that
   held it may have loaded the old address. Readers without the runtime
   lock read under the device's mutex or as a call inside it. The driver's
   word is still read by another device's queue while that device's record
   holds a wait on it its word has not passed. Answers whether this call
   moved the readers; [d] is stopped. */
value caml_rig_word_retire(value v_d) {
  struct rig_device *d = Device_val(v_d);
  caml_enter_blocking_section_no_pending();
  int n = atomic_load(&top), waited = 0;
  for (int i = 1; i <= n && !waited; i++) {
    struct rig_device *c = device_of(i);
    if (c == NULL || c == d) continue;
    mu_lock(c);
    uint64_t cw = device_word(c);
    for (int j = 0; j < c->nrecord && !waited; j++)
      waited = c->record[j].producer == d->index && c->record[j].u > cw;
    mu_unlock(c);
  }
  int moved = 0;
  if (!waited) {
    mu_lock(d);
    _Atomic uint64_t *w = atomic_load(&d->word);
    moved = w != &d->final && d->inside == 0;
    if (moved) {
      atomic_store(&d->final, atomic_load(w != NULL ? w : &d->seen));
      atomic_store(&d->word, &d->final);
    }
    mu_unlock(d);
  }
  caml_leave_blocking_section();
  return Val_bool(moved);
}

/* The minor collections the program made: each one waited for every domain
   that holds its runtime lock. */
value caml_rig_minors(value unit) {
  (void)unit;
  return Val_long((intnat)atomic_load(&caml_minor_collections_count));
}

/* The producers [d]'s unreached work waits on in its queue, each once, for
   a wait that looks for their unseen faults. */
value caml_rig_producers(value v_d) {
  struct rig_device *d = Device_val(v_d);
  int released = take(d), n = 0;
  int *seen = malloc((size_t)(d->nrecord > 0 ? d->nrecord : 1) * sizeof *seen);
  if (seen != NULL) {
    uint64_t w = device_word(d);
    for (int j = 0; j < d->nrecord; j++) {
      struct rig_entry *e = &d->record[j];
      if (e->u <= w) continue;
      int k = 0;
      while (k < n && seen[k] != e->producer) k++;
      if (k == n) seen[n++] = e->producer;
    }
  }
  give(d, released);
  if (seen == NULL) caml_raise_out_of_memory();
  value a = caml_alloc_tuple((mlsize_t)n);
  for (int i = 0; i < n; i++) Store_field(a, (mlsize_t)i, Val_int(seen[i]));
  free(seen);
  return a;
}

/* Grows [d]'s record so that [v_n] more entries fit. The array is made
   with no mutex held, which a mutex section never waits for. */
value caml_rig_ensure_record(value v_d, value v_n) {
  struct rig_device *d = Device_val(v_d);
  int n = Int_val(v_n);
  int released = take(d);
  int want = d->nrecord + n, have = d->crecord;
  give(d, released);
  if (want <= have) return Val_unit;
  int c = 2 * want < 8 ? 8 : 2 * want;
  struct rig_entry *grown = malloc((size_t)c * sizeof *grown);
  if (grown == NULL) caml_raise_out_of_memory();
  released = take(d);
  struct rig_entry *old = NULL;
  if (d->crecord < c) {
    if (d->nrecord > 0)
      memcpy(grown, d->record, (size_t)d->nrecord * sizeof *grown);
    old = d->record;
    d->record = grown;
    d->crecord = c;
    grown = NULL;
  }
  give(d, released);
  free(old);
  free(grown);
  return Val_unit;
}

/* The turn

   A driver that never blocks is called under [d]'s mutex, which is then
   the turn. One that may block is called with the mutex released and
   [turn] set, so that waits and loss go on beside it; it is a counted
   call. */

/* Takes [d]'s mutex and, for a driver that may block, waits for its turn
   to be free, returning to OCaml after the still interval: whether the
   mutex is held with the turn free or [d] lost. A driver that may block
   is waited for with the domain lock released; [*released] says whether
   it was, for [give]. Without [wait], it takes neither when either is
   held. */
static int lock_turn(struct rig_device *d, int wait, int *released) {
  if (!d->may_block) {
    if (wait) *released = take(d);
    else if (mutex_try(&d->mu)) *released = 0;
    else return 0;
    return 1;
  }
  caml_enter_blocking_section_no_pending();
  *released = 1;
  mu_lock(d);
  /* A free turn reads no clock: the submit is on every run's path. */
  if (wait && d->turn && !is_lost(d)) {
    int64_t until = now_ms() + STILL_MS, left;
    while (d->turn && !is_lost(d) && (left = until - now_ms()) > 0)
      cv_wait_ms(d, (int)left);
  }
  if (!d->turn || is_lost(d)) return 1;
  give(d, 1);
  return 0;
}

/* Takes and gives the turn of [d], whose mutex [lock_turn] took. */
static void begin_turn(struct rig_device *d) {
  if (!d->may_block) return;
  d->turn = 1;
  d->inside++;
  mu_unlock(d);
}

static void end_turn(struct rig_device *d) {
  if (!d->may_block) return;
  mu_lock(d);
  d->turn = 0;
  d->inside--;
  cv_broadcast(d);
}

/* Loses [d] for its driver's failure [why]: SUBMIT_FAILED if this call
   lost it, whose caller spreads the loss ([spread_failed]), SUBMIT_LOST if
   it was lost already. The mutex is held. */
static int fail_locked(struct rig_device *d, const char *why) {
  char *copy = strdup(why);
  if (lose_locked(d, copy != NULL ? copy : d->name, 1)) return SUBMIT_FAILED;
  free(copy);
  return SUBMIT_LOST;
}

/* Gives [d]'s mutex back and, if [r] is SUBMIT_FAILED, spreads the loss
   with the domain lock released: answers [r]. */
static int give_turn(struct rig_device *d, int released, int r) {
  mu_unlock(d);
  if (r == SUBMIT_FAILED) {
    if (!released) caml_enter_blocking_section_no_pending();
    released = 1;
    spread(d);
    finish(d);
  }
  if (released) caml_leave_blocking_section();
  return r;
}

/* Submitting */

/* Records [s]'s in-queue waits in [d]'s record, for spread, after dropping
   the entries [d]'s word shows reached, and refuses if a producer is lost.
   The mutex is held. */
static int record_waits(struct rig_device *d, struct rig_sub *s) {
  /* Nothing to drop or record reads no word: the device writes its line,
     so a read is a cache miss on every submit. */
  if (d->nrecord == 0 && s->nwaits == 0) return SUBMIT_OK;
  uint64_t w = device_word(d);
  int k = 0;
  for (int j = 0; j < d->nrecord; j++)
    if (d->record[j].u > w) d->record[k++] = d->record[j];
  d->nrecord = k;
  if (s->nwaits == 0) return SUBMIT_OK;
  if (k + s->nwaits > d->crecord) return SUBMIT_NEED_RECORD;
  uint64_t u = atomic_load(&d->submitted) + 1;
  for (int j = 0; j < s->nwaits; j++)
    d->record[k + j] =
        (struct rig_entry){s->producers[j], s->waits[j].value, u};
  d->nrecord = k + s->nwaits;
  for (int j = 0; j < s->nwaits; j++) {
    struct rig_device *p = device_of(s->producers[j]);
    if (p != NULL && is_lost(p)) {
      d->nrecord = k;
      s->producer = s->producers[j];
      return SUBMIT_PRODUCER_LOST;
    }
  }
  return SUBMIT_OK;
}

/* Asks [d]'s driver for room for [s]; once the parts fit, assigns the next
   value, stores it as submitted, hands the work over and raises [s]'s
   stamps. Called under [d]'s turn. */
static int admit(struct rig_device *d, struct rig_sub *s) {
  int room = d->driver.room(d->self, s->parts, s->nparts);
  if (room == RIG_NEVER) return SUBMIT_NEVER;
  if (room == RIG_LATER) {
    s->no_room_at = atomic_load(&d->submitted);
    return SUBMIT_NO_ROOM;
  }
  uint64_t v = atomic_load(&d->submitted) + 1;
  atomic_store_explicit(&d->submitted, v, memory_order_release);
  const char *why = NULL;
  int r = d->driver.submit(d->self, v, s->waits, s->nwaits, s->parts,
                           s->nparts, s->handles, s->nhandles, &why);
  rig_sub_raise(s, RIG_POINT(d->index, v));
  s->v = v;
  if (r == RIG_COMMITTED)
    atomic_store_explicit(&d->committed, v, memory_order_release);
  if (r != RIG_FAILED) return SUBMIT_OK;
  s->why = why != NULL ? why : "the driver's submit failed";
  return SUBMIT_FAILED;
}

/* Takes and gives the guard of the submission [v_s], which a submit holds
   throughout: two domains' submits of one submission take turns. A guard
   is taken before anything else of a submit, and nothing a submit runs
   holding a device's mutex or turn takes a guard, so a guard never waits
   on what waits for it. */
static int guard_try(struct rig_sub *s) {
  int free = 0;
  return atomic_compare_exchange_strong(&s->busy, &free, 1);
}

/* A waiter counts itself under the mutex before it tries, and a giver
   frees the guard before it reads the count: either the try sees the
   guard free, or the giver sees the waiter and wakes it under the mutex,
   which the waiter holds until it waits. [s] is read with the runtime
   released: [Submission.submit] uses the submission after the call, which
   keeps its custom block, whose finaliser frees [s], reachable. */
value caml_rig_sub_take(value v_s) {
  struct rig_sub *s = *(struct rig_sub **)Data_custom_val(v_s);
  if (guard_try(s)) return Val_unit;
  caml_enter_blocking_section_no_pending();
  mutex_lock(&s->guard);
  atomic_fetch_add(&s->waiting, 1);
  while (!guard_try(s)) cond_wait(&s->freed, &s->guard);
  atomic_fetch_sub(&s->waiting, 1);
  mutex_unlock(&s->guard);
  caml_leave_blocking_section();
  return Val_unit;
}

value caml_rig_sub_give(value v_s) {
  struct rig_sub *s = *(struct rig_sub **)Data_custom_val(v_s);
  atomic_store(&s->busy, 0);
  if (atomic_load(&s->waiting) > 0) {
    mutex_lock(&s->guard);
    cond_broadcast(&s->freed);
    mutex_unlock(&s->guard);
  }
  return Val_unit;
}

/* Submits [s] on its device: the turn, room, the value, the hand-over and
   the stamps, in one call that runs no OCaml. [s] is read with the runtime
   released: [Submission.submit] uses it after the call, which keeps its
   custom block reachable until then. */
value caml_rig_submit(value v_s) {
  struct rig_sub *s = *(struct rig_sub **)Data_custom_val(v_s);
  struct rig_device *d = s->dev;
  int released;
  if (!lock_turn(d, 1, &released)) return Val_int(SUBMIT_BUSY);
  int r = is_lost(d) ? SUBMIT_LOST : record_waits(d, s);
  if (r != SUBMIT_OK) return Val_int(give_turn(d, released, r));
  begin_turn(d);
  r = admit(d, s);
  end_turn(d);
  if (r == SUBMIT_NO_ROOM || r == SUBMIT_NEVER) d->nrecord -= s->nwaits;
  if (r == SUBMIT_FAILED) r = fail_locked(d, s->why);
  else if (r == SUBMIT_OK && is_lost(d)) r = SUBMIT_LOST_AFTER;
  return Val_int(give_turn(d, released, r));
}

/* Committing */

/* Commits [d]'s submitted work under the turn ([rig_commit_fn]), unless it
   is committed. The mutex is held, the turn free. */
static int commit(struct rig_device *d) {
  uint64_t v = atomic_load(&d->submitted);
  if (atomic_load(&d->committed) >= v) return SUBMIT_OK;
  const char *why = NULL;
  begin_turn(d);
  int r = d->driver.commit(d->self, v, &why);
  end_turn(d);
  if (r == RIG_OK) {
    atomic_store_explicit(&d->committed, v, memory_order_release);
    return SUBMIT_OK;
  }
  return fail_locked(d, why != NULL ? why : "the driver's commit failed");
}

/* Commits [d]'s submitted work, waiting for the turn if [v_wait] and
   skipping the commit if the turn is held otherwise. Answers SUBMIT_OK,
   SUBMIT_BUSY (no commit: the turn stayed held), SUBMIT_LOST, or
   SUBMIT_FAILED if the commit failed and lost [d]. Releases the runtime
   while it waits for the turn and, for a driver that may block, while it
   commits. */
value caml_rig_commit(value v_d, value v_wait) {
  struct rig_device *d = Device_val(v_d);
  int released;
  if (!lock_turn(d, Bool_val(v_wait), &released)) return Val_int(SUBMIT_BUSY);
  int r = is_lost(d) ? SUBMIT_LOST : commit(d);
  return Val_int(give_turn(d, released, r));
}
