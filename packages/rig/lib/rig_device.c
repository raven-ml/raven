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

/* How long a waiter for a turn or a lock blocks before it returns to OCaml,
   where pending signals run: the still interval of waits. */
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
static void cond_broadcast(rig_cond *c) { WakeAllConditionVariable(c); }
static void cond_wait_ms(rig_cond *c, rig_mutex *m, int ms) {
  SleepConditionVariableSRW(c, m, (DWORD)ms, 0);
}
#else
static void mutex_init(rig_mutex *m) { pthread_mutex_init(m, NULL); }
static int mutex_try(rig_mutex *m) { return pthread_mutex_trylock(m) == 0; }
static void mutex_lock(rig_mutex *m) { pthread_mutex_lock(m); }
static void mutex_unlock(rig_mutex *m) { pthread_mutex_unlock(m); }
/* A condition waits on a clock no step of the wall clock moves: macOS has no
   clock attribute for conditions and waits for the time left; elsewhere the
   condition waits on CLOCK_MONOTONIC. */
static void cond_init(rig_cond *c) {
#ifdef __APPLE__
  pthread_cond_init(c, NULL);
#else
  pthread_condattr_t monotonic;
  pthread_condattr_init(&monotonic);
  pthread_condattr_setclock(&monotonic, CLOCK_MONOTONIC);
  pthread_cond_init(c, &monotonic);
  pthread_condattr_destroy(&monotonic);
#endif
}
static void cond_broadcast(rig_cond *c) { pthread_cond_broadcast(c); }
static void cond_wait_ms(rig_cond *c, rig_mutex *m, int ms) {
  struct timespec t = {ms / 1000, (long)(ms % 1000) * 1000000};
#ifdef __APPLE__
  pthread_cond_timedwait_relative_np(c, m, &t);
#else
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  t.tv_sec += now.tv_sec;
  t.tv_nsec += now.tv_nsec;
  if (t.tv_nsec >= 1000000000) {
    t.tv_sec += 1;
    t.tv_nsec -= 1000000000;
  }
  pthread_cond_timedwait(c, m, &t);
#endif
}
#endif

static void cv_wait_ms(struct rig_device *d, int ms) {
  cond_wait_ms(&d->cv, &d->mu, ms);
}

static void mu_init(struct rig_device *d) {
  mutex_init(&d->mu);
  cond_init(&d->cv);
  mutex_init(&d->times.mu);
}
static void cv_broadcast(struct rig_device *d) { cond_broadcast(&d->cv); }
static void mu_lock(struct rig_device *d) { mutex_lock(&d->mu); }
static void mu_unlock(struct rig_device *d) { mutex_unlock(&d->mu); }

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

/* The table of devices

   A device's line lives at its index for the life of the process; its
   record lives while its OCaml record or a submission holds it. Code that
   walks the records holds [table_mu], which a record's free takes to leave
   the table, and holds no runtime lock meanwhile. Code that asks a device
   by index for its timeline reads its line. */

struct rig_line rig_lines[RIG_DEVICES + 1];

static _Atomic(struct rig_device *) devices[RIG_DEVICES];
static _Atomic int top;

/* The devices whose resources are not freed, a bit per index, the host's
   never set. A walk visits these alone, so a process that opened and
   closed many devices walks the few still open. [put] sets a device's bit,
   and [caml_rig_unlive] clears it once the device has stopped and nothing
   holds its word. Readers take no lock. */
static _Atomic uint64_t live[RIG_DEVICES / 64];

static int low_bit(uint64_t w) {
#ifdef _MSC_VER
  unsigned long n;
  _BitScanForward64(&n, w);
  return (int)n;
#else
  return __builtin_ctzll(w);
#endif
}

/* The least index from [i] on whose device's resources are not freed, or 0
   for none. */
static int next_live(int i) {
  for (int k = i / 64; k < RIG_DEVICES / 64; k++) {
    uint64_t w = atomic_load_explicit(&live[k], memory_order_acquire);
    if (k == i / 64) w &= ~(uint64_t)0 << (i % 64);
    if (w != 0) return k * 64 + low_bit(w);
  }
  return 0;
}

value caml_rig_next_live(value v_i) {
  return Val_int(next_live(Int_val(v_i)));
}

/* Takes the device of index [v_i], whose resources are freed, out of the
   walks. */
value caml_rig_unlive(value v_i) {
  int i = Int_val(v_i);
  atomic_fetch_and(&live[i / 64], ~((uint64_t)1 << (i % 64)));
  return Val_unit;
}

#ifdef _WIN32
static rig_mutex table_mu = SRWLOCK_INIT;
#else
static rig_mutex table_mu = PTHREAD_MUTEX_INITIALIZER;
#endif

/* The device record of index [i], or NULL. [table_mu] is held. */
static struct rig_device *device_of(int i) {
  if (i <= 0 || i >= RIG_DEVICES) return NULL;
  return atomic_load_explicit(&devices[i], memory_order_acquire);
}

static struct rig_line *line_of(int i) { return &rig_lines[i]; }

/* The last value the word of [l] showed. */
static uint64_t line_word(struct rig_line *l) {
  _Atomic uint64_t *w = atomic_load_explicit(&l->word, memory_order_acquire);
  if (w != NULL) return atomic_load_explicit(w, memory_order_acquire);
  return atomic_load_explicit(&l->seen, memory_order_acquire);
}

static int line_state(struct rig_line *l) {
  return atomic_load_explicit(&l->state, memory_order_acquire);
}

static uint64_t device_word(struct rig_device *d) { return line_word(d->line); }
static int state(struct rig_device *d) { return line_state(d->line); }
static int is_lost(struct rig_device *d) { return state(d) != RIG_LIVE; }

/* Whether [d] is lost and its stop has not returned. */
static int stopping(struct rig_device *d) {
  int s = state(d);
  return s == RIG_LOSING || s == RIG_OWED || s == RIG_STOPPING;
}

/* The word is read before the state: a word the stop raised shows its
   loss. An orphaned device's word is never read: it may be a page the child
   does not map. The host's line, of index 0, is live with no word, so the
   point 0 is done. */
int rig_point_done(uint64_t p) {
  struct rig_line *l = line_of(RIG_INDEX(p));
  if (line_state(l) == RIG_ORPHANED) return 0;
  uint64_t w = line_word(l);
  int s = line_state(l);
  if (s == RIG_LIVE) return w >= RIG_VALUE(p);
  return s != RIG_ORPHANED &&
         atomic_load_explicit(&l->reached, memory_order_relaxed) >=
             RIG_VALUE(p);
}

/* The reached value is stored before the state leaves RIG_LIVE, and never
   after: a point found lost stays lost. */
int rig_point_lost(uint64_t p) {
  struct rig_line *l = line_of(RIG_INDEX(p));
  int s = line_state(l);
  if (s == RIG_LIVE) return 0;
  return s == RIG_ORPHANED ||
         atomic_load_explicit(&l->reached, memory_order_relaxed) <
             RIG_VALUE(p);
}

value caml_rig_done(value v_p) {
  return Val_bool(rig_point_done(Unsigned_long_val(v_p)));
}

/* Locks */

/* A lock of rig's OCaml state, with a condition, owned by a custom block
   and freed with it. Every lock is on one list, under [locks_mu], so a
   forked child can make each anew. */
struct rig_lock {
  rig_mutex mu;
  rig_cond cv;
  struct rig_lock *prev, *next;
};

static struct rig_lock *locks;

#ifdef _WIN32
static rig_mutex locks_mu = SRWLOCK_INIT;
#else
static rig_mutex locks_mu = PTHREAD_MUTEX_INITIALIZER;
#endif

/* The process's loss facts: the devices whose stop is owed; the first
   failure, 0 for none, a device's index for its loss, -1 for [fail]'s;
   [fail]'s reason; and the forks that made this process, which tell a hold
   made before a fork. */
static _Atomic int owed;
static _Atomic int first;
static _Atomic(char *) failed;
static _Atomic int generation;

/* The reasons no line owns: a fork's, and the stand-in's. */
static char forked_why[] = "forked";
static char ended_why[] = "ended";

#define Lock_val(v) (*(struct rig_lock **)Data_custom_val(v))

/* After fork, the locks are made anew, since a thread of the parent may
   have held them; the fork holds [locks_mu] and [table_mu], so the child
   finds both lists whole. Each device is made anew and its calls in flight
   forgotten. A driver's device the child inherits is orphaned, lost with
   the reason "forked" unless it was lost: freeing would call a driver, and
   a word may be a page the child shares with its parent, or does not map.
   An io device's state is its io library's, which decides what a fork does
   to it: rig leaves it as it was, and runs a stop that a thread of the
   parent claimed or owed. A device whose record was freed has stopped: a
   driver's is orphaned too. A device the child opens is its own. */
#ifndef _WIN32
static void forking(void) {
  mutex_lock(&table_mu);
  mutex_lock(&locks_mu);
}

static void forked_parent(void) {
  mutex_unlock(&locks_mu);
  mutex_unlock(&table_mu);
}

static void forked_child(void) {
  mutex_init(&locks_mu);
  mutex_init(&table_mu);
  for (struct rig_lock *l = locks; l != NULL; l = l->next) {
    mutex_init(&l->mu);
    cond_init(&l->cv);
  }
  atomic_fetch_add(&generation, 1);
  int n = atomic_load(&top), owing = 0;
  for (int i = 0; i <= n; i++) {
    struct rig_line *l = line_of(i);
    struct rig_device *d = atomic_load(&devices[i]);
    if (d != NULL) {
      mu_init(d);
      d->turn = d->inside = 0;
    }
    if (l->io) {
      if (d != NULL && stopping(d)) atomic_store(&l->state, RIG_OWED);
      owing += line_state(l) == RIG_OWED;
    } else if (i > 0 && (d != NULL || line_state(l) != RIG_LIVE)) {
      char *none = NULL;
      atomic_compare_exchange_strong(&l->lost, &none, forked_why);
      atomic_store(&l->state, RIG_ORPHANED);
    }
  }
  atomic_store(&owed, owing);
}

static pthread_once_t atfork_once = PTHREAD_ONCE_INIT;
static void atfork(void) {
  pthread_atfork(forking, forked_parent, forked_child);
}
#endif

static void lock_finalize(value v) {
  struct rig_lock *l = Lock_val(v);
  mutex_lock(&locks_mu);
  if (l->prev != NULL) l->prev->next = l->next;
  else locks = l->next;
  if (l->next != NULL) l->next->prev = l->prev;
  mutex_unlock(&locks_mu);
#ifndef _WIN32
  pthread_mutex_destroy(&l->mu);
  pthread_cond_destroy(&l->cv);
#endif
  free(l);
}

static struct custom_operations lock_ops = {
    "rig.lock",         lock_finalize,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

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
  value v = caml_alloc_custom(&lock_ops, sizeof l, 0, 1);
  Lock_val(v) = l;
  mutex_lock(&locks_mu);
  l->prev = NULL;
  l->next = locks;
  if (locks != NULL) locks->prev = l;
  locks = l;
  mutex_unlock(&locks_mu);
  return v;
}

value caml_rig_lock_try(value v_l) {
  return Val_bool(mutex_try(&Lock_val(v_l)->mu));
}

/* Takes the lock [v_l], which a try found held: waits for it without the
   domain lock, which may collect anything but [v_l]. */
value caml_rig_lock_take(value v_l) {
  CAMLparam1(v_l);
  if (take_mutex(&Lock_val(v_l)->mu)) caml_leave_blocking_section();
  CAMLreturn(Val_unit);
}

value caml_rig_lock_give(value v_l) {
  mutex_unlock(&Lock_val(v_l)->mu);
  return Val_unit;
}

/* Gives up the lock [v_l], which the caller holds, until a broadcast on
   it, a spurious wake-up or the still interval, then takes it back; without
   the domain lock meanwhile. Callers wait in a loop on their condition, so a
   wait returns to OCaml at least every still interval, where a pending
   signal's handler runs. */
value caml_rig_lock_wait(value v_l) {
  CAMLparam1(v_l);
  struct rig_lock *l = Lock_val(v_l);
  caml_enter_blocking_section_no_pending();
  cond_wait_ms(&l->cv, &l->mu, STILL_MS);
  caml_leave_blocking_section();
  CAMLreturn(Val_unit);
}

value caml_rig_lock_broadcast(value v_l) {
  cond_broadcast(&Lock_val(v_l)->cv);
  return Val_unit;
}

/* Device records */

/* Frees [d] once nothing holds it: it leaves the table first, so no walk
   reaches it. Its line stays. Its work has stopped, so no driver writes
   its time pairs. */
static void device_free(struct rig_device *d) {
  mutex_lock(&table_mu);
  struct rig_device *expected = d;
  atomic_compare_exchange_strong(&devices[d->index], &expected, NULL);
  mutex_unlock(&table_mu);
#ifndef _WIN32
  pthread_mutex_destroy(&d->mu);
  pthread_cond_destroy(&d->cv);
  pthread_mutex_destroy(&d->times.mu);
#endif
  for (int i = 0; i < d->times.nchunks; i++) free(d->times.chunks[i]);
  free(d->times.chunks);
  free(d->times.free);
  free(d->record);
  free(d->name);
  free(d);
}

void rig_device_ref(struct rig_device *d) { atomic_fetch_add(&d->refs, 1); }

void rig_device_unref(struct rig_device *d) {
  if (atomic_fetch_sub(&d->refs, 1) == 1) device_free(d);
}

static void device_finalize(value v) { rig_device_unref(Device_val(v)); }

static struct custom_operations device_ops = {
    "rig.device",       device_finalize,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

/* A device record of index [v_index] named [v_name], not yet published,
   in a new custom block [*v] that holds it. */
static struct rig_device *record(value *v, value v_index, value v_name) {
  int index = Int_val(v_index);
  struct rig_device *d = calloc(1, sizeof *d);
  char *name = strdup(String_val(v_name));
  if (d == NULL || name == NULL) {
    free(d);
    free(name);
    caml_raise_out_of_memory();
  }
  mu_init(d);
  atomic_init(&d->refs, 1);
  atomic_init(&d->release.refs, 1);
  d->release.device = d;
  d->index = index;
  d->line = line_of(index);
  d->name = name;
  *v = caml_alloc_custom(&device_ops, sizeof d, 0, 1);
  Device_val(*v) = d;
  return d;
}

/* A driver's device of index [v_index] whose driver blocks ([v_may_block]),
   with its driver's C state [v_edge] and its word's host address (0 behind
   a transport). */
value caml_rig_device_new(value v_index, value v_name, value v_may_block,
                          value v_edge, value v_word) {
  CAMLparam5(v_index, v_name, v_may_block, v_edge, v_word);
  CAMLlocal1(v);
  void *self = (void *)Nativeint_val(v_edge);
  if (self == NULL || rig_driver_of(self) == NULL)
    caml_invalid_argument("Rig.open_: the driver's edge names no rig_driver");
  _Atomic uint64_t *word = (_Atomic uint64_t *)Nativeint_val(v_word);
  struct rig_device *d = record(&v, v_index, v_name);
  d->may_block = Bool_val(v_may_block);
  d->self = self;
  d->driver = *rig_driver_of(self);
  atomic_store(&d->line->word, word);
  CAMLreturn(v);
}

/* An io device of index [v_index]: no queue and no word; its state is its
   io library's. */
value caml_rig_io_new(value v_index, value v_name) {
  CAMLparam2(v_index, v_name);
  CAMLlocal1(v);
  struct rig_device *d = record(&v, v_index, v_name);
  d->line->io = 1;
  CAMLreturn(v);
}

static void put(struct rig_device *d) {
  atomic_store(&devices[d->index], d);
  int t = atomic_load(&top);
  while (t < d->index && !atomic_compare_exchange_weak(&top, &t, d->index)) {
  }
  if (d->index > 0)
    atomic_fetch_or(&live[d->index / 64], (uint64_t)1 << (d->index % 64));
}

/* The host's record, of index 0: never lost, with its driver, whose work
   runs at its hand-over and may take long. */
value caml_rig_host_new(value v_name) {
  CAMLparam1(v_name);
  CAMLlocal1(v);
  struct rig_device *d = record(&v, Val_int(0), v_name);
  d->may_block = 1;
  d->self = rig_host_state();
  d->driver = *rig_driver_of(d->self);
  atomic_store(&d->line->word, rig_host_word());
  put(d);
  CAMLreturn(v);
}

/* The timeline of the device of index [v_i], read from its line. */

value caml_rig_word(value v_i) {
  struct rig_line *l = line_of(Int_val(v_i));
  if (line_state(l) == RIG_ORPHANED)
    return Val_long((intnat)atomic_load(&l->seen));
  return Val_long((intnat)line_word(l));
}

/* Records [v] as read from the word of the device of index [v_i], for a
   word behind a transport. */
value caml_rig_set_seen(value v_i, value v) {
  struct rig_line *l = line_of(Int_val(v_i));
  uint64_t seen = atomic_load(&l->seen), w = (uint64_t)Long_val(v);
  while (seen < w && !atomic_compare_exchange_weak(&l->seen, &seen, w)) {
  }
  return Val_unit;
}

value caml_rig_submitted(value v_i) {
  return Val_long((intnat)atomic_load_explicit(
      &line_of(Int_val(v_i))->submitted, memory_order_acquire));
}

value caml_rig_state(value v_i) {
  return Val_int(line_state(line_of(Int_val(v_i))));
}

/* The reason of the device of index [v_i], "" if it is not lost. Its caller
   keeps the device's OCaml record reachable: the record's collection frees
   the reason ([caml_rig_forget]). */
value caml_rig_why(value v_i) {
  char *why =
      atomic_load_explicit(&line_of(Int_val(v_i))->lost, memory_order_acquire);
  return caml_copy_string(why == NULL ? "" : why);
}

/* The address of [v_d]'s release list, valid while [v_d] is reachable. */
value caml_rig_device_release(value v_d) {
  return Val_long((intnat)&Device_val(v_d)->release);
}

value caml_rig_committed(value v_d) {
  struct rig_device *d = Device_val(v_d);
  return Val_long((intnat)atomic_load_explicit(&d->committed,
                                               memory_order_acquire));
}

/* Loss */

/* Loses [d] with the reason [why], which the record keeps: answers 1 if
   this call lost it, 0 if it was lost already. A loss that [counts] is the
   process's first failure if none came before. Its stop is owed once its
   loser finished spreading it ([finish]). The mutex is held. */
static int lose_locked(struct rig_device *d, char *why, int counts) {
  char *none = NULL;
  if (!atomic_compare_exchange_strong_explicit(
          &d->line->lost, &none, why, memory_order_acq_rel, memory_order_acquire))
    return 0;
  atomic_store_explicit(&d->line->reached, device_word(d), memory_order_relaxed);
  atomic_store_explicit(&d->line->state, RIG_LOSING, memory_order_release);
  int zero = 0;
  if (counts) atomic_compare_exchange_strong(&first, &zero, d->index);
  cv_broadcast(d);
  return 1;
}

/* Claims [d]'s stop if it is owed and [d] runs no counted call: the
   claimant runs it. The mutex is held. */
static int claim_locked(struct rig_device *d) {
  if (state(d) != RIG_OWED || d->inside != 0) return 0;
  atomic_store(&d->line->state, RIG_STOPPING);
  atomic_fetch_sub(&owed, 1);
  return 1;
}

/* Ends [d]'s spreading: its stop is owed. The loser claims it next
   ([caml_rig_claim]), or, while a counted call is inside, the call that
   leaves last. The count moves under the mutex, so a counted call that
   leaves after it sees it. */
static void finish(struct rig_device *d) {
  mu_lock(d);
  atomic_store(&d->line->state, RIG_OWED);
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
   spreads from each. Each keeps the reason [shared], a process's failure
   that every device keeps and none frees, or, if it is NULL, "[p's name]
   lost". A device is lost once, so each is visited once. The domain lock
   is released, and [table_mu] is the one mutex held. */
static void spread(struct rig_device *p, char *shared) {
  char *why = shared;
  for (int i = next_live(1); i != 0; i = next_live(i + 1)) {
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
    why = shared; /* [c] keeps it */
    spread(c, shared);
    finish(c);
  }
  if (why != shared) free(why);
}

/* Loses [d] for the reason [v_why], a fault of a counted call and a
   failure of the process if [v_fault]: whether this call lost it. [v_why]
   is copied before the runtime is released. */
value caml_rig_lose(value v_d, value v_why, value v_fault) {
  CAMLparam1(v_d);
  struct rig_device *d = Device_val(v_d);
  char *why = strdup(String_val(v_why));
  if (why == NULL) caml_raise_out_of_memory();
  caml_enter_blocking_section_no_pending();
  mu_lock(d);
  int won = lose_locked(d, why, Bool_val(v_fault));
  if (won) d->faulted = Bool_val(v_fault);
  mu_unlock(d);
  if (won) {
    mutex_lock(&table_mu);
    spread(d, NULL);
    mutex_unlock(&table_mu);
    finish(d);
  } else
    free(why);
  caml_leave_blocking_section();
  CAMLreturn(Val_bool(won));
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
  CAMLparam1(v_d);
  struct rig_device *d = Device_val(v_d);
  put(d);
  char *why = atomic_load(&failed);
  if (why == NULL) CAMLreturn(Val_unit);
  caml_enter_blocking_section_no_pending();
  mu_lock(d);
  int won = lose_locked(d, why, 0);
  mu_unlock(d);
  if (won) finish(d);
  caml_leave_blocking_section();
  CAMLreturn(Val_unit);
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
  mutex_lock(&table_mu);
  for (int i = next_live(1); i != 0; i = next_live(i + 1)) {
    struct rig_device *d = device_of(i);
    if (d == NULL) continue;
    mu_lock(d);
    int won = lose_locked(d, why, 0);
    mu_unlock(d);
    if (!won) continue;
    spread(d, why);
    finish(d);
  }
  mutex_unlock(&table_mu);
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
  int claimed = 0;
  caml_enter_blocking_section_no_pending();
  mutex_lock(&table_mu);
  for (int i = next_live(1); i != 0 && claimed == 0; i = next_live(i + 1)) {
    struct rig_device *d = device_of(i);
    if (d == NULL || state(d) != RIG_OWED) continue;
    mu_lock(d);
    if (claim_locked(d)) claimed = i;
    mu_unlock(d);
  }
  mutex_unlock(&table_mu);
  caml_leave_blocking_section();
  return Val_int(claimed);
}

/* Sets up the line of the stand-in for devices whose OCaml records were
   collected, of index RIG_DEVICES: stopped, lost with the reason "ended",
   and its word and submitted value past every value. Answers its index. */
value caml_rig_forgotten(value unit) {
  (void)unit;
  struct rig_line *l = line_of(RIG_DEVICES);
  atomic_store(&l->final, (uint64_t)Max_long);
  atomic_store(&l->submitted, (uint64_t)Max_long);
  atomic_store(&l->reached, (uint64_t)Max_long);
  atomic_store(&l->lost, ended_why);
  atomic_store(&l->word, &l->final);
  atomic_store(&l->state, RIG_STOPPED);
  return Val_int(RIG_DEVICES);
}

/* Frees the reason of the lost device of index [v_i], whose OCaml record was
   collected: what asks after the device now asks the stand-in. [fail]'s
   reason, which every device it lost shares, stays. */
value caml_rig_forget(value v_i) {
  struct rig_line *l = line_of(Int_val(v_i));
  char *why = atomic_exchange(&l->lost, ended_why);
  if (why != atomic_load(&failed) && why != forked_why && why != ended_why)
    free(why);
  return Val_unit;
}

/* Records that [d]'s stop returned: ENDED for a driver's device, whose
   work may still run, and STOPPED for an io device, whose work is the
   caller's. */
value caml_rig_stopped(value v_d) {
  CAMLparam1(v_d);
  struct rig_device *d = Device_val(v_d);
  int released = take(d);
  atomic_store(&d->line->state, d->line->io ? RIG_STOPPED : RIG_ENDED);
  cv_broadcast(d);
  give(d, released);
  CAMLreturn(Val_unit);
}

/* Moves the device of index [v_i] from ENDED to STOPPED once its word
   reads its submitted value, read last through its driver for a word
   behind a transport: then no work of it runs. Answers whether this call
   moved it. */
value caml_rig_upgrade(value v_i) {
  struct rig_line *l = line_of(Int_val(v_i));
  if (line_state(l) != RIG_ENDED) return Val_false;
  if (line_word(l) < atomic_load(&l->submitted)) return Val_false;
  int ended = RIG_ENDED;
  return Val_bool(
      atomic_compare_exchange_strong(&l->state, &ended, RIG_STOPPED));
}

/* Waits with the domain lock released until [d]'s stop returned, or
   [v_ms] milliseconds passed: whether it returned. */
value caml_rig_await_stop(value v_d, value v_ms) {
  CAMLparam1(v_d);
  struct rig_device *d = Device_val(v_d);
  int64_t until = now_ms() + Long_val(v_ms), left;
  caml_enter_blocking_section_no_pending();
  mu_lock(d);
  while (stopping(d) && (left = until - now_ms()) > 0) cv_wait_ms(d, (int)left);
  int returned = !stopping(d);
  mu_unlock(d);
  caml_leave_blocking_section();
  CAMLreturn(Val_bool(returned));
}

/* Counted calls */

/* Counts a call on [d]: 0 once it is counted, 1 if [d] is lost. */
value caml_rig_enter(value v_d) {
  CAMLparam1(v_d);
  struct rig_device *d = Device_val(v_d);
  int released = take(d);
  int r = 0;
  if (is_lost(d)) r = 1;
  else d->inside++;
  give(d, released);
  CAMLreturn(Val_int(r));
}

/* Ends a counted call on [d]. */
value caml_rig_exit(value v_d) {
  CAMLparam1(v_d);
  struct rig_device *d = Device_val(v_d);
  int released = take(d);
  d->inside--;
  give(d, released);
  CAMLreturn(Val_unit);
}

/* Spins on [d]'s word, yielding the processor with the domain lock
   released, until it reads [v_target], [d] is lost, or [v_ms] milliseconds
   passed. Answers the last value read. The spin counts as a call inside
   [d], so [d]'s stop, and the end of its word, wait for it: it reads the
   word without the domain lock, which no minor collection waits for. */
value caml_rig_spin(value v_d, value v_target, value v_ms) {
  CAMLparam1(v_d);
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
    CAMLreturn(Val_long((intnat)w));
  }
  while ((w = device_word(d)) < target && !is_lost(d) && now_ms() < until)
    relax();
  mu_lock(d);
  d->inside--;
  mu_unlock(d);
  caml_leave_blocking_section();
  CAMLreturn(Val_long((intnat)w));
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
  CAMLparam1(v_d);
  struct rig_device *d = Device_val(v_d);
  caml_enter_blocking_section_no_pending();
  mutex_lock(&table_mu);
  int waited = 0;
  for (int i = next_live(1); i != 0 && !waited; i = next_live(i + 1)) {
    struct rig_device *c = device_of(i);
    if (c == NULL || c == d) continue;
    mu_lock(c);
    uint64_t cw = device_word(c);
    for (int j = 0; j < c->nrecord && !waited; j++)
      waited = c->record[j].producer == d->index && c->record[j].u > cw;
    mu_unlock(c);
  }
  mutex_unlock(&table_mu);
  int moved = 0;
  if (!waited) {
    mu_lock(d);
    _Atomic uint64_t *w = atomic_load(&d->line->word);
    moved = w != &d->line->final && d->inside == 0;
    if (moved) {
      atomic_store(&d->line->final, atomic_load(w != NULL ? w : &d->line->seen));
      atomic_store(&d->line->word, &d->line->final);
    }
    mu_unlock(d);
  }
  caml_leave_blocking_section();
  CAMLreturn(Val_bool(moved));
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
  CAMLparam1(v_d);
  CAMLlocal1(a);
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
  a = caml_alloc_tuple((mlsize_t)n);
  for (int i = 0; i < n; i++) Store_field(a, (mlsize_t)i, Val_int(seen[i]));
  free(seen);
  CAMLreturn(a);
}

/* Whether [d]'s unreached work waits in its queue on a producer's value the
   producer has not reached: [d]'s own work may not have started. */
value caml_rig_waits_in_queue(value v_d) {
  CAMLparam1(v_d);
  struct rig_device *d = Device_val(v_d);
  int released = take(d), waits = 0;
  uint64_t w = device_word(d);
  for (int j = 0; j < d->nrecord && !waits; j++) {
    struct rig_entry *e = &d->record[j];
    waits = e->u > w && !rig_point_done(RIG_POINT(e->producer, e->w));
  }
  give(d, released);
  CAMLreturn(Val_bool(waits));
}

/* Grows [d]'s record so that [v_n] more entries fit. The array is made
   with no mutex held, which a mutex section never waits for. */
value caml_rig_ensure_record(value v_d, value v_n) {
  CAMLparam1(v_d);
  struct rig_device *d = Device_val(v_d);
  int n = Int_val(v_n);
  int released = take(d);
  int want = d->nrecord + n, have = d->crecord;
  give(d, released);
  if (want <= have) CAMLreturn(Val_unit);
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
  CAMLreturn(Val_unit);
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
   it was, for [give]. */
static int lock_turn(struct rig_device *d, int *released) {
  if (!d->may_block) {
    *released = take(d);
    return 1;
  }
  caml_enter_blocking_section_no_pending();
  *released = 1;
  mu_lock(d);
  /* A free turn reads no clock: the submit is on every run's path. */
  if (d->turn && !is_lost(d)) {
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
    mutex_lock(&table_mu);
    spread(d, NULL);
    mutex_unlock(&table_mu);
    finish(d);
  }
  if (released) caml_leave_blocking_section();
  return r;
}

/* Submitting */

/* Records the run [r]'s in-queue waits in [d]'s record, for spread, after
   dropping the entries [d]'s word shows reached, and refuses if a producer
   is lost. The mutex is held. */
static int record_waits(struct rig_device *d, struct rig_run *r) {
  /* Nothing to drop or record reads no word: the device writes its line,
     so a read is a cache miss on every submit. */
  if (d->nrecord == 0 && r->nwaits == 0) return SUBMIT_OK;
  uint64_t w = device_word(d);
  int k = 0;
  for (int j = 0; j < d->nrecord; j++)
    if (d->record[j].u > w) d->record[k++] = d->record[j];
  d->nrecord = k;
  if (r->nwaits == 0) return SUBMIT_OK;
  if (k + r->nwaits > d->crecord) return SUBMIT_NEED_RECORD;
  uint64_t u = atomic_load(&d->line->submitted) + 1;
  for (int j = 0; j < r->nwaits; j++)
    d->record[k + j] =
        (struct rig_entry){r->producers[j], r->waits[j].value, u};
  d->nrecord = k + r->nwaits;
  for (int j = 0; j < r->nwaits; j++) {
    if (line_state(line_of(r->producers[j])) != RIG_LIVE) {
      d->nrecord = k;
      r->producer = r->producers[j];
      return SUBMIT_PRODUCER_LOST;
    }
  }
  return SUBMIT_OK;
}

/* Time pairs */

static uint64_t *pair_at(struct rig_times *t, int k) {
  return t->chunks[k / RIG_TIMES][k % RIG_TIMES];
}

/* A free pair of [t], zeroed, adding a chunk if none is free: its index,
   or -1 if the host has no memory for one. */
static int take_pair(struct rig_times *t) {
  int k = -1;
  mutex_lock(&t->mu);
  if (t->nfree == 0) {
    uint64_t(*chunk)[2] = calloc(RIG_TIMES, sizeof *chunk);
    uint64_t(**chunks)[2] =
        realloc(t->chunks, (size_t)(t->nchunks + 1) * sizeof *chunks);
    int *free_ = realloc(t->free, (size_t)(t->nchunks + 1) * RIG_TIMES *
                                      sizeof *free_);
    if (chunks != NULL) t->chunks = chunks;
    if (free_ != NULL) t->free = free_;
    if (chunk != NULL && chunks != NULL && free_ != NULL) {
      t->chunks[t->nchunks] = chunk;
      for (int i = RIG_TIMES - 1; i >= 0; i--)
        t->free[t->nfree++] = t->nchunks * RIG_TIMES + i;
      t->nchunks++;
    } else
      free(chunk);
  }
  if (t->nfree > 0) {
    k = t->free[--t->nfree];
    uint64_t *p = pair_at(t, k);
    p[0] = p[1] = 0;
  }
  mutex_unlock(&t->mu);
  return k;
}

/* The pair [v_k] of the device [v_d], as [(start, stop)], given back. The
   value it timed is reached, so its driver wrote it. */
value caml_rig_times_read(value v_d, value v_k) {
  CAMLparam2(v_d, v_k);
  CAMLlocal1(v);
  struct rig_times *t = &Device_val(v_d)->times;
  mutex_lock(&t->mu);
  uint64_t *p = pair_at(t, Int_val(v_k));
  uint64_t start = p[0], stop = p[1];
  t->free[t->nfree++] = Int_val(v_k);
  mutex_unlock(&t->mu);
  v = caml_alloc_tuple(2);
  Store_field(v, 0, Val_long((intnat)start));
  Store_field(v, 1, Val_long((intnat)stop));
  CAMLreturn(v);
}

/* Asks [d]'s driver for room for [s]; once the parts fit, assigns the next
   value, stores it as submitted, hands the work over with the run [r]'s
   waits, and raises the stamps. Called under [d]'s turn. */
static int admit(struct rig_device *d, const struct rig_sub *s,
                 struct rig_run *r) {
  /* A submission with copies of slots hands over the run's parts
     ([set_copies]). */
  const struct rig_part *parts = s->nslotted > 0 ? r->parts : s->parts;
  int room = d->driver.room(d->self, parts, s->nparts, s->scratch, r->args);
  if (room == RIG_NEVER) return SUBMIT_NEVER;
  if (room == RIG_LATER) {
    r->no_room_at = atomic_load(&d->line->submitted);
    return SUBMIT_NO_ROOM;
  }
  uint64_t v = atomic_load(&d->line->submitted) + 1;
  atomic_store_explicit(&d->line->submitted, v, memory_order_release);
  const char *why = NULL;
  r->pair = r->timed ? take_pair(&d->times) : -1;
  uint64_t *times = r->pair < 0 ? NULL : pair_at(&d->times, r->pair);
  int a = d->driver.submit(d->self, v, r->waits, r->nwaits, parts,
                           s->nparts, s->scratch, r->args, r->addresses,
                           r->nslots, times, &why);
  rig_sub_raise(s, r, RIG_POINT(d->index, v));
  r->v = v;
  if (a == RIG_COMMITTED)
    atomic_store_explicit(&d->committed, v, memory_order_release);
  if (a != RIG_FAILED) return SUBMIT_OK;
  r->why = why != NULL ? why : "the driver's submit failed";
  return SUBMIT_FAILED;
}

/* Submits [s] on its device with the run [r]: the turn, room, the value,
   the hand-over and the stamps, in one call that runs no OCaml. [s] and
   [r] are read with the runtime released: [Submission.submit] uses both
   after the call, which keeps their custom blocks reachable until then. */
value caml_rig_submit(value v_s, value v_r) {
  const struct rig_sub *s = *(struct rig_sub **)Data_custom_val(v_s);
  struct rig_run *run = *(struct rig_run **)Data_custom_val(v_r);
  struct rig_device *d = s->dev;
  int released;
  if (!lock_turn(d, &released)) return Val_int(SUBMIT_BUSY);
  int r = is_lost(d) ? SUBMIT_LOST : record_waits(d, run);
  if (r != SUBMIT_OK) return Val_int(give_turn(d, released, r));
  begin_turn(d);
  r = admit(d, s, run);
  end_turn(d);
  if (r == SUBMIT_NO_ROOM || r == SUBMIT_NEVER) d->nrecord -= run->nwaits;
  if (r == SUBMIT_FAILED) r = fail_locked(d, run->why);
  else if (r == SUBMIT_OK && is_lost(d)) r = SUBMIT_LOST_AFTER;
  return Val_int(give_turn(d, released, r));
}

/* Committing */

/* Commits [d]'s submitted work under the turn ([rig_commit_fn]), unless it
   is committed. The mutex is held, the turn free. */
static int commit(struct rig_device *d) {
  uint64_t v = atomic_load(&d->line->submitted);
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

/* Commits [d]'s submitted work, waiting for the turn. Answers SUBMIT_OK,
   SUBMIT_BUSY (no commit: the turn stayed held past the still interval),
   SUBMIT_LOST, or SUBMIT_FAILED if the commit failed and lost [d].
   Releases the runtime while it waits for the turn and, for a driver that
   may block, while it commits. */
value caml_rig_commit(value v_d) {
  CAMLparam1(v_d);
  struct rig_device *d = Device_val(v_d);
  int released;
  if (!lock_turn(d, &released)) CAMLreturn(Val_int(SUBMIT_BUSY));
  int r = is_lost(d) ? SUBMIT_LOST : commit(d);
  CAMLreturn(Val_int(give_turn(d, released, r)));
}
