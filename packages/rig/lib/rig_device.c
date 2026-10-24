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
#include <caml/mlvalues.h>
#include <caml/printexc.h>
#include <caml/signals.h>
#include <caml/threads.h>

#include <time.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sched.h>
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
  SUBMIT_NO_MEMORY,
  SUBMIT_GROW
};

/* Mutexes and conditions

   POSIX threads on every system, as in rig_pool.c and rig_remote_link.c:
   on Windows they are winpthreads, which OCaml's runtime links. */

static void mutex_init(rig_mutex *m) { pthread_mutex_init(m, NULL); }
static int mutex_try(rig_mutex *m) { return pthread_mutex_trylock(m) == 0; }
static void mutex_lock(rig_mutex *m) { pthread_mutex_lock(m); }
static void mutex_unlock(rig_mutex *m) { pthread_mutex_unlock(m); }
/* A condition waits on a clock no step of the wall clock moves: macOS and
   Windows wait for the time left; elsewhere the condition waits on
   CLOCK_MONOTONIC. */
static void cond_init(rig_cond *c) {
#if defined(__APPLE__) || defined(_WIN32)
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
#if defined(__APPLE__) || defined(_WIN32)
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

   A device's record lives at its index from its publication until its slot
   is freed, once the device has stopped and nothing of it remains to give
   back ([caml_rig_free]). A line lives for the process. Code that walks the
   records holds [table_mu], so the fork handler finds the table whole, and
   holds no runtime lock meanwhile. Code that asks a point's slot for its
   timeline reads its line. */

struct rig_line rig_lines[RIG_SLOTS];

static _Atomic(struct rig_device *) devices[RIG_SLOTS];
static _Atomic int top;

/* The devices whose stop has not returned, a bit per index, the host's
   never set. Spreads, failures and claims visit these alone, so an open
   and a close cost the same however many devices the process ended. [put]
   sets a device's bit and [caml_rig_stopped] clears it. Readers take no
   lock. */
static _Atomic uint64_t running[RIG_SLOTS / 64];

static int low_bit(uint64_t w) {
#ifdef _MSC_VER
  unsigned long n;
  _BitScanForward64(&n, w);
  return (int)n;
#else
  return __builtin_ctzll(w);
#endif
}

/* The least index from [i] on whose device's stop has not returned, or 0
   for none. */
static int next_running(int i) {
  for (int k = i / 64; k < RIG_SLOTS / 64; k++) {
    uint64_t w = atomic_load_explicit(&running[k], memory_order_acquire);
    if (k == i / 64) w &= ~(uint64_t)0 << (i % 64);
    if (w != 0) return k * 64 + low_bit(w);
  }
  return 0;
}

static void set_running(int i, int on) {
  uint64_t bit = (uint64_t)1 << (i % 64);
  if (on) atomic_fetch_or(&running[i / 64], bit);
  else atomic_fetch_and(&running[i / 64], ~bit);
}

static rig_mutex table_mu = PTHREAD_MUTEX_INITIALIZER;

/* The free slots, taken last freed first, and the next slot no device has
   taken, under [table_mu]. */
static int free_slots[RIG_SLOTS];
static int nfree;
static int fresh = 1;

/* The published device record of index [i], or NULL. */
static struct rig_device *device_of(int i) {
  if (i <= 0 || i >= RIG_SLOTS) return NULL;
  return atomic_load_explicit(&devices[i], memory_order_acquire);
}

/* The least index from [v_i] on of a published device, 0 for none. */
value caml_rig_next(value v_i) {
  int n = atomic_load(&top);
  for (int i = Int_val(v_i); i <= n; i++)
    if (device_of(i) != NULL) return Val_int(i);
  return Val_int(0);
}

static struct rig_line *line_of(int i) { return &rig_lines[i]; }

/* The timeline word of slot [i] in rig's array. */
static _Atomic uint64_t *slot_word(int i) {
  return (_Atomic uint64_t *)((char *)rig_words() + (size_t)i * RIG_LINE);
}

/* The host address of slot [v_i]'s word, which an opener's make is
   given. */
value caml_rig_slot_word(value v_i) {
  return caml_copy_nativeint((intnat)slot_word(Int_val(v_i)));
}

/* The last value the word of [l] showed: 0 for a line no device opened. */
static uint64_t line_word(struct rig_line *l) {
  _Atomic uint64_t *w = atomic_load_explicit(&l->word, memory_order_acquire);
  return w == NULL ? 0 : atomic_load_explicit(w, memory_order_acquire);
}

static int line_state(struct rig_line *l) {
  return atomic_load_explicit(&l->state, memory_order_acquire);
}

/* Whether [d] owns its line: read after the field it guards. */
static int owner(struct rig_device *d) {
  return atomic_load_explicit(&d->line->owner, memory_order_acquire) ==
         d->opened;
}

static uint64_t device_word(struct rig_device *d) {
  uint64_t w = line_word(d->line);
  return owner(d) ? w : d->final;
}

static int state(struct rig_device *d) {
  int s = line_state(d->line);
  return owner(d) ? s : RIG_ENDED;
}

static int is_lost(struct rig_device *d) { return state(d) != RIG_LIVE; }

/* Whether [d] is lost and its stop has not returned. */
static int stopping(struct rig_device *d) {
  int s = state(d);
  return s == RIG_LOSING || s == RIG_OWED || s == RIG_STOPPING;
}

/* The range of the line [l] that holds [v], or NULL: one compare for a
   value above every range, as every value of the slot's device is. Ranges
   are disjoint and newest first, the highest. */
static struct rig_range *range_of(struct rig_line *l, uint64_t v) {
  if (v > atomic_load_explicit(&l->lost_top, memory_order_acquire)) return NULL;
  for (struct rig_range *r = atomic_load_explicit(&l->ranges, memory_order_acquire);
       r != NULL && r->hi >= v; r = r->next)
    if (v > r->lo) return r;
  return NULL;
}

/* The word is read before the state: a word the stop raised shows its
   loss. The ranges are read last: a reader that read the line of a later
   device sees the range its slot's free published before. The point 0 is
   the host's value 0, which its word holds. */
int rig_point_done(uint64_t p) {
  struct rig_line *l = line_of(RIG_INDEX(p));
  uint64_t v = RIG_VALUE(p);
  int done;
  if (line_state(l) == RIG_ORPHANED)
    done = v <= atomic_load_explicit(&l->base, memory_order_acquire);
  else {
    /* A line becomes orphaned only in a forked child, before any other
       thread runs. */
    uint64_t w = line_word(l);
    done = line_state(l) == RIG_LIVE
               ? w >= v
               : atomic_load_explicit(&l->reached, memory_order_acquire) >= v;
  }
  return done && range_of(l, v) == NULL;
}

/* The reached value is stored before the state leaves RIG_LIVE, and never
   after, and a range is never removed: a point found lost stays lost. A
   use that depends on every point of a memory still gets past a lost
   device's read once the device has stopped ([rig_use_followed]): the lost
   point is not done, but the read it stands for no longer runs. */
int rig_point_lost_rest(uint64_t p) {
  struct rig_line *l = line_of(RIG_INDEX(p));
  uint64_t v = RIG_VALUE(p);
  int s = line_state(l);
  int lost = s == RIG_LIVE ? 0
             : s == RIG_ORPHANED
                 ? v > atomic_load_explicit(&l->base, memory_order_acquire)
                 : atomic_load_explicit(&l->reached, memory_order_acquire) < v;
  return lost || range_of(l, v) != NULL;
}

/* Whether the device of the lost point [p] has stopped: the slot's device
   once its stop returned and its word reads its last submitted value, or an
   earlier device of its slot, whose slot passed on only once it had stopped
   ([free_locked]). A device a forked child inherited never stops there.

   The base is read last. A later device's record stores its base before its
   submitted value, and its [put] stores RIG_LIVE after both, each with
   release order ([record]). A reader that sees either with acquire order
   then sees the later base, which is at least [p]'s value. Read
   first, the base could be the stopped device's, and the state the later
   device's RIG_LIVE: the stopped device's read would count as not stopped,
   and a write after it would raise Lost. The word is read after the
   submitted value, which it reaches once the device has stopped. */
static int point_stopped(uint64_t p) {
  struct rig_line *l = line_of(RIG_INDEX(p));
  int ended = line_state(l) == RIG_ENDED;
  uint64_t submitted =
      atomic_load_explicit(&l->submitted, memory_order_acquire);
  if (ended && line_word(l) >= submitted) return 1;
  return RIG_VALUE(p) <= atomic_load_explicit(&l->base, memory_order_acquire);
}

/* A write raises its device's use word too, so a lost [u] above [*w] is a
   read, and a lost [u] at [*w] is a write, which stays lost through [*w]. A
   stopped device's read no longer runs and changed no memory, so the work
   that follows it needs only its last write. [*w] is read after the line:
   a stopped device raises its words no more, so [*w] is still its last
   write, or the write of a later device that took the word, which comes
   after the take folded a lost write into the head ([takeable]). */
uint64_t rig_use_stopped(uint64_t u, _Atomic uint64_t *w) {
  if (!rig_point_lost_rest(u) || !point_stopped(u)) return u;
  return atomic_load_explicit(w, memory_order_acquire);
}

/* The device whose range of its slot holds the point [v_p]: [Some d] if
   one does. */
value caml_rig_range_device(value v_p) {
  CAMLparam1(v_p);
  uint64_t p = Unsigned_long_val(v_p);
  struct rig_range *r = range_of(line_of(RIG_INDEX(p)), RIG_VALUE(p));
  if (r == NULL) CAMLreturn(Val_none);
  CAMLreturn(caml_alloc_some(r->device));
}

value caml_rig_done(value v_p) {
  return Val_bool(rig_point_done(Unsigned_long_val(v_p)));
}

value caml_rig_lost(value v_p) {
  return Val_bool(rig_point_lost(Unsigned_long_val(v_p)));
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

/* C mutexes nest in the order [table_mu], a device's [mu], its [times.mu],
   [locks_mu]. */

static rig_mutex locks_mu = PTHREAD_MUTEX_INITIALIZER;

/* The process's loss facts: the first failure, 0 for none, a device's index
   for its loss, -1 for [fail]'s; and [fail]'s reason. */
static _Atomic int first;
static _Atomic(char *) failed;

/* The reasons no line owns: a fork's, and the one a loss keeps when its own
   could not be copied. */
static char forked_why[] = "forked";
static char no_memory_why[] = "out of memory";

#define Lock_val(v) (*(struct rig_lock **)Data_custom_val(v))

/* The fork holds [locks_mu] and [table_mu], so the child finds both lists
   whole, and its one thread, which took them, gives them back. The other
   locks are made anew, since a thread of the parent may have held them.
   Each device is made anew and its calls in flight
   forgotten. A driver's device the child inherits is orphaned, lost with
   the reason "forked" unless it was lost: freeing would call a driver, and
   a word may be a page the child shares with its parent, or does not map.
   An io device's state is its io library's, which decides what a fork does
   to it: rig leaves a live one as it was, as POSIX leaves a child its
   parent's open files. One whose stop has not returned is orphaned, since
   that stop, owed or under way, is the parent's: the child would end state
   that the parent's thread is still using. One whose stop returned stays
   as it is, and the child frees its memory as any process does. An
   orphaned device's list is ended, as a stopped one's: the child makes no
   call on it, so any drain routes what is collected onto it. A device the
   child opens is its own. */
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
  mutex_unlock(&locks_mu);
  mutex_unlock(&table_mu);
  for (struct rig_lock *l = locks; l != NULL; l = l->next) {
    mutex_init(&l->mu);
    cond_init(&l->cv);
  }
  int n = atomic_load(&top);
  for (int i = 0; i <= n; i++) {
    struct rig_line *l = line_of(i);
    struct rig_device *d = atomic_load(&devices[i]);
    if (d == NULL) continue;
    mu_init(d);
    d->turn = d->inside = 0;
    if (d->io) {
      if (stopping(d)) {
        atomic_store(&l->state, RIG_ORPHANED);
        set_running(i, 0);
        rig_release_ended(&d->release);
      }
    } else if (i > 0) {
      char *none = NULL;
      atomic_compare_exchange_strong(&d->lost, &none, forked_why);
      atomic_store(&l->state, RIG_ORPHANED);
      set_running(i, 0);
      rig_release_ended(&d->release);
    }
  }
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
  pthread_mutex_destroy(&l->mu);
  pthread_cond_destroy(&l->cv);
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

/* rig takes a lock only on a thread that holds none (lock.mli): one that
   holds a lock is in a callback whose call skipped its check, and the take
   would wait for good, or take two locks in no fixed order. A try answers
   that it took nothing, and the take that follows it raises. */

/* Takes the lock [v_l] if it is free and the thread holds no lock: whether
   it took it. */
value caml_rig_lock_try(value v_l) {
  if (rig_hold() != 0) {
    rig_unhold();
    return Val_false;
  }
  if (mutex_try(&Lock_val(v_l)->mu)) return Val_true;
  rig_unhold();
  return Val_false;
}

/* Takes the lock [v_l], which a try did not take: waits for it without the
   domain lock, which may collect anything but [v_l]. Raises
   Invalid_argument, having taken nothing, if the thread holds a lock. */
value caml_rig_lock_take(value v_l) {
  CAMLparam1(v_l);
  if (rig_hold() != 0) {
    rig_unhold();
    caml_invalid_argument("Rig: called from a signal handler, finaliser or "
                          "Gc.Memprof callback while this thread holds a "
                          "lock");
  }
  if (take_mutex(&Lock_val(v_l)->mu)) caml_leave_blocking_section();
  CAMLreturn(Val_unit);
}

value caml_rig_lock_give(value v_l) {
  mutex_unlock(&Lock_val(v_l)->mu);
  rig_unhold();
  return Val_unit;
}

/* Gives up the lock [v_l], which the caller holds, until a broadcast on
   it, a spurious wake-up or the still interval, then gives it back with its
   count, without the domain lock meanwhile. It returns to OCaml holding no
   lock, so a pending signal's handler that runs there holds none either. */
value caml_rig_lock_wait(value v_l) {
  CAMLparam1(v_l);
  struct rig_lock *l = Lock_val(v_l);
  caml_enter_blocking_section_no_pending();
  cond_wait_ms(&l->cv, &l->mu, STILL_MS);
  mutex_unlock(&l->mu);
  rig_unhold();
  caml_leave_blocking_section();
  CAMLreturn(Val_unit);
}

value caml_rig_lock_broadcast(value v_l) {
  cond_broadcast(&Lock_val(v_l)->cv);
  return Val_unit;
}

/* Device records */

/* A record goes with its block, unless its slot still holds it: a
   published record stays until its slot is freed, and its block lives as
   long as rig's table of devices holds it. A record that no open published,
   as when an exception ended its open, goes with its block. A freed
   record's tokens have all been routed and its list takes no more, so no
   token points into it; a record whose list still counts one stays, which
   leaks it rather than let the token's finaliser write freed memory. */
static void device_finalize(value v) {
  struct rig_device *d = Device_val(v);
  if (atomic_load_explicit(&devices[d->index], memory_order_acquire) == d ||
      atomic_load(&d->release.count) != 0)
    return;
  pthread_mutex_destroy(&d->mu);
  pthread_cond_destroy(&d->cv);
  pthread_mutex_destroy(&d->times.mu);
  if (d->own_why) free(atomic_load(&d->lost));
  for (int i = 0; i < d->times.nchunks; i++) free(d->times.chunks[i]);
  free(d->times.chunks);
  free(d->times.free);
  free(d->times.owed);
  free(d->record);
  free(d->name);
  free(d);
}

static struct custom_operations device_ops = {
    "rig.device",       device_finalize,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

/* A device record of index [v_index] named [v_name], not yet published,
   in a new custom block [*v] that names it. The record lives for the
   process once published. Its timeline starts after the value its slot's
   word holds. */
static struct rig_device *record(value *v, value v_index, value v_name,
                                 intnat opened) {
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
  d->line = line_of(index);
  d->name = name;
  d->opened = opened;
  /* The owner comes first, then the base ([struct rig_line]). */
  struct rig_line *l = d->line;
  atomic_store_explicit(&l->owner, opened, memory_order_release);
  _Atomic uint64_t *word = slot_word(index);
  uint64_t base = atomic_load_explicit(word, memory_order_acquire);
  d->base = base;
  atomic_store(&d->committed, base);
  atomic_store_explicit(&l->base, base, memory_order_release);
  atomic_store_explicit(&l->reached, base, memory_order_release);
  atomic_store_explicit(&l->submitted, base, memory_order_release);
  atomic_store_explicit(&l->word, word, memory_order_release);
  *v = caml_alloc_custom(&device_ops, sizeof d, 0, 1);
  Device_val(*v) = d;
  return d;
}

/* A driver's device of index [v_index] whose driver blocks ([v_may_block]),
   with its driver's C state [v_edge]. */
value caml_rig_device_new(value v_index, value v_name, value v_may_block,
                          value v_edge, value v_opened) {
  CAMLparam5(v_index, v_name, v_may_block, v_edge, v_opened);
  CAMLlocal1(v);
  void *self = (void *)Nativeint_val(v_edge);
  if (self == NULL || rig_driver_of(self) == NULL)
    caml_invalid_argument("Rig.open_: the driver's edge names no rig_driver");
  struct rig_device *d = record(&v, v_index, v_name, Long_val(v_opened));
  d->may_block = Bool_val(v_may_block);
  d->self = self;
  d->driver = *rig_driver_of(self);
  CAMLreturn(v);
}

/* An io device of index [v_index]: no queue and no word; its state is its
   io library's. */
value caml_rig_io_new(value v_index, value v_name, value v_opened) {
  CAMLparam3(v_index, v_name, v_opened);
  CAMLlocal1(v);
  struct rig_device *d = record(&v, v_index, v_name, Long_val(v_opened));
  d->io = 1;
  CAMLreturn(v);
}

static void put(struct rig_device *d) {
  atomic_store_explicit(&d->line->state, RIG_LIVE, memory_order_release);
  atomic_store(&devices[d->index], d);
  int t = atomic_load(&top);
  while (t < d->index && !atomic_compare_exchange_weak(&top, &t, d->index)) {
  }
  if (d->index > 0) set_running(d->index, 1);
}

/* Threads */

/* A number for the calling thread that no other thread of the process
   has, given at its first call. An open records its thread's, so that a
   call on that thread that would wait for the open, from its opener or a
   callback inside it, raises instead. */
static _Atomic intptr_t threads_numbered;
static _Thread_local intptr_t thread_number;

intptr_t rig_thread_number(void) {
  if (thread_number == 0)
    thread_number = atomic_fetch_add(&threads_numbered, 1) + 1;
  return thread_number;
}

value caml_rig_thread(value unit) {
  (void)unit;
  return Val_long(rig_thread_number());
}

/* Slots */

/* An open's slot, an [int Atomic.t] (dev.ml's [request]): a take and a
   give store it in the same call as the move of the slot, which runs no
   OCaml code, so no asynchronous exception comes between them. An index is
   an immediate, which a plain store writes. */
#define Slot_field(v) (*(_Atomic value *)&Field(v, 0))

/* Takes a slot for an open: the last freed one, or one no device has taken;
   stores it in [v_a] and answers it, or 0 if every slot is held. */
value caml_rig_take(value v_a) {
  CAMLparam1(v_a);
  int released = take_mutex(&table_mu), i = 0;
  if (nfree > 0) i = free_slots[--nfree];
  else if (fresh < RIG_SLOTS) i = fresh++;
  mutex_unlock(&table_mu);
  if (released) caml_leave_blocking_section();
  atomic_store(&Slot_field(v_a), Val_int(i));
  CAMLreturn(Val_int(i));
}

/* A slot whose word passed this at its free is retired: every device that
   takes a slot then has at least 2^46 values before its timeline ends, and
   only one that makes that many submissions itself is lost for it. */
#define RETIRED (UINT64_C(1) << 46)

static void push(int i) { free_slots[nfree++] = i; }

/* Gives back the slot an open stored in [v_a], and zeroes [v_a]: giving it
   twice gives it once, and a slot that [caml_rig_publish] marked, negating
   it, stays with its device. */
value caml_rig_give(value v_a) {
  CAMLparam1(v_a);
  int i = Int_val(atomic_exchange(&Slot_field(v_a), Val_int(0)));
  if (i > 0) {
    int released = take_mutex(&table_mu);
    push(i);
    mutex_unlock(&table_mu);
    if (released) caml_leave_blocking_section();
  }
  CAMLreturn(Val_unit);
}

/* Frees the slot of [d] if [d] has stopped: its stop returned and its word
   shows its last value, so nothing of it writes the word again (the stop's
   obligation, [Rig_edge.Driver.stop]); its release list has routed every
   token; and it is not the process's first failure, which [Rig.failure]
   names by its slot. The caller saw the rest of its memory given back. A
   device lost before it reached every value it was given leaves those
   values on the slot as the range [r], which the caller made and which then
   holds the device: [*used] says so. The range is published before the slot
   is given back. The record keeps final copies of its line and answers from
   them from then on; the slot goes to a later device, whose timeline goes
   on from the word, unless the word passed RETIRED. Answers whether it
   freed the slot: 0 for a slot freed before, so that one call alone answers
   1. Under [table_mu] and [d]'s mutex. */
static int free_locked(struct rig_device *d, struct rig_range *r, int *used) {
  if (atomic_load(&devices[d->index]) != d) return 0;
  struct rig_line *l = d->line;
  uint64_t w = line_word(l), v = atomic_load(&l->submitted);
  uint64_t reached = atomic_load_explicit(&l->reached, memory_order_acquire);
  if (d->index <= 0 || line_state(l) != RIG_ENDED || w < v ||
      atomic_load(&first) == d->index || (reached < v && r == NULL))
    return 0;
  /* A token made from here on is refused, and one made before is seen. */
  atomic_store(&d->release.closed, 1);
  if (atomic_load(&d->release.count) != 0) {
    atomic_store(&d->release.closed, 0);
    return 0;
  }
  if (reached < v) {
    r->lo = reached;
    r->hi = v;
    r->next = atomic_load(&l->ranges);
    atomic_store_explicit(&l->ranges, r, memory_order_release);
    atomic_store_explicit(&l->lost_top, v, memory_order_release);
    *used = 1;
  }
  d->final = w;
  /* The line names no open until the next device's record: the record
     answers its final value even if that device's make raises the word. */
  atomic_store_explicit(&l->owner, -1, memory_order_release);
  atomic_store(&devices[d->index], NULL);
  if (w <= RETIRED) push(d->index);
  return 1;
}

/* Frees the slot of [v_d], the C record of the device [v_dev]
   ([free_locked]). The range a lost device may leave is made first, with
   the runtime, which its root needs, and given back if the free left
   none. */
value caml_rig_free(value v_d, value v_dev) {
  CAMLparam2(v_d, v_dev);
  struct rig_device *d = Device_val(v_d);
  struct rig_range *r = NULL;
  if (atomic_load(&devices[d->index]) == d &&
      atomic_load(&d->line->reached) < atomic_load(&d->line->submitted)) {
    r = malloc(sizeof *r);
    if (r != NULL) {
      r->device = v_dev;
      caml_register_generational_global_root(&r->device);
    }
  }
  int used = 0;
  caml_enter_blocking_section_no_pending();
  mutex_lock(&table_mu);
  mu_lock(d);
  int freed = free_locked(d, r, &used);
  mu_unlock(d);
  mutex_unlock(&table_mu);
  caml_leave_blocking_section();
  if (r != NULL && !used) {
    caml_remove_generational_global_root(&r->device);
    free(r);
  }
  CAMLreturn(Val_bool(freed));
}

/* One past [v_d]'s last value, read under its mutex, which every submit
   checks its waits under ([record_waits]). At most one submit is between
   that check and storing its value ([begin_turn]), and that value is at
   most this one. A record whose slot passed on answers past its final
   value. */
value caml_rig_next_value(value v_d) {
  struct rig_device *d = Device_val(v_d);
  caml_enter_blocking_section_no_pending();
  mu_lock(d);
  uint64_t v = (owner(d) ? atomic_load(&d->line->submitted) : d->final) + 1;
  mu_unlock(d);
  caml_leave_blocking_section();
  return Val_long((intnat)v);
}

/* The host's record, of index 0: never lost, with its driver, whose work
   runs at its hand-over and may take long. */
value caml_rig_host_new(value v_name) {
  CAMLparam1(v_name);
  CAMLlocal1(v);
  struct rig_device *d = record(&v, Val_int(0), v_name, 0);
  d->may_block = 1;
  d->self = rig_host_state();
  d->driver = *rig_driver_of(d->self);
  put(d);
  CAMLreturn(v);
}

/* The timeline of the device of index [v_i] whose [opened] is [v_o], read
   from its line while it owns the line; -1 for the word and the last value
   given, and RIG_ENDED for the state, once a later device's record wrote
   it, which its record's [final] then answers. */

static int owns_line(struct rig_line *l, value v_o) {
  return atomic_load_explicit(&l->owner, memory_order_acquire) == Long_val(v_o);
}

value caml_rig_word(value v_i, value v_o) {
  struct rig_line *l = line_of(Int_val(v_i));
  uint64_t w = line_word(l);
  return Val_long(owns_line(l, v_o) ? (intnat)w : -1);
}

value caml_rig_submitted(value v_i, value v_o) {
  struct rig_line *l = line_of(Int_val(v_i));
  uint64_t v = atomic_load_explicit(&l->submitted, memory_order_acquire);
  return Val_long(owns_line(l, v_o) ? (intnat)v : -1);
}

value caml_rig_state(value v_i, value v_o) {
  struct rig_line *l = line_of(Int_val(v_i));
  int s = line_state(l);
  return Val_int(owns_line(l, v_o) ? s : RIG_ENDED);
}

value caml_rig_final(value v_d) {
  return Val_long((intnat)Device_val(v_d)->final);
}

/* The value [v_d]'s timeline started after. */
value caml_rig_base(value v_d) {
  return Val_long((intnat)Device_val(v_d)->base);
}

/* Raises the slot word of the device of index [v_i] to [v], read from the
   device's own word, which lies elsewhere. */
value caml_rig_raise_word(value v_i, value v) {
  rig_raise(slot_word(Int_val(v_i)), (uint64_t)Long_val(v));
  return Val_unit;
}

/* The reason of the device of the record [v_d], "" if it is not lost. */
value caml_rig_why(value v_d) {
  char *why = atomic_load_explicit(&Device_val(v_d)->lost, memory_order_acquire);
  return caml_copy_string(why == NULL ? "" : why);
}

/* The address of [v_d]'s release list. */
value caml_rig_device_release(value v_d) {
  return Val_long((intnat)&Device_val(v_d)->release);
}

value caml_rig_committed(value v_d) {
  struct rig_device *d = Device_val(v_d);
  return Val_long((intnat)atomic_load_explicit(&d->committed,
                                               memory_order_acquire));
}

/* Loss */

/* Loses [d] with the reason [why], which the record keeps, and frees if
   [own]: answers 1 if this call lost it, 0 if it was lost already. A loss
   that [counts] is the process's first failure if none came before. Its
   stop is owed once its loser finished spreading it ([finish]). The mutex
   is held. */
static int lose_locked(struct rig_device *d, char *why, int own, int counts) {
  char *none = NULL;
  if (!atomic_compare_exchange_strong_explicit(
          &d->lost, &none, why, memory_order_acq_rel, memory_order_acquire))
    return 0;
  d->own_why = own;
  atomic_store_explicit(&d->line->reached, device_word(d), memory_order_release);
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
  return 1;
}

/* Ends [d]'s spreading: its stop is owed, and a close or an open of its
   name claims it ([claim_locked]). The state moves under the mutex, so a
   counted call that leaves after it sees it, and a close waiting for the
   stop wakes to claim it. */
static void finish(struct rig_device *d) {
  mu_lock(d);
  atomic_store(&d->line->state, RIG_OWED);
  cv_broadcast(d);
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
   lost", or "out of memory" if that cannot be made. A device is lost once,
   so each is visited once. The domain lock is released, and [table_mu] is
   the one mutex held. */
static void spread(struct rig_device *p, char *shared) {
  char *why = shared;
  for (int i = next_running(1); i != 0; i = next_running(i + 1)) {
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
        won = why != NULL ? lose_locked(c, why, why != shared, 1)
                          : lose_locked(c, no_memory_why, 0, 1);
    }
    mu_unlock(c);
    if (!won) continue;
    why = shared; /* [c] keeps it */
    spread(c, shared);
    finish(c);
  }
  if (why != shared) free(why);
}

/* Loses [d] with the reason [why] as [lose_locked] does, a fault if
   [fault], and ends [ending] counted calls, under the mutex taken once: a
   loss this call won is spread and its stop owed. [why] is freed if [own]
   and the loss was not this call's. Whether this call lost [d]. The domain
   lock is released here, after the caller read its values. */
static int lose_ending(struct rig_device *d, char *why, int own, int counts,
                       int fault, int ending) {
  caml_enter_blocking_section_no_pending();
  mu_lock(d);
  int won = lose_locked(d, why, own, counts);
  if (won) d->faulted = fault;
  d->inside -= ending;
  if (ending && d->inside == 0 && state(d) == RIG_OWED) cv_broadcast(d);
  mu_unlock(d);
  if (won) {
    mutex_lock(&table_mu);
    spread(d, NULL);
    mutex_unlock(&table_mu);
    finish(d);
  } else if (own)
    free(why);
  caml_leave_blocking_section();
  return won;
}

/* Ends a counted call on [d] that loses nothing. The last one out of an
   owed device wakes a close waiting to claim its stop. */
static void leave_count(struct rig_device *d) {
  int released = take(d);
  if (--d->inside == 0 && state(d) == RIG_OWED) cv_broadcast(d);
  give(d, released);
}

/* Loses [d] for the reason [v_why], a fault of a counted call and a
   failure of the process if [v_fault]: whether this call lost it. [v_why]
   is copied before the runtime is released. */
value caml_rig_lose(value v_d, value v_why, value v_fault) {
  struct rig_device *d = Device_val(v_d);
  int fault = Bool_val(v_fault);
  char *why = strdup(String_val(v_why));
  if (why == NULL) caml_raise_out_of_memory();
  return Val_bool(lose_ending(d, why, 1, fault, fault, 0));
}

/* Ends the counted call on [v_d] that raised [v_exn], losing [v_d] first
   if [v_exn] is of [v_fault], the constructor of its driver's faults, or
   of one the scope [v_scope] names: none other for [Faults] (0), every
   other for [Any] (1), every other but its constructor's for [Any_but].
   A fault's loss keeps its message and counts as a fault
   ([caml_rig_faulted]); another exception's keeps its constructor's name
   and its integer and string arguments ([caml_format_exception]), or "out
   of memory" if the C heap has no room. Either is the process's failure if
   none came before. The loss and the end of the count take the mutex once,
   and no OCaml runs in this call, so an exception that lands in the
   caller's handler after it finds both done. Answers whether [v_exn] is a
   fault. */
value caml_rig_raised(value v_d, value v_exn, value v_fault, value v_scope) {
  struct rig_device *d = Device_val(v_d);
  value k = Tag_val(v_exn) == 0 ? Field(v_exn, 0) : v_exn;
  int fault = k == v_fault;
  int loses = fault || v_scope == Val_int(1) ||
              (Is_block(v_scope) && Field(v_scope, 0) != k);
  if (!loses) {
    leave_count(d);
    return Val_false;
  }
  char *text = fault ? NULL : caml_format_exception(v_exn);
  const char *given = fault ? String_val(Field(v_exn, 1)) : text;
  char *why = given != NULL ? strdup(given) : NULL;
  if (text != NULL) caml_stat_free(text);
  int own = why != NULL;
  lose_ending(d, own ? why : no_memory_why, own, 1, fault, 1);
  return Val_bool(fault);
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
value caml_rig_publish(value v_d, value v_a) {
  CAMLparam2(v_d, v_a);
  struct rig_device *d = Device_val(v_d);
  put(d);
  atomic_store(&Slot_field(v_a), Val_int(-d->index));
  char *why = atomic_load(&failed);
  if (why == NULL) CAMLreturn(Val_unit);
  caml_enter_blocking_section_no_pending();
  mu_lock(d);
  int won = lose_locked(d, why, 0, 0);
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
  for (int i = next_running(1); i != 0; i = next_running(i + 1)) {
    struct rig_device *d = device_of(i);
    if (d == NULL) continue;
    mu_lock(d);
    int won = lose_locked(d, why, 0, 0);
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

value caml_rig_still_ms(value unit) {
  (void)unit;
  return Val_int(STILL_MS);
}

/* The process's first failure: 0 for none, the index of the device whose
   loss it was, or -1 for [fail]'s. */
value caml_rig_first(value unit) {
  (void)unit;
  return Val_int(atomic_load(&first));
}

/* Claims the stop of [v_d] if it is owed and [v_d] runs no counted call:
   whether the caller runs it. */
value caml_rig_claim_own(value v_d) {
  struct rig_device *d = Device_val(v_d);
  int released = take(d);
  int claimed = claim_locked(d);
  give(d, released);
  return Val_bool(claimed);
}

/* Owes [d]'s stop again, which its claimant gave up before it called the
   driver's stop. */
value caml_rig_unclaim(value v_d) {
  struct rig_device *d = Device_val(v_d);
  int released = take(d);
  atomic_store(&d->line->state, RIG_OWED);
  give(d, released);
  return Val_unit;
}

/* Records that [d]'s stop returned: ENDED, out of the walks, and its list
   drained by any device's drain. */
value caml_rig_stopped(value v_d) {
  CAMLparam1(v_d);
  struct rig_device *d = Device_val(v_d);
  int released = take(d);
  atomic_store(&d->line->state, RIG_ENDED);
  set_running(d->index, 0);
  rig_release_ended(&d->release);
  cv_broadcast(d);
  give(d, released);
  CAMLreturn(Val_unit);
}

/* Waits with the domain lock released until [d]'s stop returned, its stop
   can be claimed ([claim_locked]), or [v_ms] milliseconds passed: whether
   it returned. */
value caml_rig_await_stop(value v_d, value v_ms) {
  CAMLparam1(v_d);
  struct rig_device *d = Device_val(v_d);
  int64_t until = now_ms() + Long_val(v_ms), left;
  caml_enter_blocking_section_no_pending();
  mu_lock(d);
  while (stopping(d) && !(state(d) == RIG_OWED && d->inside == 0) &&
         (left = until - now_ms()) > 0)
    cv_wait_ms(d, (int)left);
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
  leave_count(Device_val(v_d));
  return Val_unit;
}

/* Spins on [d]'s word, yielding the processor with the domain lock
   released, until it reads [v_target], [d] is lost, or [v_ms] milliseconds
   passed. Answers the last value read. The word and the line live for the
   process, so the spin holds nothing of [d]. */
value caml_rig_spin(value v_d, value v_target, value v_ms) {
  struct rig_device *d = Device_val(v_d);
  uint64_t target = (uint64_t)Long_val(v_target), w;
  int64_t until = now_ms() + Long_val(v_ms);
  caml_enter_blocking_section_no_pending();
  while ((w = device_word(d)) < target && !is_lost(d) && now_ms() < until)
    relax();
  caml_leave_blocking_section();
  return Val_long((intnat)w);
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
  if (copy != NULL ? lose_locked(d, copy, 1, 1)
                   : lose_locked(d, no_memory_why, 0, 1))
    return SUBMIT_FAILED;
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

/* Records the call [r]'s in-queue waits in [d]'s record, for spread, after
   dropping the entries [d]'s word shows reached, and refuses a wait for a
   point a lost device did not reach, of the slot's device or an earlier
   one: [r]'s [producer] is then that point. A call whose waits read words
   through mappings [d] may give back drops a wait for a point already done,
   so the driver never reads its word: that word, or [d]'s mapping of it,
   may be given back once [d] passed its next value
   ([caml_rig_next_value]). The record grows here: SUBMIT_NO_MEMORY if the
   host has no memory for it. The mutex is held. */
static int record_waits(struct rig_device *d, struct rig_call *r) {
  /* Nothing to drop or record reads no word: the device writes its line,
     so a read is a cache miss on every submit. */
  if (d->nrecord == 0 && r->nwaits == 0) return SUBMIT_OK;
  uint64_t w = device_word(d);
  int k = 0;
  for (int j = 0; j < d->nrecord; j++)
    if (d->record[j].u > w) d->record[k++] = d->record[j];
  d->nrecord = k;
  int n = 0;
  for (int j = 0; j < r->nwaits; j++) {
    uint64_t p = RIG_POINT(r->producers[j], r->waits[j].value);
    if (rig_point_lost(p)) {
      r->producer = p;
      return SUBMIT_PRODUCER_LOST;
    }
    if (r->checks && rig_point_done(p)) continue;
    r->producers[n] = r->producers[j];
    r->waits[n++] = r->waits[j];
  }
  r->nwaits = n;
  if (n == 0) return SUBMIT_OK;
  if (k + n > d->crecord) {
    int c = 2 * (k + n) < 8 ? 8 : 2 * (k + n);
    struct rig_entry *grown = realloc(d->record, (size_t)c * sizeof *grown);
    if (grown == NULL) return SUBMIT_NO_MEMORY;
    d->record = grown;
    d->crecord = c;
  }
  uint64_t u = atomic_load(&d->line->submitted) + 1;
  for (int j = 0; j < n; j++)
    d->record[k + j] =
        (struct rig_entry){r->producers[j], r->waits[j].value, u};
  d->nrecord = k + n;
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
    struct rig_owed *owed = realloc(
        t->owed, (size_t)(t->nchunks + 1) * RIG_TIMES * sizeof *owed);
    if (chunks != NULL) t->chunks = chunks;
    if (free_ != NULL) t->free = free_;
    if (owed != NULL) t->owed = owed;
    if (chunk != NULL && chunks != NULL && free_ != NULL && owed != NULL) {
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

/* The pair [v_k] of the device [v_d], as [(start, stop)]. The value it
   timed is reached, so its driver wrote it. */
value caml_rig_times_read(value v_d, value v_k) {
  CAMLparam2(v_d, v_k);
  CAMLlocal1(v);
  struct rig_times *t = &Device_val(v_d)->times;
  mutex_lock(&t->mu);
  uint64_t *p = pair_at(t, Int_val(v_k));
  uint64_t start = p[0], stop = p[1];
  mutex_unlock(&t->mu);
  v = caml_alloc_tuple(2);
  Store_field(v, 0, Val_long((intnat)start));
  Store_field(v, 1, Val_long((intnat)stop));
  CAMLreturn(v);
}

/* Gives back the pair [v_k] of the device [v_d]. */
value caml_rig_times_free(value v_d, value v_k) {
  struct rig_times *t = &Device_val(v_d)->times;
  mutex_lock(&t->mu);
  t->free[t->nfree++] = Int_val(v_k);
  mutex_unlock(&t->mu);
  return Val_unit;
}

/* Owes the pair [v_k] of the device [v_d], which timed its value [v_v]
   and which no read settled. */
value caml_rig_times_owe(value v_d, value v_k, value v_v) {
  struct rig_times *t = &Device_val(v_d)->times;
  mutex_lock(&t->mu);
  int n = atomic_load_explicit(&t->nowed, memory_order_relaxed);
  t->owed[n] = (struct rig_owed){Int_val(v_k), 0, (uint64_t)Long_val(v_v)};
  atomic_store_explicit(&t->nowed, n + 1, memory_order_release);
  mutex_unlock(&t->mu);
  return Val_unit;
}

/* Takes a pair of the live device [v_d] that is owed, untaken, and whose
   value its word shows: that value, or 0 for none. Its caller settles the
   value and gives the pair back, so no other call takes it too. One load
   while no pair is owed. */
value caml_rig_times_due(value v_d) {
  struct rig_device *d = Device_val(v_d);
  struct rig_times *t = &d->times;
  uint64_t v = 0;
  if (atomic_load_explicit(&t->nowed, memory_order_acquire) == 0 ||
      is_lost(d))
    return Val_long(0);
  uint64_t w = device_word(d);
  mutex_lock(&t->mu);
  int n = atomic_load_explicit(&t->nowed, memory_order_relaxed);
  for (int i = 0; i < n && v == 0; i++) {
    struct rig_owed *o = &t->owed[i];
    if (o->taken || o->v > w) continue;
    o->taken = 1;
    v = o->v;
  }
  mutex_unlock(&t->mu);
  return Val_long((intnat)v);
}

/* Gives back the owed pair of the value [v_v] of the device [v_d], which
   [caml_rig_times_due] took. */
value caml_rig_times_give(value v_d, value v_v) {
  struct rig_times *t = &Device_val(v_d)->times;
  uint64_t v = (uint64_t)Long_val(v_v);
  mutex_lock(&t->mu);
  int n = atomic_load_explicit(&t->nowed, memory_order_relaxed);
  for (int i = 0; i < n; i++) {
    if (t->owed[i].v != v) continue;
    t->free[t->nfree++] = t->owed[i].pair;
    t->owed[i] = t->owed[n - 1];
    atomic_store_explicit(&t->nowed, n - 1, memory_order_release);
    break;
  }
  mutex_unlock(&t->mu);
  return Val_unit;
}

/* Asks [d]'s driver for room for [s]; once the parts fit, assigns the next
   value, stores it as submitted, hands the work over with the call [r]'s
   waits and the blocks [args], and raises the stamps, but for the host.
   Called under [d]'s turn. */
static int admit(struct rig_device *d, const struct rig_sub *s,
                 struct rig_call *r, uint8_t *args) {
  /* A submission with copies of slots hands over the call's parts
     ([set_copies]). */
  const struct rig_part *parts = s->nslotted > 0 ? r->parts : s->parts;
  int room = d->driver.room(d->self, r->nwaits, parts, s->nparts, s->scratch,
                            args);
  if (room == RIG_NEVER) return SUBMIT_NEVER;
  if (room == RIG_GROW) return SUBMIT_GROW;
  if (room == RIG_LATER) {
    r->no_room_at = atomic_load(&d->line->submitted);
    return SUBMIT_NO_ROOM;
  }
  uint64_t v = atomic_load(&d->line->submitted) + 1;
  /* A value past a point's 47 bits would carry into its index: the device
     is lost instead, and assigns nothing. */
  if (v > RIG_VALUE_MASK) {
    r->why = "its timeline reached the last value a point holds, 2^47 - 1";
    return SUBMIT_FAILED;
  }
  atomic_store_explicit(&d->line->submitted, v, memory_order_release);
  const char *why = NULL;
  r->pair = r->timed ? take_pair(&d->times) : -1;
  uint64_t *times = r->pair < 0 ? NULL : pair_at(&d->times, r->pair);
  int a = d->driver.submit(d->self, v, r->waits, r->nwaits, parts,
                           s->nparts, s->scratch, args, r->addresses,
                           r->nslots, times, &why);
  /* The host reserves no word ([add_memory]). */
  if (d->index != 0) rig_sub_raise(s, r, RIG_POINT(d->index, v));
  r->v = v;
  if (a == RIG_COMMITTED)
    atomic_store_explicit(&d->committed, v, memory_order_release);
  if (a != RIG_FAILED) return SUBMIT_OK;
  r->why = why != NULL ? why : "the driver's submit failed";
  return SUBMIT_FAILED;
}

/* Submits [s] on its device with the call [v_c] and the run [v_r]'s
   blocks: the turn, room, the value, the hand-over and the stamps, in one
   call that runs no OCaml. [s] and the run are read with the runtime
   released: [Submission.submit] uses both after the call, which keeps their
   custom blocks reachable until then. A run is an OCaml record whose field
   0 holds its custom block. */
value caml_rig_submit(value v_s, value v_c, value v_r) {
  const struct rig_sub *s = *(struct rig_sub **)Data_custom_val(v_s);
  struct rig_call *c = (struct rig_call *)Long_val(v_c);
  uint8_t *args = (*(struct rig_run **)Data_custom_val(Field(v_r, 0)))->args;
  struct rig_device *d = s->dev;
  int released;
  if (!lock_turn(d, &released)) return Val_int(SUBMIT_BUSY);
  int r = is_lost(d) ? SUBMIT_LOST : record_waits(d, c);
  if (r != SUBMIT_OK) return Val_int(give_turn(d, released, r));
  begin_turn(d);
  r = admit(d, s, c, args);
  end_turn(d);
  if (r == SUBMIT_NO_ROOM || r == SUBMIT_NEVER || r == SUBMIT_GROW)
    d->nrecord -= c->nwaits;
  if (r == SUBMIT_FAILED) r = fail_locked(d, c->why);
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

/* Settling */

/* Has [d]'s driver write the times of its value [v_v] under the turn
   ([rig_settle_fn]). Answers SUBMIT_OK, SUBMIT_BUSY (the turn stayed held
   past the still interval), SUBMIT_LOST, or SUBMIT_FAILED if the settle
   failed and lost [d]. Releases the runtime while it waits for the turn
   and, for a driver that may block, while it settles. */
value caml_rig_settle(value v_d, value v_v) {
  struct rig_device *d = Device_val(v_d);
  uint64_t v = (uint64_t)Long_val(v_v);
  int released;
  if (d->driver.settle == NULL) return Val_int(SUBMIT_OK);
  if (!lock_turn(d, &released)) return Val_int(SUBMIT_BUSY);
  int r = SUBMIT_LOST;
  if (!is_lost(d)) {
    const char *why = NULL;
    begin_turn(d);
    int a = d->driver.settle(d->self, v, &why);
    end_turn(d);
    r = a == RIG_OK
            ? SUBMIT_OK
            : fail_locked(d, why != NULL ? why : "the driver's settle failed");
  }
  return Val_int(give_turn(d, released, r));
}
