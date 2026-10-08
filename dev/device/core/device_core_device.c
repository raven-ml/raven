/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Devices: the turn, the hand-over, loss, spread and the owed stop.

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
#include <caml/signals.h>
#include <caml/threads.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sched.h>
#include <time.h>
#endif

#include "device_core_stubs.h"

/* How long a turn waiter blocks before it returns to OCaml, where pending
   signals run: the still interval of waits. */
#define DC_STILL_MS 200

/* What a submit answers; DC_STOP_CLAIMED is added when the call claimed
   the device's stop, which its caller then runs. */
enum {
  DC_OK,
  DC_BUSY,
  DC_NO_ROOM,
  DC_NEVER,
  DC_LOST,
  DC_LOST_AFTER,
  DC_PRODUCER_LOST,
  DC_FAILED,
  DC_NEED_RECORD
};
#define DC_STOP_CLAIMED 16

/* Mutexes and conditions */

#ifdef _WIN32
static void mu_init(struct dc_device *d) {
  InitializeSRWLock(&d->mu);
  InitializeConditionVariable(&d->cv);
}
static int mu_try(struct dc_device *d) {
  return TryAcquireSRWLockExclusive(&d->mu);
}
static void mu_lock(struct dc_device *d) { AcquireSRWLockExclusive(&d->mu); }
static void mu_unlock(struct dc_device *d) {
  ReleaseSRWLockExclusive(&d->mu);
}
static void cv_wait_ms(struct dc_device *d, int ms) {
  SleepConditionVariableSRW(&d->cv, &d->mu, (DWORD)ms, 0);
}
static void cv_broadcast(struct dc_device *d) {
  WakeAllConditionVariable(&d->cv);
}
#else
static void mu_init(struct dc_device *d) {
  pthread_mutex_init(&d->mu, NULL);
  pthread_cond_init(&d->cv, NULL);
}
static int mu_try(struct dc_device *d) {
  return pthread_mutex_trylock(&d->mu) == 0;
}
static void mu_lock(struct dc_device *d) { pthread_mutex_lock(&d->mu); }
static void mu_unlock(struct dc_device *d) { pthread_mutex_unlock(&d->mu); }
/* A clock jump moves one timeout, which only returns a waiter early or late
   to OCaml. */
static void cv_wait_ms(struct dc_device *d, int ms) {
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
static void cv_broadcast(struct dc_device *d) {
  pthread_cond_broadcast(&d->cv);
}
#endif

/* Takes [d]'s mutex from a stub holding the domain lock: by try-lock, and
   otherwise after releasing the domain lock. Answers whether it released
   it, for [give]. */
static int take(struct dc_device *d) {
  if (mu_try(d)) return 0;
  caml_enter_blocking_section_no_pending();
  mu_lock(d);
  return 1;
}

static void give(struct dc_device *d, int released) {
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

static int64_t now_ms(void) {
#ifdef _WIN32
  return (int64_t)GetTickCount64();
#else
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (int64_t)t.tv_sec * 1000 + t.tv_nsec / 1000000;
#endif
}

/* The table of devices */

static _Atomic(struct dc_device *) devices[DC_DEVICES];
static _Atomic int top;
static _Atomic int forked;

struct dc_device *dc_device_of(int i) {
  if (i <= 0 || i >= DC_DEVICES) return NULL;
  return atomic_load_explicit(&devices[i], memory_order_acquire);
}

uint64_t dc_word(struct dc_device *d) {
  if (d->word != NULL)
    return atomic_load_explicit(d->word, memory_order_acquire);
  return atomic_load_explicit(&d->seen, memory_order_acquire);
}

static int is_lost(struct dc_device *d) {
  return atomic_load_explicit(&d->lost, memory_order_acquire) != NULL;
}

#define Device_val(v) ((struct dc_device *)Long_val(v))

/* After fork, the child sees every device lost, its stop answered
   Unknown for good: freeing would call a driver, and a word may be a page
   the child shares with its parent. The mutexes are made anew, since a
   thread of the parent may have held one. */
#ifndef _WIN32
static void forked_child(void) {
  static char why[] = "forked";
  atomic_store(&forked, 1);
  int n = atomic_load(&top);
  for (int i = 1; i <= n; i++) {
    struct dc_device *d = dc_device_of(i);
    if (d == NULL) continue;
    mu_init(d);
    char *none = NULL;
    atomic_compare_exchange_strong(&d->lost, &none, why);
    atomic_store(&d->answer, DC_UNKNOWN);
    d->turn = d->inside = d->owed = d->spreading = 0;
  }
}

static pthread_once_t atfork_once = PTHREAD_ONCE_INIT;
static void atfork(void) { pthread_atfork(NULL, NULL, forked_child); }
#endif

/* A device of index [v_index] whose driver blocks ([v_may_block]), with its
   C entries and its word's host address (0 behind a transport). */
value caml_device_core_device_new(value v_index, value v_name,
                                  value v_may_block, value v_self,
                                  value v_room, value v_submit,
                                  value v_word) {
  int index = Int_val(v_index);
  if (index <= 0 || index >= DC_DEVICES)
    caml_invalid_argument("device index out of range");
  struct dc_device *d = calloc(1, sizeof *d);
  char *name = strdup(String_val(v_name));
  if (d == NULL || name == NULL) {
    free(d);
    free(name);
    caml_raise_out_of_memory();
  }
  mu_init(d);
  d->index = index;
  d->name = name;
  d->may_block = Bool_val(v_may_block);
  d->self = (void *)Nativeint_val(v_self);
  d->room = (nx_room_fn *)Nativeint_val(v_room);
  d->submit = (nx_submit_fn *)Nativeint_val(v_submit);
  d->word = (_Atomic uint64_t *)Nativeint_val(v_word);
#ifndef _WIN32
  pthread_once(&atfork_once, atfork);
#endif
  atomic_store_explicit(&devices[index], d, memory_order_release);
  int t = atomic_load(&top);
  while (t < index && !atomic_compare_exchange_weak(&top, &t, index)) {
  }
  return Val_long((intnat)d);
}

value caml_device_core_device_new_byte(value *argv, int argn) {
  (void)argn;
  return caml_device_core_device_new(argv[0], argv[1], argv[2], argv[3],
                                     argv[4], argv[5], argv[6]);
}

value caml_device_core_word(value v_d) {
  return Val_long((intnat)dc_word(Device_val(v_d)));
}

/* Records [v] as read from [d]'s word, for a word behind a transport. */
value caml_device_core_set_seen(value v_d, value v) {
  struct dc_device *d = Device_val(v_d);
  uint64_t seen = atomic_load(&d->seen), w = (uint64_t)Long_val(v);
  while (seen < w && !atomic_compare_exchange_weak(&d->seen, &seen, w)) {
  }
  return Val_unit;
}

value caml_device_core_submitted(value v_d) {
  struct dc_device *d = Device_val(v_d);
  return Val_long((intnat)atomic_load_explicit(&d->submitted,
                                               memory_order_acquire));
}

value caml_device_core_is_lost(value v_d) {
  return Val_bool(is_lost(Device_val(v_d)));
}

value caml_device_core_why(value v_d) {
  char *why = atomic_load_explicit(&Device_val(v_d)->lost,
                                   memory_order_acquire);
  return caml_copy_string(why == NULL ? "" : why);
}

value caml_device_core_answer(value v_d) {
  return Val_int(atomic_load(&Device_val(v_d)->answer));
}

value caml_device_core_forked(value unit) {
  (void)unit;
  return Val_bool(atomic_load(&forked));
}

/* Loss */

/* Loses [d] with the reason [why], which the record keeps: answers 1 if
   this call lost it, 0 if it was lost already. Its stop is claimable once
   its loser finished spreading it ([finish]). The mutex is held. */
static int lose_locked(struct dc_device *d, char *why) {
  char *none = NULL;
  if (!atomic_compare_exchange_strong_explicit(
          &d->lost, &none, why, memory_order_acq_rel, memory_order_acquire))
    return 0;
  d->spreading = 1;
  cv_broadcast(d);
  return 1;
}

/* Claims [d]'s stop if [d] is lost, spread, owed its stop and runs no
   counted call: the claimant runs it. The mutex is held. */
static int claim_locked(struct dc_device *d) {
  if (!is_lost(d) || !d->owed || d->spreading || d->inside != 0) return 0;
  d->owed = 0;
  atomic_store(&d->answer, DC_STOPPING);
  return 1;
}

struct claims {
  int n, c;
  int *index;
};

static void claims_push(struct claims *k, int index) {
  if (k->n == k->c) {
    int c = k->c == 0 ? 8 : 2 * k->c;
    int *index = realloc(k->index, (size_t)c * sizeof *index);
    if (index == NULL) return; /* the next caller claims it */
    k->index = index;
    k->c = c;
  }
  k->index[k->n++] = index;
}

static void spread(struct dc_device *p, struct claims *k);

/* Ends [d]'s spreading: its stop is owed, and claimed here if no call is
   inside. */
static void finish(struct dc_device *d, struct claims *k) {
  mu_lock(d);
  d->spreading = 0;
  d->owed = 1;
  if (claim_locked(d)) claims_push(k, d->index);
  mu_unlock(d);
}

/* Loses every device whose unreached work waits in its queue on an
   unreached value of the lost [p], reading both words as they stand, and
   spreads from each. A device is lost once, so each is visited once. The
   domain lock is released and no mutex held. */
static void spread(struct dc_device *p, struct claims *k) {
  int n = atomic_load(&top);
  for (int i = 1; i <= n; i++) {
    struct dc_device *c = dc_device_of(i);
    if (c == NULL || c == p || is_lost(c)) continue;
    size_t len = strlen(p->name) + sizeof " lost";
    char *why = malloc(len);
    if (why != NULL) {
      memcpy(why, p->name, strlen(p->name));
      memcpy(why + strlen(p->name), " lost", sizeof " lost");
    }
    uint64_t pw = dc_word(p);
    int won = 0;
    mu_lock(c);
    uint64_t cw = dc_word(c);
    for (int j = 0; j < c->nrecord && !won; j++) {
      struct dc_entry *e = &c->record[j];
      if (e->producer == p->index && e->w > pw && e->u > cw)
        won = lose_locked(c, why != NULL ? why : p->name);
    }
    mu_unlock(c);
    if (!won) {
      free(why);
      continue;
    }
    spread(c, k);
    finish(c, k);
  }
}

static value claims_value(int won, struct claims *k) {
  value a = caml_alloc_tuple((mlsize_t)k->n + 1);
  Store_field(a, 0, Val_int(won));
  for (int i = 0; i < k->n; i++) Store_field(a, (mlsize_t)i + 1, Val_int(k->index[i]));
  free(k->index);
  return a;
}

/* Loses [d] for the reason [v_why]. The result's first element is 1 if this
   call lost it, then the indices of the devices whose stops it claimed,
   whose stops the caller runs. */
value caml_device_core_lose(value v_d, value v_why) {
  struct dc_device *d = Device_val(v_d);
  char *why = strdup(String_val(v_why));
  if (why == NULL) caml_raise_out_of_memory();
  struct claims k = {0, 0, NULL};
  caml_enter_blocking_section_no_pending();
  mu_lock(d);
  int won = lose_locked(d, why);
  mu_unlock(d);
  if (won) {
    spread(d, &k);
    finish(d, &k);
  } else
    free(why);
  caml_leave_blocking_section();
  return claims_value(won, &k);
}

/* Records [d]'s stop's answer, [DC_STOPPED] or [DC_UNKNOWN]. */
value caml_device_core_set_answer(value v_d, value v_answer) {
  atomic_store(&Device_val(v_d)->answer, Int_val(v_answer));
  return Val_unit;
}

/* Records [Stopped] in place of [Unknown] once [d]'s word reads its
   submitted value: then no work of [d] runs. Answers whether this call
   recorded it. Never in a forked child, which reads no word. */
value caml_device_core_upgrade(value v_d) {
  struct dc_device *d = Device_val(v_d);
  if (atomic_load(&forked) || d->word == NULL) return Val_false;
  if (atomic_load(&d->answer) != DC_UNKNOWN) return Val_false;
  if (dc_word(d) < atomic_load(&d->submitted)) return Val_false;
  int unknown = DC_UNKNOWN;
  return Val_bool(atomic_compare_exchange_strong(&d->answer, &unknown,
                                                 DC_STOPPED));
}

/* Counted calls */

/* Counts a call on [d]: 0 once it is counted, 1 if [d] is lost, 3 if [d]
   is lost and this call claimed its stop. */
value caml_device_core_enter(value v_d) {
  struct dc_device *d = Device_val(v_d);
  int released = take(d);
  int r = 0;
  if (is_lost(d)) r = claim_locked(d) ? 3 : 1;
  else d->inside++;
  give(d, released);
  return Val_int(r);
}

/* Ends a counted call on [d]: answers whether it claimed [d]'s stop. */
value caml_device_core_exit(value v_d) {
  struct dc_device *d = Device_val(v_d);
  int released = take(d);
  d->inside--;
  int claimed = claim_locked(d);
  give(d, released);
  return Val_bool(claimed);
}

/* Spins on [d]'s word, yielding the processor with the domain lock
   released, until it reads [v_target], [d] is lost, or [v_ms] milliseconds
   passed. Answers the last value read. */
value caml_device_core_spin(value v_d, value v_target, value v_ms) {
  struct dc_device *d = Device_val(v_d);
  uint64_t target = (uint64_t)Long_val(v_target), w;
  int64_t until = now_ms() + Long_val(v_ms);
  caml_enter_blocking_section_no_pending();
  while ((w = dc_word(d)) < target && !is_lost(d) && now_ms() < until)
    relax();
  caml_leave_blocking_section();
  return Val_long((intnat)w);
}

/* The producers [d]'s unreached work waits on in its queue, for a wait that
   looks for their unseen faults. */
value caml_device_core_producers(value v_d) {
  CAMLparam1(v_d);
  CAMLlocal1(a);
  struct dc_device *d = Device_val(v_d);
  int released = take(d), n = 0;
  int buf[64];
  uint64_t w = dc_word(d);
  for (int j = 0; j < d->nrecord && n < 64; j++)
    if (d->record[j].u > w) buf[n++] = d->record[j].producer;
  give(d, released);
  a = caml_alloc_tuple((mlsize_t)n);
  for (int i = 0; i < n; i++) Store_field(a, (mlsize_t)i, Val_int(buf[i]));
  CAMLreturn(a);
}

/* Grows [d]'s record so that [v_n] more entries fit. The array is made
   with no mutex held, which a mutex section never waits for. */
value caml_device_core_ensure_record(value v_d, value v_n) {
  struct dc_device *d = Device_val(v_d);
  int n = Int_val(v_n);
  int released = take(d);
  int want = d->nrecord + n, have = d->crecord;
  give(d, released);
  if (want <= have) return Val_unit;
  int c = 2 * want < 8 ? 8 : 2 * want;
  struct dc_entry *grown = malloc((size_t)c * sizeof *grown);
  if (grown == NULL) caml_raise_out_of_memory();
  released = take(d);
  struct dc_entry *old = NULL;
  if (d->crecord < c) {
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

/* Submitting */

/* Records [s]'s in-queue waits in [d]'s record, for spread, after dropping
   the entries [d]'s word shows reached, and refuses if a producer is lost.
   The mutex is held. */
static int record_waits(struct dc_device *d, struct dc_sub *s) {
  uint64_t w = dc_word(d);
  int k = 0;
  for (int j = 0; j < d->nrecord; j++)
    if (d->record[j].u > w) d->record[k++] = d->record[j];
  d->nrecord = k;
  if (s->nwaits == 0) return DC_OK;
  if (k + s->nwaits > d->crecord) return DC_NEED_RECORD;
  uint64_t u = atomic_load(&d->submitted) + 1;
  for (int j = 0; j < s->nwaits; j++)
    d->record[k + j] = (struct dc_entry){s->producers[j], s->waits[j].value, u};
  d->nrecord = k + s->nwaits;
  for (int j = 0; j < s->nwaits; j++) {
    struct dc_device *p = dc_device_of(s->producers[j]);
    if (p != NULL && is_lost(p)) {
      d->nrecord = k;
      s->producer = s->producers[j];
      return DC_PRODUCER_LOST;
    }
  }
  return DC_OK;
}

void dc_sub_raise(struct dc_sub *s, uint64_t p);

/* Asks [d]'s driver for room for [s]; once the parts fit, assigns the next
   value, stores it as submitted, hands the work over and raises [s]'s
   stamps. Called under [d]'s turn. */
static int admit(struct dc_device *d, struct dc_sub *s) {
  int room = d->room(d->self, s->parts, s->nparts);
  if (room == NX_NEVER) return DC_NEVER;
  if (room == NX_LATER) {
    s->no_room_at = atomic_load(&d->submitted);
    return DC_NO_ROOM;
  }
  uint64_t v = atomic_load(&d->submitted) + 1;
  atomic_store_explicit(&d->submitted, v, memory_order_release);
  const char *why = NULL;
  int r = d->submit(d->self, v, s->waits, s->nwaits, s->parts, s->nparts,
                    s->handles, s->nhandles, &why);
  dc_sub_raise(s, DC_POINT(d->index, v));
  s->v = v;
  if (r == NX_OK) return DC_OK;
  s->why = why != NULL ? why : "the driver's submit failed";
  return DC_FAILED;
}

/* Submits [s] on its device: the turn, room, the value, the hand-over and
   the stamps, in one call that runs no OCaml. */
value caml_device_core_submit(value v_s) {
  struct dc_sub *s = *(struct dc_sub **)Data_custom_val(v_s);
  struct dc_device *d = s->dev;
  struct claims k = {0, 0, NULL};
  int released = 0, r;
  s->nclaims = 0;
  if (d->may_block) {
    caml_enter_blocking_section_no_pending();
    released = 1;
    mu_lock(d);
    int64_t until = now_ms() + DC_STILL_MS;
    int64_t left;
    while (d->turn && !is_lost(d) && (left = until - now_ms()) > 0)
      cv_wait_ms(d, (int)left);
    if (d->turn && !is_lost(d)) {
      r = DC_BUSY;
      goto out;
    }
  } else
    released = take(d);
  if (is_lost(d)) {
    r = DC_LOST;
    goto out;
  }
  r = record_waits(d, s);
  if (r != DC_OK) goto out;
  if (!d->may_block) r = admit(d, s);
  else {
    d->turn = 1;
    d->inside++;
    mu_unlock(d);
    r = admit(d, s);
    mu_lock(d);
    d->turn = 0;
    d->inside--;
    cv_broadcast(d);
  }
  if (r == DC_NO_ROOM || r == DC_NEVER) d->nrecord -= s->nwaits;
  if (r == DC_FAILED) {
    char *why = strdup(s->why);
    if (!lose_locked(d, why != NULL ? why : d->name)) {
      free(why);
      r = DC_LOST;
    }
  } else if (r == DC_OK && is_lost(d))
    r = DC_LOST_AFTER;
out:;
  int claimed = claim_locked(d);
  mu_unlock(d);
  if (r == DC_FAILED) {
    if (!released) caml_enter_blocking_section_no_pending();
    released = 1;
    spread(d, &k);
    finish(d, &k);
  }
  if (released) caml_leave_blocking_section();
  if (k.n > 0 || r == DC_FAILED) {
    free(s->claims);
    s->claims = k.index;
    s->nclaims = k.n;
  }
  return Val_int(r | (claimed ? DC_STOP_CLAIMED : 0));
}
