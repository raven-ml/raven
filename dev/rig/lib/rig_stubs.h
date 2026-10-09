/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What rig's stub files share: the device record, stamp records and
   the prepared form of a submission. */

#ifndef RIG_STUBS_H
#define RIG_STUBS_H

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

#include "rig_edge.h"

#ifdef _WIN32
#include <windows.h>
typedef SRWLOCK rig_mutex;
typedef CONDITION_VARIABLE rig_cond;
#else
#include <pthread.h>
typedef pthread_mutex_t rig_mutex;
typedef pthread_cond_t rig_cond;
#endif

/* Points: a device's index in bits 47 to 62 and a value in bits 0 to 46,
   the layout of an OCaml int's non-negative range. The word 0 is no point:
   index 0 is the host, whose work is never stamped. */
#define RIG_VALUE_BITS 47
#define RIG_VALUE_MASK ((UINT64_C(1) << RIG_VALUE_BITS) - 1)
#define RIG_POINT(index, v) (((uint64_t)(index) << RIG_VALUE_BITS) | (v))
#define RIG_INDEX(p) ((int)((p) >> RIG_VALUE_BITS))
#define RIG_VALUE(p) ((p) & RIG_VALUE_MASK)

/* Device indices run from 1 to RIG_DEVICES - 1 and are never reused. */
#define RIG_DEVICES 65536

/* A device's state. A loss moves a live device to LOSING while its loser
   spreads it, then to OWED, until a claimant takes its stop (STOPPING).
   Once the stop returned it is ENDED, and STOPPED once its word reads its
   last submitted value, or at once for an io device. A driver's device a
   forked child inherited is ORPHANED: its stop never runs and its work is
   the parent's.

     LIVE -> LOSING -> OWED -> STOPPING -> ENDED -> STOPPED
     LIVE, or any loss state, -> ORPHANED (in a forked child) */
enum {
  RIG_LIVE,
  RIG_LOSING,
  RIG_OWED,
  RIG_STOPPING,
  RIG_ENDED,
  RIG_STOPPED,
  RIG_ORPHANED
};

/* One in-queue wait of a device's unreached work on another device's value:
   the work of the waiter's value [u] waits for the producer's value [w]. */
struct rig_entry {
  int producer;
  uint64_t w, u;
};

/* A device. Made at open, never freed: other devices map its word, and its
   loss is read for the life of the process. The mutex guards [turn],
   [inside], the record and the moves of [state] but ENDED to STOPPED;
   [state], [lost], [submitted] and [committed] are also read without it. */
struct rig_device {
  rig_mutex mu;
  rig_cond cv;
  int index;
  char *name;
  int io; /* an io device, whose state its library holds */
  int may_block;
  void *self; /* the driver's state, whose first member points to [driver] */
  struct rig_driver driver; /* copied at open: a call loads no pointer */
  /* The timeline word, or NULL behind a transport. Once the device is
     stopped and nothing else reads the driver's word, it points to [final],
     and the driver's word is given back ([caml_rig_word_retire]). */
  _Atomic(_Atomic uint64_t *) word;
  _Atomic uint64_t final;
  _Atomic uint64_t seen; /* the last value a read of the word showed */
  _Atomic uint64_t submitted;
  _Atomic uint64_t committed; /* the last value known committed */
  _Atomic int state;
  _Atomic(char *) lost;     /* NULL, or the loss's reason */
  _Atomic uint64_t reached; /* once lost, the word's value at the loss */
  int turn;   /* a May_block submission is between room and hand-over */
  int inside; /* counted calls in flight, the turn holder included */
  struct rig_entry *record;
  int nrecord, crecord;
};

/* A memory's stamps: the point of its last write and, per device, the point
   of its last use. A chunk never moves, so a raise is one store or
   compare-and-set; a submission reserves its device's use word before its
   hand-over, so a raise allocates nothing. [refs] counts the memories,
   holds and memories' links that share it.

   A memory in a hold links to the hold's stamps, set once, which only
   submissions made with the hold raise, as uses. The memory's uses are its
   own and the hold's: such a submission may write any of its memory, so
   a read follows every point of the hold too. */
#define RIG_USES 4

struct rig_stamps {
  _Atomic uint64_t write;
  _Atomic uint64_t use[RIG_USES];
  _Atomic(struct rig_stamps *) next;
  _Atomic int refs;                  /* in the first chunk only */
  _Atomic(struct rig_stamps *) hold; /* in the first chunk only */
};

/* The hold's stamps that the stamps [s] link to, or NULL. Most memory is
   in no hold: the load is relaxed, and only a link found acquires. */
static inline struct rig_stamps *held(struct rig_stamps *s) {
  struct rig_stamps *hold =
      atomic_load_explicit(&s->hold, memory_order_relaxed);
  if (hold != NULL) atomic_thread_fence(memory_order_acquire);
  return hold;
}

/* A slot of a prepared submission: a memory's stamps, NULL while unset,
   the handle by which the device names it, and the device's use word in
   the stamps, reserved at each submit. A slot's handle outlives its
   clearing, so a submit whose slots name the handles of the last one
   collects none. */
struct rig_slot {
  struct rig_stamps *stamps;
  uint64_t handle;
  _Atomic uint64_t *use;
};

/* A handle a collect added, by hash: an entry of an earlier epoch is
   empty. */
struct rig_seen {
  uint64_t handle, epoch;
};

/* The prepared form of a submission on one device. */
struct rig_sub {
  /* Held by a submit from its first touch of what follows to its last:
     two domains' submits take turns. A free guard is taken by one
     compare-and-set; a submit that finds it held waits on [freed]. */
  _Atomic int busy, waiting;
  rig_mutex guard;
  rig_cond freed;
  struct rig_device *dev;
  int nparts;
  struct rig_part *parts;
  int *after; /* every part's [after], one after the other */
  int nfixed; /* the buffers the parts name */
  struct rig_slot *fixed;
  unsigned char *fixed_write;
  int nreads, nwrites; /* a run's buffers: those it reads, then writes */
  struct rig_slot *slots;
  struct rig_stamps *hold; /* a run's hold's stamps, NULL for none */
  _Atomic uint64_t *hold_use;
  /* Built for one submit, cleared after it. */
  int npoints, cpoints;
  uint64_t *points;
  int nwaits, cwaits;
  struct rig_wait *waits;
  int *producers;
  /* Built by a collect once a slot's handle changed, kept after. */
  int handles_stale;
  int nhandles; /* at most one per fixed buffer and slot */
  uint64_t *handles;
  int seen_bits; /* [seen] has 2^seen_bits entries, twice the handles */
  struct rig_seen *seen;
  uint64_t epoch; /* the collect's */
  /* What the submit answered. */
  uint64_t no_room_at, v;
  const char *why;
  int producer;
};

/* Whether the work up to the point [p] is done: its device's word, as the
   host last read it, reached [p]'s value, before the loss for a lost
   device, whose stop raises the word to its last value whatever ran. A
   point of a device a forked child inherited is never done: its work is
   the parent's. */
int rig_point_done(uint64_t p);

/* The host's page size in bytes. */
size_t rig_page_bytes(void);

/* The host clock: nanoseconds of the monotonic clock; on macOS, mach time,
   the time base of Metal's command buffer times. */
uint64_t rig_now_ns(void);

/* Raises the stamps [s]'s work names to [p]. */
void rig_sub_raise(struct rig_sub *s, uint64_t p);

/* Makes and unmakes the guard of a submission no submit holds. */
void rig_guard_init(struct rig_sub *s);
void rig_guard_destroy(struct rig_sub *s);

#endif
