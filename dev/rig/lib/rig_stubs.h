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
   [state], [lost] and [submitted] are also read without it. */
struct rig_device {
  rig_mutex mu;
  rig_cond cv;
  int index;
  char *name;
  int io; /* an io device, whose state its library holds */
  int may_block;
  void *self;
  rig_room_fn *room;
  rig_submit_fn *submit;
  _Atomic uint64_t *word; /* the timeline word, or NULL behind a transport */
  _Atomic uint64_t seen;  /* the last value a read of the word showed */
  _Atomic uint64_t submitted;
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
   hand-over, so a raise allocates nothing. [refs] counts the memories and
   holds that share it. */
#define RIG_USES 4

struct rig_stamps {
  _Atomic uint64_t write;
  _Atomic uint64_t use[RIG_USES];
  _Atomic(struct rig_stamps *) next;
  _Atomic int refs; /* in the first chunk only */
};

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
  struct rig_stamps *hold;
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

/* Raises the stamps [s]'s work names to [p]. */
void rig_sub_raise(struct rig_sub *s, uint64_t p);

/* Makes and unmakes the guard of a submission no submit holds. */
void rig_guard_init(struct rig_sub *s);
void rig_guard_destroy(struct rig_sub *s);

#endif
