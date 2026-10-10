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

/* A device's time pairs for timed values, chunks of RIG_TIMES pairs that
   never move, and the indices of the free pairs, under [mu]. A hand-over
   takes a pair for a value it times; its driver writes the pair before the
   word shows the value, and the read that records the value's span gives
   it back. A lost device keeps the pairs of values it never showed: they
   live as long as the device, so a late write from its work lands in
   them. */
#define RIG_TIMES 256

struct rig_times {
  rig_mutex mu;
  uint64_t (**chunks)[2];
  int nchunks;
  int *free;
  int nfree;
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
  int faulted; /* lost for a fault of a counted call, set with [lost] */
  int turn;   /* a May_block submission is between room and hand-over */
  int inside; /* counted calls in flight, the turn holder included */
  struct rig_entry *record;
  int nrecord, crecord;
  struct rig_times times;
};

/* A memory's stamps: the point of its last write and, per device, the point
   of its last use. A chunk never moves, so a raise is one store or
   compare-and-set; a submission reserves its device's use word before its
   hand-over, so a raise allocates nothing. [refs] counts the memories
   and holds that share it. A hold's stamps hold only uses: those of the
   submissions made with it. */
#define RIG_USES 4

struct rig_stamps {
  _Atomic uint64_t write;
  _Atomic uint64_t use[RIG_USES];
  _Atomic(struct rig_stamps *) next;
  _Atomic int refs; /* in the first chunk only */
};

/* A buffer a submission names, in a part or as fixed memory: its memory's
   stamps, the handle by which the device names it, and whether the work
   writes it. */
struct rig_fixed {
  struct rig_stamps *stamps;
  uint64_t handle;
  int write;
};

/* A buffer of a run: its memory's stamps, NULL while unset, the handle by
   which the device names it, and its use word. A slot's handle outlives
   its clearing, so a submit whose slots name the handles of the last one
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

/* The prepared form of a submission on one device, fixed when it is made:
   any number of submits read it at once. */
struct rig_sub {
  uint64_t id; /* unique for the life of the process */
  struct rig_device *dev;
  int nparts;
  struct rig_part *parts;
  int *after; /* every part's [after], one after the other */
  int nfixed; /* the buffers the parts name */
  struct rig_fixed *fixed;
  int nslots;      /* a run's buffers */
  uint8_t *writes; /* per slot: whether its work writes the buffer */
  int nrefs;
  struct rig_ref *refs; /* every launch's refs, one after the other */
  size_t args;          /* where its last launch's block ends in a run */
};

/* A launch's block in a run, as [Submission.block] encodes it: where it
   starts, in bytes, above the RIG_BLOCK_BITS bits of its parameter count. */
#define RIG_BLOCK_BITS 13
#define RIG_BLOCK_START(b) ((uint64_t)(b) >> RIG_BLOCK_BITS)
#define RIG_BLOCK_PARAMS(b) ((uint64_t)(b) & ((1u << RIG_BLOCK_BITS) - 1))

/* The state of one submit: what it collects, the handles it names and
   what it answers, in the caller's run, which one submit uses at a time.
   Its arrays grow to the largest submission it served. */
struct rig_run {
  _Atomic int busy; /* taken by a submit's compare-and-set */
  int cfixed, cslots;
  int nslots; /* the slots of the submission it served last */
  /* The use words of the submission's fixed buffers. */
  _Atomic uint64_t **fixed;
  struct rig_slot *slots;
  struct rig_stamps *hold; /* the hold's stamps, NULL for none */
  _Atomic uint64_t *hold_use;
  /* Built for one submit, cleared after it. */
  int npoints, cpoints;
  uint64_t *points;
  int nwaits, cwaits;
  struct rig_wait *waits;
  int *producers;
  /* Built by a collect once a slot's handle or the submission changed,
     kept after. */
  uint64_t sub; /* the [id] of the submission the handles are of */
  int handles_stale;
  int nhandles, chandles; /* at most one per fixed buffer and slot */
  uint64_t *handles;
  int seen_bits; /* [seen] has 2^seen_bits entries, twice the handles */
  struct rig_seen *seen;
  uint64_t epoch; /* the collect's */
  /* What the submit answered. */
  uint64_t no_room_at, v;
  const char *why;
  int producer;
  /* The launches' blocks the setters store into: [nargs] bytes reach the
     end of the last block stored into, of [cargs] allocated. */
  uint8_t *args;
  size_t nargs, cargs;
  uint64_t *addresses; /* each slot's address, for launches' refs */
  /* Whether the submit times its value, and the device's time pair it took
     at the hand-over, -1 for none. */
  int timed, pair;
};

/* Whether the work up to the point [p] is done: its device's word, as the
   host last read it, reached [p]'s value, before the loss for a lost
   device, whose stop raises the word to its last value whatever ran. A
   point of a device a forked child inherited is never done: its work is
   the parent's. */
int rig_point_done(uint64_t p);

/* Whether the point [p] is a lost device's and was not reached when the
   device was lost: no wait gets past it. A point of a device a forked
   child inherited is lost. */
int rig_point_lost(uint64_t p);

/* The host's page size in bytes. */
size_t rig_page_bytes(void);

/* The host clock: nanoseconds of the monotonic clock; on macOS, mach time,
   the time base of Metal's command buffer times. */
uint64_t rig_now_ns(void);

/* Raises the stamps [s]'s work names in the run [r] to [p]. */
void rig_sub_raise(const struct rig_sub *s, struct rig_run *r, uint64_t p);

#endif
