/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What the core's stub files share: the device record, stamp records and
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

/* A stop's answer, as the device records it. */
enum {
  RIG_ANSWER_NONE,
  RIG_ANSWER_STOPPING,
  RIG_ANSWER_STOPPED,
  RIG_ANSWER_UNKNOWN
};

/* One in-queue wait of a device's unreached work on another device's value:
   the work of the waiter's value [u] waits for the producer's value [w]. */
struct rig_entry {
  int producer;
  uint64_t w, u;
};

/* A device. Made at open, never freed: other devices map its word, and its
   loss is read for the life of the process. The mutex guards [turn],
   [inside], [owed], [spreading] and the record; [lost], [answer] and
   [submitted] are also read without it. */
struct rig_device {
  rig_mutex mu;
  rig_cond cv;
  int index;
  char *name;
  int io;        /* an io device, whose state its library holds */
  int inherited; /* a driver's device a forked child holds from its parent */
  int may_block;
  void *self;
  rig_room_fn *room;
  rig_submit_fn *submit;
  _Atomic uint64_t *word; /* the timeline word, or NULL behind a transport */
  _Atomic uint64_t seen;  /* the last value a read of the word showed */
  _Atomic uint64_t submitted;
  _Atomic(char *) lost; /* NULL, or the loss's reason */
  _Atomic int answer;
  int turn;      /* a May_block submission is between room and hand-over */
  int inside;    /* counted calls in flight, the turn holder included */
  int owed;      /* lost, and its stop not yet claimed */
  int spreading; /* lost, and its loser has not finished spreading it */
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

/* A slot of a prepared submission: a memory's stamps, the handle by which
   the device names it, and the device's use word in the stamps, reserved
   at each submit. */
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
  struct rig_device *dev;
  int nparts;
  struct rig_part *parts;
  int *after; /* every part's [after], one after the other */
  int nfixed; /* the buffers the parts name */
  struct rig_slot *fixed;
  unsigned char *fixed_write;
  int nreads, nwrites; /* the read slots, then the write slots */
  struct rig_slot *slots;
  int nwait_slots;
  uint64_t *wait_slots;
  struct rig_stamps *hold;
  _Atomic uint64_t *hold_use;
  /* Built for one submit, cleared after it. */
  int npoints, cpoints;
  uint64_t *points;
  int nwaits, cwaits;
  struct rig_wait *waits;
  int *producers;
  int nhandles; /* at most one per fixed buffer and slot */
  uint64_t *handles;
  int seen_bits; /* [seen] has 2^seen_bits entries, twice the handles */
  struct rig_seen *seen;
  uint64_t epoch; /* the collect's */
  /* What the submit answered. */
  uint64_t no_room_at, v;
  const char *why;
  int producer;
  int nclaims; /* the stops a failed hand-over's spread claimed */
  int *claims;
};

/* The host's page size in bytes. */
size_t rig_page_bytes(void);

/* Raises the stamps [s]'s work names to [p]. */
void rig_sub_raise(struct rig_sub *s, uint64_t p);

#endif
