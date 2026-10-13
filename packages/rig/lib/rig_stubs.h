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

#include <caml/mlvalues.h>

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
   the 63 bits of an OCaml int. From index 32768 the int is negative, so a
   point crosses from OCaml as Unsigned_long_val reads it. The word 0 is no
   point: index 0 is the host, whose values start at 1. */
#define RIG_VALUE_BITS 47
#define RIG_VALUE_MASK ((UINT64_C(1) << RIG_VALUE_BITS) - 1)
#define RIG_POINT(index, v) (((uint64_t)(index) << RIG_VALUE_BITS) | (v))
#define RIG_INDEX(p) ((int)((p) >> RIG_VALUE_BITS))
#define RIG_VALUE(p) ((p) & RIG_VALUE_MASK)

/* A device's index is its slot, from 1 to RIG_SLOTS - 1, and its line in
   rig's array of timeline words ([rig_edge.h]). A slot passes to a later
   device once its device has stopped, and that device's timeline goes on
   from the value the slot's word holds. */

/* A device's state. A loss moves a live device to LOSING while its loser
   spreads it, then to OWED, until a claimant takes its stop (STOPPING).
   Once the stop returned it is ENDED. A driver's device a forked child
   inherited is ORPHANED: its stop never runs and its work is the
   parent's.

     LIVE -> LOSING -> OWED -> STOPPING -> ENDED
     LIVE, or any loss state, -> ORPHANED (in a forked child) */
enum { RIG_LIVE, RIG_LOSING, RIG_OWED, RIG_STOPPING, RIG_ENDED, RIG_ORPHANED };

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

/* A slot's timeline line: what a point of the slot's devices asks, kept for the
   life of the process. [word] is the slot's word in rig's array, which a device
   whose own word lies elsewhere, such as behind a transport, raises as rig
   reads its word. [owner] is the [opened] of the device whose record wrote the
   line. [base] is the value the slot's device started after: every point of an
   earlier device on the slot is at most [base]. [state] and [reached] change
   under the device's mutex, and every field is read without it. A device that
   takes the slot stores [owner] first and the rest after it with release order,
   [state] last, so a record that reads a field of its line and then finds
   itself the owner read its own. [ranges] holds, newest first, the values each
   earlier device of the slot was given and lost, under [lost_top], the highest:
   a reader checks them after the rest of the line. The free of a slot publishes
   its device's range before the slot passes on, and ranges are never moved or
   freed. */
struct rig_range;

struct rig_line {
  _Atomic intnat owner;
  _Atomic(_Atomic uint64_t *) word;
  _Atomic uint64_t base;
  _Atomic uint64_t submitted;
  _Atomic int state;
  _Atomic uint64_t reached; /* once lost, the word's value at the loss */
  _Atomic(struct rig_range *) ranges;
  _Atomic uint64_t lost_top;
};

/* The values (lo, hi] that a device lost: it was given them, and had not
   reached them when it was lost. [device] is its OCaml record, a global
   root, which a stale point in the range raises [Lost] with. */
struct rig_range {
  uint64_t lo, hi;
  value device;
  struct rig_range *next; /* the slot's earlier range, below this one */
};

extern struct rig_line rig_lines[RIG_SLOTS];

/* The [base] of slot [index]'s line. */
static inline uint64_t rig_line_base(int index) {
  return atomic_load_explicit(&rig_lines[index].base, memory_order_acquire);
}

/* A list of release records ([rig_heap.c]): those of a device lie in its
   record. [count] counts the tokens made for the list whose records its
   drains have not yet routed: a device's slot frees at none. [closed] says
   that the list takes no more tokens, as its device's slot is being freed:
   a token counts itself before it reads [closed], and the free closes the
   list before it reads [count], so one of the two sees the other. */
struct release_node;

struct release_list {
  _Atomic(struct release_node *) head;
  _Atomic intnat count;
  _Atomic int closed;
};

/* A device's record, which lives as long as the program holds the device.
   The mutex guards [turn], [inside], the record of waits, the moves of the
   line's state and the free. Once its line names another owner, the slot
   is no longer the device's: it has ended, and its word and its last value
   given are [final]. */
struct rig_device {
  rig_mutex mu;
  rig_cond cv;
  int index;
  struct rig_line *line;
  uint64_t base; /* the value its timeline started after */
  intnat opened; /* unique for the process */
  uint64_t final;
  _Atomic(char *) lost; /* NULL, or the loss's reason */
  int own_why;          /* whether the record frees [lost] */
  char *name;
  int may_block;
  int io; /* an io device, whose state its library holds */
  void *self; /* the driver's state, whose first member points to [driver] */
  struct rig_driver driver; /* copied at open: a call loads no pointer */
  _Atomic uint64_t committed; /* the last value known committed */
  int faulted; /* lost for a fault of a counted call, set with [lost] */
  int turn;   /* a May_block submission is between room and hand-over */
  int inside; /* counted calls in flight, the turn holder included */
  struct rig_entry *record;
  int nrecord, crecord;
  struct rig_times times;
  struct release_list release;
};

/* The device record of the OCaml value [v]. */
#define Device_val(v) (*(struct rig_device **)Data_custom_val(v))

/* A memory's stamps: per device, the point of its last use and the point of
   its last write. A chunk's write[i] belongs to the device of its use[i],
   and only that device's submissions store to the two, one at a time under
   its turn, so two devices' writes never replace each other. A chunk never
   moves, so a raise is one store; a submission reserves its device's use
   word before its hand-over, so a raise allocates nothing. A reservation
   takes the word of a device whose stop was claimed and whose point is
   done, so the chunks grow only with the devices that use the memory at
   once. Its write word keeps that device's done point until the new device
   writes. A reservation scans from the first chunk and takes the first
   empty or spent word of its kind, and a use word never empties, so every
   use word after an empty one is empty: a walk ends at the first empty use
   word. A word whose point a loss dropped stays until a later device takes
   its slot: its point is then folded into the first chunk's [lost_use] or
   [lost_write], which keep the first such point, and the word can be taken.
   A walk reads them after the words. [refs] counts the memories and holds
   that share it. A hold's stamps hold only uses: those of the submissions
   made with it. */
#define RIG_USES 4

struct rig_stamps {
  _Atomic uint64_t use[RIG_USES];
  _Atomic uint64_t write[RIG_USES];
  /* Per device, the last point of its work that uses the memory unordered:
     the memory returns after it, and no access follows it. */
  _Atomic uint64_t keep[RIG_USES];
  _Atomic(struct rig_stamps *) next;
  _Atomic int refs; /* in the first chunk only */
  /* In the first chunk only: a lost point of an earlier device of a slot,
     folded from a use word, and from a write word. */
  _Atomic uint64_t lost_use, lost_write;
};

/* The point folded into the stamps [s] that an access follows: [lost_write],
   or, if [every], [lost_use] when [lost_write] holds none; 0 for none, and
   for no stamps. Every reader reads the head after a word, with acquire
   order, and a fold comes before the take that changes the word: a reader
   that saw the change sees the fold, so the head's loads are relaxed. */
static inline uint64_t rig_stamps_folded(struct rig_stamps *s, int every) {
  if (s == NULL) return 0;
  uint64_t p = atomic_load_explicit(&s->lost_write, memory_order_relaxed);
  if (p != 0 || !every) return p;
  return atomic_load_explicit(&s->lost_use, memory_order_relaxed);
}

/* The point of the chunk [s]'s [i]th words that an access follows: its use
   word's if [every], its write word's otherwise, 0 for none; or -1 from
   the chunk's first empty use word on, where a walk of the stamps ends. A
   write raises its device's use word too, so the use words cover every
   write. A use word that a submission reserved holds no value until its
   first raise. */
static inline int64_t rig_stamps_followed(struct rig_stamps *s, int i,
                                          int every) {
  uint64_t u = atomic_load_explicit(&s->use[i], memory_order_acquire);
  if (u == 0) return -1;
  uint64_t p =
      every ? u : atomic_load_explicit(&s->write[i], memory_order_acquire);
  return RIG_VALUE(p) == 0 ? 0 : (int64_t)p;
}

/* How a submission's work uses a buffer, as Submission.use orders its
   constructors. */
enum rig_use { RIG_USE_READ, RIG_USE_WRITE, RIG_USE_UNORDERED };

/* A buffer a submission names, in a part or as fixed memory: its memory's
   stamps and how the work uses it. */
struct rig_fixed {
  struct rig_stamps *stamps;
  enum rig_use use;
};

/* An entry ref: the function value [code] that each submit writes at byte
   [at] of the run's blocks. */
struct rig_code {
  size_t at;
  uint64_t code;
};

/* A buffer of a run: its memory's stamps, NULL while unset, and its use
   word, its keep word for an unordered use. */
struct rig_slot {
  struct rig_stamps *stamps;
  _Atomic uint64_t *use;
};

/* What a run's buffer that a copy names passes the driver, and where its
   bytes lie. [handle] and [offset] are the copy's fields for it, which for
   [local] memory of this process are its memory's host address and the
   offset into it (RIG_LOCAL_SRC, RIG_LOCAL_DST). [device_memory] is the
   copy.device_memory fact for it. [space] and [first] are its span
   (rig_buffer_span), and [alias], unless 0, the host address of an io
   memory's mapped pages for its first byte, as Rig.Buffer.overlaps
   compares them. */
struct rig_side {
  uint64_t handle, offset, bytes;
  intptr_t space, first, alias;
  int local, device_memory;
};

/* Sets [s] for the buffer [b], which a copy passes as memory of this
   process iff [local], all but its [device_memory]. */
void rig_buffer_side(value b, int local, struct rig_side *s);

/* Whether the bytes of [a] and [b] share one, as Rig.Buffer.overlaps
   answers. */
int rig_sides_overlap(const struct rig_side *a, const struct rig_side *b);

/* A copy part of a submission: its index and, for its source and its
   destination, the run's slot, or -1 for a buffer the submission names,
   whose side is in [ends]. */
struct rig_copy {
  int part, src, dst;
  struct rig_side ends[2]; /* the source's, the destination's */
};

/* The prepared form of a submission on one device, fixed when it is made:
   any number of submits read it at once. Only its driver's room check and
   submit, under the device's turn, write its scratch. */
struct rig_sub {
  uint64_t id; /* unique for the life of the process */
  struct rig_device *dev;
  int nparts;
  struct rig_part *parts;
  void *scratch; /* RIG_SCRATCH bytes per part */
  int *after; /* every part's [after], one after the other */
  int nfixed; /* the buffers the parts name */
  struct rig_fixed *fixed;
  int nslots;      /* a run's buffers */
  uint8_t *uses; /* per slot: how its work uses the buffer, an rig_use */
  int nrefs;
  struct rig_ref *refs; /* every launch's and fill's buffer refs, in part order */
  int ncodes;
  struct rig_code *codes; /* every entry ref, in part order */
  int ncopies, nslotted; /* its copies, and those with a slot end */
  struct rig_copy *copies; /* every copy part, in part order */
  size_t args;            /* where its last block ends in a run */
};

/* A launch's or a fill's block in a run, as [Submission.block] encodes it:
   where it starts, in bytes, above the RIG_BLOCK_BITS bits of its parameter
   count. */
#define RIG_BLOCK_BITS 13
#define RIG_BLOCK_START(b) ((uint64_t)(b) >> RIG_BLOCK_BITS)
#define RIG_BLOCK_PARAMS(b) ((uint64_t)(b) & ((1u << RIG_BLOCK_BITS) - 1))

/* Who holds a run: one submit or one store at a time takes it by a
   compare-and-set from RUN_IDLE. */
enum { RUN_IDLE, RUN_SUBMIT, RUN_STORE };

/* The state of one submit: what it collects and what it answers, in the
   caller's run, which one submit uses at a time. Its arrays grow to the
   largest submission it served. */
struct rig_run {
  _Atomic int busy; /* RUN_IDLE, or who took it */
  uint64_t owner; /* the submission whose blocks [args] holds, 0 for none */
  int cfixed, cslots;
  int nslots; /* the slots of the submission it served last */
  /* The use words of the submission's fixed buffers, keep words for those
     used unordered. */
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
  uint64_t sub; /* the submission the fixed and slot arrays were fitted to */
  /* What the submit answered: for a producer the device's queue would wait
     for that is lost, the point it did not reach. */
  uint64_t no_room_at, v;
  const char *why;
  uint64_t producer;
  /* The blocks the setters store into: [nargs] bytes reach the
     end of the last block stored into, of [cargs] allocated. */
  uint8_t *args;
  size_t nargs, cargs;
  uint64_t *addresses; /* each slot's address, for launches' and fills' refs */
  struct rig_side *sides; /* each slot's side, for copies */
  /* The parts a submission with copies of slots hands its driver: its own,
     with each such copy's fields set from the run's slots. */
  struct rig_part *parts;
  int cparts;
  /* Whether the submit times its value, and the device's time pair it took
     at the hand-over, -1 for none. */
  int timed, pair;
};

/* Whether the work up to the point [p] is done: its device's word, as the
   host last read it, reached [p]'s value, before the loss for a lost
   device, whose stop raises the word to its last value whatever ran. A
   point of an earlier device of its slot is done unless its range holds
   it. A point of a device a forked child inherited is not done: its work
   is the parent's. */
int rig_point_done(uint64_t p);

/* Whether the point [p] is a lost device's and was not reached when the
   device was lost, or lies in a range of its slot: no wait gets past it. A
   point of a device a forked child inherited is lost. */
int rig_point_lost(uint64_t p);

/* The host's page size in bytes. */
size_t rig_page_bytes(void);

/* The host's driver state. Its word is slot 0's. */
void *rig_host_state(void);

/* Raises the stamps [s]'s work names in the run [r] to [p]. */
void rig_sub_raise(const struct rig_sub *s, struct rig_run *r, uint64_t p);

#endif
