/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Handing work to a device from C.

   A device's driver keeps its C state for one device, [self], whose first
   member is a pointer to the driver's struct rig_driver: three functions
   in the shapes rig_room_fn, rig_submit_fn and rig_commit_fn, which the
   caller that holds its submissions in C calls with [self] and the
   structures below. The functions are called one at a time per device,
   the room check before its submission, without the OCaml runtime: they
   call no function of it and read no OCaml value.

   A submission is the work of one value [v] of the device's timeline: an
   array of parts, each on one of the device's queues, and the waits on
   other devices' words it starts after. */

#ifndef RIG_EDGE_H
#define RIG_EDGE_H

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

/* What a room check answers: the parts fit the device's queues now; they
   fit once one of the device's values is reached; they never fit the
   device's empty queues. */
enum { RIG_FITS, RIG_LATER, RIG_NEVER };

/* What a submission and a commit answer: done; failed, and the device is
   lost; and, for a submission, done with every value up to it committed. */
enum { RIG_OK, RIG_FAILED, RIG_COMMITTED };

/* A wait's kinds: the 64-bit word at the mapped address [at] holds at
   least [value]; [at] is an object of the driver that reaches [value]. */
enum { RIG_WORD, RIG_OBJECT };

/* Which side of a copy is memory of the calling process, if any. */
enum { RIG_LOCAL_NONE, RIG_LOCAL_SRC, RIG_LOCAL_DST };

struct rig_wait {
  uint64_t at, value;
  int kind;
};

/* A part's kinds. 0 is no part: a room check answers RIG_NEVER for it. */
enum { RIG_WORDS = 1, RIG_FILL, RIG_COPY, RIG_LAUNCH };

/* The most parameter bytes a launch has: the least of the drivers' limits
   (Metal's setBytes, CUDA's launch). */
#define RIG_PARAMS 4096

/* A launch's block in a run's [args]: the groups of its grid along x, y
   and z, the threads of a group, a group's dynamic shared memory in bytes,
   then its parameters. A block starts on 16 bytes. */
struct rig_block {
  uint32_t groups[3], threads[3], shared, unused;
  uint8_t params[];
};

/* A launch's reference to a run's buffer: the 8 parameter bytes at [at]
   hold an offset into the run's buffer [slot], to which the driver adds
   the buffer's address. */
struct rig_ref {
  uint32_t at, slot;
};

/* One part of a submission, on the queue at index [queue] of the driver's
   queue list, whose work [kind] names:
   - RIG_WORDS: the [words.n] words at [words.at], placed on the queue;
   - RIG_FILL: [fill.fn] called with the queue's context, [fill.arg] and
     [v], which returns 0 or a failure. [fill.ring_units] and
     [fill.segment_bytes] bound what it writes, for a driver whose queue is
     a ring;
   - RIG_COPY: [copy.bytes] bytes from [copy.src_offset] bytes into the
     memory whose handle is [copy.src] to [copy.dst_offset] bytes into the
     memory whose handle is [copy.dst]. Where [copy.local] is RIG_LOCAL_SRC
     or RIG_LOCAL_DST, that side is memory of the calling process instead:
     its handle field holds the host address of the memory, and its offset
     field the offset into it. Only a driver of a device of another machine
     is handed one;
   - RIG_LAUNCH: the function whose entry is [launch.code] and
     [launch.launch] (the [code] and [launch] the driver's [entry] gave) run
     once, over the grid of the struct rig_block at byte [launch.block] of
     the submission's [args], with its [launch.params] bytes of parameters,
     in which the 8 bytes at each of the [launch.nrefs] refs' [at] are read
     as a 64-bit offset plus [slots[slot]]. The driver copies the parameters
     into memory of its own before it adds the addresses: [args] is the
     caller's, and is read only during the call.
   The part runs after the parts of the same submission before it on its
   queue, and after the parts whose indices the [nafter] ints at [after]
   list, each below its own: [after] orders parts of different queues. */
struct rig_part {
  int queue, kind;
  const int *after;
  int nafter;
  union {
    struct {
      const uint32_t *at;
      size_t n;
    } words;
    struct {
      int (*fn)(void *queue, void *arg, uint64_t v);
      void *arg;
      size_t ring_units, segment_bytes;
    } fill;
    struct {
      uint64_t dst, dst_offset, src, src_offset, bytes;
      int local;
    } copy;
    struct {
      uint64_t code;
      const void *launch;
      uint32_t block, params;
      const struct rig_ref *refs;
      int nrefs;
    } launch;
  };
};

/* Answers RIG_FITS, RIG_LATER or RIG_NEVER for the [n] parts at [parts],
   whose launches' blocks lie in [args]. A launch never fits whose grid or
   group has a size of 0 along an axis, whose groups or threads exceed the
   device's or the function's limits, or whose shared memory exceeds the
   function's. */
typedef int rig_room_fn(void *self, const struct rig_part *parts, int n,
                        const uint8_t *args);

/* Encodes [parts] on the device's queues as the work of [v], the value
   after the last one it received, which starts after [waits] and runs
   without another call. The word shows [v] once [v] is committed and its
   work completed. [args] holds the launches' blocks and [slots] the
   address of each of the run's [nslots] buffers, as the device's work
   addresses it, for its launches' refs. [handles] lists [self]'s own memory
   the work uses, for a driver whose submissions name it; a copy's side
   that [copy_local] names is not in it. Answers RIG_COMMITTED if every
   value up to [v] is committed, RIG_OK if [v] is encoded only, or
   RIG_FAILED with [*failure] set to the driver's message, which lives as
   long as the device. */
typedef int rig_submit_fn(void *self, uint64_t v, const struct rig_wait *waits,
                         int nwaits, const struct rig_part *parts, int nparts,
                         const uint8_t *args, const uint64_t *slots,
                         int nslots, const uint64_t *handles, int nhandles,
                         const char **failure);

/* Commits the device's work up to [v], at most the last value it received:
   the device writes [v] or a later value into the word once the work up to
   it completed. Committing a committed value does nothing. The driver also
   commits on its own, at least once every [k] values it receives, [k] a
   bound of its own. Answers RIG_OK, or RIG_FAILED with [*failure] set as a
   submission's. */
typedef int rig_commit_fn(void *self, uint64_t v, const char **failure);

/* A driver's functions, which the first member of its state for each
   device points to. */
struct rig_driver {
  rig_room_fn *room;
  rig_submit_fn *submit;
  rig_commit_fn *commit;
};

/* The functions of the driver whose C state for one device is [self]. */
static inline const struct rig_driver *rig_driver_of(const void *self) {
  return *(const struct rig_driver *const *)self;
}

/* Raises the timeline word [word] to [last] with release order, unless it
   shows [last] or later: a word never moves backwards. */
static inline void rig_raise(_Atomic uint64_t *word, uint64_t last) {
  uint64_t w = atomic_load_explicit(word, memory_order_acquire);
  while (w < last && !atomic_compare_exchange_weak_explicit(
                         word, &w, last, memory_order_release,
                         memory_order_acquire)) {
  }
}

#endif
