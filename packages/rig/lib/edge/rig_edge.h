/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Handing work to a device from C.

   A device's driver keeps its C state for one device, [self], whose first
   member is a pointer to the driver's struct rig_driver: four functions
   in the shapes rig_room_fn, rig_submit_fn, rig_commit_fn and
   rig_settle_fn, which the caller that holds its submissions in C calls
   with [self] and the structures below. The functions are called one at a
   time per device, the room check before its submission, without the
   OCaml runtime: they call no function of it and read no OCaml value.

   A submission is the work of one value [v] of the device's timeline: an
   array of parts, each on one of the device's queues, and the waits on
   other devices' words it starts after. */

#ifndef RIG_EDGE_H
#define RIG_EDGE_H

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

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

/* The most parameter bytes a launch or fill has: the least of the drivers'
   limits (Metal's setBytes, CUDA's launch). */
#define RIG_PARAMS 4096

/* A launch's or a fill's block in a run's [args]: the groups of its grid
   along x, y and z, the threads of a group, a group's dynamic shared memory
   in bytes, then its parameters. A fill uses only the parameters. A block
   starts on 16 bytes. */
struct rig_block {
  uint32_t groups[3], threads[3], shared, unused;
  uint8_t params[];
};

/* A launch's or a fill's reference to a run's buffer: the 8 parameter bytes
   at [at] hold an offset into the run's buffer [slot], to which the driver
   adds the buffer's address. */
struct rig_ref {
  uint32_t at, slot;
};

/* A launch's or a fill's parameters: the [params] bytes after the struct
   rig_block at byte [block] of the submission's [args], in which the 8
   bytes at each of the [nrefs] refs' [at] are read as a 64-bit offset plus
   [slots[slot]]. */
struct rig_args {
  uint32_t block, params;
  const struct rig_ref *refs;
  int nrefs;
};

/* One part of a submission, on the queue at index [queue] of the driver's
   queue list, whose work [kind] names:
   - RIG_WORDS: the [words.n] words at [words.at], placed on the queue;
   - RIG_FILL: [fill.fn] called with the queue's context, the host address
     of its argument, [fill.args] placed (rig_fill), and [v], which returns
     0 or a failure;
   - RIG_COPY: [copy.bytes] bytes from [copy.src_offset] bytes into the
     memory whose handle is [copy.src] to [copy.dst_offset] bytes into the
     memory whose handle is [copy.dst]. The two ranges share no byte. Where
     [copy.local] is RIG_LOCAL_SRC or RIG_LOCAL_DST, that side is memory of
     the calling process instead: its handle field holds the host address of
     the memory, and its offset field the offset into it. Only a driver of a
     device of another machine is handed one. [copy.device_memory] is 1
     exactly when both buffers' original allocations belong to this device
     and have memory kind Device. Views and borrows on the owning device
     preserve it. Pinned, Mapped, this process's host memory, peer borrows
     and unbacked empty buffers have 0. Every field of a copy part may
     differ from one call to the next, since its ends may be a run's
     buffers. The driver chooses how to copy on each call;
   - RIG_LAUNCH: the function whose entry is [launch.code] and
     [launch.launch] (the [code] and [launch] the driver's [entry] gave) run
     once, over the grid of the struct rig_block at byte [launch.args.block]
     of the submission's [args], with its parameters [launch.args]. The
     driver copies the parameters into memory of its own before it adds the
     addresses (rig_params): [args] is the caller's, and is read only during
     the call.
   The part runs after the parts of the same submission before it on its
   queue, and after the parts whose indices the [nafter] ints at [after]
   list, each below its own: [after] orders parts of different queues, and
   names at most the last part of each other queue. [parts], and what they
   point to, are valid only during the call that is handed them. */
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
      struct rig_args args;
    } fill;
    struct {
      uint64_t dst, dst_offset, src, src_offset, bytes;
      int local, device_memory;
    } copy;
    struct {
      uint64_t code;
      const void *launch;
      struct rig_args args;
    } launch;
  };
};

/* The bytes of scratch a submission holds for each of its parts, aligned
   to 8 bytes. A room check and a submit may write the [n * RIG_SCRATCH]
   bytes at [scratch] and read them back during the call. No other call
   uses them meanwhile, and a call finds whatever an earlier one left. A
   driver keeps there what it decides per part, such as which engine runs
   it, so that neither call allocates. It decides again on each call what
   it derives from a copy part's fields, which may differ on the next
   call. */
#define RIG_SCRATCH 8

/* Answers RIG_FITS, RIG_LATER or RIG_NEVER for the [n] parts at [parts],
   whose launches' and fills' blocks lie in [args]. A launch never fits whose
   grid or group has a size of 0 along an axis, whose groups or threads exceed
   the device's or the function's limits, or whose shared memory exceeds the
   function's. */
typedef int rig_room_fn(void *self, const struct rig_part *parts, int n,
                        void *scratch, const uint8_t *args);

/* Encodes [parts] on the device's queues as the work of [v], the value
   after the last one it received, which starts after [waits] and runs
   without another call. The word shows [v] once [v] is committed and its
   work completed. [args] holds the launches' and fills' blocks and [slots]
   the address of each of the run's [nslots] buffers, as the device's work
   addresses it, for their refs. [times] is NULL unless a profile is
   being taken; otherwise the driver stores at [times[0]] and [times[1]] the
   nanoseconds of rig's host clock (Rig.Profile.now: CLOCK_UPTIME_RAW on
   macOS, CLOCK_MONOTONIC elsewhere) at which [v]'s work started and ended
   on the device: before the word shows [v] if its settle is NULL, and
   before its settle of [v] returns otherwise (rig_settle_fn). [times]
   stays valid until then. It leaves both 0 if it cannot time the work.
   Answers RIG_COMMITTED if every value up to [v] is committed, RIG_OK if
   [v] is encoded only, or RIG_FAILED with [*failure] set to the driver's
   message, which lives as long as the device. */
typedef int rig_submit_fn(void *self, uint64_t v, const struct rig_wait *waits,
                         int nwaits, const struct rig_part *parts, int nparts,
                         void *scratch, const uint8_t *args,
                         const uint64_t *slots, int nslots, uint64_t *times,
                         const char **failure);

/* Commits the device's work up to [v], at most the last value it received:
   the device writes [v] or a later value into the word once the work up to
   it completed. Committing a committed value does nothing. The driver also
   commits on its own, at least once every [k] values it receives, [k] a
   bound of its own. Answers RIG_OK, or RIG_FAILED with [*failure] set as a
   submission's. */
typedef int rig_commit_fn(void *self, uint64_t v, const char **failure);

/* Writes the times of [v], whose submission was given [times], unless the
   driver wrote them already. The caller calls it after the word shows [v]
   and before it reads the times, at most once for each such value, and
   possibly never, as for a device lost first. A driver whose GPU writes
   the word has the work write stamps into the driver's own memory, and
   converts them to the host's clock here, or in a later submit once [v]'s
   work has completed. Settle never waits for the device's work. It may
   read a clock of the device, with one system call or register read.
   Answers RIG_OK, or RIG_FAILED with [*failure] set as a submission's if
   it could not write the times, such as for a stamp the work never wrote,
   which loses the device. A driver that writes the times before the word
   shows [v] sets [settle] to NULL. */
typedef int rig_settle_fn(void *self, uint64_t v, const char **failure);

/* A driver's functions, which the first member of its state for each
   device points to. */
struct rig_driver {
  rig_room_fn *room;
  rig_submit_fn *submit;
  rig_commit_fn *commit;
  rig_settle_fn *settle;
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

/* Writes into [dst] the parameters [a] of a launch or a fill: the
   [a->params] bytes of its block in [args], each ref's 8 bytes, a 64-bit
   integer in the host's byte order, plus [slots[slot]]. [dst] holds as many
   bytes and lies outside [args]. */
static inline void rig_params(uint8_t *dst, const struct rig_args *a,
                              const uint8_t *args, const uint64_t *slots) {
  const uint8_t *src = args + a->block + sizeof(struct rig_block);
  memcpy(dst, src, a->params);
  for (int i = 0; i < a->nrefs; i++) {
    const struct rig_ref *r = &a->refs[i];
    uint64_t x;
    memcpy(&x, src + r->at, sizeof x);
    x += slots[r->slot];
    memcpy(dst + r->at, &x, sizeof x);
  }
}

/* rig_now_ns () is the host's clock: nanoseconds of the system's monotonic
   clock from an unspecified start; on macOS, mach time, the time base of
   Metal's command buffer times. A driver writes a value's times and
   measures its waits on it (Rig_edge's calls section). OCaml reads it
   through the noalloc external caml_rig_edge_now, which answers an
   untagged int, and caml_rig_edge_now_byte. */
uint64_t rig_now_ns(void);

/* rig_code_sync () makes the calling thread run the code that rig linked on
   the host before the call ([Rig.Image.load] on [Rig.host]): on arm64 it
   synchronizes the thread's instruction stream with that code, once per
   link. A driver calls it before it calls a fill's function by its own
   means: the function may be such code. */
void rig_code_sync(void);

/* Calls the fill [p] with the queue's context [queue] and the value [v], on
   its argument placed into memory of the call's own (rig_params), after
   rig_code_sync (). A driver that calls the fill after its submit returns
   keeps its own copy: [args] and [slots] are the caller's. */
#if defined(__clang__)
/* A loaded fill need not have function-type metadata before its entry. */
__attribute__((no_sanitize("function")))
#endif
static inline int rig_fill(void *queue, const struct rig_part *p,
                           const uint8_t *args, const uint64_t *slots,
                           uint64_t v) {
  _Alignas(16) uint8_t arg[RIG_PARAMS];
  rig_params(arg, &p->fill.args, args, slots);
  rig_code_sync();
  return p->fill.fn(queue, arg, v);
}

#endif
