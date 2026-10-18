/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Handing work to a device from C.

   A device's driver keeps its C state for one device, [self], whose first
   member is a pointer to the driver's struct rig_driver: four functions
   in the shapes rig_room_fn, rig_submit_fn, rig_commit_fn and
   rig_settle_fn, which rig calls with [self] and the structures below.
   The section "The C functions" of rig_edge.mli states what each must do
   and how rig calls them. This header gives their C shapes and the layout
   of what they are handed. rig calls them without the OCaml runtime: they
   must call no function of it and read no OCaml value.

   A submission is the work of one value [v] of the device's timeline: an
   array of parts, each on one of the device's queues, and the waits on
   other devices' words it starts after. */

#ifndef RIG_EDGE_H
#define RIG_EDGE_H

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* What a room check answers (rig_edge.mli, The C functions). */
enum { RIG_FITS, RIG_LATER, RIG_NEVER, RIG_GROW };

/* What a submission, a commit and a settle answer (rig_edge.mli, The C
   functions). */
enum { RIG_OK, RIG_FAILED, RIG_COMMITTED };

/* Which side of a copy is memory of the calling process, if any. */
enum { RIG_LOCAL_NONE, RIG_LOCAL_SRC, RIG_LOCAL_DST };

/* A wait: the 64-bit word at the mapped address [at] holds at least
   [value]. */
struct rig_wait {
  uint64_t at, value;
};

/* A part's kinds. 0 is no part: a room check answers RIG_NEVER for it. */
enum { RIG_WORDS = 1, RIG_FILL, RIG_COPY, RIG_LAUNCH };

/* A launch's or a fill's block in a run's [args]: the groups of its grid
   along x, y and z, the threads of a group, a group's dynamic shared memory
   in bytes, then its parameters. A fill uses only the parameters. A block
   starts on 16 bytes. Its parameters, rounded up to 16 bytes, are followed
   by its place (rig_place), as rig_block_bytes lays them out. */
struct rig_block {
  uint32_t groups[3], threads[3], shared, unused;
  uint8_t params[];
};

/* [params] rounded up to the 16 bytes a block's parts start on. */
static inline uint64_t rig_round16(uint64_t params) {
  return (params + 15) & ~(uint64_t)15;
}

/* The bytes of a block of [params] parameter bytes in [args]: its header,
   its parameters and its place, each up to a multiple of 16 bytes. The next
   block starts there. Whoever lays out blocks for a submit lays them out so,
   since the driver's submit may write each place. */
static inline uint64_t rig_block_bytes(uint64_t params) {
  return sizeof(struct rig_block) + 2 * rig_round16(params);
}

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
     driver places the parameters, with the addresses added (rig_params),
     into memory of its own or into the block's place (rig_place): [args] is
     the caller's, and is used only during the call.
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
   to 8 bytes: the [n * RIG_SCRATCH] bytes at [scratch] that a room check
   and a submit are handed. rig_edge.mli's section "The C functions" states
   how the two may use them. */
#define RIG_SCRATCH 8

/* The room check of the [n] parts at [parts], whose launches' and fills'
   blocks lie in [args] (rig_edge.mli, The C functions: [room]). */
typedef int rig_room_fn(void *self, const struct rig_part *parts, int n,
                        void *scratch, const uint8_t *args);

/* Encodes [parts] as the work of [v] after [waits] (rig_edge.mli, The C
   functions: [submit]). [args] holds the launches' and fills' blocks, of
   which the driver writes only their places (rig_place), and [slots] the
   address of each of the run's [nslots] buffers, as the device's work
   addresses it, for their refs. [times] is NULL unless a profile is being
   taken; otherwise the driver stores at [times[0]] and [times[1]] the
   times of rig_now_ns at which [v]'s work started and ended on the device.
   A failure sets [*failure] to the driver's message, which must live as
   long as the device. */
typedef int rig_submit_fn(void *self, uint64_t v, const struct rig_wait *waits,
                         int nwaits, const struct rig_part *parts, int nparts,
                         void *scratch, uint8_t *args,
                         const uint64_t *slots, int nslots, uint64_t *times,
                         const char **failure);

/* Commits the device's work up to [v] (rig_edge.mli, The C functions:
   [commit]). A failure sets [*failure] as a submission's. */
typedef int rig_commit_fn(void *self, uint64_t v, const char **failure);

/* Writes the times of [v], whose submission was given [times], unless the
   driver wrote them already (rig_edge.mli, The C functions: [settle]). A
   failure sets [*failure] as a submission's. A driver that writes the
   times before the word shows [v] sets [settle] to NULL. */
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
   bytes and shares none with the block's parameters. */
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

/* The place of the block of [a] in [args]: [a->params] bytes, on 16 bytes,
   that its driver's submit may write during the call, such as to place the
   parameters (rig_params) for an API that copies them at a call or for
   code the host runs. Its bytes are undefined when the call starts, and
   rig never reads them. */
static inline uint8_t *rig_place(const struct rig_args *a, uint8_t *args) {
  return args + a->block + sizeof(struct rig_block) + rig_round16(a->params);
}

/* Timeline words

   rig keeps every device's timeline word in one static array of RIG_SLOTS
   lines of RIG_LINE bytes, the word of slot [i] in the first 8 bytes of
   line [i]. The array starts on a 64 KiB boundary and fills its pages,
   which hold nothing else, so a driver may map them whole. It lives for
   the process, and a slot's word passes from one device to the next that
   takes the slot. rig_words () is its first line. */
#define RIG_SLOTS 65536
#define RIG_LINE 64
#define RIG_WORDS_BYTES ((size_t)RIG_SLOTS * RIG_LINE)

_Atomic uint64_t *rig_words(void);

/* The OCaml primitive [caml_rig_edge_words : unit -> int * int] answers the
   array's address and RIG_WORDS_BYTES, for OCaml code that maps it whole. */

/* rig_now_ns () is the host's clock: nanoseconds of the system's monotonic
   clock from an unspecified start (CLOCK_MONOTONIC on Linux,
   QueryPerformanceCounter on Windows); on macOS, mach time
   (CLOCK_UPTIME_RAW), the time base of Metal's command buffer times. It is
   the clock of [Rig.Profile.now]. A driver writes a value's times and
   measures its waits on it (Rig_edge's calls section). OCaml reads it
   through the noalloc external caml_rig_edge_now, which answers an
   untagged int, and caml_rig_edge_now_byte. */
uint64_t rig_now_ns(void);

/* Locks

   rig_holds () is the number of locks the calling thread holds: rig's, a
   driver's and an io library's, each that OCaml code runs inside or that a
   stub holds while it runs pending signal handlers. rig_hold () counts one,
   from before its take, and answers the count before it; rig_unhold ()
   counts its give, after it. rig_edge.mli's Drivers section states which
   locks a driver counts and which of its functions must check the count
   as they start. OCaml reads the count through the noalloc external
   caml_rig_edge_holds, which answers an untagged int, and
   caml_rig_edge_holds_byte, and counts a lock of its own through the
   noalloc externals caml_rig_edge_hold and caml_rig_edge_unhold, of type
   unit -> unit, which call rig_hold and rig_unhold. A library that links
   rig.edge declares each external it uses itself. */
int rig_holds(void);
int rig_hold(void);
void rig_unhold(void);

/* rig_code_sync () makes the calling thread run the code that rig linked on
   the host before the call ([Rig.Image.load] on [Rig.host]): on arm64 it
   synchronizes the thread's instruction stream with that code, once per
   link. A driver calls it before it calls a fill's function by its own
   means: the function may be such code. */
void rig_code_sync(void);

/* Calls the fill [p] with the queue's context [queue] and the value [v], on
   its argument placed into its block's place (rig_place), after
   rig_code_sync (). A driver that calls the fill after its submit returns
   keeps its own copy: [args] and [slots] are the caller's. */
#if defined(__clang__)
/* A loaded fill need not have function-type metadata before its entry. */
__attribute__((no_sanitize("function")))
#endif
static inline int rig_fill(void *queue, const struct rig_part *p,
                           uint8_t *args, const uint64_t *slots, uint64_t v) {
  uint8_t *arg = rig_place(&p->fill.args, args);
  rig_params(arg, &p->fill.args, args, slots);
  rig_code_sync();
  return p->fill.fn(queue, arg, v);
}

#endif
