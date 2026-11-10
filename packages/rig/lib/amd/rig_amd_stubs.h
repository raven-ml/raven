/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A device's state in C, which rig_amd_ring.c writes the rings from and
   rig_amd_stubs.c fills.

   The state is allocated at open and freed with the timeline word, which
   rig gives back after the stop, once nothing reads it.
   Positions on a ring count 32-bit words placed since the ring was made and
   never wrap; a word's place is its position modulo the ring's size. */

#ifndef RIG_AMD_STUBS_H
#define RIG_AMD_STUBS_H

#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "rig_amd.h"

/* The submissions a ring keeps the end of until their value is reached. */
#define RIG_AMD_MARKS 4096

/* The queues, in the order of Rig_amd's facts. */
enum { RIG_AMD_COMPUTE, RIG_AMD_COPY, RIG_AMD_QUEUES };

/* The packets a ring reads, in Rig_amd's order. On an AQL ring a packet
   is 16 words whose first is stored last, and the writer's own PM4 words go
   in an indirect buffer in the segment; its write position counts packets.
   An SDMA ring's packets never wrap; its write position counts bytes. */
enum { RING_PM4, RING_AQL, RING_SDMA };

/* The scratch writes an AQL queue's descriptor takes at the next
   submission's first compute part, which also makes the pending scratch
   the one launches name. */
#define RIG_AMD_SCRATCH_WRITES 8

/* The bytes of an AQL queue's descriptor that hold its scratch, from
   compute_tmpring_size to scratch_wave64_lane_byte_size (amd_hsa_queue.h's
   amd_queue_t), which Rig_amd's scratch writes also name. */
#define RIG_AMD_DESC_SCRATCH 140
#define RIG_AMD_DESC_SCRATCH_BYTES 40

/* The packets the writer places, which Rig_amd encodes once per device.
   Their arguments: 0 an address, 1 a value, and for a copy 0 its
   destination, 1 its source, 2 its bytes. */
enum {
  T_WAIT,      /* compute: the 32 bits at 0 equal the low 32 bits of 1 */
  T_FLUSH,     /* compute: wait for the dispatches before to complete */
  T_ACQUIRE,   /* compute: invalidate every cache, the L2 written back */
  T_WAIT64,    /* compute: the 64 bits at 0 are at least 1 */
  T_SIGNAL,    /* compute: once the work before completed, low 32 of 1 at 0 */
  T_RELEASE,   /* compute: once the work before completed, 1 at 0, interrupt */
  T_WRITE,     /* compute: the low 32 bits of 1 at 0 */
  S_POLL,      /* copy: the 32 bits at 0 equal the low 32 bits of 1 */
  S_FENCE,     /* copy: once the work before completed, low 32 of 1 at 0 */
  S_TRAP,      /* copy: interrupt */
  S_COPY,      /* copy: 2 bytes from 1 to 0, at most the engine's packet */
  A_IB,        /* AQL compute: the 1 words of PM4 packets at 0 */
  T_INVALIDATE, /* compute: invalidate the caches above the L2 */
  T_STAMP,     /* compute: the dispatches before completed, the clock at 0 */
  S_STAMP,     /* copy: the clock at 0 */
  S_FILL,      /* copy: 2 bytes from 0, a multiple of 4, set to word 1 */
  RIG_AMD_TEMPLATES
};

#define RIG_AMD_TEMPLATE_WORDS 16
#define RIG_AMD_TEMPLATE_HOLES 4
#define RIG_AMD_TEMPLATE_ARGS 3

/* A word of a template that holds a computation on an argument [v],
   ((v + add) >> shift) | set, written as one word or as two, low first. */
struct rig_amd_hole {
  uint16_t at;
  uint8_t wide, arg, shift;
  uint64_t add, set;
};

struct rig_amd_template {
  uint32_t words[RIG_AMD_TEMPLATE_WORDS];
  int n, nholes;
  struct rig_amd_hole holes[RIG_AMD_TEMPLATE_HOLES];
};

/* Writes into [w] the [n] holes at [h], filled from [args]. */
static inline void rig_amd_holes(const struct rig_amd_hole *h, int n,
                                 const uint64_t *args, uint32_t *w) {
  for (; n > 0; n--, h++) {
    uint64_t v = ((args[h->arg] + h->add) >> h->shift) | h->set;
    w[h->at] = (uint32_t)v;
    if (h->wide == 2) w[h->at + 1] = (uint32_t)(v >> 32);
  }
}

/* Template [t]'s words in [w], its holes filled from [args]: their
   count. */
static inline int rig_amd_fill(const struct rig_amd_template *t,
                               const uint64_t *args, uint32_t *w) {
  memcpy(w, t->words, sizeof t->words);
  rig_amd_holes(t->holes, t->nholes, args, w);
  return t->n;
}

/* A function's launch, which the device's [entry] makes once per function
   of a loaded image and which lives until the image is unloaded: the words
   of its dispatch, whose holes take the hand-over's arguments below. On a
   PM4 queue they are PM4 packets, whose word at [pm4.lds_at] is
   [pm4.lds_word] plus [pm4.lds_unit] for each [pm4.lds_granule] bytes,
   rounded up, of the LDS its groups take. On an AQL queue they are one
   kernel dispatch packet, whose 16-bit workgroup sizes lie at the bytes
   [aql.threads] and whose 32-bit group segment size at [aql.group]. The
   LDS its groups take is the function's [group] bytes and the launch's
   shared memory. Its arguments take [kernarg] bytes, after the launch's
   parameters at most, and [hidden] holds the offset among them of each
   implicit argument it reads, in the order below, or -1. A launch fits
   when its parameters are its function's [explicit_size] bytes or more,
   its threads per group at most [max_threads] and its shared memory at
   most [max_shared] bytes. */

/* The hand-over's arguments: the address of the launch's arguments, the
   device's scratch, its threads per group, its groups, and its grid's
   work-items along x, y and z. */
enum {
  L_ARGS,
  L_SCRATCH,
  L_THREADS,
  L_GROUPS = L_THREADS + 3,
  L_GRID = L_GROUPS + 3,
  RIG_AMD_LAUNCH_ARGS = L_GRID + 3
};

/* The implicit arguments of code object v5 a launch writes: its groups
   along each axis (32 bits), its threads per group along each (16 bits),
   the threads of a partial last group along each (16 bits), the first
   thread's index along each (64 bits), the axes of its grid (16 bits) and
   its shared memory's bytes (32 bits). */
enum {
  H_BLOCKS,
  H_GROUP = H_BLOCKS + 3,
  H_REMAINDER = H_GROUP + 3,
  H_OFFSET = H_REMAINDER + 3,
  H_DIMS = H_OFFSET + 3,
  H_LDS,
  RIG_AMD_HIDDEN
};

/* The bytes of each implicit argument, which a launch's check and its
   writer both read. */
static const uint8_t rig_amd_hidden_bytes[RIG_AMD_HIDDEN] = {
    [H_BLOCKS] = 4,      [H_BLOCKS + 1] = 4,    [H_BLOCKS + 2] = 4,
    [H_GROUP] = 2,       [H_GROUP + 1] = 2,     [H_GROUP + 2] = 2,
    [H_REMAINDER] = 2,   [H_REMAINDER + 1] = 2, [H_REMAINDER + 2] = 2,
    [H_OFFSET] = 8,      [H_OFFSET + 1] = 8,    [H_OFFSET + 2] = 8,
    [H_DIMS] = 2,        [H_LDS] = 4};

#define RIG_AMD_LAUNCH_WORDS 64
#define RIG_AMD_LAUNCH_HOLES 12

/* The bytes of implicit arguments code object v5 places after a kernel's
   explicit ones, which start on 8 bytes. */
#define RIG_AMD_HIDDEN_BYTES 256

struct rig_amd_launch {
  uint32_t words[RIG_AMD_LAUNCH_WORDS];
  int n, nholes, packet; /* an AQL packet, else PM4 */
  struct rig_amd_hole holes[RIG_AMD_LAUNCH_HOLES];
  union {
    struct {
      uint32_t lds_at, lds_word, lds_unit, lds_granule;
    } pm4;
    struct {
      uint32_t threads[3], group;
    } aql;
  };
  uint32_t group, max_threads, max_shared, kernarg, explicit_size;
  uint32_t lanes; /* of its waves: 32 or 64 */
  int32_t hidden[RIG_AMD_HIDDEN];
};

/* Launch [l]'s dispatch in [w], its holes filled from the hand-over's first
   arguments [args], up to its grid, which they give, and its LDS from its
   groups' [shared] bytes: its count. */
static inline int rig_amd_dispatch(const struct rig_amd_launch *l,
                                   const uint64_t *args, uint32_t shared,
                                   uint32_t *w) {
  uint64_t a[RIG_AMD_LAUNCH_ARGS];
  uint32_t lds = l->group + shared;
  memcpy(a, args, L_GRID * sizeof *a);
  for (int k = 0; k < 3; k++) a[L_GRID + k] = a[L_GROUPS + k] * a[L_THREADS + k];
  memcpy(w, l->words, 4 * (size_t)l->n);
  rig_amd_holes(l->holes, l->nholes, a, w);
  if (l->packet) {
    uint8_t *b = (uint8_t *)w;
    for (int k = 0; k < 3; k++) {
      uint16_t t = (uint16_t)a[L_THREADS + k];
      memcpy(b + l->aql.threads[k], &t, 2);
    }
    memcpy(b + l->aql.group, &lds, 4);
  } else
    w[l->pm4.lds_at] =
        l->pm4.lds_word +
        (lds + l->pm4.lds_granule - 1) / l->pm4.lds_granule * l->pm4.lds_unit;
  return l->n;
}

/* A submission that used a ring or the segment: its value, and the
   position its words or bytes end at. */
struct rig_amd_mark {
  uint64_t v, end;
};

/* Room handed out in order and taken back once values are reached. */
struct rig_amd_marks {
  struct rig_amd_mark at[RIG_AMD_MARKS];
  uint32_t head, count;
};

struct rig_amd_ring {
  volatile uint32_t *words;
  uint64_t size;                /* words, a power of two */
  uint64_t most;                /* words of the largest ring the path makes */
  volatile uint64_t *write;     /* the write position the queue reads */
  volatile uint64_t *doorbell;
  int kind;                     /* RING_PM4, RING_AQL or RING_SDMA */
  uint64_t put, free;           /* placed, done */
  uint64_t released;            /* the last value this ring released */
  struct rig_amd_marks marks;
};

struct rig_amd_segment {
  uint8_t *host;
  uint64_t gpu, size;
  uint64_t put, free;           /* bytes, as a ring's positions */
  struct rig_amd_marks marks;
};

/* What a growth made (rig_amd_ring.c's Growth): an argument segment of
   [size] bytes at the host address [host] and the GPU's [gpu], which also
   names it, of 0 bytes for none; for each queue, a new queue of the path's
   on a ring of [size] words at [words], with its write-position word and
   doorbell, named by [id], the ring's GPU address, of 0 words for none;
   and for a new compute queue that reads AQL packets, its descriptor, at
   the host address [desc] and the GPU's [desc_gpu]. A new queue's read and
   write positions are 0. */
struct rig_amd_grown {
  uint8_t *host;
  uint64_t gpu, size;
  struct {
    volatile uint32_t *words;
    uint64_t size;
    volatile uint64_t *write;
    volatile uint64_t *doorbell;
    uint64_t id;
  } ring[RIG_AMD_QUEUES];
  uint8_t *desc;
  uint64_t desc_gpu;
};

/* Where a part of a submission runs: the ring that runs it; whether the
   driver's copy kernel runs it; whether a later part on the other ring
   runs after it; and the part before it on its queue, when that ran on
   the other ring, or -1. */
struct rig_amd_route {
  uint8_t ring, kernel, named;
  int32_t join;
};

/* A submission's routes lie in the scratch rig hands the driver. */
_Static_assert(sizeof(struct rig_amd_route) <= RIG_SCRATCH, "a route fits");

/* Times

   A timed value's queues write the GPU's clock into the value's slot of
   the stamp ring, in memory the host reads: each queue it enters after
   the waits, and the queue that releases it before the release. Value v
   takes slot v modulo RIG_AMD_STAMPS. A submission finds at most
   RIG_AMD_MARKS - 1 values unreached, one segment mark each, so one that
   reuses a slot finds its earlier value completed, and converts that
   value's stamps to the host's clock unless rig_amd_settle did. A record
   the GPU did not write stays 0.

   A stamp converts to the host's clock along the line through two
   measures of the clock against the host's clock, the first, at open, and
   the latest, which rig_amd_settle, and a submit that converts an earlier
   value's stamps, take again once it is RIG_AMD_MEASURE_NS old from a
   path's source, once they lie
   RIG_AMD_RATE_BASE_NS apart, and at the clock's nominal rate before. A
   read whose host readings lie more than RIG_AMD_READ_BOUND_NS apart is
   dropped. A source may count a counter of the clock's rate that runs a
   constant apart from the clock, as KFD's does on GFX 12: at open the
   device takes that constant from stamps the compute queue writes between
   two readings of the host's clock, and adds it to every read: against a
   read within RIG_AMD_READ_BOUND_NS, taken before the stamps or, if none
   is, after them. A device without a source measures at open only, with
   such stamps. With a source, a conversion that puts a value's start
   before its submit or its end after the conversion, by more than
   RIG_AMD_MOVED_NS, leaves the value untimed: the clock moved against the
   source's counter, or a read of the source was late.
   Values convert in value order, and a value's start is never before the
   end of the value converted before it: a value's work starts once the
   work of every value before it has completed, on every queue. */
#define RIG_AMD_STAMPS RIG_AMD_MARKS
#define RIG_AMD_MEASURE_NS UINT64_C(62500000)
#define RIG_AMD_READ_BOUND_NS UINT64_C(10000)
#define RIG_AMD_RATE_BASE_NS UINT64_C(1000000000)
#define RIG_AMD_MOVED_NS UINT64_C(1000000)

/* A slot's records, as the GPU writes them: counts of its clock, and the
   word the copy queue's fence before its end writes. */
struct rig_amd_stamps {
  uint64_t start[RIG_AMD_QUEUES], end, fence;
};

/* The host's side of a slot: its value, where its times go, NULL once
   they are written or dropped, the queues it entered, a bit each, and the
   host's time at its submit. */
struct rig_amd_timed {
  uint64_t v;
  uint64_t *times;
  uint8_t entered;
  uint64_t submitted;
};

/* A reading of the GPU's clock and of the host's clock at that instant. */
struct rig_amd_measure {
  uint64_t ticks, host;
};



/* The layout of the copy engine's COPY_LINEAR_SUB_WIND packet
   (Rig_amd_abi.Sdma.window): its header, the lowest bit of each field, and
   the most elements of a row, rows of a packet and elements of a pitch. */
struct rig_amd_window {
  uint32_t header;
  int element, src_x, src_pitch, dst_x, dst_pitch, width, rows;
  uint64_t most_width, most_rows, most_pitch;
};

/* Counters

   A device opened to measure counters brackets each launch of a timed
   value on its compute queue, which reads PM4 packets: a one-wave
   dispatch of the filter kernel stores the counter filter of its own VMID
   (Rig_amd_abi.Counter.filter) into the launch's record; the start
   program loads it into the filter register and starts counting; then a
   stamp, the launch and a stamp; the stop program stores the counters
   into the record. After the value's last launch the filter kernel stores
   the value's tail word. The records lie in the value's run of the
   argument segment, each [RIG_AMD_RECORD_*] bytes in, and the GPU writes
   them before the word shows the value. Before the segment's room is
   taken back, the host copies each reached value's records into its own
   (rig_amd_ring.c's Counters). */
#define RIG_AMD_PROGRAM_WORDS 16384
#define RIG_AMD_PROGRAM_HOLES 2048
#define RIG_AMD_RECORD_FILTER 0
#define RIG_AMD_RECORD_START 8
#define RIG_AMD_RECORD_END 16
#define RIG_AMD_RECORD_SAMPLES 24

/* Packets whose holes read one argument, the address of a record. */
struct rig_amd_program {
  uint32_t *words;
  struct rig_amd_hole *holes;
  int n, nholes;
};

/* A measured value's launches, keyed by the value: where each launch's
   record and the value's tail word and drain lie in the segment, at their
   host addresses, the drain's NULL for none; each launch's waves; and once
   copied, for each launch, its status, 1 if measured, its start and end
   on the host's clock and its values, as rig_edge.mli's [measures]
   writes them. A value dropped before it was copied is freed once it
   is. */
struct rig_amd_counted {
  uint64_t v;
  int n;
  int copied, dropped;
  volatile uint8_t *tail, *drain;
  volatile uint8_t **at;
  uint64_t *waves;
  uint64_t *rows;
};

struct rig_amd_counting {
  struct rig_amd_launch filter;
  /* around a launch; and the drain's, around no launch */
  struct rig_amd_program start, stop, drain_start, drain_stop;
  uint64_t values, drain_values; /* the samples' 8-byte values */
  uint64_t waves;                /* SQ_WAVES's values, the samples' first */
  uint64_t saved;                /* SQ_WAVES_SAVED's values, the next */
  uint32_t *scratch;             /* a program's words, filled */
  /* a filter word the next bracket stores in place of its filter wave's,
     or -1 (Rig_amd.stale_filter) */
  _Atomic int64_t forced;
  /* set once a VMID moved, until a drain reads nothing; every value up
     to [stale] is unmeasured */
  int marked;
  uint64_t stale;
  /* in value order, those before [uncopied] copied */
  struct rig_amd_counted **counted;
  size_t ncounted, cap, uncopied;
};

/* What a device runs work with, from its open to its stop. */
struct rig_amd {
  _Atomic uint64_t *word;
  uint64_t word_gpu;
  struct rig_amd_ring rings[RIG_AMD_QUEUES];
  struct rig_amd_segment segment;
  struct rig_amd_template templates[RIG_AMD_TEMPLATES];
  uint64_t max_copy;            /* the bytes one S_COPY moves */
  uint64_t max_fill;            /* the bytes one S_FILL sets */
  struct rig_amd_window window;
  const struct rig_amd_launch *copy_launch; /* owned until confirmed stop */
  /* the register whose store flushes the GPU's host data path, or NULL */
  volatile uint32_t *hdp;
  /* the regions another function was given, whose writes pass that path:
     each submission flushes it while any is counted */
  _Atomic int given;
  uint64_t ib_at;               /* the PM4 words not yet in an AQL packet */
  size_t ib_n;
  uint64_t scratch_gpu;         /* the scratch the launches name */
  uint64_t scratch_next;        /* the pending scratch */
  int scratch_n;                /* the descriptor's pending writes */
  uint64_t scratch_at[RIG_AMD_SCRATCH_WRITES]; /* offsets in the descriptor */
  uint32_t scratch_value[RIG_AMD_SCRATCH_WRITES];
  _Atomic int scratch_ready;
  _Atomic int scratch_lock;     /* held while the writes change or are read */
  _Atomic uint64_t scratch_taken;   /* the value that placed them */
  _Atomic uint64_t last;
  /* an AQL compute queue's descriptor, at its host and GPU addresses */
  uint8_t *desc;
  uint64_t desc_gpu;
  /* What a growth made, which the next room that needs it installs once
     the device reached every value submitted, NULL for none, and the names
     of the segment and of each queue's ring installed, under [grown_mu],
     which rooms take only to install and a growth to hand; and whether
     every room waits for the device to reach every value submitted, to
     install it: rooms, which run under rig's turn, alone read and write
     [draining]. */
  pthread_mutex_t grown_mu;
  struct rig_amd_grown *grown;
  uint64_t installed[1 + RIG_AMD_QUEUES];
  int draining;
  /* the stamp ring, as the host reads it and at the GPU's address, and its
     host side; the clock: the path's source, or NULL, the record a stamp
     of the measure at open writes, the clock's rate, the clock's count
     less the source's at one instant, the first and the latest measures,
     in the clock's counts; every timed value up to [converted] converted, and
     the timed values whose times are still to write; the end of the last
     value converted; and the failure of a conversion a submit made, which
     the next settle reports */
  volatile struct rig_amd_stamps *stamps;
  uint64_t stamps_gpu;
  struct rig_amd_timed timed[RIG_AMD_STAMPS];
  struct rig_amd_clock *source;
  volatile uint64_t *clock;
  uint64_t clock_gpu, clock_hz;
  int64_t offset;
  struct rig_amd_measure first, latest;
  int first_wide; /* the first's readings lie wider than the bound */
  uint64_t converted, unconverted, last_end;
  const char *times_failure;
  /* NULL unless the device measures counters */
  struct rig_amd_counting *counting;
};

/* A device's edge, kept for the process: its word, and its run state,
   NULL once stopped. */
struct rig_amd_edge {
  const struct rig_driver *driver;
  _Atomic uint64_t *word;
  struct rig_amd *d;
};

/* Takes [d]'s first measure of its clock: from its source, the first of
   sixteen reads whose host readings lie within RIG_AMD_READ_BOUND_NS, or
   the narrowest, then the source's constant from stamps of the compute
   queue; without one, from stamps of the compute queue. Answers
   [1], or [0] if the source could not read the clock or the GPU wrote no
   stamp within a second. Called at open, before any submission. */
int rig_amd_measure(struct rig_amd *d);

/* Orders every store before it before every store after it, as the GPU
   sees them: on arm64 the full-system barrier stores to a BAR need; on
   x86_64 mfence. */
static inline void rig_amd_barrier(void) {
#if defined(__aarch64__)
  __asm__ __volatile__("dsb sy" ::: "memory");
#elif defined(__x86_64__)
  __asm__ __volatile__("mfence" ::: "memory");
#else
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

/* What a growth on a thread makes: the bytes the argument segment and
   each ring of [d] must have for the submission whose rig_amd_room last
   answered RIG_GROW on the thread, 0 for those that hold it. */
struct rig_amd_asked {
  const struct rig_amd *d;
  uint64_t bytes, rings[RIG_AMD_QUEUES];
};

extern _Thread_local struct rig_amd_asked rig_amd_asked;

/* Frees [c] and what it holds. */
void rig_amd_counting_free(struct rig_amd_counting *c);

#endif
