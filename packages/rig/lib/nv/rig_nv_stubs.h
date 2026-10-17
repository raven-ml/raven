/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A device's state in C, which the writer (rig_nv_ring.c) and the OCaml
   stubs (rig_nv_stubs.c) share.

   The writer knows no method number: the words it writes are templates the
   OCaml side encodes with the ABI library at open (Rig_nv_abi.Template),
   bytes with patches the writer fills from its values (an address, a
   payload, a count).

   A channel's ring and its segment ring return what a submission used once
   the device's timeline word reaches the submission's value: each channel
   keeps a queue of marks, one per submission that wrote to it. */

#ifndef RIG_NV_STUBS_H
#define RIG_NV_STUBS_H

#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "rig_nv.h"

/* The channels, by queue index: COMPUTE:0, COPY:0. */
enum { COMPUTE, COPY, CHANNELS };

/* The templates the writer writes. */
enum {
  T_ACQUIRE,      /* (address, value): wait until the word is at least value */
  T_RELEASE,      /* (address, value): wait for idle, then write value */
  T_COPY_RELEASE, /* (address, value): after the copies, write value */
  T_COPY,         /* (dst, src, bytes): one copy of at most COPY_MAX bytes */
  T_LOCAL,        /* (address, per_tpc): the compute channel's local memory */
  T_SETUP,        /* (): a channel's engine binding, and compute's windows */
  T_SETUP_COPY,   /* (): the copy channel's engine binding */
  T_INVALIDATE,   /* (): the compute engine's caches */
  T_IDLE,         /* (): wait for the compute engine's launches */
  T_SCHEDULE,     /* (address): schedule the launch descriptor there */
  T_STAMP,        /* (address, value): T_RELEASE, and the timer at +8 */
  T_COPY_STAMP,   /* (address, value): T_COPY_RELEASE, and the timer at +8 */
  TEMPLATES
};

/* A channel template's bytes, and its values: two addresses or an address
   and a value, then a size. */
#define TEMPLATE_WORDS 16
#define TEMPLATE_VALUES 3
#define COPY_MAX (UINT64_C(1) << 31)

/* The most bytes of a segment: an entry's length holds 2^21 - 1 words
   (clc56f.h, GP_ENTRY1_LENGTH 30:10). */
#define SEGMENT_BYTES_MAX (4 * ((UINT64_C(1) << 21) - 1))

/* A patch, Rig_nv_abi.Template.patch: the little-endian word of [width]
   bytes (1, 2, 4 or 8) at byte [at] whose low [bits] bits take
   (value [slot] + [add]) >> [shift]; its other bits stay. */
struct patch {
  uint16_t at;
  uint8_t width, bits, slot, shift;
  uint64_t add;
};

#define STRUCTURE_BYTES 1024
#define PATCHES 16

/* A template: a channel's words or a launch's structure in memory. */
struct template {
  uint32_t nbytes, npatches;
  struct patch patches[PATCHES];
  uint8_t bytes[STRUCTURE_BYTES];
};

/* Template [t]'s bytes in [w], its patches filled from [values]: their
   count. The host is little-endian. */
static inline uint32_t rig_nv_fill(const struct template *t,
                                   const uint64_t *values, uint8_t *w) {
  memcpy(w, t->bytes, t->nbytes);
  for (uint32_t i = 0; i < t->npatches; i++) {
    const struct patch *p = &t->patches[i];
    uint64_t x = (values[p->slot] + p->add) >> p->shift;
    uint64_t mask = p->bits == 64 ? UINT64_MAX : (UINT64_C(1) << p->bits) - 1;
    uint64_t word = 0;
    memcpy(&word, w + p->at, p->width);
    word = (word & ~mask) | (x & mask);
    memcpy(w + p->at, &word, p->width);
  }
  return t->nbytes;
}

/* Launches

   A function set up for launch, as the driver's [entry] made it: its
   descriptors and the driver's parameters of its constant bank 0, as
   templates the writer fills from a launch's values, and the limits the
   room check holds its launches to. */

/* A launch's values, by slot. */
enum {
  V_GRID_X, V_GRID_Y, V_GRID_Z, /* the groups of its grid */
  V_BLOCK_X, V_BLOCK_Y, V_BLOCK_Z, /* the threads of a group */
  V_BANK0,  /* the address of its constant bank 0 */
  V_NEXT,   /* the address of the descriptor it chains */
  V_SHARED, /* its dynamic shared memory, in bytes, a multiple of 128 */
  VALUES
};

/* Descriptor and constant bank alignments, in bytes. */
#define QMD_ALIGN 256
#define BANK_ALIGN 64

struct launch {
  /* [qmd[s][c]]: without (s = 0) or with dynamic shared memory, chaining
     no descriptor (c = 0) or the one at V_NEXT */
  struct template qmd[2][2];
  struct template bank0;
  /* where the kernel's parameters start in bank 0, how many it reads, and
     the bank's bytes the descriptor names, which hold them */
  uint32_t params_at, params, bank0_bytes;
  /* the most groups of a grid and threads of a group along X, Y and Z, the
     most threads of a group, and the most dynamic shared memory */
  uint32_t max[6], max_threads, max_shared;
};

/* A submission's use of a channel: the value, and the ring's, the segment
   ring's and the launch ring's counts once it was written. */
struct mark {
  uint64_t v, put, written, launched;
};

/* A ring a growth made, at its host and GPU addresses, of [size] bytes, a
   power of two: a channel's segment ring, or COMPUTE's launch ring, in the
   GPU's memory through its BAR if [bar]. Once retired, no work after
   [until] uses it, and [next] is the ring retired before it. */
struct ring {
  uint8_t *host;
  uint64_t gpu, size;
  int bar;
  uint64_t until;
  struct ring *next;
};

/* A channel. Counts grow without wrapping; a position is a count modulo
   the ring's size, a power of two. */
struct channel {
  volatile uint64_t *ring; /* the GPFIFO's entries, as the host writes them */
  uint64_t entries, put, freed;
  volatile uint32_t *gp_put; /* in USERD */
  uint32_t token;            /* the work submit token the doorbell takes */
  /* the driver's segments, as the host writes them */
  uint8_t *segments;
  uint64_t segments_gpu, size, written, reclaimed;
  /* the launches' descriptors and banks, as the host writes them, on
     COMPUTE alone; in the GPU's memory through its BAR when [bar] */
  uint8_t *launches;
  uint64_t launches_gpu, launches_size, launched, landed;
  int bar;
  struct mark *marks; /* a queue of [entries] marks */
  uint64_t first, count;
  /* the first values whose work the segment ring and the launch ring hold:
     a mark of an earlier value counts in rings a growth replaced */
  uint64_t segments_from, launches_from;
  uint64_t released; /* the last value this channel released into the word */
  int owes_setup;    /* whether its setup words were not yet placed */
  volatile const uint8_t *notifier; /* its error notification, RM's */
  /* During a submission: where its open segment starts and its words. */
  uint64_t open, open_words;
};

/* A join tag, v << 16 | i + 1, holds a part's index in its low 16 bits:
   the tag's layout bounds a submission's parts, not the GPU. */
#define MAX_PARTS 65535

/* Where a part of a submission runs: its channel; whether the driver's
   copy kernel runs it; whether a later part on the other channel runs
   after it; and the part before it on its queue, when that ran on the
   other channel, or -1. */
struct route {
  uint8_t channel, kernel, awaited;
  int32_t join;
};

/* A submission's routes lie in its scratch. */
_Static_assert(sizeof(struct route) <= RIG_SCRATCH, "a route fits");

/* Orders the host's stores to memory before a store to a register:
   Linux's wmb(). */
static inline void store_fence(void) {
#if defined(__x86_64__)
  __asm__ __volatile__("sfence" ::: "memory");
#elif defined(__aarch64__)
  __asm__ __volatile__("dsb st" ::: "memory");
#else
  atomic_thread_fence(memory_order_seq_cst);
#endif
}

/* Orders every access before it before every access after it: Linux's
   mb(), before a load that must follow the stores before it. */
static inline void full_fence(void) {
#if defined(__x86_64__)
  __asm__ __volatile__("mfence" ::: "memory");
#elif defined(__aarch64__)
  __asm__ __volatile__("dsb sy" ::: "memory");
#else
  atomic_thread_fence(memory_order_seq_cst);
#endif
}

struct rig_nv_edge;

/* A word of a GPU's BAR that a submission reads before its doorbells, which
   returns once the host's writes through that BAR reached the GPU, while
   [count] regions written through it are live: the device's own BAR, or
   the BAR of the GPU whose Mapped memory the device's views reach. [src] is
   the edge of the device that owns the word. An entry whose count fell to
   0 serves another word: its word is stored before its count, and a
   submission reads the count first. The owner's stop takes its word out of
   every other device's entries before its path unmaps it (rig_nv_drop). */
struct flush {
  const struct rig_nv_edge *src;
  _Atomic(volatile const uint32_t *) word;
  _Atomic long count;
};

/* The device's own BAR's word first, then other devices' BAR words. */
#define FLUSHES 8

/* Times

   A timed value's channels write stamps, each a release of the value with
   the GPU's timer, into the value's slot of the stamp ring, in memory the
   host reads: one on each channel it enters, after the waits, and one on
   the channel that releases it, before the release. Value v takes slot
   v modulo the ring's slots, which are more than the values a submission
   may find unreached: a submission that reuses a slot finds its earlier
   value completed, and converts that value's stamps to the host's clock
   unless rig_nv_settle did.

   A stamp converts to the host's clock along the line through two
   measures of the timer against the host's clock, the first, at open, and
   the latest, which rig_nv_settle takes again once it is MEASURE_NS old
   from a path's source, once they lie RATE_BASE_NS apart, and at the
   timer's nominal rate before. A measure whose host readings lie more than
   READ_BOUND_NS apart is dropped. A device of the timer's registers
   measures at open only. Values convert in value order, and a value's
   start is never before the end of the value converted before it: a
   value's work starts once the work of every value before it has
   completed, on every channel. */

/* A release with the timer: the value released and the timer's
   nanoseconds. */
struct record {
  uint64_t value, stamp;
};

/* A slot's records, as the GPU writes them. */
struct stamps {
  struct record start[CHANNELS], end;
};

/* The host's side of a slot: its value, where its times go, NULL once
   they are written or dropped, and the channels it entered, a bit each. */
struct timed {
  uint64_t v;
  uint64_t *times;
  uint8_t entered;
};

/* A reading of the GPU's timer and of the host's clock at that instant. */
struct measure {
  uint64_t gpu, host;
};

/* How old the latest measure of a path's source gets before a settle
   takes another, as often as ROCr's clock correlation. */
#define MEASURE_NS UINT64_C(62500000)

/* How far apart a measure's host readings may lie, and how far apart the
   first and the latest measures lie before their line gives the rate. */
#define READ_BOUND_NS UINT64_C(10000)
#define RATE_BASE_NS UINT64_C(1000000000)

/* What a measure answers: taken, dropped for readings too far apart, or
   failed. */
enum { MEASURE_OK, MEASURE_WIDE, MEASURE_FAILED };

/* What a device runs work with, from its open to its stop. */
struct device {
  _Atomic uint64_t *word; /* the timeline word */
  uint64_t word_gpu;
  uint64_t join_gpu; /* the channels' join words follow it */
  _Atomic uint64_t last; /* the last value submitted */
  struct channel ch[CHANNELS];
  volatile uint32_t *doorbell;
  struct template t[TEMPLATES];
  /* an entry is address + base + words * word */
  uint64_t entry_base, entry_word;
  /* a pending local memory, per_tpc / 32 KiB << (1 + LOCAL_PAGE_BITS) |
     address / 4 KiB << 1, or once a submission took it, v << 1 | LOCAL_TAKEN
     for its value v */
  _Atomic uint64_t local;
  /* whether the compute caches are owed an invalidation */
  _Atomic int invalidate;
  /* The driver's byte-copy kernel; owned until confirmed channel stop. */
  struct launch *copy_launch;
  /* The rings a growth made, which the next room that needs them installs,
     and the rings installed, NULL for those the device opened with: each
     channel's segment ring, then COMPUTE's launch ring. A room retires the
     rings it replaces, and those no larger than the installed ones, which
     the OCaml side takes back. While the device runs, only the OCaml side
     frees a ring's record, under the device's local lock, so a record these
     name stays readable under it; its end frees the rest. */
  _Atomic(struct ring *) grown[CHANNELS + 1];
  _Atomic(struct ring *) current[CHANNELS + 1];
  _Atomic(struct ring *) retired;
  struct flush flushes[FLUSHES];
  /* set while a submission reads [flushes]' words, from before its full
     fence: a stop that took a word out waits until it clears */
  _Atomic int flushing;
  /* a notification's error fields */
  uint32_t info32_at, status_at;
  /* the stamp ring: its [slots] slots, a power of two, as the host reads
     them and at the GPU's address, and their host side */
  volatile struct stamps *stamps;
  uint64_t stamps_gpu, slots;
  struct timed *timed;
  /* the timer: the path's source, or NULL for the registers of the GPU's
     BAR 0 mapped at [bar]; the first and the latest measures, and whether
     the first's readings lie wider than READ_BOUND_NS; every timed
     value up to [converted] converted, and the timed values whose times
     are still to write; the end of the last value converted; and the
     failure of a conversion a submit made, which the next settle reports */
  struct rig_nv_clock *source;
  volatile const uint8_t *bar;
  struct measure first, latest;
  int first_wide;
  uint64_t converted, unconverted, last_end;
  const char *failure;
};

/* What a growth on a thread makes: the bytes each ring of [d] must hold for
   the submission whose rig_nv_room last answered RIG_GROW on the thread, 0
   for those it holds, as each channel's segment ring, then COMPUTE's launch
   ring. */
struct asked {
  const struct device *d;
  uint64_t bytes[CHANNELS + 1];
};

extern _Thread_local struct asked rig_nv_asked;

/* A device's edge, kept for the process: its word, and its run state,
   NULL once stopped. */
struct rig_nv_edge {
  const struct rig_driver *driver;
  _Atomic uint64_t *word;
  struct device *d;
  /* Taken inside one stub, around [d]'s flushes and its free at the stop,
     so no OCaml code runs while it is held. */
  pthread_mutex_t flush_mu;
  /* the path's flush word, NULL without one and from the device's stop:
     no device takes an entry for it then */
  _Atomic(volatile const uint32_t *) bar;
  /* the next live edge (rig_nv_drop) */
  struct rig_nv_edge *next;
};

/* Makes the host's writes through the BARs that [d]'s work may read reach
   their GPUs before the doorbells that follow, a store fence otherwise:
   each live word is read after a full fence, as is the device's own while
   [launched], its launches lying in its BAR. [flushing] is set before the
   fence and each word read after it, so a stop that took a word out either
   finds [flushing] set and waits, or the word is read as NULL. */
static inline void rig_nv_flush(struct device *d, int launched) {
  int fenced = 0;
  for (int i = 0; i < FLUSHES; i++) {
    struct flush *f = &d->flushes[i];
    if (atomic_load_explicit(&f->count, memory_order_acquire) <= 0 &&
        !(i == 0 && launched))
      continue;
    if (!fenced) {
      atomic_store_explicit(&d->flushing, 1, memory_order_relaxed);
      full_fence();
      fenced = 1;
    }
    volatile const uint32_t *w =
        atomic_load_explicit(&f->word, memory_order_acquire);
    if (w != NULL) (void)*w;
  }
  if (fenced) atomic_store_explicit(&d->flushing, 0, memory_order_release);
  else store_fence();
}

/* Reads [d]'s timer against the host's clock into [*m], and the interval
   of its host readings into [*width]: MEASURE_OK, MEASURE_WIDE if the
   readings lie more than READ_BOUND_NS apart, or MEASURE_FAILED if its
   source could not read them. */
int rig_nv_measure(const struct device *d, struct measure *m,
                   uint64_t *width);

/* The join word of channel [q]: its GPU address. */
#define JOIN_GPU(d, q) ((d)->join_gpu + 8 * (1 + (q)))

/* A pending local memory's fields: its address's 4 KiB page from bit 1, in
   the 37 bits of a 49-bit address's page, and its bytes per cluster in units
   of 32 KiB above. Its bit 0 is clear. */
#define LOCAL_PAGE_SHIFT 12
#define LOCAL_PAGE_BITS 37
#define LOCAL_UNIT_SHIFT 15
#define LOCAL_TAKEN UINT64_C(1)

/* The most parts of the largest size an idle device runs in one
   submission (rig_nv_ring.c). */
int rig_nv_most_parts(const struct device *d);

#endif
