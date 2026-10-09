/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A device's state in C, which the writer (rig_nv_ring.c) and the OCaml
   stubs (rig_nv_stubs.c) share.

   The writer knows no method number: the words it writes are templates the
   OCaml side encodes with the ABI library at open, each a run of words with
   holes, and a hole is a function of one of the writer's values (an
   address, a payload, a count) that the writer fills.

   A channel's ring and its segment ring return what a submission used once
   the device's timeline word reaches the submission's value: each channel
   keeps a queue of marks, one per submission that wrote to it. */

#ifndef RIG_NV_STUBS_H
#define RIG_NV_STUBS_H

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
  TEMPLATES
};

#define TEMPLATE_WORDS 16
#define TEMPLATE_HOLES 6
#define HOLE_OPS 3
#define COPY_MAX (UINT64_C(1) << 31)

/* A hole: the word at index [at], or the two words from it when [wide],
   low first, filled with the writer's value number [slot] after [nops]
   operations in order, each an addition ([shift] 0) of [n] modulo 2^64 or a
   right shift by [shift]. */
struct hole {
  uint16_t at;
  uint8_t slot, wide, nops;
  uint8_t shift[HOLE_OPS];
  uint64_t n[HOLE_OPS];
};

struct template {
  uint32_t words[TEMPLATE_WORDS];
  int nwords, nholes;
  struct hole holes[TEMPLATE_HOLES];
};

/* Template [t]'s words in [w], its holes filled from [values]: their
   count. */
static inline int rig_nv_fill(const struct template *t,
                              const uint64_t *values, uint32_t *w) {
  memcpy(w, t->words, 4 * (size_t)t->nwords);
  for (int i = 0; i < t->nholes; i++) {
    const struct hole *h = &t->holes[i];
    uint64_t x = values[h->slot];
    for (int j = 0; j < h->nops; j++)
      x = h->shift[j] ? x >> h->shift[j] : x + h->n[j];
    w[h->at] = (uint32_t)x;
    if (h->wide) w[h->at + 1] = (uint32_t)(x >> 32);
  }
  return t->nwords;
}

/* Launches

   A function set up for launch, as the driver's [entry] made it: its
   descriptors and the driver's parameters of its constant bank 0, as
   structures with holes the writer fills from a launch's values, and the
   limits the room check holds its launches to.

   A structure's field is the little-endian word of [width] bytes (1, 2, 4
   or 8) at byte [at] whose low [bits] bits take the writer's value number
   [slot] after its operations, as a template's hole does; its other bits
   stay. */
struct field {
  uint16_t at;
  uint8_t width, bits, slot, nops;
  uint8_t shift[HOLE_OPS];
  uint64_t n[HOLE_OPS];
};

#define STRUCTURE_BYTES 1024
#define STRUCTURE_FIELDS 16

struct structure {
  uint32_t nbytes, nfields;
  struct field fields[STRUCTURE_FIELDS];
  uint8_t bytes[STRUCTURE_BYTES];
};

/* A launch's values, by slot. */
enum {
  V_GRID_X, V_GRID_Y, V_GRID_Z, /* the groups of its grid */
  V_BLOCK_X, V_BLOCK_Y, V_BLOCK_Z, /* the threads of a group */
  V_BANK0,  /* the address of its constant bank 0 */
  V_NEXT,   /* the address of the descriptor it chains */
  V_SHARED, /* its dynamic shared memory, in bytes, a multiple of 128 */
  VALUES
};

/* Structure [s]'s bytes in [w], its fields filled from [values]. The host
   is little-endian. */
static inline void rig_nv_fill_structure(const struct structure *s,
                                         const uint64_t *values, uint8_t *w) {
  memcpy(w, s->bytes, s->nbytes);
  for (uint32_t i = 0; i < s->nfields; i++) {
    const struct field *f = &s->fields[i];
    uint64_t x = values[f->slot];
    for (int j = 0; j < f->nops; j++)
      x = f->shift[j] ? x >> f->shift[j] : x + f->n[j];
    uint64_t mask = f->bits == 64 ? UINT64_MAX : (UINT64_C(1) << f->bits) - 1;
    uint64_t word = 0;
    memcpy(&word, w + f->at, f->width);
    word = (word & ~mask) | (x & mask);
    memcpy(w + f->at, &word, f->width);
  }
}

/* Descriptor and constant bank alignments, in bytes. */
#define QMD_ALIGN 256
#define BANK_ALIGN 64

struct launch {
  /* [qmd[s][c]]: without (s = 0) or with dynamic shared memory, chaining
     no descriptor (c = 0) or the one at V_NEXT */
  struct structure qmd[2][2];
  struct structure bank0;
  /* where the kernel's parameters start in bank 0, and the bank's bytes the
     descriptor names */
  uint32_t params_at, bank0_bytes;
  /* the most groups of a grid and threads of a group along X, Y and Z, the
     most threads of a group, and the most dynamic shared memory */
  uint32_t max[6], max_threads, max_shared;
};

/* A submission's use of a channel: the value, and the ring's, the segment
   ring's and the launch ring's counts once it was written. */
struct mark {
  uint64_t v, put, written, launched;
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
  uint64_t released; /* the last value this channel released into the word */
  int owes_setup;    /* whether its setup words were not yet placed */
  volatile const uint8_t *notifier; /* its error notification, RM's */
  /* During a submission: where its open segment starts and its words. */
  uint64_t open, open_words;
};

/* A join tag holds a part's index in 16 bits. */
#define MAX_PARTS 65535

/* A device. The fields up to [last] live for the process; the channels'
   memory ends at a stop that answered Stopped. */
struct device {
  const struct rig_driver *driver;
  _Atomic uint64_t *word; /* the timeline word, then the two join words */
  uint64_t word_gpu;
  _Atomic uint64_t last; /* the last value submitted */
  struct channel ch[CHANNELS];
  volatile uint32_t *doorbell;
  struct template t[TEMPLATES];
  /* an entry is address + base + words * word */
  uint64_t entry_base, entry_word;
  /* a pending local memory: address | per_tpc / 32 KiB << 40 */
  _Atomic uint64_t local;
  /* whether the compute caches are owed an invalidation */
  _Atomic int invalidate;
  /* live regions the host writes through the BAR */
  _Atomic long bar_live;
  /* a word of the BAR, read to flush its writes */
  volatile const uint32_t *bar;
  /* a notification's error fields */
  uint32_t info32_at, status_at;
  /* during a submission: whether a later part on the other channel runs
     after part i */
  uint8_t awaited[MAX_PARTS];
};

/* The join word of channel [q]: its GPU address. */
#define JOIN_GPU(d, q) ((d)->word_gpu + 8 * (1 + (q)))

/* A pending local memory's fields. */
#define LOCAL_ADDRESS_BITS 40
#define LOCAL_UNIT_SHIFT 15

#endif
