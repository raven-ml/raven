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

/* A submission's use of a channel: the value, and the ring's and the
   segment ring's counts once it was written. */
struct mark {
  uint64_t v, put, written;
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
  struct mark *marks; /* a queue of [entries] marks */
  uint64_t first, count;
  uint64_t released; /* the last value this channel released into the word */
  int owes_setup;    /* whether its setup words were not yet placed */
  volatile const uint8_t *notifier; /* its error notification, RM's */
  /* During a submission: where its open segment starts and its words. */
  uint64_t open, open_words;
};

/* A device. The fields up to [last] live for the process; the channels'
   memory ends at a stop that answered Stopped. */
struct device {
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
};

/* The join word of channel [q]: its GPU address. */
#define JOIN_GPU(d, q) ((d)->word_gpu + 8 * (1 + (q)))

/* A pending local memory's fields. */
#define LOCAL_ADDRESS_BITS 40
#define LOCAL_UNIT_SHIFT 15

#endif
