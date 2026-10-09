/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A device's state in C, which rig_amd_ring.c writes the rings from and
   rig_amd_stubs.c fills.

   The state is allocated at open and never freed: it holds the timeline
   word's address, which other devices may poll after the device is gone.
   Positions on a ring count 32-bit words placed since the ring was made and
   never wrap; a word's place is its position modulo the ring's size. */

#ifndef RIG_AMD_STUBS_H
#define RIG_AMD_STUBS_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "rig_amd.h"

/* The most parts and waits of a submission, the slot words (one per part
   and slot W, which the compute queue signals once it passed the waits),
   and the submissions a ring keeps the end of until their value is
   reached. */
#define RIG_AMD_PARTS 512
#define RIG_AMD_WAITS 255
#define RIG_AMD_SLOTS (RIG_AMD_PARTS + 1)
#define RIG_AMD_SLOT_W RIG_AMD_PARTS
#define RIG_AMD_MARKS 4096
#define RIG_AMD_HDPS 8

/* The queues, in Rig_amd.queues's order. */
enum { RIG_AMD_COMPUTE, RIG_AMD_COPY, RIG_AMD_QUEUES };

/* The packets a ring reads, in Rig_amd's order. On an AQL ring a packet
   is 16 words whose first is stored last, and the writer's own PM4 words go
   in an indirect buffer in the segment; its write position counts packets.
   An SDMA ring's packets never wrap; its write position counts bytes. */
enum { RING_PM4, RING_AQL, RING_SDMA };

/* The scratch writes an AQL queue's descriptor takes at the next
   submission's first compute part. */
#define RIG_AMD_SCRATCH_WRITES 8

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
  RIG_AMD_TEMPLATES
};

#define RIG_AMD_TEMPLATE_WORDS 16
#define RIG_AMD_TEMPLATE_HOLES 4
#define RIG_AMD_HOLE_OPS 3

/* A word of a template that holds a computation on an argument: the
   argument, then each operation in turn (add, shift right, or), written as
   one word or as two, low first. */
enum { OP_ADD, OP_SHIFT, OP_OR };

struct rig_amd_hole {
  uint8_t at, wide, arg, nops;
  uint8_t op[RIG_AMD_HOLE_OPS];
  uint64_t k[RIG_AMD_HOLE_OPS];
};

struct rig_amd_template {
  uint32_t words[RIG_AMD_TEMPLATE_WORDS];
  int n, nholes;
  struct rig_amd_hole holes[RIG_AMD_TEMPLATE_HOLES];
};

/* Template [t]'s words in [w], its holes filled from [args]: their
   count. */
static inline int rig_amd_fill(const struct rig_amd_template *t,
                               const uint64_t *args, uint32_t *w) {
  memcpy(w, t->words, sizeof t->words);
  for (int i = 0; i < t->nholes; i++) {
    const struct rig_amd_hole *h = &t->holes[i];
    uint64_t v = args[h->arg];
    for (int j = 0; j < h->nops; j++) switch (h->op[j]) {
        case OP_ADD: v += h->k[j]; break;
        case OP_SHIFT: v >>= h->k[j]; break;
        default: v |= h->k[j]; break;
      }
    w[h->at] = (uint32_t)v;
    if (h->wide == 2) w[h->at + 1] = (uint32_t)(v >> 32);
  }
  return t->n;
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
  volatile uint64_t *write;     /* the write position the queue reads */
  volatile uint64_t *doorbell;
  int kind;                     /* RING_PM4, RING_AQL or RING_SDMA */
  uint64_t put, start, free;    /* placed, at the submission's start, done */
  uint64_t released;            /* the last value this ring released */
  struct rig_amd_marks marks;
};

struct rig_amd_segment {
  uint8_t *host;
  uint64_t gpu, size;
  uint64_t put, start, free;    /* bytes, as a ring's positions */
  struct rig_amd_marks marks;
};

/* A register whose store flushes a GPU's host data path, and the regions
   that need it: the device's BAR allocations, or its views of another
   device's. */
struct rig_amd_hdp {
  volatile uint32_t *reg;
  _Atomic int count;
};

struct rig_amd {
  const struct rig_driver *driver;
  _Atomic uint64_t *word;
  uint64_t word_gpu;
  volatile uint32_t *slots;     /* 64 bits apart, their low 32 bits used */
  uint64_t slots_gpu;
  uint64_t slot_last[RIG_AMD_SLOTS];
  struct rig_amd_ring rings[RIG_AMD_QUEUES];
  struct rig_amd_segment segment;
  struct rig_amd_template templates[RIG_AMD_TEMPLATES];
  uint64_t max_copy;            /* the bytes one S_COPY moves */
  struct rig_amd_hdp hdps[RIG_AMD_HDPS];
  uint64_t ib_at;               /* the PM4 words not yet in an AQL packet */
  size_t ib_n;
  int scratch_n;                /* the descriptor's pending writes */
  uint64_t scratch_at[RIG_AMD_SCRATCH_WRITES];
  uint32_t scratch_value[RIG_AMD_SCRATCH_WRITES];
  _Atomic int scratch_ready;
  _Atomic int scratch_lock;     /* held while the writes change or are read */
  _Atomic uint64_t scratch_taken;   /* the value that placed them */
  _Atomic uint64_t last;
  const char *failure;
  char failure_text[96];
};

/* A fill's writer: the device, the queue it places on, the words of the
   part's declared ring units it has not placed, and how far its declared
   segment bytes reach. The padding the writer places before an SDMA ring's
   end is not the fill's. */
struct rig_amd_writer {
  struct rig_amd *d;
  struct rig_amd_ring *q;
  uint64_t ring_left, segment_end;
};

int rig_amd_place(void *queue, const uint32_t *words, size_t n);
int rig_amd_segment(void *queue, size_t n, void **host, uint64_t *address);

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

#endif
