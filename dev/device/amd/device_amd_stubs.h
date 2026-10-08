/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A device's state in C, which device_amd_ring.c writes the rings from and
   device_amd_stubs.c fills.

   The state is allocated at open and never freed: it holds the timeline
   word's address, which other devices may poll after the device is gone.
   Positions on a ring count 32-bit words placed since the ring was made and
   never wrap; a word's place is its position modulo the ring's size. */

#ifndef DEVICE_AMD_STUBS_H
#define DEVICE_AMD_STUBS_H

#include <stddef.h>
#include <stdint.h>

#include "device_amd.h"

/* The most parts and waits of a submission, the slot words (one per part
   and slot W, which the compute queue signals once it passed the waits),
   and the submissions a ring keeps the end of until their value is
   reached. */
#define DEVICE_AMD_PARTS 512
#define DEVICE_AMD_WAITS 255
#define DEVICE_AMD_SLOTS (DEVICE_AMD_PARTS + 1)
#define DEVICE_AMD_SLOT_W DEVICE_AMD_PARTS
#define DEVICE_AMD_MARKS 4096
#define DEVICE_AMD_HDPS 8

/* The queues, in Device_amd.queues's order. */
enum { DEVICE_AMD_COMPUTE, DEVICE_AMD_COPY, DEVICE_AMD_QUEUES };

/* The packets the writer places, which Device_amd encodes once per device.
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
  DEVICE_AMD_TEMPLATES
};

#define DEVICE_AMD_TEMPLATE_WORDS 16
#define DEVICE_AMD_TEMPLATE_HOLES 4
#define DEVICE_AMD_HOLE_OPS 3

/* A word of a template that holds a computation on an argument: the
   argument, then each operation in turn (add, shift right, or), written as
   one word or as two, low first. */
enum { OP_ADD, OP_SHIFT, OP_OR };

struct device_amd_hole {
  uint8_t at, wide, arg, nops;
  uint8_t op[DEVICE_AMD_HOLE_OPS];
  uint64_t k[DEVICE_AMD_HOLE_OPS];
};

struct device_amd_template {
  uint32_t words[DEVICE_AMD_TEMPLATE_WORDS];
  int n, nholes;
  struct device_amd_hole holes[DEVICE_AMD_TEMPLATE_HOLES];
};

/* A submission that used a ring or the segment: its value, and the
   position its words or bytes end at. */
struct device_amd_mark {
  uint64_t v, end;
};

/* Room handed out in order and taken back once values are reached. */
struct device_amd_marks {
  struct device_amd_mark at[DEVICE_AMD_MARKS];
  uint32_t head, count;
};

struct device_amd_ring {
  volatile uint32_t *words;
  uint64_t size;                /* words, a power of two */
  volatile uint64_t *write;     /* the write position the queue reads */
  volatile uint64_t *doorbell;
  int sdma;                     /* packets never wrap; positions in bytes */
  uint64_t put, start, free;    /* placed, at the submission's start, done */
  uint64_t released;            /* the last value this ring released */
  struct device_amd_marks marks;
};

struct device_amd_segment {
  uint8_t *host;
  uint64_t gpu, size;
  uint64_t put, start, free;    /* bytes, as a ring's positions */
  struct device_amd_marks marks;
};

/* A register whose store flushes a GPU's host data path, and the regions
   that need it: the device's BAR allocations, or its views of another
   device's. */
struct device_amd_hdp {
  volatile uint32_t *reg;
  _Atomic int count;
};

struct device_amd {
  _Atomic uint64_t *word;
  uint64_t word_gpu;
  volatile uint32_t *slots;     /* 64 bits apart, their low 32 bits used */
  uint64_t slots_gpu;
  uint64_t slot_last[DEVICE_AMD_SLOTS];
  struct device_amd_ring rings[DEVICE_AMD_QUEUES];
  struct device_amd_segment segment;
  struct device_amd_template templates[DEVICE_AMD_TEMPLATES];
  uint64_t max_copy;            /* the bytes one S_COPY moves */
  struct device_amd_hdp hdps[DEVICE_AMD_HDPS];
  uint64_t last;
  const char *failure;
  char failure_text[96];
};

/* A fill's writer: the device, the queue it places on, and how far the
   part's declared ring units and segment bytes reach. */
struct device_amd_writer {
  struct device_amd *d;
  struct device_amd_ring *q;
  uint64_t ring_end, segment_end;
};

int device_amd_place(void *queue, const uint32_t *words, size_t n);
int device_amd_segment(void *queue, size_t n, void **host, uint64_t *address);

/* Orders every store before it before every store after it, as the GPU
   sees them: on arm64 the full-system barrier stores to a BAR need; on
   x86_64 mfence. */
static inline void device_amd_barrier(void) {
#if defined(__aarch64__)
  __asm__ __volatile__("dsb sy" ::: "memory");
#elif defined(__x86_64__)
  __asm__ __volatile__("mfence" ::: "memory");
#else
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

#endif
