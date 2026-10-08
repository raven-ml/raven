/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Arrays from C.

   C reads arrays only through nx_read, which takes every operand of a call
   with the dtype the kernel's loads assume and whether it is written. It
   checks every operand, claims every operand's memory and fills its
   descriptor, or, if it refuses one, claims nothing and fills nothing. A
   kernel then calls nx_done on every path:

     nx_operand in[3] = { { vz, dt, 1 }, { vx, dt, 0 }, { vy, dt, 0 } };
     nx_array a[3];
     nx_loop l;
     int e = nx_read(3, in, a);
     if (e) return e;
     if (!(e = nx_coalesce(3, a, &l))) run(&l);
     nx_done(3, a);
     return e;

   A descriptor holds no pointer into the OCaml heap: it stays valid after the
   kernel allocates or releases the domain lock, and nx_read keeps each
   operand's buffer reachable until nx_done. nx_read and nx_done run with the
   domain lock held, on one thread. A kernel that may release the lock reads
   no OCaml value while it is released.

   A layout maps an index (i0, …, ik-1), 0 <= ij < dj, to the element
   position offset + Σ ij·sj, counted in elements from the buffer's first
   byte. An element at position p of a dtype of b bits occupies bits p·b to
   p·b + b - 1 of the buffer, LSB first within a byte.

   The coalescer turns operands of one shape into a loop: their extent-1
   axes dropped and the adjacent axes every operand lays out as one run
   merged. */

#ifndef NX_ARRAY_H
#define NX_ARRAY_H

#include <stdint.h>

#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_dtype.h"

#define NX_MAX_RANK 32
#define NX_MAX_OPERANDS 4

/* A layout's flags, computed when it is made. */
enum {
  NX_CONTIGUOUS = 1, /* element k in C order is at offset + k */
  NX_DISTINCT = 2,   /* no two indices reach one position */
  NX_EMPTY = 4       /* no element */
};

/* A layout, as the bytes of an OCaml Layout.t. A layout is in canonical
   form: an axis of extent 1 has stride 0, and a layout with no element has
   offset 0 and every stride 0. */
typedef struct {
  int64_t rank, flags;
  int64_t offset; /* elements */
  int64_t lo, hi; /* every position lies in [lo, hi); (0, 0) if none */
  int64_t dim[];  /* rank extents, then rank strides */
} nx_layout;

/* Codes */

enum {
  NX_OK,
  NX_DTYPE,        /* an operand's dtype is not the one named */
  NX_DEAD,         /* an operand's buffer is dead */
  NX_NOT_HOST,     /* the host does not address an operand's memory */
  NX_PENDING,      /* device work on an operand is unfinished: wait, retry */
  NX_EXCLUSIVE,    /* an operand's memory is held exclusive */
  NX_READ_ONLY,    /* a written operand's memory is Read */
  NX_NOT_DISTINCT, /* a written operand reaches a position twice */
  NX_OVERLAP,      /* a written operand shares a byte with another */
  NX_LAYOUT,       /* an operand's layout is not one */
  NX_SHAPE,        /* operands of one loop have different shapes */
  NX_ARITY         /* more operands than NX_MAX_OPERANDS */
};

/* The door */

/* An operand of a call: an OCaml array, the dtype the kernel's loads assume,
   and whether the kernel writes it. */
typedef struct {
  value array;
  int dtype;
  int written;
} nx_operand;

/* An operand, read. Its memory is claimed until nx_done; [base] is NULL for an
   operand with no element, which the coalescer forms no address from. */
typedef struct {
  uint8_t *base; /* host address of the buffer's first byte */
  int dtype, bits, rank, flags;
  int64_t offset;               /* elements from base */
  int64_t dim[2 * NX_MAX_RANK]; /* rank extents, then rank strides */
  /* Private: the claimed buffer, a local root until nx_done. */
  value buffer;
  struct caml__roots_block roots;
} nx_array;

/* The dtype of the OCaml array [v], a code. */
int nx_array_dtype(value v);

/* Reads the [n] operands [in] into [out] and answers NX_OK, or answers why
   it refuses one and fills nothing. Per operand it checks the dtype, the
   layout, that the buffer lives and the host addresses it; per written
   operand that it is NX_DISTINCT and shares no byte with any other operand.
   It then claims each operand's memory, for writing if written: NX_PENDING
   while earlier device work on it is unfinished (wait on the buffer with
   Rig.Buffer.wait and read again), NX_EXCLUSIVE if it is held exclusive,
   NX_READ_ONLY if a written operand's memory is Read. */
int nx_read(int n, const nx_operand *in, nx_array *out);

/* Releases the claims of the [n] operands a successful nx_read filled. */
void nx_done(int n, nx_array *a);

/* Coalescing */

/* A loop over operands of one shape: [rank] axes, at least one, of
   [extent]s; operand k's element at a loop index is at first[k] + Σ
   index·step[k], in elements from its base. A loop with no element has
   extent[0] = 0; one of one element has rank 1 and extent 1. */
typedef struct {
  int rank;
  int64_t extent[NX_MAX_RANK];
  int64_t step[NX_MAX_OPERANDS][NX_MAX_RANK]; /* elements */
  int64_t first[NX_MAX_OPERANDS];             /* elements from base */
} nx_loop;

/* Fills [l] for the [n] operands [a] and answers NX_OK, or answers
   NX_ARITY if [n] is not in [1, NX_MAX_OPERANDS], and NX_SHAPE if their
   shapes differ. */
int nx_coalesce(int n, const nx_array *a, nx_loop *l);

/* Sub-byte elements

   Element p of a dtype of [bits] bits (1 or 4) is bits p·bits to p·bits +
   bits - 1 of the bytes from [base], LSB first. */

static inline uint32_t nx_sub_load(const uint8_t *base, int bits, int64_t p) {
  uint64_t bit = (uint64_t)p * (uint64_t)bits;
  return (base[bit >> 3] >> (bit & 7)) & ((1u << bits) - 1);
}

/* Stores the low [bits] of [v] at element [p] with one atomic
   read-modify-write of its byte: writes to the byte's other elements, from
   other threads too, are kept, and no reader sees a value of the element
   that no store wrote. */
static inline void nx_sub_store(uint8_t *base, int bits, int64_t p,
                                uint32_t v) {
  uint64_t bit = (uint64_t)p * (uint64_t)bits;
  uint8_t mask = (uint8_t)(((1u << bits) - 1) << (bit & 7));
  uint8_t set = (uint8_t)((v << (bit & 7)) & mask);
  uint8_t *byte = base + (bit >> 3);
  uint8_t old = __atomic_load_n(byte, __ATOMIC_RELAXED);
  while (!__atomic_compare_exchange_n(byte, &old, (uint8_t)((old & ~mask) | set),
                                      1, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
    ;
}

#endif /* NX_ARRAY_H */
