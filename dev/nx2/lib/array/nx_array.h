/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Arrays from C.

   A host kernel reads arrays only through the door: nx_read checks every
   operand of a call and claims every operand's memory, or refuses and claims
   none, then waits under the claims for the device work on them; the kernel
   runs over the descriptors nx_read filled, and nx_done ends the claims:

     nx_operand in[3] = { { vz, dt, 1 }, { vx, dt, 0 }, { vy, dt, 0 } };
     nx_array a[3];
     nx_loop l;
     int e = nx_read(3, in, a);
     if (e) return e;
     if (!(e = nx_coalesce(3, a, &l))) run(&l);
     nx_done(3, a);
     return e;

   The kernel answers its code to its OCaml wrapper, which hands any code but
   NX_OK to Nx_array.refused, which raises: every code but NX_OK is a
   refusal.

   A descriptor's element at index (i0, …, ik-1), 0 <= ij < dim[j], lies at
   position offset + Σ ij·dim[rank + j], counted in elements from base, as
   Layout places it (layout.mli). */

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
  NX_CONTIGUOUS = 1, /* Layout.is_contiguous */
  NX_DISTINCT = 2,   /* Layout.is_distinct */
  NX_EMPTY = 4       /* no element */
};

/* Codes */

enum {
  NX_OK,
  NX_DTYPE,        /* an operand's dtype is not the one named */
  NX_DEAD,         /* an operand's buffer is dead */
  NX_NOT_HOST,     /* the host does not address an operand's memory */
  NX_EXCLUSIVE,    /* an operand's memory is held exclusive */
  NX_READ_ONLY,    /* a written operand's memory is Read */
  NX_NOT_DISTINCT, /* a written operand reaches a position twice */
  NX_OVERLAP,      /* a written operand shares a byte with another */
  NX_LAYOUT,       /* an operand's layout is not one */
  NX_SHAPE,        /* operands of one loop have different shapes */
  NX_ARITY         /* no operand, or more than NX_MAX_OPERANDS, to a loop */
};

/* The door

   From nx_read to nx_done, each operand's memory is claimed and its buffer
   is a local root of the domain: a kernel needs no CAMLparam to keep its
   operands reachable. A descriptor holds no pointer into the OCaml heap, so
   it stays valid when the kernel allocates or releases the domain lock. In
   return the kernel keeps these rules:

   - nx_read may wait for device work, which runs OCaml code: the collector,
     signal handlers, other threads, the device's driver. A kernel that calls
     it is an external that is not [@@noalloc], and keeps as a root
     registered with CAMLparam every OCaml value other than its operands that
     it uses after nx_read. nx_read raises what Rig.Buffer.wait raises,
     holding no claim: the kernel acquires nothing before it that must be
     undone.
   - nx_read and nx_done run on one thread, with the domain lock held.
   - Descriptors stay where nx_read filled them, in the frame of the C
     function that called it, and are passed by pointer, never copied or
     moved, until nx_done.
   - nx_done runs once per successful nx_read, never after a refusal or a
     raise, before that function returns or raises.
   - nx_read and nx_done nest with root frames as CAMLparam and CAMLreturn
     do: a frame registered before nx_read is popped after nx_done, and one
     registered after nx_read is popped before nx_done. A kernel registers
     its roots with CAMLparam before it calls nx_read. nx_done ends the
     process if it finds the descriptors' roots popped.
   - While the domain lock is released, the kernel reads no OCaml value. */

/* An operand of a call: an OCaml array, the dtype the kernel's loads assume,
   and whether the kernel writes it. nx_read reads [array] while it runs;
   the descriptor it fills is what the kernel uses after. */
typedef struct {
  value array;
  int dtype;
  int written;
} nx_operand;

/* An operand, read: its descriptor. [base] is NULL for an operand with no
   element. */
typedef struct {
  uint8_t *base; /* host address of the buffer's first byte */
  int dtype, bits, rank, flags;
  int64_t offset;               /* elements from base */
  int64_t dim[2 * NX_MAX_RANK]; /* rank extents, then rank strides */
  /* Private: the claimed buffer, a local root until nx_done, and whether
     nx_read waits for device work on it. */
  value buffer;
  int wait;
  struct caml__roots_block roots;
} nx_array;

/* The dtype of the OCaml array [v], a code. */
int nx_array_dtype(value v);

/* Reads the [n] operands [in] into [out] and answers NX_OK, or answers why
   it refuses one, claims nothing and leaves [out] unspecified. Per operand
   it checks the dtype, the layout, that the buffer lives and, unless the
   operand has no element, that the host addresses it; per written operand,
   that it is NX_DISTINCT and shares no byte with another operand. It then
   claims each operand's memory, for writing if written: NX_EXCLUSIVE if the
   memory is held exclusive, NX_READ_ONLY if a written operand's memory is
   Read. Under the claims it waits for the device work each operand's access
   must follow, as Rig.Buffer.wait does; this alone runs OCaml code, and only
   while such work is unfinished. If the wait raises, as Rig.Lost does for a
   lost device, nx_read releases every claim and raises it. With no operand
   it answers NX_OK. */
int nx_read(int n, const nx_operand *in, nx_array *out);

/* Releases the claims of the [n] operands a successful nx_read filled and
   unlinks their roots. */
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
   shapes differ. The loop keeps the C order of indices: axes are dropped
   and merged, never reordered. */
int nx_coalesce(int n, const nx_array *a, nx_loop *l);

/* Block copies */

/* Copies [rows] rows of [cols] elements of [bits] bits, bits for bits:
   element j of row i from position [ps + i·src_row + j·src_col] of [src]
   to position [pd + i·dst_row + j·dst_col] of [dst]. Positions are at
   least 0 and steps of either sign, both counted in elements. The
   destination's elements are distinct and share no byte with the
   source's. Bytes that hold only copied elements take plain stores; a
   byte shared with other elements takes a compare-and-swap, as
   nx_sub_store does. Rows of adjacent elements are memcpy; where the
   source steps one element across rows and the destination one along
   them, as for a transposed source, it moves 4x4 blocks, four contiguous
   loads and four contiguous stores each. */
void nx_copy_block(uint8_t *dst, int64_t pd, int64_t dst_row, int64_t dst_col,
                   const uint8_t *src, int64_t ps, int64_t src_row,
                   int64_t src_col, int64_t rows, int64_t cols, int bits);

/* Sub-byte elements

   Element p of a dtype of [bits] bits (1 or 4) is bits p·bits to p·bits +
   bits - 1 of the bytes from [base], LSB first. A store is one
   compare-and-swap of its byte and a load one relaxed atomic load of it, so
   threads may load and store elements of one byte at once: no store loses
   another element's bits, and no load sees an element no store wrote. */

static inline uint32_t nx_sub_load(const uint8_t *base, int bits, int64_t p) {
  uint64_t bit = (uint64_t)p * (uint64_t)bits;
  uint8_t byte = __atomic_load_n(base + (bit >> 3), __ATOMIC_RELAXED);
  return (byte >> (bit & 7)) & ((1u << bits) - 1);
}

/* Stores the bits of [set] under [mask] into [*byte], keeping its other
   bits, with one compare-and-swap. */
static inline void nx_sub_put(uint8_t *byte, uint8_t mask, uint8_t set) {
  uint8_t old = __atomic_load_n(byte, __ATOMIC_RELAXED);
  set &= mask;
  while (!__atomic_compare_exchange_n(byte, &old,
                                      (uint8_t)((old & ~mask) | set), 1,
                                      __ATOMIC_RELAXED, __ATOMIC_RELAXED))
    ;
}

/* Stores the low [bits] of [v] at element [p]. */
static inline void nx_sub_store(uint8_t *base, int bits, int64_t p,
                                uint32_t v) {
  uint64_t bit = (uint64_t)p * (uint64_t)bits;
  uint8_t mask = (uint8_t)(((1u << bits) - 1) << (bit & 7));
  nx_sub_put(base + (bit >> 3), mask, (uint8_t)((v << (bit & 7)) & mask));
}

/* Runs of sub-byte elements

   The [n] elements from element [p], one per byte; [p] is at least 0, and
   a run of no element touches no byte. A run's whole bytes hold only its
   elements and are plain loads and stores; a byte it shares with other
   elements, at either end, is one atomic load or one compare-and-swap. */

/* Reads the elements of [dt], a sub-byte dtype, into [dst]: int4's
   sign-extended, the others' zero-extended. Both runs are always inlined,
   so that a caller that passes a constant [dt] or [bits] gets a loop over
   whole bytes the compiler vectorises. */
static inline __attribute__((always_inline)) void nx_sub_unpack_run(
    const uint8_t *base, int dt, int64_t p, uint8_t *dst, int64_t n) {
  int bits = dt == NX_BIT ? 1 : 4, per = 8 / bits;
  uint32_t mask = (1u << bits) - 1, half = dt == NX_INT4 ? 8 : 0;
  int64_t i = 0;
#define NX_EXTEND(v) ((uint8_t)((((v) & mask) ^ half) - half))
  for (; i < n && (p + i) % per != 0; i++)
    dst[i] = NX_EXTEND(nx_sub_load(base, bits, p + i));
  const uint8_t *b = base + (p + i) / per;
  int64_t whole = (n - i) / per;
  if (bits == 4)
    for (int64_t k = 0; k < whole; k++, i += 2) {
      dst[i] = NX_EXTEND(b[k]);
      dst[i + 1] = NX_EXTEND(b[k] >> 4);
    }
  else
    for (int64_t k = 0; k < whole; k++, i += 8)
      for (int j = 0; j < 8; j++) dst[i + j] = NX_EXTEND(b[k] >> j);
  for (; i < n; i++) dst[i] = NX_EXTEND(nx_sub_load(base, bits, p + i));
#undef NX_EXTEND
}

/* Writes the low [bits] of each byte of [src] to the elements. */
static inline __attribute__((always_inline)) void nx_sub_pack_run(
    uint8_t *base, int bits, int64_t p, const uint8_t *src, int64_t n) {
  if (n <= 0) return;
  int per = 8 / bits;
  uint32_t mask = (1u << bits) - 1;
  int64_t i = 0;
  /* The elements of the first byte, if the run starts inside it. */
  if (p % per != 0) {
    int at = (int)(p % per) * bits;
    uint8_t m = 0, set = 0;
    for (; i < n && (p + i) % per != 0; i++, at += bits) {
      m |= (uint8_t)(mask << at);
      set |= (uint8_t)((src[i] & mask) << at);
    }
    nx_sub_put(base + p / per, m, set);
  }
  uint8_t *b = base + (p + i) / per;
  int64_t whole = (n - i) / per;
  if (bits == 4)
    for (int64_t k = 0; k < whole; k++, i += 2)
      b[k] = (uint8_t)((src[i] & 15) | (src[i + 1] << 4));
  else
    for (int64_t k = 0; k < whole; k++, i += 8) {
      uint8_t x = 0;
      for (int j = 0; j < 8; j++) x |= (uint8_t)((src[i + j] & 1) << j);
      b[k] = x;
    }
  /* The elements of the last byte, if the run ends inside it. */
  if (i < n) {
    uint8_t m = 0, set = 0;
    for (int at = 0; i < n; i++, at += bits) {
      m |= (uint8_t)(mask << at);
      set |= (uint8_t)((src[i] & mask) << at);
    }
    nx_sub_put(b + whole, m, set);
  }
}

#endif /* NX_ARRAY_H */
