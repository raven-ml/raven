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
     if (e) return Val_int(e);
     if (!(e = nx_coalesce(3, a, &l))) run(&l);
     nx_done(3, a);
     return Val_int(e);

   The kernel answers Val_int of a code, an Nx_array.answer: NX_OK is Done,
   NX_DECLINED a case the kernel does not compute, and every other code a
   refusal, which Nx_array.refused raises.

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
/* The most operands of a loop: Nx_array.max_operands, which shape.ml sets
   to the same number. */
#define NX_MAX_OPERANDS 18

/* A layout's flags, computed when it is made. */
enum {
  NX_CONTIGUOUS = 1, /* Layout.is_contiguous */
  NX_DISTINCT = 2,   /* Layout.is_distinct */
  NX_EMPTY = 4       /* no element */
};

/* Codes: the constructors of Nx_array.answer, in its order. */

enum {
  NX_OK,           /* Done */
  NX_DECLINED,     /* the kernel does not compute the case */
  NX_DTYPE,        /* an operand's dtype is not the one named */
  NX_DEAD,         /* an operand's buffer is dead */
  NX_NOT_HOST,     /* the host does not address an operand's memory */
  NX_EXCLUSIVE,    /* an operand's memory is held exclusive */
  NX_READ_ONLY,    /* a written operand's memory is Read */
  NX_NOT_DISTINCT, /* a written operand reaches a position twice */
  NX_OVERLAP,      /* a written operand shares a byte with another written
                      operand, or a read one not identical to it */
  NX_SHAPE,        /* the operands' shapes do not fit the operation */
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

/* Layout.max_numel: a descriptor's extents, the product of its extents
   other than 0, and the end of its span are at most NX_MAX_NUMEL. A product
   of any of its extents, a position times a width of at most 128 bits, and
   a sum of a few of these fit in an int64_t, unchecked. */
#define NX_MAX_NUMEL ((int64_t)1 << 53)

/* An operand, read: its descriptor. [base] is NULL for an operand with no
   element. */
typedef struct {
  uint8_t *base; /* host address of the buffer's first byte */
  int dtype, bits, rank, flags;
  int64_t offset;               /* elements from base */
  int64_t dim[2 * NX_MAX_RANK]; /* rank extents, then rank strides */
  /* Read-only: a read operand identical to a written one. The kernel
     writes its bytes, so it may read it at an index only before it writes
     that index. */
  int alias;
  /* Private: the claimed buffer, a local root until nx_done, and whether
     nx_read waits for device work on it. */
  value buffer;
  int wait;
  struct caml__roots_block roots;
  struct caml__roots_block **local; /* the domain's local roots */
} nx_array;

/* The dtype of the OCaml array [v], a code. */
int nx_array_dtype(value v);

/* The layout of the array [v], an Nx_array.t, read without a claim: its
   rank, then at [dim] its [rank] extents and its [rank] strides, as
   nx_array's dim holds them, and at [offset] its first element, in
   elements. [dim] holds 2 · NX_MAX_RANK. It reads [v] alone and allocates
   nothing; what it writes is C memory, so it stays valid after [v]
   moves. */
int nx_array_layout(value v, int64_t *dim, int64_t *offset);

/* Reads the [n] operands [in] into [out] and answers NX_OK, or answers why
   it refuses one, claims nothing and leaves [out] unspecified. [n] is any
   number; with no operand it answers NX_OK. Past NX_MAX_OPERANDS it
   allocates, before any claim, and raises Out_of_memory if host memory
   runs out. It checks in this order and answers the first refusal: each
   operand's dtype; then each operand's claim of its memory, for writing if
   written: NX_DEAD if its buffer is dead, NX_EXCLUSIVE if the memory is
   held exclusive, NX_READ_ONLY if a written operand's buffer admits only
   Read; then, per operand, that a written one is NX_DISTINCT and that the
   host addresses one with an element; then that a written operand shares
   no byte with another written operand, nor with a read operand unless the
   two are identical: one width, and every index at one byte (equal extents
   and strides, one first bit). A read operand identical to a written one
   is claimed through it, once. Under the claims it waits for the device
   work each operand's access must follow, as Rig.Buffer.wait does; this
   alone runs OCaml code, and only while such work is unfinished. If the
   wait raises, as Rig.Lost does for a lost device, nx_read releases every
   claim and raises it. */
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

/* Coalesces in place the loop of [n] operands over [rank] axes, [rank] 0
   or more: their [extent]s and operand k's steps step[k], by axis. Drops
   the axes of extent 1 and merges an axis into the one before it where
   every operand's step there is this axis's step times its extent,
   keeping the C order. Answers the merged rank, at least 1: a loop with no
   element has rank 1, extent[0] = 0 and every step 0; one of one element,
   a rank-0 loop included, rank 1, extent 1 and every step 0. nx_coalesce's
   loop is this over its operands' shapes and strides. */
int nx_coalesce_dims(int n, int rank, int64_t *extent,
                     int64_t (*step)[NX_MAX_RANK]);

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

/* Copies the [len] elements of [bits] bits from positions [ps], [ps + ss],
   … of [s] to positions [pd], [pd + 1], … of [d]. The whole bytes of [d]
   come from memcpy when both runs start at one bit of a byte, from two
   bytes of [s] shifted when they start at different bits, and packed from
   loads otherwise. Out of line, so that the walks that call it stay
   small. */
void nx_sub_copy_run(uint8_t *d, int64_t pd, const uint8_t *s, int64_t ps,
                     int64_t ss, int64_t len, int bits);

/* Writes the low [bits] of [v] to every element: the bytes the run holds
   whole all at once, a byte of [v] repeated. */
static inline void nx_sub_fill_run(uint8_t *base, int bits, int64_t p,
                                   int64_t n, uint32_t v) {
  int per = 8 / bits;
  int64_t i = 0;
  for (; i < n && (p + i) % per != 0; i++) nx_sub_store(base, bits, p + i, v);
  uint8_t byte = bits == 1 ? (uint8_t)-(v & 1) : (uint8_t)((v & 15) * 0x11);
  uint8_t *b = base + (p + i) / per;
  int64_t whole = (n - i) / per;
  for (int64_t k = 0; k < whole; k++) b[k] = byte;
  for (i += whole * per; i < n; i++) nx_sub_store(base, bits, p + i, v);
}

#endif /* NX_ARRAY_H */
