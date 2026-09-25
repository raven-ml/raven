/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#ifndef NX_BUFFER_STUBS_H
#define NX_BUFFER_STUBS_H

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <stdbool.h>
#include <stdint.h>

#include "nx_dtype.h"

/* Additional types not in standard bigarray, following stdlib naming convention */
typedef uint16_t caml_ba_bfloat16;   /* BFloat16 */
typedef uint8_t caml_ba_fp8_e4m3;    /* 8-bit float: 1 sign, 4 exponent, 3 mantissa */
typedef uint8_t caml_ba_fp8_e5m2;    /* 8-bit float: 1 sign, 5 exponent, 2 mantissa */
typedef uint8_t caml_ba_bool;        /* Bool as byte (0/1) */
/* Note: int4/uint4 pack 2 values per byte — no single-element typedef */
typedef uint32_t caml_ba_uint32;     /* Unsigned 32-bit */
typedef uint64_t caml_ba_uint64;     /* Unsigned 64-bit */

/* Extended kind enumeration that continues from OCaml's bigarray kinds */
enum nx_ba_extended_kind {
  NX_BA_BFLOAT16 = CAML_BA_FIRST_UNIMPLEMENTED_KIND,
  NX_BA_BOOL,
  NX_BA_INT4,
  NX_BA_UINT4,
  NX_BA_FP8_E4M3,
  NX_BA_FP8_E5M2,
  NX_BA_UINT32,
  NX_BA_UINT64,
  NX_BA_LAST_KIND
};

#define NX_BA_EXTENDED_KIND_SHIFT 16
#define NX_BA_EXTENDED_KIND_FIELD(kind) \
  ((int)((kind) << NX_BA_EXTENDED_KIND_SHIFT))
#define NX_BA_EXTENDED_KIND_MASK NX_BA_EXTENDED_KIND_FIELD(0xFF)

static inline bool nx_buffer_is_extended_kind(int kind) {
  return kind >= NX_BA_BFLOAT16 && kind < NX_BA_LAST_KIND;
}

static inline int nx_buffer_get_stored_extended_kind(int flags) {
  return (flags & NX_BA_EXTENDED_KIND_MASK) >> NX_BA_EXTENDED_KIND_SHIFT;
}

static inline int nx_buffer_store_extended_kind(int flags, int kind) {
  flags &= ~NX_BA_EXTENDED_KIND_MASK;
  if (nx_buffer_is_extended_kind(kind))
    flags |= NX_BA_EXTENDED_KIND_FIELD(kind);
  return flags;
}

static inline int nx_buffer_get_kind_from_flags(int flags) {
  int stored = nx_buffer_get_stored_extended_kind(flags);
  if (stored != 0) return stored;
  return flags & CAML_BA_KIND_MASK;
}

static inline int nx_buffer_get_kind(const struct caml_ba_array *b) {
  return nx_buffer_get_kind_from_flags(b->flags);
}

#endif /* NX_BUFFER_STUBS_H */
