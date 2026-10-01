/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  XXH64, the content checksum of frames
  (https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md), and the
  OCaml entry points. [Compress_zstd] checks positions before it calls them.
  ---------------------------------------------------------------------------*/

#define CAML_NAME_SPACE
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#include <stdlib.h>
#include <string.h>

#include "compress_zstd.h"

#define P1 0x9E3779B185EBCA87ull
#define P2 0xC2B2AE3D27D4EB4Full
#define P3 0x165667B19E3779F9ull
#define P4 0x85EBCA77C2B2AE63ull
#define P5 0x27D4EB2F165667C5ull

static inline uint64_t rotl(uint64_t x, unsigned r) {
  return (x << r) | (x >> (64 - r));
}

static inline uint64_t le64(const uint8_t *p) {
  uint64_t v = 0;
  for (unsigned i = 0; i < 8; i++)
    v |= (uint64_t)p[i] << (8 * i);
  return v;
}

static inline uint64_t round64(uint64_t acc, uint64_t lane) {
  return rotl(acc + lane * P2, 31) * P1;
}

static inline uint64_t merge(uint64_t acc, uint64_t v) {
  return (acc ^ round64(0, v)) * P1 + P4;
}

uint64_t compress_zstd_xxh64(const uint8_t *p, size_t len) {
  const uint8_t *end = p + len;
  uint64_t h;
  if (len >= 32) {
    uint64_t v1 = P1 + P2, v2 = P2, v3 = 0, v4 = -P1;
    for (; end - p >= 32; p += 32) {
      v1 = round64(v1, le64(p));
      v2 = round64(v2, le64(p + 8));
      v3 = round64(v3, le64(p + 16));
      v4 = round64(v4, le64(p + 24));
    }
    h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
    h = merge(h, v1);
    h = merge(h, v2);
    h = merge(h, v3);
    h = merge(h, v4);
  } else {
    h = P5;
  }
  h += len;
  for (; end - p >= 8; p += 8)
    h = rotl(h ^ round64(0, le64(p)), 27) * P1 + P4;
  if (end - p >= 4) {
    uint64_t v = (uint64_t)p[0] | ((uint64_t)p[1] << 8) |
                 ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24);
    h = rotl(h ^ (v * P1), 23) * P2 + P3;
    p += 4;
  }
  for (; p < end; p++)
    h = rotl(h ^ (*p * P5), 11) * P1;
  h ^= h >> 33;
  h *= P2;
  h ^= h >> 29;
  h *= P3;
  h ^= h >> 32;
  return h;
}

#define RELEASE_THRESHOLD 65536

#define Zstd_val(v) (*((compress_zstd **)Data_custom_val(v)))

static void finalize(value v) { free(Zstd_val(v)); }

static struct custom_operations ops = {
    "compress.zstd",          finalize,
    custom_compare_default,   custom_hash_default,
    custom_serialize_default, custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

CAMLprim value caml_compress_zstd_create(value unit) {
  (void)unit;
  compress_zstd *z = malloc(sizeof(*z));
  if (z == NULL)
    caml_raise_out_of_memory();
  compress_zstd_reset(z);
  value v = caml_alloc_custom_mem(&ops, sizeof(z), sizeof(*z));
  Zstd_val(v) = z;
  return v;
}

CAMLprim value caml_compress_zstd_reset(value v) {
  compress_zstd_reset(Zstd_val(v));
  return Val_unit;
}

CAMLprim value caml_compress_zstd_overlap(value a, value b) {
  uintptr_t pa = (uintptr_t)Caml_ba_data_val(a);
  uintptr_t pb = (uintptr_t)Caml_ba_data_val(b);
  size_t la = Caml_ba_array_val(a)->dim[0], lb = Caml_ba_array_val(b)->dim[0];
  return Val_bool(la != 0 && lb != 0 && pa < pb + lb && pb < pa + la);
}

/* [io] is [| src_pos; src_end; dst_hist; dst_pos; dst_end |]; the block
   advances [dst_pos]. */
CAMLprim value caml_compress_zstd_block(value vz, value vsrc, value vdst,
                                        value vio) {
  CAMLparam4(vz, vsrc, vdst, vio);
  compress_zstd *z = Zstd_val(vz);
  const uint8_t *src = Caml_ba_data_val(vsrc);
  uint8_t *dst = Caml_ba_data_val(vdst);
  size_t pos = Long_val(Field(vio, 0)), end = Long_val(Field(vio, 1));
  size_t hist = Long_val(Field(vio, 2)), out = Long_val(Field(vio, 3));
  size_t dst_end = Long_val(Field(vio, 4));
  int status;
  if (end - pos > RELEASE_THRESHOLD / 4) {
    caml_release_runtime_system();
    status = compress_zstd_block(z, src, pos, end, dst, hist, &out, dst_end);
    caml_acquire_runtime_system();
  } else {
    status = compress_zstd_block(z, src, pos, end, dst, hist, &out, dst_end);
  }
  Field(vio, 3) = Val_long(out);
  CAMLreturn(Val_int(status));
}

CAMLprim value caml_compress_zstd_xxh64(value vb, value voff, value vlen) {
  CAMLparam1(vb);
  const uint8_t *p = (const uint8_t *)Caml_ba_data_val(vb) + Long_val(voff);
  size_t len = Long_val(vlen);
  uint64_t h;
  if (len > RELEASE_THRESHOLD) {
    caml_release_runtime_system();
    h = compress_zstd_xxh64(p, len);
    caml_acquire_runtime_system();
  } else {
    h = compress_zstd_xxh64(p, len);
  }
  CAMLreturn(Val_long(h & 0xFFFFFFFFu));
}
