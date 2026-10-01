/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  OCaml entry points. Positions and lengths come from [Compress_deflate],
  which checks them. Entry points over bytes never release the runtime;
  entry points over bigarrays release it for large spans and touch only
  bigarray and C memory while it is released.
  ---------------------------------------------------------------------------*/

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#include <stdlib.h>

#include "compress_deflate.h"

#define RELEASE_THRESHOLD 65536

static uint8_t *bigbytes(value b) { return (uint8_t *)Caml_ba_data_val(b); }
static size_t bigbytes_length(value b) { return Caml_ba_array_val(b)->dim[0]; }

CAMLprim value caml_compress_deflate_init(value unit) {
  (void)unit;
  compress_inflate_init();
  compress_crc32_init();
  return Val_unit;
}

CAMLprim value caml_compress_deflate_overlap(value a, value b) {
  uintptr_t pa = (uintptr_t)bigbytes(a), pb = (uintptr_t)bigbytes(b);
  size_t la = bigbytes_length(a), lb = bigbytes_length(b);
  return Val_bool(la != 0 && lb != 0 && pa < pb + lb && pb < pa + la);
}

/* Checksums */

CAMLprim value caml_compress_deflate_crc32_bytes(value crc, value b, value off,
                                                 value len) {
  return Val_long(compress_deflate_crc32(
      (uint32_t)Long_val(crc), Bytes_val(b) + Long_val(off), Long_val(len)));
}

CAMLprim value caml_compress_deflate_adler32_bytes(value adler, value b,
                                                   value off, value len) {
  return Val_long(compress_deflate_adler32(
      (uint32_t)Long_val(adler), Bytes_val(b) + Long_val(off), Long_val(len)));
}

CAMLprim value caml_compress_deflate_crc32_bigbytes(value vcrc, value b,
                                                    value off, value len) {
  CAMLparam1(b);
  uint32_t crc = (uint32_t)Long_val(vcrc);
  const uint8_t *p = bigbytes(b) + Long_val(off);
  size_t n = Long_val(len);
  if (n > RELEASE_THRESHOLD) {
    caml_release_runtime_system();
    crc = compress_deflate_crc32(crc, p, n);
    caml_acquire_runtime_system();
  } else {
    crc = compress_deflate_crc32(crc, p, n);
  }
  CAMLreturn(Val_long(crc));
}

CAMLprim value caml_compress_deflate_adler32_bigbytes(value vadler, value b,
                                                      value off, value len) {
  CAMLparam1(b);
  uint32_t adler = (uint32_t)Long_val(vadler);
  const uint8_t *p = bigbytes(b) + Long_val(off);
  size_t n = Long_val(len);
  if (n > RELEASE_THRESHOLD) {
    caml_release_runtime_system();
    adler = compress_deflate_adler32(adler, p, n);
    caml_acquire_runtime_system();
  } else {
    adler = compress_deflate_adler32(adler, p, n);
  }
  CAMLreturn(Val_long(adler));
}

/* Inflaters. The positions travel in an int array:
   [| src_pos; src_end; dst_hist; dst_pos; dst_end |]. */

#define Inflate_val(v) (*((compress_inflate **)Data_custom_val(v)))

static void inflate_finalize(value v) { free(Inflate_val(v)); }

static struct custom_operations inflate_ops = {
    "compress.deflate.inflate", inflate_finalize,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

CAMLprim value caml_compress_inflate_create(value unit) {
  (void)unit;
  compress_inflate *s = malloc(sizeof(*s));
  if (s == NULL)
    caml_raise_out_of_memory();
  compress_inflate_reset(s);
  value v = caml_alloc_custom_mem(&inflate_ops, sizeof(s), sizeof(*s));
  Inflate_val(v) = s;
  return v;
}

CAMLprim value caml_compress_inflate_reset(value v) {
  compress_inflate_reset(Inflate_val(v));
  return Val_unit;
}

static compress_inflate_io io_of(value io, const uint8_t *src, uint8_t *dst,
                                 value final) {
  compress_inflate_io x = {src,
                           Long_val(Field(io, 0)),
                           Long_val(Field(io, 1)),
                           Bool_val(final),
                           dst,
                           Long_val(Field(io, 2)),
                           Long_val(Field(io, 3)),
                           Long_val(Field(io, 4))};
  return x;
}

/* The fields hold immediates: writing them needs no barrier. */
static void io_update(value io, const compress_inflate_io *x) {
  Field(io, 0) = Val_long(x->src_pos);
  Field(io, 3) = Val_long(x->dst_pos);
}

CAMLprim value caml_compress_inflate_bytes(value v, value src, value dst,
                                           value io, value final) {
  compress_inflate_io x = io_of(io, Bytes_val(src), Bytes_val(dst), final);
  int status = compress_inflate_run(Inflate_val(v), &x);
  io_update(io, &x);
  return Val_int(status);
}

CAMLprim value caml_compress_inflate_bigbytes(value v, value src, value dst,
                                              value io, value final) {
  CAMLparam5(v, src, dst, io, final);
  compress_inflate_io x = io_of(io, bigbytes(src), bigbytes(dst), final);
  compress_inflate *s = Inflate_val(v);
  int status;
  if ((x.src_end - x.src_pos) + (x.dst_end - x.dst_pos) > RELEASE_THRESHOLD) {
    caml_release_runtime_system();
    status = compress_inflate_run(s, &x);
    caml_acquire_runtime_system();
  } else {
    status = compress_inflate_run(s, &x);
  }
  io_update(io, &x);
  CAMLreturn(Val_int(status));
}

/* Deflaters */

#define Deflate_val(v) (*((compress_deflate **)Data_custom_val(v)))

static void deflate_finalize(value v) { compress_deflate_free(Deflate_val(v)); }

static struct custom_operations deflate_ops = {
    "compress.deflate.deflate", deflate_finalize,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

CAMLprim value caml_compress_deflate_out_max(value unit) {
  (void)unit;
  return Val_long(COMPRESS_DEFLATE_OUT_MAX);
}

CAMLprim value caml_compress_deflate_create(value level) {
  compress_deflate *e = compress_deflate_create(Int_val(level));
  if (e == NULL)
    caml_raise_out_of_memory();
  value v = caml_alloc_custom_mem(&deflate_ops, sizeof(e), 1 << 20);
  Deflate_val(v) = e;
  return v;
}

CAMLprim value caml_compress_deflate_free(value v) {
  compress_deflate_free(Deflate_val(v));
  Deflate_val(v) = NULL;
  return Val_unit;
}

CAMLprim value caml_compress_deflate_input(value v, value src, value off,
                                           value len) {
  return Val_long(compress_deflate_input(
      Deflate_val(v), Bytes_val(src) + Long_val(off), Long_val(len)));
}

CAMLprim value caml_compress_deflate_encode(value v, value out, value eod) {
  return Val_long(
      compress_deflate_encode(Deflate_val(v), Bytes_val(out), Bool_val(eod)));
}
