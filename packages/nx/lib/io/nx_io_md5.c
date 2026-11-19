/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  MD5 (RFC 1321) fed in pieces, from the OCaml runtime's implementation,
  for FLAC's checksum of its decoded samples (RFC 9639, 8.2). Stdlib's
  Digest hashes only a whole string, bytes or channel, so checking a
  decode through it would copy the signal into a string, and a decode in
  blocks could not check at all.

  This file calls caml_MD5Init, caml_MD5Update and caml_MD5Final, which
  caml/md5.h declares CAMLextern under CAML_INTERNALS. It is an exception
  to the Layers rule of docs/dev/design.md, as munin_blake2_stubs.c is for
  BLAKE2: Digest is the layer that abstracts MD5 and it is too weak here.
  The internals stay in this file, so the decoder is not compiled with
  them. A change to the runtime's context or functions fails this file's
  build at a compiler bump, never a digest.

  A context lives in a bytes value its owner never shares with a running
  decode: each call copies it into a C local, so the decoder may hash with
  the runtime lock released, and a FLAC block's digest stays the node's.
  ---------------------------------------------------------------------------*/

#define CAML_INTERNALS

#include <caml/alloc.h>
#include <caml/md5.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include <string.h>

#include "nx_io_md5.h"

_Static_assert(sizeof(struct MD5Context) == NX_IO_MD5_CONTEXT,
               "the runtime's MD5 context is not NX_IO_MD5_CONTEXT bytes");

void nx_io_md5_update(unsigned char *ctx, const unsigned char *p, size_t n) {
  struct MD5Context c;
  memcpy(&c, ctx, sizeof c);
  caml_MD5Update(&c, (unsigned char *)p, n);
  memcpy(ctx, &c, sizeof c);
}

/* [caml_nx_io_md5_init ()] is a fresh context, as bytes. */
CAMLprim value caml_nx_io_md5_init(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(ctx);
  struct MD5Context c;
  caml_MD5Init(&c);
  ctx = caml_alloc_string(sizeof c);
  memcpy(Bytes_val(ctx), &c, sizeof c);
  CAMLreturn(ctx);
}

/* [caml_nx_io_md5_final ctx] is the digest of the context [ctx], which it
   leaves as it was. */
CAMLprim value caml_nx_io_md5_final(value ctx) {
  CAMLparam1(ctx);
  CAMLlocal1(res);
  struct MD5Context c;
  unsigned char digest[16];
  memcpy(&c, Bytes_val(ctx), sizeof c);
  caml_MD5Final(digest, &c);
  res = caml_alloc_initialized_string(16, (const char *)digest);
  CAMLreturn(res);
}
