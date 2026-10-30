/* The Zstandard decoder's paths this host can run (lib/zstd_dec_paths.h),
   for test_zstd: Lz reaches only the last, and the tests hold each path,
   the generic one included, to the same output and errors. */

#include <stdint.h>
#include <string.h>

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "zstd_dec_paths.h"

CAMLprim value lz_support_zstd_paths(value unit) {
  CAMLparam1(unit);
  CAMLlocal2(names, name);
  const char *const *p;
  size_t n = lz_zstd_dec_paths(&p);
  names = caml_alloc(n, 0);
  for (size_t i = 0; i < n; i++) {
    name = caml_copy_string(p[i]);
    Store_field(names, i, name);
  }
  CAMLreturn(names);
}

/* An empty bigarray may have no data; the kernel takes this byte. */
static uint8_t none;

/* Path [i]'s whole-buffer decode of [src] into [dst], frame after frame to
   the input's end as stubs.c's driver runs it: [(kind, at, produced,
   message)], [kind] -1 and [at] the bytes consumed when it decoded, else
   an lz.h error kind. */
CAMLprim value lz_support_zstd_decode(value i, value src, value dst) {
  CAMLparam3(i, src, dst);
  CAMLlocal2(r, msg);
  const char *const *names;
  if ((size_t)Long_val(i) >= lz_zstd_dec_paths(&names))
    caml_invalid_argument("Lz_support: no such path");
  size_t src_len = (size_t)Caml_ba_array_val(src)->dim[0];
  size_t dst_cap = (size_t)Caml_ba_array_val(dst)->dim[0];
  const uint8_t *s = src_len ? (const uint8_t *)Caml_ba_data_val(src) : &none;
  uint8_t *d = dst_cap ? (uint8_t *)Caml_ba_data_val(dst) : &none;
  struct lz_zstd_dec st;
  lz_zstd_dec_init(&st, UINT64_MAX, false);
  lz_zstd_dec_path(&st, (size_t)Long_val(i));
  size_t consumed = 0, produced = 0;
  struct lz_error err = {-1, 0, NULL};
  for (;;) {
    struct lz_io m = {.src = s + consumed,
                      .src_len = src_len - consumed,
                      .dst = d + produced,
                      .dst_cap = dst_cap - produced};
    enum lz_status k = lz_zstd_decode(&st, &m, true, &err);
    if (k == LZ_ERROR) err.at += (int64_t)consumed;
    consumed += m.consumed;
    produced += m.produced;
    if (k == LZ_NEED_OUTPUT) lz_fail(&err, LZ_TOO_LONG, (int64_t)consumed, NULL);
    if (k == LZ_NEED_INPUT) lz_fail(&err, LZ_TRUNCATED, (int64_t)src_len, NULL);
    if (k != LZ_DONE || consumed == src_len || m.consumed == 0) break;
  }
  lz_zstd_dec_free(&st);
  msg = caml_copy_string(err.msg ? err.msg : "");
  r = caml_alloc_tuple(4);
  Store_field(r, 0, Val_int(err.kind));
  Store_field(r, 1, Val_long(err.kind < 0 ? (long)consumed : (long)err.at));
  Store_field(r, 2, Val_long((long)produced));
  Store_field(r, 3, msg);
  CAMLreturn(r);
}

/* The Zstandard encoder's paths (lib/zstd_enc.h), which must all write the
   same bytes: their names, and path [i]'s whole-buffer encode of [src]
   into [dst] at [level], the encoded length or -1 when it does not fit. */

#include "zstd_enc.h"

CAMLprim value lz_support_zstd_enc_paths(value unit) {
  CAMLparam1(unit);
  CAMLlocal2(names, name);
  const char *const *p;
  size_t n = lz_zstd_enc_paths(&p);
  names = caml_alloc(n, 0);
  for (size_t i = 0; i < n; i++) {
    name = caml_copy_string(p[i]);
    Store_field(names, i, name);
  }
  CAMLreturn(names);
}

CAMLprim value lz_support_zstd_encode(value i, value level, value src,
                                      value dst) {
  const char *const *names;
  if ((size_t)Long_val(i) >= lz_zstd_enc_paths(&names))
    caml_invalid_argument("Lz_support: no such path");
  size_t src_len = (size_t)Caml_ba_array_val(src)->dim[0];
  size_t dst_cap = (size_t)Caml_ba_array_val(dst)->dim[0];
  struct lz_io io = {
      .src = src_len ? (const uint8_t *)Caml_ba_data_val(src) : &none,
      .src_len = src_len,
      .dst = dst_cap ? (uint8_t *)Caml_ba_data_val(dst) : &none,
      .dst_cap = dst_cap};
  struct lz_zstd_enc st;
  struct lz_error err;
  lz_zstd_enc_init(&st, Int_val(level), false);
  lz_zstd_enc_path(&st, (size_t)Long_val(i));
  enum lz_status k = lz_zstd_encode(&st, &io, true, &err);
  lz_zstd_enc_free(&st);
  if (k == LZ_ERROR) caml_raise_out_of_memory();
  return Val_long(k == LZ_DONE ? (long)io.produced : -1);
}

/* [lz_support_zstd_encode_limit limit step level src dst] is the frame of
   [src] at [level] by an encoder whose index limit is [limit]
   (lib/zstd_enc.h), written to [dst]: by the whole-buffer encode when
   [step] is 0, else by a stream given at most [step] bytes a step, as
   stubs.c's driver steps it. It is the frame's length, or -1 when it does
   not fit. */
CAMLprim value lz_support_zstd_encode_limit(value limit, value step,
                                            value level, value src,
                                            value dst) {
  size_t src_len = (size_t)Caml_ba_array_val(src)->dim[0];
  size_t dst_cap = (size_t)Caml_ba_array_val(dst)->dim[0];
  const uint8_t *s = src_len ? (const uint8_t *)Caml_ba_data_val(src) : &none;
  uint8_t *d = dst_cap ? (uint8_t *)Caml_ba_data_val(dst) : &none;
  size_t cut = (size_t)Long_val(step);
  struct lz_zstd_enc st;
  struct lz_error err;
  lz_zstd_enc_init(&st, Int_val(level), cut > 0);
  lz_zstd_enc_limit(&st, (uint32_t)Long_val(limit));

  if (cut == 0) {
    struct lz_io io = {.src = s, .src_len = src_len, .dst = d,
                       .dst_cap = dst_cap};
    enum lz_status k = lz_zstd_encode(&st, &io, true, &err);
    lz_zstd_enc_free(&st);
    if (k == LZ_ERROR) caml_raise_out_of_memory();
    return Val_long(k == LZ_DONE ? (long)io.produced : -1);
  }

  /* Each step's output goes to a run, as a stream's does, then to [dst]
     while it fits. */
  uint8_t *run = caml_stat_alloc_noexc(LZ_ZSTD_ENC_RUN);
  if (run == NULL) caml_raise_out_of_memory();
  size_t at = 0, out = 0;
  enum lz_status k;
  for (;;) {
    size_t n = src_len - at;
    if (n > cut) n = cut;
    if (n > LZ_ZSTD_ENC_QUOTA) n = LZ_ZSTD_ENC_QUOTA;
    struct lz_io io = {.src = s + at, .src_len = n, .dst = run,
                       .dst_cap = LZ_ZSTD_ENC_RUN};
    k = lz_zstd_encode(&st, &io, at + n == src_len, &err);
    if (k == LZ_ERROR) break;
    at += io.consumed;
    if (out + io.produced <= dst_cap) memcpy(d + out, run, io.produced);
    out += io.produced;
    if (k == LZ_DONE) break;
  }
  caml_stat_free(run);
  lz_zstd_enc_free(&st);
  if (k == LZ_ERROR) caml_raise_out_of_memory();
  return Val_long(out <= dst_cap ? (long)out : -1);
}
