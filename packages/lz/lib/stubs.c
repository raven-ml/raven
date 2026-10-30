/* The OCaml entries of lz: bigarrays in, results out. Every kernel is
   reached through the same few drivers below; lz.h states the contract
   they rely on.

   A call releases the runtime lock when its input holds at least its
   kernel's threshold of bytes, so that other domains and threads run while
   it works. Below the threshold the release and reacquisition would cost a
   measurable share of the call. Every OCaml value a call needs is read
   before the release, and nothing on the heap is touched until the lock
   is back; a bigarray's data never moves, and its value stays rooted by
   CAMLparam. */

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>

#include <string.h>

#include "checksum.h"
#include "deflate.h"
#include "deflate_enc.h"
#include "lz.h"
#include "lz4.h"
#include "snappy.h"
#include "zstd.h"
#include "zstd_enc.h"

/* Input bytes from which each kernel releases the runtime lock: the
   smallest power of two whose call lasts at least 100 times a release and
   reacquisition, so that the release costs under 1 % of the call. The call
   is timed at the speed of the fastest vendor build of the codec, on
   compressible input, and the release with the systhreads library linked
   and a second domain running, where it costs the most. A decoder's input
   is compressed bytes: its speed counts them, its decoded speed divided by
   the ratio. On incompressible input a call runs faster, and the release
   can cost more than 1 % of a call near the threshold. */
#define RELEASE_DEFLATE_DECODE (8 * 1024)
#define RELEASE_DEFLATE_ENCODE (4 * 1024)
#define RELEASE_ZSTD_DECODE (8 * 1024)
#define RELEASE_ZSTD_ENCODE (8 * 1024)
#define RELEASE_LZ4_DECODE (16 * 1024)
#define RELEASE_LZ4_ENCODE (8 * 1024)
#define RELEASE_SNAPPY_DECODE (16 * 1024)
#define RELEASE_SNAPPY_ENCODE (8 * 1024)

void *lz_alloc(size_t n) { return caml_stat_alloc_noexc(n); }

void lz_free(void *p) { caml_stat_free(p); }

/* Buffers */

/* An empty bigarray may have a NULL data pointer, and C forbids even
   adding zero to NULL: an empty buffer is this byte, with length 0. */
static uint8_t empty[1];

static size_t ba_len(value ba) { return Caml_ba_array_val(ba)->dim[0]; }

static uint8_t *ba_data(value ba) {
  return ba_len(ba) == 0 ? empty : (uint8_t *)Caml_ba_data_val(ba);
}

static struct lz_io io_of(value src, value dst) {
  struct lz_io io = {
      .src = ba_data(src),
      .src_len = ba_len(src),
      .dst = ba_data(dst),
      .dst_cap = ba_len(dst),
      .consumed = 0,
      .produced = 0,
  };
  return io;
}

/* A whole-buffer kernel reads [src] after it has written [dst]: the two
   must share no byte. */
static void disjoint(value src, value dst) {
  uintptr_t s = (uintptr_t)ba_data(src), d = (uintptr_t)ba_data(dst);
  size_t ls = ba_len(src), ld = ba_len(dst);
  if (ls > 0 && ld > 0 && s < d + ld && d < s + ls)
    caml_invalid_argument("Lz: src and dst overlap");
}

CAMLprim value lz_blit_in(value s, value b) {
  memcpy(ba_data(b), String_val(s), caml_string_length(s));
  return Val_unit;
}

CAMLprim value lz_blit_out(value b, value s, value n) {
  memcpy(Bytes_val(s), ba_data(b), (size_t)Long_val(n));
  return Val_unit;
}

/* Results */

/* [Ok { consumed; produced }]. */
static value ok_progress(size_t consumed, size_t produced) {
  CAMLparam0();
  CAMLlocal2(p, r);
  p = caml_alloc_small(2, 0);
  Field(p, 0) = Val_long(consumed);
  Field(p, 1) = Val_long(produced);
  r = caml_alloc_small(1, 0);
  Field(r, 0) = p;
  CAMLreturn(r);
}

/* lz.ml's [outcome]: [{ taken; start; len; fin }]. */
static value outcome(size_t taken, size_t start, size_t len, bool fin) {
  value o = caml_alloc_small(4, 0);
  Field(o, 0) = Val_long(taken);
  Field(o, 1) = Val_long(start);
  Field(o, 2) = Val_long(len);
  Field(o, 3) = Val_bool(fin);
  return o;
}

/* lz.ml's [lent]: [{ run; buf }], a decoder step's outcome and the buffer
   its run lies in. */
static value lent(size_t taken, size_t start, size_t len, bool fin,
                  value buf) {
  CAMLparam1(buf);
  CAMLlocal2(o, r);
  o = outcome(taken, start, len, fin);
  r = caml_alloc_small(2, 0);
  Field(r, 0) = o;
  Field(r, 1) = buf;
  CAMLreturn(r);
}

static value ok(value v) {
  CAMLparam1(v);
  CAMLlocal1(r);
  r = caml_alloc_small(1, 0);
  Field(r, 0) = v;
  CAMLreturn(r);
}

/* [Error { at = base + err.at; why }]. [why]'s constant constructors are
   Truncated (0) and Too_long (1); its blocks Malformed (tag 0) and
   Unsupported (tag 1). */
static value error_result(const struct lz_error *err, int64_t base) {
  CAMLparam0();
  CAMLlocal4(msg, why, e, r);
  switch (err->kind) {
  case LZ_TRUNCATED:
    why = Val_int(0);
    break;
  case LZ_TOO_LONG:
    why = Val_int(1);
    break;
  case LZ_MALFORMED:
  case LZ_UNSUPPORTED:
    msg = caml_copy_string(err->msg ? err->msg : "");
    why = caml_alloc_small(1, err->kind == LZ_MALFORMED ? 0 : 1);
    Field(why, 0) = msg;
    break;
  default:
    caml_raise_out_of_memory();
  }
  e = caml_alloc_small(2, 0);
  Field(e, 0) = Val_long(base + err->at);
  Field(e, 1) = why;
  r = caml_alloc_small(1, 1);
  Field(r, 0) = e;
  CAMLreturn(r);
}

static value some_int(size_t n) {
  value r = caml_alloc_small(1, 0);
  Field(r, 0) = Val_long(n);
  return r;
}

/* A bound as an OCaml int, or -1 past max_int, for lz.ml to refuse. A
   bound of a size up to max_int fits a size_t. */
static value bound_val(size_t b) {
  return Val_long(b > (size_t)Max_long ? -1 : (intnat)b);
}

/* Drivers */

/* A format's entries, so that one driver serves them all. A [members]
   stream is a sequence of members read to the input's end (gzip,
   Zstandard); otherwise it is one stream with an exact end. [run] is the
   run an encoder step fills from at most [quota] bytes of input; a
   decoder's buffer is empty until the decoder asks for one (lz.h). */
struct codec {
  enum lz_status (*decode)(void *st, struct lz_io *io, bool final,
                           struct lz_error *err);
  enum lz_status (*encode)(void *st, struct lz_io *io, bool last,
                           struct lz_error *err);
  void (*reset)(void *st);
  void (*free)(void *st);
  bool members;
  size_t run;
  size_t quota;
  size_t release;
};

/* The encoders state their quota and run in deflate_enc.h and
   zstd_enc.h; the decoders their runs in deflate.h and zstd.h. */

/* The whole-buffer decode: the stream, and for a [members] format every
   member after it to the input's end, each decoded in place after the
   last. Where bytes remain, the kernel begins a member there and refuses a
   byte that begins none. Output past [dst_cap] is Too_long at the input
   offset the kernel had reached, unless the kernel placed it itself. */
static enum lz_status whole(const struct codec *c, void *st, struct lz_io *io,
                            struct lz_error *err) {
  for (;;) {
    struct lz_io m = {
        .src = io->src + io->consumed,
        .src_len = io->src_len - io->consumed,
        .dst = io->dst + io->produced,
        .dst_cap = io->dst_cap - io->produced,
        .consumed = 0,
        .produced = 0,
    };
    size_t base = io->consumed;
    enum lz_status s = c->decode(st, &m, true, err);
    io->consumed += m.consumed;
    io->produced += m.produced;
    switch (s) {
    case LZ_ERROR:
      err->at += (int64_t)base;
      return s;
    case LZ_NEED_OUTPUT:
    case LZ_NEED_BUFFER: /* only a stream asks for a buffer */
      return lz_fail(err, LZ_TOO_LONG, (int64_t)io->consumed, NULL);
    case LZ_NEED_INPUT:
      return lz_fail(err, LZ_TRUNCATED, (int64_t)io->src_len, NULL);
    case LZ_DONE:
      break;
    }
    if (!c->members || io->consumed == io->src_len || m.consumed == 0)
      return LZ_DONE;
  }
}

static value decode_whole(const struct codec *c, void *st, value src,
                          value dst) {
  CAMLparam2(src, dst);
  disjoint(src, dst);
  struct lz_io io = io_of(src, dst);
  struct lz_error err;
  bool release = io.src_len >= c->release;
  if (release) caml_enter_blocking_section();
  enum lz_status s = whole(c, st, &io, &err);
  if (release) caml_leave_blocking_section();
  c->free(st);
  if (s == LZ_ERROR) CAMLreturn(error_result(&err, 0));
  CAMLreturn(ok_progress(io.consumed, io.produced));
}

static value encode_whole(const struct codec *c, void *st, value src,
                          value dst) {
  CAMLparam2(src, dst);
  disjoint(src, dst);
  struct lz_io io = io_of(src, dst);
  struct lz_error err;
  bool release = io.src_len >= c->release;
  if (release) caml_enter_blocking_section();
  enum lz_status s = c->encode(st, &io, true, &err);
  if (release) caml_leave_blocking_section();
  c->free(st);
  if (s == LZ_ERROR) caml_raise_out_of_memory();
  if (s != LZ_DONE) CAMLreturn(Val_none);
  CAMLreturn(some_int(io.produced));
}

/* A streaming state: the kernel's state, its codec, the input taken since
   creation or the last Done (to which an error's offset is relative),
   whether a member has ended with no byte of another taken, and for a
   decoder where its next byte goes in its buffer. lz.ml keeps the buffer,
   who holds the state and whether it is dead. A step copies the box
   before it releases the lock, since compaction may move the custom
   block, and writes it back after. */
struct box {
  void *st;
  const struct codec *codec;
  int64_t base;
  bool boundary;
  size_t pos;
};

#define Box_val(v) ((struct box *)Data_custom_val(v))

static void box_finalize(value v) {
  struct box *b = Box_val(v);
  if (b->st == NULL) return;
  b->codec->free(b->st);
  caml_stat_free(b->st);
  b->st = NULL;
}

static struct custom_operations box_ops = {
    "raven.lz.state",           box_finalize,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default,
};

/* [box c st mem] takes ownership of [st], a [caml_stat_alloc] block that
   its init has set up and that holds [mem] bytes outside the heap, which
   the collector counts to pace itself. */
static value box(const struct codec *c, void *st, size_t mem) {
  value v = caml_alloc_custom_mem(&box_ops, sizeof(struct box), mem);
  struct box *b = Box_val(v);
  b->st = st;
  b->codec = c;
  b->base = 0;
  b->boundary = false;
  b->pos = 0;
  return v;
}

CAMLprim value lz_run_size(value v) {
  return Val_long(Box_val(v)->codec->run);
}

/* One decoder step over [io], its input and the decoder's buffer (lz.h):
   members are decoded until the input is used, the run is full or the
   kernel asks for a larger buffer. With [last], a [members] stream ends
   where a member ends at the input's end; a stream with an exact end ends
   at it whatever [last] says, and the kernel is reset for the next. */
static enum lz_status dec_run(struct box *k, struct lz_io *io, bool last,
                              bool *fin, struct lz_error *err) {
  const struct codec *c = k->codec;
  for (;;) {
    if (k->boundary && io->consumed == io->src_len) {
      if (!last) return LZ_NEED_INPUT;
      c->reset(k->st);
      k->boundary = false;
      *fin = true;
      return LZ_DONE;
    }
    struct lz_io m = {
        .src = io->src + io->consumed,
        .src_len = io->src_len - io->consumed,
        .dst = io->dst,
        .dst_cap = io->dst_cap,
        .produced = io->produced,
        .run = io->run,
    };
    size_t before = io->consumed;
    enum lz_status s = c->decode(k->st, &m, last, err);
    io->consumed += m.consumed;
    io->produced = m.produced;
    io->run = m.run;
    io->need = m.need;
    if (m.consumed > 0) k->boundary = false;
    switch (s) {
    case LZ_ERROR:
      err->at += (int64_t)before;
      return s;
    case LZ_NEED_INPUT:
    case LZ_NEED_OUTPUT:
    case LZ_NEED_BUFFER:
      return s;
    case LZ_DONE:
      if (!c->members) {
        c->reset(k->st);
        *fin = true;
        return LZ_DONE;
      }
      k->boundary = true;
      break;
    }
  }
}

/* [grow io buf] is a buffer of [io->need] bytes that replaces [buf],
   holding the run at its start, with [io] moved onto it. */
static value grow(struct lz_io *io, value buf) {
  CAMLparam1(buf);
  CAMLlocal1(b);
  size_t len = io->produced - io->run;
  b = caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT, 1, NULL,
                         (intnat)io->need);
  uint8_t *d = (uint8_t *)Caml_ba_data_val(b);
  memcpy(d, io->dst + io->run, len);
  io->dst = d;
  io->dst_cap = io->need;
  io->produced = len;
  io->run = 0;
  CAMLreturn(b);
}

/* A step: the run begins where the last step's ended. A kernel that asks
   for a larger buffer gets it under the lock, and the step goes on in it,
   so that [f] sees one run. */
CAMLprim value lz_dec_step(value v, value last, value src, value buf) {
  CAMLparam4(v, last, src, buf);
  struct box k = *Box_val(v);
  struct lz_io io = io_of(src, buf);
  io.produced = k.pos;
  io.run = k.pos;
  struct lz_error err;
  bool l = Bool_val(last), fin = false;
  enum lz_status s;
  for (;;) {
    bool release = io.src_len - io.consumed >= k.codec->release;
    if (release) caml_enter_blocking_section();
    s = dec_run(&k, &io, l, &fin, &err);
    if (release) caml_leave_blocking_section();
    if (s != LZ_NEED_BUFFER) break;
    buf = grow(&io, buf);
  }
  if (s == LZ_ERROR) {
    if (err.kind == LZ_NOMEM) caml_raise_out_of_memory();
    CAMLreturn(error_result(&err, k.base));
  }
  k.base = fin ? 0 : k.base + (int64_t)io.consumed;
  k.pos = io.produced;
  *Box_val(v) = k;
  CAMLreturn(
      ok(lent(io.consumed, io.run, io.produced - io.run, fin, buf)));
}

CAMLprim value lz_enc_step(value v, value last, value src, value buf) {
  CAMLparam4(v, last, src, buf);
  struct box k = *Box_val(v);
  struct lz_io io = io_of(src, buf);
  struct lz_error err;
  bool l = Bool_val(last);
  if (io.src_len > k.codec->quota) {
    io.src_len = k.codec->quota;
    l = false;
  }
  bool release = io.src_len >= k.codec->release;
  if (release) caml_enter_blocking_section();
  enum lz_status s = k.codec->encode(k.st, &io, l, &err);
  if (release) caml_leave_blocking_section();
  if (s == LZ_ERROR) caml_raise_out_of_memory();
  CAMLreturn(outcome(io.consumed, 0, io.produced, s == LZ_DONE));
}

/* Deflate, zlib and gzip */

static enum lz_status deflate_decode(void *st, struct lz_io *io, bool final,
                                     struct lz_error *err) {
  return lz_deflate_decode(st, io, final, err);
}

static void deflate_dec_reset(void *st) { lz_deflate_dec_reset(st); }

static void deflate_dec_free(void *st) { lz_deflate_dec_free(st); }

static enum lz_status deflate_encode(void *st, struct lz_io *io, bool last,
                                     struct lz_error *err) {
  return lz_deflate_encode(st, io, last, err);
}

static void deflate_enc_reset(void *st) { lz_deflate_enc_reset(st); }

static void deflate_enc_free(void *st) { lz_deflate_enc_free(st); }

static const struct codec deflate_dec = {
    .decode = deflate_decode,
    .reset = deflate_dec_reset,
    .free = deflate_dec_free,
    .members = false,
    .release = RELEASE_DEFLATE_DECODE,
};

static const struct codec gzip_dec = {
    .decode = deflate_decode,
    .reset = deflate_dec_reset,
    .free = deflate_dec_free,
    .members = true,
    .release = RELEASE_DEFLATE_DECODE,
};

static const struct codec deflate_enc = {
    .encode = deflate_encode,
    .reset = deflate_enc_reset,
    .free = deflate_enc_free,
    .run = LZ_DEFLATE_ENC_RUN,
    .quota = LZ_DEFLATE_ENC_QUOTA,
    .release = RELEASE_DEFLATE_ENCODE,
};

static const struct codec *deflate_dec_of(int wrap) {
  return wrap == LZ_GZIP ? &gzip_dec : &deflate_dec;
}

CAMLprim value lz_deflate_decode_whole(value wrap, value src, value dst) {
  struct lz_deflate_dec st;
  lz_deflate_dec_init(&st, Int_val(wrap), false);
  return decode_whole(deflate_dec_of(Int_val(wrap)), &st, src, dst);
}

CAMLprim value lz_deflate_encode_whole(value wrap, value level, value src,
                                       value dst) {
  struct lz_deflate_enc st;
  lz_deflate_enc_init(&st, Int_val(wrap), Int_val(level), false);
  return encode_whole(&deflate_enc, &st, src, dst);
}

CAMLprim value lz_deflate_bound_stub(value wrap, value n) {
  return bound_val(lz_deflate_bound(Int_val(wrap), Long_val(n)));
}

CAMLprim value lz_deflate_decoder(value wrap) {
  struct lz_deflate_dec *st = caml_stat_alloc(sizeof *st);
  lz_deflate_dec_init(st, Int_val(wrap), true);
  return box(deflate_dec_of(Int_val(wrap)), st, sizeof *st);
}

CAMLprim value lz_deflate_encoder(value wrap, value level) {
  struct lz_deflate_enc *st = caml_stat_alloc(sizeof *st);
  lz_deflate_enc_init(st, Int_val(wrap), Int_val(level), true);
  return box(&deflate_enc, st, sizeof *st + lz_deflate_enc_mem(Int_val(level)));
}

/* Zstandard */

static enum lz_status zstd_decode(void *st, struct lz_io *io, bool final,
                                  struct lz_error *err) {
  return lz_zstd_decode(st, io, final, err);
}

static void zstd_dec_reset(void *st) { lz_zstd_dec_reset(st); }

static void zstd_dec_free(void *st) { lz_zstd_dec_free(st); }

static enum lz_status zstd_encode(void *st, struct lz_io *io, bool last,
                                  struct lz_error *err) {
  return lz_zstd_encode(st, io, last, err);
}

static void zstd_enc_reset(void *st) { lz_zstd_enc_reset(st); }

static void zstd_enc_free(void *st) { lz_zstd_enc_free(st); }

static const struct codec zstd_dec = {
    .decode = zstd_decode,
    .reset = zstd_dec_reset,
    .free = zstd_dec_free,
    .members = true,
    .release = RELEASE_ZSTD_DECODE,
};

static const struct codec zstd_enc = {
    .encode = zstd_encode,
    .reset = zstd_enc_reset,
    .free = zstd_enc_free,
    .run = LZ_ZSTD_ENC_RUN,
    .quota = LZ_ZSTD_ENC_QUOTA,
    .release = RELEASE_ZSTD_ENCODE,
};

/* A whole-buffer decode accepts any window the format allows: the
   destination holds it. */
CAMLprim value lz_zstd_decode_whole(value src, value dst) {
  struct lz_zstd_dec st;
  lz_zstd_dec_init(&st, UINT64_MAX, false);
  return decode_whole(&zstd_dec, &st, src, dst);
}

CAMLprim value lz_zstd_encode_whole(value level, value src, value dst) {
  struct lz_zstd_enc st;
  lz_zstd_enc_init(&st, Int_val(level), false);
  return encode_whole(&zstd_enc, &st, src, dst);
}

CAMLprim value lz_zstd_bound_stub(value n) {
  return bound_val(lz_zstd_bound(Long_val(n)));
}

/* The buffer is allocated when a frame header fixes the window; the
   gathered block and the literals, at the first frame, outside what the
   collector is told here. */
CAMLprim value lz_zstd_decoder(value window) {
  struct lz_zstd_dec *st = caml_stat_alloc(sizeof *st);
  lz_zstd_dec_init(st, (uint64_t)Long_val(window), true);
  return box(&zstd_dec, st, sizeof *st);
}

CAMLprim value lz_zstd_encoder(value level) {
  struct lz_zstd_enc *st = caml_stat_alloc(sizeof *st);
  lz_zstd_enc_init(st, Int_val(level), true);
  return box(&zstd_enc, st, sizeof *st + lz_zstd_enc_mem(Int_val(level)));
}

/* LZ4 and Snappy: whole-buffer only, each entry keeping its state on its
   own stack. */

static value simple_decode(enum lz_status (*f)(struct lz_io *,
                                               struct lz_error *),
                           size_t threshold, value src, value dst) {
  CAMLparam2(src, dst);
  disjoint(src, dst);
  struct lz_io io = io_of(src, dst);
  struct lz_error err;
  bool release = io.src_len >= threshold;
  if (release) caml_enter_blocking_section();
  enum lz_status s = f(&io, &err);
  if (release) caml_leave_blocking_section();
  if (s == LZ_ERROR) CAMLreturn(error_result(&err, 0));
  CAMLreturn(ok_progress(io.consumed, io.produced));
}

static value simple_encode(enum lz_status (*f)(struct lz_io *),
                           size_t threshold, value src, value dst) {
  CAMLparam2(src, dst);
  disjoint(src, dst);
  struct lz_io io = io_of(src, dst);
  bool release = io.src_len >= threshold;
  if (release) caml_enter_blocking_section();
  enum lz_status s = f(&io);
  if (release) caml_leave_blocking_section();
  /* An encoder fails only when its table cannot be allocated. */
  if (s == LZ_ERROR) caml_raise_out_of_memory();
  if (s != LZ_DONE) CAMLreturn(Val_none);
  CAMLreturn(some_int(io.produced));
}

CAMLprim value lz_lz4_block_decode_stub(value src, value dst) {
  return simple_decode(lz_lz4_block_decode, RELEASE_LZ4_DECODE, src, dst);
}

CAMLprim value lz_lz4_block_encode_stub(value src, value dst) {
  return simple_encode(lz_lz4_block_encode, RELEASE_LZ4_ENCODE, src, dst);
}

CAMLprim value lz_lz4_block_bound_stub(value n) {
  return bound_val(lz_lz4_block_bound(Long_val(n)));
}

CAMLprim value lz_lz4_frame_decode_stub(value src, value dst) {
  return simple_decode(lz_lz4_frame_decode, RELEASE_LZ4_DECODE, src, dst);
}

CAMLprim value lz_lz4_frame_encode_stub(value src, value dst) {
  return simple_encode(lz_lz4_frame_encode, RELEASE_LZ4_ENCODE, src, dst);
}

CAMLprim value lz_lz4_frame_bound_stub(value n) {
  return bound_val(lz_lz4_frame_bound(Long_val(n)));
}

CAMLprim value lz_snappy_decode_stub(value src, value dst) {
  return simple_decode(lz_snappy_decode, RELEASE_SNAPPY_DECODE, src, dst);
}

CAMLprim value lz_snappy_encode_stub(value src, value dst) {
  return simple_encode(lz_snappy_encode, RELEASE_SNAPPY_ENCODE, src, dst);
}

CAMLprim value lz_snappy_bound_stub(value n) {
  return bound_val(lz_snappy_bound(Long_val(n)));
}

/* Checksums. A call below lz.ml's threshold takes the [noalloc] entry,
   which neither allocates nor releases the lock; a longer one takes the
   entry that releases it. A string's bytes may move during a collection,
   so a string never releases the lock. Native code passes unboxed values;
   the [_byte] entries serve bytecode. */

static const uint8_t *str_at(value s, intnat first) {
  return (const uint8_t *)String_val(s) + first;
}

static const uint8_t *ba_at(value b, intnat first) {
  return ba_data(b) + first;
}

/* CRC-32, as an OCaml [int] in [0, 2^32). */

intnat lz_crc32_ba(intnat crc, value b, intnat first, intnat len) {
  return (intnat)lz_crc32((uint32_t)crc, ba_at(b, first), (size_t)len);
}

CAMLprim value lz_crc32_ba_byte(value crc, value b, value first, value len) {
  return Val_long(
      lz_crc32_ba(Long_val(crc), b, Long_val(first), Long_val(len)));
}

intnat lz_crc32_str(intnat crc, value s, intnat first, intnat len) {
  return (intnat)lz_crc32((uint32_t)crc, str_at(s, first), (size_t)len);
}

CAMLprim value lz_crc32_str_byte(value crc, value s, value first, value len) {
  return Val_long(
      lz_crc32_str(Long_val(crc), s, Long_val(first), Long_val(len)));
}

CAMLprim value lz_crc32_ba_long(value crc, value b, value first, value len) {
  CAMLparam1(b);
  uint32_t c = (uint32_t)Long_val(crc);
  const uint8_t *p = ba_at(b, Long_val(first));
  size_t n = (size_t)Long_val(len);
  caml_enter_blocking_section();
  c = lz_crc32(c, p, n);
  caml_leave_blocking_section();
  CAMLreturn(Val_long((intnat)c));
}

/* Adler-32 */

uint32_t lz_adler32_ba(uint32_t a, value b, intnat first, intnat len) {
  return lz_adler32(a, ba_at(b, first), (size_t)len);
}

CAMLprim value lz_adler32_ba_byte(value a, value b, value first, value len) {
  return caml_copy_int32((int32_t)lz_adler32_ba((uint32_t)Int32_val(a), b,
                                                Long_val(first),
                                                Long_val(len)));
}

uint32_t lz_adler32_str(uint32_t a, value s, intnat first, intnat len) {
  return lz_adler32(a, str_at(s, first), (size_t)len);
}

CAMLprim value lz_adler32_str_byte(value a, value s, value first,
                                   value len) {
  return caml_copy_int32((int32_t)lz_adler32_str((uint32_t)Int32_val(a), s,
                                                 Long_val(first),
                                                 Long_val(len)));
}

CAMLprim value lz_adler32_ba_long(value a, value b, value first, value len) {
  CAMLparam1(b);
  uint32_t c = (uint32_t)Int32_val(a);
  const uint8_t *p = ba_at(b, Long_val(first));
  size_t n = (size_t)Long_val(len);
  caml_enter_blocking_section();
  c = lz_adler32(c, p, n);
  caml_leave_blocking_section();
  CAMLreturn(caml_copy_int32((int32_t)c));
}

/* XXH32 */

uint32_t lz_xxh32_ba(uint32_t seed, value b, intnat first, intnat len) {
  return lz_xxh32(seed, ba_at(b, first), (size_t)len);
}

CAMLprim value lz_xxh32_ba_byte(value seed, value b, value first,
                                value len) {
  return caml_copy_int32((int32_t)lz_xxh32_ba((uint32_t)Int32_val(seed), b,
                                              Long_val(first),
                                              Long_val(len)));
}

uint32_t lz_xxh32_str(uint32_t seed, value s, intnat first, intnat len) {
  return lz_xxh32(seed, str_at(s, first), (size_t)len);
}

CAMLprim value lz_xxh32_str_byte(value seed, value s, value first,
                                 value len) {
  return caml_copy_int32((int32_t)lz_xxh32_str((uint32_t)Int32_val(seed), s,
                                               Long_val(first),
                                               Long_val(len)));
}

CAMLprim value lz_xxh32_ba_long(value seed, value b, value first,
                                value len) {
  CAMLparam1(b);
  uint32_t c = (uint32_t)Int32_val(seed);
  const uint8_t *p = ba_at(b, Long_val(first));
  size_t n = (size_t)Long_val(len);
  caml_enter_blocking_section();
  c = lz_xxh32(c, p, n);
  caml_leave_blocking_section();
  CAMLreturn(caml_copy_int32((int32_t)c));
}

/* XXH64 */

uint64_t lz_xxh64_ba(uint64_t seed, value b, intnat first, intnat len) {
  return lz_xxh64(seed, ba_at(b, first), (size_t)len);
}

CAMLprim value lz_xxh64_ba_byte(value seed, value b, value first,
                                value len) {
  return caml_copy_int64((int64_t)lz_xxh64_ba((uint64_t)Int64_val(seed), b,
                                              Long_val(first),
                                              Long_val(len)));
}

uint64_t lz_xxh64_str(uint64_t seed, value s, intnat first, intnat len) {
  return lz_xxh64(seed, str_at(s, first), (size_t)len);
}

CAMLprim value lz_xxh64_str_byte(value seed, value s, value first,
                                 value len) {
  return caml_copy_int64((int64_t)lz_xxh64_str((uint64_t)Int64_val(seed), s,
                                               Long_val(first),
                                               Long_val(len)));
}

CAMLprim value lz_xxh64_ba_long(value seed, value b, value first,
                                value len) {
  CAMLparam1(b);
  uint64_t c = (uint64_t)Int64_val(seed);
  const uint8_t *p = ba_at(b, Long_val(first));
  size_t n = (size_t)Long_val(len);
  caml_enter_blocking_section();
  c = lz_xxh64(c, p, n);
  caml_leave_blocking_section();
  CAMLreturn(caml_copy_int64((int64_t)c));
}
