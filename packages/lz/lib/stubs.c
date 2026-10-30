/* The OCaml entries of lz: bigarrays in, results out. Every kernel is
   reached through the same few drivers below; lz.h states the contract
   they rely on.

   A call releases the runtime lock from inside its kernel, once the work
   the kernel reports reaches the kernel's threshold (lz.h), so that the
   domain's other threads run while it works, and takes it back when the
   kernel returns. A call that does less work holds the lock, whatever
   the size of its input or its destination: a release would cost a
   measurable share of it, and while the lock is released another thread
   of the domain may take it, and the call then waits on return until
   that thread yields it, up to its time slice. Work counts what a call
   does, not what it may do: a 2 KiB Zstandard frame decoding to 64 MiB
   releases at its second block, and a step on 100 bytes that writes a
   few hundred holds the lock.

   A step copies its box to the C stack before its kernel runs, since
   compaction may move the custom block while the lock is released, and
   writes it back after. Every other value a call needs is read before
   the kernel runs; a bigarray's data never moves, and its value stays
   rooted by CAMLparam. */

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
#include "checksum_paths.h"
#include "deflate.h"
#include "deflate_enc.h"
#include "lz.h"
#include "lz4.h"
#include "snappy.h"
#include "zstd.h"
#include "zstd_dec_paths.h"
#include "zstd_enc.h"

/* Bytes of work from which each kernel releases the runtime lock: the
   smallest power of two of input whose call lasts at least 100 times a
   release and reacquisition, so that the release costs under 1 % of the
   call. The call is timed at the speed of the fastest vendor build of the
   codec, on compressible input, and the release with the systhreads
   library linked and a second domain running, where it costs the most. A
   decoder's input is compressed bytes: its speed counts them, its decoded
   speed divided by the ratio. An encoder's work is its input, as
   measured. A decoder's counts its input and its output, so on input of
   ratio r it reaches the threshold after 1 / (1 + r) of the input
   measured, where the release costs up to 1 + r % of the call. On
   incompressible input a call runs faster, and the release can cost more
   than 1 % of a call near the threshold. The cost leaves out the wait for
   the lock when another thread of the domain has taken it. */
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

/* The lock */

/* A call's work (lz.h) and whether its kernel has released the lock. [w]
   comes first, so that [reach] finds the call from it. */
struct call {
  struct lz_work w;
  bool released;
};

/* The releases the calling thread has made, and the work its call had
   counted at the last, for the tests (lz.mli). */
static _Thread_local intnat releases;
static _Thread_local size_t released_at;

/* Every release goes through here. It runs no signal handler: a handler
   may raise, and a raise from inside a kernel would skip what its stub
   does after it, such as freeing an encode's tables. A signal that
   arrives while the lock is released is handled at the first poll after
   the stub takes it back. */
static void release(struct call *c) {
  caml_enter_blocking_section_no_pending();
  releases++;
  released_at = c->w.done;
  c->released = true;
  c->w.limit = SIZE_MAX;
}

static void reach(struct lz_work *w) { release((struct call *)w); }

static void start(struct call *c, size_t threshold) {
  c->w.done = 0;
  c->w.limit = threshold;
  c->w.reach = reach;
  c->released = false;
}

/* Takes the lock back if the call's kernel released it. */
static void reacquire(struct call *c) {
  if (!c->released) return;
  caml_leave_blocking_section();
  c->released = false;
}

/* [tally w before did] counts the [did] bytes a decoder kernel took and
   wrote, of which [w] held [before] when it began: the work after its
   last report, added without a release (lz.h). */
static void tally(struct lz_work *w, size_t before, size_t did) {
  if (w->done - before < did) w->done = before + did;
}

CAMLprim value lz_releases(value unit) {
  (void)unit;
  return Val_long(releases);
}

CAMLprim value lz_released_at(value unit) {
  (void)unit;
  return Val_long(released_at);
}

/* Buffers */

/* An empty bigarray may have a NULL data pointer, and C forbids even
   adding zero to NULL: an empty buffer is this byte, with length 0. */
static uint8_t empty[1];

static size_t ba_len(value ba) { return Caml_ba_array_val(ba)->dim[0]; }

static uint8_t *ba_data(value ba) {
  return ba_len(ba) == 0 ? empty : (uint8_t *)Caml_ba_data_val(ba);
}

static struct lz_io io_of(value src, value dst, struct call *c) {
  struct lz_io io = {
      .src = ba_data(src),
      .src_len = ba_len(src),
      .dst = ba_data(dst),
      .dst_cap = ba_len(dst),
      .consumed = 0,
      .produced = 0,
      .work = &c->w,
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
   decoder's buffer is empty until the decoder asks for one (lz.h).
   [release] is the kernel's threshold (above). */
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
        .work = io->work,
    };
    size_t base = io->consumed, before = io->work->done;
    enum lz_status s = c->decode(st, &m, true, err);
    tally(io->work, before, m.consumed + m.produced);
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
    /* Another member follows: check the work tallied (lz.h). */
    lz_did(io->work, 0);
  }
}

static value decode_whole(const struct codec *c, void *st, value src,
                          value dst) {
  CAMLparam2(src, dst);
  disjoint(src, dst);
  struct call k;
  start(&k, c->release);
  struct lz_io io = io_of(src, dst, &k);
  struct lz_error err;
  enum lz_status s = whole(c, st, &io, &err);
  reacquire(&k);
  c->free(st);
  if (s == LZ_ERROR) CAMLreturn(error_result(&err, 0));
  CAMLreturn(ok_progress(io.consumed, io.produced));
}

static value encode_whole(const struct codec *c, void *st, value src,
                          value dst) {
  CAMLparam2(src, dst);
  disjoint(src, dst);
  struct call k;
  start(&k, c->release);
  struct lz_io io = io_of(src, dst, &k);
  struct lz_error err;
  enum lz_status s = c->encode(st, &io, true, &err);
  reacquire(&k);
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
   before its kernel runs, since compaction may move the custom block
   while the kernel has the lock released, and writes it back after. */
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
    /* A member follows the last one: check the work tallied (lz.h). */
    if (k->boundary) lz_did(io->work, 0);
    struct lz_io m = {
        .src = io->src + io->consumed,
        .src_len = io->src_len - io->consumed,
        .dst = io->dst,
        .dst_cap = io->dst_cap,
        .produced = io->produced,
        .run = io->run,
        .work = io->work,
    };
    /* The run so far: a kernel that wraps the buffer does so while the
       run is empty, so its growth is the kernel's output. */
    size_t before = io->consumed, told = io->work->done;
    size_t ran = io->produced - io->run;
    enum lz_status s = c->decode(k->st, &m, last, err);
    tally(io->work, told, m.consumed + (m.produced - m.run) - ran);
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
   so that [f] sees one run; a step that had released the lock releases
   it again to go on. */
CAMLprim value lz_dec_step(value v, value last, value src, value buf) {
  CAMLparam4(v, last, src, buf);
  struct box k = *Box_val(v);
  struct call c;
  start(&c, k.codec->release);
  struct lz_io io = io_of(src, buf, &c);
  io.produced = k.pos;
  io.run = k.pos;
  struct lz_error err;
  bool l = Bool_val(last), fin = false;
  enum lz_status s;
  for (;;) {
    s = dec_run(&k, &io, l, &fin, &err);
    bool released = c.released;
    reacquire(&c);
    if (s != LZ_NEED_BUFFER) break;
    buf = grow(&io, buf);
    if (released) release(&c);
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
  struct call c;
  start(&c, k.codec->release);
  struct lz_io io = io_of(src, buf, &c);
  struct lz_error err;
  bool l = Bool_val(last);
  if (io.src_len > k.codec->quota) {
    io.src_len = k.codec->quota;
    l = false;
  }
  enum lz_status s = k.codec->encode(k.st, &io, l, &err);
  reacquire(&c);
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
  struct call c;
  start(&c, threshold);
  struct lz_io io = io_of(src, dst, &c);
  struct lz_error err;
  enum lz_status s = f(&io, &err);
  reacquire(&c);
  if (s == LZ_ERROR) CAMLreturn(error_result(&err, 0));
  CAMLreturn(ok_progress(io.consumed, io.produced));
}

static value simple_encode(enum lz_status (*f)(struct lz_io *),
                           size_t threshold, value src, value dst) {
  CAMLparam2(src, dst);
  disjoint(src, dst);
  struct call c;
  start(&c, threshold);
  struct lz_io io = io_of(src, dst, &c);
  enum lz_status s = f(&io);
  reacquire(&c);
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

/* Checksums. A checksum's work is its input, known before it starts: a
   call below lz.ml's threshold takes the [noalloc] entry, which neither
   allocates nor releases the lock, and a longer one the entry that
   releases it at its start. A string's bytes may move during a
   collection, so a string never releases the lock. Native code passes
   unboxed values; the [_byte] entries serve bytecode. */

static void release_now(struct call *k) {
  start(k, 0);
  release(k);
}

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
  struct call k;
  release_now(&k);
  c = lz_crc32(c, p, n);
  reacquire(&k);
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
  struct call k;
  release_now(&k);
  c = lz_adler32(c, p, n);
  reacquire(&k);
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
  struct call k;
  release_now(&k);
  c = lz_xxh32(c, p, n);
  reacquire(&k);
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
  struct call k;
  release_now(&k);
  c = lz_xxh64(c, p, n);
  reacquire(&k);
  CAMLreturn(caml_copy_int64((int64_t)c));
}

/* Paths, for the tests that hold every path of a kernel to the same bits.
   Each entry runs path [i] of a kernel through the same driver as the
   modules' own entries, which take the last path. */

static void check_path(value i, size_t n) {
  if (Long_val(i) < 0 || (size_t)Long_val(i) >= n)
    caml_invalid_argument("Lz: no such path");
}

static value names_of(const char *const *names, size_t n) {
  CAMLparam0();
  CAMLlocal2(a, s);
  a = caml_alloc(n, 0);
  for (size_t i = 0; i < n; i++) {
    s = caml_copy_string(names[i]);
    Store_field(a, i, s);
  }
  CAMLreturn(a);
}

/* [kind] is 0 for CRC-32, 1 for Adler-32. */
static size_t checksum_paths(value kind, const struct lz_checksum_path **p) {
  return Int_val(kind) == 0 ? lz_crc32_paths(p) : lz_adler32_paths(p);
}

CAMLprim value lz_checksum_path_names(value kind) {
  CAMLparam1(kind);
  CAMLlocal2(a, s);
  const struct lz_checksum_path *p;
  size_t n = checksum_paths(kind, &p);
  a = caml_alloc(n, 0);
  for (size_t i = 0; i < n; i++) {
    s = caml_copy_string(p[i].name);
    Store_field(a, i, s);
  }
  CAMLreturn(a);
}

/* Path [i] of [kind] over all of [b], from the running value [c]. */
CAMLprim value lz_checksum_path(value kind, value i, value c, value b) {
  const struct lz_checksum_path *p;
  check_path(i, checksum_paths(kind, &p));
  uint32_t r = p[Long_val(i)].fn((uint32_t)Int32_val(c), ba_data(b),
                                 ba_len(b));
  return caml_copy_int32((int32_t)r);
}

CAMLprim value lz_zstd_dec_path_names(value unit) {
  (void)unit;
  const char *const *names;
  size_t n = lz_zstd_dec_paths(&names);
  return names_of(names, n);
}

CAMLprim value lz_zstd_decode_path(value i, value src, value dst) {
  const char *const *names;
  check_path(i, lz_zstd_dec_paths(&names));
  struct lz_zstd_dec st;
  lz_zstd_dec_init(&st, UINT64_MAX, false);
  lz_zstd_dec_path(&st, (size_t)Long_val(i));
  return decode_whole(&zstd_dec, &st, src, dst);
}

CAMLprim value lz_zstd_enc_path_names(value unit) {
  (void)unit;
  const char *const *names;
  size_t n = lz_zstd_enc_paths(&names);
  return names_of(names, n);
}

CAMLprim value lz_zstd_encode_path(value i, value level, value src,
                                   value dst) {
  const char *const *names;
  check_path(i, lz_zstd_enc_paths(&names));
  struct lz_zstd_enc st;
  lz_zstd_enc_init(&st, Int_val(level), false);
  lz_zstd_enc_path(&st, (size_t)Long_val(i));
  return encode_whole(&zstd_enc, &st, src, dst);
}

CAMLprim value lz_deflate_enc_path_names(value unit) {
  (void)unit;
  const char *const *names;
  size_t n = lz_deflate_enc_paths(&names);
  return names_of(names, n);
}

CAMLprim value lz_deflate_encode_path(value i, value level, value src,
                                      value dst) {
  const char *const *names;
  check_path(i, lz_deflate_enc_paths(&names));
  struct lz_deflate_enc st;
  lz_deflate_enc_init(&st, LZ_RAW, Int_val(level), false);
  lz_deflate_enc_path(&st, (size_t)Long_val(i));
  return encode_whole(&deflate_enc, &st, src, dst);
}

/* Zstandard's encoders with the match finders' index limit lowered to
   [limit] (zstd_enc.h), for the tests that hold the bytes equal across
   limits. */

CAMLprim value lz_zstd_encode_limit(value limit, value level, value src,
                                    value dst) {
  struct lz_zstd_enc st;
  lz_zstd_enc_init(&st, Int_val(level), false);
  lz_zstd_enc_limit(&st, (uint32_t)Long_val(limit));
  return encode_whole(&zstd_enc, &st, src, dst);
}

CAMLprim value lz_zstd_encoder_limit(value limit, value level) {
  struct lz_zstd_enc *st = caml_stat_alloc(sizeof *st);
  lz_zstd_enc_init(st, Int_val(level), true);
  lz_zstd_enc_limit(st, (uint32_t)Long_val(limit));
  return box(&zstd_enc, st, sizeof *st + lz_zstd_enc_mem(Int_val(level)));
}
