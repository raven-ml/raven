/* Zstandard frames (RFC 8878 section 3.1): the stream's stages, frame
   headers, blocks and the window.

   The input is read in units: a magic number, a frame header descriptor,
   the rest of the header, a block header, a block, a checksum. A unit the
   input holds whole is read in place; in a stream, one that arrives in
   pieces is gathered first, in [unit] or, for a block, in [in], on every
   step, the last included. A unit is checked once it is whole, so that
   the same bytes fail with the same error however they are sliced. An
   error is at the first byte of the field it refuses.

   The whole-buffer decode writes each block straight into the
   destination, which is the window. A streaming decoder writes each block
   straight into its buffer (lz.h), where the step's run gathers blocks
   while the next one fits in it:

     buffer  [ frame output | run: blocks ... | ... | earlier segment | ]
              0 after a wrap  [run]   [produced]             [ext_end]

   The buffer holds the window, a run and twice the slack. When a step's
   run would pass its end, the run goes back to the start and the bytes
   before become the earlier segment, still in place behind the new
   output: at least a window of history stays readable while the new
   output and its slack overwrite the segment's start. */

#include "zstd_dec.h"
#include "zstd_dec_paths.h"

enum stage { MAGIC, DESCRIPTOR, HEADER, BLOCK_HEADER, BLOCK, CHECKSUM, END,
             SKIP_SIZE, SKIP };

#define FRAME_MAGIC 0xFD2FB528u
#define BLOCK_RAW 0
#define BLOCK_RLE 1
#define BLOCK_COMPRESSED 2

static const char bad_magic[] = "Magic_Number of no frame";
static const char reserved[] = "reserved Frame_Header_Descriptor bit";
static const char dictionary[] = "Dictionary_ID names a dictionary";
static const char too_wide[] = "Window_Size above the decoder's limit";
static const char bad_type[] = "reserved Block_Type";
static const char too_large[] = "Block_Size above Block_Maximum_Size";
static const char bad_size[] = "Frame_Content_Size not the frame's size";
static const char bad_check[] = "Content_Checksum mismatch";

/* Section 3.1.1 and 3.1.2: [0x28 0xB5 0x2F 0xFD] begins a frame,
   [0x5? 0x2A 0x4D 0x18] a skippable frame. [prefix b n] is the index of
   the first of the [n] bytes [b] that neither can begin with, or [n]. */
static size_t prefix(const uint8_t *b, size_t n) {
  static const uint8_t frame[4] = {0x28, 0xB5, 0x2F, 0xFD};
  static const uint8_t skip[4] = {0x50, 0x2A, 0x4D, 0x18};
  bool f = true, s = true;
  for (size_t i = 0; i < n; i++) {
    f = f && b[i] == frame[i];
    s = s && (i == 0 ? (b[0] & 0xF0) == 0x50 : b[i] == skip[i]);
    if (!f && !s) return i;
  }
  return n;
}

/* Paths */

static const char *const path_names[] = {"generic", "bmi2"};

static size_t path_count(void) {
#if LZ_ZSTD_BMI2
  if (__builtin_cpu_supports("bmi") && __builtin_cpu_supports("bmi2") &&
      __builtin_cpu_supports("lzcnt"))
    return 2;
#endif
  return 1;
}

size_t lz_zstd_dec_paths(const char *const **names) {
  *names = path_names;
  return path_count();
}

void lz_zstd_dec_path(struct lz_zstd_dec *st, size_t i) {
  st->path = i == 1 ? LZ_ZSTD_WITH_BMI2 : LZ_ZSTD_GENERIC;
}

void lz_zstd_dec_init(struct lz_zstd_dec *st, uint64_t window_max,
                      bool stream) {
  st->window_max = window_max;
  st->stream = stream;
  lz_zstd_dec_path(st, path_count() - 1);
  st->defaults = false;
  st->in = NULL;
  st->lits = NULL;
  lz_zstd_dec_reset(st);
}

void lz_zstd_dec_reset(struct lz_zstd_dec *st) {
  st->stage = MAGIC;
  st->total = 0;
  st->have = 0;
  st->fresh = true;
  st->seg = 0;
  st->ext_end = 0;
  st->ext = 0;
}

void lz_zstd_dec_free(struct lz_zstd_dec *st) {
  lz_free(st->in);
  lz_free(st->lits);
  st->in = NULL;
  st->lits = NULL;
}

/* Errors and positions */

/* [here st io] is the stream offset of the next input byte. */
static uint64_t here(const struct lz_zstd_dec *st, const struct lz_io *io) {
  return st->call + io->consumed;
}

/* [fail st err kind pos msg] reports an error at the byte at stream
   offset [pos], relative to the call's input. */
static enum lz_status fail(const struct lz_zstd_dec *st, struct lz_error *err,
                           int kind, uint64_t pos, const char *msg) {
  return lz_fail(err, kind, (int64_t)(pos - st->call), msg);
}

/* [take st io n buf p] points [*p] at the next [n] bytes of the stream:
   in place when the input holds them and none is gathered, else, in a
   stream, gathered into [buf] across calls. It is false while they are
   not all there; a stream has then consumed all the input. The
   whole-buffer decode, whose input is all there is, gathers nothing. */
static bool take(struct lz_zstd_dec *st, struct lz_io *io, size_t n,
                 uint8_t *buf, const uint8_t **p) {
  size_t avail = io->src_len - io->consumed;
  if (st->have == 0 && avail >= n) {
    *p = io->src + io->consumed;
    io->consumed += n;
    return true;
  }
  if (!st->stream) return false;
  size_t k = n - st->have < avail ? n - st->have : avail;
  memcpy(buf + st->have, io->src + io->consumed, k);
  st->have += k;
  io->consumed += k;
  if (st->have < n) return false;
  st->have = 0;
  *p = buf;
  return true;
}

/* The input ran out inside a unit: the end of the stream when [final]. */
static enum lz_status starved(struct lz_io *io, bool final,
                              struct lz_error *err) {
  if (final) return lz_fail(err, LZ_TRUNCATED, (int64_t)io->src_len, NULL);
  return LZ_NEED_INPUT;
}

/* Frames */

/* Section 3.1.1.1.1: the sizes of the fields the descriptor announces. */
static size_t header_size(uint8_t d) {
  static const uint8_t did[4] = {0, 1, 2, 4};
  static const uint8_t fcs[4] = {0, 2, 4, 8};
  bool single = (d >> 5) & 1;
  size_t fcs_size = fcs[d >> 6];
  if (single && fcs_size == 0) fcs_size = 1;
  return (single ? 0 : 1) + did[d & 3] + fcs_size;
}

static uint64_t le(const uint8_t *p, size_t n) {
  uint64_t v = 0;
  for (size_t i = 0; i < n; i++) v |= (uint64_t)p[i] << (8 * i);
  return v;
}

/* Section 3.1.1.1: the header after its descriptor [d], at [h]. */
static enum lz_status frame(struct lz_zstd_dec *st, uint8_t d,
                            const uint8_t *h, struct lz_error *err) {
  static const uint8_t did_sizes[4] = {0, 1, 2, 4};
  uint64_t at = st->frame_at + 5;
  bool single = (d >> 5) & 1;
  uint64_t window = 0;
  if (!single) {
    unsigned exponent = h[0] >> 3, mantissa = h[0] & 7;
    uint64_t base = 1ull << (10 + exponent);
    window = base + (base >> 3) * mantissa;
    if (window > st->window_max)
      return fail(st, err, LZ_UNSUPPORTED, at, too_wide);
    h++;
    at++;
  }
  size_t did = did_sizes[d & 3];
  if (le(h, did) != 0) return fail(st, err, LZ_UNSUPPORTED, at, dictionary);
  h += did;
  at += did;
  size_t fcs = header_size(d) - (single ? 0 : 1) - did;
  st->sized = fcs > 0;
  st->fcs_at = at;
  st->content_size = le(h, fcs) + (fcs == 2 ? 256 : 0);
  if (single) {
    window = st->content_size;
    if (window > st->window_max)
      return fail(st, err, LZ_UNSUPPORTED, at, too_wide);
  }

  st->window = window;
  st->block_max = window < LZ_ZSTD_BLOCK_MAX ? (uint32_t)window
                                             : LZ_ZSTD_BLOCK_MAX;
  st->checked = (d >> 2) & 1;
  st->produced = 0;
  st->rep[0] = 1;
  st->rep[1] = 4;
  st->rep[2] = 8;
  st->huf = 0;
  st->seq_tables = false;
  st->fresh = true;
  if (st->checked) lz_xxh64_init(&st->hash, 0);
  if (!st->stream) return LZ_DONE;

  if (st->in == NULL) st->in = lz_alloc(LZ_ZSTD_BLOCK_MAX);
  if (st->lits == NULL) st->lits = lz_alloc(LZ_ZSTD_BLOCK_MAX + LZ_ZSTD_SLACK);
  if (st->in == NULL || st->lits == NULL)
    return lz_fail(err, LZ_NOMEM, 0, NULL);
  return LZ_DONE;
}

/* [need st] is the buffer a streaming decoder needs for the frame: its
   window, or its content when that is smaller, a run and twice the
   slack. */
static size_t need(const struct lz_zstd_dec *st) {
  uint64_t hist = st->sized && st->content_size < st->window
                      ? st->content_size
                      : st->window;
  return (size_t)hist + LZ_ZSTD_RUN + 2 * LZ_ZSTD_SLACK;
}

/* [place st io] gives the block whose header was read its place: the
   frame's first block sets where the frame's output begins. In a stream,
   it is false when the block might pass the step's run, which then ends,
   holding at least a block's maximum (zstd.h);
   an empty run that leaves less than a run and the slack before the
   buffer's end goes back to its start, and the bytes behind become the
   window's earlier segment. A wrap leaves more than the window and the
   slack behind, since the buffer holds the window, a run and twice the
   slack. */
static bool place(struct lz_zstd_dec *st, struct lz_io *io) {
  if (st->fresh) {
    st->seg = io->produced;
    st->ext = 0;
    st->fresh = false;
  }
  if (!st->stream) return true;
  size_t pos = io->produced;
  size_t size = st->block_type == BLOCK_COMPRESSED ? st->block_max
                                                   : st->block_size;
  if (pos > io->run)
    return pos - io->run + size <= LZ_ZSTD_RUN &&
           pos + size + LZ_ZSTD_SLACK <= io->dst_cap;
  if (pos + LZ_ZSTD_RUN + LZ_ZSTD_SLACK > io->dst_cap) {
    st->ext = pos - st->seg;
    st->ext_end = pos;
    st->seg = 0;
    io->produced = 0;
    io->run = 0;
  }
  return true;
}

/* Blocks */

/* [block st io final out err] reads the block whose header was read and
   writes it at [out->op]; [*n] is its output size. Output the block asks
   for and the destination lacks is Too_long at the block's header. */
static enum lz_status block(struct lz_zstd_dec *st, struct lz_io *io,
                            bool final, const struct lz_zstd_out *out,
                            size_t *n, struct lz_error *err) {
  const uint8_t *p;
  size_t size = st->block_size;
  size_t room = (size_t)(out->end - out->op);

  if (st->block_type == BLOCK_RLE) {
    if (!take(st, io, 1, st->unit, &p)) return starved(io, final, err);
    if (size > room)
      return fail(st, err, LZ_TOO_LONG, st->block_at, NULL);
    memset(out->op, p[0], size);
    *n = size;
    return LZ_DONE;
  }

  if (st->block_type == BLOCK_RAW) {
    uint8_t *buf = st->stream ? out->op : NULL;
    if (!take(st, io, size, buf, &p)) return starved(io, final, err);
    if (size > room)
      return fail(st, err, LZ_TOO_LONG, st->block_at, NULL);
    if (p != out->op) memcpy(out->op, p, size);
    *n = size;
    return LZ_DONE;
  }

  if (!take(st, io, size, st->in, &p)) return starved(io, final, err);
  uint64_t content = here(st, io) - size;
  struct lz_zstd_fail f;
  struct lz_zstd_lits l;
  long k = lz_zstd_literals(st, p, size, out, &l, &f);
  if (k >= 0)
    k = lz_zstd_sequences(st, p + k, size - (size_t)k, (size_t)k, &l, out, &f);
  if (k < 0) {
    if (f.kind == LZ_TOO_LONG)
      return fail(st, err, LZ_TOO_LONG, st->block_at, NULL);
    /* An element with no byte in the block is at the block's last. */
    int64_t at = f.at < (int64_t)size ? f.at : (int64_t)size - 1;
    return fail(st, err, f.kind, content + (uint64_t)at, f.msg);
  }
  *n = (size_t)k;
  return LZ_DONE;
}

/* [fits st total] is whether [total] bytes, the frame's output after the
   current block, keep to the declared content size. */
static bool fits(const struct lz_zstd_dec *st, uint64_t total) {
  return total <= st->content_size && (!st->last || total == st->content_size);
}

/* A block's work is reported before the next block's header (lz.h),
   once there is TELL_MIN of it, so that a stream of small blocks does
   not pay a report for each: [told] is the input taken at the last
   report, [wrote] the output written since. */
#define TELL_MIN 4096

static enum lz_status step(struct lz_zstd_dec *st, struct lz_io *io,
                           bool final, struct lz_error *err) {
  const uint8_t *p;
  size_t told = io->consumed, wrote = 0;
  for (;;) {
    switch (st->stage) {
    case MAGIC: {
      /* Bytes no magic number begins with end the stream: no frame
         begins at the first of them. */
      if (st->have == 0) st->frame_at = here(st, io);
      uint8_t b[4];
      size_t avail = io->src_len - io->consumed;
      size_t n = st->have;
      memcpy(b, st->unit, n);
      size_t k = 4 - n < avail ? 4 - n : avail;
      memcpy(b + n, io->src + io->consumed, k);
      if (prefix(b, n + k) < n + k)
        return fail(st, err, LZ_MALFORMED, st->frame_at, bad_magic);
      if (!take(st, io, 4, st->unit, &p)) return starved(io, final, err);
      st->stage = lz_le32(p) == FRAME_MAGIC ? DESCRIPTOR : SKIP_SIZE;
      break;
    }

    case DESCRIPTOR:
      if (!take(st, io, 1, st->unit + 4, &p)) return starved(io, final, err);
      if (p[0] & 0x08)
        return fail(st, err, LZ_MALFORMED, st->frame_at + 4, reserved);
      st->unit[4] = p[0];
      st->stage = HEADER;
      break;

    case HEADER: {
      uint8_t d = st->unit[4];
      if (!take(st, io, header_size(d), st->unit + 5, &p))
        return starved(io, final, err);
      enum lz_status s = frame(st, d, p, err);
      if (s == LZ_ERROR) return s;
      st->stage = BLOCK_HEADER;
      if (st->stream && io->dst_cap < need(st)) {
        io->need = need(st);
        return LZ_NEED_BUFFER;
      }
      break;
    }

    case BLOCK_HEADER: {
      if (st->have == 0) st->block_at = here(st, io);
      size_t work = io->consumed - told + wrote;
      if (st->have == 0 && work >= TELL_MIN) {
        lz_did(io->work, work);
        told = io->consumed;
        wrote = 0;
      }
      if (!take(st, io, 3, st->unit, &p)) return starved(io, final, err);
      uint32_t h = lz_le24(p);
      st->last = h & 1;
      st->block_type = (h >> 1) & 3;
      st->block_size = h >> 3;
      /* Every field begins in the header's first byte. A raw or RLE
         block states its output: past or short of the content size, it
         makes Frame_Content_Size wrong. */
      if (st->block_type == 3)
        return fail(st, err, LZ_MALFORMED, st->block_at, bad_type);
      if (st->block_size > st->block_max)
        return fail(st, err, LZ_MALFORMED, st->block_at, too_large);
      if (st->sized && st->block_type != BLOCK_COMPRESSED &&
          !fits(st, st->produced + st->block_size))
        return fail(st, err, LZ_MALFORMED, st->fcs_at, bad_size);
      st->placed = false;
      st->stage = BLOCK;
      break;
    }

    case BLOCK: {
      if (!st->placed) {
        if (!place(st, io)) return LZ_NEED_OUTPUT;
        st->placed = true;
      }
      struct lz_zstd_out out = {
          .base = io->dst + st->seg,
          .op = io->dst + io->produced,
          .end = io->dst + io->dst_cap,
          .ext_end = io->dst + st->ext_end,
          .ext = st->ext,
          .window = st->window,
          .block_max = st->block_max,
      };
      size_t n;
      enum lz_status s = block(st, io, final, &out, &n, err);
      if (s != LZ_DONE) return s;
      if (st->checked) lz_xxh64_update(&st->hash, out.op, n);
      st->produced += n;
      io->produced += n;
      wrote += n;
      if (st->sized && !fits(st, st->produced))
        return fail(st, err, LZ_MALFORMED, st->fcs_at, bad_size);
      st->stage = !st->last ? BLOCK_HEADER : st->checked ? CHECKSUM : END;
      break;
    }

    case CHECKSUM:
      if (!take(st, io, 4, st->unit, &p)) return starved(io, final, err);
      if (lz_le32(p) != (uint32_t)lz_xxh64_digest(&st->hash))
        return fail(st, err, LZ_MALFORMED, here(st, io) - 4, bad_check);
      st->stage = END;
      break;

    case END:
      st->stage = MAGIC;
      return LZ_DONE;

    case SKIP_SIZE:
      if (!take(st, io, 4, st->unit, &p)) return starved(io, final, err);
      st->skip = lz_le32(p);
      st->stage = SKIP;
      break;

    case SKIP: {
      size_t avail = io->src_len - io->consumed;
      size_t k = st->skip < avail ? (size_t)st->skip : avail;
      io->consumed += k;
      st->skip -= k;
      if (st->skip > 0) return starved(io, final, err);
      st->stage = END;
      break;
    }
    }
  }
}

enum lz_status lz_zstd_decode(struct lz_zstd_dec *st, struct lz_io *io,
                              bool final, struct lz_error *err) {
  st->call = st->total;
  enum lz_status s = step(st, io, final, err);
  st->total += io->consumed;
  return s;
}
