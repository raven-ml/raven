/* The vendors' side of bench/README.md's table: the rows of
   vendor_table.ml, the same buffers and protocol, timed against
   libdeflate, zlib (for reference, deflate rows only), zstd, lz4 and
   Google's snappy. Never linked into lz; vendor_table.sh builds it
   against the vendor builds under ~/wt/lz-vendors/ and runs it.

     vendor_table [-r ROW] DATA DECODE_RUNS ENCODE_RUNS CORPUS...

   Output as vendor_table.ml's, the implementation column naming the
   vendor. Every row runs single-threaded into a destination allocated
   and faulted in once. [-r ROW] keeps the row named ROW. Zstandard frames carry the XXH64 content checksum
   and the content size, as lz's do; LZ4 frames have independent blocks of
   the smallest maximum size that holds the input, the content size and
   the content checksum, as lz's do. */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <zlib.h>

#include "libdeflate.h"
#include "lz4.h"
#include "lz4frame.h"
#include "snappy-c.h"
#include "zstd.h"

static const char *only;

static int keep(const char *row) { return only == NULL || strcmp(row, only) == 0; }

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static void die(const char *what, const char *row) {
  fprintf(stderr, "%s: %s\n", row, what);
  exit(1);
}

static unsigned char *slurp(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) die("cannot open", path);
  fseek(f, 0, SEEK_END);
  *n = (size_t)ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char *b = malloc(*n + 1);
  if (b == NULL || fread(b, 1, *n, f) != *n) die("cannot read", path);
  fclose(f);
  return b;
}

/* The task a row times: [run] reads [src] and writes [dst], returning the
   bytes written or 0 on failure. */
struct task {
  size_t (*run)(struct task *);
  const unsigned char *src;
  size_t src_len;
  unsigned char *dst;
  size_t dst_cap;
  int level;
  void *state;
};

static double best(struct task *t, int runs) {
  double b = 1e30;
  for (int i = 0; i < runs; i++) {
    double t0 = now();
    if (t->run(t) == 0) die("failed", "timed run");
    double dt = now() - t0;
    if (dt < b) b = dt;
  }
  return b;
}

static void report(const char *corpus, const char *row, const char *impl,
                   size_t n, size_t size, int runs, double t) {
  printf("%s\t%s\t%s\t%.0f\t%zu\t%d\n", corpus, row, impl, n / t / 1e6, size,
         runs);
  fflush(stdout);
}

/* libdeflate */

static size_t ld_raw_dec(struct task *t) {
  size_t out;
  return libdeflate_deflate_decompress(t->state, t->src, t->src_len, t->dst,
                                       t->dst_cap, &out) == LIBDEFLATE_SUCCESS
             ? out
             : 0;
}

static size_t ld_zlib_dec(struct task *t) {
  size_t out;
  return libdeflate_zlib_decompress(t->state, t->src, t->src_len, t->dst,
                                    t->dst_cap, &out) == LIBDEFLATE_SUCCESS
             ? out
             : 0;
}

static size_t ld_gzip_dec(struct task *t) {
  size_t out;
  return libdeflate_gzip_decompress(t->state, t->src, t->src_len, t->dst,
                                    t->dst_cap, &out) == LIBDEFLATE_SUCCESS
             ? out
             : 0;
}

static size_t ld_raw_enc(struct task *t) {
  return libdeflate_deflate_compress(t->state, t->src, t->src_len, t->dst,
                                     t->dst_cap);
}

static size_t ld_zlib_enc(struct task *t) {
  return libdeflate_zlib_compress(t->state, t->src, t->src_len, t->dst,
                                  t->dst_cap);
}

static size_t ld_gzip_enc(struct task *t) {
  return libdeflate_gzip_compress(t->state, t->src, t->src_len, t->dst,
                                  t->dst_cap);
}

/* zlib: a stream reset per run, raw (window bits -15), zlib (15) or gzip
   (31) by [level] for a decoder. */

static size_t z_dec(struct task *t) {
  z_stream *z = t->state;
  if (inflateReset(z) != Z_OK) return 0;
  z->next_in = (unsigned char *)t->src;
  z->avail_in = (uInt)t->src_len;
  z->next_out = t->dst;
  z->avail_out = (uInt)t->dst_cap;
  return inflate(z, Z_FINISH) == Z_STREAM_END ? z->total_out : 0;
}

static size_t z_enc(struct task *t) {
  z_stream *z = t->state;
  if (deflateReset(z) != Z_OK) return 0;
  z->next_in = (unsigned char *)t->src;
  z->avail_in = (uInt)t->src_len;
  z->next_out = t->dst;
  z->avail_out = (uInt)t->dst_cap;
  return deflate(z, Z_FINISH) == Z_STREAM_END ? z->total_out : 0;
}

/* zstd */

static size_t zstd_dec(struct task *t) {
  size_t r =
      ZSTD_decompressDCtx(t->state, t->dst, t->dst_cap, t->src, t->src_len);
  return ZSTD_isError(r) ? 0 : r;
}

static size_t zstd_enc(struct task *t) {
  ZSTD_CCtx *c = t->state;
  ZSTD_CCtx_reset(c, ZSTD_reset_session_and_parameters);
  ZSTD_CCtx_setParameter(c, ZSTD_c_compressionLevel, t->level);
  ZSTD_CCtx_setParameter(c, ZSTD_c_checksumFlag, 1);
  ZSTD_CCtx_setParameter(c, ZSTD_c_contentSizeFlag, 1);
  ZSTD_CCtx_setParameter(c, ZSTD_c_nbWorkers, 0);
  size_t r = ZSTD_compress2(c, t->dst, t->dst_cap, t->src, t->src_len);
  return ZSTD_isError(r) ? 0 : r;
}

/* lz4 */

static size_t lz4b_dec(struct task *t) {
  int r = LZ4_decompress_safe((const char *)t->src, (char *)t->dst,
                              (int)t->src_len, (int)t->dst_cap);
  return r < 0 ? 0 : (size_t)r;
}

static size_t lz4b_enc(struct task *t) {
  return (size_t)LZ4_compress_fast_extState(
      t->state, (const char *)t->src, (char *)t->dst, (int)t->src_len,
      (int)t->dst_cap, 1);
}

static size_t lz4f_dec(struct task *t) {
  LZ4F_dctx *d = t->state;
  LZ4F_resetDecompressionContext(d);
  LZ4F_decompressOptions_t o = {.stableDst = 1};
  size_t in = t->src_len, out = t->dst_cap;
  size_t r = LZ4F_decompress(d, t->dst, &out, t->src, &in, &o);
  return r != 0 || in != t->src_len ? 0 : out;
}

static LZ4F_blockSizeID_t lz4f_block(size_t n) {
  if (n <= 64 << 10) return LZ4F_max64KB;
  if (n <= 256 << 10) return LZ4F_max256KB;
  if (n <= 1 << 20) return LZ4F_max1MB;
  return LZ4F_max4MB;
}

static size_t lz4f_enc(struct task *t) {
  LZ4F_preferences_t p = {0};
  p.frameInfo.blockSizeID = lz4f_block(t->src_len);
  p.frameInfo.blockMode = LZ4F_blockIndependent;
  p.frameInfo.contentChecksumFlag = LZ4F_contentChecksumEnabled;
  p.frameInfo.contentSize = t->src_len;
  size_t r = LZ4F_compressFrame_usingCDict(t->state, t->dst, t->dst_cap,
                                           t->src, t->src_len, NULL, &p);
  return LZ4F_isError(r) ? 0 : r;
}

/* snappy */

static size_t snappy_dec(struct task *t) {
  size_t out = t->dst_cap;
  return snappy_uncompress((const char *)t->src, t->src_len, (char *)t->dst,
                           &out) == SNAPPY_OK
             ? out
             : 0;
}

static size_t snappy_enc(struct task *t) {
  size_t out = t->dst_cap;
  return snappy_compress((const char *)t->src, t->src_len, (char *)t->dst,
                         &out) == SNAPPY_OK
             ? out
             : 0;
}

/* CRC-32 returns 1: it writes nothing. */

static size_t ld_crc(struct task *t) {
  t->level = (int)libdeflate_crc32(0, t->src, t->src_len);
  return 1;
}

static size_t z_crc(struct task *t) {
  t->level = (int)crc32(0, t->src, (uInt)t->src_len);
  return 1;
}

/* Rows */

#define CRC_RUNS 50

struct ctx {
  const char *data, *corpus;
  const unsigned char *raw;
  size_t n;
  unsigned char *buf, *back;
  size_t cap;
  int drun, erun;
  struct libdeflate_decompressor *ld;
  z_stream *zd[3];
  ZSTD_DCtx *zdc;
  ZSTD_CCtx *zcc;
  LZ4F_dctx *lzd;
  LZ4F_cctx *lzc;
  void *lz4_state;
};

static void decode(struct ctx *c, const char *suffix, const char *row,
                   const char *impl, size_t (*run)(struct task *),
                   void *state) {
  if (!keep(row)) return;
  char path[4096];
  snprintf(path, sizeof path, "%s/%s.%s", c->data, c->corpus, suffix);
  size_t len;
  unsigned char *src = slurp(path, &len);
  struct task t = {run, src, len, c->buf, c->n, 0, state};
  if (run(&t) != c->n || memcmp(c->buf, c->raw, c->n) != 0)
    die("decodes wrong bytes", row);
  report(c->corpus, row, impl, c->n, 0, c->drun, best(&t, c->drun));
  free(src);
}

/* [check] decodes an encoder's output with the vendor's decoder of its
   format. */
static void encode(struct ctx *c, const char *row, const char *impl,
                   size_t (*run)(struct task *), int level, void *state,
                   size_t (*check)(struct task *), void *check_state) {
  if (!keep(row)) return;
  struct task t = {run, c->raw, c->n, c->buf, c->cap, level, state};
  size_t size = run(&t);
  if (size == 0) die("encode failed", row);
  struct task d = {check, c->buf, size, c->back, c->n, 0, check_state};
  if (check(&d) != c->n || memcmp(c->back, c->raw, c->n) != 0)
    die("does not decode to the input", row);
  report(c->corpus, row, impl, c->n, size, c->erun, best(&t, c->erun));
}

static void corpus(struct ctx *c) {
  char path[4096], row[64];
  snprintf(path, sizeof path, "%s/%s.raw", c->data, c->corpus);
  unsigned char *raw = slurp(path, &c->n);
  c->raw = raw;
  size_t n = c->n;
  c->cap = n + n / 4 + (1 << 20); /* above every bound, snappy's n/6 */
  c->buf = malloc(c->cap);
  c->back = malloc(n);
  if (c->buf == NULL || c->back == NULL) die("out of memory", c->corpus);
  memset(c->buf, 1, c->cap);
  memset(c->back, 1, n);

  decode(c, "deflate6", "decode deflate", "libdeflate", ld_raw_dec, c->ld);
  decode(c, "deflate6", "decode deflate", "zlib", z_dec, c->zd[0]);
  decode(c, "zlib6", "decode zlib", "libdeflate", ld_zlib_dec, c->ld);
  decode(c, "zlib6", "decode zlib", "zlib", z_dec, c->zd[1]);
  decode(c, "gz9", "decode gzip", "libdeflate", ld_gzip_dec, c->ld);
  decode(c, "gz9", "decode gzip", "zlib", z_dec, c->zd[2]);
  const char *zst[] = {"1", "3", "12", "19"};
  for (int i = 0; i < 4; i++) {
    char suffix[16];
    snprintf(suffix, sizeof suffix, "zst%s", zst[i]);
    snprintf(row, sizeof row, "decode zstd %s", zst[i]);
    decode(c, suffix, row, "zstd", zstd_dec, c->zdc);
  }
  decode(c, "lz4b", "decode lz4 block", "lz4", lz4b_dec, NULL);
  decode(c, "lz4f", "decode lz4 frame", "lz4", lz4f_dec, c->lzd);
  decode(c, "snappy", "decode snappy", "snappy", snappy_dec, NULL);

  for (int level = 1; level <= 9; level++) {
    struct libdeflate_compressor *ld = libdeflate_alloc_compressor(level);
    snprintf(row, sizeof row, "encode deflate %d", level);
    encode(c, row, "libdeflate", ld_raw_enc, level, ld, ld_raw_dec, c->ld);
    if (keep(row)) {
      z_stream z = {0};
      if (deflateInit2(&z, level, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY) !=
          Z_OK)
        die("deflateInit2", row);
      encode(c, row, "zlib", z_enc, level, &z, ld_raw_dec, c->ld);
      deflateEnd(&z);
    }
    snprintf(row, sizeof row, "encode zlib %d", level);
    encode(c, row, "libdeflate", ld_zlib_enc, level, ld, ld_zlib_dec, c->ld);
    snprintf(row, sizeof row, "encode gzip %d", level);
    encode(c, row, "libdeflate", ld_gzip_enc, level, ld, ld_gzip_dec, c->ld);
    libdeflate_free_compressor(ld);
  }
  for (int level = 1; level <= 12; level++) {
    snprintf(row, sizeof row, "encode zstd %d", level);
    encode(c, row, "zstd", zstd_enc, level, c->zcc, zstd_dec, c->zdc);
  }
  encode(c, "encode lz4 block", "lz4", lz4b_enc, 0, c->lz4_state, lz4b_dec,
         NULL);
  encode(c, "encode lz4 frame", "lz4", lz4f_enc, 0, c->lzc, lz4f_dec, c->lzd);
  encode(c, "encode snappy", "snappy", snappy_enc, 0, NULL, snappy_dec, NULL);

  if (keep("crc32")) {
    struct task t = {ld_crc, raw, n, NULL, 0, 0, NULL};
    report(c->corpus, "crc32", "libdeflate", n, 0, CRC_RUNS,
           best(&t, CRC_RUNS));
    t.run = z_crc;
    report(c->corpus, "crc32", "zlib", n, 0, CRC_RUNS, best(&t, CRC_RUNS));
  }
  free(raw);
  free(c->buf);
  free(c->back);
}

static void usage(void) {
  fprintf(stderr,
          "usage: vendor_table [-r ROW] DATA DECODE_RUNS ENCODE_RUNS "
          "CORPUS...\n");
  exit(2);
}

int main(int argc, char **argv) {
  int a = 1;
  if (argc > 2 && strcmp(argv[1], "-r") == 0) {
    only = argv[2];
    a = 3;
  }
  if (argc - a < 4) usage();
  struct ctx c = {0};
  c.data = argv[a];
  c.drun = atoi(argv[a + 1]);
  c.erun = atoi(argv[a + 2]);
  if (c.drun <= 0 || c.erun <= 0) usage();

  c.ld = libdeflate_alloc_decompressor();
  int bits[3] = {-15, 15, 31};
  for (int i = 0; i < 3; i++) {
    c.zd[i] = calloc(1, sizeof(z_stream));
    if (c.zd[i] == NULL || inflateInit2(c.zd[i], bits[i]) != Z_OK)
      die("inflateInit2", "zlib");
  }
  c.zdc = ZSTD_createDCtx();
  c.zcc = ZSTD_createCCtx();
  if (LZ4F_isError(LZ4F_createDecompressionContext(&c.lzd, LZ4F_VERSION)) ||
      LZ4F_isError(LZ4F_createCompressionContext(&c.lzc, LZ4F_VERSION)))
    die("context", "lz4");
  c.lz4_state = malloc((size_t)LZ4_sizeofState());
  if (c.ld == NULL || c.zdc == NULL || c.zcc == NULL || c.lz4_state == NULL)
    die("out of memory", "vendors");

  for (int i = a + 3; i < argc; i++) {
    c.corpus = argv[i];
    corpus(&c);
  }
  return 0;
}
