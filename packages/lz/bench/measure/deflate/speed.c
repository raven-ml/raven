/* Deflate decode timing for bench/measurements.md, "Deflate decode": lz's
   decoder on a file, whole-buffer or in steps, best of runs.

   Built against the kernels dune compiles (lz's flags, -O3), from the
   tree's root:

     dune build packages/lz/lib/liblz_stubs.a
     gcc -std=c11 -O2 -I packages/lz/lib -o speed \
       packages/lz/bench/measure/deflate/speed.c _build/default/packages/lz/lib/liblz_stubs.a
     flock ~/wt/TIMING.lock taskset -c 0-5 ./speed WRAP FILE RAW [STEP [SECS]]

   WRAP is 0 (raw deflate), 1 (zlib) or 2 (gzip); RAW the file FILE
   decodes to, which a whole-buffer run checks; STEP 0 decodes the whole
   buffer, otherwise steps of STEP input bytes, each writing its run into
   the decoder's buffer (lib/lz.h), which it asks for as stubs.c gives it;
   SECS the time to fill with runs (default 1). The program defines
   lz_alloc itself, so the archive's OCaml stubs are not linked.

   libdeflate's side is ldspeed.c. The "window copy" rows timed the design
   before the lent buffer, a copy of lib/deflate_decode.c whose keep() had
   its two memcpy calls removed. */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "deflate.h"

void *lz_alloc(size_t n) { return malloc(n); }
void lz_free(void *p) { free(p); }

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static uint8_t *slurp(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) exit(2);
  fseek(f, 0, SEEK_END);
  *n = (size_t)ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t *b = malloc(*n + 1);
  if (fread(b, 1, *n, f) != *n) exit(2);
  fclose(f);
  return b;
}

static struct lz_deflate_dec st;

/* The stubs' whole-buffer loop: for gzip, a member wherever bytes
   remain. */
static size_t whole(int wrap, const uint8_t *src, size_t n, uint8_t *dst,
                    size_t cap) {
  struct lz_error e;
  size_t in = 0, out = 0;
  lz_deflate_dec_init(&st, wrap, false);
  for (;;) {
    struct lz_io m = {.src = src + in,
                      .src_len = n - in,
                      .dst = dst + out,
                      .dst_cap = cap - out};
    enum lz_status r = lz_deflate_decode(&st, &m, true, &e);
    in += m.consumed;
    out += m.produced;
    if (r != LZ_DONE) {
      fprintf(stderr, "status %d: %s at %lld\n", r, e.msg ? e.msg : "",
              (long long)e.at);
      exit(1);
    }
    if (wrap != LZ_GZIP || in == n) return out;
  }
}

/* The decoder's buffer, kept across runs as a decoder keeps it across
   steps: empty at first, a byte that is never NULL, as stubs.c's. */
static uint8_t empty[1];
static uint8_t *buf = empty;
static size_t buf_cap;

/* Each call is a step: its run begins where the last one ended. A call
   that asks for a buffer gets one holding the run at its start, as
   stubs.c's lz_dec_step does, and the step goes on. */
static size_t stepped(int wrap, const uint8_t *src, size_t n, size_t step) {
  struct lz_error e;
  size_t in = 0, out = 0, pos = 0;
  lz_deflate_dec_init(&st, wrap, true);
  for (;;) {
    size_t len = n - in < step ? n - in : step;
    struct lz_io io = {.src = src + in, .src_len = len, .dst = buf,
                       .dst_cap = buf_cap, .produced = pos, .run = pos};
    enum lz_status r = lz_deflate_decode(&st, &io, false, &e);
    if (r == LZ_ERROR) {
      fprintf(stderr, "%s at %lld\n", e.msg ? e.msg : "", (long long)e.at);
      exit(1);
    }
    in += io.consumed;
    out += io.produced - io.run;
    pos = io.produced;
    if (r == LZ_NEED_BUFFER) {
      uint8_t *b = malloc(io.need);
      memcpy(b, io.dst + io.run, io.produced - io.run);
      if (buf != empty) free(buf);
      buf = b;
      buf_cap = io.need;
      pos = io.produced - io.run;
    }
    if (r == LZ_DONE && in == n) {
      lz_deflate_dec_free(&st);
      return out;
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: speed WRAP FILE RAW [STEP [SECS]]\n");
    return 2;
  }
  int wrap = atoi(argv[1]);
  size_t n, rawn;
  uint8_t *src = slurp(argv[2], &n), *raw = slurp(argv[3], &rawn);
  size_t step = argc > 4 ? (size_t)atol(argv[4]) : 0;
  double secs = argc > 5 ? atof(argv[5]) : 1.0;
  uint8_t *dst = malloc(rawn + 1);
  size_t got = step ? stepped(wrap, src, n, step)
                    : whole(wrap, src, n, dst, rawn);
  if (got != rawn || (!step && memcmp(dst, raw, rawn) != 0)) {
    fprintf(stderr, "wrong output\n");
    return 1;
  }
  double best = 1e9, start = now();
  int runs = 0;
  while (now() - start < secs || runs < 3) {
    double t = now();
    if (step) stepped(wrap, src, n, step);
    else whole(wrap, src, n, dst, rawn);
    t = now() - t;
    if (t < best) best = t;
    runs++;
  }
  printf("%s %s %.1f MB/s (best of %d, %.2f ms)\n", argv[2],
         step ? argv[4] : "whole", rawn / best / 1e6, runs, best * 1e3);
  return 0;
}
