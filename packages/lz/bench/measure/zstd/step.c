/* Zstandard stepped decode timing for bench/measurements.md, "Zstd
   decode", measurements 2 and 3: lz's kernel driven as
   Lz.Zstd.Decoder.step drives it, with input slices of STEP bytes, each
   step writing its run into the decoder's buffer (lib/lz.h), which it
   asks for as stubs.c gives it, best of runs.

   Built as whole.c is, from the tree's root:

     dune build packages/lz/lib/liblz_stubs.a
     gcc -std=c11 -O2 -I packages/lz/lib -o step \
       packages/lz/bench/measure/zstd/step.c _build/default/packages/lz/lib/liblz_stubs.a
     flock ~/wt/TIMING.lock taskset -c 0-5 ./step FILE STEP [SECS]

   The decoder's window limit is 1 GiB, so that every file of the
   measurements decodes. Measurement 2's files are Silesia at level 3 with
   three windows, written by the vendor build under ~/wt/lz-vendors/:

     ~/wt/lz-vendors/zstd-1.5.7/programs/zstd -3 --zstd=wlog=N silesia.raw

   for N in 19, 23, 27; the vendor's streaming decoder on them is
   `zstd -q -d -c --memory=256MB FILE > /dev/null`, timed whole. */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "zstd.h"

void *lz_alloc(size_t n) { return malloc(n); }
void lz_free(void *p) { free(p); }

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static struct lz_zstd_dec st;

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: step FILE STEP [SECS]\n");
    return 2;
  }
  FILE *f = fopen(argv[1], "rb");
  if (f == NULL) return 2;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t *src = malloc(n);
  if (fread(src, 1, n, f) != (size_t)n) return 2;
  fclose(f);
  size_t step = strtoul(argv[2], 0, 10);
  double secs = argc > 3 ? atof(argv[3]) : 2.0;

  /* The decoder's buffer, kept across runs as a decoder keeps it across
     frames: empty at first, a byte that is never NULL, as stubs.c's. */
  static uint8_t empty[1];
  uint8_t *buf = empty;
  size_t cap = 0, pos = 0;

  lz_zstd_dec_init(&st, (uint64_t)1 << 30, true);
  double best = 1e9, end = now() + secs;
  size_t produced = 0;
  int runs = 0;
  while (now() < end || runs < 3) {
    double t0 = now();
    lz_zstd_dec_reset(&st);
    size_t at = 0, out = 0;
    /* Each call is a step: its run begins where the last one ended. A
       call that asks for a buffer gets one holding the run at its start,
       as stubs.c's lz_dec_step does, and the step goes on. */
    for (;;) {
      size_t len = (size_t)n - at < step ? (size_t)n - at : step;
      struct lz_io io = {.src = src + at, .src_len = len, .dst = buf,
                         .dst_cap = cap, .produced = pos, .run = pos};
      struct lz_error err;
      enum lz_status s = lz_zstd_decode(&st, &io, false, &err);
      if (s == LZ_ERROR) {
        printf("error %d at %ld\n", err.kind, (long)err.at);
        return 1;
      }
      at += io.consumed;
      out += io.produced - io.run;
      pos = io.produced;
      if (s == LZ_NEED_BUFFER) {
        uint8_t *b = malloc(io.need);
        memcpy(b, io.dst + io.run, io.produced - io.run);
        if (buf != empty) free(buf);
        buf = b;
        cap = io.need;
        pos = io.produced - io.run;
      }
      if (s == LZ_DONE && at == (size_t)n) break;
    }
    double t = now() - t0;
    if (t < best) best = t;
    produced = out;
    runs++;
  }
  printf("%s in %zu-byte steps: %zu bytes, %.1f MB/s (best of %d)\n",
         argv[1], step, produced, produced / best / 1e6, runs);
  return 0;
}
