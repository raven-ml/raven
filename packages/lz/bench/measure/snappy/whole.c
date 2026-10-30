/* Snappy whole-buffer timing for bench/measurements.md, "Snappy": lz's
   kernels on a raw corpus and on a block of it, best of RUNS, in MB/s of
   the raw size; the same command line and report as the vendor harness
   packages/lz/test/gen/snappy_ref.cc, which times Google's snappy.

   Built against the kernels dune compiles (lz's flags, -O3), from the
   tree's root:

     dune build packages/lz/lib/liblz_stubs.a
     gcc -std=c11 -O2 -I packages/lz/lib -o whole \
       packages/lz/bench/measure/snappy/whole.c \
       _build/default/packages/lz/lib/liblz_stubs.a
     flock ~/wt/TIMING.lock taskset -c 0-5 \
       ./whole bench packages/lz/bench/data/text.raw packages/lz/bench/data/text.snappy 10

   compare.sh runs it beside the vendor builds of ~/wt/lz-vendors/ on every
   corpus, with the decoder built without its pattern expansion
   (measurement 11). [whole enc RAW OUT] writes lz's block of RAW, for a
   byte comparison with [snappy_ref compress]. */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "snappy.h"

/* The kernels allocate through these; the stubs' versions need the OCaml
   runtime. */
void *lz_alloc(size_t n) { return malloc(n); }
void lz_free(void *p) { free(p); }

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static uint8_t *slurp(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    perror(path);
    exit(2);
  }
  fseek(f, 0, SEEK_END);
  *n = (size_t)ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t *b = malloc(*n ? *n : 1);
  if (fread(b, 1, *n, f) != *n) exit(2);
  fclose(f);
  return b;
}

static int bench(const char *raw_path, const char *z_path, int runs) {
  size_t rn, zn;
  uint8_t *raw = slurp(raw_path, &rn), *z = slurp(z_path, &zn);
  size_t cap = lz_snappy_bound(rn);
  uint8_t *out = malloc(cap), *back = malloc(rn ? rn : 1);
  double best_enc = 1e30, best_dec = 1e30;
  size_t zlen = 0;
  for (int i = 0; i < runs; i++) {
    struct lz_io io = {raw, rn, out, cap, 0, 0};
    double t = now();
    if (lz_snappy_encode(&io) != LZ_DONE) return 1;
    t = now() - t;
    zlen = io.produced;
    if (t < best_enc) best_enc = t;
  }
  for (int i = 0; i < runs; i++) {
    struct lz_io io = {z, zn, back, rn, 0, 0};
    struct lz_error e;
    double t = now();
    if (lz_snappy_decode(&io, &e) != LZ_DONE) return 1;
    t = now() - t;
    if (t < best_dec) best_dec = t;
  }
  if (memcmp(back, raw, rn) != 0) {
    fprintf(stderr, "decode differs\n");
    return 1;
  }
  printf("%s: encode %.0f MB/s (ratio %.3f), decode %.0f MB/s\n", raw_path,
         rn / 1e6 / best_enc, (double)rn / zlen, rn / 1e6 / best_dec);
  return 0;
}

static int enc(const char *raw_path, const char *out_path) {
  size_t n;
  uint8_t *raw = slurp(raw_path, &n);
  size_t cap = lz_snappy_bound(n);
  uint8_t *out = malloc(cap);
  struct lz_io io = {raw, n, out, cap, 0, 0};
  if (lz_snappy_encode(&io) != LZ_DONE) return 1;
  FILE *f = fopen(out_path, "wb");
  if (!f || fwrite(out, 1, io.produced, f) != io.produced) return 2;
  fclose(f);
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 5 && strcmp(argv[1], "bench") == 0)
    return bench(argv[2], argv[3], atoi(argv[4]));
  if (argc == 4 && strcmp(argv[1], "enc") == 0) return enc(argv[2], argv[3]);
  fprintf(stderr, "usage: whole bench RAW BLOCK RUNS | whole enc RAW OUT\n");
  return 2;
}
