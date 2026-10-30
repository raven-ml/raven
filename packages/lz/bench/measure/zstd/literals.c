/* Zstandard literals timing for bench/measurements.md, "Zstd decode",
   measurement 8: the literals section of a frame's first block, decoded
   as zstd's tests/fullbench -b21 decodes it, in MB/s of the block's
   128 KiB.

   Built against the kernels dune compiles, from the tree's root; it
   reaches the kernel's internal entry through lib/zstd_dec.h:

     dune build packages/lz/lib/liblz_stubs.a
     gcc -std=c11 -O2 -I packages/lz/lib -o literals \
       packages/lz/bench/measure/zstd/literals.c _build/default/packages/lz/lib/liblz_stubs.a
     head -c 10000000 silesia.raw > silesia10m.raw
     ~/wt/lz-vendors/zstd-1.5.7/programs/zstd -3 silesia10m.raw -o first.zst
     flock ~/wt/TIMING.lock taskset -c 0-5 ./literals first.zst [SECS]

   The vendor's side, on the same first block (fullbench compresses its
   input at the level given, then times ZSTD_decodeLiteralsBlock on the
   first block), from the builds under ~/wt/lz-vendors/ (`make -C tests
   fullbench`, with MOREFLAGS=-DZSTD_DISABLE_ASM for the C path):

     ~/wt/lz-vendors/zstd-1.5.7/tests/fullbench -b21 -l3 -i4 silesia10m.raw
     ~/wt/lz-vendors/zstd-1.5.7-noasm/tests/fullbench -b21 -l3 -i4 silesia10m.raw

   The literals are decoded into a destination with the room a
   whole-buffer decode has mid-file, so they land past the block's
   output, as there. */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "zstd_dec.h"

void *lz_alloc(size_t n) { return malloc(n); }
void lz_free(void *p) { free(p); }

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static struct lz_zstd_dec st;

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: literals FILE [SECS]\n");
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

  /* RFC 8878 section 3.1.1.1: skip the frame header to the first block. */
  static const int did[4] = {0, 1, 2, 4}, fcs[4] = {0, 2, 4, 8};
  uint8_t d = src[4];
  int single = (d >> 5) & 1, fs = fcs[d >> 6];
  if (single && fs == 0) fs = 1;
  size_t h = 5 + (single ? 0 : 1) + did[d & 3] + fs;
  uint32_t header = src[h] | src[h + 1] << 8 | (uint32_t)src[h + 2] << 16;
  if (((header >> 1) & 3) != 2) {
    fprintf(stderr, "the first block is not compressed\n");
    return 2;
  }
  size_t size = header >> 3;
  const uint8_t *block = src + h + 3;

  size_t cap = 4 * LZ_ZSTD_BLOCK_MAX;
  uint8_t *dst = malloc(cap);
  struct lz_zstd_out o = {dst, dst, dst + cap, NULL, 0, UINT64_MAX,
                          LZ_ZSTD_BLOCK_MAX};
  lz_zstd_dec_init(&st, UINT64_MAX, false);
  st.block_max = LZ_ZSTD_BLOCK_MAX;

  double best = 1e9, end = now() + (argc > 2 ? atof(argv[2]) : 2.0);
  int runs = 0;
  struct lz_zstd_lits l;
  while (now() < end || runs < 100) {
    double t0 = now();
    for (int k = 0; k < 100; k++) {
      struct lz_zstd_fail fail;
      st.huf = 0;
      if (lz_zstd_literals(&st, block, size, &o, &l, &fail) < 0) {
        printf("error %d at %lld\n", fail.kind, (long long)fail.at);
        return 1;
      }
    }
    double t = (now() - t0) / 100;
    if (t < best) best = t;
    runs++;
  }
  printf("%s: first block %zu bytes, %zu literals, %.1f MB/s of 128 KiB, "
         "%.1f MB/s of literals\n",
         argv[1], size, l.len, 131072 / best / 1e6, l.len / best / 1e6);
  return 0;
}
