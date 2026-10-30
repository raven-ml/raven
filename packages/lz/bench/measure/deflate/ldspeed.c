/* libdeflate 1.26 decoding the same files as speed.c, best of runs, for
   bench/measurements.md. Linked with libdeflate here only, never with lz.

     LD=~/wt/lz-vendors/libdeflate-1.26   # cmake -B build, Release
     gcc -std=c11 -O2 -I $LD -o ldspeed \
       packages/lz/bench/measure/deflate/ldspeed.c $LD/build/libdeflate.a
     flock ~/wt/TIMING.lock taskset -c 0-5 ./ldspeed WRAP FILE RAW [SECS]

   Arguments as speed.c's, whole-buffer only. */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "libdeflate.h"

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static unsigned char *slurp(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) exit(2);
  fseek(f, 0, SEEK_END);
  *n = (size_t)ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char *b = malloc(*n + 1);
  if (fread(b, 1, *n, f) != *n) exit(2);
  fclose(f);
  return b;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: ldspeed WRAP FILE RAW [SECS]\n");
    return 2;
  }
  int wrap = atoi(argv[1]);
  size_t n, rawn, got;
  unsigned char *src = slurp(argv[2], &n), *raw = slurp(argv[3], &rawn);
  double secs = argc > 4 ? atof(argv[4]) : 1.0;
  unsigned char *dst = malloc(rawn + 1);
  struct libdeflate_decompressor *d = libdeflate_alloc_decompressor();
#define RUN()                                                                \
  (wrap == 0   ? libdeflate_deflate_decompress(d, src, n, dst, rawn, &got)   \
   : wrap == 1 ? libdeflate_zlib_decompress(d, src, n, dst, rawn, &got)      \
               : libdeflate_gzip_decompress(d, src, n, dst, rawn, &got))
  if (RUN() != LIBDEFLATE_SUCCESS || got != rawn ||
      memcmp(dst, raw, rawn) != 0) {
    fprintf(stderr, "wrong output\n");
    return 1;
  }
  double best = 1e9, start = now();
  int runs = 0;
  while (now() - start < secs || runs < 3) {
    double t = now();
    RUN();
    t = now() - t;
    if (t < best) best = t;
    runs++;
  }
  printf("%s libdeflate %.1f MB/s (best of %d, %.2f ms)\n", argv[2],
         rawn / best / 1e6, runs, best * 1e3);
  return 0;
}
