/* Zstandard whole-buffer decode timing for bench/measurements.md, "Zstd
   decode": lz's kernel over every frame of a file, as stubs.c's
   whole-buffer driver runs it, best of runs.

   Built against the kernels dune compiles (lz's flags, -O3), from the
   tree's root:

     dune build packages/lz/lib/liblz_stubs.a
     gcc -std=c11 -O2 -I packages/lz/lib -o whole \
       packages/lz/bench/measure/zstd/whole.c _build/default/packages/lz/lib/liblz_stubs.a
     flock ~/wt/TIMING.lock taskset -c 0-5 ./whole FILE CAP [SECS]

   CAP is the destination's size (at least the decoded size), SECS the time
   to fill with runs (default 2). The vendor's side is zstd's own driver on
   the same file, from the builds under ~/wt/lz-vendors/:

     ~/wt/lz-vendors/zstd-1.5.7/programs/zstd -b -d -i2 FILE
     ~/wt/lz-vendors/zstd-1.5.7-noasm/programs/zstd -b -d -i2 FILE

   (zstd-1.5.7-noasm is `make -C programs zstd-release
   MOREFLAGS=-DZSTD_DISABLE_ASM`.) Measurement 10's "without" column is
   this program built with a copy of lib/zstd_decode.c whose checksum
   update and comparison are removed:

     sed 's|      if (st->checked) lz_xxh64_update(&st->hash, out.op, n);|      (void)0;|; s|      if (lz_le32(p) != (uint32_t)lz_xxh64_digest(\&st->hash))|      if (0)|' \
       packages/lz/lib/zstd_decode.c > /tmp/zstd_decode_nohash.c
     gcc -std=c11 -O3 -I packages/lz/lib -o whole_nohash \
       packages/lz/bench/measure/zstd/whole.c /tmp/zstd_decode_nohash.c \
       packages/lz/lib/zstd_dec_fse.c packages/lz/lib/zstd_dec_lit.c \
       packages/lz/lib/zstd_dec_seq.c _build/default/packages/lz/lib/liblz_stubs.a

   The program defines lz_alloc itself, so the archive's OCaml stubs are
   not linked. */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
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
    fprintf(stderr, "usage: whole FILE CAP [SECS]\n");
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
  size_t cap = strtoul(argv[2], 0, 10);
  double secs = argc > 3 ? atof(argv[3]) : 2.0;
  uint8_t *dst = malloc(cap);

  double best = 1e9, end = now() + secs;
  size_t produced = 0;
  int runs = 0;
  while (now() < end || runs < 3) {
    double t0 = now();
    lz_zstd_dec_init(&st, UINT64_MAX, false);
    struct lz_io io = {src, (size_t)n, dst, cap, 0, 0};
    struct lz_error err;
    /* One call per frame, as stubs.c's whole() makes them. */
    while (io.consumed < io.src_len) {
      struct lz_io m = {io.src + io.consumed, io.src_len - io.consumed,
                        io.dst + io.produced, io.dst_cap - io.produced, 0, 0};
      if (lz_zstd_decode(&st, &m, true, &err) != LZ_DONE) {
        printf("error %d at %ld\n", err.kind, (long)err.at);
        return 1;
      }
      io.consumed += m.consumed;
      io.produced += m.produced;
    }
    double t = now() - t0;
    if (t < best) best = t;
    produced = io.produced;
    runs++;
  }
  printf("%s: %ld -> %zu bytes, %.1f MB/s (best of %d)\n", argv[1], n,
         produced, produced / best / 1e6, runs);
  return 0;
}
