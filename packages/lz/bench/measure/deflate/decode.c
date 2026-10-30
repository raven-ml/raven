/* One whole-buffer decode for zlib_fuzz.py: decodes FILE into a
   destination of CAP bytes, writes the output to stdout, and prints
   "consumed C produced P" or the error ("kind K at A: MESSAGE") to stderr.
   Build as speed.c:

     gcc -std=c11 -O2 -I packages/lz/lib -o decode \
       packages/lz/bench/measure/deflate/decode.c _build/default/packages/lz/lib/liblz_stubs.a
     ./decode WRAP FILE CAP */

#include <stdio.h>
#include <stdlib.h>

#include "deflate.h"

void *lz_alloc(size_t n) { return malloc(n); }
void lz_free(void *p) { free(p); }

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: decode WRAP FILE CAP\n");
    return 2;
  }
  FILE *f = fopen(argv[2], "rb");
  if (f == NULL) return 2;
  fseek(f, 0, SEEK_END);
  size_t n = (size_t)ftell(f), cap = (size_t)atol(argv[3]);
  fseek(f, 0, SEEK_SET);
  uint8_t *src = malloc(n + 1), *dst = malloc(cap + 1);
  if (fread(src, 1, n, f) != n) return 2;
  fclose(f);

  /* The stubs' whole-buffer loop, as lz.h describes it. */
  static struct lz_deflate_dec st;
  struct lz_error e = {0, 0, NULL};
  size_t in = 0, out = 0;
  lz_deflate_dec_init(&st, atoi(argv[1]), false);
  for (;;) {
    struct lz_io m = {src + in, n - in, dst + out, cap - out, 0, 0};
    enum lz_status r = lz_deflate_decode(&st, &m, true, &e);
    size_t base = in;
    in += m.consumed;
    out += m.produced;
    if (r == LZ_NEED_OUTPUT)
      r = lz_fail(&e, LZ_TOO_LONG, (int64_t)m.consumed, NULL);
    if (r == LZ_ERROR) {
      e.at += (int64_t)base;
      fprintf(stderr, "kind %d at %lld: %s\n", e.kind, (long long)e.at,
              e.msg ? e.msg : "");
      return 1;
    }
    if (atoi(argv[1]) != LZ_GZIP || in == n) break;
  }
  fwrite(dst, 1, out, stdout);
  fprintf(stderr, "consumed %zu produced %zu\n", in, out);
  return 0;
}
