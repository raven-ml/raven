/* §7.7: CRC-32 kernels as a standalone x86-64 microbenchmark, in GB/s at
   4 KiB, 64 KiB and 8 MiB. These are measured variants, not lz's code.

   - bytewise: one 256-entry table lookup per byte.
   - slice-by-8: eight tables, eight bytes per step.
   - crc32 instruction: SSE4.2's crc32 computes CRC-32C (Castagnoli,
     0x1EDC6F41), not gzip's CRC-32 (0x04C11DB7); it is timed only as a
     speed reference, as one dependent chain and as three independent chains
     (the latter is not one CRC: combining three chains costs a carry-less
     multiply per chain, left out).
   - PCLMULQDQ folding with 4, 8 and 12 xmm accumulators, and VPCLMULQDQ on
     ymm (AVX2 width) with 4 and 8 accumulators.

   Folding. The CRC register of a message M is M(x) x^32 mod G(x), with the
   register's initial value XORed into M's first 32 bits. Bits are
   reflected: bit j of a little-endian 64-bit lane is the coefficient of
   x^(63-j) of that lane. A carry-less multiply of a 64-bit lane L by a
   32-bit reflected constant K yields, read as a 128-bit reflected value,
   L K x^33. Folding a 128-bit accumulator A = L x^64 + H forward by n
   blocks of 128 bits needs A x^(128 n) mod G, which is
     clmul(L, x^(128n+31) mod G) + clmul(H, x^(128n-33) mod G),
   a 128-bit value that is XORed into the block n ahead. The constants are
   computed below from G; no table is copied.

   Reduction of the last 128 bits to 32: two folds by x^63 mod G bring 128
   bits to 96 and 96 to 64 (the multiply's x^33 offset accounted as above),
   then Barrett's reduction of V(x) x^32 with mu = floor(x^96 / G):
     q = V + floor(V (mu - x^64) / x^64),  R = (q (G - x^32)) mod x^32.

   Build: gcc -O3 -mpclmul -msse4.2 -mavx2 -mvpclmulqdq (dune does this).
   Run: crc32_bench.exe, under the timing lock. */

#define _POSIX_C_SOURCE 200809L

#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define POLY_NORMAL 0x104C11DB7ull /* G(x) with its x^32 term */
#define POLY_REFLECTED 0xEDB88320u /* G(x) - x^32, reflected */
#define POLY_C_REFLECTED 0x82F63B78u /* CRC-32C, reflected */
#define MAX_FOLD 24

static uint32_t table[8][256];
static uint32_t table_c[256];

/* Fold constants: lane 0 multiplies the accumulator's low lane (the earlier
   bytes), lane 1 its high lane. */
static __m128i fold_k[MAX_FOLD + 1];
static uint64_t k63;    /* x^63 mod G, reflected */
static uint64_t mu_rep; /* floor(x^96 / G) - x^64, reflected over 64 bits */

static uint32_t rev32(uint32_t x) {
  uint32_t r = 0;
  for (int i = 0; i < 32; i++) r |= ((x >> i) & 1u) << (31 - i);
  return r;
}

static uint64_t rev64(uint64_t x) {
  uint64_t r = 0;
  for (int i = 0; i < 64; i++) r |= ((x >> i) & 1u) << (63 - i);
  return r;
}

/* x^n mod G, normal bit order. */
static uint32_t xpow_mod(unsigned n) {
  uint64_t r = 1;
  for (unsigned i = 0; i < n; i++) {
    r <<= 1;
    if (r >> 32) r ^= POLY_NORMAL;
  }
  return (uint32_t)r;
}

static void init(void) {
  for (uint32_t b = 0; b < 256; b++) {
    uint32_t c = b, d = b;
    for (int k = 0; k < 8; k++) {
      c = c & 1 ? (c >> 1) ^ POLY_REFLECTED : c >> 1;
      d = d & 1 ? (d >> 1) ^ POLY_C_REFLECTED : d >> 1;
    }
    table[0][b] = c;
    table_c[b] = d;
  }
  for (int t = 1; t < 8; t++)
    for (int b = 0; b < 256; b++)
      table[t][b] = (table[t - 1][b] >> 8) ^ table[0][table[t - 1][b] & 0xFF];

  for (unsigned n = 1; n <= MAX_FOLD; n++) {
    uint64_t lo = rev32(xpow_mod(128 * n + 31));
    uint64_t hi = rev32(xpow_mod(128 * n - 33));
    fold_k[n] = _mm_set_epi64x((long long)hi, (long long)lo);
  }
  k63 = rev32(xpow_mod(63));

  /* mu = floor(x^96 / G) by long division; its x^64 term is implicit. */
  unsigned __int128 num = (unsigned __int128)1 << 96, q = 0;
  for (int d = 96; d >= 32; d--)
    if ((num >> d) & 1) {
      q |= (unsigned __int128)1 << (d - 32);
      num ^= (unsigned __int128)POLY_NORMAL << (d - 32);
    }
  mu_rep = rev64((uint64_t)q);
}

/* Tables */

static uint32_t crc_bytewise_raw(uint32_t c, const uint8_t *p, size_t n) {
  for (size_t i = 0; i < n; i++) c = table[0][(c ^ p[i]) & 0xFF] ^ (c >> 8);
  return c;
}

static uint32_t crc_bytewise(uint32_t crc, const uint8_t *p, size_t n) {
  return ~crc_bytewise_raw(~crc, p, n);
}

static uint32_t crc_slice8_raw(uint32_t c, const uint8_t *p, size_t n) {
  while (n >= 8) {
    uint64_t v;
    memcpy(&v, p, 8);
    v ^= c;
    c = table[7][v & 0xFF] ^ table[6][(v >> 8) & 0xFF] ^
        table[5][(v >> 16) & 0xFF] ^ table[4][(v >> 24) & 0xFF] ^
        table[3][(v >> 32) & 0xFF] ^ table[2][(v >> 40) & 0xFF] ^
        table[1][(v >> 48) & 0xFF] ^ table[0][v >> 56];
    p += 8;
    n -= 8;
  }
  return crc_bytewise_raw(c, p, n);
}

static uint32_t crc_slice8(uint32_t crc, const uint8_t *p, size_t n) {
  return ~crc_slice8_raw(~crc, p, n);
}

/* CRC-32C */

static uint32_t crc32c_bytewise(uint32_t crc, const uint8_t *p, size_t n) {
  uint32_t c = ~crc;
  for (size_t i = 0; i < n; i++) c = table_c[(c ^ p[i]) & 0xFF] ^ (c >> 8);
  return ~c;
}

static uint32_t crc32c_hw(uint32_t crc, const uint8_t *p, size_t n) {
  uint64_t c = ~crc;
  while (n >= 8) {
    uint64_t v;
    memcpy(&v, p, 8);
    c = _mm_crc32_u64(c, v);
    p += 8;
    n -= 8;
  }
  uint32_t c32 = (uint32_t)c;
  while (n--) c32 = _mm_crc32_u8(c32, *p++);
  return ~c32;
}

/* Three independent chains over the thirds of the input: the instruction's
   throughput, not a CRC of the whole. */
static uint32_t crc32c_hw3(uint32_t crc, const uint8_t *p, size_t n) {
  size_t third = n / 24 * 8;
  uint64_t a = ~crc, b = 0, c = 0;
  const uint8_t *q = p + third, *r = p + 2 * third;
  for (size_t i = 0; i < third; i += 8) {
    uint64_t u, v, w;
    memcpy(&u, p + i, 8);
    memcpy(&v, q + i, 8);
    memcpy(&w, r + i, 8);
    a = _mm_crc32_u64(a, u);
    b = _mm_crc32_u64(b, v);
    c = _mm_crc32_u64(c, w);
  }
  return ~(uint32_t)(a ^ b ^ c);
}

/* Folding */

static inline __m128i fold(__m128i a, __m128i k) {
  return _mm_xor_si128(_mm_clmulepi64_si128(a, k, 0x00),
                       _mm_clmulepi64_si128(a, k, 0x11));
}

static inline __m128i load(const uint8_t *p) {
  return _mm_loadu_si128((const __m128i *)p);
}

/* The CRC register of the 16 bytes [v] holds, as derived at the top. */
static inline uint32_t reduce(__m128i v) {
  __m128i k = _mm_set_epi64x(0, (long long)k63);
  /* 128 -> 96: S = clmul(lo, x^63) + (hi << 32). */
  __m128i s = _mm_xor_si128(_mm_clmulepi64_si128(v, k, 0x00),
                            _mm_slli_si128(_mm_srli_si128(v, 8), 4));
  /* 96 -> 64: V = clmul(S[0..31], x^63) + S[32..95]. */
  uint64_t s_lo = (uint64_t)_mm_cvtsi128_si64(s);
  uint64_t s_hi = (uint64_t)_mm_extract_epi64(s, 1);
  __m128i t = _mm_cvtsi64_si128((long long)(s_lo & 0xFFFFFFFFu));
  uint64_t c2 = (uint64_t)_mm_cvtsi128_si64(_mm_clmulepi64_si128(t, k, 0x00));
  uint64_t u = (s_lo >> 32) | (s_hi << 32);
  uint64_t v64 = c2 ^ u;
  /* Barrett. */
  __m128i vv = _mm_cvtsi64_si128((long long)v64);
  __m128i m = _mm_cvtsi64_si128((long long)mu_rep);
  uint64_t c3 = (uint64_t)_mm_cvtsi128_si64(_mm_clmulepi64_si128(vv, m, 0x00));
  uint64_t q = v64 ^ (c3 << 1);
  __m128i g = _mm_cvtsi64_si128((long long)POLY_REFLECTED);
  __m128i c4 = _mm_clmulepi64_si128(_mm_cvtsi64_si128((long long)q), g, 0x00);
  uint64_t lo = (uint64_t)_mm_cvtsi128_si64(c4);
  uint64_t hi = (uint64_t)_mm_extract_epi64(c4, 1);
  return (uint32_t)((lo >> 63) | (hi << 1));
}

/* [w] xmm accumulators, 16 w bytes per step. */
static inline __attribute__((always_inline)) uint32_t
crc_fold(uint32_t crc, const uint8_t *p, size_t n, int w) {
  if (n < (size_t)(32 * w)) return crc_slice8(crc, p, n);
  __m128i acc[12];
  for (int i = 0; i < w; i++) acc[i] = load(p + 16 * i);
  acc[0] = _mm_xor_si128(acc[0], _mm_cvtsi32_si128((int)~crc));
  p += 16 * w;
  n -= 16 * (size_t)w;
  __m128i k = fold_k[w];
  while (n >= (size_t)(16 * w)) {
    for (int i = 0; i < w; i++)
      acc[i] = _mm_xor_si128(fold(acc[i], k), load(p + 16 * i));
    p += 16 * w;
    n -= 16 * (size_t)w;
  }
  __m128i v = acc[w - 1];
  for (int i = 0; i < w - 1; i++)
    v = _mm_xor_si128(v, fold(acc[i], fold_k[w - 1 - i]));
  while (n >= 16) {
    v = _mm_xor_si128(fold(v, fold_k[1]), load(p));
    p += 16;
    n -= 16;
  }
  return ~crc_slice8_raw(reduce(v), p, n);
}

static uint32_t crc_fold4(uint32_t c, const uint8_t *p, size_t n) {
  return crc_fold(c, p, n, 4);
}

static uint32_t crc_fold8(uint32_t c, const uint8_t *p, size_t n) {
  return crc_fold(c, p, n, 8);
}

static uint32_t crc_fold12(uint32_t c, const uint8_t *p, size_t n) {
  return crc_fold(c, p, n, 12);
}

/* [w] ymm accumulators, 32 w bytes per step, each ymm two blocks. */
static inline __attribute__((always_inline)) uint32_t
crc_vfold(uint32_t crc, const uint8_t *p, size_t n, int w) {
  if (n < (size_t)(64 * w)) return crc_slice8(crc, p, n);
  __m256i acc[8];
  for (int i = 0; i < w; i++)
    acc[i] = _mm256_loadu_si256((const __m256i *)(p + 32 * i));
  __m128i c0 = _mm_cvtsi32_si128((int)~crc);
  acc[0] = _mm256_xor_si256(acc[0], _mm256_zextsi128_si256(c0));
  p += 32 * w;
  n -= 32 * (size_t)w;
  __m256i k = _mm256_broadcastsi128_si256(fold_k[2 * w]);
  while (n >= (size_t)(32 * w)) {
    for (int i = 0; i < w; i++) {
      __m256i a = _mm256_xor_si256(_mm256_clmulepi64_epi128(acc[i], k, 0x00),
                                   _mm256_clmulepi64_epi128(acc[i], k, 0x11));
      __m256i d = _mm256_loadu_si256((const __m256i *)(p + 32 * i));
      acc[i] = _mm256_xor_si256(a, d);
    }
    p += 32 * w;
    n -= 32 * (size_t)w;
  }
  /* Blocks in message order: acc[i]'s low half, then its high half. */
  __m128i v = _mm256_extracti128_si256(acc[w - 1], 1);
  for (int j = 0; j < 2 * w - 1; j++) {
    __m128i b = j & 1 ? _mm256_extracti128_si256(acc[j / 2], 1)
                      : _mm256_castsi256_si128(acc[j / 2]);
    v = _mm_xor_si128(v, fold(b, fold_k[2 * w - 1 - j]));
  }
  while (n >= 16) {
    v = _mm_xor_si128(fold(v, fold_k[1]), load(p));
    p += 16;
    n -= 16;
  }
  return ~crc_slice8_raw(reduce(v), p, n);
}

static uint32_t crc_vfold4(uint32_t c, const uint8_t *p, size_t n) {
  return crc_vfold(c, p, n, 4);
}

static uint32_t crc_vfold8(uint32_t c, const uint8_t *p, size_t n) {
  return crc_vfold(c, p, n, 8);
}

/* Driver */

typedef uint32_t (*crc_fn)(uint32_t, const uint8_t *, size_t);

struct variant {
  const char *name;
  crc_fn f;
  crc_fn ref; /* the bytewise CRC it must agree with, or NULL */
};

static const struct variant variants[] = {
    {"bytewise (CRC-32)", crc_bytewise, crc_bytewise},
    {"slice-by-8 (CRC-32)", crc_slice8, crc_bytewise},
    {"crc32 instruction, 1 chain (CRC-32C)", crc32c_hw, crc32c_bytewise},
    {"crc32 instruction, 3 chains, no combine (CRC-32C)", crc32c_hw3, NULL},
    {"PCLMULQDQ fold, 4 xmm (CRC-32)", crc_fold4, crc_bytewise},
    {"PCLMULQDQ fold, 8 xmm (CRC-32)", crc_fold8, crc_bytewise},
    {"PCLMULQDQ fold, 12 xmm (CRC-32)", crc_fold12, crc_bytewise},
    {"VPCLMULQDQ fold, 4 ymm (CRC-32)", crc_vfold4, crc_bytewise},
    {"VPCLMULQDQ fold, 8 ymm (CRC-32)", crc_vfold8, crc_bytewise},
};

#define NVARIANTS (sizeof variants / sizeof variants[0])

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static uint64_t rng = 0x9E3779B97F4A7C15ull;

static uint64_t next(void) {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return rng;
}

static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

/* Every length 0 to 1100 at every alignment 0 to 15, with a random initial
   CRC, and the check value of "123456789". */
static void check(const uint8_t *buf) {
  if (crc_bytewise(0, (const uint8_t *)"123456789", 9) != 0xCBF43926u ||
      crc32c_bytewise(0, (const uint8_t *)"123456789", 9) != 0xE3069283u) {
    fprintf(stderr, "bytewise check values wrong\n");
    exit(1);
  }
  for (size_t v = 0; v < NVARIANTS; v++) {
    if (!variants[v].ref) continue;
    for (size_t al = 0; al < 16; al++)
      for (size_t n = 0; n <= 1100; n++) {
        uint32_t c = (uint32_t)next();
        uint32_t want = variants[v].ref(c, buf + al, n);
        uint32_t got = variants[v].f(c, buf + al, n);
        if (want != got) {
          fprintf(stderr, "%s: length %zu, alignment %zu: %08x, want %08x\n",
                  variants[v].name, n, al, got, want);
          exit(1);
        }
      }
    size_t big = 8u << 20;
    if (variants[v].f(0, buf, big) != variants[v].ref(0, buf, big)) {
      fprintf(stderr, "%s: 8 MiB differs\n", variants[v].name);
      exit(1);
    }
  }
}

#define ROUNDS 11
#define ROUND_SECONDS 0.02

int main(void) {
  init();
  size_t cap = (8u << 20) + 64;
  uint8_t *buf = aligned_alloc(64, cap);
  if (!buf) return 1;
  for (size_t i = 0; i < cap; i++) buf[i] = (uint8_t)next();
  check(buf);
  printf("| kernel | 4 KiB GB/s | 64 KiB GB/s | 8 MiB GB/s |\n");
  printf("|---|---|---|---|\n");
  static const size_t sizes[] = {4096, 65536, 8u << 20};
  for (size_t v = 0; v < NVARIANTS; v++) {
    printf("| %s |", variants[v].name);
    for (size_t s = 0; s < 3; s++) {
      size_t n = sizes[s];
      uint32_t c = 0;
      double t0 = now();
      c = variants[v].f(c, buf, n);
      double once = now() - t0;
      long calls = (long)(ROUND_SECONDS / (once > 1e-9 ? once : 1e-9)) + 1;
      double r[ROUNDS];
      for (int k = 0; k < ROUNDS; k++) {
        t0 = now();
        for (long i = 0; i < calls; i++) c = variants[v].f(c, buf, n);
        r[k] = (now() - t0) / (double)calls;
      }
      qsort(r, ROUNDS, sizeof r[0], cmp);
      /* The median; each call continues the last one's CRC. */
      printf(" %.1f |", (double)n / r[ROUNDS / 2] / 1e9);
      if (c == 0x12345678u) printf("!");
    }
    printf("\n");
  }
  free(buf);
  return 0;
}
