/* Checksums: the generic paths, written for any C11 host, and the choice
   of path, made once when the program starts.

   CRC-32 is RFC 1952 section 8's: the reflected polynomial 0xEDB88320,
   the register preset to all ones and inverted at the end. The generic
   path reads 8 bytes per step through 8 tables ("slicing by 8"): table k
   holds the CRC of a byte followed by k zero bytes, so the 8 lookups of
   one step are independent and XOR together.

   Adler-32 is RFC 1950 section 2.2's: s1 is 1 plus the sum of the bytes,
   s2 the sum of the s1 after each byte, both modulo 65521. The sums are
   reduced once per ADLER_NMAX bytes, the most that cannot overflow 32
   bits (section 8.2).

   XXH32 and XXH64 are the xxHash specification's (version 0.2.0), in its
   steps: four accumulators over 16- or 32-byte stripes, convergence, the
   length, the tail, the avalanche. */

#include "checksum.h"
#include "checksum_paths.h"

#include <string.h>

/* Little-endian loads, which compilers turn into one load on the hosts lz
   builds on. */

static inline uint32_t le32(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
         (uint32_t)p[3] << 24;
}

static inline uint64_t le64(const uint8_t *p) {
  return (uint64_t)le32(p) | (uint64_t)le32(p + 4) << 32;
}

/* CRC-32 */

#define CRC32_POLY 0xEDB88320u

/* crc32_table[k][b] is the register after the byte b followed by k zero
   bytes, from a zero register. */
static uint32_t crc32_table[8][256];

static void crc32_tables(void) {
  for (uint32_t b = 0; b < 256; b++) {
    uint32_t r = b;
    for (int i = 0; i < 8; i++) r = (r >> 1) ^ (CRC32_POLY & (0u - (r & 1)));
    crc32_table[0][b] = r;
  }
  for (int k = 1; k < 8; k++)
    for (int b = 0; b < 256; b++) {
      uint32_t r = crc32_table[k - 1][b];
      crc32_table[k][b] = (r >> 8) ^ crc32_table[0][r & 0xff];
    }
}

uint32_t lz_crc32_bytes(uint32_t r, const uint8_t *p, size_t n) {
  for (size_t i = 0; i < n; i++)
    r = (r >> 8) ^ crc32_table[0][(r ^ p[i]) & 0xff];
  return r;
}

uint32_t lz_crc32_slice8(uint32_t r, const uint8_t *p, size_t n) {
  const uint32_t(*t)[256] = crc32_table;
  for (; n >= 8; p += 8, n -= 8) {
    uint32_t a = r ^ le32(p), b = le32(p + 4);
    r = t[7][a & 0xff] ^ t[6][(a >> 8) & 0xff] ^ t[5][(a >> 16) & 0xff] ^
        t[4][a >> 24] ^ t[3][b & 0xff] ^ t[2][(b >> 8) & 0xff] ^
        t[1][(b >> 16) & 0xff] ^ t[0][b >> 24];
  }
  return lz_crc32_bytes(r, p, n);
}

static uint32_t crc32_generic(uint32_t crc, const uint8_t *p, size_t n) {
  return ~lz_crc32_slice8(~crc, p, n);
}

/* Adler-32 */

#define ADLER_MOD 65521u

/* With s1, s2 < 65521 at the start of a block of n bytes of 255, s2 grows
   by at most 255 n (n + 1) / 2 + (n + 1) (65521 - 1): 5552 is the largest
   n for which that stays under 2^32. */
#define ADLER_NMAX 5552

uint32_t lz_adler32_generic(uint32_t adler, const uint8_t *p, size_t n) {
  uint32_t s1 = adler & 0xffff, s2 = adler >> 16;
  while (n > 0) {
    size_t k = n < ADLER_NMAX ? n : ADLER_NMAX;
    n -= k;
    for (; k >= 4; k -= 4, p += 4) {
      s1 += p[0];
      s2 += s1;
      s1 += p[1];
      s2 += s1;
      s1 += p[2];
      s2 += s1;
      s1 += p[3];
      s2 += s1;
    }
    for (; k > 0; k--, p++) {
      s1 += *p;
      s2 += s1;
    }
    s1 %= ADLER_MOD;
    s2 %= ADLER_MOD;
  }
  return s2 << 16 | s1;
}

/* XXH32 */

#define P32_1 0x9E3779B1u
#define P32_2 0x85EBCA77u
#define P32_3 0xC2B2AE3Du
#define P32_4 0x27D4EB2Fu
#define P32_5 0x165667B1u

static inline uint32_t rotl32(uint32_t x, int s) {
  return x << s | x >> (32 - s);
}

/* gcc turns the four rounds of a stripe into one SSE2 vector, whose 32-bit
   multiplies become shifts and adds: less than half the scalar speed on
   fudge. The empty asm keeps each accumulator in a scalar register, as
   xxhash.h's XXH_COMPILER_GUARD does. */
static inline uint32_t xxh32_round(uint32_t acc, uint32_t lane) {
  acc = rotl32(acc + lane * P32_2, 13) * P32_1;
#if defined(__GNUC__)
  __asm__("" : "+r"(acc));
#endif
  return acc;
}

static void xxh32_seed(uint32_t acc[4], uint32_t seed) {
  acc[0] = seed + P32_1 + P32_2;
  acc[1] = seed + P32_2;
  acc[2] = seed;
  acc[3] = seed - P32_1;
}

/* Runs step 2 over every whole stripe of [p, p + n); returns the bytes it
   consumed. */
static size_t xxh32_stripes(uint32_t acc[4], const uint8_t *p, size_t n) {
  uint32_t a = acc[0], b = acc[1], c = acc[2], d = acc[3];
  size_t done = n & ~(size_t)15;
  for (const uint8_t *e = p + done; p < e; p += 16) {
    a = xxh32_round(a, le32(p));
    b = xxh32_round(b, le32(p + 4));
    c = xxh32_round(c, le32(p + 8));
    d = xxh32_round(d, le32(p + 12));
  }
  acc[0] = a, acc[1] = b, acc[2] = c, acc[3] = d;
  return done;
}

/* Steps 3 to 6 for a [total]-byte input whose stripes left [acc] and whose
   last [n] bytes, fewer than a stripe, are [p]. */
static uint32_t xxh32_finish(const uint32_t acc[4], uint32_t seed,
                             uint64_t total, const uint8_t *p, size_t n) {
  uint32_t h = total >= 16 ? rotl32(acc[0], 1) + rotl32(acc[1], 7) +
                                 rotl32(acc[2], 12) + rotl32(acc[3], 18)
                           : seed + P32_5;
  h += (uint32_t)total;
  for (; n >= 4; p += 4, n -= 4) h = rotl32(h + le32(p) * P32_3, 17) * P32_4;
  for (; n > 0; p++, n--) h = rotl32(h + *p * P32_5, 11) * P32_1;
  h ^= h >> 15;
  h *= P32_2;
  h ^= h >> 13;
  h *= P32_3;
  h ^= h >> 16;
  return h;
}

uint32_t lz_xxh32(uint32_t seed, const uint8_t *p, size_t n) {
  uint32_t acc[4];
  xxh32_seed(acc, seed);
  size_t done = xxh32_stripes(acc, p, n);
  return xxh32_finish(acc, seed, n, p + done, n - done);
}

void lz_xxh32_init(struct lz_xxh32 *h, uint32_t seed) {
  memset(h, 0, sizeof *h);
  h->seed = seed;
  xxh32_seed(h->acc, seed);
}

void lz_xxh32_update(struct lz_xxh32 *h, const uint8_t *p, size_t n) {
  h->total += n;
  if (h->buffered > 0) {
    size_t k = sizeof h->buf - h->buffered;
    if (n < k) {
      memcpy(h->buf + h->buffered, p, n);
      h->buffered += (uint32_t)n;
      return;
    }
    memcpy(h->buf + h->buffered, p, k);
    xxh32_stripes(h->acc, h->buf, sizeof h->buf);
    p += k, n -= k;
    h->buffered = 0;
  }
  size_t done = xxh32_stripes(h->acc, p, n);
  memcpy(h->buf, p + done, n - done);
  h->buffered = (uint32_t)(n - done);
}

uint32_t lz_xxh32_digest(const struct lz_xxh32 *h) {
  return xxh32_finish(h->acc, h->seed, h->total, h->buf, h->buffered);
}

/* XXH64 */

#define P64_1 0x9E3779B185EBCA87u
#define P64_2 0xC2B2AE3D27D4EB4Fu
#define P64_3 0x165667B19E3779F9u
#define P64_4 0x85EBCA77C2B2AE63u
#define P64_5 0x27D4EB2F165667C5u

static inline uint64_t rotl64(uint64_t x, int s) {
  return x << s | x >> (64 - s);
}

static inline uint64_t xxh64_round(uint64_t acc, uint64_t lane) {
  return rotl64(acc + lane * P64_2, 31) * P64_1;
}

static inline uint64_t xxh64_merge(uint64_t h, uint64_t acc) {
  return (h ^ xxh64_round(0, acc)) * P64_1 + P64_4;
}

static void xxh64_seed(uint64_t acc[4], uint64_t seed) {
  acc[0] = seed + P64_1 + P64_2;
  acc[1] = seed + P64_2;
  acc[2] = seed;
  acc[3] = seed - P64_1;
}

static size_t xxh64_stripes(uint64_t acc[4], const uint8_t *p, size_t n) {
  uint64_t a = acc[0], b = acc[1], c = acc[2], d = acc[3];
  size_t done = n & ~(size_t)31;
  for (const uint8_t *e = p + done; p < e; p += 32) {
    a = xxh64_round(a, le64(p));
    b = xxh64_round(b, le64(p + 8));
    c = xxh64_round(c, le64(p + 16));
    d = xxh64_round(d, le64(p + 24));
  }
  acc[0] = a, acc[1] = b, acc[2] = c, acc[3] = d;
  return done;
}

static uint64_t xxh64_finish(const uint64_t acc[4], uint64_t seed,
                             uint64_t total, const uint8_t *p, size_t n) {
  uint64_t h = seed + P64_5;
  if (total >= 32) {
    h = rotl64(acc[0], 1) + rotl64(acc[1], 7) + rotl64(acc[2], 12) +
        rotl64(acc[3], 18);
    for (int i = 0; i < 4; i++) h = xxh64_merge(h, acc[i]);
  }
  h += total;
  for (; n >= 8; p += 8, n -= 8)
    h = rotl64(h ^ xxh64_round(0, le64(p)), 27) * P64_1 + P64_4;
  if (n >= 4) {
    h = rotl64(h ^ (uint64_t)le32(p) * P64_1, 23) * P64_2 + P64_3;
    p += 4, n -= 4;
  }
  for (; n > 0; p++, n--) h = rotl64(h ^ *p * P64_5, 11) * P64_1;
  h ^= h >> 33;
  h *= P64_2;
  h ^= h >> 29;
  h *= P64_3;
  h ^= h >> 32;
  return h;
}

uint64_t lz_xxh64(uint64_t seed, const uint8_t *p, size_t n) {
  uint64_t acc[4];
  xxh64_seed(acc, seed);
  size_t done = xxh64_stripes(acc, p, n);
  return xxh64_finish(acc, seed, n, p + done, n - done);
}

void lz_xxh64_init(struct lz_xxh64 *h, uint64_t seed) {
  memset(h, 0, sizeof *h);
  h->seed = seed;
  xxh64_seed(h->acc, seed);
}

void lz_xxh64_update(struct lz_xxh64 *h, const uint8_t *p, size_t n) {
  h->total += n;
  if (h->buffered > 0) {
    size_t k = sizeof h->buf - h->buffered;
    if (n < k) {
      memcpy(h->buf + h->buffered, p, n);
      h->buffered += (uint32_t)n;
      return;
    }
    memcpy(h->buf + h->buffered, p, k);
    xxh64_stripes(h->acc, h->buf, sizeof h->buf);
    p += k, n -= k;
    h->buffered = 0;
  }
  size_t done = xxh64_stripes(h->acc, p, n);
  memcpy(h->buf, p + done, n - done);
  h->buffered = (uint32_t)(n - done);
}

uint64_t lz_xxh64_digest(const struct lz_xxh64 *h) {
  return xxh64_finish(h->acc, h->seed, h->total, h->buf, h->buffered);
}

/* The paths, chosen once */

struct lz_crc32_folds lz_crc32_folds;

/* x^d mod G, reflected: bit 31 - i holds the coefficient of x^i. Each step
   multiplies by x, folding x^32 back as G's lower terms. */
static uint32_t crc32_xpow(unsigned d) {
  uint32_t r = 0x80000000u;
  for (unsigned i = 0; i < d; i++)
    r = (r >> 1) ^ (CRC32_POLY & (0u - (r & 1)));
  return r;
}

static struct lz_crc32_fold crc32_fold(unsigned d) {
  return (struct lz_crc32_fold){.hi = crc32_xpow(d + 31),
                                .lo = crc32_xpow(d - 33)};
}

/* The paths this host can run, in the order the probe found them: each
   faster than the one before. The last is the one taken. */
#define MAX_PATHS 4

static struct lz_checksum_path crc32_list[MAX_PATHS] = {
    {"generic", crc32_generic}};
static struct lz_checksum_path adler32_list[MAX_PATHS] = {
    {"generic", lz_adler32_generic}};
static size_t crc32_count = 1, adler32_count = 1;
static lz_crc32_fn *crc32_path = crc32_generic;
static lz_adler32_fn *adler32_path = lz_adler32_generic;

uint32_t lz_crc32(uint32_t crc, const uint8_t *p, size_t n) {
  return crc32_path(crc, p, n);
}

uint32_t lz_adler32(uint32_t adler, const uint8_t *p, size_t n) {
  return adler32_path(adler, p, n);
}

size_t lz_crc32_paths(const struct lz_checksum_path **paths) {
  *paths = crc32_list;
  return crc32_count;
}

size_t lz_adler32_paths(const struct lz_checksum_path **paths) {
  *paths = adler32_list;
  return adler32_count;
}

#if defined(__x86_64__) || defined(__aarch64__)
/* Adds a path, if its unit was compiled, and takes it. */
static void add_crc32(const char *name, lz_crc32_fn *fn) {
  if (fn == NULL || crc32_count == MAX_PATHS) return;
  crc32_list[crc32_count++] = (struct lz_checksum_path){name, fn};
  crc32_path = fn;
}
#endif

#if defined(__x86_64__)
static void add_adler32(const char *name, lz_adler32_fn *fn) {
  if (fn == NULL || adler32_count == MAX_PATHS) return;
  adler32_list[adler32_count++] = (struct lz_checksum_path){name, fn};
  adler32_path = fn;
}

#include <cpuid.h>

/* CPUID leaf 1 ECX: SSE4.1 bit 19, PCLMULQDQ bit 1, OSXSAVE bit 27, AVX
   bit 28; leaf 7 EBX: AVX2 bit 5; leaf 7 ECX: VPCLMULQDQ bit 10; leaf 7
   subleaf 1 EAX: AVX-VNNI bit 4. The 256-bit registers need the operating
   system to save them: XCR0 bits 1 and 2, read by XGETBV. Intel SDM vol.
   2A, CPUID; vol. 1, section 14.3. */
static void choose(void) {
  unsigned a, b, c, d;
  if (!__get_cpuid(1, &a, &b, &c, &d)) return;
  int sse41 = (c >> 19) & 1, pclmul = (c >> 1) & 1;
  int osxsave = (c >> 27) & 1, avx = (c >> 28) & 1;
  if (sse41 && pclmul) add_crc32("pclmul", lz_crc32_pclmul);

  if (!(osxsave && avx)) return;
  unsigned lo, hi;
  __asm__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  if ((lo & 6) != 6 || !__get_cpuid_count(7, 0, &a, &b, &c, &d)) return;
  int avx2 = (b >> 5) & 1, vpclmul = (c >> 10) & 1;
  if (!avx2) return;
  add_adler32("avx2", lz_adler32_avx2);
  if (vpclmul && pclmul) add_crc32("vpclmul", lz_crc32_vpclmul);
  if (__get_cpuid_count(7, 1, &a, &b, &c, &d) && ((a >> 4) & 1))
    add_adler32("vnni", lz_adler32_vnni);
}
#endif

#if defined(__aarch64__)
#if defined(__linux__)
#include <sys/auxv.h>

/* AT_HWCAP's bits for arm64: HWCAP_PMULL 1 << 4, HWCAP_CRC32 1 << 7
   (Linux, arch/arm64/include/uapi/asm/hwcap.h). */
static int has_crc32(void) { return (getauxval(AT_HWCAP) >> 7) & 1; }
static int has_pmull(void) { return (getauxval(AT_HWCAP) >> 4) & 1; }
#else
/* Elsewhere, as on macOS, what the compiler's target promises: Apple's
   arm64 target has both, so every host that runs the program has them. */
#if defined(__ARM_FEATURE_CRC32)
static int has_crc32(void) { return 1; }
#else
static int has_crc32(void) { return 0; }
#endif
#if defined(__ARM_FEATURE_AES)
static int has_pmull(void) { return 1; }
#else
static int has_pmull(void) { return 0; }
#endif
#endif

static void choose(void) {
  if (!has_crc32()) return;
  add_crc32("crc", lz_crc32_arm);
  if (has_pmull()) add_crc32("pmull", lz_crc32_pmull);
}
#endif

/* Runs before main, so before any domain or blocking section can call a
   path: the tables and pointers are written once, by one thread, then only
   read. */
__attribute__((constructor)) static void checksum_probe(void) {
  crc32_tables();
  lz_crc32_folds = (struct lz_crc32_folds){
      .d128 = crc32_fold(128),
      .d256 = crc32_fold(256),
      .d384 = crc32_fold(384),
      .d512 = crc32_fold(512),
      .d768 = crc32_fold(768),
      .d1024 = crc32_fold(1024),
      .d1536 = crc32_fold(1536),
      .d2048 = crc32_fold(2048),
  };
#if defined(__x86_64__) || defined(__aarch64__)
  choose();
#endif
}
