/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Contraction kernels on the matrix unit of Apple's M1 (amx.h), and the amx
   table: base's, with these kernels for chain order.

   A register holds L = 64 / w elements of w bytes: 16 float32, 8 float64.
   A kernel adds 2L × 2L outputs per step of k, 32 × 32 float32 or 16 × 16
   float64, as four L × L accumulators in Z, q = 2·(row half) + (column
   half): output (i, j) lives in Z row w·(i mod L) + q, lane j mod L. A
   step loads a's 2L elements into Y and b's 2L into X, 128 bytes each, and
   runs four outer products, one per accumulator: on one thread, the
   unit's peak, fma32 at 1.49 TFLOP/s and fma64 at 372 GFLOP/s on the M1
   Max, with operands in the caches. Four steps load at a time, into all
   eight X and Y registers. a is packed, each step on 128 bytes; b may be
   read in place, by pairs of registers where its steps lie on 128 bytes,
   else a register at a time. Each output adds a[i]·b[j] fused, in
   increasing k: the chain NEON's kernels compute, bit for bit on every
   result but a NaN, which the unit gives as the default NaN.

   The kernel and the pack run in a live state of the unit that enter makes
   and leave ends, once per body of the driver's jobs (gemm.c): a state per
   call drained the unit at each end. A thread holds none between jobs. A
   call of a whole KC block is 4,096 outer products, against which the
   tile's stores from Z cost about 3%; a first block starts from +0, which
   four outer products write, so that a product of at most KC steps loads
   no tile. An M1 has one unit per cluster of cores, which every core of
   the cluster feeds: products run on all the performance cores, which on
   the M1 Max beat one thread per unit (1024³ float32: 1.09 against 2.02
   ms).

   A kernel's speed prices its jobs (cpu.h): at the unit's, a 128³ product
   runs on one thread, which beats eight (5.9 against 8.0 us). The cores of
   a cluster share its unit, so that the M1 Max's eight performance cores
   hold two: products of at most 8 rows, as decoding is, run faster on the
   cores' vector units (base's kernels, the table's other), and the unit
   packs only in jobs of at most one thread per unit (gemm.c).

   The functions below take the element's bytes [w]: 4 or 8, a constant
   in each width's kernel and pack, into which they are inlined. */

#include "cpu.h"

#if defined(__APPLE__) && defined(__aarch64__)

#include <string.h>
#include <sys/sysctl.h>

#include "amx.h"

/* The unit's speed (cpu.h), against the 100 GB/s a performance core's
   memcpy moves: fma32 at 1.36 TFLOP/s from one thread of the M1 Max in a
   product, fma64 at a quarter of it, as each outer product adds a quarter
   as many. */
#define SPEED32 13.0
#define SPEED64 (SPEED32 / 4)

#define INLINE static inline __attribute__((always_inline))

/* The elements of a register. */
#define LANES(w) (64 / (w))

/* The Z row of row [i]'s column half [h] of a tile. */
#define ZROW(w, i, h) ((w) * ((i) % LANES(w)) + 2 * ((i) / LANES(w)) + (h))

/* Whether [p], and every [step] bytes from it, lie on 128 bytes: where a
   pair load moves two registers at once. */
static int on128(const void *p, int64_t step) {
  return ((uintptr_t)p | (uintptr_t)step) % 128 == 0;
}

/* Moves the first [rows] rows of the tile at [c], [ldc] bytes apart, into
   Z and back: a row of the tile is the Z rows ZROW(w, i, 0) and
   ZROW(w, i, 1), adjacent, so that where its rows start on 128 bytes each
   moves in one instruction, else in two. */
INLINE void zload(int w, const uint8_t *c, int64_t ldc, int64_t rows) {
  int paired = on128(c, ldc);
  for (int i = 0; i < rows; i++) {
    const uint8_t *r = c + i * ldc;
    if (paired) NX_AMX_LDZ((uint64_t)r | NX_AMX_PAIR, ZROW(w, i, 0));
    else
      for (int h = 0; h < 2; h++) NX_AMX_LDZ(r + 64 * h, ZROW(w, i, h));
  }
}

INLINE void zstore(int w, uint8_t *c, int64_t ldc, int64_t rows) {
  int paired = on128(c, ldc);
  for (int i = 0; i < rows; i++) {
    uint8_t *r = c + i * ldc;
    if (paired) NX_AMX_STZ((uint64_t)r | NX_AMX_PAIR, ZROW(w, i, 0));
    else
      for (int h = 0; h < 2; h++) NX_AMX_STZ(r + 64 * h, ZROW(w, i, h));
  }
}

/* 128 bytes of +0, a pair of Y registers for zeroing Z. */
static _Alignas(128) const uint8_t zeros[128];

/* Sets Z to +0: fma32's four accumulators, its 64 rows, take 1·y, y +0. */
static void zzero(void) {
  NX_AMX_LDY((uint64_t)zeros | NX_AMX_PAIR, 0);
  for (int q = 0; q < 4; q++)
    NX_AMX_OP(12, NX_AMX_OUTER(0, 0, q) | NX_AMX_SKIP_Z | NX_AMX_SKIP_X);
}

/* 128 bytes from [s] into X's registers [r] and [r + 1]: in one load
   where [pair], else in two. */
static void ldx2(const uint8_t *s, int r, int pair) {
  if (pair) {
    NX_AMX_LDX((uint64_t)s | NX_AMX_PAIR, r);
    return;
  }
  NX_AMX_LDX(s, r);
  NX_AMX_LDX(s + 64, r + 1);
}

/* The four outer products of the step in X and Y's registers [r] and
   [r + 1]. */
INLINE void outer4(int w, int r) {
  int o = 64 * r;
  for (int q = 0; q < 4; q++) {
    uint64_t op = NX_AMX_OUTER(o + 64 * (q % 2), o + 64 * (q / 2), q);
    if (w == 4) NX_AMX_OP(12, op);
    else NX_AMX_OP(10, op);
  }
}

/* The first [rows] rows of the tile at [c]: a's packed steps from [a],
   each on 128 bytes, times b's 2L columns; steps [a_k] and [b_k] bytes
   apart. The rows past [rows] are computed from a's padding and not
   stored. */
INLINE void tile(int w, int64_t k, const uint8_t *a, int64_t a_k,
                 const uint8_t *b, int64_t b_k, uint8_t *c, int64_t ldc,
                 int64_t rows, nx_cpu_from from) {
  int pair_b = on128(b, b_k);
  if (from == NX_CPU_FROM_TILE) zload(w, c, ldc, rows);
  else zzero();
  int64_t p = 0;
  for (; p + 4 <= k; p += 4) {
    for (int s = 0; s < 4; s++) {
      NX_AMX_LDY((uint64_t)(a + (p + s) * a_k) | NX_AMX_PAIR, 2 * s);
      ldx2(b + (p + s) * b_k, 2 * s, pair_b);
    }
    for (int s = 0; s < 4; s++) outer4(w, 2 * s);
  }
  for (; p < k; p++) {
    NX_AMX_LDY((uint64_t)(a + p * a_k) | NX_AMX_PAIR, 0);
    ldx2(b + p * b_k, 0, pair_b);
    outer4(w, 0);
  }
  zstore(w, c, ldc, rows);
}

/* A tile of [rows] and [n] columns, fewer than 2L, through a whole one on
   the stack. Apart from the kernel, so that the kernel's
   frame holds no tile, which macOS would probe page by page at each
   call. */
static __attribute__((noinline)) void edge(
    int w, int64_t k, const uint8_t *a, int64_t a_k, const uint8_t *b,
    int64_t b_k, uint8_t *y, int64_t ldc, int64_t rows, int64_t n,
    nx_cpu_from from) {
  _Alignas(128) uint8_t t[NX_CPU_TILE];
  for (int64_t i = 0; from == NX_CPU_FROM_TILE && i < rows; i++)
    memcpy(t + 128 * i, y + i * ldc, (size_t)(n * w));
  if (w == 4) tile(4, k, a, a_k, b, b_k, t, 128, rows, from);
  else tile(8, k, a, a_k, b, b_k, t, 128, rows, from);
  for (int64_t i = 0; i < rows; i++)
    memcpy(y + i * ldc, t + 128 * i, (size_t)(n * w));
}

/* a is packed (NX_CPU_A_PACKED), its rows adjacent. */
INLINE void kernel(int w, int64_t m, int64_t n, int64_t k, const void *va,
                   int64_t a_row, int64_t a_k, const void *vb, int64_t b_k,
                   void *vc, int64_t ldc, nx_cpu_from from) {
  const uint8_t *a = va, *b = vb;
  uint8_t *c = vc;
  int64_t mr = 2 * LANES(w);
  for (int64_t i0 = 0; i0 < m; i0 += mr) {
    int64_t rows = m - i0 < mr ? m - i0 : mr;
    const uint8_t *ai = a + i0 * a_row * w;
    uint8_t *ci = c + i0 * ldc * w;
    if (n == mr)
      tile(w, k, ai, a_k * w, b, b_k * w, ci, ldc * w, rows, from);
    else edge(w, k, ai, a_k * w, b, b_k * w, ci, ldc * w, rows, n, from);
  }
}

static void kernel_f32(int64_t m, int64_t n, int64_t k, const void *a,
                       int64_t a_row, int64_t a_k, const void *b,
                       int64_t b_k, void *c, int64_t ldc, nx_cpu_from from) {
  kernel(4, m, n, k, a, a_row, a_k, b, b_k, c, ldc, from);
}

static void kernel_f64(int64_t m, int64_t n, int64_t k, const void *a,
                       int64_t a_row, int64_t a_k, const void *b,
                       int64_t b_k, void *c, int64_t ldc, nx_cpu_from from) {
  kernel(8, m, n, k, a, a_row, a_k, b, b_k, c, ldc, from);
}

/* Transposes 2L rows × 2L steps from [s] into [d], rows [ld] and steps
   [pitch] bytes apart, on the unit. Row i of row half h loads into the Z
   rows w·i + 2h and the next, its steps 0 to L - 1 and L to 2L - 1: Z
   holds four quarters, q = 2·(row half) + (step half), each row i of a
   quarter in Z row w·i + q. Step k is then column k mod L of quarters k / L
   and 2 + k / L, moved into Y and stored. Rows and steps that start on 128
   bytes move in one instruction each, others in two. A move rounds
   nothing, so every bit moves, a NaN's payload included. */
INLINE void transpose(int w, const uint8_t *s, int64_t ld, uint8_t *d,
                      int64_t pitch) {
  int l = LANES(w), in_pairs = on128(s, ld), out_pairs = on128(d, pitch);
  for (int h = 0; h < 2; h++)
    for (int i = 0; i < l; i++) {
      const uint8_t *r = s + (l * h + i) * ld;
      if (in_pairs) NX_AMX_LDZ((uint64_t)r | NX_AMX_PAIR, w * i + 2 * h);
      else {
        NX_AMX_LDZ(r, w * i + 2 * h);
        NX_AMX_LDZ(r + 64, w * i + 2 * h + 1);
      }
    }
  for (int k = 0; k < 2 * l; k++) {
    int c = w * (k % l) + k / l;
    uint8_t *r = d + k * pitch;
    if (w == 4) {
      NX_AMX_EXTRV32(0, c);
      NX_AMX_EXTRV32(64, c + 2);
    } else {
      NX_AMX_EXTRV64(0, c);
      NX_AMX_EXTRV64(64, c + 2);
    }
    if (out_pairs) NX_AMX_STY((uint64_t)r | NX_AMX_PAIR, 0);
    else {
      NX_AMX_STY(r, 0);
      NX_AMX_STY(r + 64, 1);
    }
  }
}

/* The pack. Contiguous rows move 2L elements at a time through X, four
   rows to the eight registers; rows contiguous across, 2L × 2L at a time
   through Z (transpose). The rest moves element by element. */
INLINE void pack(int w, int64_t n0, int64_t n1, const void *vs, int64_t s0,
                 int64_t s1, void *vd, int64_t pitch) {
  const uint8_t *s = vs;
  uint8_t *d = vd;
  int64_t mr = 2 * LANES(w), i_most = n0 / mr * mr, j_most = n1 / mr * mr;
  if (s0 == 1) {
    int64_t j4 = n1 / 4 * 4;
    for (int64_t i = 0; i < i_most; i += mr)
      for (int64_t j = 0; j < j4; j += 4) {
        for (int u = 0; u < 4; u++) {
          NX_AMX_LDX(s + ((j + u) * s1 + i) * w, 2 * u);
          NX_AMX_LDX(s + ((j + u) * s1 + i) * w + 64, 2 * u + 1);
        }
        for (int u = 0; u < 4; u++) {
          NX_AMX_STX(d + ((j + u) * pitch + i) * w, 2 * u);
          NX_AMX_STX(d + ((j + u) * pitch + i) * w + 64, 2 * u + 1);
        }
      }
    for (int64_t j = 0; j < n1; j++)
      for (int64_t i = j < j4 ? i_most : 0; i < n0; i++)
        memcpy(d + (j * pitch + i) * w, s + (j * s1 + i) * w, w);
    return;
  }
  for (int64_t i = 0; i < i_most; i += mr)
    for (int64_t j = 0; j < j_most; j += mr)
      transpose(w, s + (i * s0 + j) * w, s0 * w, d + (j * pitch + i) * w,
                pitch * w);
  for (int64_t j = 0; j < n1; j++)
    for (int64_t i = j < j_most ? i_most : 0; i < n0; i++)
      memcpy(d + (j * pitch + i) * w, s + (i * s0 + j) * w, w);
}

static void pack_f32(int64_t n0, int64_t n1, const void *s, int64_t s0,
                     int64_t s1, void *d, int64_t pitch) {
  pack(4, n0, n1, s, s0, s1, d, pitch);
}

static void pack_f64(int64_t n0, int64_t n1, const void *s, int64_t s0,
                     int64_t s1, void *d, int64_t pitch) {
  pack(8, n0, n1, s, s0, s1, d, pitch);
}

static int sysctl_int(const char *name) {
  int v = 0;
  size_t n = sizeof v;
  return sysctlbyname(name, &v, &n, NULL, 0) == 0 ? v : 0;
}

static void enter(void) { NX_AMX_SET(); }
static void leave(void) { NX_AMX_CLR(); }

/* Whether the host has the matrix unit these kernels use: the M1's cores,
   hw.cpufamily CPUFAMILY_ARM_FIRESTORM_ICESTORM of <mach/machine.h>. M2
   and M3 carry the unit too and join when a machine here runs the suite on
   them; M4 and later have SME instead. */
int nx_cpu_has_amx(void) {
  return (uint32_t)sysctl_int("hw.cpufamily") == 0x1b588bb3u;
}

/* The table's kernels: the unit's, priced against base's. */
void nx_cpu_set_amx(nx_cpu_target *t) {
  /* The cores that share a unit: a cluster's, which share its L2. */
  int shared = sysctl_int("hw.perflevel0.cpusperl2");
  shared = shared > 0 ? shared : 1;
  t->gemm[NX_FLOAT32] = (nx_cpu_gemm){
      .kernel = {kernel_f32, 32, 32, NX_CPU_A_PACKED, SPEED32, shared},
      .mc = 128,
      .kc = 1024,
      .nc = 2048,
      .pack = pack_f32,
      .other = &nx_cpu_base.gemm[NX_FLOAT32],
      .enter = enter,
      .leave = leave};
  t->gemm[NX_FLOAT64] = (nx_cpu_gemm){
      .kernel = {kernel_f64, 16, 16, NX_CPU_A_PACKED, SPEED64, shared},
      .mc = 128,
      .kc = 1024,
      .nc = 2048,
      .pack = pack_f64,
      .other = &nx_cpu_base.gemm[NX_FLOAT64],
      .enter = enter,
      .leave = leave};
}

#else

/* ISO C forbids an empty unit. */
typedef int nx_cpu_no_amx;

#endif
