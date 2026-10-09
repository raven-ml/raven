/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The mma kernels: bfloat16 or float16 operands summed in float32, int8 in
   int32, on mma.sync. Each output sums its k in steps of 16 (32 for int8)
   in increasing k, one mma.sync a step; the unit truncates its sum to
   float32 (23 bits kept: each addition errs by at most 2u).

     global --cp.async 16 B (zero past the extents)--> shared ring of
     STAGES k-tiles, 16-byte chunks swizzled --ldmatrix (.trans for an
     operand whose free axis is contiguous)--> fragments --mma.sync-->
     accumulators --> combine (splits) --> + init --> y

   Part of contract.cu's translation unit, after elements.cuh. */

#ifndef NX_CUDA_MMA_CUH
#define NX_CUDA_MMA_CUH

/* The block's BM x BN outputs, held in the mma accumulator layout,
   stored through shared memory ([CAP] bytes at [smem], the pipeline's) in
   passes of whole warp rows: a pass's warps write their accumulators,
   then every thread stores 8 outputs at a time. No accumulator is live
   across a call, and y's stores are whole rows. */
template <typename Acc, int BM, int BN, int WM, int WN, int THREADS, int CAP>
__device__ void epilogue(const contract_params &p, uint8_t *smem,
                                const Acc (&acc)[WM / 16][WN / 8][4], int z,
                                int m0, int n0, int wm0, int wn0) {
  constexpr int LD = BN + 8;
  constexpr int ROWS = CAP / (LD * (int)sizeof(Acc)) / WM * WM;
  static_assert(ROWS >= WM, "a pass holds a warp row");
  Acc *st = (Acc *)smem;
  const int tid = threadIdx.x, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
#pragma unroll
  for (int base = 0; base < BM; base += ROWS) {
    __syncthreads();
    if (wm0 >= base && wm0 < base + ROWS)
#pragma unroll
      for (int i = 0; i < WM / 16; i++)
#pragma unroll
        for (int h = 0; h < 2; h++)
#pragma unroll
          for (int j = 0; j < WN / 8; j++) {
            Acc *d = st + (wm0 - base + i * 16 + g + h * 8) * LD + wn0 + j * 8 + t4 * 2;
            d[0] = acc[i][j][2 * h], d[1] = acc[i][j][2 * h + 1];
          }
    __syncthreads();
    const int rows = min(ROWS, BM - base);
    for (int e = tid; e < rows * (BN / 8); e += THREADS) {
      const int r = e / (BN / 8), c = e % (BN / 8) * 8;
      store8(p, z, m0 + base + r, n0 + c, st + r * LD + c);
    }
  }
}

/* The kinds of operands an mma kernel sums, as kernels.h's list names them:
   any is each of the others, chosen by a's dtype at the kernel's entry. */
#define KIND_bf16 0
#define KIND_f16 1
#define KIND_s8 2
#define KIND_any 3

/* The accumulator of an mma kind: int32 for int8, float32 otherwise. */
template <int KIND> struct Mma_acc { typedef float type; };
template <> struct Mma_acc<KIND_s8> { typedef int type; };

/* One mma.sync step of the kind [KIND] into the accumulators [c]. */
template <int KIND>
__device__ void mma(typename Mma_acc<KIND>::type *c, const uint32_t *a,
                    const uint32_t *b) {
  if constexpr (KIND == KIND_bf16)
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  else if constexpr (KIND == KIND_f16)
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  else
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

/* Four 8 x 8 matrices of 16-bit elements from shared memory, transposed if
   [T]. */
template <bool T> __device__ void ldsm(uint32_t *r, uint32_t addr) {
  if constexpr (T)
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(addr));
  else
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(addr));
}

/* [bytes] of the 16 at [src] to shared [dst], the rest zero. */
__device__ void cp_async(uint32_t dst, const void *src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst),
               "l"(src), "r"(bytes));
}

__device__ void cp_commit(void) {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N> __device__ void cp_wait(void) {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

/* The chunk a row's chunk [c] is stored at, in a tile whose rows hold [C]
   16-byte chunks: the 8 rows an ldmatrix phase reads then hit 8 distinct
   bank groups. */
__device__ int swz(int r, int c, int C) {
  return C >= 8 ? c ^ (r & 7) : c ^ ((r / (8 / C)) % C);
}

/* One operand of a block: its rows (m for a, n for b) from [r0], [R] of
   them, k-tiles of [BK] elements of [ES] bytes, loaded in 16-byte chunks
   along the contiguous axis: a row-major tile ([R] rows of k) if [T] is
   false, k rows of [R] elements if [T]. Its rows are whole vectors: the
   plan packs an operand whose are not.

   A thread loads the chunk at its column of every RS-th tile row, the
   same each k-tile: one address, moved along the rows and along k, and
   the bounds of its chunks, kept from make. */
template <int R, int BK, int ES, bool T, int THREADS> struct Operand {
  static constexpr int CHE = 16 / ES;               /* elements per chunk */
  static constexpr int C = T ? R / CHE : BK / CHE;  /* chunks per tile row */
  static constexpr int ROWS = T ? BK : R;           /* tile rows */
  static constexpr int BYTES = R * BK * ES;
  static constexpr int RS = THREADS / C;            /* rows a pass loads */
  static_assert(THREADS % C == 0, "a thread keeps its tile column");

  const char *first; /* this thread's first chunk at k-tile 0 */
  int64_t step;      /* bytes from a pass's chunk to the next pass's */
  int left;          /* rows from its first to the extent; if T, its
                        chunk's elements inside the rows */
  int kofs;          /* its k in a k-tile (its first, if T) */

  __device__ void make(const char *base, int64_t lead, int rows, int r0,
                       int tid) {
    const int tc = tid % C, tr = tid / C;
    step = RS * lead;
    if (T) {
      first = base + tr * lead + (r0 + tc * CHE) * ES;
      left = min(CHE, rows - (r0 + tc * CHE));
      kofs = tr;
    } else {
      first = base + (r0 + tr) * lead + tc * CHE * ES;
      left = rows - (r0 + tr);
      kofs = tc * CHE;
    }
  }

  /* The tile of k-tile [kt] of an operand of [k] k into the shared buffer
     at [dst], from thread [tid]: the chunks past the extents are zero. */
  __device__ void load(uint32_t dst, int kt, int k, int tid) const {
    static_assert(!T || BK % RS == 0, "a k-tile is whole passes");
    const int tc = tid % C, tr = tid / C;
    const char *src = first + (T ? kt * (BK / RS) * step : (int64_t)kt * BK * ES);
#pragma unroll
    for (int it = 0; it < (ROWS + RS - 1) / RS; it++) {
      const int row = tr + it * RS;
      if (ROWS % RS && row >= ROWS) break;
      const int kk = kt * BK + kofs + (T ? it * RS : 0);
      const int inside = T ? (kk < k ? left : 0)
                           : (it * RS < left ? min(CHE, k - kk) : 0);
      cp_async(dst + (row * C + swz(row, tc, C)) * 16,
               inside > 0 ? src + it * step : first, inside > 0 ? inside * ES : 0);
    }
  }
};

template <int KIND, bool A_T, bool B_T, typename Tile>
__device__ void mma_contract(const contract_params &p) {
  typedef typename Mma_acc<KIND>::type Acc;
  constexpr int BM = Tile::BM, BN = Tile::BN, BKB = Tile::BKB, WM = Tile::WM,
                WN = Tile::WN, STAGES = Tile::STAGES;
  constexpr int ES = KIND == KIND_s8 ? 1 : 2, BK = BKB / ES;
  constexpr int WARPS_N = BN / WN, THREADS = (BM / WM) * WARPS_N * 32;
  constexpr int MI = WM / 16, NI = WN / 8, KS = BKB / 32;
  typedef Operand<BM, BK, ES, A_T, THREADS> A;
  typedef Operand<BN, BK, ES, B_T, THREADS> B;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sa = __cvta_generic_to_shared(smem);
  const uint32_t sb = sa + STAGES * A::BYTES;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm0 = (warp / WARPS_N) * WM, wn0 = (warp % WARPS_N) * WN;

  /* Grouped rasterisation: 8 row-tiles sweep the column tiles together, so
     that a column tile of b is read from L2 by the 8 blocks in flight. */
  const int tiles_m = (p.m + BM - 1) / BM, tiles_n = (p.n + BN - 1) / BN;
  const int pid = blockIdx.x, group = 8 * tiles_n, first = (pid / group) * 8;
  const int gsz = min(tiles_m - first, 8);
  const int tm = first + (pid % group) % gsz, tn = (pid % group) / gsz;
  const int m0 = tm * BM, n0 = tn * BN, z = blockIdx.z;

  const int kts = (p.k + BK - 1) / BK, split = blockIdx.y;
  const int kt0 = split * kts / p.splits;
  const int KT = (split + 1) * kts / p.splits - kt0;

  A a;
  B b;
  a.make((const char *)p.a + z * p.sa[0] * ES, (A_T ? p.sa[2] : p.sa[1]) * ES,
         p.m, m0, tid);
  b.make((const char *)p.b + z * p.sb[0] * ES, (B_T ? p.sb[2] : p.sb[1]) * ES,
         p.n, n0, tid);

  auto load = [&](int slot, int kt) {
    a.load(sa + slot * A::BYTES, kt0 + kt, p.k, tid);
    b.load(sb + slot * B::BYTES, kt0 + kt, p.k, tid);
  };

  /* The fragments of k-step [ks] of a slot. A row-major tile gives them by
     ldmatrix; a transposed one by ldmatrix.trans of the 8 x 8 blocks the
     fragment's registers hold, in their order. */
  auto frag = [&](uint32_t (&fa)[MI][4], uint32_t (&fb)[NI][2], int slot,
                  int ks) {
    const uint32_t ta = sa + slot * A::BYTES, tb = sb + slot * B::BYTES;
    const int q = lane >> 3, r8 = lane & 7;
#pragma unroll
    for (int i = 0; i < MI; i++) {
      if (A_T) {
        const int r = ks * 16 + r8 + 8 * (q >> 1), c = (wm0 + i * 16) / 8 + (q & 1);
        ldsm<true>(fa[i], ta + (r * A::C + swz(r, c, A::C)) * 16);
      } else {
        const int r = wm0 + i * 16 + (lane & 15), c = 2 * ks + (lane >> 4);
        ldsm<false>(fa[i], ta + (r * A::C + swz(r, c, A::C)) * 16);
      }
    }
#pragma unroll
    for (int j = 0; j < NI / 2; j++) {
      uint32_t t[4];
      if (B_T) {
        const int r = ks * 16 + r8 + 8 * (q & 1), c = (wn0 + j * 16) / 8 + (q >> 1);
        ldsm<true>(t, tb + (r * B::C + swz(r, c, B::C)) * 16);
      } else {
        const int r = wn0 + j * 16 + r8 + ((lane >> 4) << 3);
        const int c = 2 * ks + ((lane >> 3) & 1);
        ldsm<false>(t, tb + (r * B::C + swz(r, c, B::C)) * 16);
      }
      fb[2 * j][0] = t[0], fb[2 * j][1] = t[1];
      fb[2 * j + 1][0] = t[2], fb[2 * j + 1][1] = t[3];
    }
  };

  Acc acc[MI][NI][4] = {};
  uint32_t fa[2][MI][4], fb[2][NI][2];

#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < KT) load(s, s);
    cp_commit();
  }
  cp_wait<STAGES - 2>();
  __syncthreads();
  frag(fa[0], fb[0], 0, 0);

  /* Software pipeline: k-step ks multiplies the fragments loaded during
     ks - 1 while ldmatrix fetches ks + 1; the barrier for the next stage
     sits before the last k-step, so the next tile's first fragments load
     under its mma. The tile STAGES - 1 ahead refills the slot tile kt - 1
     freed: a's copies at the first k-step, b's at the last, which commits
     them as one group, so that the copies spread over the k-tile. */
  for (int kt = 0; kt < KT; kt++) {
    const int slot = kt % STAGES;
#pragma unroll
    for (int ks = 0; ks < KS; ks++) {
      const bool wrap = ks == KS - 1;
      if (wrap) {
        /* Tile kt + 1's group, which has STAGES - 3 committed after it:
           tile kt + STAGES - 1's commits after this wait. */
        cp_wait<STAGES - 3>();
        __syncthreads();
      }
      if (!wrap || kt + 1 < KT)
        frag(fa[(ks + 1) & 1], fb[(ks + 1) & 1],
             wrap ? (kt + 1) % STAGES : slot, (ks + 1) % KS);
      const int next = kt + STAGES - 1;
      if (ks == 0 && next < KT) a.load(sa + next % STAGES * A::BYTES, kt0 + next, p.k, tid);
      if (ks == KS - 1) {
        if (next < KT) b.load(sb + next % STAGES * B::BYTES, kt0 + next, p.k, tid);
        cp_commit();
      }
#pragma unroll
      for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NI; j++)
          mma<KIND>(acc[i][j], fa[ks & 1][i], fb[ks & 1][j]);
    }
  }
  cp_wait<0>();

  if (p.splits > 1) {
    Acc(&flat)[MI * NI * 4] = *reinterpret_cast<Acc(*)[MI * NI * 4]>(acc);
    const int tile = z * tiles_m * tiles_n + tm * tiles_n + tn;
    if (!combine(flat, (Acc *)p.partials + (size_t)tile * p.splits * MI * NI * 4 * THREADS,
                 p.tickets + tile, p.splits, split))
      return;
  }

  epilogue<Acc, BM, BN, WM, WN, THREADS, STAGES * (BM + BN) * BKB>(
      p, smem, acc, z, m0, n0, wm0, wn0);
}

#endif
