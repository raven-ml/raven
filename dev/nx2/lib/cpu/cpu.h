/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx.cpu's private interface: targets, jobs, walks and the stage.

   A kernel reads its operands through the door (nx_array.h) and coalesces
   them into a loop. The walk cuts the loop into 2-D blocks and hands them
   to a job, which runs them on the pool's threads. Per block the kernel
   reads each operand in place where the target's runs can, and otherwise
   stages it into a buffer in L1; it writes its result through the unstage,
   which converts as a cast does.

     kernel ─► walk ─► job ─► rig.pool threads ─► block
                                                   │
                     operand ─► stage ─► buffer ─► unstage ─► result
                                     (target's runs)
*/

#ifndef NX_CPU_H
#define NX_CPU_H

#include <stdint.h>
#include <string.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__AVX__)
#include <immintrin.h>
#elif defined(__x86_64__)
#include <emmintrin.h>
#endif

#include <caml/signals.h>

#include "nx_array.h"
#include "nx_kernel.h"
#include "rig_pool.h"

/* The counts of nx_kernel.h's kinds and constructors that nx.cpu's tables
   and switches take: a code added there fails nx.cpu's build here until
   nx.cpu computes it. */
_Static_assert(NX_OP1_COUNT == 31 && NX_OP2_COUNT == 18 && NX_OP3_COUNT == 2,
               "nx.cpu computes every kind");
_Static_assert(NX_LOAD_COUNT == 2 && NX_MONOID_COUNT == 4 &&
                   NX_REDUCTION_BLOCKS == 2 && NX_REDUCTION_CONSTANTS == 2 &&
                   NX_COMBINE_COUNT == 4 && NX_TRIANGLE_COUNT == 2,
               "nx.cpu reads every constructor");

/* Targets

   A target is the instruction set the kernels' runs are compiled for: one
   table per target, chosen once at initialisation. Every table computes the
   same bits. */

/* The target's vector, in bytes: 32 where the unit is compiled with AVX,
   as every unit of nx.cpu but target.c is on x86-64, else 16. */
#if defined(__AVX__)
#define NX_VECTOR_BYTES 32
#else
#define NX_VECTOR_BYTES 16
#endif

/* Whether a kernel that reads operand rows several times at a step other
   than one copies them into unit steps first: on x86-64, where strided
   loads cost more than the copy. In cache, adding rows 9 elements apart
   took 0.26 ns an element on an Arrow Lake core and 0.15 on an M1 Max
   one, whose strided adds outrun the copy. */
#if defined(__x86_64__)
#define NX_CPU_COPY_STRIDED 1
#else
#define NX_CPU_COPY_STRIDED 0
#endif

/* A run converts [n] contiguous elements at [src] into [dst]. */
typedef void (*nx_cpu_run)(const void *src, void *dst, int64_t n);

/* The conversions by source then destination (convert.c): a loop for
   every pair of a carrier and a dtype and from each narrow float to
   float32, and the runs where they outrun the loop, NULL elsewhere. */
extern const nx_cpu_run nx_cpu_convert_loops[NX_DTYPE_COUNT][NX_DTYPE_COUNT];
extern const nx_cpu_run nx_cpu_convert_runs[NX_DTYPE_COUNT][NX_DTYPE_COUNT];

/* The run that converts elements of the dtype [s] into the dtype [d], as a
   cast does, where s is a carrier, or where s is a narrow float and d
   float32. A sub-byte dtype has one element per byte, in its low bits:
   int4 and uint4 their value modulo 16, float4 its code, bit 0 or 1. */
static inline nx_cpu_run nx_cpu_convert(int s, int d) {
  nx_cpu_run r = nx_cpu_convert_runs[s][d];
  return r != NULL ? r : nx_cpu_convert_loops[s][d];
}

/* The lanes of a sum whose order is fixed by shape: term t of a block of
   NX_CPU_LANE_BLOCK consecutive terms adds into lane t modulo
   NX_CPU_LANES. */
#define NX_CPU_LANES 16
#define NX_CPU_LANE_BLOCK 1024

/* The lanes' tree: lane i takes lane i + 8 for i < 8, then i + 4, i + 2
   and i + 1, lane 0 then holding the block's value. [FOLD] is a statement
   folding lane [j] into lane [i]. */
#define NX_CPU_LANE_TREE(i, j, FOLD)               \
  for (int h_ = NX_CPU_LANES / 2; h_ > 0; h_ /= 2) \
    for (int i = 0, j = h_; i < h_; i++, j++) {    \
      FOLD;                                        \
    }

/* The blocks' tree: the binary tree whose left part holds the largest
   power of two of blocks below their count. A binary counter computes it
   in one pass over a stack of [top] values: after the [k]th value (from
   1) is pushed, NX_CPU_TREE_STEP folds the newest into the one below it
   once per trailing zero of k; after the last, NX_CPU_TREE_END folds what
   remains, newest first. [JOIN] is a statement folding stack entry
   top - 1 into top - 2. A stack of NX_CPU_TREE_DEPTH holds any count:
   2^48 blocks exceed any array. */
#define NX_CPU_TREE_DEPTH 48
#define NX_CPU_TREE_STEP(k, top, JOIN)                         \
  for (int64_t k_ = (k); (k_ & 1) == 0; k_ >>= 1, (top)--) { \
    JOIN;                                                      \
  }
#define NX_CPU_TREE_END(top, JOIN) \
  for (; (top) > 1; (top)--) {     \
    JOIN;                          \
  }

/* Where a microkernel's sums start: the tile's outputs, or +0, the tile
   then only written. */
typedef enum { NX_CPU_FROM_TILE, NX_CPU_FROM_ZERO } nx_cpu_from;

/* A contraction's microkernel adds to the [m] × [n] outputs at [c], n at
   most NR, output (i, j) at c + (i·ldc + j)·w, the products of [k] steps:
   step p's element of a's row i at a + (i·a_row + p·a_k)·w, of b's column
   j at b + (p·b_k + j)·w. It runs MR rows at a time, so that b's steps
   stay in the caches across them, and reads b as nx_cpu_b_read says. Each
   output adds its products in increasing p, each fused into its addition,
   to the start [from] names. */
typedef void (*nx_cpu_kernel)(int64_t m, int64_t n, int64_t k, const void *a,
                              int64_t a_row, int64_t a_k, const void *b,
                              int64_t b_k, void *c, int64_t ldc,
                              nx_cpu_from from);

/* How a microkernel reads a: an element at a time, so that a is read in
   place however its rows lie; or MR elements of a step at once, from a
   packed block, a_row 1, whose rows past m up to a multiple of MR are
   padding, any value (nx_cpu_pack). */
typedef enum { NX_CPU_A_ELEMENTS, NX_CPU_A_PACKED } nx_cpu_a_read;

/* How a microkernel reads b. NX_CPU_B_SLIVER: NR columns a step, so that
   a sliver of fewer columns is packed. NX_CPU_B_COLUMNS: the n columns it
   is given, so that such a sliver is read in place too. NX_CPU_B_HALVES:
   the n columns it is given where n is at least NR / 2, so that only a
   sliver of fewer is packed. */
typedef enum {
  NX_CPU_B_SLIVER,
  NX_CPU_B_COLUMNS,
  NX_CPU_B_HALVES
} nx_cpu_b_read;

/* A dot adds, for each of [r] contiguous rows of [a], row i at
   a + i·lda·w, r at most NX_CPU_DOT_ROWS, the [n] products with the
   contiguous [b] into row i's lanes at lanes + i·NX_CPU_LANES·w: term t
   into lane t modulo NX_CPU_LANES, each fused. */
#define NX_CPU_DOT_ROWS 4
typedef void (*nx_cpu_dot)(const void *a, int64_t lda, int r, const void *b,
                           int64_t n, void *lanes);

/* An axpy adds, for each of [r] rows, the element of [a] at a + i·lda·w
   times each of the [n] contiguous elements of [b] into row i's [n]
   contiguous accumulators at y + i·ldy·w, each fused. It prefetches the
   [n] elements at [next], the b of a later call. */
typedef void (*nx_cpu_axpy)(const void *a, int64_t lda, int r, const void *b,
                            int64_t n, void *y, int64_t ldy,
                            const void *next);

/* A lane tile sets y[j], for each of the [c] columns of b at [b], c at
   most NX_CPU_LANES, row t at b + t·b_k·w, to lane order's sum of one
   block of [k] products, k at most NX_CPU_LANE_BLOCK, of a's row, its
   element t at a + t·a_k·w, and the column: term t into lane t modulo
   NX_CPU_LANES, fused, the lanes summed by their tree (gemm.c). It reads
   no column past c. */
typedef void (*nx_cpu_lane_tile)(const void *a, int64_t a_k, const void *b,
                                 int64_t b_k, int64_t k, int64_t c,
                                 void *y);

/* A gemv adds into each of the [m] contiguous accumulators y[i] the [k]
   products of a's element at a + (i + p·lda)·w and x's [p]th, in
   increasing p, each fused: y += A x for the m × k A whose columns lie
   lda apart. */
typedef void (*nx_cpu_gemv)(int64_t m, int64_t k, const void *a, int64_t lda,
                            const void *x, void *y);

/* The float4 codes along k that MXFP4 scales as one block: a run. */
#define NX_CPU_F4_RUN 32

/* A float4 stage sets, for each of [lines] lines of NX_CPU_F4_RUN·[runs]
   float4 codes, two a byte from its low bits, line l's first at
   codes + l·sl, element u of line l at y + (u·du + l·dl)·w to the code's
   value times its run's scale, rounded once in acc: run r of line l's at
   s + (r·lines + l)·w. */
typedef void (*nx_cpu_f4)(const uint8_t *codes, int64_t sl, int64_t lines,
                          int64_t runs, const void *s, void *y, int64_t du,
                          int64_t dl);

/* A quads dot sets y[i·ldy + c], for each of 2 rows i of a, at a + i·lda,
   and 4 columns c of b, at b + c·ldb, to the sum of the products of the
   [n] words of the row and of the column: each word four elements of
   quads[d]'s 8-bit dtype along k, adjacent (gemm_quads.c). */
typedef void (*nx_cpu_quads_dot)(const uint32_t *a, int64_t lda,
                                 const uint32_t *b, int64_t ldb, int64_t n,
                                 uint32_t *y, int64_t ldy);

/* A quads axpy adds into y[i·ldy + j], for each of [rows] rows i of a,
   at a + i·lda, and [n] columns j, the sum over the [k] elements t of the
   row, adjacent, of a[t] · b[t·ldb + j], b's rows adjacent along their
   columns: elements of quads[d]'s 8-bit dtype, the sum in 32 bits
   (gemm_quads.c). A row of a holds zeros past k to a multiple of 4, which
   the axpy may read. It may prefetch the [k] rows of [n] elements at
   [next], [ldb] apart, the b of a later call. */
typedef void (*nx_cpu_quads_axpy)(const uint8_t *a, int64_t lda,
                                  int64_t rows, const uint8_t *b, int64_t ldb,
                                  int64_t k, int64_t n, uint32_t *y,
                                  int64_t ldy, const uint8_t *next);

/* A pack moves [n1] rows of [n0] elements of acc from [s], stepping [s0]
   along a row and [s1] across rows, one of them 1, into [d]: element i of
   row j to d + (j·pitch + i)·w. It may write any value to a row's
   elements past n0, up to the next multiple of the kernel's MR, for a's
   packed blocks, or of its NR, for b's slivers, which [pitch] leaves room
   for: a kernel's outputs from those elements lie past m or n and are
   never stored, and a kernel on hardware that slows on subnormal inputs
   masks its loads past n. It packs without the stage, whose moves serve
   every dtype and layout. */
typedef void (*nx_cpu_pack)(int64_t n0, int64_t n1, const void *s,
                            int64_t s0, int64_t s1, void *d, int64_t pitch);

/* A microkernel, its tile of MR × NR outputs, how it reads a and b, its
   speed: the flops it runs on one thread in the time memcpy moves a byte,
   which prices its jobs; and the cores that share the hardware it runs on:
   1 for a core's vector unit, a cluster's for the M1's matrix unit. */
typedef struct {
  nx_cpu_kernel f;
  int mr, nr;
  nx_cpu_a_read a;
  nx_cpu_b_read b;
  double flops_per_byte;
  int shared;
} nx_cpu_micro;

/* A contraction's kernels for chain order in one accumulator dtype. */
typedef struct nx_cpu_gemm {
  nx_cpu_micro kernel;
  int64_t mc, kc, nc;    /* the driver's blocks of rows, of k and of columns */
  int64_t few;           /* the rows at most which a product may stream b
                            past all of a (gemm.c's streams) */
  nx_cpu_pack pack;      /* or NULL: operands pack through the stage */
  const struct nx_cpu_gemm *other; /* or NULL: kernels that compute a
                                      product where they take less time */
  void (*enter)(void), (*leave)(void); /* or NULL: on a thread, its calls
                                          of kernel and pack run between
                                          enter and leave */
  double start; /* the cost, in bytes of memcpy, that each thread of a job
                   on these kernels adds to its start (nx_cpu_job_at); 0
                   where it adds none */
  double call;  /* the cost, in bytes of memcpy, that a product on these
                   kernels on one thread adds to its flops' (gemm.c's
                   fastest); 0 where it adds none */
} nx_cpu_gemm;

/* A row of a kind of one operand: n elements of [d] from [x], each stepping
   its own count of elements; of two, from [x] and [y]; of three, from [c],
   [x] and [y]. Each operand's elements and [d]'s are disjoint or identical:
   the same address and step. */
typedef void (*nx_cpu_row1)(int64_t n, uint8_t *d, int64_t sd,
                            const uint8_t *x, int64_t sx);
typedef void (*nx_cpu_row2)(int64_t n, uint8_t *d, int64_t sd,
                            const uint8_t *x, int64_t sx, const uint8_t *y,
                            int64_t sy);
typedef void (*nx_cpu_row3)(int64_t n, uint8_t *d, int64_t sd,
                            const uint8_t *c, int64_t sc, const uint8_t *x,
                            int64_t sx, const uint8_t *y, int64_t sy);
/* A fill's row: n elements of [d] set to the element whose bits are at
   [bits]. */
typedef void (*nx_cpu_row0)(int64_t n, uint8_t *d, int64_t sd,
                            const uint8_t *bits);

/* The rows (rows.c). op1[k][dt] and op2[k][dt] compute the kind of one or
   two operands k (nx_kernel.h's code) at dt, fma[dt] fma: NULL where
   Nx_kernel.Prog refuses the kind at dt. where[i] and fill[i] move
   elements of 2^i bytes, i in 0..4. */
typedef struct {
  nx_cpu_row1 op1[NX_OP1_COUNT][NX_DTYPE_COUNT];
  nx_cpu_row2 op2[NX_OP2_COUNT][NX_DTYPE_COUNT];
  nx_cpu_row3 fma[NX_DTYPE_COUNT];
  nx_cpu_row3 where[5];
  nx_cpu_row0 fill[5];
} nx_cpu_row_table;

extern const nx_cpu_row_table nx_cpu_rows;

/* The bytes of the largest tile of any target's microkernel: 32 × 32
   float32 on the matrix unit. */
#define NX_CPU_TILE 4096

/* The bytes of L2 cache of a core, as the host reports them when the
   program starts (target.c). */
extern int64_t nx_cpu_l2;
/* The bytes of L1 data cache of the cores a job runs on (target.c). */
extern int64_t nx_cpu_l1;
/* The bytes of the largest cache the host reports (target.c). */
extern int64_t nx_cpu_llc;

/* A target's table: the contraction kernels of its instructions. */
typedef struct {
  const char *name;
  /* gemm[acc] contracts in acc in chain order; its kernel is NULL where
     the target has none. dot[acc], axpy[acc] and tile[acc] are lane
     order's; gemv[dt] serves the factorisations' leaves (linalg.c).
     quads[d] contracts operands of the 8-bit dtype d into 32 bits in
     chain order, each element of a and b a 32-bit word of four of d along
     k, each product of words the sum of their four products
     (gemm_quads.c); its kernel is NULL where the target has none. quads_dot[d]
     and quads_axpy[d] are its dot and axpy for products of few rows, or
     NULL where the target has none; from quads_zip_rows rows (INT_MAX
     where it never does), quads[d]'s kernel over b's rows zipped into
     words outruns the axpy. */
  nx_cpu_gemm gemm[NX_DTYPE_COUNT];
  nx_cpu_gemm quads[NX_DTYPE_COUNT];
  nx_cpu_quads_dot quads_dot[NX_DTYPE_COUNT];
  nx_cpu_quads_axpy quads_axpy[NX_DTYPE_COUNT];
  int quads_zip_rows;
  nx_cpu_dot dot[NX_DTYPE_COUNT];
  nx_cpu_axpy axpy[NX_DTYPE_COUNT];
  nx_cpu_lane_tile tile[NX_DTYPE_COUNT];
  nx_cpu_gemv gemv[NX_DTYPE_COUNT];
  /* f4[acc] stages a scaled operand's float4 codes adjacent along k
     (gemm.c, stage_f4), in acc float32 or float64. */
  nx_cpu_f4 f4[NX_DTYPE_COUNT];
} nx_cpu_target;

/* The tables, each set when the program starts on a host that runs it, by
   gemm_generic.c, then by the kernels of the host's instructions:
   gemm_neon.c's on arm64, gemm_avx2.c's on x86-64, and gemm_vnni.c's
   quads where the host has AVX-VNNI-INT8. amx is base's table with
   gemm_amx.c's kernels for the matrix unit of Apple's M1. On x86-64 every
   unit but target.c is compiled for x86-64-v3, and base is set only on a
   host that has it (target.c). */
extern nx_cpu_target nx_cpu_base;
void nx_cpu_set_gemm(nx_cpu_target *t);
#if defined(__aarch64__)
void nx_cpu_set_neon(nx_cpu_target *t);
#endif
#if defined(__APPLE__) && defined(__aarch64__)
extern nx_cpu_target nx_cpu_amx;
int nx_cpu_has_amx(void);
void nx_cpu_set_amx(nx_cpu_target *t);
#endif
#if defined(__x86_64__)
void nx_cpu_set_avx2(nx_cpu_target *t);
void nx_cpu_set_vnni(nx_cpu_target *t);
#endif

/* The table the kernels run: the best one whose instructions the host
   has. */
extern const nx_cpu_target *nx_cpu_table;

/* The dtype a block of [dt] is staged in: float32 for a narrow float, int8
   for int4, uint8 for uint4, bool for bit, and [dt] itself otherwise. */
static inline int nx_cpu_carrier(int dt) {
  switch (dt) {
    case NX_FLOAT16:
    case NX_BFLOAT16:
    case NX_FLOAT8_E4M3FN:
    case NX_FLOAT8_E5M2:
    case NX_FLOAT4_E2M1FN: return NX_FLOAT32;
    case NX_INT4: return NX_INT8;
    case NX_UINT4: return NX_UINT8;
    case NX_BIT: return NX_BOOL;
    default: return dt;
  }
}

/* The bytes a staged element of [dt] takes: a sub-byte one takes a byte. */
static inline int nx_cpu_width(int dt) {
  int bits = nx_dtype_row_of(dt).bits;
  return bits < 8 ? 1 : bits / 8;
}

/* The index of a width of 1, 2, 4, 8 or 16 bytes in the rows by
   width (nx_cpu_row_table's where and fill): its log2. */
static inline int nx_cpu_width_index(int bytes) {
  return __builtin_ctz((unsigned)bytes);
}

/* The elements of [a]. */
static inline double nx_cpu_elements(const nx_array *a) {
  double n = 1;
  for (int i = 0; i < a->rank; i++) n *= (double)a->dim[i];
  return n;
}

/* The bytes of [a]'s elements, each nx_cpu_width wide. */
static inline double nx_cpu_bytes(const nx_array *a) {
  return nx_cpu_elements(a) * nx_cpu_width(a->dtype);
}

/* Calls and jobs

   A call keeps other domains waiting to collect for at most about one
   core's copy of NX_CPU_HOLD_BYTES, and never waits on another domain's
   job while it holds the domain lock.

   One number decides both whether a call holds the domain lock and how
   many threads its jobs take: a work's cost, in bytes one core's memcpy
   moves in the work's time, the reference nx.cpu's one-thread copy of a
   filled 128 KiB array. Each family computes it by one function, at its
   external's entry for the whole call, every pass, fill and copy it runs
   inline included, and for each job on the job's part of the work. The
   order is always cost, release, threads: between nx_read and nx_done an
   external calls nx_cpu_release with its call's cost, then runs its work,
   whose jobs each take the threads their share pays for (nx_cpu_threads),
   then nx_cpu_reacquire. The domain lock is released for the work where
   the cost reaches NX_CPU_RELEASE_BYTES, and a held call runs its jobs on
   the calling thread alone (job.c). The work reads no OCaml value and
   raises nothing, held or released, so every call that passes
   nx_cpu_release reaches nx_cpu_reacquire. A call reads many operands
   through nx_read_each, and a long program or many reductions as
   NX_CPU_ITEM says, so that no count of them lengthens a held stretch;
   a table sized by such a count is allocated and freed with the lock
   released (nx_alloc_released, nx_free_released).

   NX_CPU_HOLD_BYTES is 1 MiB: one core copies it in 12 to 15 us (M1 Max,
   fudge's Core Ultra 5 235), and a release and reacquisition cost 9 to
   150 ns, at most 1% of that, with or without systhreads and another
   domain collecting. NX_CPU_RELEASE_BYTES, a quarter of it, is the least
   cost at which a job takes a second thread, so a call whose jobs could
   use the pool releases first. That quarter is the one margin: a held
   call's work stays within NX_CPU_HOLD_BYTES while no cost underestimates
   its work by more than four times, but for the plans a transform makes
   on a cache's miss, which fft.c holds within NX_CPU_HOLD_BYTES.

   A cost is the work's time on one core in three classes:
   - streaming: work that moves its elements in order, a cost an element
     (nx_cpu_loop_cost);
   - lines: an element or term that touches a cache line of its own,
     NX_CPU_LINE each;
   - scalar: work bound by arithmetic, its operations times their cost:
     an elementwise kind's per element (nx_cpu_kind_cost), a family's own
     for its routines.
   Each family keeps its values in one table, which names the machines
   and the date they were measured. The values bound their class: each is
   1.5 times the most one core took of its class's members, on the M1 Max
   and fudge under the timing lock, rounded up to a power of two, and
   names its worst member; a value measured on one of them alone names
   it. */

/* The most a held call may cost. */
#define NX_CPU_HOLD_BYTES (1024 * 1024)

/* The cost a job takes for each of its threads (nx_cpu_threads). */
#define NX_CPU_THREAD_BYTES (128 * 1024)

/* The least price at which a call releases the domain lock: the least
   cost at which a job takes a second thread. */
#define NX_CPU_RELEASE_BYTES (2 * NX_CPU_THREAD_BYTES)

/* An item of a call's own reading and analysis, before its work: a
   program's node or a reduction, each copied, checked and priced. A node
   took about 20 ns (M1 alone, a 20000-node map under load), 1.6 KiB of
   memcpy. A call whose items cost NX_CPU_RELEASE_BYTES releases the domain
   lock before it analyses them, and a held loop that reads them from
   OCaml lets other domains collect (nx_let_collect) every NX_CPU_ITEMS
   items, a release's worth. */
#define NX_CPU_ITEM 4096
#define NX_CPU_ITEMS (NX_CPU_RELEASE_BYTES / NX_CPU_ITEM)

/* A cache line's fetch for an element or term that touches its own, on
   top of the term's own price: Logsumexp along strided float32 terms, its
   worst member, took 900 a term (M1 alone), which this and the term's 512
   (reduce.c) cover. */
#define NX_CPU_LINE 1024

/* The cost of computing the kind [k] of [arity] operands, nx_kernel.h's
   code, at one element of [dt] that it computes in [as]: the kind's at
   [as], and [dt]'s staging where [as] is another dtype (rows.c). */
double nx_cpu_kind_cost(int arity, int k, int dt, int as);

/* The cost of casting an element of [s] to [d] as copy.c does, through
   [s]'s carrier: the cast's at the wider dtype, the staging of each of
   [s] and [d] that is not its own carrier, and a rounding from float64
   where [d] is a narrow float and [s]'s carrier is not float32 (rows.c,
   ROUND_COST). */
double nx_cpu_cast_cost(int s, int d);

/* What an element of [dt] that a kind computes in [as] adds for its
   staging: none where [as] is [dt] (rows.c). */
double nx_cpu_stage_cost(int dt, int as);

/* Whether the calling thread runs a held call's work, whose jobs then run
   on it alone (job.c): nx_cpu_release writes it at every call's entry and
   nx_cpu_reacquire clears it at its exit. */
extern _Thread_local int nx_cpu_held;

/* Whether the calling thread runs every job on itself alone, as a test of
   nx_cpu.ml's one_thread asks: the bits are those of any threads. */
extern _Thread_local int nx_cpu_one;

/* Releases the domain lock for a call's work that costs [price],
   NX_CPU_RELEASE_BYTES or more, else marks the call held; answers
   whether it released, for nx_cpu_reacquire. */
static inline int nx_cpu_release(double price) {
  nx_cpu_held = price < NX_CPU_RELEASE_BYTES;
  if (nx_cpu_held) return 0;
  /* The release that runs no pending signal handler: one that raised
     there would leave the door's claims held. */
  caml_enter_blocking_section_no_pending();
  return 1;
}

/* Reacquires the domain lock if nx_cpu_release released it, else ends
   the held call. Pending handlers run once the external returns. */
static inline void nx_cpu_reacquire(int released) {
  if (released) caml_leave_blocking_section();
  else nx_cpu_held = 0;
}

/* Chunks a job cuts its units into per thread: a thread that finishes its
   strip takes the chunks left in the others', so a thread that runs
   slower, an efficiency core's say, runs fewer. */
#define NX_CPU_CHUNKS 8

/* Runs [body] over the units [0, total) of work that touches [bytes] bytes
   and costs [cost] (Calls and jobs), its share of its call's cost, on as
   many of the pool's threads as the cost pays for: [body] reads no OCaml
   value, as a worker is no OCaml thread. The cost is a double: one that
   multiplies extents of several operands may pass an int64_t. A body may
   begin a job of its own: it runs on the body's thread alone where the
   outer job runs on several (rig_pool.h, Scheduling). */
void nx_cpu_job(int64_t total, int64_t bytes, double cost, rig_pool_body body,
                void *ctx);

/* The threads nx_cpu_job gives such work, if it has as many units: at
   least 1, one per NX_CPU_THREAD_BYTES of cost, the bytes a core moves in
   the time a job takes to start, and 1 in a held call or under
   nx_cpu_one (job.c). */
int nx_cpu_threads(int64_t bytes, double cost);

/* nx_cpu_job and nx_cpu_threads with one thread per [thread_bytes] of
   cost: the bytes a core moves in the time a job of that work takes to
   start, where it starts slower than a job's NX_CPU_THREAD_BYTES. */
void nx_cpu_job_at(int64_t total, int64_t bytes, double cost,
                   double thread_bytes, rig_pool_body body, void *ctx);
int nx_cpu_threads_at(int64_t bytes, double cost, double thread_bytes);

/* Boxes (copy.c) */

/* A box: [extent[0]] planes of [extent[1]] rows of [extent[2]] elements, in
   C order, as nx_loop's axes. Element (p, i, j) of operand k, 0 the
   destination and 1 the source, lies at position
   first[k] + p·step[k][0] + i·step[k][1] + j·step[k][2], counted in
   elements from its base. Positions are at least 0; steps have either
   sign. */
typedef struct {
  int64_t extent[3];
  int64_t first[2];
  int64_t step[2][3]; /* elements, by operand then axis */
} nx_box;

/* Copies the box [b] of elements of [bits] bits from [src] to [dst], bits
   for bits. The destination's elements are distinct and share no byte with
   the source's. Bytes that hold only copied elements take plain stores; a
   byte shared with other elements takes a compare-and-swap, as
   nx_sub_store does. Each plane copies as one block: rows of adjacent
   elements are memcpy; where one operand steps one element across rows and
   the other one along them, whichever operand is transposed, it moves
   square blocks through registers, 8x8 of 4-byte elements, 32 bytes from
   each of eight columns into each of eight rows, and 4x4 of the others.
   Two adjacent runs of the source interleaved into adjacent elements, as a
   window two elements wide reads its rows, zip in registers. The case is
   chosen once per box. */
void nx_copy_box(uint8_t *dst, const uint8_t *src, const nx_box *b,
                 int bits);

/* nx_copy_box, its square blocks of 4- and 8-byte elements storing each
   row's 32 bytes past the caches (nx_stream32) where the destination's
   rows, and each plane's pieces of them, lie on whole 64-byte lines,
   drained before it returns: for a destination far larger than the
   caches. A line that plain and streamed stores share, or that a stream
   fills in parts far apart in time, is written in parts: on x86-64 a
   transposed 4000x4000 float32 copy in tiles 65 elements wide took twice
   as long streamed. */
void nx_stream_box(uint8_t *dst, const uint8_t *src, const nx_box *b,
                   int bits);

/* Stores past the caches

   A store into a line the cache lacks first reads the line. A result far
   larger than the caches, written in pieces that lie apart, as a
   transpose's rows are, reads each line it writes: stores that bypass the
   caches read none. NX_STREAMS is 1 where the target has them: x86-64's
   movntdq, arm64's stnp. Which results gain is each kernel's rule. */

#if defined(__x86_64__) || defined(__aarch64__)
#define NX_STREAMS 1
#else
#define NX_STREAMS 0
#endif

/* Stores the 32 bytes at [s] at [d], on 32 bytes, past the caches where
   NX_STREAMS: one AVX store or two SSE2 ones on x86-64, arm64's stnp in
   assembly, since clang lowers __builtin_nontemporal_store to plain
   stores there. Inline, a source in registers stays in them. */
static inline __attribute__((always_inline)) void nx_stream32(void *d,
                                                              const void *s) {
#if defined(__AVX__)
  _mm256_stream_si256((__m256i *)d, _mm256_loadu_si256((const __m256i *)s));
#elif defined(__x86_64__)
  _mm_stream_si128((__m128i *)d, _mm_loadu_si128((const __m128i *)s));
  _mm_stream_si128((__m128i *)d + 1,
                   _mm_loadu_si128((const __m128i *)s + 1));
#elif defined(__aarch64__)
  uint8x16_t lo = vld1q_u8((const uint8_t *)s);
  uint8x16_t hi = vld1q_u8((const uint8_t *)s + 16);
  __asm__ volatile("stnp %q0, %q1, [%2]" : : "w"(lo), "w"(hi), "r"(d)
                   : "memory");
#else
  __builtin_memcpy(d, s, 32);
#endif
}

/* Orders the stores past the caches before the stores that follow: x86's
   are weakly ordered and need a fence. arm64's stnp relaxes ordering only
   for a load's address dependency (the non-temporal hint), so like any
   store it is ordered by the release with which a job's worker leaves
   (rig_pool.c) and the acquire with which its caller sees the job end. */
static inline void nx_stream_drain(void) {
#if defined(__x86_64__)
  _mm_sfence();
#endif
}

/* Scratch */

/* [bytes] of scratch on 128 bytes, NULL if they cannot be allocated: one
   of the calling thread's kept buffers where one is free and [bytes] fit
   in it (job.c), else a new allocation. nx_cpu_give gives it back. */
void *nx_cpu_take(int64_t bytes);
void nx_cpu_give(void *p);

/* Contraction views (gemm_view.c) */

/* A contraction's operands and result grouped into four axes, its arrays
   indexed by the operands and axes below. [offset] is each operand's first
   element and [stride] its stride along each axis, in elements; an axis an
   operand lacks, and an absent Init, read 0. Kernels write
   v.stride[NX_VIEW_A][NX_VIEW_CONTRACTED]. */
enum { NX_VIEW_A, NX_VIEW_B, NX_VIEW_INIT, NX_VIEW_DST };
enum { NX_VIEW_BATCH, NX_VIEW_ROW, NX_VIEW_COLUMN, NX_VIEW_CONTRACTED };

typedef struct {
  int64_t extent[4];    /* by axis */
  int64_t offset[4];    /* by operand */
  int64_t stride[4][4]; /* by operand, then axis */
} nx_contract_view;

/* A contraction's groups as runs, indexed as nx_contract_view's arrays:
   each group's axes coalesced over its operands (nx_coalesce_dims) into
   [runs[g]] runs, at least one, in C order. Run [r] of group [g] has
   [extent[g][r]] elements, and operand [o] steps [stride[o][g][r]] along
   it, 0 in a group it lacks and for an absent init. A view is the runs
   where every group has one. */
typedef struct {
  int runs[4];
  int64_t extent[4][NX_MAX_RANK];
  int64_t stride[4][4][NX_MAX_RANK];
  int64_t offset[4];
} nx_contract_runs;

/* An array's layout as nx_array_layout reads it: its rank, its extents
   then its strides, and its first element. */
typedef struct {
  int rank;
  int64_t dim[2 * NX_MAX_RANK];
  int64_t offset;
} nx_layout;

/* Groups the axes of the arrays [a] and [b], the init [i], an option, and
   the result [y], over the pairs [batch] and [contracting], as
   Nx_kernel.S.contract takes them, into [w]: one axis of each kind, where
   every operand lays out each group's axes as one run of strides, as a
   matrix product reads them. [a] has the batch, row and contracted axes,
   [b] the batch, column and contracted ones, [i] and [y] the batch, row
   and column ones. It reads each pair once and answers an nx_array.h
   code: NX_SHAPE if the arrays do not fit the pairs (ranks the pairs do
   not fit, an axis in two pairs, or extents that differ within a group);
   otherwise NX_DECLINED, with [r] filled, where a group makes several
   runs; NX_OK once grouped. It allocates nothing and raises nothing. */
int nx_contract_fill(nx_contract_view *w, nx_contract_runs *r, value batch,
                     value contracting, value y, value a, value b, value i);

/* Groups as nx_contract_fill does, the layouts read already: [y], [a],
   [b] and the init [i], NULL for none, over the [nb] batch pairs then the
   [nc] contracting pairs [pairs], each a's axis then b's. It reads no
   OCaml value. */
int nx_contract_fill_layouts(nx_contract_view *w, nx_contract_runs *r,
                             int nb, int nc, const int (*pairs)[2],
                             const nx_layout *y, const nx_layout *a,
                             const nx_layout *b, const nx_layout *i);

/* Matrices */

/* A contraction's order of summation (gemm.c). */
enum { NX_CPU_CHAIN = 0, NX_CPU_LANE = 1 };

/* Stores into op[NX_VIEW_DST] the contraction in [acc], float32 or
   float64, and in [order], of op[NX_VIEW_A], op[NX_VIEW_B] and
   op[NX_VIEW_INIT] (or NULL), arrays of [acc] laid out as [v] (gemm.c).
   Answers 0 if an allocation failed. */
int nx_cpu_contract_on(int acc, int order, const nx_contract_view *v,
                       const nx_array *const *op);

/* The speed (nx_cpu_micro) of the core's vector kernel of products in
   [acc], float32 or float64: the table's, or its other kernels' where the
   table's runs on a unit that cores share, as the M1's matrix unit
   (gemm.c). It prices work whose units each run on one thread. */
double nx_cpu_core_speed(int acc);

/* Walks */

/* A block of a loop: [n2] planes of [n1] rows of [n0] elements. Operand k's
   first element is at position at[k] from its base, and it steps s0[k]
   elements along a row, s1[k] from a row to the next and s2[k] from a plane
   to the next. */
typedef struct {
  int64_t n0, n1, n2;
  int64_t at[NX_MAX_OPERANDS];
  int64_t s0[NX_MAX_OPERANDS];
  int64_t s1[NX_MAX_OPERANDS];
  int64_t s2[NX_MAX_OPERANDS];
} nx_cpu_block;

typedef void (*nx_cpu_block_fn)(const nx_cpu_block *b, void *ctx);

/* Calls [f] on blocks of at most [most] elements that cover the loop [l]
   over the [n] operands [a], operand 0 the written one, each element once,
   from a job, or on the calling thread for a loop that one block holds
   and one thread runs. An element of [f] costs [per], and the job takes
   the threads nx_cpu_walk_threads gives. Order and threads decide no bit:
   [f] computes each element from operands' elements at its index alone.
   Inline, so that a kernel's call of [f] on its one block is direct
   (walk.c, rule 0). */
static inline void nx_cpu_walk(int n, const nx_array *a, const nx_loop *l,
                               int64_t most, double per, nx_cpu_block_fn f,
                               void *ctx);

/* Stores the element of bits [bits] into the elements of block [b] of the
   array [d], its operand 0: rows by rows.c's fill of their width, a
   sub-byte dtype's rows of unit step as runs (nx_array.h) (apply.c). */
void nx_cpu_fill_block(const nx_array *d, const nx_cpu_block *b,
                       const uint8_t *bits);

/* A walk that moves elements one by one, through tiles or along strides,
   costs as much time as this many times its bytes moved by memcpy: a
   transposed 512x512 float32 copy, tile by tile, takes 4.6 times memcpy's
   time on one kimchi core. */
#define NX_CPU_STRIDED_COST 4

/* The cost (Calls and jobs) of a walk of the loop [l] over the [n]
   operands [a], [per] an element: NX_CPU_STRIDED_COST times that where
   the walk's blocks are tiles, an operand steps more than one element
   along its rows, or the walk runs several rows shorter than its SHORT,
   as it orders the loop's axes (walk.c). An
   elementwise call's cost at its entry, [per] its kind's
   (nx_cpu_kind_cost). */
double nx_cpu_loop_cost(int n, const nx_array *a, const nx_loop *l,
                        double per);

/* The threads the walk of the loop [l] over the [n] operands [a], [per] an
   element, takes: nx_cpu_threads of its operands' bytes and its cost. */
int nx_cpu_walk_threads(int n, const nx_array *a, const nx_loop *l,
                        double per);

/* Places

   A loop's operands placed by the virtual position of an index, its
   place in C order of the loop's shape, as an index operand of no buffer
   reads it: a kernel of more operands than a block holds (map.c, reduce.c)
   walks or reduces the virtual positions and places each operand from
   them. The loop of the virtual operand and the [n] operands coalesced
   has [rank] axes of extents [ext], along which virtual positions step
   [vs], C-contiguous; operand k's first element is at first[k], and it
   steps step[k][d] along axis d. A position is placed along each axis of
   more than one element; a step moves along one axis, [s / vs[d]] places
   along axis d, the innermost whose step divides it into fewer places
   than its extent. */
typedef struct {
  int n, rank;
  int64_t ext[NX_MAX_RANK], vs[NX_MAX_RANK];
  const int64_t *first;
  int64_t (*step)[NX_MAX_RANK];
} nx_cpu_places;

/* Each operand's position at the virtual position [p], into at[k]. */
static inline void nx_cpu_places_at(const nx_cpu_places *pl, int64_t p,
                                    int64_t *at) {
  for (int k = 0; k < pl->n; k++) at[k] = pl->first[k];
  for (int d = 0; d < pl->rank; d++) {
    if (pl->ext[d] <= 1) continue;
    int64_t i = p / pl->vs[d] % pl->ext[d];
    for (int k = 0; k < pl->n; k++) at[k] += i * pl->step[k][d];
  }
}

/* Operand [k]'s position at the virtual position [p]. */
static inline int64_t nx_cpu_place(const nx_cpu_places *pl, int k,
                                   int64_t p) {
  int64_t at = pl->first[k];
  for (int d = 0; d < pl->rank; d++)
    if (pl->ext[d] > 1) at += p / pl->vs[d] % pl->ext[d] * pl->step[k][d];
  return at;
}

/* The axis the virtual step [s] moves along, or -1 for no step. */
static inline int nx_cpu_places_axis(const nx_cpu_places *pl, int64_t s) {
  int d = pl->rank - 1;
  while (d >= 0 && (s == 0 || pl->ext[d] <= 1 || s % pl->vs[d] != 0 ||
                    s / pl->vs[d] >= pl->ext[d]))
    d--;
  return d;
}

/* Each operand's step along the virtual step [s], into st[k]: 0 for no
   step. */
static inline void nx_cpu_places_along(const nx_cpu_places *pl, int64_t s,
                                       int64_t *st) {
  int d = nx_cpu_places_axis(pl, s);
  for (int k = 0; k < pl->n; k++)
    st[k] = d >= 0 ? s / pl->vs[d] * pl->step[k][d] : 0;
}

/* Operand [k]'s step along the virtual step [s]. */
static inline int64_t nx_cpu_place_step(const nx_cpu_places *pl, int k,
                                        int64_t s) {
  int d = nx_cpu_places_axis(pl, s);
  return d >= 0 ? s / pl->vs[d] * pl->step[k][d] : 0;
}

/* nx_cpu_walk for a loop that rule 0 does not run (walk.c, rules 1 to
   3). */
void nx_cpu_cut(int n, const nx_array *a, const nx_loop *l, int64_t most,
                double per, nx_cpu_block_fn f, void *ctx);

static inline void nx_cpu_walk(int n, const nx_array *a, const nx_loop *l,
                               int64_t most, double per, nx_cpu_block_fn f,
                               void *ctx) {
  if (l->rank > 1 || l->extent[0] > most ||
      nx_cpu_walk_threads(n, a, l, per) > 1) {
    nx_cpu_cut(n, a, l, most, per, f, ctx);
    return;
  }
  nx_cpu_block b = {.n0 = l->extent[0], .n1 = 1, .n2 = 1};
  for (int k = 0; k < n; k++) {
    b.at[k] = l->first[k];
    b.s0[k] = l->step[k][0];
  }
  if (b.n0 > 0) f(&b, ctx);
}

/* Copies */

/* Copies the elements of [bits] bits of the loop [l]'s operand 1 from [src]
   to its operand 0 at [dst], bits for bits, as a copy's walk does
   (copy.c), an element costing [per], the caller's share for it. */
void nx_cpu_copy_loop(uint8_t *dst, const uint8_t *src, int bits,
                      const nx_loop *l, double per);

/* Casts the loop [l]'s operand 1, [src], into its operand 0, [dst], as a
   cast's walk does (copy.c): each element rounds once to [dst]'s dtype and
   costs [per], the caller's share for it. */
void nx_cpu_cast_loop(const nx_array *dst, const nx_array *src,
                      const nx_loop *l, double per);

/* Programs

   A map's program as map.c runs it, block by block, for a kernel that
   reads a program's outputs as its terms, as reduce.c does. The program
   is copied out of OCaml before the door, which can run OCaml code, bound
   to the loads the door read, then run over blocks of them: each node
   computes as a map's does, so an output's elements have the map's bits.

   A bound program places its loads, then, where it reads a coordinate, the
   index, by the virtual positions of its loop (Places): the index's are
   the virtual positions themselves. */
typedef struct nx_cpu_prog nx_cpu_prog;

/* The Prog.t at the root [vp] copied, or NULL if host memory runs out.
   Reads [vp] alone, letting other domains collect every NX_CPU_ITEMS
   nodes. */
nx_cpu_prog *nx_cpu_prog_copy(const value *vp);
void nx_cpu_prog_free(nx_cpu_prog *j);

/* [j]'s nodes, its loads and outputs, and their dtypes. */
int nx_cpu_prog_nodes(const nx_cpu_prog *j);
int nx_cpu_prog_ins(const nx_cpu_prog *j);
int nx_cpu_prog_in(const nx_cpu_prog *j, int k);
int nx_cpu_prog_outs(const nx_cpu_prog *j);
int nx_cpu_prog_out(const nx_cpu_prog *j, int o);

/* Whether output [o] of [j] is the product of two of its loads, each read
   at its own dtype or a float cast into the product's: its dtype, float32
   or float64, with the loads into [k], else -1. A load of another dtype
   than the product's, staged as nx_cpu_stage_as stages it, gives that
   cast's elements. */
int nx_cpu_prog_product(const nx_cpu_prog *j, int o, int *k);

/* Binds [j] to its loads [a], read by the door, over a loop of [rank] axes
   of extents [shape]: NX_OK, NX_SHAPE for a coordinate past the rank, or
   -1 if host memory runs out. Sets [index] to whether the program reads
   the index. */
int nx_cpu_prog_bind(nx_cpu_prog *j, const nx_array *a, int rank,
                     const int64_t *shape, int *index);

/* The dtype output [o] of [j] is held in: its own, a narrow float's
   codes included, or its carrier for a sub-byte dtype. */
int nx_cpu_prog_held(const nx_cpu_prog *j, int o);

/* Runs the bound [j] over [n1] rows of [n0] elements from the virtual
   position [p], stepping [s0] along a row and [s1] across rows, each a
   step along one axis, its loads then the index placed by [pl], and
   stores its output [o] into [out], C-contiguous in its held dtype: row
   r's element i at out + (r·n0 + i)·w, w that dtype's width. */
void nx_cpu_prog_terms(const nx_cpu_prog *j, const nx_cpu_places *pl,
                       int o, int64_t p, int64_t s0, int64_t n0, int64_t s1,
                       int64_t n1, uint8_t *out);

/* What an element of [j] costs (Calls and jobs): the kinds it runs, each
   at its cost (nx_cpu_kind_cost). */
double nx_cpu_prog_cost(const nx_cpu_prog *j);

/* Whether a run of [j] could not have an arena or its tables on the heap,
   as a program of many values or operands needs: its outputs are then
   unspecified. */
int nx_cpu_prog_failed(const nx_cpu_prog *j);

/* Sorts of 64-bit keys with their positions (sort.c, index.c) */

/* The bytes of the span [r] of keys, at least 1. */
static inline int nx_cpu_span_bytes(uint64_t r) {
  int b = 1;
  while (b < 8 && r >> (8 * b) != 0) b++;
  return b;
}

/* Sorts the [n] keys [k] with their positions [p] by insertion, stably. */
static inline void nx_cpu_insertion(uint64_t *k, int64_t *p, int64_t n) {
  for (int64_t i = 1; i < n; i++) {
    uint64_t x = k[i];
    int64_t y = p[i], j = i;
    for (; j > 0 && k[j - 1] > x; j--) {
      k[j] = k[j - 1];
      p[j] = p[j - 1];
    }
    k[j] = x;
    p[j] = y;
  }
}

/* Sorts the [n] >= 1 keys [k] with their positions [p] by LSD radix,
   stably, through [k2] and [p2] of n entries: by the [kb] low bytes of
   each key less [lo], a digit of 8 bits a pass, skipping a digit every
   key shares. Answers whether the sorted entries ended in k2 and p2. */
static inline int nx_cpu_radix(uint64_t *k, int64_t *p, uint64_t *k2,
                               int64_t *p2, int64_t n, int kb, uint64_t lo) {
  int64_t count[8][256];
  memset(count, 0, (size_t)kb * sizeof count[0]);
  for (int64_t i = 0; i < n; i++)
    for (int d = 0; d < kb; d++) count[d][((k[i] - lo) >> (8 * d)) & 255]++;
  int moved = 0;
  for (int d = 0; d < kb; d++) {
    int64_t *c = count[d], at = 0;
    if (c[((k[0] - lo) >> (8 * d)) & 255] == n) continue;
    for (int b = 0; b < 256; b++) {
      int64_t x = c[b];
      c[b] = at;
      at += x;
    }
    for (int64_t i = 0; i < n; i++) {
      int64_t j = c[((k[i] - lo) >> (8 * d)) & 255]++;
      k2[j] = k[i];
      p2[j] = p[i];
    }
    uint64_t *kt = k;
    k = k2;
    k2 = kt;
    int64_t *pt = p;
    p = p2;
    p2 = pt;
    moved = !moved;
  }
  return moved;
}

/* The stage */

/* The bytes of a staged block: a walk for the stage takes blocks of at most
   NX_CPU_SLOT / w elements, w the widest form its elements take, so that a
   block's buffers stay in L1. */
#define NX_CPU_SLOT (16 * 1024)

/* Copies operand [k] of block [b], the array [a], into [dst] in its carrier:
   row j's element i at [dst + j·row + i·w], w the carrier's width. */
void nx_cpu_stage(const nx_array *a, const nx_cpu_block *b, int k, uint8_t *dst,
                  int64_t row);

/* Writes [src], elements of the dtype [c] with row j's element i at
   [src + j·row + i·w], w [c]'s width, into operand [k] of block [b], the
   array [a], converted as a cast does. [c] is a carrier. */
void nx_cpu_unstage(const nx_array *a, const nx_cpu_block *b, int k,
                    const uint8_t *src, int64_t row, int c);

/* Copies the [n1] rows of [n0] elements of [a] from position [at], each
   stepping [s0] along a row and [s1] across rows, into [dst] in the
   carrier [d], converted as a cast does: row j's element i at
   dst + (j·pitch + i)·w, w [d]'s width. An operand of the dtype [d] is
   one stage, whatever its size; another goes through a slot, in pieces
   whose every form fits it. */
void nx_cpu_stage_as(const nx_array *a, int64_t at, int64_t s0, int64_t s1,
                     int64_t n0, int64_t n1, int d, uint8_t *dst,
                     int64_t pitch);

#endif /* NX_CPU_H */
