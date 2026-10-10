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

#include "nx_array.h"
#include "nx_spec.h"
#include "rig_pool.h"

/* Targets

   A target is the instruction set the kernels' runs are compiled for: one
   table per target, chosen once at initialisation. Every table computes the
   same bits. */

/* A run converts [n] contiguous elements at [src] into [dst]. */
typedef void (*nx_cpu_run)(const void *src, void *dst, int64_t n);

/* The lanes of a sum whose order is fixed by shape: term t of a block of
   NX_CPU_FOLD_BLOCK consecutive terms adds into lane t modulo
   NX_CPU_LANES. */
#define NX_CPU_LANES 16
#define NX_CPU_FOLD_BLOCK 1024

/* A contraction's microkernel adds to the tile of MR × NR outputs at [c],
   output (i, j) at c + (i·ldc + j)·w, the products of [k] steps of packed
   operands: step p holds MR elements of a at a + p·lda·w and NR of b at
   b + p·NR·w. Each output adds its products in increasing p, each fused
   into its addition. */
typedef void (*nx_cpu_kernel)(int64_t k, const void *a, int64_t lda,
                              const void *b, void *c, int64_t ldc);

/* A dot adds the [n] products of the contiguous [a] and [b] into the lanes
   at [lanes], term t into lane t modulo NX_CPU_LANES, each fused. */
typedef void (*nx_cpu_dot)(const void *a, const void *b, int64_t n,
                           void *lanes);

/* A microkernel and its tile of MR × NR outputs. */
typedef struct {
  nx_cpu_kernel f;
  int mr, nr;
} nx_cpu_micro;

/* A contraction's kernels for chain order in one accumulator dtype. */
typedef struct {
  nx_cpu_micro kernel;
  int64_t mc, kc, nc;    /* the driver's blocks of rows, of k and of columns */
  nx_cpu_micro thin[3];  /* of 1, 2 and 4 rows; f NULL where none */
} nx_cpu_gemm;

/* A row of a kind of one operand: n elements of [d] from [x], each stepping
   its own count of elements; of two, from [x] and [y]; of three, from [c],
   [x] and [y]. */
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

/* A monoid's fold at a dtype (folds.c). [lanes] adds the [n] terms at [x],
   stepping [s] elements, into the NX_CPU_LANES lanes at [l]: term i into
   lane (first + i) mod NX_CPU_LANES. [combine] adds the [n] elements at
   [x], stepping [s], into the [n] contiguous accumulators at [a], element
   by element. [scan] runs [k] sequences, at most 4, side by side: sequence
   j adds the [n] elements at [x + j·xs], stepping [s], into the
   accumulator [a]'s element j in order, storing each sum at
   [y + j·ys], stepping [sy]. [blocks] stores into [v] the values of the [n] blocks of
   NX_CPU_FOLD_BLOCK contiguous terms from [x], each its lanes from the
   identity [e] and the lanes' tree, as [lanes] and [combine] give them.
   [few] stores into [y], stepping [sy], the values of [w] outputs of [n]
   terms each, n at most NX_CPU_LANES: output j's term t at
   [x + j·s + off[t]], in lane t from the identity [e], then the lanes'
   tree, as one block of [lanes] gives them. [column] adds into each of the
   [w] contiguous accumulators at [a] in turn the [n] rows of [x], row i
   at [x + i·st] and its element j [s] further than j - 1: accumulator j
   takes row 0's element j, then row 1's, and so on. */
typedef struct {
  void (*lanes)(const uint8_t *x, int64_t s, int64_t n, uint8_t *l,
                int first);
  void (*combine)(uint8_t *a, const uint8_t *x, int64_t s, int64_t n);
  void (*scan)(uint8_t *a, const uint8_t *x, int64_t s, int64_t xs,
               uint8_t *y, int64_t sy, int64_t ys, int64_t n, int k);
  void (*blocks)(const uint8_t *x, int64_t n, const uint8_t *e, uint8_t *v);
  void (*few)(const uint8_t *x, const int64_t *off, int n, int64_t s,
              int64_t w, const uint8_t *e, uint8_t *y, int64_t sy);
  void (*column)(uint8_t *a, const uint8_t *x, int64_t st, int64_t n,
                 int64_t s, int64_t w);
} nx_cpu_fold;

/* The bytes of the largest tile of any target's microkernel: 8 × 12
   float32 on arm64, 6 × 16 on x86-64, 1 × 96 thin. */
#define NX_CPU_TILE 384

typedef struct {
  const char *name;
  /* convert[s][d] converts elements of the dtype s into the dtype d, as a
     cast does, where s is a carrier, or where s is a narrow float and d
     float32. A sub-byte dtype has one element per byte, in its low bits:
     int4 and uint4 their value modulo 16, float4 its code, bit 0 or 1. */
  nx_cpu_run convert[NX_DTYPE_COUNT][NX_DTYPE_COUNT];
  /* gemm[acc] contracts in acc in chain order; its kernel is NULL where
     the target has none. dot[acc] is lane order's dot. */
  nx_cpu_gemm gemm[NX_DTYPE_COUNT];
  nx_cpu_dot dot[NX_DTYPE_COUNT];
  /* op1[k][dt] and op2[k][dt] compute the kind of one or two operands k
     (nx_spec.h's code) at dt, fma[dt] fma: NULL where the table declines
     (rows.c). */
  nx_cpu_row1 op1[NX_OP1_COUNT][NX_DTYPE_COUNT];
  nx_cpu_row2 op2[NX_OP2_COUNT][NX_DTYPE_COUNT];
  nx_cpu_row3 fma[NX_DTYPE_COUNT];
  /* where[i] and fill[i] move elements of 2^i bytes, i in 0..4. */
  nx_cpu_row3 where[5];
  nx_cpu_row0 fill[5];
  /* fold[m][dt] folds the monoid m (nx_spec.h's NX_SUM to NX_MIN) at dt:
     lanes NULL where the table declines. */
  nx_cpu_fold fold[NX_MIN + 1][NX_DTYPE_COUNT];
} nx_cpu_target;

/* The tables, each set when the program starts on a host that runs it, by
   convert.c, gemm_generic.c, rows.c and folds.c compiled for its target,
   then by the kernels of its instructions: gemm_neon.c's on arm64,
   gemm_avx2.c's for v3. */
extern nx_cpu_target nx_cpu_base;
void nx_cpu_set_convert_base(nx_cpu_target *t);
void nx_cpu_set_gemm_base(nx_cpu_target *t);
void nx_cpu_set_rows_base(nx_cpu_target *t);
void nx_cpu_set_folds_base(nx_cpu_target *t);
#if defined(__aarch64__)
void nx_cpu_set_neon(nx_cpu_target *t);
#endif
#if defined(__x86_64__)
extern nx_cpu_target nx_cpu_v3;
void nx_cpu_set_convert_v3(nx_cpu_target *t);
void nx_cpu_set_gemm_v3(nx_cpu_target *t);
void nx_cpu_set_rows_v3(nx_cpu_target *t);
void nx_cpu_set_folds_v3(nx_cpu_target *t);
void nx_cpu_set_avx2(nx_cpu_target *t);
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

/* Jobs */

/* Runs [body] over the units [0, total) of work that touches [bytes] bytes
   and takes as long as memcpy of [cost] bytes, on as many of the pool's
   threads as the cost pays for. A job of more than one thread, or a long
   one, runs with the runtime released: [body] reads no OCaml value. */
void nx_cpu_job(int64_t total, int64_t bytes, int64_t cost, rig_pool_body body,
                void *ctx);

/* The threads nx_cpu_job gives such work, if it has as many units: at
   least 1. */
int nx_cpu_threads(int64_t bytes, int64_t cost);

/* Folds */

/* nx.cpu's reduce and scan (fold.c) of the descriptor [s], the
   destinations [dsts] and the operands [ops]: on one thread where
   [threads] is 1, on as many as the job gives where it is 0. The bits do
   not change. */
value nx_cpu_reduce_on(value s, value dsts, value ops, int threads);
value nx_cpu_scan_on(value s, value dsts, value ops, int threads);

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
   from a job, or on the calling thread for a loop that one block holds.
   Order and threads decide no bit: [f] computes each element from
   operands' elements at its index alone. Inline, so that a kernel's call
   of [f] on its one block is direct (walk.c, rule 0). */
static inline void nx_cpu_walk(int n, const nx_array *a, const nx_loop *l,
                               int64_t most, nx_cpu_block_fn f, void *ctx);

/* nx_cpu_walk for a loop that one block does not hold (walk.c, rules 1 to
   3). */
void nx_cpu_cut(int n, const nx_array *a, const nx_loop *l, int64_t most,
                nx_cpu_block_fn f, void *ctx);

static inline void nx_cpu_walk(int n, const nx_array *a, const nx_loop *l,
                               int64_t most, nx_cpu_block_fn f, void *ctx) {
  if (l->rank > 1 || l->extent[0] > most) {
    nx_cpu_cut(n, a, l, most, f, ctx);
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
   (copy.c). */
void nx_cpu_copy_loop(uint8_t *dst, const uint8_t *src, int bits,
                      const nx_loop *l);

/* Padded loads */

/* Fills [y] with the shape an operand of shape [x] and rank [r] has once
   padded by [p], its windows' axes last, as Spec.shapes computes it.
   Answers NX_OK, or NX_SHAPE where Spec.shapes answers [Error]: [r] other
   than [p]'s rank, an extent below zero or past int64, or a window that
   does not fit its padded axis. Reads [p]'s geometry only once [r] fits
   it. */
int nx_cpu_padded_shape(const nx_spec_pad *p, int r, const int64_t *x,
                        int64_t *y);

/* Fills [out] with the padded load [p] of the operand [a]: a plain
   descriptor over a C-contiguous copy of [a] padded with [p]'s fill, laid
   out by [p]'s windows, which a kernel reads as any operand (assemble.c).
   [a]'s shape is one nx_cpu_padded_shape answers NX_OK for. The copy
   lives in C-heap memory at [out->base], which the caller frees; it is
   NULL for a load with no element. Answers 0, or 1 if host memory runs
   out, having allocated nothing. Reads no OCaml value. */
int nx_cpu_unpad(const nx_array *a, const nx_spec_pad *p, nx_array *out);

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
