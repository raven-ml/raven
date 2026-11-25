/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx.metal's kernels as the host and the device both see them: the kernel
   table and each kernel's parameter words.

   A kernel reads its parameters at buffer 0 as a struct of 64-bit fields,
   field i at byte 8 i, which the host writes as words at the indices this
   header's enumerations name; rig turns each address field into the
   address of a buffer of the submit. Every fact both sides use is an
   enumeration's constant or a row's column, so that kernels/gen.py writes
   kernels.ml from what the pinned metal reads here. Formulas of extents
   live on the host alone, which passes their values as words. */

#ifndef NX_METAL_KERNELS_H
#define NX_METAL_KERNELS_H

/* X(name, family, threads, words) for every kernel of the metallib, in the
   order of kernels.ml's variant: the family whose pipelines are made
   together at the first call that launches one of them, the threads of
   each of its groups and the words of its parameters. A family lists its
   rows in its own macro, which its unit expands to define its kernels. */
#define NX_METAL_KERNELS(X)                                                   \
  NX_METAL_COPY_KERNELS(X) NX_METAL_LOOP_KERNELS(X)                           \
  NX_METAL_CONTRACT_KERNELS(X) NX_METAL_FFT_KERNELS(X)                        \
  NX_METAL_SORT_KERNELS(X) NX_METAL_FACTOR_KERNELS(X)                         \
  NX_METAL_ROUTINE_KERNELS(X) NX_METAL_INDEX_KERNELS(X)                       \
  NX_METAL_SPECTRAL_KERNELS(X)                                                \
  NX_METAL_ATTENTION_KERNELS(X)

/* The loaders' keys: L(X, b, B, a, A, T) for each, b's dtype and
   scale [b], a's dtype [a], each also in upper case, and the staged
   dtype [T], float32: the scaled weights nx.quant makes, each by a of
   float32, float16 or bfloat16. b of float4_e2m1fn by Exponent scales
   (MXFP4), of int8 by Value scales of float16 (Q8_0) and of int8 by
   Value scales of float32 (Q6_K's). */
#define NX_CONTRACT_LOADER_KEYS(L, X)                                         \
  L(X, e2m1_by_exp, E2M1_BY_EXP, f32, F32, float)                             \
  L(X, e2m1_by_exp, E2M1_BY_EXP, f16, F16, float)                             \
  L(X, e2m1_by_exp, E2M1_BY_EXP, bf16, BF16, float)                           \
  L(X, i8_by_f16, I8_BY_F16, f32, F32, float)                                 \
  L(X, i8_by_f16, I8_BY_F16, f16, F16, float)                                 \
  L(X, i8_by_f16, I8_BY_F16, bf16, BF16, float)                               \
  L(X, i8_by_f32, I8_BY_F32, f32, F32, float)                                 \
  L(X, i8_by_f32, I8_BY_F32, f16, F16, float)                                 \
  L(X, i8_by_f32, I8_BY_F32, bf16, BF16, float)

#define NX_LOADER_FAMILIES(X, b, B, a, A, T)                                  \
  NX_FAMILY_LOADER_##B##_##A##_KN, NX_FAMILY_LOADER_##B##_##A##_NK,

/* The keys of the banded loaders, L(X, b, B): b's of each loader key, a
   cast into float32. */
#define NX_CONTRACT_LOADER_BANDS(L, X)                                        \
  L(X, e2m1_by_exp, E2M1_BY_EXP)                                              \
  L(X, i8_by_f16, I8_BY_F16)                                                  \
  L(X, i8_by_f32, I8_BY_F32)

#define NX_LOADER_BAND_FAMILIES(X, b, B) NX_FAMILY_LOADER_##B##_BAND,

/* Families */

enum {
  /* Kernels any family's call may launch: the zeroing of fresh tickets. */
  NX_FAMILY_TRUNK,
  /* The copies of maps that move bits. */
  NX_FAMILY_COPY,
  /* Loops' maps. */
  NX_FAMILY_MAP,
  /* Loops' reductions, which scans of several chunks a slice also run. */
  NX_FAMILY_REDUCE,
  /* Loops' scans. */
  NX_FAMILY_SCAN,
  /* Loops of 64-bit indices, past 2^32 elements. */
  NX_FAMILY_WIDE,
  /* Direct reductions of a float load along its row, a kernel for each
     reduction and dtype. */
  NX_FAMILY_ROW,
  /* Casts of one load into float32, float16 or bfloat16, a kernel for
     each source and target. */
  NX_FAMILY_CAST,
  /* Contractions' tiles, one family for each staged dtype and b's
     layout, k x n or n x k. */
  NX_FAMILY_CONTRACT_F32_KN,
  NX_FAMILY_CONTRACT_F32_NK,
  NX_FAMILY_CONTRACT_F16_KN,
  NX_FAMILY_CONTRACT_F16_NK,
  NX_FAMILY_CONTRACT_BF16_KN,
  NX_FAMILY_CONTRACT_BF16_NK,
  /* Integer contractions' tiles on the half units, for b's layouts. */
  NX_FAMILY_CONTRACT_I8_KN,
  NX_FAMILY_CONTRACT_I8_NK,
  /* Contractions' finish. */
  NX_FAMILY_FINISH,
  /* Complex contractions' tiles, which stage b's real blocks, for b's
     layouts. */
  NX_FAMILY_CONTRACT_C64_KN,
  NX_FAMILY_CONTRACT_C64_NK,
  /* Scaled operands' elements times their scales, and cast operands'
     bands. */
  NX_FAMILY_CONTRACT_SCALE,
  /* The SIMT kernel's sums, their integer finish, and Contracting
     segments' prepass. */
  NX_FAMILY_CONTRACT_SIMT,
  /* Rows segments' prepass and grouped tiles. */
  NX_FAMILY_CONTRACT_ROWS,
  /* Loaders, one family for each key and b's layout. */
  NX_CONTRACT_LOADER_KEYS(NX_LOADER_FAMILIES, _)
  /* The vector kernels, one family for each key of the wider
     operand. */
  NX_FAMILY_CONTRACT_MV_F32,
  NX_FAMILY_CONTRACT_MV_F16,
  NX_FAMILY_CONTRACT_MV_BF16,
  NX_FAMILY_CONTRACT_MV_E2M1,
  NX_FAMILY_CONTRACT_MV_I8F16,
  NX_FAMILY_CONTRACT_MV_I8F32,
  /* The twins of each key that has them (NX_CONTRACT_MV_TWINS), apart
     from its kernels, so that a call compiles only the family it
     launches. */
  NX_FAMILY_CONTRACT_MV_E2M1_W32,
  NX_FAMILY_CONTRACT_MV_I8F16_W32,
  NX_FAMILY_CONTRACT_MV_I8F32_W32,
  /* The Fourier transforms' moves. */
  NX_FAMILY_FFT,
  /* Their moves of 64-bit offsets, for a side past 2^31 - 1 elements. */
  NX_FAMILY_FFT_WIDE,
  /* Sorts and numberings. */
  NX_FAMILY_SORT,
  /* The factorisations' steps at float32, at complex64, and those that
     take either. */
  NX_FAMILY_FACTOR_F32,
  NX_FAMILY_FACTOR_C64,
  NX_FAMILY_FACTOR,
  /* The routines' products' float32 tiles, plain and negated, for each
     layout of b, and their copies. */
  NX_FAMILY_CONTRACT_F32_RT_KN,
  NX_FAMILY_CONTRACT_F32_RT_NK,
  NX_FAMILY_CONTRACT_F32_NEG_KN,
  NX_FAMILY_CONTRACT_F32_NEG_NK,
  NX_FAMILY_CONTRACT_COPY,
  /* The steps specialised by function constants, in fft_steps.metallib,
     whose pipelines are made a key at a time. */
  NX_FAMILY_FFT_STEPS,
  /* Gathers, scatters, assemblies and folds (index.metal). */
  NX_FAMILY_INDEX,
  /* eigh's and svd's steps at float32 and at complex64. */
  NX_FAMILY_EIGH_F32,
  NX_FAMILY_EIGH_C64,
  NX_FAMILY_SVD_F32,
  NX_FAMILY_SVD_C64,
  /* eigh's steps of either format. */
  NX_FAMILY_EIGH,
  /* The steps eigh and svd share. */
  NX_FAMILY_SPECTRAL,
  /* svd's steps of either format. */
  NX_FAMILY_SVD,
  /* eig's steps at float32, at complex64, and of either format. */
  NX_FAMILY_EIG_F32,
  NX_FAMILY_EIG_C64,
  NX_FAMILY_EIG,
  /* Attention. */
  NX_FAMILY_ATTENTION,
  /* The tiles that read an operand through its band, plain and grouped,
     for each staged dtype, and the banded loaders for each key of b. */
  NX_FAMILY_CONTRACT_F32_BAND,
  NX_FAMILY_CONTRACT_F16_BAND,
  NX_FAMILY_CONTRACT_BF16_BAND,
  NX_CONTRACT_LOADER_BANDS(NX_LOADER_BAND_FAMILIES, _)
};

/* Copies (copy.metal) */

#define NX_METAL_COPY_KERNELS(X)                                              \
  X(copy, NX_FAMILY_COPY, NX_COPY_THREADS, NX_COPY_WORDS)                     \
  X(copy_8, NX_FAMILY_COPY, NX_COPY_THREADS, NX_COPY_WORDS)                   \
  X(copy_16, NX_FAMILY_COPY, NX_COPY_THREADS, NX_COPY_WORDS)                  \
  X(copy_32, NX_FAMILY_COPY, NX_COPY_THREADS, NX_COPY_WORDS)                  \
  X(copy_64, NX_FAMILY_COPY, NX_COPY_THREADS, NX_COPY_WORDS)                  \
  X(copy_run, NX_FAMILY_COPY, NX_COPY_RUN_THREADS, NX_COPY_RUN_WORDS)         \
  X(copy_rows, NX_FAMILY_COPY, NX_COPY_ROWS_THREADS, NX_COPY_ROWS_WORDS)      \
  X(copy_rows_wide, NX_FAMILY_COPY, NX_COPY_ROWS_THREADS, NX_COPY_ROWS_WORDS) \
  X(copy_step_8, NX_FAMILY_COPY, NX_COPY_ROWS_THREADS, NX_COPY_ROWS_WORDS)    \
  X(copy_step_16, NX_FAMILY_COPY, NX_COPY_ROWS_THREADS, NX_COPY_ROWS_WORDS)   \
  X(copy_step_32, NX_FAMILY_COPY, NX_COPY_ROWS_THREADS, NX_COPY_ROWS_WORDS)   \
  X(copy_step_64, NX_FAMILY_COPY, NX_COPY_ROWS_THREADS, NX_COPY_ROWS_WORDS)   \
  X(widen, NX_FAMILY_COPY, NX_COPY_THREADS, NX_COPY_WORDS)                    \
  X(zero, NX_FAMILY_TRUNK, NX_ZERO_THREADS, NX_ZERO_WORDS)

/* An axis of a copy's source: its extent, its stride in elements, a
   signed 64-bit integer, and its extent's divisor (device.h, divide), 0
   for an extent past 32 bits. */
enum {
  NX_AXIS_EXTENT,
  NX_AXIS_STRIDE,
  NX_AXIS_DIVISOR,
  NX_AXIS_WORDS,
};

/* The most axes a copy's parameters hold: 8, which fill rig's 256 bytes
   of parameters a launch. */
enum { NX_COPY_HOLDS = 8 };

/* copy: the elements [first] to [count] - 1 of a source, in C order of
   their indices, each into the element of its number from [dst]. An
   element is [units] units of [unit] bytes, 1, 2, 4 or 8, which both ends
   are aligned on. The source's element at index (i₀, …, i₍ᵣ₋₁₎) lies at
   [src] + Σ iⱼ sⱼ elements, the extents eⱼ and strides sⱼ, [rank] of
   each, being NX_AXIS_WORDS words an axis from the outermost: from
   [held] where [rank] is at most NX_COPY_HOLDS, and from [axes]
   otherwise, a constant buffer. A copy too long for one launch's time
   runs as launches of its pieces, each from its own [first].

   copy_8, copy_16, copy_32 and copy_64: as copy, of a launch whose
   [count] is at most 2^32 - NX_COPY_THREADS, so that each of its
   threads' indices fits 32 bits, the last group's past [count]
   included, and whose [unit] is 1, 2, 4 or 8 bytes, 8 to 64 bits.

   widen: as copy, of a source of a dtype narrower than a byte, each
   element cast into a byte of the dtype that holds its values: [unit] is
   the source's dtype code, plus the byte's times 256, and [units] the
   bit of [src]'s byte at which the source's position 0 starts. */
enum {
  NX_COPY_DST,
  NX_COPY_SRC,
  NX_COPY_AXES,
  NX_COPY_FIRST,
  NX_COPY_COUNT,
  NX_COPY_RANK,
  NX_COPY_UNIT,
  NX_COPY_UNITS,
  NX_COPY_HELD,
  NX_COPY_WORDS = NX_COPY_HELD + NX_AXIS_WORDS * NX_COPY_HOLDS,
};

/* A copy's group: each thread copies one element. */
enum { NX_COPY_THREADS = 256 };

/* copy_run: the bytes of a run of consecutive bytes from [src] into the
   same bytes from [dst], up to byte [end] - 1, a thread a block of
   NX_COPY_RUN_BYTES bytes of [dst]'s memory from the launch's block
   [first]. [dst]'s first byte lies [head] bytes into its block, and the
   source's bytes for a block lie [shift] bytes past a multiple of
   NX_COPY_RUN_BYTES: (src - dst) modulo NX_COPY_RUN_BYTES. A thread reads
   the one or two aligned words that hold its source bytes, or, where a
   block is clipped by either end or its words would reach past them,
   copies its bytes one at a time. A run too long for one launch's time
   runs as launches of its blocks, each from its own [first] to its own
   [end]. */
enum {
  NX_COPY_RUN_DST,
  NX_COPY_RUN_SRC,
  NX_COPY_RUN_FIRST,
  NX_COPY_RUN_END,
  NX_COPY_RUN_HEAD,
  NX_COPY_RUN_SHIFT,
  NX_COPY_RUN_WORDS,
};

/* The bytes a thread of copy_run copies, and its group's threads. */
enum { NX_COPY_RUN_BYTES = 16 };
enum { NX_COPY_RUN_THREADS = 256 };

/* The most axes copy_rows' parameters hold, the row's included: 7, which
   with its other words fill rig's 256 bytes of parameters a launch. */
enum { NX_COPY_ROWS_HOLDS = 7 };
enum { NX_COPY_ROWS_THREADS = 256 };

/* copy_rows: the bytes of a source of rows, each of consecutive bytes,
   into consecutive bytes from [dst], up to byte [end] - 1 of [dst], a
   thread a block of NX_COPY_RUN_BYTES bytes of [dst]'s memory from the
   launch's block [first], as copy_run. Its [rank] axes, NX_AXIS_WORDS
   words each from the outermost, from [held] where [rank] is at most
   NX_COPY_ROWS_HOLDS and from [axes] otherwise, are in bytes: the last
   is the row, of its extent of bytes, at least NX_COPY_RUN_BYTES, and
   stride 1; the row's byte at index (i₀, …, i₍ᵣ₋₁₎) lies at [src] +
   Σ iⱼ sⱼ. [dst]'s first byte lies [head] bytes into its block, and
   [src] [shift] bytes past a multiple of NX_COPY_RUN_BYTES. Every byte
   the source's rows reach lies in [src] + [lo] to [src] + [hi] - 1: a
   thread reads the aligned words that hold its block's bytes, at most
   two a row, where they lie in that span, and copies its bytes one at a
   time where they do not or the block is clipped by either end. Its
   launch's [end] is at most 2^32 - NX_COPY_THREADS, so that its threads'
   byte indices fit 32 bits, and every extent, the row's in bytes
   included, is below 2^32, so that its coordinates' bounds do.

   copy_rows_wide: as copy_rows, of any [end], its byte indices 64-bit.

   copy_step_8, copy_step_16, copy_step_32 and copy_step_64: as copy_rows,
   of a source of rows of every other element of 1, 2, 4 or 8 bytes, its
   elements and [dst]'s on multiples of their width: the last axis is
   the row, of its extent of elements, at least NX_COPY_RUN_BYTES bytes
   of them, at a stride of twice the element's width. A thread copies its
   block's elements one at a time where they cross a row's end. Every
   extent is below 2^32, as [end] is. */
enum {
  NX_COPY_ROWS_DST,
  NX_COPY_ROWS_SRC,
  NX_COPY_ROWS_AXES,
  NX_COPY_ROWS_FIRST,
  NX_COPY_ROWS_END,
  NX_COPY_ROWS_HEAD,
  NX_COPY_ROWS_SHIFT,
  NX_COPY_ROWS_RANK,
  NX_COPY_ROWS_LO,
  NX_COPY_ROWS_HI,
  NX_COPY_ROWS_HELD,
  NX_COPY_ROWS_WORDS = NX_COPY_ROWS_HELD + NX_AXIS_WORDS * NX_COPY_ROWS_HOLDS,
};

/* zero: [count] 32-bit words from [dst]; fresh tickets, which a call's
   first launch zeroes. */
enum {
  NX_ZERO_DST,
  NX_ZERO_COUNT,
  NX_ZERO_WORDS,
};

enum { NX_ZERO_THREADS = 256 };

/* A row's kernels (loop.metal, by_row), a float load's direct reduction
   along its row: R(X, name, reduction, dtype, how, element) for each
   reduction a direct reduction takes and each float dtype it decodes,
   and Logsumexp's and Moments' second passes of a float32 load. */
#define NX_LOOP_ROWS(R, X, r, rk)                                              \
  R(X, loop_row_##r##_f32, rk, NX_FLOAT32, NX_DIRECT_AS_IS, uint)              \
  R(X, loop_row_##r##_bf16, rk, NX_BFLOAT16, NX_DIRECT_BF16, ushort)           \
  R(X, loop_row_##r##_f16, rk, NX_FLOAT16, NX_DIRECT_F16, ushort)
#define NX_METAL_ROW_KERNELS(R, X)                                             \
  NX_LOOP_ROWS(R, X, sum, NX_LOOP_SUM)                                         \
  NX_LOOP_ROWS(R, X, max, NX_LOOP_MAX)                                         \
  NX_LOOP_ROWS(R, X, min, NX_LOOP_MIN)                                         \
  NX_LOOP_ROWS(R, X, argmax, NX_LOOP_ARG_MAX)                                  \
  NX_LOOP_ROWS(R, X, argmin, NX_LOOP_ARG_MIN)                                 \
  R(X, loop_row_logsumexp_f32, NX_LOOP_EXPS, NX_FLOAT32, NX_DIRECT_AS_IS,      \
    uint)                                                                      \
  R(X, loop_row_moments_f32, NX_LOOP_SQUARES, NX_FLOAT32, NX_DIRECT_AS_IS,     \
    uint)
#define NX_ROW_ROW(X, name, rk, dt, how, E)                                    \
  X(name, NX_FAMILY_ROW, NX_LOOP_THREADS, NX_LOOP_WORDS)

/* A cast's kernels (direct.metal, cast_map), a map of one plain load
   cast into a dtype of 8, 16, 32 or 64 bits but complex64: one kernel a
   source class and target width, the target read from its descriptor;
   but int64 and uint64 into 64 bits, which are copies (moves.ml). A
   class is a source dtype, read as its element, or the byte or the
   nibble sources, which take every result from a table:
   C(X, name, class, element, width), by NX_CAST_WIDTHS for the four
   widths or NX_CAST_NARROW for those below 64. A cast's sources and
   targets, in
   the order of their tables:
   - NX_CAST_BYTE_SOURCES and NX_CAST_NIBBLE_SOURCES, the classes'
     NX_CAST_BYTES and NX_CAST_NIBBLES;
   - NX_CAST_TARGETS_8, _16, _32 and _64, by store width.
   The classes' codes lie past every dtype's (direct.metal). */
enum { NX_CAST_BYTES = 64, NX_CAST_NIBBLES };
#define NX_CAST_BYTE_SOURCES(S)                                                \
  S(NX_UINT8) S(NX_INT8) S(NX_BOOL) S(NX_FLOAT8_E4M3FN) S(NX_FLOAT8_E5M2)
#define NX_CAST_NIBBLE_SOURCES(S) S(NX_UINT4) S(NX_INT4) S(NX_FLOAT4_E2M1FN)
#define NX_CAST_TARGETS_8(T)                                                   \
  T(NX_INT8) T(NX_UINT8) T(NX_BOOL) T(NX_FLOAT8_E4M3FN) T(NX_FLOAT8_E5M2)
#define NX_CAST_TARGETS_16(T)                                                  \
  T(NX_FLOAT16) T(NX_BFLOAT16) T(NX_INT16) T(NX_UINT16)
#define NX_CAST_TARGETS_32(T) T(NX_FLOAT32) T(NX_INT32) T(NX_UINT32)
#define NX_CAST_TARGETS_64(T) T(NX_INT64) T(NX_UINT64)
#define NX_CAST_NARROW(C, X, c, K, E)                                          \
  C(X, loop_cast_##c##_8, K, E, 8)                                             \
  C(X, loop_cast_##c##_16, K, E, 16)                                           \
  C(X, loop_cast_##c##_32, K, E, 32)
#define NX_CAST_WIDTHS(C, X, c, K, E)                                          \
  NX_CAST_NARROW(C, X, c, K, E) C(X, loop_cast_##c##_64, K, E, 64)
#define NX_METAL_CAST_KERNELS(C, X)                                            \
  NX_CAST_WIDTHS(C, X, f32, NX_FLOAT32, uint)                                  \
  NX_CAST_WIDTHS(C, X, f16, NX_FLOAT16, ushort)                                \
  NX_CAST_WIDTHS(C, X, bf16, NX_BFLOAT16, ushort)                              \
  NX_CAST_NARROW(C, X, i64, NX_INT64, long)                                    \
  NX_CAST_NARROW(C, X, u64, NX_UINT64, ulong)                                  \
  NX_CAST_WIDTHS(C, X, i32, NX_INT32, int)                                     \
  NX_CAST_WIDTHS(C, X, u32, NX_UINT32, uint)                                   \
  NX_CAST_WIDTHS(C, X, i16, NX_INT16, short)                                   \
  NX_CAST_WIDTHS(C, X, u16, NX_UINT16, ushort)                                 \
  NX_CAST_WIDTHS(C, X, c64, NX_COMPLEX64, ulong)                               \
  NX_CAST_WIDTHS(C, X, bit, NX_BIT, uchar)                                     \
  NX_CAST_WIDTHS(C, X, byte, NX_CAST_BYTES, uchar)                             \
  NX_CAST_WIDTHS(C, X, nibble, NX_CAST_NIBBLES, uchar)
#define NX_CAST_ROW(X, name, K, E, W)                                          \
  X(name, NX_FAMILY_CAST, NX_LOOP_THREADS, NX_LOOP_WORDS)

/* Loops (loop.metal and its headers tape.h and fold.h, which state how
   they run, and direct.metal) */

/* A narrow float's direct maps (NX_LOOP_BITS16), as direct32's. */
#define NX_DIRECT16_KERNELS(X, c)                                              \
  X(loop_map_direct16_##c, NX_FAMILY_MAP, NX_LOOP_THREADS, NX_LOOP_WORDS)      \
  X(loop_map_direct16_##c##_tiled, NX_FAMILY_MAP, NX_LOOP_THREADS,             \
    NX_LOOP_WORDS)                                                             \
  X(loop_map_direct16_##c##_long, NX_FAMILY_MAP, NX_LOOP_THREADS,              \
    NX_LOOP_WORDS)                                                             \
  X(loop_map_direct16_##c##_long_tiled, NX_FAMILY_MAP, NX_LOOP_THREADS,        \
    NX_LOOP_WORDS)

#define NX_METAL_LOOP_KERNELS(X)                                              \
  X(loop_map_v4, NX_FAMILY_MAP, NX_LOOP_THREADS, NX_LOOP_WORDS)               \
  X(loop_map_v1, NX_FAMILY_MAP, NX_LOOP_THREADS, NX_LOOP_WORDS)               \
  X(loop_map_tiled, NX_FAMILY_MAP, NX_LOOP_THREADS, NX_LOOP_WORDS)            \
  X(loop_map_padded, NX_FAMILY_MAP, NX_LOOP_THREADS, NX_LOOP_WORDS)           \
  X(loop_map_direct, NX_FAMILY_MAP, NX_LOOP_THREADS, NX_LOOP_WORDS)           \
  X(loop_map_direct_tiled, NX_FAMILY_MAP, NX_LOOP_THREADS, NX_LOOP_WORDS)     \
  X(loop_map_direct32, NX_FAMILY_MAP, NX_LOOP_THREADS, NX_LOOP_WORDS)         \
  X(loop_map_direct32_tiled, NX_FAMILY_MAP, NX_LOOP_THREADS, NX_LOOP_WORDS)   \
  X(loop_map_direct32_long, NX_FAMILY_MAP, NX_LOOP_THREADS, NX_LOOP_WORDS)    \
  X(loop_map_direct32_long_tiled, NX_FAMILY_MAP, NX_LOOP_THREADS,             \
    NX_LOOP_WORDS)                                                             \
  NX_DIRECT16_KERNELS(X, bf16)                                                \
  NX_DIRECT16_KERNELS(X, f16)                                                 \
  X(loop_map_direct8, NX_FAMILY_MAP, NX_LOOP_THREADS, NX_LOOP_WORDS)          \
  X(loop_map_direct8_tiled, NX_FAMILY_MAP, NX_LOOP_THREADS, NX_LOOP_WORDS)    \
  NX_METAL_CAST_KERNELS(NX_CAST_ROW, X)                                       \
  X(loop_map_direct32_flat, NX_FAMILY_MAP, NX_LOOP_THREADS, NX_LOOP_WORDS)    \
  X(loop_map_direct32_long_flat, NX_FAMILY_MAP, NX_LOOP_THREADS,              \
    NX_LOOP_WORDS)                                                             \
  X(loop_map_wide, NX_FAMILY_WIDE, NX_LOOP_THREADS, NX_LOOP_WORDS)            \
  X(loop_reduce_v4, NX_FAMILY_REDUCE, NX_LOOP_THREADS, NX_LOOP_WORDS)         \
  X(loop_reduce_v1, NX_FAMILY_REDUCE, NX_LOOP_THREADS, NX_LOOP_WORDS)         \
  X(loop_reduce_direct, NX_FAMILY_REDUCE, NX_LOOP_THREADS, NX_LOOP_WORDS)     \
  X(loop_reduce_centered, NX_FAMILY_REDUCE, NX_LOOP_THREADS, NX_LOOP_WORDS)   \
  NX_METAL_ROW_KERNELS(NX_ROW_ROW, X)                                         \
  X(loop_reduce_wide, NX_FAMILY_WIDE, NX_LOOP_THREADS, NX_LOOP_WORDS)         \
  X(loop_merge, NX_FAMILY_REDUCE, NX_LOOP_THREADS, NX_LOOP_WORDS)             \
  X(loop_scan_v4, NX_FAMILY_SCAN, NX_LOOP_THREADS, NX_LOOP_WORDS)             \
  X(loop_scan_v1, NX_FAMILY_SCAN, NX_LOOP_THREADS, NX_LOOP_WORDS)             \
  X(loop_scan_wide, NX_FAMILY_WIDE, NX_LOOP_THREADS, NX_LOOP_WORDS)           \
  X(loop_carries, NX_FAMILY_SCAN, NX_LOOP_THREADS, NX_LOOP_WORDS)             \
  X(loop_network_v4, NX_FAMILY_SCAN, NX_LOOP_THREADS, NX_LOOP_WORDS)          \
  X(loop_network_v1, NX_FAMILY_SCAN, NX_LOOP_THREADS, NX_LOOP_WORDS)          \
  X(loop_network_wide, NX_FAMILY_WIDE, NX_LOOP_THREADS, NX_LOOP_WORDS)

/* A loop's parameters are NX_LOOP_WORDS words: NX_LOOP_BUFFERS buffer
   addresses, the address of a constant tail, the header's words from
   NX_LOOP_MODE, then NX_LOOP_HOLDS words that hold the loop's table where
   it fits them; past them the tail holds it. The table holds the loop's
   extents, then each extent's divisor, then its operands' descriptors and
   its tape, which the header points at by word. A group has
   NX_LOOP_THREADS threads, each of [v] elements of a tile, at most
   NX_LOOP_MOST. A reduction keeps nx.cpu's lane order: blocks of
   NX_LOOP_BLOCK terms, NX_LOOP_LANES lanes a block; a scan its chunks of
   NX_LOOP_CHUNK terms. A state in the workspace takes NX_LOOP_STATE
   32-bit words. A map's tile of NX_LOOP_TILE x NX_LOOP_TILE indices
   (NX_LOOP_TILED) stages each staged load in NX_LOOP_STAGE_WORDS words
   of the group's memory: NX_LOOP_TILE columns of NX_LOOP_TILE + 1, the
   last word apart, so that a column's words lie in distinct banks. */
enum {
  NX_LOOP_WORDS = 128,
  NX_LOOP_BUFFERS = 12,
  NX_LOOP_THREADS = 256,
  NX_LOOP_MOST = 4,
  NX_LOOP_BLOCK = 1024,
  NX_LOOP_LANES = 16,
  NX_LOOP_CHUNK = 4096,
  NX_LOOP_STATE = 8,
  NX_LOOP_TILE = 32,
  NX_LOOP_STAGE_WORDS = NX_LOOP_TILE * (NX_LOOP_TILE + 1),
};

/* The parameters' words. */
enum {
  NX_LOOP_BUFFER,    /* the first buffer's address */
  NX_LOOP_TAIL = NX_LOOP_BUFFER + NX_LOOP_BUFFERS, /* the tail's address */
  NX_LOOP_MODE,      /* v, then NX_LOOP_WIDE, NX_LOOP_APART, NX_LOOP_QUADS */
  NX_LOOP_FIRST,     /* the launch's first tile, unit or group */
  NX_LOOP_END,       /* past its last */
  NX_LOOP_RANK,      /* the loop's axes */
  NX_LOOP_NUMEL,     /* its indices */
  NX_LOOP_TABLE,     /* the table's words */
  NX_LOOP_LOADS,     /* the word of the loads' table: each load's word */
  NX_LOOP_DSTS,      /* the word of the destinations' table */
  NX_LOOP_TAPE,      /* the word of the tape's first instruction */
  NX_LOOP_PROLOGUE,  /* the words of the tape's constants, written once */
  NX_LOOP_BODY,      /* the words run on each tile, after the prologue */
  NX_LOOP_EPILOGUE,  /* the words run after a scan's results, after the body */
  NX_LOOP_TERM,      /* the plane of a reduction's or a scan's terms */
  NX_LOOP_RESULT,    /* the first of a scan's result planes */
  NX_LOOP_CENTER,    /* the plane of each term's center; a network's first
                        plane past its planes */
  NX_LOOP_REDUCTION, /* reduction, then compute type << 8, carrier << 16,
                        finish << 24 */
  NX_LOOP_TERMS,     /* a row's terms */
  NX_LOOP_SEGMENT,   /* an output's terms at most: a segment of a row */
  NX_LOOP_SEGMENTS,  /* a row's segments */
  NX_LOOP_OUTPUTS,   /* the outputs: rows times segments */
  NX_LOOP_GROUP,     /* a reduction's blocks a group, a power of two; a
                        scan's threads a chunk */
  NX_LOOP_GROUPS,    /* an output's groups */
  NX_LOOP_SPAN,      /* a reduction's lanes an output: NX_LOOP_LANES, or
                        for an output of at most half as many terms, the
                        least power of two at or past them */
  NX_LOOP_RUNS,      /* the runs a launch takes of a reduction's block, of
                        its threads times v terms, or of a scan's unit */
  NX_LOOP_FROM,      /* a reduction's first run a launch takes */
  NX_LOOP_WORK,      /* the workspace's buffer */
  NX_LOOP_TICKETS,   /* the tickets' buffer, a zero word an output */
  NX_LOOP_PARTIALS,  /* the byte of the groups' states, or of a network
                        chain's levels */
  NX_LOOP_TOTALS,    /* the byte of a scan's chunk or block totals */
  NX_LOOP_CENTERS,   /* the byte of the outputs' or slices' centers */
  NX_LOOP_CARRY,     /* a scan's carry: its load, or -1; a direct map's
                        or reduction's form */
  NX_LOOP_HELD,      /* the first word of the held table */
  NX_LOOP_HOLDS = NX_LOOP_WORDS - NX_LOOP_HELD,
};

/* NX_LOOP_MODE's bits past v: positions and indices of 64 bits; a
   reduction's groups leave their states apart, for loop_merge; a map's
   thread takes 4 consecutive indices, which each operand reads and writes
   as one vector where it can (tape.h, Quads); a network's groups
   leave their blocks' totals alone (loop.metal, Networks); a
   reduction's groups take their runs in pieces from NX_LOOP_FROM; a
   map's groups take tiles (NX_LOOP_TILE_A); a map's threads keep their
   quads' coordinates in group memory (NX_LOOP_COORD_AT); a map reads a
   padded load; a direct map's loads and destination are each 32 bits
   wide, its form takes each element and gives its result as it is, and
   its threads take quads; a cast's operands are contiguous over one axis
   of quads, each from an element on its quad's size (direct.metal, Casts
   by name); a direct map's loads and destination are each a bfloat16's,
   or each a float16's, its form decodes each element and encodes its
   result by that float's codec, and its threads take quads; a direct
   map's loads and destination are each 8 bits wide, its form takes each
   element as it is or as a truth, computes no long kind
   (NX_DIRECT_LONG_INT1) and gives its result as it is or wrapped into 8
   bits, and its threads take quads; a Logsumexp's or Moments' direct
   second pass folds its centers in the same launch first (loop.metal,
   once_center). A kernel's v, width, tiles, padded loads and 32-bit
   lanes are its own (its name's _v4, _v1, _wide, _tiled, _padded or 32),
   so that a path it does not take costs it nothing; loop_merge and
   loop_carries take the width from the header. Each bit is named by its
   index, counted on from v's 8 bits, so that no two bits can be one. */
enum {
  NX_LOOP_WIDE_AT = 8,
  NX_LOOP_APART_AT,
  NX_LOOP_QUADS_AT,
  NX_LOOP_UP_AT,
  NX_LOOP_PIECES_AT,
  NX_LOOP_TILED_AT,
  NX_LOOP_COORDS_AT,
  NX_LOOP_PADDED_AT,
  NX_LOOP_BITS32_AT,
  NX_LOOP_FLAT_AT,
  NX_LOOP_ONCE_AT,
  NX_LOOP_BITS16_AT,
  NX_LOOP_BITS8_AT,
  NX_LOOP_MODE_END
};
enum {
  NX_LOOP_WIDE = 1 << NX_LOOP_WIDE_AT,
  NX_LOOP_APART = 1 << NX_LOOP_APART_AT,
  NX_LOOP_QUADS = 1 << NX_LOOP_QUADS_AT,
  NX_LOOP_UP = 1 << NX_LOOP_UP_AT,
  NX_LOOP_PIECES = 1 << NX_LOOP_PIECES_AT,
  NX_LOOP_TILED = 1 << NX_LOOP_TILED_AT,
  NX_LOOP_COORDS = 1 << NX_LOOP_COORDS_AT,
  NX_LOOP_PADDED = 1 << NX_LOOP_PADDED_AT,
  NX_LOOP_BITS32 = 1 << NX_LOOP_BITS32_AT,
  NX_LOOP_FLAT = 1 << NX_LOOP_FLAT_AT,
  NX_LOOP_ONCE = 1 << NX_LOOP_ONCE_AT,
  NX_LOOP_BITS16 = 1 << NX_LOOP_BITS16_AT,
  NX_LOOP_BITS8 = 1 << NX_LOOP_BITS8_AT
};
static_assert(NX_LOOP_MODE_END <= 32, "NX_LOOP_MODE's bits fit a word");

/* A map of quads with padded loads: the word of group memory its
   threads' coordinates start at, where NX_LOOP_COORDS is set (tape.h,
   Padded quads), in a header word no map reads otherwise. */
enum { NX_LOOP_COORD_AT = NX_LOOP_RUNS };

/* A tiled map's words, in header words no map reads otherwise: the
   extents of a and b, the walk's last two axes of extent past 1; the
   log2 of the tiles along b a group run takes in turn before the next
   along a, so that tiles that share a staged load's lines run together;
   the word of group memory its stages start at; its staged loads, their
   count in the low 4 bits, then each one's load in 8 bits; and the
   divisors (divide) of the tiles along a and of a plane's groups, its
   tiles along b rounded up to whole runs. A group takes the tile of
   NX_LOOP_TILE indices along b by NX_LOOP_TILE consecutive along a at
   its number (tape.h, Tiles); a group past a plane's last tile along b
   has none. */
enum {
  NX_LOOP_TILE_A = NX_LOOP_TERMS,
  NX_LOOP_TILE_B = NX_LOOP_SEGMENT,
  NX_LOOP_TILE_RUN = NX_LOOP_SEGMENTS,
  NX_LOOP_STAGE = NX_LOOP_OUTPUTS,
  NX_LOOP_STAGED = NX_LOOP_GROUP,
  NX_LOOP_TILES_A = NX_LOOP_GROUPS,
  NX_LOOP_TILES = NX_LOOP_SPAN,
};

/* A direct map's form, which loop_map_direct runs in place of the tape
   (direct.metal), in the header word of a scan's carry, which no map
   reads: its op in its low 16 bits, a kind's (kernels.h's ops),
   NX_LOOP_WHERE1, or NX_LOOP_CAST for its first operand as it is; its
   operands' count at NX_DIRECT_COUNT; operand j's load at
   NX_DIRECT_OPERANDS + 4 j and, 2 bits above, how its element comes to
   the op: as it is, as a truth, or decoded from bfloat16 or float16; and
   at NX_DIRECT_RESULT how the op's result comes to the store: as it is,
   wrapped to the integer dtype at NX_DIRECT_WRAP_DT, or encoded into
   bfloat16 or float16. A direct reduction's form, which
   loop_reduce_direct, loop_reduce_centered or a row's kernel runs
   (loop.metal), in the same word, which no
   reduction reads either: its one operand, as a map's operand 0, whose
   element, so taken, is the term. */
enum { NX_LOOP_DIRECT = NX_LOOP_CARRY };
enum {
  NX_DIRECT_COUNT = 16,
  NX_DIRECT_OPERANDS = 20,
  NX_DIRECT_RESULT = 32,
  NX_DIRECT_WRAP_DT = 40,
};
enum { NX_DIRECT_AS_IS, NX_DIRECT_TRUTH, NX_DIRECT_BF16, NX_DIRECT_F16 };
enum { NX_DIRECT_WRAP = NX_DIRECT_TRUTH };

/* Where a direct map's slot k (NX_DIRECT_SLOTS) finds its elements, 2
   bits at NX_DIRECT_PLACES + 2 k of its form: at the positions its
   descriptor places; flat, the element of each index at that index past
   its buffer's word, the walk being one axis along which the slot's
   stride is 1; flat with every quad's elements on 4 times their size; or
   at that word for every index, its stride 0 along the walk's one axis.
   A 32-bit direct map none of whose slots is placed runs
   loop_map_direct32_flat, or loop_map_direct32_long_flat for its long
   kinds: kernels that read no descriptor and check no alignment, and
   hold no path that does. A slot's places are 0 in every other map. */
enum { NX_DIRECT_PLACES = 48 };
enum { NX_DIRECT_PLACED, NX_DIRECT_FLAT, NX_DIRECT_FLAT_ON, NX_DIRECT_FIRST };

/* A direct map of 32-bit lanes runs its kind in one of two kernels, by
   the kind's cost: loop_map_direct32 holds the kinds of one instruction
   or a few, Where and Cast; loop_map_direct32_long the long kinds, so
   that neither kind's registers and code bound the other's. The long
   kinds, a bit an op1's or an op2's code (nx_kernel.h): at float32
   Recip, Sqrt, Exp, Exp2, Log, Log2, Log1p, Expm1, Sin, Cos, Tan, Asin,
   Acos, Atan, Sinh, Cosh, Tanh and Erf, and Fdiv, Mod, Pow and Atan2; at
   int32 and uint32 Recip, and Idiv, Mod and Pow. direct.metal checks
   each set against its kernel's kinds. */
enum {
  NX_DIRECT_LONG_REAL1 = 0x47fff90,
  NX_DIRECT_LONG_REAL2 = 0xe8,
  NX_DIRECT_LONG_INT1 = 0x10,
  NX_DIRECT_LONG_INT2 = 0x70,
};

/* A direct map's table, held (direct.metal): the walk's extents, then
   their divisors, over at most NX_DIRECT_RANK axes, in the first
   2 NX_DIRECT_RANK words; from NX_DIRECT_SLOTS, a descriptor a slot of
   NX_DIRECT_STRIDE words, its destination's in slot 0 and its form's
   operand j's in slot 1 + j; then the loads' and the destinations'
   tables, which name the slots. Slot k's buffer is the parameters'
   buffer k. Every word a thread reads to find its first element so lies
   at a fixed place, and no read waits on another for its address. */
enum {
  NX_DIRECT_RANK = 8,
  NX_DIRECT_SLOTS = 2 * NX_DIRECT_RANK,
  NX_DIRECT_STRIDE = 2 + NX_DIRECT_RANK,
};

/* A descriptor's words: its dtype, then whether it is padded << 8, its
   buffer << 16, the bit its position 0 lies past << 24, for a dtype of
   fewer than 8 bits, and a staged load's stage, from 1, << 32; the
   position of its element at index 0 from its buffer's address, in
   elements; then a plain operand's stride along each axis of the loop,
   or a padded one's fill (two words), its rank r and,
   per axis of the operand, NX_LOOP_PAD words: its lo, its interior, its
   extent, its stride, the loop axes it reads (the second << 16, 0xffff
   for none), their coefficients, and the divisor (divide) of its
   interior + 1. */
enum { NX_LOOP_PAD = 8 };

/* An instruction: its op, then its planes d, a, b and c of 12 bits from
   bits 16, 28, 40 and 52. A kind's op is its compute type << 7 and its
   code: an op1's, NX_LOOP_BINARY plus an op2's, NX_LOOP_TERNARY plus an
   op3's (nx_kernel.h's codes, in Prog's order: an op1 Unary's from
   NX_LOOP_UNARY, an op2 Compare's from NX_LOOP_COMPARE). A constant's two
   words follow it. Fma at a narrow float is a kind at NX_LOOP_F32 past the
   op3s' codes: nx_kinds.h's nx_fma_narrow, or nx_fma_bf16 for bfloat16,
   the exact value rounded to odd, so that the encode after it rounds
   once. */
enum {
  NX_LOOP_UNARY = 3,
  NX_LOOP_COMPARE = 14,
  NX_LOOP_BINARY = 32,
  NX_LOOP_TERNARY = 64,
  NX_LOOP_FMA_NARROW = NX_LOOP_TERNARY + 2,
  NX_LOOP_FMA_BF16 = NX_LOOP_TERNARY + 3,
};
enum { NX_LOOP_F32, NX_LOOP_C64, NX_LOOP_I32, NX_LOOP_U32, NX_LOOP_I64,
       NX_LOOP_U64 };
enum {
  NX_LOOP_LOAD = 1024, /* d: load a's element */
  NX_LOOP_COORD,       /* d: the index along loop axis a */
  NX_LOOP_CONST,       /* d: the two words after, of b words */
  NX_LOOP_CAST,        /* d: a of dtype b cast to dtype c */
  NX_LOOP_BITCAST,     /* d: a of dtype b read in dtype c */
  NX_LOOP_WHERE1,      /* d: b where a, else c: of one word */
  NX_LOOP_WHERE2,      /* of two */
  NX_LOOP_WHERE4,      /* of four */
  NX_LOOP_DECODE,      /* d: the float32 of a, a code of dtype b */
  NX_LOOP_ENCODE,      /* d: the code of dtype b of a, a float32 */
  NX_LOOP_WRAP,        /* d: a wrapped to the integer dtype b */
  NX_LOOP_TRUTH,       /* d: whether a, a boolean's byte, is not 0 */
  NX_LOOP_STORE        /* destination b: a */
};

/* Reductions: a monoid's fold, Arg's, Logsumexp's second pass, a sum of
   exp (x - c), and Moments', sums of (x - c)^2 and x - c. A network
   scans by Logsumexp's or Moments' states. */
enum {
  NX_LOOP_SUM, NX_LOOP_PROD, NX_LOOP_MAX, NX_LOOP_MIN, NX_LOOP_ARG_MAX,
  NX_LOOP_ARG_MIN, NX_LOOP_EXPS, NX_LOOP_SQUARES, NX_LOOP_LOGSUMEXP,
  NX_LOOP_MOMENTS
};

/* What a reduction does with an output's state: stores it, as a total of
   a scan's chunk, as Logsumexp's center, as Moments' mean, as Logsumexp,
   as Moments. */
enum {
  NX_LOOP_PUT, NX_LOOP_PUT_TOTAL, NX_LOOP_PUT_CENTER, NX_LOOP_PUT_MEAN,
  NX_LOOP_PUT_LOGSUMEXP, NX_LOOP_PUT_MOMENTS
};

/* Contractions (contract.metal, whose header states how they run) */

/* A staged dtype's tiles with b k x n, each of an aligned (_a), a wide
   edge (_e) and a narrow edge (_n) function, and with b n x k, each of
   its edge function. */
#define NX_CONTRACT_TILE_ROWS_KN(X, stage, family)                            \
  X(contract_##stage##_64_kn_a, family, NX_CONTRACT_THREADS,                  \
    NX_CONTRACT_WORDS)                                                        \
  X(contract_##stage##_64_kn_e, family, NX_CONTRACT_THREADS,                  \
    NX_CONTRACT_WORDS)                                                        \
  X(contract_##stage##_64_kn_n, family, NX_CONTRACT_THREADS,                  \
    NX_CONTRACT_WORDS)                                                        \
  X(contract_##stage##_32_kn_a, family, NX_CONTRACT_THREADS,                  \
    NX_CONTRACT_WORDS)                                                        \
  X(contract_##stage##_32_kn_e, family, NX_CONTRACT_THREADS,                  \
    NX_CONTRACT_WORDS)                                                        \
  X(contract_##stage##_32_kn_n, family, NX_CONTRACT_THREADS,                  \
    NX_CONTRACT_WORDS)                                                        \
  X(contract_##stage##_16_kn_a, family, NX_CONTRACT_THREADS,                  \
    NX_CONTRACT_WORDS)                                                        \
  X(contract_##stage##_16_kn_e, family, NX_CONTRACT_THREADS,                  \
    NX_CONTRACT_WORDS)                                                        \
  X(contract_##stage##_16_kn_n, family, NX_CONTRACT_THREADS,                  \
    NX_CONTRACT_WORDS)
#define NX_CONTRACT_TILE_ROWS_NK(X, stage, family)                            \
  X(contract_##stage##_64_nk, family, NX_CONTRACT_THREADS, NX_CONTRACT_WORDS) \
  X(contract_##stage##_32_nk, family, NX_CONTRACT_THREADS, NX_CONTRACT_WORDS) \
  X(contract_##stage##_16_nk, family, NX_CONTRACT_THREADS, NX_CONTRACT_WORDS)

/* An integer or converting tile's rows, of its edge function alone,
   for each layout of b; a layout kn_n names an integer tile's narrow
   edge functions. */
#define NX_CONTRACT_TILE_ROWS_EDGE(X, stage, layout, family)                  \
  X(contract_##stage##_64_##layout, family, NX_CONTRACT_THREADS,              \
    NX_CONTRACT_WORDS)                                                        \
  X(contract_##stage##_32_##layout, family, NX_CONTRACT_THREADS,              \
    NX_CONTRACT_WORDS)                                                        \
  X(contract_##stage##_16_##layout, family, NX_CONTRACT_THREADS,              \
    NX_CONTRACT_WORDS)

/* A staged dtype's grouped tiles, 32 x 64 and 16 x 64, for a layout of
   b, each of its edge function; a layout kn_n the narrow edge
   functions of b k x n. */
#define NX_CONTRACT_GROUPED_OF(X, stage, layout, family)                      \
  X(contract_rows_##stage##_32_##layout, family, NX_CONTRACT_THREADS,         \
    NX_CONTRACT_WORDS)                                                        \
  X(contract_rows_##stage##_16_##layout, family, NX_CONTRACT_THREADS,         \
    NX_CONTRACT_WORDS)
#define NX_CONTRACT_GROUPED(X, stage, layout)                                 \
  NX_CONTRACT_GROUPED_OF(X, stage, layout, NX_FAMILY_CONTRACT_ROWS)

/* A loader key's tiles for a layout of b: 32 x 32 and 16 x 64, and the
   grouped 32 x 64 and 16 x 64, each of its edge function. */
#define NX_LOADER_ROWS(X, b, a, layout, family)                               \
  X(contract_ld_##b##_##a##_32_##layout, family, NX_CONTRACT_THREADS,         \
    NX_CONTRACT_WORDS)                                                        \
  X(contract_ld_##b##_##a##_16_##layout, family, NX_CONTRACT_THREADS,         \
    NX_CONTRACT_WORDS)                                                        \
  X(contract_rows_ld_##b##_##a##_32_##layout, family, NX_CONTRACT_THREADS,    \
    NX_CONTRACT_WORDS)                                                        \
  X(contract_rows_ld_##b##_##a##_16_##layout, family, NX_CONTRACT_THREADS,    \
    NX_CONTRACT_WORDS)
#define NX_LOADER_KERNELS(X, b, B, a, A, T)                                   \
  NX_LOADER_ROWS(X, b, a, kn, NX_FAMILY_LOADER_##B##_##A##_KN)                \
  NX_LOADER_ROWS(X, b, a, nk, NX_FAMILY_LOADER_##B##_##A##_NK)
#define NX_LOADER_BAND_KERNELS(X, b, B)                                       \
  NX_LOADER_ROWS(X, b, f32, kn_band, NX_FAMILY_LOADER_##B##_BAND)             \
  NX_LOADER_ROWS(X, b, f32, nk_band, NX_FAMILY_LOADER_##B##_BAND)

/* A staged dtype's banded tiles: each tile's edge functions with b
   k x n, wide and narrow, and its function with b n x k, then the
   grouped tiles' functions. */
#define NX_CONTRACT_BAND_ROWS(X, stage, family)                               \
  NX_CONTRACT_TILE_ROWS_EDGE(X, stage, kn_e_band, family)                     \
  NX_CONTRACT_TILE_ROWS_EDGE(X, stage, kn_n_band, family)                     \
  NX_CONTRACT_TILE_ROWS_EDGE(X, stage, nk_band, family)                       \
  NX_CONTRACT_GROUPED_OF(X, stage, kn_band, family)                           \
  NX_CONTRACT_GROUPED_OF(X, stage, kn_n_band, family)                         \
  NX_CONTRACT_GROUPED_OF(X, stage, nk_band, family)

#define NX_METAL_CONTRACT_KERNELS(X)                                          \
  NX_CONTRACT_TILE_ROWS_KN(X, f32, NX_FAMILY_CONTRACT_F32_KN)                 \
  NX_CONTRACT_TILE_ROWS_NK(X, f32, NX_FAMILY_CONTRACT_F32_NK)                 \
  NX_CONTRACT_TILE_ROWS_KN(X, f16, NX_FAMILY_CONTRACT_F16_KN)                 \
  NX_CONTRACT_TILE_ROWS_NK(X, f16, NX_FAMILY_CONTRACT_F16_NK)                 \
  NX_CONTRACT_TILE_ROWS_KN(X, bf16, NX_FAMILY_CONTRACT_BF16_KN)               \
  NX_CONTRACT_TILE_ROWS_NK(X, bf16, NX_FAMILY_CONTRACT_BF16_NK)               \
  NX_CONTRACT_TILE_ROWS_KN(X, c64, NX_FAMILY_CONTRACT_C64_KN)                 \
  NX_CONTRACT_TILE_ROWS_NK(X, c64, NX_FAMILY_CONTRACT_C64_NK)                 \
  NX_CONTRACT_TILE_ROWS_EDGE(X, i8, kn, NX_FAMILY_CONTRACT_I8_KN)             \
  NX_CONTRACT_TILE_ROWS_EDGE(X, i8, kn_n, NX_FAMILY_CONTRACT_I8_KN)           \
  NX_CONTRACT_TILE_ROWS_EDGE(X, i8, nk, NX_FAMILY_CONTRACT_I8_NK)             \
  X(contract_finish, NX_FAMILY_FINISH, NX_FINISH_THREADS, NX_FINISH_WORDS)    \
  X(contract_scale, NX_FAMILY_CONTRACT_SCALE, NX_SCALE_THREADS,              \
    NX_SCALE_WORDS)                                                           \
  X(contract_band, NX_FAMILY_CONTRACT_SCALE, NX_BAND_THREADS, NX_BAND_WORDS)  \
  X(contract_simt, NX_FAMILY_CONTRACT_SIMT, NX_SIMT_THREADS, NX_SIMT_WORDS)   \
  X(contract_finish_int, NX_FAMILY_CONTRACT_SIMT, NX_FINISH_THREADS,          \
    NX_FINISH_WORDS)                                                          \
  X(contract_compact, NX_FAMILY_CONTRACT_SIMT, NX_COMPACT_THREADS,            \
    NX_COMPACT_WORDS)                                                         \
  X(contract_buckets, NX_FAMILY_CONTRACT_ROWS, NX_BUCKETS_THREADS,            \
    NX_BUCKETS_WORDS)                                                         \
  NX_CONTRACT_GROUPED(X, f32, kn)                                             \
  NX_CONTRACT_GROUPED(X, f32, kn_n)                                           \
  NX_CONTRACT_GROUPED(X, f32, nk)                                             \
  NX_CONTRACT_GROUPED(X, f16, kn)                                             \
  NX_CONTRACT_GROUPED(X, f16, kn_n)                                           \
  NX_CONTRACT_GROUPED(X, f16, nk)                                             \
  NX_CONTRACT_GROUPED(X, bf16, kn)                                            \
  NX_CONTRACT_GROUPED(X, bf16, kn_n)                                          \
  NX_CONTRACT_GROUPED(X, bf16, nk)                                            \
  NX_CONTRACT_GROUPED(X, c64, kn)                                             \
  NX_CONTRACT_GROUPED(X, c64, kn_n)                                           \
  NX_CONTRACT_GROUPED(X, c64, nk)                                             \
  NX_CONTRACT_LOADER_KEYS(NX_LOADER_KERNELS, X)                               \
  NX_METAL_CONTRACT_MV_KERNELS(X)                                             \
  NX_CONTRACT_BAND_ROWS(X, f32, NX_FAMILY_CONTRACT_F32_BAND)                  \
  NX_CONTRACT_BAND_ROWS(X, f16, NX_FAMILY_CONTRACT_F16_BAND)                  \
  NX_CONTRACT_BAND_ROWS(X, bf16, NX_FAMILY_CONTRACT_BF16_BAND)                \
  NX_CONTRACT_LOADER_BANDS(NX_LOADER_BAND_KERNELS, X)

/* The vector kernels of a key [key] at [r] rows a pass, for each dtype
   of the other operand: contract_mv_<key>_<other>_r<r>_nk, its sums
   float32, and of a float16 or bfloat16 other operand
   contract_mv_<key>_<other>_r<r>_nk_typed, its sums of that dtype. */
#define NX_CONTRACT_MV(X, key, family, r)                                     \
  X(contract_mv_##key##_f32_r##r##_nk, family, NX_MV_THREADS, NX_MV_WORDS)   \
  X(contract_mv_##key##_f16_r##r##_nk, family, NX_MV_THREADS, NX_MV_WORDS)   \
  X(contract_mv_##key##_bf16_r##r##_nk, family, NX_MV_THREADS, NX_MV_WORDS)  \
  X(contract_mv_##key##_f16_r##r##_nk_typed, family, NX_MV_THREADS,          \
    NX_MV_WORDS)                                                              \
  X(contract_mv_##key##_bf16_r##r##_nk_typed, family, NX_MV_THREADS,         \
    NX_MV_WORDS)

/* M(X, key, family, r) for each key of the wider operand and its rows a
   pass: r 1 for every key, which one row and a Rows call take; 2 for
   float16 and bfloat16. */
#define NX_CONTRACT_MV_KEYS(M, X)                                             \
  M(X, f32, NX_FAMILY_CONTRACT_MV_F32, 1)                                     \
  M(X, f16, NX_FAMILY_CONTRACT_MV_F16, 1)                                     \
  M(X, f16, NX_FAMILY_CONTRACT_MV_F16, 2)                                     \
  M(X, bf16, NX_FAMILY_CONTRACT_MV_BF16, 1)                                   \
  M(X, bf16, NX_FAMILY_CONTRACT_MV_BF16, 2)                                   \
  M(X, e2m1, NX_FAMILY_CONTRACT_MV_E2M1, 1)                                   \
  M(X, i8f16, NX_FAMILY_CONTRACT_MV_I8F16, 1)                                 \
  M(X, i8f32, NX_FAMILY_CONTRACT_MV_I8F32, 1)

/* The k x n vector kernels of a key [key], its weight's columns
   contiguous, at [r] rows a pass, for each dtype of the other operand:
   contract_mv_<key>_<other>_r<r>_kn, and _kn_typed, as the n x k ones. */
#define NX_CONTRACT_MV_KN(X, key, family, r)                                  \
  X(contract_mv_##key##_f32_r##r##_kn, family, NX_MV_KN_THREADS, NX_MV_WORDS) \
  X(contract_mv_##key##_f16_r##r##_kn, family, NX_MV_KN_THREADS, NX_MV_WORDS) \
  X(contract_mv_##key##_bf16_r##r##_kn, family, NX_MV_KN_THREADS, NX_MV_WORDS) \
  X(contract_mv_##key##_f16_r##r##_kn_typed, family, NX_MV_KN_THREADS,       \
    NX_MV_WORDS)                                                              \
  X(contract_mv_##key##_bf16_r##r##_kn_typed, family, NX_MV_KN_THREADS,      \
    NX_MV_WORDS)

/* M(X, key, family, r) for each key of a k x n weight and its rows a
   pass: r 1 for every key; 4 for the scaled keys past one row, whose
   lanes hold 8 columns. */
#define NX_CONTRACT_MV_KN_KEYS(M, X)                                          \
  M(X, f32, NX_FAMILY_CONTRACT_MV_F32, 1)                                     \
  M(X, f16, NX_FAMILY_CONTRACT_MV_F16, 1)                                     \
  M(X, bf16, NX_FAMILY_CONTRACT_MV_BF16, 1)                                   \
  M(X, e2m1, NX_FAMILY_CONTRACT_MV_E2M1, 1)                                   \
  M(X, e2m1, NX_FAMILY_CONTRACT_MV_E2M1, 4)                                   \
  M(X, i8f16, NX_FAMILY_CONTRACT_MV_I8F16, 1)                                 \
  M(X, i8f16, NX_FAMILY_CONTRACT_MV_I8F16, 4)                                 \
  M(X, i8f32, NX_FAMILY_CONTRACT_MV_I8F32, 1)                                 \
  M(X, i8f32, NX_FAMILY_CONTRACT_MV_I8F32, 4)

/* The twins of a key's n x k kernels at one row a pass,
   contract_mv_<key>_<other>_r1_nk_w32 and _nk_typed_w32: the same sums,
   by a kernel that runs only whole spans of one form (NX_MV_FORM) at 32
   lanes, so that its registers are that loop's alone. The host launches
   a twin where its call's form is the twin's, k is a multiple of 32 and
   its lanes are 32, else the kernel. */
#define NX_CONTRACT_MV_TWIN(X, key, family, form)                             \
  X(contract_mv_##key##_f32_r1_nk_w32, family, NX_MV_THREADS, NX_MV_WORDS)   \
  X(contract_mv_##key##_f16_r1_nk_w32, family, NX_MV_THREADS, NX_MV_WORDS)   \
  X(contract_mv_##key##_bf16_r1_nk_w32, family, NX_MV_THREADS, NX_MV_WORDS)  \
  X(contract_mv_##key##_f16_r1_nk_typed_w32, family, NX_MV_THREADS,          \
    NX_MV_WORDS)                                                              \
  X(contract_mv_##key##_bf16_r1_nk_typed_w32, family, NX_MV_THREADS,         \
    NX_MV_WORDS)

/* M(X, key, family, form) for each key with twins and the form they
   run: e2m1's (MXFP4) 16-byte words, int8 by float16's (Q8_0's 34-byte
   blocks) 2-byte words, int8 by float32's 16-byte words with two scale
   blocks a span (Q6_K's of 16 terms). */
#define NX_CONTRACT_MV_TWINS(M, X)                                            \
  M(X, e2m1, NX_FAMILY_CONTRACT_MV_E2M1_W32, 2)                               \
  M(X, i8f16, NX_FAMILY_CONTRACT_MV_I8F16_W32, 1)                             \
  M(X, i8f32, NX_FAMILY_CONTRACT_MV_I8F32_W32, 3)

#define NX_METAL_CONTRACT_MV_KERNELS(X)                                       \
  NX_CONTRACT_MV_KEYS(NX_CONTRACT_MV, X)                                      \
  NX_CONTRACT_MV_TWINS(NX_CONTRACT_MV_TWIN, X)                                \
  NX_CONTRACT_MV_KN_KEYS(NX_CONTRACT_MV_KN, X)

/* A tile's group, of four SIMD-groups, and the terms of a span: a range
   of k holds whole spans, and a chunk of it too. */
enum { NX_CONTRACT_THREADS = 128, NX_CONTRACT_SPAN = 32 };

/* The fields of a tile's [codes] (below), by their first bit: a's and
   b's dtype codes, 8 bits each; how a's and b's bytes read, 2 bits
   each; the flips, 3 bits; b's phase, 3 bits; narrow; the swizzle, 4
   bits; a routine's start from y, its triangle, 2 bits, and its pairs;
   whether b's runs are read as words; a's band and b's. Bits 23 and
   39-63 are free. */
enum {
  NX_CODES_DTYPE_A = 0,
  NX_CODES_DTYPE_B = 8,
  NX_CODES_KIND_A = 16,
  NX_CODES_KIND_B = 18,
  NX_CODES_FLIPS = 20,
  NX_CODES_PHASE = 24,
  NX_CODES_NARROW = 27,
  NX_CODES_SWIZZLE = 28,
  NX_CODES_FIRST = 32,
  NX_CODES_TRI = 33,
  NX_CODES_PAIR = 35,
  NX_CODES_WORDS = 36,
  NX_CODES_BAND_A = 37,
  NX_CODES_BAND_B = 38,
};

/* A tile kernel's launch: groups (tiles, 1, ranges), x the tiles from
   [first] in the call's order, z the range. [a] and [b] address their
   first element of batch element 0, [y] the outputs' of batch element 0
   and range 0. Strides are in elements: [lda] a's rows'; [ldb] b's
   terms' (k x n) or columns' (n x k); [sa], [sb] and [sy] the batch
   element's; [ldy] and [cy] y's rows' and columns'; [plane] the range's.
   [m], [n] and [k] are the tiles' rows, columns and terms; [tiles]
   holds the tile rows tiles_m in its low 32 bits and the tile columns
   tiles_n in its high, which with the swizzle, [codes]'
   NX_CODES_SWIZZLE, and [padded] give the call's order (contract.mli);
   [ranges] and [spans] the ranges of k and its spans; [length] a
   chunk's terms, [0] for whole ranges, and [chunk] the launch's chunk
   of each; [codes]' NX_CODES_NARROW, 1 where y is a dst of the staged
   dtype, into which the sums round, 0 where it holds float32. An
   integer tile's y holds 64-bit sums, or where narrow their low 32
   bits, and [codes]' NX_CODES_KIND_A and NX_CODES_KIND_B say how a's
   and b's bytes read: an NX_BYTES_* constant. A converting tile reads
   each element at its dtype and casts it into the staged dtype:
   [codes]' NX_CODES_DTYPE_A and NX_CODES_DTYPE_B hold a's and b's dtype
   codes, each of 8 bits or more, and the strides count elements of each
   operand's own width. A routine's product sets [codes]'
   NX_CODES_FIRST, range 0 starts from y; NX_CODES_TRI's two bits, only
   tiles holding an element of the lower or upper triangle compute; and
   NX_CODES_PAIR, a column is a part of a complex element, its element's
   column the column halved. Other calls hold them 0. Bit
   NX_CODES_BAND_A of [codes] bands a, bit NX_CODES_BAND_B b: the
   operand is a band of chunks (contract_band), and a launch reads range
   s's chunk from its term s [length] in place of its chunk's first
   term; [a] or [b] then addresses the band.

   A complex tile computes a complex64 call's real form: [m], [n] and
   [k] are its rows, columns and terms, n and k twice the call's, and
   [a] and [y] are read and written as float32 parts. Its [b] is the
   complex64 b, [ldb] and [sb] in float32 parts (twice its elements),
   whose element br + i bi the tile stages as the real block [br, bi]
   above [-bi, br]. [codes]' NX_CODES_FLIPS, three bits, flip the
   block's signs for a routine's product: the first conjugates b, the
   second a (the lower row negated), the third negates it whole. Other
   calls hold them 0.

   A loader is a converting tile made for its operands' dtypes, which it
   holds as constants in place of [codes]' dtype codes. A loader of a b
   narrower than a byte reads it in place: [b] addresses the byte of b's
   index 0, at the bit that [codes]' NX_CODES_PHASE gives; where
   [codes]' NX_CODES_WORDS is set, every run of b's elements a loader
   reads starts on 4 bytes, and it reads the run as 32-bit words. A
   loader of a scaled b reads b's scale at [bs]: b's element of batch element z,
   group j, column c and term t is scaled by the scale's element at
   z sz + j sj + (c / rn) sn + (t / rk) sk elements from [bs], [bsz]
   holding sz in its low 32 bits and sj in its high, [bsn] sn and rn,
   [bsk] sk and rk, each stride signed and each block's elements below
   2^32. A loader of b n x k reads its term t at
   (t / bl) bs + t mod bl elements past its column's first, [bkb]
   holding bl in its low 32 bits and bs, signed, in its high, where bl
   is not 0: blocks of bl contiguous terms bs elements apart, as Q8_0's
   quants lie between their scales; other calls hold [bkb] 0. A loader
   divides by rn, rk and bl through their divisor words in [drn], [drk]
   and [dbl]: d's holds a multiplier m in its low 32 bits and shifts s1
   and s2 in bits 32-39 and 40-47, and u / d is
   (t + ((u - t) >> s1)) >> s2, t the high 32 bits of u m, for every
   32-bit u (contract.mli's divisor).

   A grouped tile's call order is of one batch element, of [padded] =
   tiles_m tiles_n tiles and no swizzle; its tile row i is entry i of
   [table], whose rows are entries of [list] (contract_buckets). [groups]
   holds the groups g in its low 32 bits and the bucket of no group in
   its high. The tile's rows y, of batch element z = y / m and row
   y mod m, multiply b's group of bucket e = z' g + j, at z' [sb] + j
   [sg]; the bucket of no group has no products. */
enum {
  NX_CONTRACT_A,
  NX_CONTRACT_B,
  NX_CONTRACT_Y,
  NX_CONTRACT_LDA,
  NX_CONTRACT_LDB,
  NX_CONTRACT_SA,
  NX_CONTRACT_SB,
  NX_CONTRACT_LDY,
  NX_CONTRACT_CY,
  NX_CONTRACT_SY,
  NX_CONTRACT_PLANE,
  NX_CONTRACT_M,
  NX_CONTRACT_N,
  NX_CONTRACT_K,
  NX_CONTRACT_TILES,
  NX_CONTRACT_BKB,
  NX_CONTRACT_PADDED,
  NX_CONTRACT_FIRST,
  NX_CONTRACT_RANGES,
  NX_CONTRACT_SPANS,
  NX_CONTRACT_LENGTH,
  NX_CONTRACT_CHUNK,
  NX_CONTRACT_CODES,
  NX_CONTRACT_LIST,
  NX_CONTRACT_TABLE,
  NX_CONTRACT_GROUPS,
  NX_CONTRACT_SG,
  NX_CONTRACT_BS,
  NX_CONTRACT_BSZ,
  NX_CONTRACT_BSN,
  NX_CONTRACT_BSK,
  NX_CONTRACT_DRN,
  NX_CONTRACT_DRK,
  NX_CONTRACT_DBL,
  NX_CONTRACT_WORDS,
};

/* contract_finish: groups (ceil (cols / NX_FINISH_THREADS), outputs' rows
   from [first], 1), a thread an output. [planes] addresses range 0's
   plane, [ranges] of them [plane] elements apart, each [rows] x [cols]
   for each batch element; [init] and [dst] their element at batch, row
   and column 0, of the dtype codes [init_dtype], -1 for no init, and
   [dst_dtype], with their strides in elements, and the accumulator's
   code [acc]. A complex64 call's init and dst hold column c's sum as
   part c mod 2 of their element c / 2.

   contract_finish_int, and contract_finish for a dst of another dtype
   than float32, float16 and bfloat16 of a float32 call, address init
   and dst by the words past [dst_c]: the position of their element at
   batch, row and column 0 from the address, at which no element they
   reach lies before it, and the bit of the address's byte at which
   position 0 starts, for a dtype narrower than a byte. contract_finish
   takes such an init and dst by their casts. contract_finish_int reads
   64-bit planes.

   Where [none] is not -1, contract_finish takes an output of row y
   whose entry of [buckets] is [none] as one of no products. Where
   [kgroups] is not 0, of a Contracting call, row y's batch element z =
   y / rows is group e = z / [batch] of batch element z mod [batch], and
   an output sums that group's ranges alone, as contract_simt counts
   them from [koff] and [cap], or is one of no products where the group
   has no terms. Where [tri] is not 0, a routine's, contract_finish
   leaves the outputs outside the lower (bit 0) or upper (bit 1)
   triangle, a column a part of a complex element where bit 2 is set. */
enum {
  NX_FINISH_PLANES,
  NX_FINISH_INIT,
  NX_FINISH_DST,
  NX_FINISH_ROWS,
  NX_FINISH_COLS,
  NX_FINISH_PLANE,
  NX_FINISH_RANGES,
  NX_FINISH_FIRST,
  NX_FINISH_INIT_DTYPE,
  NX_FINISH_INIT_Z,
  NX_FINISH_INIT_R,
  NX_FINISH_INIT_C,
  NX_FINISH_DST_DTYPE,
  NX_FINISH_DST_Z,
  NX_FINISH_DST_R,
  NX_FINISH_DST_C,
  NX_FINISH_ACC,
  NX_FINISH_INIT_AT,
  NX_FINISH_INIT_PHASE,
  NX_FINISH_DST_AT,
  NX_FINISH_DST_PHASE,
  NX_FINISH_BUCKETS,
  NX_FINISH_NONE,
  NX_FINISH_KOFF,
  NX_FINISH_KGROUPS,
  NX_FINISH_BATCH,
  NX_FINISH_CAP,
  NX_FINISH_TRI,
  NX_FINISH_WORDS,
};

enum { NX_FINISH_THREADS = 256 };

enum { NX_CCOPY_THREADS = 256 };

/* contract_copy: groups (ceil (c / NX_CCOPY_THREADS), rows from
   [first], 1), c the columns from [start] that the launch takes, a
   thread an element (z, r, j) of a float32 or complex64 matrix of
   [parts] 1 or 2 float32 parts, its row y = z rows + r and its column j
   from [start], below [cols]. [src] and [dst] address element 0, their
   strides along the batch, rows and columns in elements. */
enum {
  NX_CCOPY_SRC,
  NX_CCOPY_DST,
  NX_CCOPY_SZ,
  NX_CCOPY_SR,
  NX_CCOPY_SC,
  NX_CCOPY_DZ,
  NX_CCOPY_DR,
  NX_CCOPY_DC,
  NX_CCOPY_ROWS,
  NX_CCOPY_COLS,
  NX_CCOPY_PARTS,
  NX_CCOPY_FIRST,
  NX_CCOPY_START,
  NX_CCOPY_WORDS,
};

/* An axis of a scaled operand: its extent, the operand's stride and the
   scale's, signed 64-bit integers. */
enum {
  NX_SCALE_AXIS_EXTENT,
  NX_SCALE_AXIS_SRC,
  NX_SCALE_AXIS_SCALE,
  NX_SCALE_AXIS_WORDS,
};

/* The most axes a scale's parameters hold: 7, which fill rig's 256 bytes
   of parameters a launch. */
enum { NX_SCALE_HOLDS = 7 };

/* contract_scale: the elements [first] to [count] - 1 of a scaled
   operand, in C order of their indices, each its cast into float32 times
   its scale, into consecutive float32 elements from [dst]. The element at
   index (i₀, …, i₍ᵣ₋₁₎) starts at bit [src_bit] + w Σ iⱼ sⱼ of [src], w
   its dtype's bits, and its scale at bit [scale_bit] + v Σ iⱼ tⱼ of
   [scale], v the scale's: the extents eⱼ and strides sⱼ and tⱼ, [rank]
   of each, NX_SCALE_AXIS_WORDS words an axis from the outermost, from
   [held] where [rank] is at most NX_SCALE_HOLDS and from [axes], a
   constant buffer, otherwise. [codes] is the operand's dtype code, plus
   the scale's times 256, plus NX_SCALE_EXPONENT for a scale of codes. */
enum {
  NX_SCALE_DST,
  NX_SCALE_SRC,
  NX_SCALE_SCALE,
  NX_SCALE_AXES,
  NX_SCALE_FIRST,
  NX_SCALE_COUNT,
  NX_SCALE_RANK,
  NX_SCALE_CODES,
  NX_SCALE_SRC_BIT,
  NX_SCALE_SCALE_BIT,
  NX_SCALE_HELD,
  NX_SCALE_WORDS = NX_SCALE_HELD + NX_SCALE_AXIS_WORDS * NX_SCALE_HOLDS,
};

enum { NX_SCALE_EXPONENT = 1 << 16 };

enum { NX_SCALE_THREADS = 256 };

/* contract_band: groups (ceil ([length] / NX_BAND_THREADS), S, G), a
   thread an element of one chunk of a line of a cast operand, for each
   of a tile call's S ranges of k terms and each of the line's G groups
   (one but for a Rows call's b). Range s holds the spans from
   ⌊s r / S⌋ to below ⌊(s + 1) r / S⌋ of the r = ⌈k / 32⌉, as a tile
   computes them, from [quot] = r / S, [rem] = r mod S and S's divisor
   word [ds] (contract.mli's divisor). The thread j of group (x, s)
   of group z takes term t = 32 ⌊s r / S⌋ + [chunk] [length] + j, if it
   lies in its range and below [k]: the element at bit
   [src_bit] + w (z [group] + t [stride]) of [src], w its dtype's bits,
   its cast into the staged dtype, stored as element (z S + s) [length] + j
   of [dst]. [codes] holds the operand's dtype
   code in bits 0-7 and the staged dtype's in bits 8-15. */
enum {
  NX_BAND_DST,
  NX_BAND_SRC,
  NX_BAND_SRC_BIT,
  NX_BAND_STRIDE,
  NX_BAND_GROUP,
  NX_BAND_CODES,
  NX_BAND_K,
  NX_BAND_QUOT,
  NX_BAND_REM,
  NX_BAND_DS,
  NX_BAND_LENGTH,
  NX_BAND_CHUNK,
  NX_BAND_WORDS,
};

enum { NX_BAND_THREADS = 256 };

/* contract_simt: groups (ceil ((last - first) / NX_SIMT_THREADS), 1,
   ranges), a thread an output o from [first] to below [last], of batch
   element z, row r and column c with o = (z m + r) n + c, over range s,
   the group's z. Range s's terms are a tile's, its chunk of [length] terms
   too. [a] and [b] address their operands as the finish's init, at
   positions [a_at] and [b_at], of the dtype codes [a_dtype] and
   [b_dtype], with strides in elements along the batch, a's rows, the
   terms and b's columns. Each element enters the accumulator of code
   [acc] by its cast. An integer accumulator's 64-bit sums wrap; a
   float32 one's are fused multiply-adds; a complex64 one's columns are
   its real form's, column c part c mod 2 of complex column c / 2, two
   fused multiply-adds a term. [y] holds the planes of 64-bit sums, or
   of float32 ones, [plane] elements apart: output o of range s at
   s plane + o. A chunk past its range's first adds to the plane. Where
   [segmented], row z m + r multiplies b's group j at j [b_g] from b's
   batch element, its entry of [buckets] z [groups] + j, or has no
   products where j is not below [groups].

   Where [kgroups] is not 0, of a Contracting call, z is group e = z /
   [batch] of operand batch element z' = z mod [batch], whose terms are
   entries [koff] z' (kgroups + 1) + e to below the next of [klist]: the
   output's K is their count, its spans of NX_CONTRACT_SPAN terms of the
   real form, and its ranges max (1, min ([cap], spans / 8)). A thread
   of a range past them writes an integer sum of 0 and returns. */
enum {
  NX_SIMT_A,
  NX_SIMT_B,
  NX_SIMT_Y,
  NX_SIMT_A_AT,
  NX_SIMT_A_PHASE,
  NX_SIMT_A_DTYPE,
  NX_SIMT_A_Z,
  NX_SIMT_A_R,
  NX_SIMT_A_K,
  NX_SIMT_B_AT,
  NX_SIMT_B_PHASE,
  NX_SIMT_B_DTYPE,
  NX_SIMT_B_Z,
  NX_SIMT_B_K,
  NX_SIMT_B_C,
  NX_SIMT_ACC,
  NX_SIMT_M,
  NX_SIMT_N,
  NX_SIMT_K,
  NX_SIMT_PLANE,
  NX_SIMT_FIRST,
  NX_SIMT_LAST,
  NX_SIMT_RANGES,
  NX_SIMT_SPANS,
  NX_SIMT_LENGTH,
  NX_SIMT_CHUNK,
  NX_SIMT_BUCKETS,
  NX_SIMT_GROUPS,
  NX_SIMT_B_G,
  NX_SIMT_SEGMENTED,
  NX_SIMT_KOFF,
  NX_SIMT_KLIST,
  NX_SIMT_KGROUPS,
  NX_SIMT_BATCH,
  NX_SIMT_CAP,
  NX_SIMT_WORDS,
};

enum { NX_SIMT_THREADS = 256 };

/* contract_compact: a Contracting call's prepass, in launches by
   [phase], over [batch] batch elements of [terms] terms each, cut into
   [parts] parts of ceil (terms / parts). Term q of batch element z is in
   group id where id, element z [ids_z] + ((q / inner) mod extent)
   [ids_r] of [ids], int64, has 0 <= id < [groups], and in none
   otherwise. [counts] holds each part's count of each group, at
   (z groups + e) parts + p, a batch element's groups and their parts in
   the order of [list]. A group holds NX_COMPACT_SHARED bytes of
   threadgroup memory.

     0  groups of the words: zeroes [counts]
     1  a group a part, (z parts + p): counts its terms' groups
     2  a group a batch element: each count becomes its part's first
        entry past z terms, the exclusive scan of the batch element's
        counts in order, chunks of NX_COMPACT_CHUNK at a time; [offsets]
        at z (groups + 1) + e gets group e's first entry of [list], from
        z terms, and past the last group its end
     3  a SIMD-group a part, NX_COMPACT_THREADS / 32 parts a group, in
        order: lists the part's terms in increasing order, so that a
        group's terms keep their order */
enum {
  NX_COMPACT_IDS,
  NX_COMPACT_COUNTS,
  NX_COMPACT_OFFSETS,
  NX_COMPACT_LIST,
  NX_COMPACT_IDS_Z,
  NX_COMPACT_IDS_R,
  NX_COMPACT_BATCH,
  NX_COMPACT_TERMS,
  NX_COMPACT_INNER,
  NX_COMPACT_EXTENT,
  NX_COMPACT_GROUPS,
  NX_COMPACT_PARTS,
  NX_COMPACT_PHASE,
  NX_COMPACT_WORDS,
};

/* A scan's chunk holds 8 counts a thread; its threadgroup memory, each
   SIMD-group's sum, the chunks' carry and the chunk's total. */
enum {
  NX_COMPACT_THREADS = 256,
  NX_COMPACT_CHUNK = 8 * NX_COMPACT_THREADS,
  NX_COMPACT_SHARED = 4 * (NX_COMPACT_THREADS / 32 + 2),
};

/* contract_buckets: one group, the prepass of a Rows call of [rows] rows
   y, batch element z = y / m and row r = y mod m, in [buckets] buckets.
   Row y's id is element z [ids_z] + ((r / inner) mod extent) [ids_r] of
   [ids], int64: its bucket is z [groups] + id where 0 <= id < [groups],
   else the last, [buckets] - 1, which [of] gets. Where [sort], over
   chunks of NX_BUCKETS_CHUNK buckets, [firsts] gets each bucket's first
   entry of [list], then from [buckets] + 1 its first tile row of
   [tile] rows, each with a last one past the others; [list] the rows,
   bucket after bucket; and each tile row i below [bound] of [table], 3
   words at 3 i, its bucket, its first entry of [list] and its rows, or
   2^32 - 1 alone where no rows are left. Its threadgroup memory
   holds 8 NX_BUCKETS_CHUNK + 8 (NX_BUCKETS_THREADS / 32 + 2) bytes. */
enum {
  NX_BUCKETS_IDS,
  NX_BUCKETS_OF,
  NX_BUCKETS_LIST,
  NX_BUCKETS_TABLE,
  NX_BUCKETS_FIRSTS,
  NX_BUCKETS_IDS_Z,
  NX_BUCKETS_IDS_R,
  NX_BUCKETS_ROWS,
  NX_BUCKETS_M,
  NX_BUCKETS_INNER,
  NX_BUCKETS_EXTENT,
  NX_BUCKETS_GROUPS,
  NX_BUCKETS_BUCKETS,
  NX_BUCKETS_TILE,
  NX_BUCKETS_BOUND,
  NX_BUCKETS_SORT,
  NX_BUCKETS_WORDS,
};

enum { NX_BUCKETS_THREADS = 1024, NX_BUCKETS_CHUNK = 2048 };

/* A vector kernel's launch: groups (passes, column groups, 1), x the
   pass of [r] rows from row r x, y the column group [first] + y of the
   call's batch elements' [groups_n] each. Its threadgroup's four
   SIMD-groups each hold 32 / [lanes] columns of c a lane, the lanes of
   a column dealt its spans. [a] addresses the other operand's element
   at batch, row and term 0, [lda] and [sa] its rows' and batch
   elements' strides in elements, its terms contiguous; [b] the wider
   operand's at batch, group, column and term 0, a float4_e2m1fn's at a
   whole byte, [ldb], [sb] and [sj] its columns', batch elements' and
   groups' strides in its elements, its terms contiguous but where
   [blocks] holds bl in its low 32 bits and bs, signed, in its high, bl
   not 0: blocks of bl contiguous terms, a multiple of 32, bs elements
   apart. A scaled key's element of batch element z, group j, column c
   and term t is scaled by its scale's element at z sz + j sj + (c / rn)
   sn + (t / rk) sk elements past [bs], as a loader's, rk a multiple of
   8 or at least [k]. [y] addresses the sums at batch, row and column
   0, float32, or the other operand's float16 or bfloat16 for a kernel
   named _typed, rounded as contract_finish rounds them, [ldy], [cy] and
   [sy] their strides in elements; [m], [n] and [k] the
   rows, columns and terms a batch element. A k x n kernel's [b] has its
   columns contiguous and [ldb] its terms' stride, and no blocks or
   scale; its threadgroup's eight SIMD-groups each hold [32 / rho']
   column lanes of [rho'] spans, rho' = min rho S, and [S / rho']
   SIMD-groups a column block. A Rows call ([rows] bit 0)
   takes one row a pass: row r of batch element z reads its id at z
   [ids_z] + ((r / [inner]) mod [extent]) [ids_r] of [ids], tested
   within [0, g), [groups] holding g in its low 32 bits and the bucket
   of no group in its high; a row of no group has no products and
   stores +0, and where [rows] bit 1 is set, the row's entry of [of], at
   z m + r, takes its bucket, z g + id, or the bucket of no group, as
   contract_buckets gives it the finish. [form] is the loads of an n x k
   kernel's whole spans, the same for every lane: 2, by 16-byte words,
   where every span of [b]'s columns and of [a]'s rows starts on 16
   bytes; 1, by 2-byte words, where [b]'s start on 2 bytes and [a]'s on
   16; with a scaled key's whole spans each within one scale block. 3,
   by 16-byte words as 2, where an int8 [b]'s scale blocks are of 16
   terms, two a span: only int8 by float32's twin runs it, and a kernel
   takes it as 0. At 0, and for a span ending past k, a lane loads by
   its own addresses. */
enum {
  NX_MV_A,
  NX_MV_B,
  NX_MV_Y,
  NX_MV_BS,
  NX_MV_IDS,
  NX_MV_OF,
  NX_MV_LDA,
  NX_MV_SA,
  NX_MV_LDB,
  NX_MV_SB,
  NX_MV_SJ,
  NX_MV_BLOCKS,
  NX_MV_BSZ,
  NX_MV_BSN,
  NX_MV_BSK,
  NX_MV_LDY,
  NX_MV_CY,
  NX_MV_SY,
  NX_MV_M,
  NX_MV_N,
  NX_MV_K,
  NX_MV_LANES,
  NX_MV_GROUPS_N,
  NX_MV_FIRST,
  NX_MV_IDS_Z,
  NX_MV_IDS_R,
  NX_MV_INNER,
  NX_MV_EXTENT,
  NX_MV_GROUPS,
  NX_MV_ROWS,
  NX_MV_FORM,
  NX_MV_WORDS,
};

/* The n x k kernels' threads, and the k x n kernels': [rho] span rows
   a SIMD-group, eight SIMD-groups a threadgroup. A k x n kernel's lane
   holds 16 bytes of adjacent columns, and its threadgroup memory 4
   bytes a column of each thread. */
enum { NX_MV_THREADS = 128, NX_MV_KN_THREADS = 256, NX_MV_RHO = 4 };

/* How an integer tile reads an operand's bytes: as unsigned or signed
   integers, or as booleans, true iff not zero (dtype.mli). */
enum { NX_BYTES_UNSIGNED = 0, NX_BYTES_SIGNED = 1, NX_BYTES_BOOLEAN = 2 };

/* Fourier transforms (fft.metal, whose header states how they run) */

/* M(X, code, input, scale, output) for each transpose fft.ml plans: what it
   reads, multiplies its points by and writes, named fft_t_<code>, and its
   64-bit twin fft_t_<code>_w. */
#define NX_METAL_FFT_MOVES(M, X)                                              \
  M(X, cpc, NX_FFT_IN_COMPLEX, NX_FFT_PLAIN, NX_FFT_OUT_COMPLEX)              \
  M(X, rpr, NX_FFT_IN_REAL, NX_FFT_PLAIN, NX_FFT_OUT_REAL)                    \
  M(X, rpc, NX_FFT_IN_REAL, NX_FFT_PLAIN, NX_FFT_OUT_COMPLEX)                 \
  M(X, bpc, NX_FFT_IN_BINS, NX_FFT_PLAIN, NX_FFT_OUT_COMPLEX)                 \
  M(X, ctc, NX_FFT_IN_COMPLEX, NX_FFT_TWIDDLE, NX_FFT_OUT_COMPLEX)            \
  M(X, cph, NX_FFT_IN_COMPLEX, NX_FFT_PLAIN, NX_FFT_OUT_HALF)                 \
  M(X, cpr, NX_FFT_IN_COMPLEX, NX_FFT_PLAIN, NX_FFT_OUT_REAL)                 \
  M(X, cic, NX_FFT_IN_COMPLEX, NX_FFT_CHIRP_IN, NX_FFT_OUT_COMPLEX)           \
  M(X, ric, NX_FFT_IN_REAL, NX_FFT_CHIRP_IN, NX_FFT_OUT_COMPLEX)              \
  M(X, bic, NX_FFT_IN_BINS, NX_FFT_CHIRP_IN, NX_FFT_OUT_COMPLEX)              \
  M(X, ckc, NX_FFT_IN_COMPLEX, NX_FFT_KERNEL, NX_FFT_OUT_COMPLEX)             \
  M(X, coc, NX_FFT_IN_COMPLEX, NX_FFT_CHIRP_OUT, NX_FFT_OUT_COMPLEX)          \
  M(X, coh, NX_FFT_IN_COMPLEX, NX_FFT_CHIRP_OUT, NX_FFT_OUT_HALF)             \
  M(X, cor, NX_FFT_IN_COMPLEX, NX_FFT_CHIRP_OUT, NX_FFT_OUT_REAL)

#define NX_FFT_MOVE_ROWS_(X, code, input, scale, output)                      \
  X(fft_t_##code, NX_FAMILY_FFT, NX_FFT_MOVE_THREADS, NX_FFT_MOVE_WORDS)      \
  X(fft_t_##code##_w, NX_FAMILY_FFT_WIDE, NX_FFT_MOVE_THREADS,                \
    NX_FFT_MOVE_WORDS)

#define NX_METAL_FFT_KERNELS(X)                                               \
  NX_METAL_FFT_MOVES(NX_FFT_MOVE_ROWS_, X)                                    \
  X(fft_split, NX_FAMILY_FFT, NX_FFT_PAIR_THREADS, NX_FFT_PAIR_WORDS)         \
  X(fft_split_w, NX_FAMILY_FFT_WIDE, NX_FFT_PAIR_THREADS, NX_FFT_PAIR_WORDS)  \
  X(fft_prepass, NX_FAMILY_FFT, NX_FFT_PAIR_THREADS, NX_FFT_PAIR_WORDS)       \
  X(fft_prepass_w, NX_FAMILY_FFT_WIDE, NX_FFT_PAIR_THREADS, NX_FFT_PAIR_WORDS)  \
  X(fft_s4096, NX_FAMILY_FFT, NX_FFT_STEP_THREADS, NX_FFT_STEP_WORDS)        \
  X(fft_c4096_first, NX_FAMILY_FFT, NX_FFT_STEP_THREADS, NX_FFT_COLS_WORDS)  \
  X(fft_c4096_last, NX_FAMILY_FFT, NX_FFT_STEP_THREADS, NX_FFT_COLS_WORDS)  \
  X(fft_step_p8, NX_FAMILY_FFT_STEPS, NX_FFT_KEY_THREADS_P8, NX_FFT_KEY_WORDS) \
  X(fft_step_p16, NX_FAMILY_FFT_STEPS, NX_FFT_KEY_THREADS_P16,                 \
    NX_FFT_KEY_WORDS)                                                          \
  X(fft_side_p8, NX_FAMILY_FFT_STEPS, NX_FFT_SIDE_THREADS, NX_FFT_KEY_WORDS)  \
  X(fft_side_p16, NX_FAMILY_FFT_STEPS, NX_FFT_SIDE_THREADS, NX_FFT_KEY_WORDS)

/* What a transpose reads, a point at a time: a complex point, or a pair of
   floats; one float, its imaginary part +0; a c2r's bins up to n / 2,
   completed to n points. */
enum { NX_FFT_IN_COMPLEX, NX_FFT_IN_REAL, NX_FFT_IN_BINS };

/* What it multiplies a point by: nothing; the twiddle w_(a b)^(r c) of row r
   and column c; Bluestein's chirp at its operand's point, zero past n;
   Bluestein's kernel at its bin, then the conjugate; the conjugate, then the
   chirp at its bin, written below n alone. */
enum {
  NX_FFT_PLAIN,
  NX_FFT_TWIDDLE,
  NX_FFT_CHIRP_IN,
  NX_FFT_KERNEL,
  NX_FFT_CHIRP_OUT,
};

/* What it writes: each point, complex or a pair of floats; an r2c's bins up
   to n / 2; a c2r's real parts. */
enum { NX_FFT_OUT_COMPLEX, NX_FFT_OUT_HALF, NX_FFT_OUT_REAL };

/* A side of a transpose: its lines' first point, at [first] + Σ i_k s_k
   elements for line (i_0, ...) of extents e_k and strides s_k, and a line's
   point j at [ps] j elements past it; [rank] axes, held from [axes] where
   [rank] is at most NX_FFT_SIDE_HOLDS, and past that from the transpose's
   tail, src's axes then dst's; [width] its elements' bytes, 4 or 8. A complex
   point of a 4-byte side is the pair of floats 2 j and 2 j + 1. An axis is
   its extent e, its stride, and e's divisor for device.h's nx_divide, so
   that a line's coordinates take no division. */
enum {
  NX_FFT_AXIS_EXTENT,
  NX_FFT_AXIS_STRIDE,
  NX_FFT_AXIS_DIVISOR,
  NX_FFT_AXIS_WORDS,
};

enum { NX_FFT_SIDE_HOLDS = 2 };

enum {
  NX_FFT_SIDE_FIRST,
  NX_FFT_SIDE_PS,
  NX_FFT_SIDE_RANK,
  NX_FFT_SIDE_WIDTH,
  NX_FFT_SIDE_AXES,
  NX_FFT_SIDE_WORDS = NX_FFT_SIDE_AXES + NX_FFT_AXIS_WORDS * NX_FFT_SIDE_HOLDS,
};

/* A transpose: blocks [first] on, the launch's groups, each a tile of 32
   rows r < [a] by 32 columns c < [b] of a grid, rows = ⌈a / 32⌉ tiles of
   rows a column of tiles and tiles = rows ⌈b / 32⌉ a run, a run's by column,
   then by row, with [rows_divisor] and [tiles_divisor] their divisors. Over a
   split's rows of [b] points, [a] of them a run, run i is line i of both
   sides, its point j = r b + c written as point c a + r; a move
   (NX_FFT_MOVE_ONE) is one run whose [a] rows are the lines and whose [b]
   columns their points, each written where it was read. [n] is the line's
   points, for the chirp, the bins and the halves. [mode]'s bits also say
   whether it reads, and whether it writes, with lanes along r, else along
   c, and with NX_FFT_CONJ, whether it conjugates each point it loads and
   each it stores, a negation of the imaginary part, around the rest: a
   read's +0 imaginary parts are set after the load's. [lo] and [hi] are
   the twiddle's tables, or [lo] the chirp or the kernel, each of the
   forward transform. */
enum {
  NX_FFT_MOVE_SRC,
  NX_FFT_MOVE_DST,
  NX_FFT_MOVE_LO,
  NX_FFT_MOVE_HI,
  NX_FFT_MOVE_TAIL,
  NX_FFT_MOVE_FIRST,
  NX_FFT_MOVE_ROWS_DIVISOR,
  NX_FFT_MOVE_TILES_DIVISOR,
  NX_FFT_MOVE_A,
  NX_FFT_MOVE_B,
  NX_FFT_MOVE_N,
  NX_FFT_MOVE_MODE,
  NX_FFT_MOVE_IN,
  NX_FFT_MOVE_OUT = NX_FFT_MOVE_IN + NX_FFT_SIDE_WORDS,
  NX_FFT_MOVE_WORDS = NX_FFT_MOVE_OUT + NX_FFT_SIDE_WORDS,
};

enum {
  NX_FFT_READ_ALONG_R = 1,
  NX_FFT_WRITE_ALONG_R = 2,
  NX_FFT_MOVE_ONE = 4,
  NX_FFT_CONJ = 8,
};

/* A split or a pre-pass of a long halved line of [h] points: blocks [first]
   on, the launch's groups, blocks = ⌈(h / 2 + 1) / NX_FFT_PAIR_BLOCK⌉ of
   them a line, [blocks_divisor] its divisor, each [NX_FFT_PAIR_BLOCK] of its
   pairs k <= h / 2. Pair k takes the root w = w_2h^k of the forward
   transform, lo[k mod 2^NX_FFT_LOW_BITS] hi[k / 2^NX_FFT_LOW_BITS], [lo] and
   [hi] the twiddle's of 2 h. With E = a + conj b and D = a - conj b,
   U = fma(w.re, D.im, w.im D.re) and V = fma(-w.re, D.re, w.im D.im), the
   parts of -i w D: a split, in place on its out side, reads a = Z_k and
   b = Z_(h-k), Z_0 at k = 0, and writes X_k = ((E.re + U) / 2,
   (E.im + V) / 2), then X_(h-k) = ((E.re - U) / 2, (V - E.im) / 2); a
   pre-pass reads bins a = X_k and b = X_(h-k) of its in side, their
   imaginary parts +0 at k = 0, and writes Z'_k = (E.re + U, E.im + V) on
   its out side, and Z'_(h-k) = (E.re - U, V - E.im) where 0 < k and
   2 k != h. With NX_FFT_CONJ in [mode], each conjugates the points it
   loads and those it stores, as a transpose. A side's points are complex
   elements; its tail as a transpose's. */
enum {
  NX_FFT_PAIR_SRC,
  NX_FFT_PAIR_DST,
  NX_FFT_PAIR_LO,
  NX_FFT_PAIR_HI,
  NX_FFT_PAIR_TAIL,
  NX_FFT_PAIR_FIRST,
  NX_FFT_PAIR_BLOCKS_DIVISOR,
  NX_FFT_PAIR_H,
  NX_FFT_PAIR_MODE,
  NX_FFT_PAIR_IN,
  NX_FFT_PAIR_OUT = NX_FFT_PAIR_IN + NX_FFT_SIDE_WORDS,
  NX_FFT_PAIR_WORDS = NX_FFT_PAIR_OUT + NX_FFT_SIDE_WORDS,
};

/* A step of NX_FFT_STEP_POINTS = 4096 points, radices 16 16 16, one line a
   group: blocks [first] on, the launch's groups, block i the line i of both
   sides, read from its in side and written to its out side, which may be
   the same. Its NX_FFT_STEP_THREADS threads take 16 points each. Pass 0
   reads point t + 256 r into a thread's v[r], r < 16; each pass takes a
   butterfly of 16 points a thread, a 16-point DFT of the form below, and
   pass p > 0 first multiplies v[r], r > 0, by w_4096^(16^(2-p) (t mod
   16^p) r) from [lo], the inner table, whose entry e < 2048 is w_4096^e,
   w_4096^(e + 2048) = -lo[e]. Pass p writes v[r] to plane slot
   (t / 16^p) 16^(p+1) + t mod 16^p + 16^p r, the next pass reading slot
   t + 256 r; the last pass writes v[r] to point t + 256 r. The DFT of 16
   points: a DFT of 4 over n2 of v[n1 + 4 n2] for each n1 < 4, v[n1 + 4
   k2] times w_16^(n1 k2) as fft.metal's fft_dft16 forms it, then a DFT
   of 4 over n1 of each k2 into v[k2 + 4 k1]. A DFT of 4 over (a0, a1,
   a2, a3): t0 = a0 + a2, t1 = a0 - a2, t2 = a1 + a3, t3 = a1 - a3, then
   (t0 + t2, (t1.re + t3.im, t1.im - t3.re), t0 - t2,
   (t1.re - t3.im, t1.im + t3.re)). The plane holds 4096 complex points,
   slot x at x ^ ((x >> 4) & 15) ^ ((x >> 8) & 15), so that each
   exchange's half SIMD groups fall in distinct banks. With NX_FFT_CONJ in
   [mode], it conjugates each point it loads and each it stores, as a
   transpose. */
enum {
  NX_FFT_STEP_SRC,
  NX_FFT_STEP_DST,
  NX_FFT_STEP_LO,
  NX_FFT_STEP_HI,
  NX_FFT_STEP_TAIL,
  NX_FFT_STEP_FIRST,
  NX_FFT_STEP_MODE,
  NX_FFT_STEP_IN,
  NX_FFT_STEP_OUT = NX_FFT_STEP_IN + NX_FFT_SIDE_WORDS,
  NX_FFT_STEP_WORDS = NX_FFT_STEP_OUT + NX_FFT_SIDE_WORDS,
};

enum {
  NX_FFT_STEP_THREADS = 256,
  NX_FFT_STEP_POINTS = 4096,
  NX_FFT_STEP_SHARED = 8 * NX_FFT_STEP_POINTS,
};

/* A column step: the step above on 16 adjacent lines a group, in two
   launches through the workspace, fft_c4096_first then fft_c4096_last. A
   side's lines lie in runs of [lines] adjacent ones, one element apart, the
   runs in C order over its other axes; block b of a run holds its lines
   16 b to 16 b + 15 below [lines]. Groups [first] on, the launch's groups,
   each take block g mod [blocks] of run g / (16 [blocks]) and part
   k = (g / [blocks]) mod 16, [blocks_divisor] the divisor of [blocks], and
   lane i line 16 b + i mod 16 of the block. fft_c4096_first: thread
   t = 16 k + i / 16 of each line reads points t + 256 r, r < 16, takes
   their DFT of 16, the step's pass 0, and writes value r to point
   256 r + t of its out side. fft_c4096_last, class m = k: thread
   t = m + 16 u, u = i / 16, reads point 256 m + u + 16 r of its in side
   where the step's pass 1 reads plane position t + 256 r, takes the step's
   passes 1 and 2 through a plane of the block's lines, value p of line c
   at slot 16 p + c, and writes value r to point t + 256 r of its out side.
   With NX_FFT_CONJ in [mode], the first conjugates each point it loads and
   the last each it stores. */
enum {
  NX_FFT_COLS_SRC,
  NX_FFT_COLS_DST,
  NX_FFT_COLS_LO,
  NX_FFT_COLS_HI,
  NX_FFT_COLS_TAIL,
  NX_FFT_COLS_FIRST,
  NX_FFT_COLS_MODE,
  NX_FFT_COLS_BLOCKS,
  NX_FFT_COLS_BLOCKS_DIVISOR,
  NX_FFT_COLS_LINES,
  NX_FFT_COLS_IN,
  NX_FFT_COLS_OUT = NX_FFT_COLS_IN + NX_FFT_SIDE_WORDS,
  NX_FFT_COLS_WORDS = NX_FFT_COLS_OUT + NX_FFT_SIDE_WORDS,
};

/* A specialised step (fft_steps.metal): a step of any 7-smooth number of
   points [l] up to 4096, its passes' radices [r_0], ..., [r_(passes-1)],
   each 2, 3, 4, 5, 7, 8 or 16, whose product is [l], and its group's lines
   and threads constants of its pipeline, which function constants k of
   NX_FFT_FC_* give: a key. fft_step_p8 and fft_side_p8 hold 8 points a
   thread, every radix at most 8; fft_step_p16 and fft_side_p16, 16. Of
   fft_step_p8 and fft_step_p16, line i < [fc_lines] of a group takes its
   threads tl i to tl i + tl - 1, tl = [fc_tl], thread j of a line being
   thread tl i + j; of fft_side_p8 and fft_side_p16, its lines side by
   side, thread j of line i is thread [fc_lines] j + i, j < tl. Threads
   past them take part in the group's barriers alone. Blocks [first] on,
   the launch's groups, block b of [fc_lines] lines, b [fc_lines] + i,
   below [lines], the count of each side's lines; a line past it loads and
   stores nothing.

   Pass q of radix R after ns = r_0 ... r_(q-1) points, Q = l / R
   butterflies: butterfly b = j + tl k, k < ⌊p / R⌋, b < Q, takes points
   b + Q r, r < R, from the in side at pass 0 and from the plane past it,
   where pass q > 0 first multiplies point r > 0 by
   w_l^((b mod ns) (l / (ns R)) r) from [lo], the inner table: w_l^e at
   entry e of an odd l's, and of an even l's for e < l / 2, whose
   w_l^(e + l/2) = -lo[e]. Then the DFT of R points of the form below, and
   value r to point (b / ns) ns R + b mod ns + ns r: of the plane where a
   pass follows, of the out side at the last pass. A pass that writes the
   plane first waits for every thread's reads, and its writes for the next
   pass's reads. Of fft_step_p8 and fft_step_p16, a line's plane holds l
   complex points, l + l / 16 for a power of two up to 2048, point x at
   slot x + x / 16. Of fft_side_p8 and fft_side_p16, the group's plane
   holds one float a point of each line, point x of line i at
   x [fc_lines] + i, and a pass moves the real parts, then the imaginary
   parts, through it. A step of one point has no pass: its point times
   [lo]'s entry 0, a part at a time, so that a subnormal part flushes as
   every operation's operand does.

   The DFTs: of 4 and 16 as fft_s4096's; of 2, (a0 + a1, a0 - a1); of 8,
   a DFT of 4 of the sums a_k + a_(k+4), k < 4, into the even outputs, and
   of the differences d_k times w_8^k, the products by c = 1/√2 fused into
   the odd outputs as fft_steps.metal forms them; of a prime P of 3, 5 or
   7, H = (P - 1) / 2: t_j = a_j + a_(P-j), d_j = a_j - a_(P-j), y_0 = a_0 +
   t_1 + ... + t_H in order, and for k <= H, m = fma(cos(2π jk/P), t_j, m)
   over j from 1 on m = a_0, n = sin(2π k/P) d_1, then fma(sin(2π jk/P),
   d_j, n) over j from 2, y_k = m - i n and y_(P-k) = m + i n, each
   constant the float32 nearest it. With NX_FFT_CONJ in [mode], it
   conjugates each point it loads and each it stores, as a transpose.

   Its sides by [fc_input] and [fc_output]: NX_FFT_IN_REAL loads a float a
   point, its imaginary part +0, or for a split floats 2 p and 2 p + 1 as
   point p; NX_FFT_IN_BINS, a c2r's odd line, bin p <= n / 2, bin 0's
   imaginary part +0, and past it conj X_(n - p); NX_FFT_IN_NONE, +0.
   NX_FFT_OUT_HALF stores points p <= n / 2 alone; NX_FFT_OUT_REAL a
   point's real part, or for a pre-pass both parts at floats 2 p and
   2 p + 1; n is the line's points, l but for a convolution. A halved
   line of h = n points takes [hi], w_2h^k at entry
   k <= h / 2. Its pair k with w = w_2h^k, from a and b, forms
   E = a + conj b, D = a - conj b, U = fma(w.re, D.im, w.im D.re) and
   V = fma(-w.re, D.re, w.im D.im). A split (NX_FFT_HALVED_SPLIT) writes
   the last pass's points Z to the plane in natural order, after a
   barrier, and each thread's pairs k = j + tl m <= h / 2, from a = Z_k
   and b = Z_(h-k), Z_0 at k = 0, store bins
   X_k = ((E.re + U) / 2, (E.im + V) / 2), then
   X_(h-k) = ((E.re - U) / 2, (V - E.im) / 2), each conjugated in an
   inverse. A pre-pass (NX_FFT_HALVED_PREPASS) loads bins a = X_k and
   b = X_(h-k) of each such pair, bins 0 and h at k = 0, conjugated in an
   inverse, their imaginary parts then +0 at k = 0, and writes
   Z'_k = (E.re + U, E.im + V) and, where 0 < k and 2 k != h,
   Z'_(h-k) = (E.re - U, V - E.im) to the plane, which pass 0 reads
   unconjugated. A step of one point splits its product as Z_0, and a
   pre-pass makes its point before the product.

   A convolution by [fc_conv] runs the FFT of the passes above, F, twice
   on the line of [n] points the word gives, its products fft_cmul's.
   Rader's (NX_FFT_CONV_RADER), n = l + 1 prime, g its least primitive
   root, [logs] holding s at entry g^s mod n, [kernel] its B: each thread
   loads x_0, conjugated in an inverse; a_s = x_(g^s), s < l, loaded
   into the plane at slot logs[g^s], then pass 0's points read from it;
   A = F a; the thread of A_0 takes y_0 = A_0 + x_0; c = F conj(A_k B_k)
   through the plane; then c into the plane in natural order and point
   p > 0 stores x_0 + conj c_q, q = (l - logs[p]) mod l, and the line's
   first thread y_0, each conjugated in an inverse. Bluestein's
   (NX_FFT_CONV_BLUESTEIN), 2 n - 1 <= l, [chirp] its chirp of n,
   [kernel] its B of l: pass 0's points z_j = x_j chirp_j for j < n, as
   loaded or as a pre-pass makes them, +0 past n; Z = F z;
   c = F conj(Z_k B_k) through the plane; then point k < n,
   conj(c_k) chirp_k, stored or split. */
enum {
  NX_FFT_KEY_SRC,
  NX_FFT_KEY_DST,
  NX_FFT_KEY_LO,
  NX_FFT_KEY_HI,
  NX_FFT_KEY_TAIL,
  NX_FFT_KEY_FIRST,
  NX_FFT_KEY_MODE,
  NX_FFT_KEY_LINES,
  NX_FFT_KEY_KERNEL,
  NX_FFT_KEY_CHIRP,
  NX_FFT_KEY_LOGS,
  NX_FFT_KEY_N,
  NX_FFT_KEY_IN,
  NX_FFT_KEY_OUT = NX_FFT_KEY_IN + NX_FFT_SIDE_WORDS,
  NX_FFT_KEY_WORDS = NX_FFT_KEY_OUT + NX_FFT_SIDE_WORDS,
};

/* A specialised step's function constants, by index: every one required.
   [fc_l] (uint), its points; [fc_tl] (ushort), the threads a line takes;
   [fc_lines] (ushort), the lines a group takes; [fc_passes] (uchar), at
   most NX_FFT_FC_RADICES; [fc_r0] to [fc_r6] (uchar), the passes'
   radices, 0 past [fc_passes]; [fc_halved] (uchar), an NX_FFT_HALVED_*;
   [fc_input] (uchar), an NX_FFT_IN_*; [fc_output] (uchar), an
   NX_FFT_OUT_*; and [fc_conv] (uchar), an NX_FFT_CONV_*. */
enum {
  NX_FFT_FC_L,
  NX_FFT_FC_TL,
  NX_FFT_FC_LINES,
  NX_FFT_FC_PASSES,
  NX_FFT_FC_R0,
  NX_FFT_FC_RADICES = 7,
  NX_FFT_FC_HALVED = NX_FFT_FC_R0 + NX_FFT_FC_RADICES,
  NX_FFT_FC_INPUT,
  NX_FFT_FC_OUTPUT,
  NX_FFT_FC_CONV,
  NX_FFT_FC_COUNT,
};

/* The convolution a step's line runs in its group, of [n] points by the
   step's FFT of l: none, n = l; Rader's, n = l + 1 prime; or Bluestein's,
   2 n - 1 <= l. */
enum {
  NX_FFT_CONV_NONE,
  NX_FFT_CONV_RADER,
  NX_FFT_CONV_BLUESTEIN,
};

/* Whether a step's line is a halved real line's, of [h] points: none;
   an r2c's, whose store splits its FFT into bins; or a c2r's, whose load
   makes the points its FFT takes from bins. */
enum {
  NX_FFT_HALVED_NONE,
  NX_FFT_HALVED_SPLIT,
  NX_FFT_HALVED_PREPASS,
};

/* What a step's load reads, past a transpose's NX_FFT_IN_*: nothing, an
   r2c's line of no point. A step's NX_FFT_IN_REAL reads a float a point,
   or a pair of floats a point for a split; its NX_FFT_OUT_REAL writes a
   point's real part, or both its parts for a pre-pass. */
enum { NX_FFT_IN_NONE = NX_FFT_IN_BINS + 1 };

/* The most threads of a group of fft_step_p8, whose lines take at most 256,
   and of fft_step_p16, whose longest take 288, one line a group; the
   threads of a group of fft_side_p8 and fft_side_p16. */
enum {
  NX_FFT_KEY_THREADS_P8 = 256,
  NX_FFT_KEY_THREADS_P16 = 288,
  NX_FFT_SIDE_THREADS = 512,
};

/* A pair's group: 256 threads, 4 pairs each. */
enum { NX_FFT_PAIR_THREADS = 256, NX_FFT_PAIR_BLOCK = 1024 };

/* A transpose's group: 256 threads, 4 points each of a 32 x 32 tile held in
   threadgroup memory as 32 rows of 33 complex points, and past it, for a
   move, its 32 lines' first points on each side, src's then dst's, a 64-bit
   position each: NX_FFT_MOVE_SHARED bytes. The twiddle's low table holds
   2^NX_FFT_LOW_BITS entries. */
enum {
  NX_FFT_MOVE_THREADS = 256,
  NX_FFT_TILE = 32,
  NX_FFT_LOW_BITS = 10,
  NX_FFT_MOVE_SHARED = 8 * NX_FFT_TILE * (NX_FFT_TILE + 1) + 8 * 2 * NX_FFT_TILE,
};

/* Sorts and numberings (sort.metal)

   A sort's kernels come in three widths of element, [_4] for elements of
   at most 4 bytes, [_8] and [_16], so that a narrow element's registers
   and code bound no wider one's; sort_scatter's [w] variants carry 64-bit
   positions, for slices past NX_SORT_WIDE's bound. */

#define NX_SORT_WIDTHS(X, k)                                                   \
  X(k##_4, NX_FAMILY_SORT, NX_SORT_THREADS, NX_SORT_WORDS)                     \
  X(k##_8, NX_FAMILY_SORT, NX_SORT_THREADS, NX_SORT_WORDS)                     \
  X(k##_16, NX_FAMILY_SORT, NX_SORT_THREADS, NX_SORT_WORDS)

#define NX_METAL_SORT_KERNELS(X)                                               \
  NX_SORT_WIDTHS(X, sort_block)                                                \
  NX_SORT_WIDTHS(X, sort_hist)                                                 \
  NX_SORT_WIDTHS(X, sort_count)                                                \
  NX_SORT_WIDTHS(X, sort_scatter)                                              \
  NX_SORT_WIDTHS(X, sort_scatter_wide)                                         \
  NX_SORT_WIDTHS(X, sort_round)                                                \
  NX_SORT_WIDTHS(X, sort_gather)                                               \
  X(unique_count, NX_FAMILY_SORT, NX_SORT_THREADS, NX_UNIQUE_WORDS)            \
  X(unique_heads, NX_FAMILY_SORT, NX_SORT_THREADS, NX_UNIQUE_WORDS)            \
  X(unique_rank, NX_FAMILY_SORT, NX_SORT_THREADS, NX_UNIQUE_WORDS)             \
  X(unique_ids, NX_FAMILY_SORT, NX_SORT_THREADS, NX_UNIQUE_WORDS)

/* A group of NX_SORT_THREADS threads. sort_block holds NX_SORT_BLOCK_ITEMS
   elements a thread, NX_SORT_BLOCK_ITEMS16 of 16 bytes, so a slice of at
   most NX_SORT_CAP or NX_SORT_CAP16 elements sorts in one group.
   sort_hist, sort_round and sort_gather take a slice's NX_SORT_CHUNK
   elements a group; sort_count and sort_scatter a part of a slice's
   tiles, of NX_SORT_THREADS times NX_SORT_TILE_ITEMS elements a width, at
   most NX_SORT_PARTS parts a slice. A numbering's kernels take
   NX_UNIQUE_TILE rows a step. */
enum {
  NX_SORT_THREADS = 256,
  NX_SORT_BLOCK_ITEMS = 16,
  NX_SORT_BLOCK_ITEMS16 = 8,
  NX_SORT_CAP = NX_SORT_THREADS * NX_SORT_BLOCK_ITEMS,
  NX_SORT_CAP16 = NX_SORT_THREADS * NX_SORT_BLOCK_ITEMS16,
  NX_SORT_CHUNK = 16384,
  NX_SORT_TILE_ITEMS4 = 8,
  NX_SORT_TILE_ITEMS8 = 8,
  NX_SORT_TILE_ITEMS16 = 4,
  NX_SORT_PARTS = 256,
  NX_UNIQUE_TILE = 4096,
};

/* The threadgroup memory of nx_rank's counts, starts and sums (device.h),
   and of a scatter's beside its exchange: those and 256 64-bit places. */
enum {
  NX_SORT_RANK_BYTES =
      (NX_SORT_THREADS / 32) * 257 * 2 + 2 * 257 * 4 + (NX_SORT_THREADS / 32) * 4,
  NX_SORT_SCATTER_REST = NX_SORT_RANK_BYTES + 256 * 8,
};

/* A key's format (sort.metal, Keys). */
enum {
  NX_SORT_UNSIGNED,
  NX_SORT_SIGNED,
  NX_SORT_FLOAT,
  NX_SORT_BOOL,
  NX_SORT_COMPLEX,
};

/* A sort's flags: it stores the values; its first pass reads x through
   [order]; sort_block sorts each slice's candidates; its slices' counts
   and places take 64 bits, and its passes' positions; its slices'
   offsets take 64-bit divisions. */
enum {
  NX_SORT_STORE = 1,
  NX_SORT_ORDERED = 2,
  NX_SORT_CANDIDATES = 4,
  NX_SORT_WIDE = 8,
  NX_SORT_HUGE = 16,
};

/* A kept axis of a sort's slices, NX_SORT_AXIS_WORDS words: its extent,
   its divisor below 2^32 (device.h's divide), and the steps of x, the
   values and the positions along it, in elements. */
enum {
  NX_SORT_AXIS_EXTENT,
  NX_SORT_AXIS_DIV,
  NX_SORT_AXIS_SX,
  NX_SORT_AXIS_SV,
  NX_SORT_AXIS_SP,
  NX_SORT_AXIS_WORDS,
};

/* The kept axes a sort's parameters hold: past them, a constant buffer at
   [axes] holds them. */
enum { NX_SORT_HOLDS = 8 };

/* A sort's parameters, one form for every sort kernel. Buffers: x; the
   values (under NX_SORT_STORE) and positions; the tickets; a slice's
   bins, or its select; each slice's passes ([meta]); each part's counts,
   then places; the passes' buffers, each slice's elements then its
   positions, [stride] bytes a slice from the buffer's [e] word; a
   select's candidates, elements and positions apart; x's order (under
   NX_SORT_ORDERED); the kept axes past NX_SORT_HOLDS. Then the elements
   a slice keeps, the flags, the pass (a key byte, or a select's round),
   the launch's units from [first] to [end] - 1, a slice's [units] and
   their divisor, a slice's tiles, a part's [span] of tiles, a pass
   buffer's bytes a slice, the slices a
   block's group takes, the bytes of a select's positions, the bit [carry]
   past which a wide bin's low word carries into its high one, and the
   bits before a sub-byte values' buffer's first byte in its 32-bit word.
   Then the key's format: its kind, its element's bits, its bytes, the
   magnitude past which a float is NaN, the key's mask, the bits a key
   flips where its sign is clear and where it is set, and the bits a
   descending sort flips, low and high. Then the slices: [n] elements
   each and n's divisor, [count] of them, the steps of x, the values and
   the positions along the axis, the offsets of their first elements, and
   the [kept] axes, held from [held]. */
enum {
  NX_SORT_X,
  NX_SORT_VALUES,
  NX_SORT_POSITIONS,
  NX_SORT_TICKETS,
  NX_SORT_HIST,
  NX_SORT_META,
  NX_SORT_COUNTS,
  NX_SORT_E0,
  NX_SORT_P0,
  NX_SORT_E1,
  NX_SORT_P1,
  NX_SORT_CAND,
  NX_SORT_CAND_P,
  NX_SORT_ORDER,
  NX_SORT_AXES,
  NX_SORT_KEEP,
  NX_SORT_FLAGS,
  NX_SORT_PASS,
  NX_SORT_FIRST,
  NX_SORT_END,
  NX_SORT_UNITS,
  NX_SORT_UNITS_DIV,
  NX_SORT_TILES,
  NX_SORT_SPAN,
  NX_SORT_STRIDE,
  NX_SORT_PER,
  NX_SORT_PBYTES,
  NX_SORT_CARRY,
  NX_SORT_SKEW,
  NX_SORT_KIND,
  NX_SORT_EBITS,
  NX_SORT_KBYTES,
  NX_SORT_NAN_ABS,
  NX_SORT_KMASK,
  NX_SORT_FLIP,
  NX_SORT_FLIP_SIGNED,
  NX_SORT_DMASK_LO,
  NX_SORT_DMASK_HI,
  NX_SORT_N,
  NX_SORT_N_DIV,
  NX_SORT_COUNT,
  NX_SORT_SX,
  NX_SORT_SV,
  NX_SORT_SP,
  NX_SORT_OFF_X,
  NX_SORT_OFF_V,
  NX_SORT_OFF_P,
  NX_SORT_KEPT,
  NX_SORT_HELD,
  NX_SORT_WORDS = NX_SORT_HELD + NX_SORT_AXIS_WORDS * NX_SORT_HOLDS,
};

/* A slice's bin in sort_hist's counts: a 32-bit count, then its first
   place, NX_SORT_BIN words; under NX_SORT_WIDE a low and a high word,
   then a 64-bit place, NX_SORT_BIN_WIDE. A select's state:
   NX_SORT_SELECT_BINS counts of a bin each, then the prefix's 32 bytes,
   the elements still to find, the candidates, the rounds, and whether it
   is done; NX_SORT_SELECT_WORDS 32-bit words, NX_SORT_SELECT_WORDS_WIDE
   under NX_SORT_WIDE. */
enum {
  NX_SORT_BIN = 2,
  NX_SORT_BIN_WIDE = 4,
  NX_SORT_SELECT_BINS = 256,
  NX_SORT_SELECT_PREFIX = 0,
  NX_SORT_SELECT_NEED = 8,
  NX_SORT_SELECT_CNT = 10,
  NX_SORT_SELECT_ROUNDS = 12,
  NX_SORT_SELECT_DONE = 13,
  NX_SORT_SELECT_STATE = 16,
  NX_SORT_SELECT_WORDS = NX_SORT_SELECT_STATE + NX_SORT_SELECT_BINS,
  NX_SORT_SELECT_WORDS_WIDE = NX_SORT_SELECT_STATE + 2 * NX_SORT_SELECT_BINS,
};

/* A numbering's flags: its rows are sorted into [order]; its one word is
   sorted into [skeys]. */
enum {
  NX_UNIQUE_BY_ORDER = 1,
  NX_UNIQUE_BY_SKEYS = 2,
};

/* A numbering's parameters. Buffers: the ids, first rows, counts and
   count it writes; the keys, or the one word sorted; the sorted rows'
   order; each part's heads; the tickets; each sorted row's key, each
   key's first row and first sorted row; the key count; a bit a first
   row, 32 a word, the first rows set before each word, and each part of
   the words' first bit. Then [m] entries of first and counts, [n] rows of
   [w] words at steps [s0] and [s1], the rows a part takes ([span]) and
   the words ([wspan]), the flags, and the launch's parts from [first] to
   [end] - 1. */
enum {
  NX_UNIQUE_IDS,
  NX_UNIQUE_FIRST,
  NX_UNIQUE_COUNTS,
  NX_UNIQUE_COUNT,
  NX_UNIQUE_KEYS,
  NX_UNIQUE_SKEYS,
  NX_UNIQUE_ORDER,
  NX_UNIQUE_HEADS,
  NX_UNIQUE_TICKETS,
  NX_UNIQUE_GID,
  NX_UNIQUE_GFIRST,
  NX_UNIQUE_GSTART,
  NX_UNIQUE_K,
  NX_UNIQUE_BITS,
  NX_UNIQUE_BELOW,
  NX_UNIQUE_SET,
  NX_UNIQUE_M,
  NX_UNIQUE_N,
  NX_UNIQUE_W,
  NX_UNIQUE_S0,
  NX_UNIQUE_S1,
  NX_UNIQUE_SPAN,
  NX_UNIQUE_WSPAN,
  NX_UNIQUE_FLAGS,
  NX_UNIQUE_FIRST_UNIT,
  NX_UNIQUE_END,
  NX_UNIQUE_WORDS,
};

/* Factorisations (factor.metal)

   Matrices of a buffer: element (i, j) of batch element z at the address
   word's byte plus z, i and j times their strides, counted in elements of
   the routine's format, float32 or complex64. A routine's own matrices
   are C-contiguous. An address word that is 0 names no matrix. A launch
   takes the units from [first], a group each, or for a kernel of a thread
   an element, the elements from [first], until [count]. */

#define NX_METAL_FACTOR_KERNELS(X)                                            \
  X(la_trsm_f32, NX_FAMILY_FACTOR_F32, NX_LA_TRSM_THREADS, NX_LA_TRSM_WORDS)  \
  X(la_trsm_c64, NX_FAMILY_FACTOR_C64, NX_LA_TRSM_THREADS, NX_LA_TRSM_WORDS)  \
  X(la_lu_panel_f32, NX_FAMILY_FACTOR_F32, NX_LA_THREADS, NX_LA_PANEL_WORDS)  \
  X(la_lu_panel_c64, NX_FAMILY_FACTOR_C64, NX_LA_THREADS, NX_LA_PANEL_WORDS)  \
  X(la_chol_f32, NX_FAMILY_FACTOR_F32, NX_LA_THREADS, NX_LA_CHOL_WORDS)       \
  X(la_chol_c64, NX_FAMILY_FACTOR_C64, NX_LA_THREADS, NX_LA_CHOL_WORDS)       \
  X(la_qr_panel_f32, NX_FAMILY_FACTOR_F32, NX_LA_THREADS, NX_LA_QR_WORDS)     \
  X(la_qr_panel_c64, NX_FAMILY_FACTOR_C64, NX_LA_THREADS, NX_LA_QR_WORDS)     \
  X(la_gather, NX_FAMILY_FACTOR, NX_LA_THREADS, NX_LA_GATHER_WORDS)           \
  X(la_chol_finish, NX_FAMILY_FACTOR, NX_LA_THREADS, NX_LA_FINISH_WORDS)      \
  X(la_diag_zero, NX_FAMILY_FACTOR, NX_LA_THREADS, NX_LA_DIAG_WORDS)          \
  X(la_nan_where, NX_FAMILY_FACTOR, NX_LA_THREADS, NX_LA_DIAG_WORDS)          \
  X(la_qr_out, NX_FAMILY_FACTOR, NX_LA_THREADS, NX_LA_QR_OUT_WORDS)           \
  X(la_dense_v, NX_FAMILY_FACTOR, NX_LA_THREADS, NX_LA_DENSE_WORDS)           \
  X(la_iota, NX_FAMILY_FACTOR, NX_LA_THREADS, NX_LA_IOTA_WORDS)

/* The matrix routines' products' kernels (contract.metal): the float32
   tiles that read a routine's options, plain and negated, and the
   flushing copy. */
#define NX_METAL_ROUTINE_KERNELS(X)                                           \
  NX_CONTRACT_TILE_ROWS_KN(X, f32_rt, NX_FAMILY_CONTRACT_F32_RT_KN)           \
  NX_CONTRACT_TILE_ROWS_NK(X, f32_rt, NX_FAMILY_CONTRACT_F32_RT_NK)           \
  NX_CONTRACT_TILE_ROWS_KN(X, f32_neg, NX_FAMILY_CONTRACT_F32_NEG_KN)         \
  NX_CONTRACT_TILE_ROWS_NK(X, f32_neg, NX_FAMILY_CONTRACT_F32_NEG_NK)         \
  X(contract_copy, NX_FAMILY_CONTRACT_COPY, NX_CCOPY_THREADS, NX_CCOPY_WORDS)

enum {
  /* A routine's formats: the [fmt] word of a kernel of either. */
  NX_LA_F32,
  NX_LA_C64,
};

enum {
  /* The threads of a group of every factorisation kernel but trsm. */
  NX_LA_THREADS = 256,
  /* trsm's threads, a column of x each. */
  NX_LA_TRSM_THREADS = 64,
  /* The most rows of trsm, the columns of an LU panel and the order of a
     Cholesky block: 64 at float32, 32 at complex64, so that their tiles
     fit 32 KiB of threadgroup memory. */
  NX_LA_LEAF = 64,
  /* The bytes of an LU meeting slot before its row: the candidate. */
  NX_LA_SLOT_HEAD = 32,
  /* QR's sums over rows: blocks of NX_LA_QBLK rows from row 0. */
  NX_LA_QBLK = 256,
  /* The blocks of a superblock, whose partials QR's sums combine by a
     tree before adding the superblocks in order. */
  NX_LA_QSUP = 32,
  /* qr_panel's stages with P > 1: a step's first and second launch, and
     the Gram pairs' launches. */
  NX_LA_QR_STEP_A = 1,
  NX_LA_QR_STEP_B = 2,
  NX_LA_QR_STEP_G = 3,
  /* qr_panel's sum stages: the totals that step k's second launch, step
     k's first launch and the Gram pairs' launch k read. */
  NX_LA_QR_SUM_B = 4,
  NX_LA_QR_SUM_A = 5,
  NX_LA_QR_SUM_G = 6,
  /* The 4-byte words of a row that a gather group moves. */
  NX_LA_GATHER_SPAN = 32,
};

/* la_trsm: x <- op(l)^-1 x for the lower triangular n x n view op(l), n
   at most the format's leaf, and the n x r view x of each of [matrices],
   a thread a column of x, the columns from [first] to [count] - 1 of the
   matrices' r each: op(l) conjugated under [conj], its diagonal taken as
   ones under [unit]. [src] (or 0) is b, read in place of x with x's
   strides; with [whole], a zero on the diagonal makes the column NaN.
   [rdiv] is r's divisor (device.h's divide), which the launch takes
   where its units are below 2^32, else under [wide] a 64-bit division. */
enum {
  NX_LA_TRSM_L,
  NX_LA_TRSM_X,
  NX_LA_TRSM_SRC,
  NX_LA_TRSM_SL,
  NX_LA_TRSM_LRS,
  NX_LA_TRSM_LCS,
  NX_LA_TRSM_SX,
  NX_LA_TRSM_XRS,
  NX_LA_TRSM_XCS,
  NX_LA_TRSM_N,
  NX_LA_TRSM_R,
  NX_LA_TRSM_RDIV,
  NX_LA_TRSM_WIDE,
  NX_LA_TRSM_COUNT,
  NX_LA_TRSM_CONJ,
  NX_LA_TRSM_UNIT,
  NX_LA_TRSM_WHOLE,
  NX_LA_TRSM_FIRST,
  NX_LA_TRSM_WORDS,
};

/* la_lu_panel: LU of the panel of columns [c0, c0 + w) of rows [c0, m)
   of each matrix of [a] (rows lda apart, matrices sbat apart), in place,
   pivots at [piv] (spiv a matrix). G threads a matrix (16, 32 or the
   group), [mats] matrices a group, and P groups a matrix, group q its
   rows [q per, (q + 1) per); with P > 1, one launch of min(m - c0, w) + 1
   a step, [k] the launch's, the groups passing candidates through
   [slots] (slots_stride bytes a matrix, the second step parity's from
   byte [slots_parity]). [src_of] (m - c0 a matrix) receives each
   position's row for gather; with [src], the operand read in place of
   a, [perm] (sperm a matrix) receives it instead. Where [lv_in] is not
   -1, step k's launch reads step k - 1's winner from the [n_in]
   candidates of [levels] from candidate lv_in (lv_stride candidates a
   matrix, NX_LA_SLOT_HEAD bytes each) instead of from the P slots. A
   launch of [level] L > 0 takes the best of each run of NX_LA_THREADS of
   its [n_in] candidates, step k's slots where L = 1 and [levels] from
   candidate lv_in otherwise, into [levels] from candidate lv_out, a unit
   a matrix's run. A matrix's tile is in
   threadgroup memory where [in_lds], rows [ld] elements apart, else at
   [spill], spill_bytes a matrix, columns [ld] elements apart. Threadgroup memory holds the candidates from byte 0, the
   second step parity's from byte [cands_parity], with P > 1 the pivot
   row from byte [row], and from byte [head] the matrices' tiles,
   spill_bytes apart; a tile's positions start [pos] bytes in. The
   launch takes the units, a group's matrices or one group of a
   matrix's P, from [first] to [count] - 1: [pdiv] is P's divisor and
   [runs_div] a level's runs' (device.h's divide), which the launch takes
   where its units are below 2^32, else under [wide] a 64-bit
   division. */
enum {
  NX_LA_PANEL_A,
  NX_LA_PANEL_PIV,
  NX_LA_PANEL_SRC_OF,
  NX_LA_PANEL_SRC,
  NX_LA_PANEL_PERM,
  NX_LA_PANEL_SPILL,
  NX_LA_PANEL_SLOTS,
  NX_LA_PANEL_LEVELS,
  NX_LA_PANEL_SBAT,
  NX_LA_PANEL_LDA,
  NX_LA_PANEL_SPIV,
  NX_LA_PANEL_SPERM,
  NX_LA_PANEL_M,
  NX_LA_PANEL_C0,
  NX_LA_PANEL_W,
  NX_LA_PANEL_MATRICES,
  NX_LA_PANEL_PER,
  NX_LA_PANEL_P,
  NX_LA_PANEL_G,
  NX_LA_PANEL_MATS,
  NX_LA_PANEL_LD,
  NX_LA_PANEL_IN_LDS,
  NX_LA_PANEL_SPILL_BYTES,
  NX_LA_PANEL_SLOT_BYTES,
  NX_LA_PANEL_SLOTS_STRIDE,
  NX_LA_PANEL_SLOTS_PARITY,
  NX_LA_PANEL_CANDS_PARITY,
  NX_LA_PANEL_ROW,
  NX_LA_PANEL_HEAD,
  NX_LA_PANEL_POS,
  NX_LA_PANEL_LEVEL,
  NX_LA_PANEL_LV_STRIDE,
  NX_LA_PANEL_LV_IN,
  NX_LA_PANEL_LV_OUT,
  NX_LA_PANEL_N_IN,
  NX_LA_PANEL_K,
  NX_LA_PANEL_PDIV,
  NX_LA_PANEL_RUNS_DIV,
  NX_LA_PANEL_WIDE,
  NX_LA_PANEL_FIRST,
  NX_LA_PANEL_COUNT,
  NX_LA_PANEL_WORDS,
};

/* la_chol: Cholesky of the diagonal block of order nb at (k, k) of each
   matrix, in place, element (i, j) at i lda + j cs, G threads a matrix
   and [mats] matrices a group, rows ld apart in threadgroup memory and
   matrices stride bytes apart; [fail], a 32-bit word a matrix, set where
   it fails; with [src], the whole matrix read from it and stored whole,
   its Lower factor or, under [upper], the Upper one. The launch takes
   the units of [mats] matrices from [first] to [count] - 1. */
enum {
  NX_LA_CHOL_A,
  NX_LA_CHOL_FAIL,
  NX_LA_CHOL_SRC,
  NX_LA_CHOL_SBAT,
  NX_LA_CHOL_LDA,
  NX_LA_CHOL_CS,
  NX_LA_CHOL_K,
  NX_LA_CHOL_NB,
  NX_LA_CHOL_MATRICES,
  NX_LA_CHOL_G,
  NX_LA_CHOL_MATS,
  NX_LA_CHOL_LD,
  NX_LA_CHOL_STRIDE,
  NX_LA_CHOL_UPPER,
  NX_LA_CHOL_FIRST,
  NX_LA_CHOL_COUNT,
  NX_LA_CHOL_WORDS,
};

/* la_qr_panel: Householder QR of the panel of columns [c0, c0 + w) of
   rows [c0, m) of each matrix of [a], in place (factor.metal states its
   sums' order), tau at [tau] (stau a matrix); dense V at [v] (sv a
   matrix) and T at [t] (st a matrix, rows ldto apart) where not 0.
   Where the panel is the whole matrix, [q] (m x qc) and [r] (qc x n)
   receive Q's first qc columns and R. G threads a matrix and [mats]
   matrices a group; with P > 1, launches of [stage] and [k]; [parts]
   holds a matrix's partials (parts_stride elements) where not in
   threadgroup memory, [spill] its tile where not there. A matrix's
   threadgroup memory, [stride] bytes from the group's byte 0 for each
   matrix before it, holds its sums' values from byte 0 and from bytes
   [tc], [ts], [gram] and [mx] a step's w_c, T, the Gram pairs and M, the
   partials from byte [parts_at] and the tile from byte [tile_at] where
   they are there; a tile there has its partials there too. The partials
   hold a sum's blocks, the second parity's from element [store], step
   j's row of R from element [drow] and, with P > 1, the pairs from
   element [gstore]. Where [summed], a meeting's sums are read from the
   totals at element [tot], which the launches of a sum stage, a unit a
   matrix, compute from superblocks [from, to) of 32 blocks. The launch
   takes the units from [first] to [count] - 1. */
enum {
  NX_LA_QR_A,
  NX_LA_QR_TAU,
  NX_LA_QR_V,
  NX_LA_QR_T,
  NX_LA_QR_Q,
  NX_LA_QR_R,
  NX_LA_QR_SPILL,
  NX_LA_QR_PARTS,
  NX_LA_QR_SBAT,
  NX_LA_QR_LDA,
  NX_LA_QR_STAU,
  NX_LA_QR_SV,
  NX_LA_QR_ST,
  NX_LA_QR_LDTO,
  NX_LA_QR_QC,
  NX_LA_QR_N,
  NX_LA_QR_M,
  NX_LA_QR_C0,
  NX_LA_QR_W,
  NX_LA_QR_MATRICES,
  NX_LA_QR_PER,
  NX_LA_QR_P,
  NX_LA_QR_G,
  NX_LA_QR_MATS,
  NX_LA_QR_LD,
  NX_LA_QR_IN_LDS,
  NX_LA_QR_PARTS_IN_LDS,
  NX_LA_QR_MOST,
  NX_LA_QR_SPILL_BYTES,
  NX_LA_QR_PARTS_STRIDE,
  NX_LA_QR_STRIDE,
  NX_LA_QR_TC,
  NX_LA_QR_TS,
  NX_LA_QR_GRAM,
  NX_LA_QR_MX,
  NX_LA_QR_PARTS_AT,
  NX_LA_QR_TILE_AT,
  NX_LA_QR_STORE,
  NX_LA_QR_DROW,
  NX_LA_QR_GSTORE,
  NX_LA_QR_TOT,
  NX_LA_QR_SUMMED,
  NX_LA_QR_FROM,
  NX_LA_QR_TO,
  NX_LA_QR_STAGE,
  NX_LA_QR_K,
  NX_LA_QR_FIRST,
  NX_LA_QR_COUNT,
  NX_LA_QR_WORDS,
};

/* la_gather: an LU panel's row interchanges on each matrix's columns
   outside it and on perm (sperm a matrix), elements of [fmt], the units
   of NX_LA_GATHER_SPAN words of each moved row from [first] to
   [count] - 1, [per] a matrix, [pdiv] per's divisor (device.h's
   divide), which the launch takes where its units are below 2^32, else
   under [wide] a 64-bit division. */
enum {
  NX_LA_GATHER_A,
  NX_LA_GATHER_PIV,
  NX_LA_GATHER_SRC_OF,
  NX_LA_GATHER_PERM,
  NX_LA_GATHER_SBAT,
  NX_LA_GATHER_LDA,
  NX_LA_GATHER_N,
  NX_LA_GATHER_SPIV,
  NX_LA_GATHER_SPERM,
  NX_LA_GATHER_M,
  NX_LA_GATHER_C0,
  NX_LA_GATHER_W,
  NX_LA_GATHER_STEPS,
  NX_LA_GATHER_FMT,
  NX_LA_GATHER_PER,
  NX_LA_GATHER_PDIV,
  NX_LA_GATHER_WIDE,
  NX_LA_GATHER_FIRST,
  NX_LA_GATHER_COUNT,
  NX_LA_GATHER_WORDS,
};

/* la_chol_finish: each n x n matrix NaN where [fail], else its other
   triangle +0 and, under [upper], the conjugate transpose of its lower
   one; under [tr], the matrix holds the factor's transpose. A thread an
   element, from [first] to [count] - 1. */
enum {
  NX_LA_FINISH_A,
  NX_LA_FINISH_FAIL,
  NX_LA_FINISH_N,
  NX_LA_FINISH_UPPER,
  NX_LA_FINISH_TR,
  NX_LA_FINISH_FMT,
  NX_LA_FINISH_FIRST,
  NX_LA_FINISH_COUNT,
  NX_LA_FINISH_WORDS,
};

/* la_diag_zero and la_nan_where: a triangular solve's diagonal, [sd]
   apart: [flag], a 32-bit word a matrix, set where a diagonal element of
   a is zero; then each of x's xn elements NaN where it is set. A thread
   a diagonal element, or an element of x, from [first] to [count] - 1. */
enum {
  NX_LA_DIAG_A,
  NX_LA_DIAG_X,
  NX_LA_DIAG_FLAG,
  NX_LA_DIAG_SA,
  NX_LA_DIAG_SD,
  NX_LA_DIAG_N,
  NX_LA_DIAG_SX,
  NX_LA_DIAG_XN,
  NX_LA_DIAG_FMT,
  NX_LA_DIAG_FIRST,
  NX_LA_DIAG_COUNT,
  NX_LA_DIAG_WORDS,
};

/* la_qr_out: q, the first qc columns of the m x m identity, and r, the
   upper triangle of f's first rows, rows past m zero; f 0 for a matrix
   of no column. A thread an element, from [first] to [count] - 1. */
enum {
  NX_LA_QR_OUT_F,
  NX_LA_QR_OUT_Q,
  NX_LA_QR_OUT_R,
  NX_LA_QR_OUT_M,
  NX_LA_QR_OUT_N,
  NX_LA_QR_OUT_QC,
  NX_LA_QR_OUT_FMT,
  NX_LA_QR_OUT_FIRST,
  NX_LA_QR_OUT_COUNT,
  NX_LA_QR_OUT_WORDS,
};

/* la_dense_v: the dense V of f's columns [c0, c0 + w), rows [c0, m). A
   thread an element, from [first] to [count] - 1. */
enum {
  NX_LA_DENSE_F,
  NX_LA_DENSE_V,
  NX_LA_DENSE_M,
  NX_LA_DENSE_N,
  NX_LA_DENSE_C0,
  NX_LA_DENSE_W,
  NX_LA_DENSE_FMT,
  NX_LA_DENSE_FIRST,
  NX_LA_DENSE_COUNT,
  NX_LA_DENSE_WORDS,
};

/* la_iota: perm[g] = g mod m for g from [first] to [count] - 1. */
enum {
  NX_LA_IOTA_PERM,
  NX_LA_IOTA_M,
  NX_LA_IOTA_FIRST,
  NX_LA_IOTA_COUNT,
  NX_LA_IOTA_WORDS,
};

/* Gathers and assemblies (index.metal) */

#define NX_METAL_INDEX_KERNELS(X)                                             \
  X(move, NX_FAMILY_INDEX, NX_MOVE_THREADS, NX_MOVE_WORDS)                    \
  X(move_fill, NX_FAMILY_INDEX, NX_MOVE_THREADS, NX_MOVE_WORDS)               \
  X(runs, NX_FAMILY_INDEX, NX_RUNS_THREADS, NX_RUNS_WORDS)                    \
  X(runs_carry, NX_FAMILY_INDEX, NX_RUNS_THREADS, NX_RUNS_WORDS)            \
  X(runs_positions, NX_FAMILY_INDEX, NX_RUNS_THREADS, NX_POSITIONS_WORDS)     \
  X(fold_0, NX_FAMILY_INDEX, NX_FOLD_THREADS, NX_FOLD_WORDS)                  \
  X(fold_1, NX_FAMILY_INDEX, NX_FOLD_THREADS, NX_FOLD_WORDS)                  \
  X(fold_2, NX_FAMILY_INDEX, NX_FOLD_THREADS, NX_FOLD_WORDS)                  \
  X(fold_3, NX_FAMILY_INDEX, NX_FOLD_THREADS, NX_FOLD_WORDS)                  \
  X(fold_n, NX_FAMILY_INDEX, NX_FOLD_THREADS, NX_FOLD_WORDS)                  \
  X(fold_w, NX_FAMILY_INDEX, NX_FOLD_THREADS, NX_FOLD_WORDS)

/* An axis of a move: its extent; the divisor of its extent, device.h's
   [divide]'s below 2^32 elements a launch and [nx_divide]'s past them;
   and the strides along it of the destination, the source and the
   positions. */
enum {
  NX_MOVE_AXIS_EXTENT,
  NX_MOVE_AXIS_DIVISOR,
  NX_MOVE_AXIS_SD,
  NX_MOVE_AXIS_SS,
  NX_MOVE_AXIS_SI,
  NX_MOVE_AXIS_WORDS,
};

/* The most axes a move's parameters hold. */
enum { NX_MOVE_HOLDS = 4 };

/* A move's modes: where a unit's bytes come from. A copy and a gather
   read the source, at its position along the moved axis the destination's
   own and [idx]'s; a fill reads none. */
enum { NX_MOVE_COPY, NX_MOVE_GATHER, NX_MOVE_FILL };

/* The fields of a move's [how]: the mode in its low 2 bits, then at these
   bits whether the launch's elements reach 2^32, the log2 of a unit's
   bits (0 and 2 for elements of 1 and 4 bits, 3 to 7 for units of 1 to
   16 bytes), the bit at which the destination's and the source's
   position 0 starts in its word, for units of 1 and 4 bits, whether a
   gather's positions are 32-bit (else 64-bit), and the rank. runs and
   runs_carry hold at the unit's bits whether their rows are the first
   level's, and at the source's bit the log2 of their chunk's entries;
   runs_positions holds at the positions' bit whether it writes them as
   32 bits. A fold holds its dtype's code in 5 bits at the dtype's bit. */
enum {
  NX_HOW_WIDE = 2,
  NX_HOW_UNIT = 3,
  NX_HOW_DST_BIT = 8,
  NX_HOW_SRC_BIT = 13,
  NX_HOW_CHUNK = 13,
  NX_HOW_NARROW = 18,
  NX_HOW_DTYPE = 18,
  NX_HOW_RANK = 24,
};

/* move: the units [first] to [count] - 1 of the destination, in C order
   of the move's axes, [rank] of them, each from the source: the unit at
   index (i_0, ...) lies at [dst] + Σ i_j sd_j units, and reads [src] at
   Σ i_j ss_j units, plus, for a gather, p [sa], p = [idx]'s element at
   Σ i_j si_j - [from], or writes zero bits where p lies outside [0, d),
   [idx]'s elements of 32 or 64 bits by [how]. A fill writes, in place of
   a source's unit, the bytes of the 16-byte pattern [d], [sa] (each
   little-endian) at the unit's offset from [dst] modulo 16: [dst] lies
   on an element, whose bits the pattern repeats from its first byte, or
   from bit 0 for elements of 1 and 4 bits. A unit
   is 1, 2, 4, 8 or 16 bytes, on which every address it reaches lies, or an
   element of 1 or 4 bits, which a store changes by device.h's
   [store_bits]; their addresses' words are [dst]'s and [src]'s. The axes
   are NX_MOVE_AXIS_WORDS words each from the outermost, from [held]
   where [rank] is at most NX_MOVE_HOLDS and from [axes] otherwise, a
   constant buffer. A move too long for one launch's time runs as
   launches of its pieces, each from its own [first]. move_fill takes
   the fills, move the copies and gathers. */
enum {
  NX_MOVE_DST,
  NX_MOVE_SRC,
  NX_MOVE_IDX,
  NX_MOVE_AXES,
  NX_MOVE_FIRST,
  NX_MOVE_COUNT,
  NX_MOVE_HOW,
  NX_MOVE_D,
  NX_MOVE_SA,
  NX_MOVE_FROM,
  NX_MOVE_HELD,
  NX_MOVE_WORDS = NX_MOVE_HELD + NX_MOVE_AXIS_WORDS * NX_MOVE_HOLDS,
};

enum { NX_MOVE_THREADS = 256 };

/* runs: of [count] - [first] groups from group [first], group g the
   chunk g mod [chunks] of 2^c entries of row g / [chunks], c [how]'s
   chunk, at most NX_RUNS_CHUNK, whose
   quotient takes [divisor], each row's running maximum, as signed 64-bit
   integers, of [n] entries: read from [src] [stride] apart, written into
   [dst], each chunk's from its first entry, and each chunk's last into
   [top], [chunks] a row. A row's places are a move's, of its [rank]
   axes: at [dst] + sd, [src] + ss; at the first level, a range gather's
   ends, each row's [dst] is the [n] runs' maxima after two words, the
   row's starts' and ends' places, ss and si.

   runs_carry: of the entries [first] to [count] - 1 of [n] a row, whose
   quotient takes [divisor], from chunk 1 of each row on, each entry of
   [dst] the maximum of itself and the entry of [top] of the chunk before
   its own: the running maxima of chunks made whole. A row of [dst]
   starts [stride] entries after the one before, two words in at the
   first level. */
enum {
  NX_RUNS_DST,
  NX_RUNS_SRC,
  NX_RUNS_TOP,
  NX_RUNS_AXES,
  NX_RUNS_FIRST,
  NX_RUNS_COUNT,
  NX_RUNS_HOW,
  NX_RUNS_N,
  NX_RUNS_CHUNKS,
  NX_RUNS_DIVISOR,
  NX_RUNS_STRIDE,
  NX_RUNS_HELD,
  NX_RUNS_WORDS = NX_RUNS_HELD + NX_MOVE_AXIS_WORDS * NX_MOVE_HOLDS,
};

/* A runs group: 256 threads, 16 consecutive entries each; its threadgroup
   memory holds its SIMD-groups' maxima, NX_RUNS_SHARED bytes. */
enum {
  NX_RUNS_THREADS = 256,
  NX_RUNS_EACH = 16,
  NX_RUNS_CHUNK = NX_RUNS_THREADS * NX_RUNS_EACH,
  NX_RUNS_SHARED = 8 * (NX_RUNS_THREADS / 32),
};

/* runs_positions: the positions [first] to [count] - 1 of a range
   gather's rows, [size] a row, whose quotient takes [divisor]: position
   p of row r is the source position of the first run j whose running
   maximum of ends (runs, [maxima], a row of [k] + 2 words: its starts'
   and ends' places, then its maxima) exceeds p, found by halving, its
   start plus p less the end of run j - 1, modulo 2^64; -1 past every
   end and where it lies outside [0, d). [ss] and [es] step [starts] and
   [ends] along the runs. Written into [dst] as int64, or as int32 by
   [how], position e of row r, e = r [size] + p, at e - [from]. */
enum {
  NX_POSITIONS_DST,
  NX_POSITIONS_MAXIMA,
  NX_POSITIONS_STARTS,
  NX_POSITIONS_ENDS,
  NX_POSITIONS_FIRST,
  NX_POSITIONS_COUNT,
  NX_POSITIONS_HOW,
  NX_POSITIONS_K,
  NX_POSITIONS_SIZE,
  NX_POSITIONS_DIVISOR,
  NX_POSITIONS_SS,
  NX_POSITIONS_ES,
  NX_POSITIONS_D,
  NX_POSITIONS_FROM,
  NX_POSITIONS_WORDS,
};

/* Folds

   fold_W: the elements [first] to [count] - 1 of [dst], C-contiguous of
   [rank] axes, each the sum by Prog.Add of the elements of [x] a Padded
   load reads from its index, in C order of their taps, those of tap
   numbers [t0] to [t1] - 1 alone: an element whose first tap lies before
   [t0] adds onto the sum [dst] holds, one whose first lies in the range
   starts from it, one whose first lies past [t1] is left, and one no tap
   reads is +0. The kernel's windows of more than one tap, W of them, or
   [windows] for fold_n and fold_w, are the odometer's. fold_0 to fold_n
   compute positions, taps and tap numbers in 32 bits, where each lies
   below 2^30; fold_w in 64.

   [axes] is a constant buffer: [rank] axis records from the outermost,
   then a copy of each odometer window's, in order. Element index i_a of
   axis a lies at padded position P = lo + i_a space. An axis outside the
   odometer, a window of one tap or no window (step 1 and [nwin] the
   padded extent), reads at window q = P / step where step divides P and
   0 <= q < nwin, at q sw elements of [x]; no element otherwise. An
   odometer window reads tap j at q = (P - j dil) / step under the same
   conditions, at q sw + j st; the taps that read one position lie
   [apart] = step / gcd (step, dil) apart, at windows [drop] = dil /
   gcd (step, dil) apart; [span] is the product of the
   later windows' taps, a tap's weight in its tap number. Where [direct],
   dil divides step, and the taps that read P, a multiple of dil, are
   those congruent to P / dil modulo [apart]. The divisors, of the
   extent, the step, the dilation and [apart], are
   device.h's [divide]'s, or [nx_divide]'s where [how] is wide: elements
   past 2^32, or a padded position or a step past 2^31. [dst]'s and
   [x]'s words are their position 0's, at [how]'s bits for elements of 1
   or 4 bits. Where [t0] > 0, [seek] is the word of [axes] from which
   [t0]'s digits lie, its tap along each odometer window in order, so
   that no thread divides [t0]. */
enum {
  NX_FOLD_AXIS_EXTENT,
  NX_FOLD_AXIS_DIVISOR,
  NX_FOLD_AXIS_LO,
  NX_FOLD_AXIS_SPACE,
  NX_FOLD_AXIS_TAPS,
  NX_FOLD_AXIS_DIL,
  NX_FOLD_AXIS_STEP,
  NX_FOLD_AXIS_STEP_DIVISOR,
  NX_FOLD_AXIS_DIL_DIVISOR,
  NX_FOLD_AXIS_APART,
  NX_FOLD_AXIS_APART_DIVISOR,
  NX_FOLD_AXIS_DIRECT,
  NX_FOLD_AXIS_DROP,
  NX_FOLD_AXIS_NWIN,
  NX_FOLD_AXIS_SW,
  NX_FOLD_AXIS_ST,
  NX_FOLD_AXIS_SPAN,
  NX_FOLD_AXIS_WORDS,
};

enum {
  NX_FOLD_DST,
  NX_FOLD_X,
  NX_FOLD_AXES,
  NX_FOLD_FIRST,
  NX_FOLD_COUNT,
  NX_FOLD_HOW,
  NX_FOLD_T0,
  NX_FOLD_T1,
  NX_FOLD_WINDOWS,
  NX_FOLD_SEEK,
  NX_FOLD_WORDS,
};

/* A fold's group, and the most odometer windows fold_n holds in thread
   memory: windows of more than one tap each double [x]'s elements, so 63
   hold any [x] that fits in memory. */
enum {
  NX_FOLD_THREADS = 256,
  NX_FOLD_MOST = 64,
};

/* Spectral routines (spectral.metal)

   The spectral routines' steps at float32 and complex64, the
   contraction's products between them. A step that computes in
   workspace finds matrix b's at its [ws] word's byte plus b [per]; one
   of parts alone reads an element as [parts] floats, 1 at float32 and 2
   at complex64. An address word that is 0 names no array. A launch
   takes its units, a group's or a thread's as the kernel states, from
   [first] to [count] − 1, and no bit follows where a launch's units
   start or end. */

#define NX_METAL_SPECTRAL_KERNELS(X)                                          \
  X(sp_eigh_jacobi_f32, NX_FAMILY_EIGH_F32, NX_SP_THREADS,                    \
    NX_SP_EIGH_JACOBI_WORDS)                                                  \
  X(sp_eigh_jacobi_c64, NX_FAMILY_EIGH_C64, NX_SP_THREADS,                    \
    NX_SP_EIGH_JACOBI_WORDS)                                                  \
  X(sp_svd_jacobi_f32, NX_FAMILY_SVD_F32, NX_SP_THREADS,                      \
    NX_SP_SVD_JACOBI_WORDS)                                                   \
  X(sp_svd_jacobi_c64, NX_FAMILY_SVD_C64, NX_SP_THREADS,                      \
    NX_SP_SVD_JACOBI_WORDS)                                                   \
  X(sp_td_stage, NX_FAMILY_EIGH, NX_SP_THREADS, NX_SP_TD_STAGE_WORDS)         \
  X(sp_td_step_f32, NX_FAMILY_EIGH_F32, NX_SP_THREADS, NX_SP_TD_STEP_WORDS)   \
  X(sp_td_step_c64, NX_FAMILY_EIGH_C64, NX_SP_THREADS, NX_SP_TD_STEP_WORDS)   \
  X(sp_td_hemv, NX_FAMILY_EIGH, NX_SP_THREADS, NX_SP_TD_HEMV_WORDS)           \
  X(sp_td_diag, NX_FAMILY_EIGH, NX_SP_THREADS, NX_SP_TD_DIAG_WORDS)           \
  X(sp_td_vcopy, NX_FAMILY_EIGH, NX_SP_THREADS, NX_SP_TD_VCOPY_WORDS)         \
  X(sp_td_zinit, NX_FAMILY_EIGH, NX_SP_THREADS, NX_SP_TD_ZINIT_WORDS)         \
  X(sp_td_out, NX_FAMILY_EIGH, NX_SP_THREADS, NX_SP_TD_OUT_WORDS)             \
  X(sp_dc_leaves, NX_FAMILY_EIGH, NX_SP_THREADS, NX_SP_DC_LEAVES_WORDS)       \
  X(sp_dc_scatter, NX_FAMILY_EIGH, NX_SP_THREADS, NX_SP_DC_SCATTER_WORDS)     \
  X(sp_dc_prepare, NX_FAMILY_EIGH, NX_SP_THREADS, NX_SP_MERGE_WORDS)          \
  X(sp_dc_deflate, NX_FAMILY_EIGH, NX_SP_THREADS, NX_SP_DC_DEFLATE_WORDS)     \
  X(sp_dc_rotate, NX_FAMILY_EIGH, NX_SP_THREADS, NX_SP_MERGE_WORDS)           \
  X(sp_dc_rank, NX_FAMILY_EIGH, NX_SP_THREADS, NX_SP_MERGE_WORDS)             \
  X(sp_dc_gather, NX_FAMILY_EIGH, NX_SP_THREADS, NX_SP_MERGE_WORDS)           \
  X(sp_dc_secular, NX_FAMILY_EIGH, NX_SP_HALF, NX_SP_DC_SECULAR_WORDS)        \
  X(sp_dc_product, NX_FAMILY_EIGH, NX_SP_THREADS, NX_SP_MERGE_WORDS)          \
  X(sp_dc_sort, NX_FAMILY_EIGH, NX_SP_THREADS, NX_SP_MERGE_WORDS)             \
  X(sp_dc_place, NX_FAMILY_EIGH, NX_SP_THREADS, NX_SP_MERGE_WORDS)            \
  X(sp_larft, NX_FAMILY_SPECTRAL, NX_SP_WAVE, NX_SP_LARFT_WORDS)              \
  X(sp_tri_bounds, NX_FAMILY_SPECTRAL, NX_SP_HALF, NX_SP_TRI_BOUNDS_WORDS)    \
  X(sp_tri_bisect, NX_FAMILY_SPECTRAL, NX_SP_HALF, NX_SP_TRI_BISECT_WORDS)    \
  X(sp_eye, NX_FAMILY_SPECTRAL, NX_SP_THREADS, NX_SP_EYE_WORDS)               \
  X(sp_bd_stage, NX_FAMILY_SVD, NX_SP_THREADS, NX_SP_BD_STAGE_WORDS)          \
  X(sp_bd_left_f32, NX_FAMILY_SVD_F32, NX_SP_THREADS, NX_SP_BD_COL_WORDS)     \
  X(sp_bd_left_c64, NX_FAMILY_SVD_C64, NX_SP_THREADS, NX_SP_BD_COL_WORDS)     \
  X(sp_bd_wh, NX_FAMILY_SVD, NX_SP_THREADS, NX_SP_BD_COL_WORDS)               \
  X(sp_bd_lsum, NX_FAMILY_SVD, NX_SP_THREADS, NX_SP_BD_CHUNK_WORDS)           \
  X(sp_bd_lfin, NX_FAMILY_SVD, NX_SP_THREADS, NX_SP_BD_CHUNK_WORDS)           \
  X(sp_bd_lscale, NX_FAMILY_SVD, NX_SP_THREADS, NX_SP_BD_CHUNK_WORDS)         \
  X(sp_bd_whc, NX_FAMILY_SVD, NX_SP_THREADS, NX_SP_BD_CHUNK_WORDS)            \
  X(sp_bd_whfin, NX_FAMILY_SVD, NX_SP_THREADS, NX_SP_BD_CHUNK_WORDS)          \
  X(sp_bd_updl, NX_FAMILY_SVD, NX_SP_THREADS, NX_SP_BD_COL_WORDS)             \
  X(sp_bd_right_f32, NX_FAMILY_SVD_F32, NX_SP_THREADS, NX_SP_BD_COL_WORDS)    \
  X(sp_bd_right_c64, NX_FAMILY_SVD_C64, NX_SP_THREADS, NX_SP_BD_COL_WORDS)    \
  X(sp_bd_wn, NX_FAMILY_SVD, NX_SP_HALF, NX_SP_BD_COL_WORDS)                  \
  X(sp_bd_updr, NX_FAMILY_SVD, NX_SP_THREADS, NX_SP_BD_COL_WORDS)             \
  X(sp_bd_tgk, NX_FAMILY_SVD, NX_SP_THREADS, NX_SP_BD_TGK_WORDS)              \
  X(sp_bd_values, NX_FAMILY_SVD, NX_SP_THREADS, NX_SP_BD_VALUES_WORDS)        \
  X(sp_bd_ginit, NX_FAMILY_SVD, NX_SP_THREADS, NX_SP_BD_WORDS)                \
  X(sp_bd_gen, NX_FAMILY_SVD, NX_SP_WAVE, NX_SP_BD_WORDS)                     \
  X(sp_bd_apply, NX_FAMILY_SVD, NX_SP_HALF, NX_SP_BD_WORDS)                   \
  X(sp_bd_xinit, NX_FAMILY_SVD, NX_SP_THREADS, NX_SP_BD_XINIT_WORDS)          \
  X(sp_bd_vcopy, NX_FAMILY_SVD, NX_SP_THREADS, NX_SP_BD_VCOPY_WORDS)          \
  X(sp_bd_rank, NX_FAMILY_SVD, NX_SP_THREADS, NX_SP_BD_WORDS)                 \
  X(sp_bd_out, NX_FAMILY_SVD, NX_SP_THREADS, NX_SP_BD_OUT_WORDS)              \
  X(sp_eig_in_f32, NX_FAMILY_EIG_F32, NX_SP_HALF, NX_SP_EIG_WORDS)            \
  X(sp_eig_in_c64, NX_FAMILY_EIG_C64, NX_SP_HALF, NX_SP_EIG_WORDS)            \
  X(sp_eig_count_f32, NX_FAMILY_EIG_F32, NX_SP_HALF, NX_SP_EIG_WORDS)         \
  X(sp_eig_count_c64, NX_FAMILY_EIG_C64, NX_SP_HALF, NX_SP_EIG_WORDS)         \
  X(sp_eig_colcnt_f32, NX_FAMILY_EIG_F32, NX_SP_HALF, NX_SP_EIG_WORDS)        \
  X(sp_eig_colcnt_c64, NX_FAMILY_EIG_C64, NX_SP_HALF, NX_SP_EIG_WORDS)        \
  X(sp_eig_iso_f32, NX_FAMILY_EIG_F32, NX_SP_HALF, NX_SP_EIG_WORDS)           \
  X(sp_eig_iso_c64, NX_FAMILY_EIG_C64, NX_SP_HALF, NX_SP_EIG_WORDS)           \
  X(sp_eig_rows_f32, NX_FAMILY_EIG_F32, NX_SP_HALF, NX_SP_EIG_WORDS)          \
  X(sp_eig_rows_c64, NX_FAMILY_EIG_C64, NX_SP_HALF, NX_SP_EIG_WORDS)          \
  X(sp_eig_big_f32, NX_FAMILY_EIG_F32, NX_SP_HALF, NX_SP_EIG_WORDS)           \
  X(sp_eig_big_c64, NX_FAMILY_EIG_C64, NX_SP_HALF, NX_SP_EIG_WORDS)           \
  X(sp_eig_scale_f32, NX_FAMILY_EIG_F32, NX_SP_HALF, NX_SP_EIG_WORDS)         \
  X(sp_eig_scale_c64, NX_FAMILY_EIG_C64, NX_SP_HALF, NX_SP_EIG_WORDS)         \
  X(sp_eig_hess_f32, NX_FAMILY_EIG_F32, NX_SP_HALF, NX_SP_EIG_WORDS)          \
  X(sp_eig_hess_c64, NX_FAMILY_EIG_C64, NX_SP_HALF, NX_SP_EIG_WORDS)          \
  X(sp_eig_orghr_f32, NX_FAMILY_EIG_F32, NX_SP_HALF, NX_SP_EIG_WORDS)         \
  X(sp_eig_orghr_c64, NX_FAMILY_EIG_C64, NX_SP_HALF, NX_SP_EIG_WORDS)         \
  X(sp_eig_qr_f32, NX_FAMILY_EIG_F32, NX_SP_HALF, NX_SP_EIG_WORDS)            \
  X(sp_eig_qr_c64, NX_FAMILY_EIG_C64, NX_SP_HALF, NX_SP_EIG_WORDS)            \
  X(sp_eig_w_f32, NX_FAMILY_EIG_F32, NX_SP_HALF, NX_SP_EIG_WORDS)             \
  X(sp_eig_w_c64, NX_FAMILY_EIG_C64, NX_SP_HALF, NX_SP_EIG_WORDS)             \
  X(sp_eig_norms_f32, NX_FAMILY_EIG_F32, NX_SP_HALF, NX_SP_EIG_WORDS)         \
  X(sp_eig_norms_c64, NX_FAMILY_EIG_C64, NX_SP_HALF, NX_SP_EIG_WORDS)         \
  X(sp_eig_vectors_f32, NX_FAMILY_EIG_F32, NX_SP_HALF, NX_SP_EIG_WORDS)       \
  X(sp_eig_vectors_c64, NX_FAMILY_EIG_C64, NX_SP_HALF, NX_SP_EIG_WORDS)       \
  X(sp_eig_finish, NX_FAMILY_EIG, NX_SP_HALF, NX_SP_EIG_WORDS)

/* Most steps' threads a group; Jacobi's orders: eigh of order at most
   NX_SP_JACOBI, svd of both extents at most it. */
enum {
  NX_SP_THREADS = 256,
  NX_SP_JACOBI = 32,
};

/* eigh's: a half group's threads and a SIMD-group's; the tridiagonal
   reduction's panel; divide and conquer's leaves. The threadgroup
   memory of a step that reduces over its group, a float a thread of
   NX_SP_THREADS; td_step's, NX_SP_SHARED then t1 and t2, NX_SP_TNB
   elements of at most 8 bytes each, from their bytes; dc_product's, two
   tiles of 16 x 65 floats; dc_leaves', a word. */
enum {
  NX_SP_HALF = 128,
  NX_SP_WAVE = 32,
  NX_SP_TNB = 32,
  NX_SP_LEAF = 32,
  NX_SP_SHARED = 1024,
  NX_SP_TD_T1 = 1024,
  NX_SP_TD_T2 = 1280,
  NX_SP_TD_LDS = 1536,
  NX_SP_DC_PRODUCT_LDS = 8320,
  NX_SP_DC_LDS = 16,
};

/* A secular root's state, in words. */
enum { NX_SP_SECULAR = 8 };

/* Bisection's: its most halvings; a matrix's bounds, in floats; a
   value's state, in words. */
enum {
  NX_SP_HALVINGS = 128,
  NX_SP_BOUNDS = 4,
  NX_SP_BISECT = 6,
};

/* svd's: the bidiagonal reflectors' blocks; the most threadgroup memory
   a group takes, in bytes. */
enum {
  NX_SP_BD_NB = 32,
  NX_SP_LDS_MOST = 32768,
};

/* sp_eigh_jacobi: w and, where v is not 0, v of the [nb] Hermitian n x n
   matrices at [a] (row-major, one after another), 1 <= n <= NX_SP_JACOBI,
   [per] matrices a group, NX_SP_THREADS / n² or 1 where that is 0,
   [sweeps] its limit (60 but in tests). A unit is a group's per
   matrices. Threadgroup memory: 16 + 20 per bytes rounded up to 16,
   then per matrices of (2 n² + 2 n) elements, then 2 per n words, one
   more where per n is odd. */
enum {
  NX_SP_EIGH_JACOBI_A,
  NX_SP_EIGH_JACOBI_W,
  NX_SP_EIGH_JACOBI_V,
  NX_SP_EIGH_JACOBI_NB,
  NX_SP_EIGH_JACOBI_N,
  NX_SP_EIGH_JACOBI_PER,
  NX_SP_EIGH_JACOBI_SWEEPS,
  NX_SP_EIGH_JACOBI_FIRST,
  NX_SP_EIGH_JACOBI_COUNT,
  NX_SP_EIGH_JACOBI_WORDS,
};

/* sp_td_stage: the Hermitian n x n matrices at [src] into the tridiagonal
   reduction's workspace, in phase [phase]: 0 a thread a matrix, 1 a
   thread a row, 2 a thread an element; [du] the divisor (divide) of the
   phase's units a matrix, n or n², and [dn] n's. */
enum {
  NX_SP_TD_STAGE_SRC,
  NX_SP_TD_STAGE_WS,
  NX_SP_TD_STAGE_N,
  NX_SP_TD_STAGE_PARTS,
  NX_SP_TD_STAGE_PER,
  NX_SP_TD_STAGE_PHASE,
  NX_SP_TD_STAGE_DU,
  NX_SP_TD_STAGE_DN,
  NX_SP_TD_STAGE_FIRST,
  NX_SP_TD_STAGE_COUNT,
  NX_SP_TD_STAGE_WORDS,
};

/* sp_td_step: step [i] of the panel of [nbk] columns from [j0], a group a
   matrix, NX_SP_TD_LDS bytes of threadgroup memory. */
enum {
  NX_SP_TD_STEP_WS,
  NX_SP_TD_STEP_N,
  NX_SP_TD_STEP_J0,
  NX_SP_TD_STEP_NBK,
  NX_SP_TD_STEP_I,
  NX_SP_TD_STEP_PER,
  NX_SP_TD_STEP_FIRST,
  NX_SP_TD_STEP_COUNT,
  NX_SP_TD_STEP_WORDS,
};

/* sp_td_hemv: W's column [i] from column [c], a SIMD-group a row: a unit
   is a matrix's NX_SP_THREADS / 32 rows, ⌈(n − c − 1) / 8⌉ units a
   matrix. */
enum {
  NX_SP_TD_HEMV_WS,
  NX_SP_TD_HEMV_N,
  NX_SP_TD_HEMV_C,
  NX_SP_TD_HEMV_I,
  NX_SP_TD_HEMV_PARTS,
  NX_SP_TD_HEMV_PER,
  NX_SP_TD_HEMV_FIRST,
  NX_SP_TD_HEMV_COUNT,
  NX_SP_TD_HEMV_WORDS,
};

/* sp_td_diag: d from A's diagonal, a thread an element of it. */
enum {
  NX_SP_TD_DIAG_WS,
  NX_SP_TD_DIAG_N,
  NX_SP_TD_DIAG_PARTS,
  NX_SP_TD_DIAG_PER,
  NX_SP_TD_DIAG_FIRST,
  NX_SP_TD_DIAG_COUNT,
  NX_SP_TD_DIAG_WORDS,
};

/* sp_td_vcopy: the reflectors of the panel of [nbk] columns from [j0]
   into Vc, a thread an element of it. */
enum {
  NX_SP_TD_VCOPY_WS,
  NX_SP_TD_VCOPY_N,
  NX_SP_TD_VCOPY_J0,
  NX_SP_TD_VCOPY_NBK,
  NX_SP_TD_VCOPY_PARTS,
  NX_SP_TD_VCOPY_PER,
  NX_SP_TD_VCOPY_FIRST,
  NX_SP_TD_VCOPY_COUNT,
  NX_SP_TD_VCOPY_WORDS,
};

/* sp_td_zinit: Z from the real n x n matrices at [q], [qs] floats apart,
   a thread an element of Z. */
enum {
  NX_SP_TD_ZINIT_Q,
  NX_SP_TD_ZINIT_Z,
  NX_SP_TD_ZINIT_QS,
  NX_SP_TD_ZINIT_N,
  NX_SP_TD_ZINIT_PARTS,
  NX_SP_TD_ZINIT_FIRST,
  NX_SP_TD_ZINIT_COUNT,
  NX_SP_TD_ZINIT_WORDS,
};

/* sp_td_out: w from d ([ds] floats a matrix) and, where v is not 0, v
   from Z; [dcx] (or 0), the solver's flag and exponent, [es] words a
   matrix. A thread an element of v. */
enum {
  NX_SP_TD_OUT_WS,
  NX_SP_TD_OUT_D,
  NX_SP_TD_OUT_DCX,
  NX_SP_TD_OUT_Z,
  NX_SP_TD_OUT_W,
  NX_SP_TD_OUT_V,
  NX_SP_TD_OUT_DS,
  NX_SP_TD_OUT_ES,
  NX_SP_TD_OUT_N,
  NX_SP_TD_OUT_PARTS,
  NX_SP_TD_OUT_PER,
  NX_SP_TD_OUT_FIRST,
  NX_SP_TD_OUT_COUNT,
  NX_SP_TD_OUT_WORDS,
};

/* sp_dc_leaves: T's diagonal [d] and subdiagonal [e], [stride] floats a
   matrix, scaled into divide and conquer's workspace and torn into
   leaves at [full] and [tail]. A group a matrix, NX_SP_DC_LDS bytes of
   threadgroup memory. */
enum {
  NX_SP_DC_LEAVES_WS,
  NX_SP_DC_LEAVES_D,
  NX_SP_DC_LEAVES_E,
  NX_SP_DC_LEAVES_FULL,
  NX_SP_DC_LEAVES_TAIL,
  NX_SP_DC_LEAVES_STRIDE,
  NX_SP_DC_LEAVES_N,
  NX_SP_DC_LEAVES_PER,
  NX_SP_DC_LEAVES_FIRST,
  NX_SP_DC_LEAVES_COUNT,
  NX_SP_DC_LEAVES_WORDS,
};

/* sp_dc_scatter: the leaves' values [wf], [wt] and vectors [vf], [vt]
   into the workspace, a thread an element of Q. */
enum {
  NX_SP_DC_SCATTER_WS,
  NX_SP_DC_SCATTER_WF,
  NX_SP_DC_SCATTER_VF,
  NX_SP_DC_SCATTER_WT,
  NX_SP_DC_SCATTER_VT,
  NX_SP_DC_SCATTER_N,
  NX_SP_DC_SCATTER_PER,
  NX_SP_DC_SCATTER_FIRST,
  NX_SP_DC_SCATTER_COUNT,
  NX_SP_DC_SCATTER_WORDS,
};

/* sp_dc_prepare, sp_dc_rotate, sp_dc_rank, sp_dc_gather, sp_dc_product,
   sp_dc_sort, sp_dc_place: a phase of the merges of level [lev], over
   units from [first] to [count] − 1. prepare and sort take a thread a
   merge (matrix, ⌈n / (NX_SP_LEAF 2^lev)⌉ a matrix); rotate a thread a
   row and rank a thread an index (matrix, n a matrix); gather and place
   a thread an element of a merge (matrix, n min(NX_SP_LEAF 2^lev, n) a
   matrix); product the merges along
   the grid's z, tiles of 64 rows along x and 64 roots along y from tile
   [tx] and [ty], NX_SP_DC_PRODUCT_LDS bytes of threadgroup memory. The
   others take [tx] and [ty] as 0. prepare, sort, gather and place
   divide by [du], the divisor (divide) of their units a matrix, and
   gather and place also by [dr], min(NX_SP_LEAF 2^lev, n)'s. */
enum {
  NX_SP_MERGE_WS,
  NX_SP_MERGE_N,
  NX_SP_MERGE_LEV,
  NX_SP_MERGE_PER,
  NX_SP_MERGE_TX,
  NX_SP_MERGE_TY,
  NX_SP_MERGE_DU,
  NX_SP_MERGE_DR,
  NX_SP_MERGE_FIRST,
  NX_SP_MERGE_COUNT,
  NX_SP_MERGE_WORDS,
};

/* sp_dc_deflate: the decisions of deflation of the merges of level
   [lev], a thread a merge (matrix, ⌈n / (NX_SP_LEAF 2^lev)⌉ a matrix),
   over units from [first] to [count] − 1, [du] the divisor (divide) of
   a matrix's merges. A launch runs at most [q] steps of each, a value a
   step, and saves its state. */
enum {
  NX_SP_DC_DEFLATE_WS,
  NX_SP_DC_DEFLATE_N,
  NX_SP_DC_DEFLATE_LEV,
  NX_SP_DC_DEFLATE_PER,
  NX_SP_DC_DEFLATE_Q,
  NX_SP_DC_DEFLATE_DU,
  NX_SP_DC_DEFLATE_FIRST,
  NX_SP_DC_DEFLATE_COUNT,
  NX_SP_DC_DEFLATE_WORDS,
};

/* sp_dc_secular: phase [phase] (0 roots, 1 weights, 2 vectors) of the
   merges of level [lev], a thread an index of a matrix; [steps] the
   roots' limit (30 but in tests). A launch of the roots runs at most [q]
   steps of each, its start the first where [start] is 1, and saves its
   state, NX_SP_SECULAR words a root. */
enum {
  NX_SP_DC_SECULAR_WS,
  NX_SP_DC_SECULAR_N,
  NX_SP_DC_SECULAR_LEV,
  NX_SP_DC_SECULAR_PHASE,
  NX_SP_DC_SECULAR_STEPS,
  NX_SP_DC_SECULAR_PER,
  NX_SP_DC_SECULAR_Q,
  NX_SP_DC_SECULAR_START,
  NX_SP_DC_SECULAR_FIRST,
  NX_SP_DC_SECULAR_COUNT,
  NX_SP_DC_SECULAR_WORDS,
};

/* sp_larft: T of the block reflector of [nbk] columns from G and τ, a
   group a matrix, the matrices' T, G and τ [per] bytes apart. */
enum {
  NX_SP_LARFT_TM,
  NX_SP_LARFT_G,
  NX_SP_LARFT_TAU,
  NX_SP_LARFT_NBK,
  NX_SP_LARFT_PARTS,
  NX_SP_LARFT_PER,
  NX_SP_LARFT_FIRST,
  NX_SP_LARFT_COUNT,
  NX_SP_LARFT_WORDS,
};

/* sp_tri_bounds: the bounds bisection starts from of each tridiagonal of
   order [n], diagonal [d] and subdiagonal [e], [stride] floats a matrix,
   into [bd], NX_SP_BOUNDS floats a matrix; a thread a matrix. */
enum {
  NX_SP_TRI_BOUNDS_D,
  NX_SP_TRI_BOUNDS_E,
  NX_SP_TRI_BOUNDS_BD,
  NX_SP_TRI_BOUNDS_STRIDE,
  NX_SP_TRI_BOUNDS_N,
  NX_SP_TRI_BOUNDS_FIRST,
  NX_SP_TRI_BOUNDS_COUNT,
  NX_SP_TRI_BOUNDS_WORDS,
};

/* sp_tri_bisect: eigenvalues [i0, i0 + cnt) of the tridiagonals of
   sp_tri_bounds from its bounds [bd] into [w], cnt a matrix, a thread
   each: at most [q] steps of each, its state at [st], NX_SP_BISECT words
   a value, which it starts from the bounds where [start] is 1. */
enum {
  NX_SP_TRI_BISECT_D,
  NX_SP_TRI_BISECT_E,
  NX_SP_TRI_BISECT_BD,
  NX_SP_TRI_BISECT_ST,
  NX_SP_TRI_BISECT_W,
  NX_SP_TRI_BISECT_STRIDE,
  NX_SP_TRI_BISECT_N,
  NX_SP_TRI_BISECT_I0,
  NX_SP_TRI_BISECT_CNT,
  NX_SP_TRI_BISECT_Q,
  NX_SP_TRI_BISECT_START,
  NX_SP_TRI_BISECT_FIRST,
  NX_SP_TRI_BISECT_COUNT,
  NX_SP_TRI_BISECT_WORDS,
};

/* sp_eye: [nz] matrices at [x], each the r x c identity's first rows and
   columns, in [parts] floats an element; a thread an element. */
enum {
  NX_SP_EYE_X,
  NX_SP_EYE_NZ,
  NX_SP_EYE_R,
  NX_SP_EYE_C,
  NX_SP_EYE_PARTS,
  NX_SP_EYE_WORDS,
};

/* sp_svd_jacobi: s and, where u and vh are not 0, u's first [uc] columns
   and vh's first [vr] rows of the [nb] m x n matrices at [a] (row-major),
   both extents at most NX_SP_JACOBI, [per] matrices a group, [sweeps]
   the limit (60 but in tests). A unit is a group's per matrices. With M
   and K the larger and smaller extents, [span] is a matrix's elements of
   threadgroup memory, M K + K² + M² + 2 K + M. Threadgroup memory: 16 +
   20 per bytes rounded up to 16, then per spans, then 2 per K words,
   then per (K + M) floats. */
enum {
  NX_SP_SVD_JACOBI_A,
  NX_SP_SVD_JACOBI_S,
  NX_SP_SVD_JACOBI_U,
  NX_SP_SVD_JACOBI_VH,
  NX_SP_SVD_JACOBI_NB,
  NX_SP_SVD_JACOBI_M,
  NX_SP_SVD_JACOBI_N,
  NX_SP_SVD_JACOBI_UC,
  NX_SP_SVD_JACOBI_VR,
  NX_SP_SVD_JACOBI_PER,
  NX_SP_SVD_JACOBI_SPAN,
  NX_SP_SVD_JACOBI_SWEEPS,
  NX_SP_SVD_JACOBI_FIRST,
  NX_SP_SVD_JACOBI_COUNT,
  NX_SP_SVD_JACOBI_WORDS,
};

/* A large svd's workspace layout, words of each bd step's parameters:
   the bytes of a matrix's workspace, the byte offsets of its arrays,
   [cap], the rotations a pass records, and [pt], the partials of a
   column past a chunk. */
enum {
  NX_SP_BDL_PER,
  NX_SP_BDL_G,
  NX_SP_BDL_X,
  NX_SP_BDL_Y,
  NX_SP_BDL_VC,
  NX_SP_BDL_WV,
  NX_SP_BDL_TAUQ,
  NX_SP_BDL_TAUP,
  NX_SP_BDL_D,
  NX_SP_BDL_E,
  NX_SP_BDL_UB,
  NX_SP_BDL_VB,
  NX_SP_BDL_REC,
  NX_SP_BDL_SC,
  NX_SP_BDL_SW,
  NX_SP_BDL_ST,
  NX_SP_BDL_CAP,
  NX_SP_BDL_PT,
  NX_SP_BDL_WORDS,
};

/* A large svd's state words a matrix, at the layout's [st]. */
enum {
  NX_SP_ST_M,
  NX_SP_ST_ITER,
  NX_SP_ST_SWEEPS,
  NX_SP_ST_OLDLL,
  NX_SP_ST_OLDM,
  NX_SP_ST_DONE,
  NX_SP_ST_FAIL,
  NX_SP_ST_OK,
  NX_SP_ST_EX,
  NX_SP_ST_NSW,
  NX_SP_ST_NREC,
  NX_SP_ST_BAD,
  NX_SP_ST_BIG,
  NX_SP_ST_SCALED,
  NX_SP_ST_KNT,
  NX_SP_ST_SCRE,
  NX_SP_ST_SCIM,
  NX_SP_ST_XBIG,
  NX_SP_ST_WORDS,
};

/* The floats of a staged matrix one thread of sp_bd_stage's scan
   reads. */
enum { NX_SP_SCAN_RUN = 256 };

/* sp_bd_stage: g of the m x n matrices at [a], M x K, in phase [phase]:
   0 a thread a matrix, 1 a thread a run of NX_SP_SCAN_RUN floats, 2 a
   thread an element; [du] the divisor (divide) of the phase's units a
   matrix, and [dm] M's. */
enum {
  NX_SP_BD_STAGE_A,
  NX_SP_BD_STAGE_WS,
  NX_SP_BD_STAGE_M,
  NX_SP_BD_STAGE_K,
  NX_SP_BD_STAGE_ROWS,
  NX_SP_BD_STAGE_COLS,
  NX_SP_BD_STAGE_PARTS,
  NX_SP_BD_STAGE_PHASE,
  NX_SP_BD_STAGE_DU,
  NX_SP_BD_STAGE_DM,
  NX_SP_BD_STAGE_LAYOUT,
  NX_SP_BD_STAGE_FIRST = NX_SP_BD_STAGE_LAYOUT + NX_SP_BDL_WORDS,
  NX_SP_BD_STAGE_COUNT,
  NX_SP_BD_STAGE_WORDS,
};

/* sp_bd_left, sp_bd_wh, sp_bd_updl, sp_bd_right, sp_bd_wn, sp_bd_updr:
   the reduction's steps at column or row [c]; a step of parts alone
   reads [parts]. left, right and wh take a group a matrix and
   NX_SP_SHARED bytes of threadgroup memory; wh a group a column. */
enum {
  NX_SP_BD_COL_WS,
  NX_SP_BD_COL_M,
  NX_SP_BD_COL_K,
  NX_SP_BD_COL_C,
  NX_SP_BD_COL_PARTS,
  NX_SP_BD_COL_LAYOUT,
  NX_SP_BD_COL_FIRST = NX_SP_BD_COL_LAYOUT + NX_SP_BDL_WORDS,
  NX_SP_BD_COL_COUNT,
  NX_SP_BD_COL_WORDS,
};

/* The rows of a chunk: a column's sums past it are taken in chunks of
   it, combined in chunk order. */
enum { NX_SP_BD_CHUNK = 65536 };

/* sp_bd_lsum, sp_bd_lfin, sp_bd_lscale, sp_bd_whc, sp_bd_whfin: the
   steps of column [c] past a chunk, in [phase] (lsum and lfin); [chunks]
   the column's chunks, [dchunks] their divisor (divide), [dcols] the
   divisor of the K − c − 1 columns past c. lsum, lscale and whc take a
   group a chunk, lfin a group a matrix, whfin a group a column, each
   NX_SP_SHARED bytes of threadgroup memory. */
enum {
  NX_SP_BD_CHUNK_WS,
  NX_SP_BD_CHUNK_M,
  NX_SP_BD_CHUNK_K,
  NX_SP_BD_CHUNK_C,
  NX_SP_BD_CHUNK_PARTS,
  NX_SP_BD_CHUNK_PHASE,
  NX_SP_BD_CHUNK_CHUNKS,
  NX_SP_BD_CHUNK_DCHUNKS,
  NX_SP_BD_CHUNK_DCOLS,
  NX_SP_BD_CHUNK_LAYOUT,
  NX_SP_BD_CHUNK_FIRST = NX_SP_BD_CHUNK_LAYOUT + NX_SP_BDL_WORDS,
  NX_SP_BD_CHUNK_COUNT,
  NX_SP_BD_CHUNK_WORDS,
};

/* sp_bd_tgk: B's Golub-Kahan tridiagonal into [dt] and [et], 2K floats a
   matrix each. */
enum {
  NX_SP_BD_TGK_WS,
  NX_SP_BD_TGK_DT,
  NX_SP_BD_TGK_ET,
  NX_SP_BD_TGK_K,
  NX_SP_BD_TGK_LAYOUT,
  NX_SP_BD_TGK_FIRST = NX_SP_BD_TGK_LAYOUT + NX_SP_BDL_WORDS,
  NX_SP_BD_TGK_COUNT,
  NX_SP_BD_TGK_WORDS,
};

/* sp_bd_values: s from the tridiagonal's eigenvalues [lam], K a
   matrix. */
enum {
  NX_SP_BD_VALUES_WS,
  NX_SP_BD_VALUES_LAM,
  NX_SP_BD_VALUES_S,
  NX_SP_BD_VALUES_K,
  NX_SP_BD_VALUES_LAYOUT,
  NX_SP_BD_VALUES_FIRST = NX_SP_BD_VALUES_LAYOUT + NX_SP_BDL_WORDS,
  NX_SP_BD_VALUES_COUNT,
  NX_SP_BD_VALUES_WORDS,
};

/* sp_bd_ginit, sp_bd_gen, sp_bd_apply, sp_bd_rank: the QR iteration of B,
   K x K. gen's iteration fails past [rotations] K steps (xBDSQR's
   maxit), and a pass records at most the layout's [cap] rotations. gen
   takes a group a matrix, its first thread working, and B's d and e in
   2 K floats of threadgroup memory where they fit NX_SP_LDS_MOST bytes,
   else none. */
enum {
  NX_SP_BD_WS,
  NX_SP_BD_K,
  NX_SP_BD_ROTATIONS,
  NX_SP_BD_LAYOUT,
  NX_SP_BD_FIRST = NX_SP_BD_LAYOUT + NX_SP_BDL_WORDS,
  NX_SP_BD_COUNT,
  NX_SP_BD_WORDS,
};

/* sp_bd_xinit: X, M x [gc], and Y from U_B and V_B. */
enum {
  NX_SP_BD_XINIT_WS,
  NX_SP_BD_XINIT_M,
  NX_SP_BD_XINIT_K,
  NX_SP_BD_XINIT_GC,
  NX_SP_BD_XINIT_PARTS,
  NX_SP_BD_XINIT_LAYOUT,
  NX_SP_BD_XINIT_FIRST = NX_SP_BD_XINIT_LAYOUT + NX_SP_BDL_WORDS,
  NX_SP_BD_XINIT_COUNT,
  NX_SP_BD_XINIT_WORDS,
};

/* sp_bd_vcopy: the reflectors of the panel of [nbk] from [j0] into Vc,
   of Q where [side] is 0, of P where 1. */
enum {
  NX_SP_BD_VCOPY_WS,
  NX_SP_BD_VCOPY_M,
  NX_SP_BD_VCOPY_K,
  NX_SP_BD_VCOPY_J0,
  NX_SP_BD_VCOPY_NBK,
  NX_SP_BD_VCOPY_SIDE,
  NX_SP_BD_VCOPY_PARTS,
  NX_SP_BD_VCOPY_LAYOUT,
  NX_SP_BD_VCOPY_FIRST = NX_SP_BD_VCOPY_LAYOUT + NX_SP_BDL_WORDS,
  NX_SP_BD_VCOPY_COUNT,
  NX_SP_BD_VCOPY_WORDS,
};

/* sp_bd_out: s, and where u and vh are not 0, u's first [uc] columns and
   vh's first [vr] rows of the m x n matrices. */
enum {
  NX_SP_BD_OUT_WS,
  NX_SP_BD_OUT_S,
  NX_SP_BD_OUT_U,
  NX_SP_BD_OUT_VH,
  NX_SP_BD_OUT_M,
  NX_SP_BD_OUT_K,
  NX_SP_BD_OUT_ROWS,
  NX_SP_BD_OUT_COLS,
  NX_SP_BD_OUT_UC,
  NX_SP_BD_OUT_VR,
  NX_SP_BD_OUT_PARTS,
  NX_SP_BD_OUT_LAYOUT,
  NX_SP_BD_OUT_FIRST = NX_SP_BD_OUT_LAYOUT + NX_SP_BDL_WORDS,
  NX_SP_BD_OUT_COUNT,
  NX_SP_BD_OUT_WORDS,
};

/* eig's steps: [a] the n x n matrices (row-major), [w] the eigenvalues
   (complex), [zx] Z X (row-major, real at float32), [v] the vectors
   (complex), [ws] the workspace, [per] bytes a matrix. [vectors] is 1
   where Z is formed. Balancing: iso at most [q] isolations, of the
   columns where [from] is 1, of the rows where 0; rows at most [q] rows
   of at most [passes] passes. hess and orghr take columns [from, to),
   orghr's ranges launched from the highest; qr at most [q] sweeps,
   [itq] max(10, h) in all. finish reads [parts], 1 at float32 and 2 at
   complex64. A unit is a matrix, a group a matrix; for in, big and
   scale a row or column b n + j of matrix b, a group each; for count,
   colcnt, norms and vectors, a thread each. rows takes 2 NX_SP_HALF
   words of threadgroup memory; in, big, scale, iso and hess
   NX_SP_HALF. */
enum {
  NX_SP_EIG_A,
  NX_SP_EIG_W,
  NX_SP_EIG_ZX,
  NX_SP_EIG_V,
  NX_SP_EIG_WS,
  NX_SP_EIG_N,
  NX_SP_EIG_VECTORS,
  NX_SP_EIG_FROM,
  NX_SP_EIG_TO,
  NX_SP_EIG_ITQ,
  NX_SP_EIG_Q,
  NX_SP_EIG_PARTS,
  NX_SP_EIG_PER,
  NX_SP_EIG_PASSES,
  NX_SP_EIG_FIRST,
  NX_SP_EIG_COUNT,
  NX_SP_EIG_WORDS,
};

/* eig's iteration state, words of a matrix's workspace: its phase, the
   active block's last row [i] and first [l], its sweeps and the sweeps
   since its last deflation; then balancing's: its phase, the rows
   [k, l] it has left, its scaling's row [i], pass and whether the pass
   rescaled a row. */
enum {
  NX_SP_EIG_PHASE,
  NX_SP_EIG_I,
  NX_SP_EIG_L,
  NX_SP_EIG_ITS,
  NX_SP_EIG_KDEFL,
  NX_SP_EIG_BAL,
  NX_SP_EIG_BK,
  NX_SP_EIG_BL,
  NX_SP_EIG_BI,
  NX_SP_EIG_BPASS,
  NX_SP_EIG_BNOCONV,
  NX_SP_EIG_STATE,
};

/* Balancing's phases: the NaN test, the rows' isolation, waiting for the
   columns' counts, the columns' isolation, the scaling, done. */
enum {
  NX_SP_BAL_START,
  NX_SP_BAL_ROWS,
  NX_SP_BAL_WAIT,
  NX_SP_BAL_COLS,
  NX_SP_BAL_SCALE,
  NX_SP_BAL_DONE,
};

/* eig's phases. */
enum {
  NX_SP_EIG_BEGIN,
  NX_SP_EIG_RUN,
  NX_SP_EIG_DONE,
  NX_SP_EIG_FAILED,
};

/* Attention (attention.metal)

   attention_tile_<dtype>_<d>_<c> runs a unit of a key head's rows, at
   most NX_ATT_ROWS, over its keys in blocks of NX_ATT_BLOCK, as four
   SIMD-groups of 8 rows: dk of at most d = 64 or 128 held whole, or (d =
   w) any dk in chunks of NX_ATT_CHUNK, and a piece of at most c columns
   of the values, NX_ATT_PIECE at most: 32, 64 or 128. Grid x counts units
   from [first_unit], grid y the pieces. */
#define NX_ATT_TILE_ROW(X, code, d, c)                                         \
  X(attention_tile_##code##_##d##_##c, NX_FAMILY_ATTENTION, NX_ATT_THREADS,    \
    NX_ATT_WORDS)
#define NX_ATT_TILES_D(X, code, d)                                             \
  NX_ATT_TILE_ROW(X, code, d, 32)                                              \
  NX_ATT_TILE_ROW(X, code, d, 64) NX_ATT_TILE_ROW(X, code, d, 128)
#define NX_ATT_TILES(X, code)                                                  \
  NX_ATT_TILES_D(X, code, 64)                                                  \
  NX_ATT_TILES_D(X, code, 128) NX_ATT_TILES_D(X, code, w)
#define NX_METAL_ATTENTION_KERNELS(X)                                          \
  NX_ATT_TILES(X, f32) NX_ATT_TILES(X, f16) NX_ATT_TILES(X, bf16)

/* A group's threadgroup memory, for elements of [sz] bytes and [cw] dims
   of q and k staged: each row's visible keys of the block (16 bits a
   quarter), each SIMD-group's flag of a value not finite, each key's
   presence, each row's, each key's k and v row offsets, each row's q
   offset; then q's rows of [cw] dims and the staging of [sk] keys of k
   (transposed, a row a dim) or of a block's keys of v, NX_ATT_VCOLS
   columns at a time, rows padded by 16 bytes. [sk] is 16 at float32,
   which holds a group of 64 dims to 15,472 bytes and a core to four
   groups, else 32. */
#define NX_ATT_STAGE_KEYS(sz) ((sz) == 4 ? 16 : 32)
#define NX_ATT_VCOLS 16
#define NX_ATT_LDQ(sz, cw) ((cw) + 16 / (sz))
#define NX_ATT_LDK(sz, sk) ((sk) + 16 / (sz))
#define NX_ATT_LDV(sz, c) ((c) + 16 / (sz))
#define NX_ATT_AT_STAGE(sz, cw)                                                \
  (NX_ATT_AT_Q + NX_ATT_ROWS * NX_ATT_LDQ(sz, cw) * (sz))
#define NX_ATT_STAGE_ELEMS(sz, cw)                                             \
  ((cw) * NX_ATT_LDK(sz, NX_ATT_STAGE_KEYS(sz)) >                              \
           NX_ATT_BLOCK * NX_ATT_LDV(sz, NX_ATT_VCOLS)                         \
       ? (cw) * NX_ATT_LDK(sz, NX_ATT_STAGE_KEYS(sz))                          \
       : NX_ATT_BLOCK * NX_ATT_LDV(sz, NX_ATT_VCOLS))
#define NX_ATT_SHARED_OF(sz, cw)                                               \
  (NX_ATT_AT_STAGE(sz, cw) + NX_ATT_STAGE_ELEMS(sz, cw) * (sz))

enum {
  NX_ATT_THREADS = 128,
  NX_ATT_ROWS = 32,
  NX_ATT_BLOCK = 64,
  NX_ATT_PIECE = 128,
  NX_ATT_CHUNK = 128,
  NX_ATT_AT_VIS = 0,
  NX_ATT_AT_BAD = NX_ATT_AT_VIS + NX_ATT_ROWS * 8,
  NX_ATT_AT_SEEN = NX_ATT_AT_BAD + 16,
  NX_ATT_AT_LIVE = NX_ATT_AT_SEEN + NX_ATT_BLOCK,
  NX_ATT_AT_KENT = NX_ATT_AT_LIVE + NX_ATT_ROWS,
  NX_ATT_AT_VENT = NX_ATT_AT_KENT + NX_ATT_BLOCK * 8,
  NX_ATT_AT_QROW = NX_ATT_AT_VENT + NX_ATT_BLOCK * 8,
  NX_ATT_AT_Q = NX_ATT_AT_QROW + NX_ATT_ROWS * 8,
  /* Each kernel's bytes, by element bytes and dims held. */
  NX_ATT_SHARED_4_64 = NX_ATT_SHARED_OF(4, 64),
  NX_ATT_SHARED_4_128 = NX_ATT_SHARED_OF(4, 128),
  NX_ATT_SHARED_2_64 = NX_ATT_SHARED_OF(2, 64),
  NX_ATT_SHARED_2_128 = NX_ATT_SHARED_OF(2, 128),
};

/* A call's flags: its keys are paged; it has lengths; output 0 reads
   [mask]; output 1 is the score with [x] added, subtracted, subtracted
   from, or multiplied, or [x] alone (stored scores); the launch stores
   each pair's score into [scores] and nothing else; [o] is float32; q's,
   k's or v's rows load 4 elements a vector; a block's keys of k and of v
   lie less than 2^31 elements from its first key's, where dense, and a
   row's elements less than 2^31 from its first. */
enum {
  NX_ATT_PAGED = 1,
  NX_ATT_LENGTHS = 2,
  NX_ATT_MASK = 4,
  NX_ATT_ADD = 8,
  NX_ATT_SUB = 16,
  NX_ATT_RSUB = 32,
  NX_ATT_MUL = 64,
  NX_ATT_X = 128,
  NX_ATT_SCORES = 256,
  NX_ATT_O32 = 512,
  NX_ATT_QVEC = 1024,
  NX_ATT_KVEC = 2048,
  NX_ATT_VVEC = 4096,
  NX_ATT_NARROW = 8192,
};

/* The arrays a lead index addresses, in the lead table's order. */
enum {
  NX_ATT_LQ,
  NX_ATT_LK,
  NX_ATT_LV,
  NX_ATT_LO,
  NX_ATT_LLSE,
  NX_ATT_LLEN,
  NX_ATT_LTAB,
  NX_ATT_LMASK,
  NX_ATT_LX,
  NX_ATT_LEAD_ARRAYS,
};

/* An attention's parameters. Buffers, each at its layout's offset: q, k
   and v (dense, or the pools), o, lse, lengths, the table, the mask's
   bools, x's float32, the scores stored, and the lead table: [leads]
   extents, their divisors (Fft.divisor), then for each array above its
   element steps along each lead axis. Then the flags, the scale's float32 bits, dk, dv, tq, tk, hq,
   hk, the heads and the rows of each head a unit holds, the page's slots
   (0 for dense keys) and the pools' pages, the lead axes and their
   elements, the launch's first unit, the query heads of a key head, the
   tiles of a head's rows and the chunks of a key head's query heads, the
   divisors of the tiles, the chunks and hk (Fft.divisor), a unit's rows
   (Plan.divisor) and the page's slots (Fft.divisor); and the element
   steps: q's along a head, a row and a dimension; k's and v's along a
   key head, a key (a slot), a dimension and a page; o's along a head, a row and a column;
   lse's along a head and a row; the table's along its entries; the
   mask's and x's along a head, a row and a key. The scores stored are
   C-contiguous over [lead…; hq; tq; tk]. */
enum {
  NX_ATT_Q,
  NX_ATT_K,
  NX_ATT_V,
  NX_ATT_O,
  NX_ATT_LSE,
  NX_ATT_LENGTHS_BUF,
  NX_ATT_TABLE,
  NX_ATT_MASK_BUF,
  NX_ATT_X_BUF,
  NX_ATT_SCORES_BUF,
  NX_ATT_LEAD,
  NX_ATT_FLAGS,
  NX_ATT_SCALE,
  NX_ATT_DK,
  NX_ATT_DV,
  NX_ATT_TQ,
  NX_ATT_TK,
  NX_ATT_HQ,
  NX_ATT_HK,
  NX_ATT_UNIT_HEADS,
  NX_ATT_UNIT_ROWS,
  NX_ATT_PAGE,
  NX_ATT_PAGES,
  NX_ATT_LEADS,
  NX_ATT_NB,
  NX_ATT_FIRST_UNIT,
  NX_ATT_G,
  NX_ATT_TILES_N,
  NX_ATT_CHUNKS_N,
  NX_ATT_TILES_DIV,
  NX_ATT_CHUNKS_DIV,
  NX_ATT_HK_DIV,
  NX_ATT_ROWS_DIV,
  NX_ATT_PAGE_DIV,
  NX_ATT_Q_H,
  NX_ATT_Q_T,
  NX_ATT_Q_D,
  NX_ATT_K_H,
  NX_ATT_K_T,
  NX_ATT_K_D,
  NX_ATT_K_P,
  NX_ATT_V_H,
  NX_ATT_V_T,
  NX_ATT_V_D,
  NX_ATT_V_P,
  NX_ATT_O_H,
  NX_ATT_O_T,
  NX_ATT_O_D,
  NX_ATT_LSE_H,
  NX_ATT_LSE_T,
  NX_ATT_TAB_P,
  NX_ATT_MASK_H,
  NX_ATT_MASK_T,
  NX_ATT_MASK_K,
  NX_ATT_X_H,
  NX_ATT_X_T,
  NX_ATT_X_K,
  NX_ATT_WORDS,
};

#endif
