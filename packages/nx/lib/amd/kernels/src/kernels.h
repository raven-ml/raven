/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx.amd's kernels as the host and the device both see them: the kernel
   table and each kernel's parameter words.

   A kernel reads one argument, its words: 64-bit integers the host writes
   at the indices this header's enumerations name, rig turning each address
   word into the address of a buffer of the submit. Every fact both sides
   use is an enumeration's constant or a row's column, so that kernels/gen.py
   writes kernels.ml from what clang reads here. Formulas of extents live on
   the host alone, which passes their values as words. */

#ifndef NX_AMD_KERNELS_H
#define NX_AMD_KERNELS_H

/* X(name, threads, words) for every kernel of the code object, in the order
   of kernels.ml's variant: the threads of each of its groups and the words
   of its argument. A family lists its rows in its own macro, which its unit
   expands to define its kernels. */
#define NX_AMD_KERNELS(X)                                                     \
  NX_AMD_ZERO_KERNELS(X) NX_AMD_LOOP_KERNELS(X) NX_AMD_MOVES_KERNELS(X)        \
      NX_AMD_FFT_KERNELS(X) NX_AMD_SORT_KERNELS(X)                             \
          NX_AMD_CONTRACT_KERNELS(X) NX_AMD_FACTOR_KERNELS(X)                 \
              NX_AMD_SPECTRAL_KERNELS(X)

/* Shared constants */

enum {
  /* The most axes a layout has (Nx_array.Layout.max_rank). */
  NX_MAX_RANK = 32,
};

/* An axis record's first words (device.h's Axes, Plan.axis): the axis's
   extent, its divisor word, then a stride for each operand that walks
   it, from NX_AXIS_STRIDES. */
enum {
  NX_AXIS_N,
  NX_AXIS_DIV,
  NX_AXIS_STRIDES,
};

/* A layout's words, from a kernel's index of its first: its rank, the
   position of its first element, whether it has fewer than 2^32 elements
   (device.h's small), then each axis's record of its extent, divisor and
   stride, axis j's from NX_LAYOUT_AXES + NX_LAYOUT_AXIS j, so that a
   layout of rank r uses its first 3 + 3r words. Positions and strides
   count elements from the buffer's first byte; the position of index
   (i₀, …, i₍ᵣ₋₁₎) is [at + Σ iⱼ strides[j]]. */
enum {
  NX_LAYOUT_RANK,
  NX_LAYOUT_AT,
  NX_LAYOUT_SMALL,
  NX_LAYOUT_AXES,
  NX_LAYOUT_AXIS = NX_AXIS_STRIDES + 1,
  NX_LAYOUT_WORDS = NX_LAYOUT_AXES + NX_LAYOUT_AXIS * NX_MAX_RANK,
};

/* Zeroing (zero.hip) */

#define NX_AMD_ZERO_KERNELS(X) X(zero, NX_ZERO_THREADS, NX_ZERO_WORDS)

/* zero: [count] 32-bit words from [dst], over [groups] groups along x;
   fresh tickets, which a call's first launch zeroes. */
enum {
  NX_ZERO_DST,
  NX_ZERO_COUNT,
  NX_ZERO_GROUPS,
  NX_ZERO_WORDS,
};

enum { NX_ZERO_THREADS = 256 };

/* Loops (loop.hip, whose header states how they run) */

#define NX_AMD_LOOP_KERNELS(X)                                                \
  X(loop_map, NX_LOOP_THREADS, NX_LOOP_WORDS)                                 \
  X(loop_reduce, NX_LOOP_THREADS, NX_LOOP_WORDS)                              \
  X(loop_scan, NX_LOOP_THREADS, NX_LOOP_WORDS)                                \
  X(loop_network, NX_LOOP_THREADS, NX_LOOP_WORDS)                             \
  X(loop_carries, NX_LOOP_THREADS, NX_LOOP_WORDS)                             \
  X(loop_plain, NX_LOOP_THREADS, NX_LOOP_WORDS)                               \
  X(loop_plain_scan, NX_LOOP_THREADS, NX_LOOP_WORDS)                          \
  X(loop_plain_carries, NX_LOOP_THREADS, NX_LOOP_WORDS)

/* A loop reads NX_LOOP_WORDS words: a header of NX_LOOP_HEADER words, the
   address of each buffer it reads or writes from word NX_LOOP_HEADER, then
   the loop's extents, its operands' descriptors and its tape, which the
   header points at by word, NX_LOOP_USED words in all. A group has
   NX_LOOP_THREADS work-items, each of [v] elements of a tile
   (NX_LOOP_MODE's bits 8 to 15), at most NX_LOOP_MOST. A reduction keeps
   nx.cpu's lane order: blocks of NX_LOOP_BLOCK terms, NX_LOOP_LANES lanes
   a block; a scan its chunks of NX_LOOP_CHUNK terms. A state in the
   workspace takes NX_LOOP_STATE 32-bit words. A plain reduction merges
   its groups' values in LDS NX_LOOP_PLAIN_CHUNK at a time, and takes
   columns of at most NX_LOOP_PLAIN_TALL blocks. */
enum {
  NX_LOOP_WORDS = 512,
  NX_LOOP_THREADS = 256,
  NX_LOOP_MOST = 8,
  NX_LOOP_BLOCK = 1024,
  NX_LOOP_LANES = 16,
  NX_LOOP_CHUNK = 4096,
  NX_LOOP_STATE = 8,
  NX_LOOP_PLAIN_CHUNK = 2048,
  NX_LOOP_PLAIN_TALL = 16,
};

/* The header's words. */
enum {
  NX_LOOP_USED,      /* the words the loop reads, from word 0 */
  NX_LOOP_MODE,      /* v << 8, then a reduction's work-items a block << 16 */
  NX_LOOP_GRID,      /* the launch's groups along x */
  NX_LOOP_RANK,      /* the loop's axes */
  NX_LOOP_SHAPE,     /* the word of its first axis's head (NX_AXIS_N and
                        NX_AXIS_DIV), the others' following it */
  NX_LOOP_NUMEL,     /* its indices */
  NX_LOOP_TAPE,      /* the word of the tape's first instruction */
  NX_LOOP_PROLOGUE,  /* the words of the tape's constants, written once */
  NX_LOOP_BODY,      /* the words run on each tile, after the prologue */
  NX_LOOP_EPILOGUE,  /* the words run after a scan's results, after the body */
  NX_LOOP_LOADS,     /* the word of the loads' table: each load's word */
  NX_LOOP_DSTS,      /* the word of the destinations' table */
  NX_LOOP_TERM,      /* the plane of a reduction's or a scan's terms */
  NX_LOOP_STATE_AT,  /* the first of a reduction's NX_LOOP_STATE planes */
  NX_LOOP_RESULT,    /* the first of a scan's result planes */
  NX_LOOP_CENTER,    /* the plane of each term's center */
  NX_LOOP_REDUCTION, /* reduction, then compute type << 8, carrier << 16,
                        finish << 24 */
  NX_LOOP_TERMS,     /* a row's terms */
  NX_LOOP_SEGMENT,   /* an output's terms at most: a segment of a row; a
                        network's slot's log2 */
  NX_LOOP_SEGMENTS,  /* a row's segments */
  NX_LOOP_OUTPUTS,   /* the outputs: rows times segments */
  NX_LOOP_GROUP,     /* a reduction's blocks a group, a power of two; a
                        scan's work-items a chunk */
  NX_LOOP_GROUPS,    /* an output's groups */
  NX_LOOP_RUNS,      /* a block's runs, of its work-items times v terms; a
                        scan's chunk's; a network's slice's */
  NX_LOOP_WORK,      /* the workspace's address word, past the header */
  NX_LOOP_TICKETS,   /* the tickets' address word, a zero word an output */
  NX_LOOP_PARTIALS,  /* the byte of the groups' states */
  NX_LOOP_TOTALS,    /* the byte of a scan's chunk totals */
  NX_LOOP_CENTERS,   /* the byte of the outputs' centers */
  NX_LOOP_CARRY,     /* a scan's carry: its load, or -1 */
  NX_LOOP_CELLS,     /* a plain reduction's cells at work a group */
  NX_LOOP_HEADER
};

/* A descriptor's words: its dtype, then whether it is padded << 8 and its
   address word << 16; its first element's position; then a plain
   operand's stride along each axis of the loop, or a padded one's fill
   (two words), its rank r and, per axis of the operand, NX_LOOP_PAD words:
   its lo, its interior, its extent, its stride, the loop axes it reads
   (the second << 16, 0xffff for none), and their coefficients. */
enum { NX_LOOP_PAD = 7 };

/* An instruction: its op, then its planes d, a, b and c of 12 bits from
   bits 16, 28, 40 and 52. A kind's op is its compute type << 7 and its
   code: an op1's, NX_LOOP_BINARY plus an op2's, NX_LOOP_TERNARY plus an
   op3's (nx_kernel.h). A constant's two words follow it. Fma at a narrow
   float is a kind at NX_LOOP_F32 past the op3s' codes: nx_kinds.h's
   nx_fma_narrow, or nx_fma_bf16 for bfloat16, the exact value rounded to
   odd, so that the encode after it rounds once. */
enum {
  NX_LOOP_BINARY = 32,
  NX_LOOP_TERNARY = 64,
  NX_LOOP_FMA_NARROW = NX_LOOP_TERNARY + 2,
  NX_LOOP_FMA_BF16 = NX_LOOP_TERNARY + 3,
};
enum {
  NX_LOOP_F32, NX_LOOP_F64, NX_LOOP_C64, NX_LOOP_C128,
  NX_LOOP_I32, NX_LOOP_U32, NX_LOOP_I64, NX_LOOP_U64
};
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

/* Moves (moves.hip, whose header states how they run)

   A move's words address each array by its buffer's first byte and give
   its first element's position, in elements as a layout states them: a
   sub-byte element at position p holds bits p b to p b + b - 1. An element
   is [count] units of [unit] bits, its dtype's alignment: 1, 4, 8, 16, 32
   or 64, and 2 units for a complex number. Arrays have up to NX_MAX_RANK
   axes. A launch's groups stride over its work, [groups] of them along
   x. */

#define NX_AMD_MOVES_KERNELS(X)                                               \
  X(move, NX_MOVE_THREADS, NX_MOVE_WORDS)                                     \
  X(runs_chunk, NX_MOVE_THREADS, NX_RUNS_WORDS)                               \
  X(runs_carry, NX_MOVE_THREADS, NX_RUNS_WORDS)                               \
  X(runs_merge, NX_MOVE_THREADS, NX_RUNS_WORDS)                               \
  X(assemble, NX_MOVE_THREADS, NX_ASSEMBLE_WORDS)                             \
  X(fold, NX_FOLD_THREADS, NX_FOLD_WORDS)                                     \
  X(scatter, NX_MOVE_THREADS, NX_SCATTER_WORDS)                               \
  X(scatter_tile_f32, NX_MOVE_THREADS, NX_SCATTER_WORDS)                      \
  X(scatter_tile_f64, NX_MOVE_THREADS, NX_SCATTER_WORDS)                      \
  X(scatter_rows_f32, NX_MOVE_THREADS, NX_SCATTER_WORDS)                      \
  X(scatter_rows_f64, NX_MOVE_THREADS, NX_SCATTER_WORDS)                      \
  X(scatter_tally, NX_MOVE_THREADS, NX_SCATTER_WORDS)                         \
  X(scatter_pack_n, NX_MOVE_THREADS, NX_SCATTER_WORDS)                        \
  X(scatter_pack_w, NX_MOVE_THREADS, NX_SCATTER_WORDS)                        \
  X(scatter_count_n, NX_MOVE_THREADS, NX_SCATTER_WORDS)                       \
  X(scatter_count_w, NX_MOVE_THREADS, NX_SCATTER_WORDS)                       \
  X(scatter_move_n, NX_MOVE_THREADS, NX_SCATTER_WORDS)                        \
  X(scatter_move_w, NX_MOVE_THREADS, NX_SCATTER_WORDS)                        \
  X(scatter_sum_f32_n, NX_TREE_THREADS, NX_SCATTER_WORDS)                     \
  X(scatter_sum_f32_w, NX_TREE_THREADS, NX_SCATTER_WORDS)                     \
  X(scatter_sum_f64_n, NX_TREE_THREADS, NX_SCATTER_WORDS)                     \
  X(scatter_sum_f64_w, NX_TREE_THREADS, NX_SCATTER_WORDS)

enum {
  /* A move's group, a fold's, and a scatter tree's, whose work-items each
     keep a stack in LDS. */
  NX_MOVE_THREADS = 256,
  NX_FOLD_THREADS = 64,
  NX_TREE_THREADS = 64,
};

/* move: dst, C-contiguous over the space of [numel] elements, gets each
   element as [mode] says: src at its index c; src at c with axis [axis]
   replaced by the position idx holds at c, or zero bits for one outside
   [0, d); or src at the position a range gather's runs give. [sa]
   is src's stride along [axis], [d] its extent there; [k] a range
   gather's runs a row, of [starts], [ends] and [maxima], their ends'
   running maximum. [small]: [numel] < 2^32. An axis of the space: its
   record (NX_AXIS_*), of the strides of dst (C order), src and the
   positions (idx; for NX_MOVE_RANGES, the rows of the runs). A copy does not read the
   words from NX_MOVE_GATHER_WORDS to NX_MOVE_AX.

   With [chunks] > 0, the space's last axis is a run: dst and src step by
   one element along it and the positions by none, so that one value
   (the source row, or zeros) holds along it. A wave then copies a chunk
   of NX_MOVE_CHUNK bytes of a run at a time, [chunks] a run, an axis's
   head (NX_AXIS_*), in 16-byte
   accesses where dst and src lie alike modulo 16. The host takes this
   path for runs of NX_MOVE_RUN bytes or more. */
enum {
  NX_MOVE_CHUNK = 2048,
  NX_MOVE_RUN = 256,
};

enum {
  NX_MOVE_AXIS_SD = NX_AXIS_STRIDES,
  NX_MOVE_AXIS_SS,
  NX_MOVE_AXIS_SI,
  NX_MOVE_AXIS_WORDS,
};

enum {
  NX_MOVE_DST,
  NX_MOVE_SRC,
  NX_MOVE_NUMEL,
  NX_MOVE_DST_OFF,
  NX_MOVE_SRC_OFF,
  NX_MOVE_IDX_OFF,
  NX_MOVE_RANK,
  NX_MOVE_AXIS,
  NX_MOVE_MODE,
  NX_MOVE_SMALL,
  NX_MOVE_UNIT,
  NX_MOVE_COUNT,
  NX_MOVE_GROUPS,
  NX_MOVE_CHUNKS,
  NX_MOVE_CHUNKS_DIV,
  NX_MOVE_GATHER_WORDS,
  NX_MOVE_IDX = NX_MOVE_GATHER_WORDS,
  NX_MOVE_STARTS,
  NX_MOVE_ENDS,
  NX_MOVE_MAXIMA,
  NX_MOVE_D,
  NX_MOVE_SA,
  NX_MOVE_K,
  NX_MOVE_AX,
  NX_MOVE_WORDS = NX_MOVE_AX + NX_MAX_RANK * NX_MOVE_AXIS_WORDS,
};

enum { NX_MOVE_COPY, NX_MOVE_GATHER, NX_MOVE_RANGES };

/* runs_*: a range gather's runs, [rows] lead rows of [k] runs, copied from
   [in_s] and [in_e] (strides [s_k] and [e_k] along the runs, the axes'
   along the lead axes) to [starts] and [ends], and their ends' running
   maximum to [maxima], in chunks of NX_RUNS_CHUNK runs, each chunk's
   maximum in [carry]. A lead axis: its record (NX_AXIS_*), of the
   strides of [in_s] and [in_e]. */
enum { NX_RUNS_CHUNK = 4096 };

enum {
  NX_RUNS_AXIS_SS = NX_AXIS_STRIDES,
  NX_RUNS_AXIS_SE,
  NX_RUNS_AXIS_WORDS,
};

enum {
  NX_RUNS_IN_S,
  NX_RUNS_IN_E,
  NX_RUNS_STARTS,
  NX_RUNS_ENDS,
  NX_RUNS_MAXIMA,
  NX_RUNS_CARRY,
  NX_RUNS_ROWS,
  NX_RUNS_K,
  NX_RUNS_CHUNKS,
  NX_RUNS_S_K,
  NX_RUNS_E_K,
  NX_RUNS_S_OFF,
  NX_RUNS_E_OFF,
  NX_RUNS_LEAD,
  NX_RUNS_SMALL,
  NX_RUNS_GROUPS,
  NX_RUNS_AX,
  NX_RUNS_WORDS = NX_RUNS_AX + NX_MAX_RANK * NX_RUNS_AXIS_WORDS,
};

/* assemble: up to NX_ASSEMBLE_PIECES pieces a launch, [n] of them. dst
   has the [rank] axes' heads (NX_AXIS_*) from [dn], NX_ASSEMBLE_DN_AXIS
   words each; the pieces' records follow them, of NX_ASSEMBLE_HEAD + rank
   NX_ASSEMBLE_AXIS words each: the piece's buffer, its first element, its
   first element's place in dst from [dst_off], the elements of the
   launch's pieces before it, the mask of the launch's later pieces whose
   regions meet its own and the chunks of its last axis's rows, an axis's
   head; then per axis its record (NX_AXIS_*) of its stride, and its
   region's start and step along dst's axis. A launch of [mode] NX_ASSEMBLE_FILL stores [fill], skipping with
   [check] what the launch's pieces cover, which are then every piece; one
   of NX_ASSEMBLE_COPY copies its [total] elements of pieces.

   With [runs], each row of dst's last axis, or of a piece's, is a run
   of NX_MOVE_RUN bytes or more, copied by a wave a chunk as move copies
   its runs (NX_MOVE_CHUNK). A fill's pieces then each step by 1 or -1
   along the last axis, so that a row's covered elements are a span a
   piece; a copy's step by 1 along it, their sources by one element, and
   meet no later piece of the launch; [total] and each piece's [prefix]
   count chunks, and [chunks], an axis's head, those of dst's rows. */
enum {
  NX_ASSEMBLE_PIECES = 16,
  NX_ASSEMBLE_TABLE = 448,
};

enum {
  NX_ASSEMBLE_REC_SRC,
  NX_ASSEMBLE_REC_OFF,
  NX_ASSEMBLE_REC_BASE,
  NX_ASSEMBLE_REC_PREFIX,
  NX_ASSEMBLE_REC_LATER,
  NX_ASSEMBLE_REC_CHUNKS,
  NX_ASSEMBLE_REC_CHUNKS_DIV,
  NX_ASSEMBLE_HEAD,
};

enum {
  NX_ASSEMBLE_AXIS_SX = NX_AXIS_STRIDES,
  NX_ASSEMBLE_AXIS_START,
  NX_ASSEMBLE_AXIS_STEP,
  NX_ASSEMBLE_AXIS,
};

enum { NX_ASSEMBLE_FILL, NX_ASSEMBLE_COPY };

enum {
  NX_ASSEMBLE_DST,
  NX_ASSEMBLE_TOTAL,
  NX_ASSEMBLE_DST_OFF,
  NX_ASSEMBLE_FILL_LO,
  NX_ASSEMBLE_FILL_HI,
  NX_ASSEMBLE_RANK,
  NX_ASSEMBLE_N,
  NX_ASSEMBLE_SMALL,
  NX_ASSEMBLE_CHECK,
  NX_ASSEMBLE_UNIT,
  NX_ASSEMBLE_COUNT,
  NX_ASSEMBLE_MODE,
  NX_ASSEMBLE_GROUPS,
  NX_ASSEMBLE_RUNS,
  NX_ASSEMBLE_CHUNKS,
  NX_ASSEMBLE_CHUNKS_DIV,
  NX_ASSEMBLE_DN,
  NX_ASSEMBLE_DN_AXIS = NX_AXIS_STRIDES,
  NX_ASSEMBLE_WORDS =
      NX_ASSEMBLE_DN + NX_ASSEMBLE_DN_AXIS * NX_MAX_RANK + NX_ASSEMBLE_TABLE,
};

/* fold: the adjoint of a Padded load of dst's shape. Per axis of dst, its
   extent and divisor (NX_AXIS_*), the padding before it and between its
   elements, the padded
   extent and x's stride; a windowed axis ([win] >= 0, the window's number
   among the [windows]) its windows' taps, step, dilation and count and
   x's strides along the windows and the taps. A work-item keeps, for each
   window, its padded position and its tap: 16 bytes of the group's
   dynamic LDS a window and work-item.

   With [narrow], every index, padded position and tap's reach fits 31
   bits and the divisors 32: a windowed axis's [ms] is its step's
   Plan.divisor word. A launch of NX_FOLD_FEW
   windows or fewer then keeps each window's position and tap in
   registers, the windows in the order of their axes, and divides by
   device.h's divide. */
enum { NX_FOLD_FEW = 3 };

enum {
  NX_FOLD_AXIS_LO = NX_AXIS_STRIDES,
  NX_FOLD_AXIS_INTER,
  NX_FOLD_AXIS_PEXT,
  NX_FOLD_AXIS_SX,
  NX_FOLD_AXIS_TAPS,
  NX_FOLD_AXIS_STEP,
  NX_FOLD_AXIS_DIL,
  NX_FOLD_AXIS_NWIN,
  NX_FOLD_AXIS_SW,
  NX_FOLD_AXIS_ST,
  NX_FOLD_AXIS_WIN,
  NX_FOLD_AXIS_MS,
  NX_FOLD_AXIS_WORDS,
};

enum {
  NX_FOLD_DST,
  NX_FOLD_X,
  NX_FOLD_NUMEL,
  NX_FOLD_DST_OFF,
  NX_FOLD_X_OFF,
  NX_FOLD_RANK,
  NX_FOLD_DTYPE,
  NX_FOLD_SMALL,
  NX_FOLD_UNIT,
  NX_FOLD_COUNT,
  NX_FOLD_WINDOWS,
  NX_FOLD_GROUPS,
  NX_FOLD_NARROW,
  NX_FOLD_AX,
  NX_FOLD_WORDS = NX_FOLD_AX + NX_MAX_RANK * NX_FOLD_AXIS_WORDS,
};

/* scatter: one pass of [op]: stores under ~unique; each target's last
   update, its position plus one by atomic maximum into [owner], then the
   owner's store; an integer Add by atomics; Max and Min; a float Add under
   ~unique; and a complex Max or Min ([greater] for Max): for complex128
   in three passes, each target's greatest real part's key, then of the
   updates of that real part the greatest imaginary part's, by atomic
   maximum into [owner] (two 64-bit keys a target), then each target's
   store; for complex64 in two, both parts' key at once (one 64-bit key a
   target), then the store; and each target's last update of 1, 2 or 4
   bytes, of a position below 2^32, packed: ((position + 1) << 32) | its
   bits by atomic maximum into [owner] (a 64-bit word a target), then each
   target's store. */
enum {
  NX_SCATTER_SET,
  NX_SCATTER_OWN,
  NX_SCATTER_OWNED,
  NX_SCATTER_ADD_INT,
  NX_SCATTER_MAX,
  NX_SCATTER_MIN,
  NX_SCATTER_ADD_UNIQUE,
  NX_SCATTER_TOP_RE,
  NX_SCATTER_TOP_IM,
  NX_SCATTER_TOP_STORE,
  NX_SCATTER_OWN_PACKED,
  NX_SCATTER_PACKED_STORE,
};

/* A float Add's tiles: NX_SCATTER_TILE positions of one slice, or of
   several short ones; a row tile's NX_SCATTER_ROWS positions by up to
   2^lcols columns, NX_SCATTER_ROW_BYTES bytes of values. Its sort by
   target: at most NX_SCATTER_PARTS parts, each a group's, of tiles of
   NX_SCATTER_ITEMS_N narrow or NX_SCATTER_ITEMS_W wide items a
   work-item. Its run sums: a run of at most NX_SCATTER_SHORT entries
   summed whole by one work-item, a longer one by levels of
   2^NX_SCATTER_LEVEL_BITS positions; a sum's stack of NX_SCATTER_LEVELS
   levels a work-item. */
enum {
  NX_SCATTER_TILE = 1024,
  NX_SCATTER_TILE_BITS = 10,
  NX_SCATTER_ROWS = 256,
  NX_SCATTER_ROWS_BITS = 8,
  NX_SCATTER_ROW_BYTES = 32768,
  NX_SCATTER_PARTS = 256,
  NX_SCATTER_ITEMS_N = 16,
  NX_SCATTER_ITEMS_W = 8,
  NX_SCATTER_SHORT = 64,
  NX_SCATTER_LEVEL_BITS = 8,
  NX_SCATTER_LEVELS = 32,
};

/* A float Add's entries: the updates, or the tiles' entries. */
enum {
  NX_SCATTER_FROM_UPDATES,
  NX_SCATTER_FROM_TILES,
};

/* An axis of the updates: its record (NX_AXIS_*), of the strides of dst
   (C order of into's shape), idx and the updates. */
enum {
  NX_SCATTER_AXIS_SD = NX_AXIS_STRIDES,
  NX_SCATTER_AXIS_SI,
  NX_SCATTER_AXIS_SU,
  NX_SCATTER_AXIS_WORDS,
};

/* A float Add past one tile sorts its entries ([from]) by target: the
   updates in C order of their slices, then position; or with row tiles,
   rows of updates, an entry's key its target row; or each tile's
   entries, at most [est] a tile: [ent_g] their keys, [ent_v] their rows
   of 2^lcols values, [ent_n] their count, packed as items into [items1]
   before the first pass. An item is an entry's key and
   index, a narrow one in one word, the key above [pb] bits, a wide one
   in two. A pass ([pass]) of [nparts] parts moves the items by a key byte
   between [items0] and [items1], its counts and then each part's places
   in [counts] behind a ticket; [ctl] holds the sorted count, of
   the updates in range or the tiles' entries. A run sum's
   level ([level]) sums each run's blocks of positions, [pbits] of them,
   into [part], their ends in [plen0] or [plen1] and their levels in
   [mark]; a run of at most [whole] entries, if not 0, whole. A tile
   entry's position is its index over [span]. Each of [m], [last], [nc]
   and [span] is an axis's head (NX_AXIS_*). A pass of scatter does not read
   the words from NX_SCATTER_TILED_WORDS to NX_SCATTER_AX. */
enum {
  NX_SCATTER_DST,
  NX_SCATTER_IDX,
  NX_SCATTER_UPD,
  NX_SCATTER_OWNER,
  NX_SCATTER_NUMEL_U,
  NX_SCATTER_NUMEL_D,
  NX_SCATTER_DST_OFF,
  NX_SCATTER_IDX_OFF,
  NX_SCATTER_UPD_OFF,
  NX_SCATTER_D,
  NX_SCATTER_SDA,
  NX_SCATTER_RANK,
  NX_SCATTER_AXIS,
  NX_SCATTER_OP,
  NX_SCATTER_DTYPE,
  NX_SCATTER_SMALL,
  NX_SCATTER_WIDE,
  NX_SCATTER_GREATER,
  NX_SCATTER_UNIT,
  NX_SCATTER_COUNT,
  NX_SCATTER_GROUPS,
  NX_SCATTER_TILED_WORDS,
  NX_SCATTER_ENT_G = NX_SCATTER_TILED_WORDS,
  NX_SCATTER_ENT_V,
  NX_SCATTER_ENT_N,
  NX_SCATTER_ITEMS0,
  NX_SCATTER_ITEMS1,
  NX_SCATTER_COUNTS,
  NX_SCATTER_TICKET,
  NX_SCATTER_CTL,
  NX_SCATTER_PART,
  NX_SCATTER_PLEN0,
  NX_SCATTER_PLEN1,
  NX_SCATTER_MARK,
  NX_SCATTER_M,
  NX_SCATTER_M_DIV,
  NX_SCATTER_SLICES,
  NX_SCATTER_RUN,
  NX_SCATTER_GROUP,
  NX_SCATTER_NB,
  NX_SCATTER_NSG,
  NX_SCATTER_TILES,
  NX_SCATTER_EST,
  NX_SCATTER_LAST,
  NX_SCATTER_LAST_DIV,
  NX_SCATTER_NC,
  NX_SCATTER_NC_DIV,
  NX_SCATTER_LCOLS,
  NX_SCATTER_FROM,
  NX_SCATTER_ENTRIES,
  NX_SCATTER_PB,
  NX_SCATTER_PASS,
  NX_SCATTER_NPARTS,
  NX_SCATTER_SPAN,
  NX_SCATTER_SPAN_DIV,
  NX_SCATTER_PBITS,
  NX_SCATTER_LEVEL,
  NX_SCATTER_WHOLE,
  NX_SCATTER_AX,
  NX_SCATTER_WORDS = NX_SCATTER_AX + NX_MAX_RANK * NX_SCATTER_AXIS_WORDS,
};

/* Fourier transforms (fft.hip)

   fft.hip runs a transform of N points as Stockham steps, a launch of fft
   each (fft.ml plans them). A step's virtual line v is batch line
   l = v / m and butterfly b = v mod m, m = N / l: it loads the l points
   b + r m of line l, from [nsrc] on +0, twiddles point r by
   w_N^((b mod ns) r gts) (gts = N / (ns l), none where 0), runs its [ops]
   on them in LDS, the inner FFT of [np] passes among them, and stores
   output k at (b / ns) ns l + b mod ns + k ns, below [limit]. A line that
   fits a group is one step, m = ns = 1. A group holds [g] lines of [tl]
   work-items each, [lp] slots apart, side by side in memory where [lf_ld]
   (loads) and [lf_st] (stores), over [groups] groups along x.

   A plan is one buffer of tables, which a launch reaches through [plan],
   each table at a byte offset from its first: the inner FFT's w_l^k [tw]
   (k < l / 2 where [half]), a long line's w_N^m = glo[m mod 2^g_bits]
   ghi[m >> g_bits], a real line's w_2h^k by [rlo], [rhi] and [r_bits],
   Bluestein's chirp and its kernel's transform [bspec], or Rader's
   kernel [bspec] and discrete logarithms [rlog]. fft_real splits a long
   real line's bins from its half-length transform in place, or makes the
   half-length points from the bins. */

#define NX_AMD_FFT_KERNELS(X)                                                 \
  X(fft_f32, NX_FFT_THREADS, NX_FFT_WORDS)                                    \
  X(fft_f64, NX_FFT_THREADS, NX_FFT_WORDS)                                    \
  X(fft_real_f32, NX_FFT_REAL_THREADS, NX_FFT_REAL_WORDS)                     \
  X(fft_real_f64, NX_FFT_REAL_THREADS, NX_FFT_REAL_WORDS)

enum {
  /* A step's group: its lines' work-items first, the others idle. */
  NX_FFT_THREADS = 512,
  /* fft_real's group: a work-item a pair of bins at a time. */
  NX_FFT_REAL_THREADS = 256,
  /* The inner FFT's most passes and a step's most ops. */
  NX_FFT_PASSES = 16,
  NX_FFT_OPS = 8,
};

/* A pass of the inner FFT, from its first word: radix r after ns points,
   an axis's head (NX_AXIS_*), q = l / r butterflies, twiddle step
   ts = l / (ns r). */
enum {
  NX_FFT_PASS_R,
  NX_FFT_PASS_NS,
  NX_FFT_PASS_NS_DIV,
  NX_FFT_PASS_Q,
  NX_FFT_PASS_TS,
  NX_FFT_PASS_WORDS,
};

/* Loads: complex points, real points, real pairs as complex points,
   bins completed to n = nh points by conjugation. Stores: complex points,
   real parts, complex points as real pairs. */
enum { NX_FFT_LD_C, NX_FFT_LD_R, NX_FFT_LD_RP, NX_FFT_LD_HC };
enum { NX_FFT_ST_C, NX_FFT_ST_RE, NX_FFT_ST_RP };

/* Ops in LDS: the inner FFT; Bluestein's chirp, kernel product and last
   chirp; a real line's split and pre-pass (h = nh); Rader's kernel
   product. Stores' pointwise steps: Bluestein's kernel product or last
   chirp. */
enum {
  NX_FFT_OP_FFT,
  NX_FFT_OP_CHIRP,
  NX_FFT_OP_BMUL,
  NX_FFT_OP_BFIN,
  NX_FFT_OP_SPLIT,
  NX_FFT_OP_HPRE,
  NX_FFT_OP_RMUL,
};
enum { NX_FFT_POST_NONE, NX_FFT_POST_BMUL, NX_FFT_POST_BFIN };

/* fft: [nv] virtual lines, small where fewer than 2^32, each of [m]
   butterflies of [ns] points, [m] and [ns] axes' heads (NX_AXIS_*); [lld]
   points loaded and [lst] stored a line;
   [conj_in] and [conj_out] conjugate loads and stores; [chirp_in]
   multiplies loads by the chirp; [rader] permutes a prime line's points;
   [gshift] is log2 g. A side's line l starts at the position of index l
   of its layout and its point j [ps_in] or [ps_out] further, in elements
   of the side's type, a complex or a real. From [op], one after another:
   [nops] ops, [np] passes, the source's layout and the destination's, of
   one shape, [used] words in all, the ones a group copies. NX_FFT_WORDS
   counts the most. */
enum {
  NX_FFT_SRC,
  NX_FFT_DST,
  NX_FFT_PLAN,
  NX_FFT_GROUPS,
  NX_FFT_NV,
  NX_FFT_SMALL,
  NX_FFT_M,
  NX_FFT_M_DIV,
  NX_FFT_NS,
  NX_FFT_NS_DIV,
  NX_FFT_NSRC,
  NX_FFT_LIMIT,
  NX_FFT_GTS,
  NX_FFT_NH,
  NX_FFT_TW,
  NX_FFT_GLO,
  NX_FFT_GHI,
  NX_FFT_RLO,
  NX_FFT_RHI,
  NX_FFT_CHIRP,
  NX_FFT_BSPEC,
  NX_FFT_RLOG,
  NX_FFT_L,
  NX_FFT_LLD,
  NX_FFT_LST,
  NX_FFT_LP,
  NX_FFT_G,
  NX_FFT_TL,
  NX_FFT_GSHIFT,
  NX_FFT_LF_LD,
  NX_FFT_LF_ST,
  NX_FFT_LD,
  NX_FFT_ST,
  NX_FFT_CONJ_IN,
  NX_FFT_CONJ_OUT,
  NX_FFT_CHIRP_IN,
  NX_FFT_POST,
  NX_FFT_NP,
  NX_FFT_HALF,
  NX_FFT_NOPS,
  NX_FFT_G_BITS,
  NX_FFT_R_BITS,
  NX_FFT_RADER,
  NX_FFT_PS_IN,
  NX_FFT_PS_OUT,
  NX_FFT_USED,
  NX_FFT_OP,
  NX_FFT_WORDS = NX_FFT_OP + NX_FFT_OPS + NX_FFT_PASSES * NX_FFT_PASS_WORDS +
                 2 * NX_LAYOUT_WORDS,
};

/* fft_real: [nl] lines of [h] complex points, [pairs] = h / 2 + 1 pairs
   of bins each, an axis's head (NX_AXIS_*), small where fewer than 2^32
   pairs in all, split in place on [dst] ([split]) or made from [src]'s
   bins into [dst], over [groups] groups along x. Its sides' lines are as fft's: from [in], the source's layout
   and the destination's, [used] words in all. */
enum {
  NX_FFT_REAL_SRC,
  NX_FFT_REAL_DST,
  NX_FFT_REAL_PLAN,
  NX_FFT_REAL_GROUPS,
  NX_FFT_REAL_NL,
  NX_FFT_REAL_H,
  NX_FFT_REAL_PAIRS,
  NX_FFT_REAL_PAIRS_DIV,
  NX_FFT_REAL_SMALL,
  NX_FFT_REAL_RLO,
  NX_FFT_REAL_RHI,
  NX_FFT_REAL_R_BITS,
  NX_FFT_REAL_SPLIT,
  NX_FFT_REAL_PS_IN,
  NX_FFT_REAL_PS_OUT,
  NX_FFT_REAL_USED,
  NX_FFT_REAL_IN,
  NX_FFT_REAL_WORDS = NX_FFT_REAL_IN + 2 * NX_LAYOUT_WORDS,
};

/* Contractions (contract.hip) */

/* X(name, threads, words) of the contraction's kernels: the pack, the
   prepasses of Rows and Contracting segments, then per compute type the
   SIMT tiles reading operands of the compute type (native) or of any real
   dtype (convert), and gemv. A GEMM row's name states its tile, which
   NX_AMD_GEMM_TILES gives. */
#define NX_AMD_CONTRACT_KERNELS(X)                                            \
  X(contract_pack, NX_PACK_THREADS, NX_PACK_WORDS)                            \
  X(contract_rows, NX_SEG_THREADS, NX_SEG_WORDS)                              \
  X(contract_ks, NX_SEG_THREADS, NX_SEG_WORDS)                                \
  NX_AMD_GEMM_TILES(NX_AMD_GEMM_ROW, X)                                       \
  NX_AMD_WMMA_TILES(NX_AMD_WMMA_ROW, X)                                       \
  X(contract_gemv_f32, NX_GEMV_THREADS, NX_CONTRACT_WORDS)                    \
  X(contract_gemv_f64, NX_GEMV_THREADS, NX_CONTRACT_WORDS)                    \
  X(contract_gemv_u32, NX_GEMV_THREADS, NX_CONTRACT_WORDS)                    \
  X(contract_gemv_u64, NX_GEMV_THREADS, NX_CONTRACT_WORDS)
#define NX_AMD_GEMM_ROW(X, name, t, bm, bn, bk, tm, tn, lm, wm, wn, load)     \
  X(name, (wm) * (wn) * 32, NX_CONTRACT_WORDS)

/* R(X, name, t, BM, BN, BK, TM, TN, LM, WM, WN, load) for every SIMT
   tile, X passed through to R: its compute type (f32, f64, u32, u64); a
   group's BM x BN outputs over k-tiles of BK; a lane's TM x TN outputs;
   its wave's LM rows of lanes; the WM x WN waves; and its load, native or
   convert. */
#define NX_AMD_GEMM_TILES(R, X)                                               \
  R(X, contract_f32_128x128, f32, 128, 128, 16, 8, 8, 4, 4, 2, native)        \
  R(X, contract_f32_64x64, f32, 64, 64, 16, 4, 4, 4, 4, 2, native)            \
  R(X, contract_f32_32x32, f32, 32, 32, 16, 4, 4, 4, 2, 1, native)            \
  R(X, contract_f32_r8, f32, 8, 256, 16, 8, 2, 1, 1, 4, native)               \
  R(X, contract_f64_64x64, f64, 64, 64, 16, 4, 4, 4, 4, 2, native)            \
  R(X, contract_f64_32x32, f64, 32, 32, 16, 4, 4, 4, 2, 1, native)            \
  R(X, contract_f64_r8, f64, 8, 128, 16, 8, 1, 1, 1, 4, native)               \
  R(X, contract_u32_64x64, u32, 64, 64, 16, 4, 4, 4, 4, 2, native)            \
  R(X, contract_u64_64x64, u64, 64, 64, 16, 4, 4, 4, 4, 2, native)            \
  R(X, contract_cv_f32_128x128, f32, 128, 128, 16, 8, 8, 4, 4, 2, convert)    \
  R(X, contract_cv_f64_64x64, f64, 64, 64, 16, 4, 4, 4, 4, 2, convert)        \
  R(X, contract_cv_u32_64x64, u32, 64, 64, 16, 4, 4, 4, 4, 2, convert)        \
  R(X, contract_cv_u64_64x64, u64, 64, 64, 16, 4, 4, 4, 4, 2, convert)

/* A SIMT tile's dynamic LDS bytes, of compute type [t]: two stages of BK
   k-major rows of a's BM and b's BN elements, each row padded by a vector
   of 16 bytes. */
#define NX_GEMM_LDS(t, bm, bn, bk)                                            \
  (2 * (bk) * ((bm) + (bn) + 32 / NX_GEMM_BYTES_##t) * NX_GEMM_BYTES_##t)
#define NX_GEMM_BYTES_f32 4
#define NX_GEMM_BYTES_f64 8
#define NX_GEMM_BYTES_u32 4
#define NX_GEMM_BYTES_u64 8

/* M(X, name, kind, BM, BN, BKB, WM, WN) for every matrix-unit tile, X
   passed through to M: the operands it sums (f16, bf16, 8bit: the float8
   and 8-bit integer pairs, any: every pair), chosen by their dtypes at
   the kernel's entry; a group's BM x BN outputs over k-tiles of BKB bytes
   a row; its WM x WN waves. */
#define NX_AMD_WMMA_TILES(M, X)                                               \
  M(X, contract_wmma_f16_128x128, f16, 128, 128, 64, 2, 4)                    \
  M(X, contract_wmma_bf16_128x128, bf16, 128, 128, 64, 2, 4)                  \
  M(X, contract_wmma_8bit_128x128, 8bit, 128, 128, 64, 2, 4)                  \
  M(X, contract_wmma_any_16x64, any, 16, 64, 64, 1, 4)
#define NX_AMD_WMMA_ROW(X, name, kind, bm, bn, bkb, wm, wn)                   \
  X(name, (wm) * (wn) * 32, NX_CONTRACT_WORDS)

/* A matrix-unit tile's dynamic LDS bytes: two stages of its BM + BN rows,
   each a k-tile's BKB bytes and 16 of padding. */
#define NX_WMMA_LDS(bm, bn, bkb) (2 * ((bm) + (bn)) * ((bkb) + 16))

/* An operand a or b as a contraction kernel reads it, from a kernel's word
   [o]: the address of its first element; its strides along batch, its rows
   (a's) or columns (b's), and k, in elements; its rows, past which loads
   read the last; its dtype's code, a sub-byte dtype's elements one a
   byte; and NX_OP_ bits. */
enum {
  NX_OPERAND_P,
  NX_OPERAND_S,
  NX_OPERAND_ROWS = NX_OPERAND_S + 3,
  NX_OPERAND_DTYPE,
  NX_OPERAND_FLAGS,
  NX_OPERAND_WORDS,
};

/* NX_OPERAND_FLAGS' bits. */
enum {
  /* Its free axis is contiguous, not its k. */
  NX_OP_FREE = 1,
};

/* contract_*: y[z, i, j] = init[z, i, j] + sum_k a[z, i, k] * b[z, j, k] in
   the accumulator [acc_dtype], rounded once to y's dtype, for z < batch,
   i < m, j < n, over real dtypes: a complex call is the real contraction
   of its parts, which the plan lays out, n = 2 n' columns and k = 2 k'
   terms. [init] is 0 for none; init and y have one element per output,
   strides [si] and [sy] along batch, rows and columns.

   The SIMT kernels (GEMM, GEMV) share k out as contract.hip states: tiles
   [0, main) of the call's [tiles_m] x [tiles_n] tiles a batch element sum
   in [s1] ranges of k, the others in [s2], each range a group; the split
   tiles' ranges' sums go to [partials] and [tickets] holds a zero word a
   tile. NX_CONTRACT_ bits in [flags].

   Rows segments: a's rows grouped by their ids; [desc] a row tile each,
   four int32 words (batch element, group, first, count), [ntiles] of them;
   the tile's rows are perm[first ..] and multiply b's group's slice, [sbg]
   elements apart in b; group [groups] is the rows in no group, which have
   no products. Contracting segments: output slice e of batch element z
   sums the k kperm[koff[z * (kgroups + 1) + e] ..] before the next
   group's, in increasing k; the slices lie [syg] (y) and [sig] (init)
   elements apart; the call's batch, for its tiles, is kgroups * batch,
   slice e of z at e * batch + z; with NX_CONTRACT_K_PAIRS, a complex
   call's, its terms 2q and 2q + 1 those of the index kperm takes as q.
   koff's words are int64, kperm's int64 with NX_CONTRACT_K_WIDE and int32
   otherwise. Each 0 for a call without them. */
enum {
  NX_CONTRACT_A,
  NX_CONTRACT_B = NX_CONTRACT_A + NX_OPERAND_WORDS,
  NX_CONTRACT_INIT = NX_CONTRACT_B + NX_OPERAND_WORDS,
  NX_CONTRACT_Y,
  NX_CONTRACT_SI,
  NX_CONTRACT_SY = NX_CONTRACT_SI + 3,
  NX_CONTRACT_INIT_DTYPE = NX_CONTRACT_SY + 3,
  NX_CONTRACT_Y_DTYPE,
  NX_CONTRACT_ACC_DTYPE,
  NX_CONTRACT_FLAGS,
  NX_CONTRACT_BATCH,
  NX_CONTRACT_M,
  NX_CONTRACT_N,
  NX_CONTRACT_KGROUPS,
  NX_CONTRACT_K,
  NX_CONTRACT_TILES_M,
  NX_CONTRACT_TILES_N,
  NX_CONTRACT_MAIN,
  NX_CONTRACT_S1,
  NX_CONTRACT_S2,
  NX_CONTRACT_GROUPS,
  NX_CONTRACT_PARTIALS,
  NX_CONTRACT_TICKETS,
  NX_CONTRACT_PERM,
  NX_CONTRACT_DESC,
  NX_CONTRACT_NTILES,
  NX_CONTRACT_KPERM,
  NX_CONTRACT_KOFF,
  NX_CONTRACT_SBG,
  NX_CONTRACT_SYG,
  NX_CONTRACT_SIG,
  NX_CONTRACT_WORDS,
};

/* NX_CONTRACT_FLAGS' bits. */
enum {
  /* y's outputs store whole vectors: y of the accumulator's own dtype, its
     columns contiguous, its rows on 16-byte boundaries. */
  NX_CONTRACT_Y_VECTORS = 1,
  NX_CONTRACT_K_PAIRS = 2,
  /* An unsegmented SIMT call's first range of k sums from init, the others
     from +0, and init is not added after: with one range, each output is
     init then its products in increasing k, one fused multiply-add each. */
  NX_CONTRACT_INIT_FIRST = 4,
  /* An unsegmented SIMT call computes no tile whose outputs (i, j) all
     have i < j (LOWER) or all i > j (UPPER), j a complex column on a
     complex accumulator; the others' outputs are all stored. */
  NX_CONTRACT_LOWER = 8,
  NX_CONTRACT_UPPER = 16,
  /* A Contracting call's kperm holds int64 words: its contracted extent
     passes 2^31 - 1. */
  NX_CONTRACT_K_WIDE = 32,
};

enum {
  /* The terms a range of a SIMT kernel's split sum takes a whole number
     of. */
  NX_CONTRACT_RUN = 32,
  /* The terms a range of a matrix-unit kernel's takes a whole number of. */
  NX_WMMA_RUN = 128,
  /* gemv's rows at most, its threads, and the k of a's rows a group holds
     at once. Its dynamic LDS: a's rows, m x NX_GEMV_STRETCH elements of
     the compute type, rounded up to 16 bytes, then NX_GEMV_STAGE_* bytes a
     wave for b's values: a row of 16 values and 16 bytes a lane, of a
     compute type of 4 or 8 bytes. */
  NX_GEMV_ROWS = 8,
  NX_GEMV_THREADS = 256,
  NX_GEMV_STRETCH = 256,
  NX_GEMV_STAGE_4 = 32 * (16 * 4 + 16),
  NX_GEMV_STAGE_8 = 32 * (16 * 8 + 16),
};

/* contract_pack: [numel] elements in C order of [rank] axes, small where
   fewer than 2^32, each a record of NX_PACK_AXIS words from NX_PACK_AX:
   its extent and divisor, its strides [ss], [ds] and [cs], and [ratio],
   an axis's head (NX_AXIS_*). Element i of the walk lies at [src_first]
   plus i's indices times [ss] of [src], of [src_dtype], to [dst_first]
   plus its indices times [ds] of [dst], of [dst_dtype], positions in
   elements, over [groups] groups along x. A sub-byte element lies at bit position times its bits of its
   buffer, or one a byte (NX_PACK_SPREAD_). A raw pack copies an element's
   bits; NX_PACK_VALUE stores its value as a cast into [acc_dtype] holds
   it, times its scale where [scale] is not 0, rounded once into
   [acc_dtype]: [scale_dtype]'s element at [scale_first] plus the indices
   divided by [ratio] times [cs], or with NX_PACK_EXPONENT 2^(e - 127) of
   the code e, NaN of 255, then stored into [dst_dtype], which holds it
   (float16 or bfloat16 of a float8 value, an 8-bit integer of a 4-bit
   one). A complex value into a complex dtype stores both parts; with
   NX_PACK_BLOCK into a real one, the 2 x 2 block [[re, -im], [im, re]]
   from its position, the second row [block] elements on. */
enum {
  NX_PACK_AXES = 32,
  NX_PACK_THREADS = 256,
};

enum {
  NX_PACK_AXIS_SS = NX_AXIS_STRIDES,
  NX_PACK_AXIS_DS,
  NX_PACK_AXIS_CS,
  NX_PACK_AXIS_RATIO,
  NX_PACK_AXIS_RATIO_DIV,
  NX_PACK_AXIS,
};

enum {
  NX_PACK_SRC,
  NX_PACK_DST,
  NX_PACK_SCALE,
  NX_PACK_SRC_FIRST,
  NX_PACK_DST_FIRST,
  NX_PACK_SCALE_FIRST,
  NX_PACK_NUMEL,
  NX_PACK_RANK,
  NX_PACK_SRC_DTYPE,
  NX_PACK_DST_DTYPE,
  NX_PACK_SCALE_DTYPE,
  NX_PACK_MODE,
  NX_PACK_ACC_DTYPE,
  NX_PACK_BLOCK_WORD,
  NX_PACK_GROUPS,
  NX_PACK_SMALL,
  NX_PACK_AX,
  NX_PACK_WORDS = NX_PACK_AX + NX_PACK_AXES * NX_PACK_AXIS,
};

/* NX_PACK_MODE's bits: NX_PACK_VALUE, else a raw copy; an Exponent scale;
   a sub-byte source or destination one element a byte, as in the
   workspace; NX_PACK_CAST, a sum of the workspace's dtype (the
   accumulator's, or its compute type's) into [dst_dtype] as a cast from
   [acc_dtype] writes it; with NX_PACK_BLOCK, NX_PACK_CONJ blocks the
   value's conjugate and NX_PACK_FLIP negates the block's second row, the
   one a's imaginary parts multiply, so that the sums take a's conjugate;
   NX_PACK_NEG, a value negated, which is exact. */
enum {
  NX_PACK_VALUE = 1,
  NX_PACK_EXPONENT = 2,
  NX_PACK_SPREAD_SRC = 4,
  NX_PACK_SPREAD_DST = 8,
  NX_PACK_CAST = 16,
  NX_PACK_BLOCK = 32,
  NX_PACK_CONJ = 64,
  NX_PACK_FLIP = 128,
  NX_PACK_NEG = 256,
};

/* A segment prepass's words: the ids' address, the tables' (counts,
   cursor, perm, desc, ntiles; a word of no table holds 0); the ids'
   strides along batch and the segmented axis; the elements of a's axes
   after it ([inner]); the rows (Rows) or contracted indices (Contracting)
   a batch element holds ([count]), the segmented axis's extent, the first
   of the window's rows; the batch elements, the groups, a row tile's rows
   (Rows) or the parts of k (Contracting), the phase, whether perm holds
   int64 indices, and the groups the launch has. */
enum {
  NX_SEG_IDS,
  NX_SEG_COUNTS,
  NX_SEG_CURSOR,
  NX_SEG_PERM,
  NX_SEG_DESC,
  NX_SEG_NTILES,
  NX_SEG_SZ,
  NX_SEG_SX,
  NX_SEG_INNER,
  NX_SEG_COUNT,
  NX_SEG_EXTENT,
  NX_SEG_FIRST,
  NX_SEG_BATCH,
  NX_SEG_GROUPS,
  NX_SEG_BM,
  NX_SEG_PHASE,
  NX_SEG_WIDE,
  NX_SEG_GRID,
  NX_SEG_WORDS,
};

/* A prepass group's work-items. */
enum { NX_SEG_THREADS = 256 };

/* Sorts and numberings (sort.hip, whose header states the paths) */

#define NX_AMD_SORT_KERNELS(X)                                                \
  X(sort_block, NX_SORT_THREADS, NX_SORT_BLOCK_WORDS)                         \
  X(sort_hist, NX_SORT_THREADS, NX_SORT_HIST_WORDS)                           \
  X(sort_count8, NX_SORT_THREADS, NX_SORT_PASS_WORDS)                         \
  X(sort_count12, NX_SORT_THREADS, NX_SORT_PASS_WORDS)                        \
  X(sort_count24, NX_SORT_THREADS, NX_SORT_PASS_WORDS)                        \
  X(sort_scatter8, NX_SORT_THREADS, NX_SORT_PASS_WORDS)                       \
  X(sort_scatter12, NX_SORT_THREADS, NX_SORT_PASS_WORDS)                      \
  X(sort_scatter24, NX_SORT_THREADS, NX_SORT_PASS_WORDS)                      \
  X(sort_round, NX_SORT_THREADS, NX_SORT_SELECT_WORDS)                        \
  X(sort_gather, NX_SORT_THREADS, NX_SORT_SELECT_WORDS)                       \
  X(unique_count, NX_SORT_THREADS, NX_UNIQUE_WORDS)                           \
  X(unique_heads, NX_SORT_THREADS, NX_UNIQUE_WORDS)                           \
  X(unique_rank, NX_SORT_THREADS, NX_UNIQUE_WORDS)                            \
  X(unique_ids, NX_SORT_THREADS, NX_UNIQUE_WORDS)

enum {
  /* A group's work-items. */
  NX_SORT_THREADS = 256,
  /* The most axes of a slice beside the sorted one. */
  NX_SORT_RANK = NX_MAX_RANK - 1,
  /* The elements a group of sort_hist, sort_round or sort_gather reads. */
  NX_SORT_CHUNK = 16384,
  /* The longest slice sort_block sorts, and for 16-byte keys. */
  NX_SORT_CAP = 4096,
  NX_SORT_CAP16 = 2048,
  /* The most parts a pass cuts a slice's tiles into. */
  NX_SORT_PARTS = 256,
  /* The items a pass's work-item ranks: 8-, 12- and 24-byte items. */
  NX_SORT_ITEMS8 = 16,
  NX_SORT_ITEMS12 = 8,
  NX_SORT_ITEMS24 = 5,
  /* The bytes of a slice's select (sort.hip, sort_select). */
  NX_SORT_SELECT_BYTES = 2104,
  /* The rows or bit words a numbering's group takes at once. */
  NX_UNIQUE_TILE = 4096,
};

/* Key formats: how an element's bits become its key (sort.hip, Keys). */
enum {
  NX_SORT_UNSIGNED,
  NX_SORT_SIGNED,
  NX_SORT_FLOAT,
  NX_SORT_BOOL,
  NX_SORT_COMPLEX,
};

/* The bits of a sort's [flags]. */
enum {
  NX_SORT_VALUES = 1,     /* store the values */
  NX_SORT_ORDER = 2,      /* the first pass reads x through [order] */
  NX_SORT_CANDIDATES = 4, /* sort_block sorts each slice's candidates */
};

/* A key's format, from a kernel's index of its first word: an element of
   [ebits] bits by the format [kind] makes a key of [kbytes] bytes;
   [nan_abs] is a float's greatest magnitude not NaN, all ones for the
   others, [kmask] a real element's NaN key, [flip] and [flip_signed] the
   bits xored into a key whose sign is clear or set, [dmask_lo] and
   [dmask_hi] the bits xored into every key, all ones descending. */
enum {
  NX_SORT_KEY_KIND,
  NX_SORT_KEY_EBITS,
  NX_SORT_KEY_KBYTES,
  NX_SORT_KEY_NAN_ABS,
  NX_SORT_KEY_KMASK,
  NX_SORT_KEY_FLIP,
  NX_SORT_KEY_FLIP_SIGNED,
  NX_SORT_KEY_DMASK_LO,
  NX_SORT_KEY_DMASK_HI,
  NX_SORT_KEY_WORDS,
};

/* A sort's slices, from a kernel's index of its first word, its last
   words: [count] slices of [n] elements along the sorted axis, [n] an
   axis's head (NX_AXIS_*), small where they hold fewer than 2^32 elements
   in all, along which x, the values and the positions step [sx], [sv] and
   [sp] elements; the first elements of the first slice, [off_x], [off_v]
   and [off_p]; then [kept] axes' records, each its extent and divisor,
   then x's, the values' and the positions' strides, so that slices of
   [kept] axes use the first NX_SORT_SLICES_AXES + NX_SORT_SLICES_AXIS
   kept words. Offsets and strides count elements from the buffers'
   addresses. */
enum {
  NX_SORT_SLICES_N,
  NX_SORT_SLICES_N_DIV,
  NX_SORT_SLICES_SMALL,
  NX_SORT_SLICES_COUNT,
  NX_SORT_SLICES_SX,
  NX_SORT_SLICES_SV,
  NX_SORT_SLICES_SP,
  NX_SORT_SLICES_OFF_X,
  NX_SORT_SLICES_OFF_V,
  NX_SORT_SLICES_OFF_P,
  NX_SORT_SLICES_KEPT,
  NX_SORT_SLICES_AXES,
  NX_SORT_SLICES_AXIS = NX_AXIS_STRIDES + 3,
  NX_SORT_SLICES_WORDS = NX_SORT_SLICES_AXES + NX_SORT_SLICES_AXIS * NX_SORT_RANK,
};

/* sort_block: slices of at most [cap] elements, [per] of them a group,
   from x, or under NX_SORT_CANDIDATES each slice's candidates [cand] of
   its select [sel], whose positions take [pbytes] bytes, into [values]
   (under NX_SORT_VALUES) and [positions], [keep] of each slice's, over
   [groups] groups along x. */
enum {
  NX_SORT_BLOCK_X,
  NX_SORT_BLOCK_VALUES,
  NX_SORT_BLOCK_POSITIONS,
  NX_SORT_BLOCK_SEL,
  NX_SORT_BLOCK_CAND,
  NX_SORT_BLOCK_KEEP,
  NX_SORT_BLOCK_FLAGS,
  NX_SORT_BLOCK_GROUPS,
  NX_SORT_BLOCK_CAP,
  NX_SORT_BLOCK_PER,
  NX_SORT_BLOCK_PBYTES,
  NX_SORT_BLOCK_KEY,
  NX_SORT_BLOCK_SLICES = NX_SORT_BLOCK_KEY + NX_SORT_KEY_WORDS,
  NX_SORT_BLOCK_WORDS = NX_SORT_BLOCK_SLICES + NX_SORT_SLICES_WORDS,
};

/* sort_hist: each slice's count of each digit of its keys into [hist],
   zero before, made each digit's first place, and each slice's passes into
   [meta], [hblocks] groups a slice of NX_SORT_CHUNK elements, an axis's
   head (NX_AXIS_*), over [groups] groups along x, a ticket a slice from
   [tickets]. */
enum {
  NX_SORT_HIST_X,
  NX_SORT_HIST_HIST,
  NX_SORT_HIST_META,
  NX_SORT_HIST_TICKETS,
  NX_SORT_HIST_GROUPS,
  NX_SORT_HIST_HBLOCKS,
  NX_SORT_HIST_HBLOCKS_DIV,
  NX_SORT_HIST_KEY,
  NX_SORT_HIST_SLICES = NX_SORT_HIST_KEY + NX_SORT_KEY_WORDS,
  NX_SORT_HIST_WORDS = NX_SORT_HIST_SLICES + NX_SORT_SLICES_WORDS,
};

/* sort_count<b> and sort_scatter<b>, a pass over key byte [pass] of items
   of b bytes: each slice's [tiles] tiles cut into [parts] parts of [span]
   tiles, [parts] an axis's head (NX_AXIS_*), over [groups] groups along
   x. sort_count counts each part's
   digits into [counts], then the slice's last part to take its ticket
   from [tickets] makes them each part's first places, from [hist];
   sort_scatter moves each part's items there, from x (through [order]
   under NX_SORT_ORDER) or [buf0] or [buf1], to [buf0] or [buf1], or, in
   the slice's last pass, into [values] and [positions], [keep] of each
   slice's. Each slice's passes are [meta]'s. */
enum {
  NX_SORT_PASS_X,
  NX_SORT_PASS_VALUES,
  NX_SORT_PASS_POSITIONS,
  NX_SORT_PASS_TICKETS,
  NX_SORT_PASS_ORDER,
  NX_SORT_PASS_HIST,
  NX_SORT_PASS_META,
  NX_SORT_PASS_COUNTS,
  NX_SORT_PASS_BUF0,
  NX_SORT_PASS_BUF1,
  NX_SORT_PASS_KEEP,
  NX_SORT_PASS_FLAGS,
  NX_SORT_PASS_PASS,
  NX_SORT_PASS_GROUPS,
  NX_SORT_PASS_TILES,
  NX_SORT_PASS_PARTS,
  NX_SORT_PASS_PARTS_DIV,
  NX_SORT_PASS_SPAN,
  NX_SORT_PASS_KEY,
  NX_SORT_PASS_SLICES = NX_SORT_PASS_KEY + NX_SORT_KEY_WORDS,
  NX_SORT_PASS_WORDS = NX_SORT_PASS_SLICES + NX_SORT_SLICES_WORDS,
};

/* sort_round and sort_gather, a select of at most [keep] elements of each
   slice: round [pass] counts digit [pass] of (key, position), positions
   of [pbytes] bytes, into each slice's select [sel], zero before its
   first round, the slice's last group to take its ticket from [tickets]
   extending its prefix, until at most [cap] elements lie below or match
   it; sort_gather copies those into each slice's [cap] candidates
   [cand]. [hblocks] groups a slice of NX_SORT_CHUNK elements, an axis's
   head (NX_AXIS_*), over [groups] groups along x. */
enum {
  NX_SORT_SELECT_X,
  NX_SORT_SELECT_SEL,
  NX_SORT_SELECT_TICKETS,
  NX_SORT_SELECT_KEEP,
  NX_SORT_SELECT_PASS,
  NX_SORT_SELECT_CAND,
  NX_SORT_SELECT_GROUPS,
  NX_SORT_SELECT_HBLOCKS,
  NX_SORT_SELECT_HBLOCKS_DIV,
  NX_SORT_SELECT_CAP,
  NX_SORT_SELECT_PBYTES,
  NX_SORT_SELECT_KEY,
  NX_SORT_SELECT_SLICES = NX_SORT_SELECT_KEY + NX_SORT_KEY_WORDS,
  NX_SORT_SELECT_WORDS = NX_SORT_SELECT_SLICES + NX_SORT_SLICES_WORDS,
};

/* The bits of a numbering's [flags]. */
enum {
  NX_UNIQUE_BY_ORDER = 1, /* the rows are sorted into [order] */
  NX_UNIQUE_BY_SKEYS = 2, /* their one word sorted into [skeys] */
};

/* A numbering of [n] rows of [w] words, word j of row i at keys[i s0 + j
   s1], after a stable sort of the rows: [ids], [first] and [counts] of
   [m] entries, and [count]. The rows are cut into parts of [span] rows,
   and their bit words into parts of [wspan] words, over [groups] groups
   along x. The workspace holds each sorted row's key in sorted order
   [gid], each key's first row [gfirst] and first sorted row [gstart], a
   bit at each first row [bits], zero before, and the bits set before each
   word [below], each part's first key [heads] and its bits set [set],
   zero before, and the keys counted [k]. unique_count and unique_heads
   take a ticket from [tickets]. */
enum {
  NX_UNIQUE_IDS,
  NX_UNIQUE_FIRST,
  NX_UNIQUE_COUNTS,
  NX_UNIQUE_COUNT,
  NX_UNIQUE_M,
  NX_UNIQUE_KEYS,
  NX_UNIQUE_SKEYS,
  NX_UNIQUE_W,
  NX_UNIQUE_S0,
  NX_UNIQUE_S1,
  NX_UNIQUE_HEADS,
  NX_UNIQUE_TICKETS,
  NX_UNIQUE_SPAN,
  NX_UNIQUE_ORDER,
  NX_UNIQUE_GID,
  NX_UNIQUE_GFIRST,
  NX_UNIQUE_GSTART,
  NX_UNIQUE_K,
  NX_UNIQUE_FLAGS,
  NX_UNIQUE_BITS,
  NX_UNIQUE_BELOW,
  NX_UNIQUE_SET,
  NX_UNIQUE_N,
  NX_UNIQUE_GROUPS,
  NX_UNIQUE_WSPAN,
  NX_UNIQUE_WORDS,
};

/* Factorisations (factor.hip)

   Matrices of a buffer: element (i, j) of batch element z at the address
   word's byte plus z, i and j times their strides, counted in elements of
   the routine's format. A routine's own matrices are C-contiguous. An
   address word that is 0 names no matrix. Each kernel strides over its
   units along x, [groups] groups of them. */

#define NX_AMD_FACTOR_KERNELS(X)                                              \
  X(la_trsm_f32, NX_LA_TRSM_THREADS, NX_LA_TRSM_WORDS)                        \
  X(la_trsm_f64, NX_LA_TRSM_THREADS, NX_LA_TRSM_WORDS)                        \
  X(la_trsm_c64, NX_LA_TRSM_THREADS, NX_LA_TRSM_WORDS)                        \
  X(la_trsm_c128, NX_LA_TRSM_THREADS, NX_LA_TRSM_WORDS)                       \
  X(la_lu_panel_f32, NX_LA_THREADS, NX_LA_PANEL_WORDS)                        \
  X(la_lu_panel_f64, NX_LA_THREADS, NX_LA_PANEL_WORDS)                        \
  X(la_lu_panel_c64, NX_LA_THREADS, NX_LA_PANEL_WORDS)                        \
  X(la_lu_panel_c128, NX_LA_THREADS, NX_LA_PANEL_WORDS)                       \
  X(la_chol_f32, NX_LA_THREADS, NX_LA_CHOL_WORDS)                             \
  X(la_chol_f64, NX_LA_THREADS, NX_LA_CHOL_WORDS)                             \
  X(la_chol_c64, NX_LA_THREADS, NX_LA_CHOL_WORDS)                             \
  X(la_chol_c128, NX_LA_THREADS, NX_LA_CHOL_WORDS)                            \
  X(la_qr_panel_f32, NX_LA_THREADS, NX_LA_QR_WORDS)                           \
  X(la_qr_panel_f64, NX_LA_THREADS, NX_LA_QR_WORDS)                           \
  X(la_qr_panel_c64, NX_LA_THREADS, NX_LA_QR_WORDS)                           \
  X(la_qr_panel_c128, NX_LA_THREADS, NX_LA_QR_WORDS)                          \
  X(la_gather, NX_LA_THREADS, NX_LA_GATHER_WORDS)                             \
  X(la_chol_finish, NX_LA_THREADS, NX_LA_FINISH_WORDS)                        \
  X(la_diag_zero, NX_LA_THREADS, NX_LA_DIAG_WORDS)                            \
  X(la_nan_where, NX_LA_THREADS, NX_LA_DIAG_WORDS)                            \
  X(la_qr_out, NX_LA_THREADS, NX_LA_QR_OUT_WORDS)                             \
  X(la_dense_v, NX_LA_THREADS, NX_LA_DENSE_WORDS)                             \
  X(la_iota, NX_LA_THREADS, NX_LA_IOTA_WORDS)

enum {
  /* A routine's formats: the [fmt] word of a kernel of every format. */
  NX_LA_F32,
  NX_LA_F64,
  NX_LA_C64,
  NX_LA_C128,
};

enum {
  /* The threads of a group of every factorisation kernel but trsm. */
  NX_LA_THREADS = 256,
  /* trsm's threads, a column of x each. */
  NX_LA_TRSM_THREADS = 64,
  /* The most rows of trsm at 4 and 8 bytes an element, half at 16: a
     thread's column fills at most 32 KiB of LDS a group. */
  NX_LA_LEAF = 64,
  /* The bytes of an LU meeting slot before its row: the candidate. */
  NX_LA_SLOT_HEAD = 32,
  /* QR's sums over rows: blocks of NX_LA_QBLK rows from row 0. */
  NX_LA_QBLK = 256,
  /* qr_panel's stages with P > 1: a step's first and second launch, and
     the Gram pairs' launches. */
  NX_LA_QR_STEP_A = 1,
  NX_LA_QR_STEP_B = 2,
  NX_LA_QR_STEP_G = 3,
};

/* trsm: x <- op(l)^-1 x for the lower triangular n x n view op(l), n at
   most NX_LA_LEAF (half at complex128), and the n x r view x of each of
   [matrices], a thread a column of x: op(l) conjugated under [conj], its
   diagonal taken as ones under [unit]. [src] (or 0) is b, read in place of
   x with x's strides; with [whole], a zero on the diagonal makes the
   column NaN. */
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
  NX_LA_TRSM_MATRICES,
  NX_LA_TRSM_CONJ,
  NX_LA_TRSM_UNIT,
  NX_LA_TRSM_WHOLE,
  NX_LA_TRSM_GROUPS,
  NX_LA_TRSM_WORDS,
};

/* lu_panel: LU of the panel of columns [c0, c0 + w) of rows [c0, m) of
   each matrix of [a] (rows lda apart, matrices sbat apart), in place,
   pivots at [piv] (spiv a matrix). G threads a matrix (16, 32 or the
   group), [mats] matrices a group, and P groups a matrix, group q its rows
   [q per, (q + 1) per); with P > 1, one launch of min(m - c0, w) + 1 a
   step, [k] the launch's, the groups passing candidates through [slots]
   (slots_stride bytes a matrix, the second step parity's from byte
   [slots_parity]). [src_of] (m - c0 a matrix) receives each
   position's row for gather; with [src], the operand read in place of a,
   [perm] (sperm a matrix) receives it instead. A matrix's tile is in LDS
   where [in_lds], else at [spill], spill_bytes a matrix. LDS holds the
   candidates from byte 0, the second step parity's from byte
   [cands_parity], with P > 1 the pivot row from byte [row], and
   from byte [head] the matrices' tiles, spill_bytes apart; a tile's
   positions start [pos] bytes in. */
enum {
  NX_LA_PANEL_A,
  NX_LA_PANEL_PIV,
  NX_LA_PANEL_SRC_OF,
  NX_LA_PANEL_SRC,
  NX_LA_PANEL_PERM,
  NX_LA_PANEL_SPILL,
  NX_LA_PANEL_SLOTS,
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
  NX_LA_PANEL_K,
  NX_LA_PANEL_GROUPS,
  NX_LA_PANEL_WORDS,
};

/* chol: Cholesky of the diagonal block of order nb at (k, k) of each
   matrix, in place, element (i, j) at i lda + j cs, G threads a matrix and
   [mats] matrices a group, rows ld apart in LDS and matrices stride bytes
   apart; [fail], a 32-bit word a
   matrix, set where it fails; with [src], the whole matrix read from it
   and stored whole, its Lower factor or, under [upper], the Upper one. */
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
  NX_LA_CHOL_GROUPS,
  NX_LA_CHOL_WORDS,
};

/* qr_panel: Householder QR of the panel of columns [c0, c0 + w) of rows
   [c0, m) of each matrix of [a], in place (factor.hip states its sums'
   order), tau at [tau] (stau a matrix); dense V at [v] (sv a matrix) and T
   at [t] (st a matrix, rows ldto apart) where not 0. Where the panel is
   the whole matrix, [q] (m x qc) and [r] (qc x n) receive Q's first qc
   columns and R. G threads a matrix and [mats] matrices a group; with
   P > 1, launches of [stage] and [k]; [parts] holds a matrix's partials
   (parts_stride elements) where not in LDS, [spill] its tile where not in
   LDS. A matrix's LDS, [stride] bytes from the group's byte 0 for each
   matrix before it, holds its sums' values from byte 0 and from bytes
   [tc], [ts], [gram] and [mx] a step's w_c, T, the Gram pairs and M,
   the partials from byte [parts_at] and the tile from byte [tile_at]
   where they are in LDS. The partials hold a sum's blocks, the second
   parity's from element [store], step j's row of R from element [drow]
   and, with P > 1, the pairs from element [gstore]. */
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
  NX_LA_QR_STAGE,
  NX_LA_QR_K,
  NX_LA_QR_GROUPS,
  NX_LA_QR_WORDS,
};

/* gather: an LU panel's row interchanges on each matrix's columns outside
   it and on perm (sperm a matrix), elements of [fmt]. */
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
  NX_LA_GATHER_MATRICES,
  NX_LA_GATHER_FMT,
  NX_LA_GATHER_GROUPS,
  NX_LA_GATHER_WORDS,
};

/* chol_finish: each n x n matrix NaN where [fail], else its other
   triangle +0 and, under [upper], the conjugate transpose of its lower
   one; under [tr], the matrix holds the factor's transpose. */
enum {
  NX_LA_FINISH_A,
  NX_LA_FINISH_FAIL,
  NX_LA_FINISH_N,
  NX_LA_FINISH_MATRICES,
  NX_LA_FINISH_UPPER,
  NX_LA_FINISH_TR,
  NX_LA_FINISH_FMT,
  NX_LA_FINISH_GROUPS,
  NX_LA_FINISH_WORDS,
};

/* diag_zero and nan_where: a triangular solve's diagonal, [sd] apart:
   [flag], a 32-bit word a matrix, set where a diagonal element of a is
   zero; then each of x's xn elements NaN where it is set. */
enum {
  NX_LA_DIAG_A,
  NX_LA_DIAG_X,
  NX_LA_DIAG_FLAG,
  NX_LA_DIAG_SA,
  NX_LA_DIAG_SD,
  NX_LA_DIAG_N,
  NX_LA_DIAG_SX,
  NX_LA_DIAG_XN,
  NX_LA_DIAG_MATRICES,
  NX_LA_DIAG_FMT,
  NX_LA_DIAG_GROUPS,
  NX_LA_DIAG_WORDS,
};

/* qr_out: q, the first qc columns of the m x m identity, and r, the upper
   triangle of f's first rows, rows past m zero; f 0 for a matrix of no
   column. */
enum {
  NX_LA_QR_OUT_F,
  NX_LA_QR_OUT_Q,
  NX_LA_QR_OUT_R,
  NX_LA_QR_OUT_M,
  NX_LA_QR_OUT_N,
  NX_LA_QR_OUT_QC,
  NX_LA_QR_OUT_MATRICES,
  NX_LA_QR_OUT_FMT,
  NX_LA_QR_OUT_GROUPS,
  NX_LA_QR_OUT_WORDS,
};

/* dense_v: the dense V of f's columns [c0, c0 + w), rows [c0, m). */
enum {
  NX_LA_DENSE_F,
  NX_LA_DENSE_V,
  NX_LA_DENSE_M,
  NX_LA_DENSE_N,
  NX_LA_DENSE_C0,
  NX_LA_DENSE_W,
  NX_LA_DENSE_MATRICES,
  NX_LA_DENSE_FMT,
  NX_LA_DENSE_GROUPS,
  NX_LA_DENSE_WORDS,
};

/* iota: perm[g] = g mod m for g < total. */
enum {
  NX_LA_IOTA_PERM,
  NX_LA_IOTA_M,
  NX_LA_IOTA_TOTAL,
  NX_LA_IOTA_GROUPS,
  NX_LA_IOTA_WORDS,
};

/* Spectral routines (spectral.hip) */

#define NX_AMD_SPECTRAL_KERNELS(X)                                            \
  X(sp_eigh_jacobi_f32, NX_SP_THREADS, NX_SP_PLAIN)                           \
  X(sp_eigh_jacobi_f64, NX_SP_THREADS, NX_SP_PLAIN)                           \
  X(sp_eigh_jacobi_c64, NX_SP_THREADS, NX_SP_PLAIN)                           \
  X(sp_eigh_jacobi_c128, NX_SP_THREADS, NX_SP_PLAIN)                          \
  X(sp_svd_jacobi_f32, NX_SP_THREADS, NX_SP_PLAIN)                            \
  X(sp_svd_jacobi_f64, NX_SP_THREADS, NX_SP_PLAIN)                            \
  X(sp_svd_jacobi_c64, NX_SP_THREADS, NX_SP_PLAIN)                            \
  X(sp_svd_jacobi_c128, NX_SP_THREADS, NX_SP_PLAIN)                           \
  X(sp_td_stage, NX_SP_THREADS, NX_SP_PLAIN)                                  \
  X(sp_td_step_f32, NX_SP_THREADS, NX_SP_PLAIN)                               \
  X(sp_td_step_f64, NX_SP_THREADS, NX_SP_PLAIN)                               \
  X(sp_td_step_c64, NX_SP_THREADS, NX_SP_PLAIN)                               \
  X(sp_td_step_c128, NX_SP_THREADS, NX_SP_PLAIN)                              \
  X(sp_td_hemv, NX_SP_THREADS, NX_SP_PLAIN)                                   \
  X(sp_td_diag, NX_SP_THREADS, NX_SP_PLAIN)                                   \
  X(sp_td_vcopy, NX_SP_THREADS, NX_SP_PLAIN)                                  \
  X(sp_larft, NX_SP_WAVE, NX_SP_PLAIN)                                        \
  X(sp_td_zinit, NX_SP_THREADS, NX_SP_PLAIN)                                  \
  X(sp_td_out, NX_SP_THREADS, NX_SP_PLAIN)                                    \
  X(sp_dc_leaves, NX_SP_THREADS, NX_SP_PLAIN)                                 \
  X(sp_dc_scatter, NX_SP_THREADS, NX_SP_PLAIN)                                \
  X(sp_dc_deflate_f32, NX_SP_THREADS, NX_SP_PLAIN)                            \
  X(sp_dc_deflate_f64, NX_SP_THREADS, NX_SP_PLAIN)                            \
  X(sp_dc_secular_f32, NX_SP_HALF, NX_SP_PLAIN)                               \
  X(sp_dc_secular_f64, NX_SP_HALF, NX_SP_PLAIN)                               \
  X(sp_dc_product, NX_SP_THREADS, NX_SP_PLAIN)                                \
  X(sp_dc_sort, NX_SP_THREADS, NX_SP_PLAIN)                                   \
  X(sp_tri_bisect, NX_SP_HALF, NX_SP_PLAIN)                                   \
  X(sp_bd_stage, NX_SP_THREADS, NX_SP_WORDS)                                  \
  X(sp_bd_left_f32, NX_SP_THREADS, NX_SP_WORDS)                               \
  X(sp_bd_left_f64, NX_SP_THREADS, NX_SP_WORDS)                               \
  X(sp_bd_left_c64, NX_SP_THREADS, NX_SP_WORDS)                               \
  X(sp_bd_left_c128, NX_SP_THREADS, NX_SP_WORDS)                              \
  X(sp_bd_wh, NX_SP_THREADS, NX_SP_WORDS)                                     \
  X(sp_bd_updl, NX_SP_THREADS, NX_SP_WORDS)                                   \
  X(sp_bd_right_f32, NX_SP_THREADS, NX_SP_WORDS)                              \
  X(sp_bd_right_f64, NX_SP_THREADS, NX_SP_WORDS)                              \
  X(sp_bd_right_c64, NX_SP_THREADS, NX_SP_WORDS)                              \
  X(sp_bd_right_c128, NX_SP_THREADS, NX_SP_WORDS)                             \
  X(sp_bd_wn, NX_SP_HALF, NX_SP_WORDS)                                        \
  X(sp_bd_updr, NX_SP_THREADS, NX_SP_WORDS)                                   \
  X(sp_bd_tgk, NX_SP_THREADS, NX_SP_WORDS)                                    \
  X(sp_bd_values, NX_SP_THREADS, NX_SP_WORDS)                                 \
  X(sp_bd_ginit, NX_SP_THREADS, NX_SP_WORDS)                                  \
  X(sp_bd_gen_f32, NX_SP_WAVE, NX_SP_WORDS)                                   \
  X(sp_bd_gen_f64, NX_SP_WAVE, NX_SP_WORDS)                                   \
  X(sp_bd_gen_c64, NX_SP_WAVE, NX_SP_WORDS)                                   \
  X(sp_bd_gen_c128, NX_SP_WAVE, NX_SP_WORDS)                                  \
  X(sp_bd_apply, NX_SP_HALF, NX_SP_WORDS)                                     \
  X(sp_bd_xinit, NX_SP_THREADS, NX_SP_WORDS)                                  \
  X(sp_bd_vcopy, NX_SP_THREADS, NX_SP_WORDS)                                  \
  X(sp_bd_rank, NX_SP_THREADS, NX_SP_WORDS)                                   \
  X(sp_bd_out, NX_SP_THREADS, NX_SP_WORDS)                                    \
  X(sp_eig_schur_f32, NX_SP_HALF, NX_SP_PLAIN)                                \
  X(sp_eig_schur_f64, NX_SP_EIG_DOUBLE, NX_SP_PLAIN)                          \
  X(sp_eig_schur_c64, NX_SP_HALF, NX_SP_PLAIN)                                \
  X(sp_eig_schur_c128, NX_SP_EIG_DOUBLE, NX_SP_PLAIN)                         \
  X(sp_eig_norms, NX_SP_HALF, NX_SP_PLAIN)                                    \
  X(sp_eig_vectors_f32, NX_SP_HALF, NX_SP_PLAIN)                              \
  X(sp_eig_vectors_f64, NX_SP_HALF, NX_SP_PLAIN)                              \
  X(sp_eig_vectors_c64, NX_SP_HALF, NX_SP_PLAIN)                              \
  X(sp_eig_vectors_c128, NX_SP_HALF, NX_SP_PLAIN)                             \
  X(sp_eig_finish, NX_SP_HALF, NX_SP_PLAIN)                                   \
  X(sp_eye, NX_SP_THREADS, NX_SP_PLAIN)

/* A spectral step reads its parameters from words [0, NX_SP_FMT) in order,
   each an address, an integer or a size, as spectral.hip's entries name
   them; then the format of a kernel of every format, the groups along
   each axis and a group's work-items; the bytes of each matrix's
   workspace, or Jacobi's matrices a group and its bytes of one; the
   iterations' limits: Jacobi's sweeps, a secular equation's steps, eig's
   QR iterations an eigenvalue over max(10, h), h the order balancing
   leaves, and the bidiagonal QR's rotations over k²: NX_SP_PLAIN words in
   all. Past them, which only a large svd's steps (sp_bd.h) read and
   stage, its workspace's layout: the byte offsets of its arrays in a
   matrix's bytes. The host lays out every workspace and passes its
   offsets. */
enum {
  NX_SP_FMT = 12,
  NX_SP_GX,
  NX_SP_GY,
  NX_SP_GZ,
  NX_SP_THREADS_OF,
  NX_SP_PER,
  NX_SP_SPAN,
  NX_SP_SWEEPS,
  NX_SP_STEPS,
  NX_SP_QR,
  NX_SP_ROTATIONS,
  NX_SP_PLAIN,
  NX_SP_BD_G = NX_SP_PLAIN,
  NX_SP_BD_X,
  NX_SP_BD_Y,
  NX_SP_BD_VC,
  NX_SP_BD_TM,
  NX_SP_BD_GG,
  NX_SP_BD_W1,
  NX_SP_BD_W2,
  NX_SP_BD_WV,
  NX_SP_BD_TAUQ,
  NX_SP_BD_TAUP,
  NX_SP_BD_D,
  NX_SP_BD_E,
  NX_SP_BD_UB,
  NX_SP_BD_VB,
  NX_SP_BD_REC,
  NX_SP_BD_SC,
  NX_SP_BD_SW,
  NX_SP_BD_ST,
  NX_SP_BD_PER,
  NX_SP_BD_CAP,
  NX_SP_WORDS,
};

/* The formats a step computes in, the word NX_SP_FMT of a kernel of
   every format. */
enum { NX_SP_F32, NX_SP_F64, NX_SP_C64, NX_SP_C128 };

enum {
  /* A group's work-items: most steps', a half and a wave's; eig's Schur
     step past order 32 at float64 and complex128. */
  NX_SP_THREADS = 256,
  NX_SP_HALF = 128,
  NX_SP_WAVE = 32,
  NX_SP_EIG_DOUBLE = 64,
  /* The tridiagonal reduction's panel; divide and conquer's leaves; the
     bidiagonal reflectors' blocks. */
  NX_SP_TNB = 32,
  NX_SP_LEAF = 32,
  NX_SP_BD_NB = 32,
  /* Jacobi's orders: eigh of order at most this, svd of both extents at
     most this. */
  NX_SP_JACOBI = 32,
};

/* A large svd's state words a matrix, from NX_SP_BD_ST. */
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
  NX_SP_ST_WORDS,
};

#endif
