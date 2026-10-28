/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Reductions and scans of one operand by Sum, Prod, Max, Min, Logsumexp
   and Arg, and by Moments.

   Association. An output's terms are numbered t = 0, 1, … in C order of the
   reduced axes' indices. They fall into blocks of NX_CPU_LANE_BLOCK
   consecutive terms and, within a block, into NX_CPU_LANES lanes from the
   identity, term t into lane t mod 16. Lane i takes lane i + 8 for i < 8,
   then lane i + 4, i + 2 and i + 1, but not a lane holding no term: lane 0
   is the block's value. A complex product's lanes start from their first
   term: 1 + 0i times a term would change its signed zeros and make NaN
   parts from infinite ones, where a real 1 changes nothing. The blocks
   combine by the binary tree whose left part holds the largest power of
   two of blocks below their count. A binary counter computes it in one
   pass: it combines the two newest values once per trailing zero of the
   number of blocks done, then what remains, newest first.

   A scan cuts each slice along its axis into chunks of SCAN_CHUNK terms from
   its start. A chunk's total is its terms reduced in the order above. The
   carry into chunk 0 is the slice's given carry, else the identity, but a
   complex product's chunk 0 given none has none and starts from its first
   term; the carry into chunk c + 1 is the
   carry into c combined with c's total, and result i of chunk c is the
   carry combined with the chunk's terms up to i, left to right.

   Only float sums and float and complex products depend on this order:
   integers wrap and extremes are exact. Neither the layout, the path nor
   the thread count changes it. A thread takes a group of blocks, a power
   of two of them from a multiple of that power, and such a group's value
   is a subtree of the blocks' tree: the groups' values combine by the same
   counter.

   NaN. A float result that a NaN term reaches, or that the arithmetic
   makes from numbers, is the NaN the instructions give: its sign and
   payload are not fixed.

   Paths. The operand and the destination, read at the operand's shape with
   step 0 along the reduced axes, coalesce: reduced and kept axes never
   merge, and the axes keep their order. A unit of work reduces W outputs:
   - one output, its runs of terms along the innermost reduced axis into
     lanes, where a reduced axis steps least through the operand;
   - W outputs along the kept axis that steps least, each term a row of W
     elements combined into one of 16 accumulator rows, elsewhere and where
     outputs have few terms. A 2x2 window's maximum takes this path.

   Stage. A monoid's terms reduce in their carrier (cpu.h): a narrow
   float's in float32, int4's and uint4's in int8 and uint8, bit's in bool.
   A narrow operand's terms are converted into it as the paths read them,
   at most 32 KiB at a time, which fits an L1 data cache of 32 KiB or more:
   as many of one output's full blocks as that holds, a block of it, or 16
   terms of a row's outputs; a network scan's operand is converted whole
   first. Arg, Logsumexp and Moments reduce in the carrier too.
   Results go into the destination where it is of the carrier, else into a
   carrier copy of it, rounded into it once at the end, as a cast does.

   States. Logsumexp, Arg and Moments reduce by the same blocks, tree and
   paths, an output's value a state of its terms in place of a monoid's
   element: a kernel makes a block's state from its terms, contiguous in
   the operand or copied so, and merges two states as the tree combines
   values. Logsumexp's and Moments' first pass finds each output's center,
   which the second takes. A Moments unit runs both in turn, the second
   over terms the first left in the caches, but where an output's blocks
   fall into groups on several threads, whose first passes all end before
   any second starts. Logsumexp's passes, whose time is their exp's,
   always run so. Arg reads a staged operand in the carrier too; its
   extreme into the operand's dtype is the term at its position, copied.

   Arg's scans. An Arg scan cuts the same chunks and carries a state per
   slice: a chunk's total is its terms' state from none, the carry into
   chunk c + 1 the carry into c merged with c's total, and result i of
   chunk c the carry taken on through the chunk's terms up to i. A unit's
   sequences, slices side by side or one slice's chunks, are the kernel's
   lanes, read and written in place, but a staged operand's, which go
   through rows of the carrier. Arg is exact, so its chunk totals are its
   reduction's.

   Programs. A reduction of a program's output reads its terms through
   the program: each place that reads terms asks map.c's evaluator for
   them (nx_cpu_prog), in the output's held dtype, converted into the
   carrier as a narrow operand's are, so the paths that stage a narrow
   operand read a program's output too. Each term is the output there as
   a map computes it, and the order is a function of the shapes alone, so
   the reduction has the bits of the output mapped into an array, then
   reduced. A Sum of a lane or fewer terms an output, each the product of
   two loads at float32 or float64, as a depthwise convolution's, reads
   the loads itself: the table's products multiply and sum them in
   registers, in the lanes' order.

   Networks. A Logsumexp or Moments scan computes, on each slice, the
   network of nx_kernel.mli's scans section over nx_cpu.mli's states, op by
   op with the kinds' arithmetic, so that every path and thread count gives
   the same bits. It holds O(log n) states per slice. Slices few for the
   threads are cut into tiles of 256 terms, the threads computing their
   states, then the network over them, then their results: the same
   combinations in the same order (TILES).

   The kernels and their table are in reduce_body.h; the paths and the
   entries follow. */

#include "cpu.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nx_kinds.h"

#include "reduce_body.h"

#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/mlvalues.h>


#define LANES NX_CPU_LANES
#define SCAN_CHUNK 4096

/* A streaming unit's row of outputs: 512 bytes, 128 float32. Byte buffers
   hold elements of the dtype, read and written as such, so they are aligned
   as the widest. */
#define ROW 512

/* A monoid scan's row of outputs on one thread, where its outputs lie side
   by side: 16 KiB, so that a unit sweeps rows of terms as long runs. A
   scan of 1024x4096 float32 down its columns took 3.75 ms on one M1 core
   in units of ROW, whose 512-byte runs lie 16 KiB apart, and 0.56 ms in
   these; on ten threads, units of ROW took 0.64 ms and of 410 outputs
   0.80. */
#define SCAN_ROW 16384

/* The rows ahead whose terms a scan of a unit's row of outputs
   prefetches, and on ARM its results: its runs of at most 512 bytes lie a
   row of the operand apart, which the hardware fetches late. On ten M1
   threads, scan-sum-f32-1024x4096-axis0 took about 545 us without, about
   370 with both at 2, 4 or 8 rows ahead, 460 at 16, and the terms' alone
   gained nothing. On fudge's six threads, the terms' alone took it from
   308 to 296 us, and with the results' to 353. */
#define SCAN_AHEAD 8

/* The bytes of the widest element, a complex128. */
#define WIDEST 16



/* Fewer terms per output take the streaming path. */
#define FEW 16

/* A scan's slices a unit runs side by side where its axis steps least. */
#define SLICES 4

/* Full blocks whose values the table computes in one call. */
#define RUN 64

/* The outputs a unit of a sum of products aims for: a 3x3 depthwise
   convolution of one 56x56x64 image took 165 us on one fudge core in
   units of 1024, 147 in these. */
#define PRODUCTS_UNIT 8192

/* Units the groups of blocks aim for where outputs are few. */
#define UNITS 64

/* What a call computes: a reduction, a scan, or a scan by nx_kernel.mli's
   network, Logsumexp's or Moments'. */
typedef enum { REDUCE, SCAN, NETWORK } task;

/* A program's output as the terms (Programs): output [k] of [prog], bound
   to its [nloads] loads [loads], read at virtual positions, a term's place
   in C order of the loop's shape, its loads then, where it reads it, the
   index placed by [pl] (cpu.h, Places). [rows] has a row for each load and
   four more, for the loops that coalesce the loads with the destinations,
   the operand and a carry. Its tables follow it in one block of
   fused_bytes: the loads, [pl]'s steps after the virtual operand's, their
   first positions, then [rows]. */
typedef struct {
  const nx_cpu_prog *prog;
  nx_array *loads;
  int k, nloads;
  nx_cpu_places pl;
  int64_t (*rows)[NX_MAX_RANK];
} fused;

/* The bytes of a fused of [nl] loads with its tables. */
static int64_t fused_bytes(int nl) {
  return (int64_t)(sizeof(fused) + (size_t)nl * sizeof(nx_array) +
                   (size_t)(2 * nl + 6) * NX_MAX_RANK * sizeof(int64_t) +
                   (size_t)(nl + 1) * sizeof(int64_t));
}

/* Lays out [z]'s tables in its block, for [nl] loads. */
static void fused_tables(fused *z, int nl) {
  z->loads = (nx_array *)(z + 1);
  int64_t (*place)[NX_MAX_RANK] = (int64_t (*)[NX_MAX_RANK])(z->loads + nl);
  z->pl.step = place + 1;
  z->pl.first = (const int64_t *)(place + nl + 2);
  z->rows = (int64_t (*)[NX_MAX_RANK])((int64_t *)(place + nl + 2) + nl + 1);
}

/* An output's value is an element of the dtype under a monoid, a state
   under a stateful reduction. */
typedef struct {
  const nx_cpu_monoid *f;   /* a monoid's kernels, or NULL */
  const nx_cpu_pass *r;     /* a stateful reduction's pass, or NULL */
  const nx_cpu_pass *center; /* the pass each unit runs before [r], or NULL */
  const nx_cpu_scan_pass *z; /* an Arg scan's kernels, or NULL */
  double *centers;          /* its outputs' centers, a row of them a unit */
  int centering;            /* the pass stores the centers */
  int w;
  int first;                /* lanes, rows and slices start from their
                               first term: a complex product's */
  int carrier, staged;      /* the dtype terms reduce in; src is not in it */
  int own;                  /* Arg's extreme is of src's dtype, not the
                               carrier's: its term, copied */
  int arg;                  /* an Arg, whose chunk totals are reductions */
  int64_t size;             /* the bytes of an output's value */
  _Alignas(16) uint8_t id[WIDEST];
  /* The destinations, the second the first again for one, and the operand
     as loaded, or a program's virtual one, of no buffer, read through
     [z]. */
  const nx_array *dst, *dst1, *src;
  const fused *fused;
  /* A scan's carries, of the carrier, or NULL: the destination's shape but
     one along its axis. */
  const nx_array *cy;
  /* The reduced axes in C order: extents and the operand's steps. */
  int nr;
  int64_t re[NX_MAX_RANK], rs[NX_MAX_RANK];
  /* The kept axes, the row axis last: extents, the operand's and the
     destinations' steps. */
  int nk;
  int64_t ke[NX_MAX_RANK], ks[NX_MAX_RANK], kd[NX_MAX_RANK], kd1[NX_MAX_RANK];
  int64_t kc[NX_MAX_RANK]; /* the carries' steps */
  int64_t x0, d0, d1, c0;
  int64_t rd, rd1; /* a scan's destinations' steps along its axis */
  int slices; /* a unit's outputs are slices scanned side by side */
  int64_t terms, blocks;
  int threads;    /* the most threads the call's jobs run on */
  int64_t row;    /* outputs per unit along the row axis; 1 per output */
  int64_t chunks; /* units along the row axis */
  int64_t band;   /* indices per unit of the kept axis before the row axis */
  int64_t bands;  /* units along that axis */
  int64_t outer;  /* units along the other kept axes */
  int64_t group;  /* blocks per group, a power of two */
  int64_t groups;
  uint8_t *scratch; /* each unit's group's values, where groups > 1 */
  const uint8_t *staged_unit; /* a unit's terms in the carrier, or NULL */
  int64_t off[LANES]; /* a lane or fewer terms: their positions from the
                         output's first */
  /* A program's terms that are the products of two of its loads, a lane or
     fewer an output, which the table's products sum: the loads, or -1, and
     each one's positions of an output's terms from its first, and its steps
     along the row axis and the band's. */
  int factor[2];
  int64_t foff[2][LANES], fs[2], fs2[2];
  _Atomic int *failed; /* set where a thread's work could not be taken */
} plan;

/* A thread's scratch for its units: every buffer the paths under a unit
   hold, in one of the thread's kept buffers (nx_cpu_take), so that its
   stack holds frames alone. A thread OCaml makes may have 128 KiB of
   stack (musl's default), and a unit runs on it under the door's and the
   plan's frames: the deepest chain from an entry of this file, through the
   other files of nx.cpu, took at most 31 KiB, and 40 KiB from a program's
   reduction, 12 KiB of it the door's descriptors (gcc's -O3 call graphs,
   arm64 and x86-64); a Moments unit's passes in turn (center_units) add
   2.4 KiB to either. Each buffer is named for the paths that use it: no
   path writes a buffer while a caller's data in it is live. */
typedef struct {
  /* The counter's values: the blocks' (blocks) or the groups'
     (finish_units), and a run of the table's blocks. The buffers every
     reduction takes come first, so that a short one touches the work's
     first pages alone. */
  _Alignas(64) uint8_t run[RUN * WIDEST];
  _Alignas(64) uint8_t tree[NX_CPU_TREE_DEPTH][ROW];
  /* A block's lanes (block, staged_block), or a few-term reduction's
     (few). */
  _Alignas(64) uint8_t lanes[NX_CPU_FEW_BYTES];
  /* Terms in the carrier, 32 KiB: as many of one output's full blocks as
     it holds (blocks), a block's or several outputs' blocks (states,
     staged_block), a unit's (few, center_units), a sum of products'
     factors (products), or a piece's (run_states, quad). */
  _Alignas(64) uint8_t terms[2 * BLOCK * WIDEST];
  /* Arg's extremes in the carrier, before their terms (run_states). */
  _Alignas(64) uint8_t extremes[LANES * ROW];
  /* A program's terms in their held dtype (eval_in). */
  _Alignas(64) uint8_t held[2 * LANES * ROW];
  /* A scan's carry, results and chunk total (scan_units, carries). */
  _Alignas(64) uint8_t carry[SCAN_ROW], a[SCAN_ROW], v[SCAN_ROW];
  /* A staged operand's or a program's terms, a piece of a chunk (rescan). */
  _Alignas(64) uint8_t pieces[2 * SCAN_ROW];
} work;

_Static_assert(LANES * ROW <= NX_CPU_FEW_BYTES, "a block's lanes fit few's");

/* The float dtype of [dt]'s parts: a complex's, or [dt] itself. */
static int part_of(int dt) {
  return dt == NX_COMPLEX64    ? NX_FLOAT32
         : dt == NX_COMPLEX128 ? NX_FLOAT64
                               : dt;
}

/* [n] parts of the float dtype [p], each [v] rounded, at [d]. */
static void parts_at(uint8_t *d, int p, double v, int n) {
  float f = (float)v;
  int w = nx_cpu_width(p);
  int64_t c = p == NX_FLOAT32 || p == NX_FLOAT64 ? 0 : nx_double_to_bits(p, v);
  for (int i = 0; i < n; i++)
    if (p == NX_FLOAT32) memcpy(d + 4 * i, &f, 4);
    else if (p == NX_FLOAT64) memcpy(d + 8 * i, &v, 8);
    else memcpy(d + w * i, &c, w); /* little-endian hosts: the low bytes */
}

/* A complex identity is 1 + 0i, or both parts the float's infinity. */
static void identity(int monoid, int dt, uint8_t *id) {
  memset(id, 0, WIDEST);
  int w = nx_cpu_width(dt), p = part_of(dt), n = p == dt ? 1 : 2;
  int fl = p == NX_FLOAT32 || p == NX_FLOAT64;
  switch (monoid) {
    case NX_MONOID_SUM: return;
    case NX_MONOID_PROD:
      if (fl) parts_at(id, p, 1., 1);
      else id[0] = 1; /* little-endian hosts: the low byte */
      return;
    case NX_MONOID_MAX:
      if (fl) parts_at(id, p, -INFINITY, n);
      else if (nx_dtype_row_of(dt).kind == NX_KIND_SIGNED) id[w - 1] = 0x80;
      return;
    default: /* NX_MONOID_MIN */
      if (fl) parts_at(id, p, INFINITY, n);
      else if (dt == NX_BOOL) id[0] = 1;
      else {
        memset(id, 0xFF, w);
        if (nx_dtype_row_of(dt).kind == NX_KIND_SIGNED) id[w - 1] = 0x7F;
      }
  }
}

/* An Arg scan's states of no term at [v], [n] of them, their first
   term [t0]. */
static void none(nx_cpu_state *v, int64_t n, int64_t t0) {
  for (int64_t j = 0; j < n; j++)
    v[j] = (nx_cpu_state){.n = t0, .at = -1};
}

/* [n] identities of the plan's monoid at [d], copied in doubling runs. */
static void ids(const plan *q, uint8_t *d, int64_t n) {
  if (n == 0) return;
  memcpy(d, q->id, q->w);
  for (int64_t k = 1; k < n; k *= 2)
    memcpy(d + k * q->w, d, (k < n - k ? k : n - k) * q->w);
}

/* [n] values of no term at [d]: identities, or an Arg scan's states
   of no term. */
static void fill(const plan *q, uint8_t *d, int64_t n) {
  if (q->z != NULL) {
    none((nx_cpu_state *)d, n, 0);
    return;
  }
  ids(q, d, n);
}

static const uint8_t *at(const plan *q, int64_t p) {
  return q->src->base + p * q->w;
}

/* Programs */

/* [z] for the program bound to its [z->nloads] loads [z->loads] over the
   virtual operand [x], C-contiguous over the loop's shape, the index read
   where [index]: the loop of [x] and the loads coalesced, the index
   stepping as [x]. Apart, so that its loop's frame is off the stack the
   jobs run on. */
static __attribute__((noinline)) void fuse(fused *z, const nx_array *x,
                                           int index) {
  int r = x->rank, nl = z->nloads;
  int64_t (*st)[NX_MAX_RANK] = z->pl.step - 1;
  int64_t *first = (int64_t *)(st + nl + 2);
  for (int i = 0; i < r; i++) {
    z->pl.ext[i] = x->dim[i];
    st[0][i] = x->dim[r + i];
    for (int k = 0; k < nl; k++) st[1 + k][i] = z->loads[k].dim[r + i];
  }
  z->pl.rank = nx_coalesce_dims(1 + nl, r, z->pl.ext, st);
  memcpy(z->pl.vs, st[0], sizeof(int64_t) * (size_t)z->pl.rank);
  memcpy(st[1 + nl], st[0], sizeof(int64_t) * (size_t)z->pl.rank);
  for (int k = 0; k < nl; k++) first[k] = z->loads[k].offset;
  first[nl] = 0;
  z->pl.n = nl + index;
}

/* The program's output over [n1] rows of [n0] terms from the virtual
   position [p], stepping [s0] along a row and [s1] across rows, each a
   step of one axis, into [out] in its held dtype: row r's term i at
   out + (r·n0 + i)·w. */
static void eval(const fused *z, int64_t p, int64_t s0, int64_t n0,
                 int64_t s1, int64_t n1, uint8_t *out) {
  nx_cpu_prog_terms(z->prog, &z->pl, z->k, p, s0, n0, s1, n1, out);
}

/* As eval, into [out] in the dtype [c]: the held values, through [held],
   converted into it where they differ. [held] holds the held values. */
static void eval_in(const fused *z, int c, int64_t p, int64_t s0, int64_t n0,
                    int64_t s1, int64_t n1, uint8_t *out, uint8_t *held) {
  int h = nx_cpu_prog_held(z->prog, z->k);
  if (h == c) {
    eval(z, p, s0, n0, s1, n1, out);
    return;
  }
  eval(z, p, s0, n0, s1, n1, held);
  nx_cpu_convert(h, c)(held, out, n0 * n1);
}

/* The operand's element at [pos] in the carrier: in place, or staged into
   [tmp], or a program's computed into it. */
static const uint8_t *element(const plan *q, int64_t pos, uint8_t *tmp) {
  if (q->fused != NULL) {
    _Alignas(16) uint8_t held[WIDEST];
    eval_in(q->fused, q->carrier, pos, 0, 1, 0, 1, tmp, held);
    return tmp;
  }
  if (!q->staged) return at(q, pos);
  nx_cpu_stage_as(q->src, pos, 1, 0, 1, 1, q->carrier, tmp, 1);
  return tmp;
}

/* Whether the paths read terms through stage_terms and element: a narrow
   operand's or a program's. */
static int sourced(const plan *q) { return q->staged || q->fused != NULL; }

/* The operand's position of term [t] of the output at [p]. */
static int64_t term(const plan *q, int64_t p, int64_t t) {
  for (int i = q->nr - 1; i >= 0; i--) {
    p += t % q->re[i] * q->rs[i];
    t /= q->re[i];
  }
  return p;
}

/* The terms' positions step through the reduced axes as an odometer: the
   position of term [t] of the output at [p], and its indices into [idx]. */
static int64_t odometer(const plan *q, int64_t p, int64_t t, int64_t *idx) {
  for (int i = q->nr - 1; i >= 0; i--) {
    idx[i] = t % q->re[i];
    p += idx[i] * q->rs[i];
    t /= q->re[i];
  }
  return p;
}

/* The position of the term after the one at [pos], its indices [idx]
   stepped. */
static int64_t next(const plan *q, int64_t *idx, int64_t pos) {
  int i = q->nr - 1;
  pos += q->rs[i];
  while (++idx[i] == q->re[i] && i > 0) {
    pos -= q->re[i] * q->rs[i];
    idx[i--] = 0;
    pos += q->rs[i];
  }
  return pos;
}

/* The stage */

/* The program's terms as stage_terms gives the operand's, in the carrier.
   Its rows take at most 2·LANES·ROW bytes, as every caller's buffer, their
   held values through [held]. */
static void fused_terms(const plan *q, int64_t p, int64_t s, int64_t W,
                        int64_t t, int64_t m, uint8_t *buf, uint8_t *held) {
  int r = q->nr - 1, c = q->carrier;
  if (W == 1)
    for (int64_t i = 0; i < m;) {
      int64_t len = q->re[r] - (t + i) % q->re[r];
      if (len > m - i) len = m - i;
      eval_in(q->fused, c, term(q, p, t + i), q->rs[r], len, 0, 1,
              buf + i * q->w, held);
      i += len;
    }
  else if (q->nr == 1)
    eval_in(q->fused, c, p + t * q->rs[0], s, W, q->rs[0], m, buf, held);
  else {
    int64_t idx[NX_MAX_RANK], pos = odometer(q, p, t, idx);
    for (int64_t i = 0; i < m; i++, pos = next(q, idx, pos))
      eval_in(q->fused, c, pos, s, W, 0, 1, buf + i * W * q->w, held);
  }
}

/* Converts into the carrier at [buf] the [W] elements, stepping [s], of
   each of the [m] terms from [t] of the outputs from [p]: term t + i's row
   at buf + i·W·w. One output's terms stage run by run along the innermost
   reduced axis; several outputs' along one reduced axis, at once. A
   program's held values go through [wk]. */
static void stage_terms(const plan *q, work *wk, int64_t p, int64_t s,
                        int64_t W, int64_t t, int64_t m, uint8_t *buf) {
  int r = q->nr - 1;
  if (q->fused != NULL) {
    fused_terms(q, p, s, W, t, m, buf, wk->held);
    return;
  }
  if (W == 1) {
    for (int64_t i = 0; i < m;) {
      int64_t len = q->re[r] - (t + i) % q->re[r];
      if (len > m - i) len = m - i;
      nx_cpu_stage_as(q->src, term(q, p, t + i), q->rs[r], 0, len, 1,
                      q->carrier, buf + i * q->w, len);
      i += len;
    }
    return;
  }
  if (q->nr == 1) {
    nx_cpu_stage_as(q->src, p + t * q->rs[0], s, q->rs[0], W, m, q->carrier,
                    buf, W);
    return;
  }
  int64_t idx[NX_MAX_RANK], pos = odometer(q, p, t, idx);
  for (int64_t i = 0; i < m; i++, pos = next(q, idx, pos))
    nx_cpu_stage_as(q->src, pos, s, 0, W, 1, q->carrier, buf + i * W * q->w,
                    W);
}

/* Blocks */

/* Adds terms [t0, t0 + n) of the output at [p] into the lanes [l], run by
   run along the innermost reduced axis. */
static void runs(const plan *q, int64_t p, int64_t t0, int64_t n, uint8_t *l) {
  int r = q->nr - 1;
  int64_t inner = q->re[r], s = q->rs[r];
  for (int64_t t = t0; t < t0 + n;) {
    int64_t i = t % inner, len = inner - i;
    if (len > t0 + n - t) len = t0 + n - t;
    q->f->lanes(at(q, term(q, p, t)), s, len, q->id, l,
                (int)(t & (LANES - 1)));
    t += len;
  }
}

/* Copies term [t0 + i] of the [W] outputs, the first at [p] and each next
   [s] further along the operand, into row i of [W] values of the lanes
   [l], for i below [used]: the lanes start from their first terms. */
static void seed(const plan *q, int64_t p, int64_t s, int64_t W, int64_t t0,
                 int used, uint8_t *l) {
  for (int i = 0; i < used; i++)
    for (int64_t j = 0; j < W; j++)
      memcpy(l + (i * W + j) * q->w, at(q, term(q, p + j * s, t0 + i)), q->w);
}

/* The lanes' tree over [W] outputs: 16 rows of [W] values, [used] of them
   holding terms, which alone take part. */
static void lane_tree(const plan *q, uint8_t *l, int64_t W, int used) {
  int64_t row = W * q->w;
  NX_CPU_LANE_TREE(
      i, j, if (j < used) q->f->combine(l + i * row, l + j * row, 1, W));
}

/* Block [b]'s states of the unit's [W] outputs, the first at [p] and each
   next [s] further along the operand, their centers [c] or 0 where NULL,
   into [v]: each output's terms where they lie contiguous, else copied into
   a block. Along one reduced axis, the blocks of as many outputs as 32 KiB
   holds are copied at once: a row of the operand then serves them all,
   where one output at a time would read each row once per output. */
static void states(const plan *q, work *wk, int64_t p, int64_t s, int64_t W,
                   const double *c, int64_t b, nx_cpu_state *v) {
  int64_t t0 = b * BLOCK, n = q->terms - t0 < BLOCK ? q->terms - t0 : BLOCK;
  uint8_t *buf = wk->terms;
  int contiguous = q->nr == 1 && q->rs[0] == 1;
  if (!contiguous && q->nr == 1 && W > 1) {
    int64_t most = (int64_t)sizeof wk->terms / (n * q->w), rs = q->rs[0];
    for (int64_t j = 0; j < W; j += most) {
      int64_t k = W - j < most ? W - j : most, o = p + j * s + t0 * rs;
      if (q->fused != NULL)
        eval_in(q->fused, q->carrier, o, rs, n, s, k, buf, wk->held);
      else nx_cpu_stage_as(q->src, o, rs, s, n, k, q->carrier, buf, n);
      for (int64_t i = 0; i < k; i++)
        q->r->block(buf + i * n * q->w, n, t0, c != NULL ? c[j + i] : 0,
                    &v[j + i]);
    }
    return;
  }
  for (int64_t j = 0; j < W; j++) {
    int64_t o = p + j * s;
    const uint8_t *x = buf;
    if (q->staged_unit != NULL) x = q->staged_unit + t0 * q->w;
    else if (sourced(q) || !contiguous) stage_terms(q, wk, o, 0, 1, t0, n, buf);
    else x = at(q, o + t0);
    q->r->block(x, n, t0, c != NULL ? c[j] : 0, &v[j]);
  }
}

/* Combines the values [b] of [W] outputs into [a]'s. */
static void join(const plan *q, uint8_t *a, const uint8_t *b, int64_t W) {
  if (q->f != NULL) {
    q->f->combine(a, b, 1, W);
    return;
  }
  for (int64_t j = 0; j < W; j++)
    q->r->merge((nx_cpu_state *)a + j, (const nx_cpu_state *)b + j);
}

/* Block [b]'s values of the unit's [W] outputs from a staged operand or a
   program, as block's: its terms in the carrier, a buffer's of one
   output's at a time, else 16 terms at a time, each a row of the outputs'.
   The lanes start from their first terms where the plan's do. */
static void staged_block(const plan *q, work *wk, int64_t p, int64_t s,
                         int64_t W, int64_t b, uint8_t *v) {
  int64_t t0 = b * BLOCK, n = q->terms - t0 < BLOCK ? q->terms - t0 : BLOCK;
  int used = n < LANES ? (int)n : LANES;
  int64_t row = W * q->w, step = W == 1 ? LANES * ROW / q->w : LANES;
  uint8_t *l = wk->lanes, *buf = wk->terms;
  if (!q->first) ids(q, l, used * W);
  for (int64_t t = 0; t < n; t += step) {
    int64_t m = n - t < step ? n - t : step;
    stage_terms(q, wk, p, s, W, t0 + t, m, buf);
    /* The rows the lanes start from: the block's first ones. */
    int64_t k = t == 0 && q->first ? used : 0;
    if (k > 0) memcpy(l, buf, (size_t)(k * row));
    if (W == 1)
      q->f->lanes(buf + k * q->w, 1, m - k, q->id, l,
                  (int)((t + k) & (LANES - 1)));
    else
      for (int64_t i = k; i < m; i++)
        q->f->combine(l + i * row, buf + i * row, 1, W);
  }
  lane_tree(q, l, W, used);
  memcpy(v, l, row);
}

/* Block [b]'s values of the unit's [W] outputs, the first at [p] and each
   next [s] further along the operand, of centers [c], into [v]. */
static void block(const plan *q, work *wk, int64_t p, int64_t s, int64_t W,
                  const double *c, int64_t b, uint8_t *v) {
  if (q->r != NULL) {
    states(q, wk, p, s, W, c, b, (nx_cpu_state *)v);
    return;
  }
  if (sourced(q)) {
    staged_block(q, wk, p, s, W, b, v);
    return;
  }
  int64_t t0 = b * BLOCK, n = q->terms - t0 < BLOCK ? q->terms - t0 : BLOCK;
  int used = n < LANES ? (int)n : LANES;
  /* The terms the lanes start from, where they start from their first. */
  int k = q->first ? used : 0;
  if (W == 1) {
    _Alignas(16) uint8_t l[LANES * WIDEST];
    if (q->first) seed(q, p, s, 1, t0, used, l);
    else ids(q, l, LANES);
    runs(q, p, t0 + k, n - k, l);
    lane_tree(q, l, 1, used);
    memcpy(v, l, q->w);
    return;
  }
  uint8_t *l = wk->lanes;
  int64_t row = W * q->w;
  if (q->first) seed(q, p, s, W, t0, used, l);
  else
    for (int i = 0; i < used; i++) ids(q, l + i * row, W);
  /* Along one reduced axis, lane i's terms lie LANES apart: the table adds
     each lane's in order, its accumulators in registers. */
  if (q->nr == 1) {
    int64_t st = q->rs[0], from = q->first ? LANES : 0;
    for (int i = 0; i < used; i++)
      q->f->column(l + i * row, at(q, p + (t0 + i + from) * st), LANES * st,
                   (n - i - from + LANES - 1) / LANES, s, W);
    lane_tree(q, l, W, used);
    memcpy(v, l, row);
    return;
  }
  int64_t idx[NX_MAX_RANK], pos = odometer(q, p, t0 + k, idx);
  for (int64_t t = t0 + k; t < t0 + n; t++, pos = next(q, idx, pos))
    q->f->combine(l + (t & (LANES - 1)) * row, at(q, pos), s, W);
  lane_tree(q, l, W, used);
  memcpy(v, l, row);
}

/* The full blocks of one output's staged or computed terms that the
   work's buffers hold at once: its terms in the carrier, and a program's
   held values where they differ from it. */
static int64_t staged_run(const plan *q, const work *wk) {
  int64_t most = (int64_t)sizeof wk->terms / (BLOCK * q->w);
  if (q->fused == NULL) return most;
  int h = nx_cpu_prog_held(q->fused->prog, q->fused->k);
  if (h == q->carrier) return most;
  int64_t held = (int64_t)sizeof wk->held / (BLOCK * nx_cpu_width(h));
  return held < most ? held : most;
}

/* The value of blocks [b0, b1) of the unit's outputs, of centers [c], by
   the counter, into [v]: the identity for no block. */
static void blocks(const plan *q, work *wk, int64_t p, int64_t s, int64_t W,
                   const double *c, int64_t b0, int64_t b1, uint8_t *v) {
  uint8_t(*stack)[ROW] = wk->tree, *run = wk->run;
  int top = 0;
  int64_t row = W * q->size;
  /* A monoid's output's full blocks take the table's blocks, whose lanes
     start from the identity: RUN at a time of contiguous terms, or as many
     as the work holds of staged or computed ones, staged in one call. On
     one fudge core, Σ exp (x - m) over [64, 128256] float32 took 7.1 ms a
     block at a time through the lanes, 5.7 so. */
  int contiguous = W == 1 && q->nr == 1 && q->rs[0] == 1;
  int fits = sourced(q) ? W == 1 : contiguous;
  int64_t full = q->f != NULL && !q->first && fits ? q->terms / BLOCK : 0;
  int64_t most = sourced(q) ? staged_run(q, wk) : RUN;
  for (int64_t b = b0; b < b1;) {
    int64_t n = (b1 < full ? b1 : full) - b;
    if (n > most) n = most;
    if (n > 0 && sourced(q)) {
      stage_terms(q, wk, p, 0, 1, b * BLOCK, n * BLOCK, wk->terms);
      q->f->blocks(wk->terms, n, q->id, run);
    } else if (n > 0)
      q->f->blocks(at(q, p + b * BLOCK), n, q->id, run);
    else
      n = 1;
    for (int64_t i = 0; i < n; i++, b++) {
      if (b < full) memcpy(stack[top++], run + i * q->w, q->w);
      else block(q, wk, p, s, W, c, b, stack[top++]);
      NX_CPU_TREE_STEP(b - b0 + 1, top,
                       join(q, stack[top - 2], stack[top - 1], W));
    }
  }
  NX_CPU_TREE_END(top, join(q, stack[top - 2], stack[top - 1], W));
  if (top == 0) fill(q, v, W);
  else memcpy(v, stack[0], row);
}

/* Units */

/* Unit [u]'s outputs: [H] rows of [W], the first one's positions in the
   operand, the destinations and a scan's carries, and the operand's,
   destinations' and carries' steps from one output to the next in a row,
   and from one row to the next. Units run outer index, then band, then
   chunk, then group. */
typedef struct {
  int64_t p, d, d1, c, W, s, sd, sd1, sc, H, s2, sd2, g;
} unit;

static unit unit_of(const plan *q, int64_t u) {
  unit x = {.p = q->x0, .d = q->d0, .d1 = q->d1, .c = q->c0, .W = 1, .H = 1,
            .g = u % q->groups};
  u /= q->groups;
  int last = q->nk - 1;
  if (q->row > 1) {
    int64_t c = u % q->chunks;
    u /= q->chunks;
    x.s = q->ks[last];
    x.sd = q->kd[last];
    x.sd1 = q->kd1[last];
    x.sc = q->kc[last];
    x.W = q->ke[last] - c * q->row < q->row ? q->ke[last] - c * q->row : q->row;
    x.p += c * q->row * x.s;
    x.d += c * q->row * x.sd;
    x.d1 += c * q->row * x.sd1;
    x.c += c * q->row * x.sc;
    last--;
  }
  if (q->band > 1) {
    int64_t c = u % q->bands;
    u /= q->bands;
    x.s2 = q->ks[last];
    x.sd2 = q->kd[last];
    x.H = q->ke[last] - c * q->band < q->band ? q->ke[last] - c * q->band
                                              : q->band;
    x.p += c * q->band * x.s2;
    x.d += c * q->band * x.sd2;
    x.d1 += c * q->band * q->kd1[last];
    x.c += c * q->band * q->kc[last];
    last--;
  }
  for (int i = last; i >= 0; i--) {
    x.p += u % q->ke[i] * q->ks[i];
    x.d += u % q->ke[i] * q->kd[i];
    x.d1 += u % q->ke[i] * q->kd1[i];
    x.c += u % q->ke[i] * q->kc[i];
    u /= q->ke[i];
  }
  return x;
}

/* Copies element [s] of [a] into element [d] of [b], of [a]'s dtype, bits
   for bits. */
static void copy_element(const nx_array *b, int64_t d, const nx_array *a,
                         int64_t s) {
  int bits = nx_dtype_row_of(a->dtype).bits;
  if (bits < 8) nx_sub_store(b->base, bits, d, nx_sub_load(a->base, bits, s));
  else memcpy(b->base + d * (bits / 8), a->base + s * (bits / 8), bits / 8);
}

/* Arg's extreme of the operand's dtype into element [d] of the
   destination: the operand's term at [pos], or a program's computed
   there. */
static void own_term(const plan *q, int64_t d, int64_t pos) {
  int64_t w = nx_cpu_width(q->dst->dtype);
  if (q->fused == NULL) copy_element(q->dst, d, q->src, pos);
  else eval(q->fused, pos, 0, 1, 0, 1, q->dst->base + d * w);
}

/* Stores the results of the unit's states [v], of centers [c], Arg's
   extreme of the operand's dtype copied from its term; or, where the pass
   finds them, the centers. */
static void store_states(const plan *q, const unit *x, double *c,
                         const nx_cpu_state *v) {
  int w1 = nx_cpu_width(q->dst1->dtype);
  for (int64_t j = 0; j < x->W; j++) {
    if (q->centering) {
      q->r->finish(&v[j], 0, (uint8_t *)&c[j], NULL);
      continue;
    }
    int64_t p = x->p + j * x->s;
    uint8_t *d0 = q->dst->base + (x->d + j * x->sd) * q->w;
    uint8_t *d1 = q->dst1->base + (x->d1 + j * x->sd1) * w1;
    if (q->own) {
      _Alignas(16) uint8_t e[WIDEST];
      q->r->finish(&v[j], 0, e, d1);
      own_term(q, x->d + j * x->sd, term(q, p, v[j].at));
      continue;
    }
    q->r->finish(&v[j], c != NULL ? c[j] : 0, d0, d1);
  }
}

/* Stores the unit's values [v], of centers [c]. */
static void store(const plan *q, const unit *x, double *c, const uint8_t *v) {
  if (q->r != NULL) {
    store_states(q, x, c, (const nx_cpu_state *)v);
    return;
  }
  uint8_t *d = q->dst->base + x->d * q->w;
  if (x->sd == 1) memcpy(d, v, x->W * q->w);
  else
    for (int64_t j = 0; j < x->W; j++)
      memcpy(d + j * x->sd * q->w, v + j * q->w, q->w);
}

/* The terms of a factor of a sum of products that a thread staged last
   (products): the position of their row's first and its outputs, or -1. */
typedef struct {
  int64_t at[2], w[2];
} staged;

/* The unit's sums of the products of two loads, a lane or fewer an output.
   Each factor's terms are read in place where they
   are of the carrier and step one along a row, else staged into the
   carrier, a row of the unit's outputs a term, once for the rows and units
   that read the same ones, as a broadcast factor's do: the table then
   computes the unit's rows at once, but one at a time where a staged
   factor steps along them. */
static void products(const plan *q, work *wk, const unit *x, staged *kept) {
  const fused *z = q->fused;
  int64_t at[2], rows[LANES];
  for (int f = 0; f < 2; f++) at[f] = nx_cpu_place(&z->pl, q->factor[f], x->p);
  int n = (int)q->terms, place[2];
  for (int t = 0; t < n; t++) rows[t] = t * x->W;
  int64_t each = x->H;
  for (int f = 0; f < 2; f++) {
    const nx_array *l = &z->loads[q->factor[f]];
    place[f] = l->dtype == q->carrier && (q->fs[f] == 1 || x->W == 1);
    if (!place[f] && q->fs2[f] != 0) each = 1;
  }
  uint8_t *buf[2] = {wk->terms, wk->terms + sizeof wk->terms / 2};
  const uint8_t *base[2];
  const int64_t *off[2];
  int64_t step[2];
  for (int64_t h = 0; h < x->H; h += each) {
    for (int f = 0; f < 2; f++) {
      const nx_array *l = &z->loads[q->factor[f]];
      int64_t p = at[f] + h * q->fs2[f];
      if (place[f]) {
        base[f] = l->base + p * q->w;
        off[f] = q->foff[f];
        step[f] = q->fs2[f];
        continue;
      }
      base[f] = buf[f];
      off[f] = rows;
      step[f] = 0;
      if (kept->at[f] == p && kept->w[f] == x->W) continue;
      for (int t = 0; t < n; t++)
        nx_cpu_stage_as(l, p + q->foff[f][t], q->fs[f], 0, x->W, 1,
                        q->carrier, buf[f] + t * x->W * q->w, x->W);
      kept->at[f] = p;
      kept->w[f] = x->W;
    }
    table.products[q->carrier](
        n, x->W, each, base[0], off[0], step[0], base[1], off[1], step[1],
        q->dst->base + (x->d + h * x->sd2) * q->w, x->sd, x->sd2);
  }
}

/* Outputs of a lane or fewer each, one block: their terms' positions from
   an output's first are the same for every output, and the table reduces
   the unit's rows at once, its lanes side by side. */
static void few(const plan *q, work *wk, const unit *x, staged *kept) {
  int64_t n = q->terms;
  if (q->factor[0] >= 0) {
    products(q, wk, x, kept);
  } else if (!sourced(q)) {
    q->f->few(at(q, x->p), q->off, (int)n, x->s, x->W, x->s2, x->H, q->id,
              q->dst->base + x->d * q->w, x->sd, x->sd2, wk->lanes);
  } else {
    /* The terms in the carrier, term t of row h a row of the W outputs'
       at buf + (h n + t)·W·w: within a unit's H·W < 2·ROW / w outputs of
       at most 16 terms each. */
    uint8_t *buf = wk->terms;
    int64_t off[LANES];
    for (int64_t h = 0; h < x->H; h++)
      stage_terms(q, wk, x->p + h * x->s2, x->s, x->W, 0, n,
                  buf + h * n * x->W * q->w);
    for (int64_t t = 0; t < n; t++) off[t] = t * x->W;
    q->f->few(buf, off, (int)n, 1, x->W, n * x->W, x->H, q->id,
              q->dst->base + x->d * q->w, x->sd, x->sd2, wk->lanes);
  }
}

static void reduce_units(const plan *q, work *wk, int64_t lo, int64_t hi) {
  _Alignas(16) uint8_t v[ROW];
  staged kept = {{-1, -1}, {-1, -1}};
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u);
    if (q->f != NULL && !q->first && q->terms <= LANES) {
      few(q, wk, &x, &kept);
      continue;
    }
    int64_t b0 = x.g * q->group, b1 = b0 + q->group;
    if (b1 > q->blocks) b1 = q->blocks;
    double *c = q->centers ? q->centers + u / q->groups * q->row : NULL;
    /* The center pass's blocks take no centers: it stores them. */
    const double *bc = q->centering ? NULL : c;
    if (q->groups == 1) {
      blocks(q, wk, x.p, x.s, x.W, bc, b0, b1, v);
      store(q, &x, c, v);
    } else
      blocks(q, wk, x.p, x.s, x.W, bc, b0, b1,
             q->scratch + u * q->row * q->size);
  }
}

/* Combines each unit's groups by the counter and stores them. */
static void finish_units(const plan *q, work *wk, int64_t lo, int64_t hi) {
  int64_t row = q->row * q->size;
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u * q->groups);
    uint8_t(*stack)[ROW] = wk->tree;
    int top = 0;
    for (int64_t g = 0; g < q->groups; g++) {
      memcpy(stack[top++], q->scratch + (u * q->groups + g) * row, row);
      NX_CPU_TREE_STEP(g + 1, top,
                       join(q, stack[top - 2], stack[top - 1], x.W));
    }
    NX_CPU_TREE_END(top, join(q, stack[top - 2], stack[top - 1], x.W));
    store(q, &x, q->centers ? q->centers + u * q->row : NULL, stack[0]);
  }
}

/* Each unit's two passes in turn, so that the second reads the unit's
   terms from the caches: its centers by [q->center], then its values by
   [q->r]. A unit's terms are one output's, or a row's of 16 outputs: 16
   KiB for rows of 4096 float32 terms, 512 KiB for columns of 8192. One
   output's that a pass would stage, converted, gathered or computed, and
   that fit the work's terms, are staged once for both, block by block as
   a pass stages them. Both passes run on one copy of the plan. */
static void center_units(const plan *q, work *wk, int64_t lo, int64_t hi) {
  plan p = *q;
  int contiguous = q->nr == 1 && q->rs[0] == 1;
  int once = q->row == 1 && (sourced(q) || !contiguous) &&
             q->terms * q->w <= (int64_t)sizeof wk->terms;
  if (once) p.staged_unit = wk->terms;
  for (int64_t u = lo; u < hi; u++) {
    if (once) {
      unit x = unit_of(q, u);
      for (int64_t t = 0; t < q->terms; t += BLOCK) {
        int64_t n = q->terms - t < BLOCK ? q->terms - t : BLOCK;
        stage_terms(q, wk, x.p, 0, 1, t, n, wk->terms + t * q->w);
      }
    }
    p.r = q->center;
    p.centering = 1;
    reduce_units(&p, wk, u, u + 1);
    p.r = q->r;
    p.centering = 0;
    reduce_units(&p, wk, u, u + 1);
  }
}

static int64_t magnitude(int64_t x) { return x < 0 ? -x : x; }

/* A plan's units [lo, hi) on one thread, with the thread's work. */
typedef void (*units_body)(const plan *q, work *wk, int64_t lo, int64_t hi);

typedef struct {
  units_body body;
  const plan *q;
} plan_job;

/* A plan's job's range on one thread: its work taken for the range, or the
   call failed where it cannot be. */
static void with_work(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const plan_job *j = ctx;
  work *wk = nx_cpu_take(sizeof *wk);
  if (wk == NULL) {
    *j->q->failed = 1;
    return;
  }
  j->body(j->q, wk, lo, hi);
  nx_cpu_give(wk);
}

/* nx_cpu_job of the plan [q]'s units each by [body]. Once a range's work
   could not be taken, no later job runs: it would read what that range
   left unwritten. */
static void units_job(int64_t total, int64_t bytes, double cost,
                      units_body body, const plan *q) {
  if (*q->failed) return;
  plan_job j = {body, q};
  nx_cpu_job(total, bytes, cost, with_work, &j);
}

/* As units_job, on the calling thread alone: each unit reads the results
   of the units before it. */
static void units_serial(int64_t total, units_body body, const plan *q) {
  if (*q->failed) return;
  plan_job j = {body, q};
  with_work(0, total, 0, &j);
}

/* Adds the axis of extent [e] and steps [s] (operand), [d] and [d1]
   (destinations) and [c] (carries) to the kept axes, if it has more than
   one element. */
static void keep(plan *q, int64_t e, int64_t s, int64_t d, int64_t d1,
                 int64_t c) {
  if (e <= 1) return;
  q->ke[q->nk] = e;
  q->ks[q->nk] = s;
  q->kd[q->nk] = d;
  q->kc[q->nk] = c;
  q->kd1[q->nk++] = d1;
}

/* Picks the path for [q]'s axes and lays out its units. The kept axis that
   steps least through the operand becomes the row axis, last, where it
   steps less than every reduced axis or the outputs have few terms. A
   scan's slices otherwise run SLICES side by side along that axis, so that
   as many running sums are in flight. Outputs of a lane of terms or fewer
   whose rows are short take bands of rows of the kept axis before the row
   axis, so that a unit holds about a row's worth of outputs. */
static void choose(plan *q, task op) {
  int64_t outputs = 1;
  for (int i = 0; i < q->nk; i++) outputs *= q->ke[i];
  int least = -1;
  int64_t kept = INT64_MAX, red = INT64_MAX;
  for (int i = 0; i < q->nk; i++)
    if (magnitude(q->ks[i]) < kept) {
      least = i;
      kept = magnitude(q->ks[i]);
    }
  for (int i = 0; i < q->nr; i++)
    if (q->re[i] > 1 && magnitude(q->rs[i]) < red) red = magnitude(q->rs[i]);
  q->row = 1;
  q->chunks = 1;
  q->band = 1;
  q->bands = 1;
  q->outer = outputs;
  q->blocks = (q->terms + BLOCK - 1) / BLOCK;
  int rows = least >= 0 && (kept < red || q->terms < FEW);
  if (least < 0 || (!rows && op != SCAN)) return;
  int64_t e = q->ke[least], s = q->ks[least], d = q->kd[least],
          d1 = q->kd1[least], c = q->kc[least];
  for (int i = least; i < q->nk - 1; i++) {
    q->ke[i] = q->ke[i + 1];
    q->ks[i] = q->ks[i + 1];
    q->kd[i] = q->kd[i + 1];
    q->kd1[i] = q->kd1[i + 1];
    q->kc[i] = q->kc[i + 1];
  }
  q->ke[q->nk - 1] = e;
  q->ks[q->nk - 1] = s;
  q->kd[q->nk - 1] = d;
  q->kd1[q->nk - 1] = d1;
  q->kc[q->nk - 1] = c;
  q->slices = !rows;
  int64_t most = q->slices ? SLICES : ROW / q->size;
  /* A monoid scan on one thread sweeps long rows; on several, its units
     of ROW share the work more evenly, which gains more. Its bits do not
     depend on the rows. */
  if (!q->slices && op == SCAN && q->f != NULL && q->threads == 1)
    most = SCAN_ROW / q->size;
  q->row = most < e ? most : e;
  q->chunks = (e + q->row - 1) / q->row;
  q->outer = outputs / e;
  if (op == SCAN || q->f == NULL || q->first || q->terms > LANES ||
      q->nk < 2 || q->chunks > 1)
    return;
  /* A sum of products stores its rows as it computes them: its units take
     more of them. */
  int64_t per = q->factor[0] >= 0 ? PRODUCTS_UNIT : most;
  int64_t next = q->ke[q->nk - 2], want = (per + q->row - 1) / q->row;
  q->bands = (next + want - 1) / want;
  q->band = (next + q->bands - 1) / q->bands;
  q->outer /= next;
}

/* Sets the plan's factors where its terms, a lane or fewer an output, are
   a Sum's of the products of two of a program's loads at the carrier, which
   the table's products sum: the products' bits are the program's, and the
   sums' the lanes'. */
static void factors(plan *q) {
  int k[2];
  const fused *z = q->fused;
  q->factor[0] = q->factor[1] = -1;
  if (z == NULL || q->f != &table.monoid[NX_MONOID_SUM][q->carrier] ||
      q->terms > LANES || table.products[q->carrier] == NULL ||
      nx_cpu_prog_product(z->prog, z->k, k) != q->carrier)
    return;
  for (int64_t t = 0; t < q->terms; t++)
    for (int f = 0; f < 2; f++)
      q->foff[f][t] = nx_cpu_place(&z->pl, k[f], q->off[t]) -
                      nx_cpu_place(&z->pl, k[f], 0);
  q->factor[0] = k[0];
  q->factor[1] = k[1];
}

/* The factors' steps along the row axis and the band's, where the plan has
   factors and those axes. */
static void factor_steps(plan *q) {
  if (q->factor[0] < 0) return;
  const nx_cpu_places *pl = &q->fused->pl;
  int64_t s = q->row > 1 ? q->ks[q->nk - 1] : 0;
  int64_t s2 = q->band > 1 ? q->ks[q->nk - 2] : 0;
  for (int f = 0; f < 2; f++) {
    q->fs[f] = nx_cpu_place_step(pl, q->factor[f], s);
    q->fs2[f] = nx_cpu_place_step(pl, q->factor[f], s2);
  }
}

/* Lays out a reduction from the coalesced loop of [rank] axes of extents
   [extent] over the destinations, steps step[0] and step[1], and the
   operand, step[2], the destinations stepping 0 along the reduced axes.
   Where units are few and blocks many, each unit takes a group of blocks,
   its size from the shape alone. */
static void plan_reduce(plan *q, int rank, const int64_t *extent,
                        int64_t (*step)[NX_MAX_RANK]) {
  q->nr = q->nk = 0;
  for (int i = 0; i < rank; i++)
    if (step[0][i] != 0)
      keep(q, extent[i], step[2][i], step[0][i], step[1][i], 0);
    else if (extent[i] > 1) {
      q->re[q->nr] = extent[i];
      q->rs[q->nr++] = step[2][i];
    }
  if (q->nr == 0) {
    q->re[0] = 1;
    q->rs[0] = 0;
    q->nr = 1;
  }
  for (int64_t t = 0; t < q->terms && t < LANES; t++) q->off[t] = term(q, 0, t);
  factors(q);
  choose(q, REDUCE);
  factor_steps(q);
  int64_t units = q->outer * q->bands * q->chunks;
  q->group = q->blocks > 0 ? q->blocks : 1;
  if (units < UNITS && q->blocks > 1) {
    int64_t want = (UNITS + units - 1) / units;
    int64_t per = (q->blocks + want - 1) / want;
    for (q->group = 1; q->group < per;) q->group *= 2;
  }
  q->groups = q->blocks > 0 ? (q->blocks + q->group - 1) / q->group : 1;
}

/* [d]'s axes coalesced (nx_coalesce_dims) into [extent] and [step]: its
   rank, the last axis its rows. */
static int rows_of(const nx_array *d, int64_t *extent,
                   int64_t (*step)[NX_MAX_RANK]) {
  for (int i = 0; i < d->rank; i++) {
    extent[i] = d->dim[i];
    step[0][i] = d->dim[d->rank + i];
  }
  return nx_coalesce_dims(1, d->rank, extent, step);
}

/* The elements of [d], each of [w] bytes, all set to the bits [e]: a
   reduction of no term. */
static void fill_dst(const nx_array *d, const uint8_t *e, int w) {
  int64_t extent[NX_MAX_RANK], step[1][NX_MAX_RANK];
  int r = rows_of(d, extent, step);
  int64_t n = extent[r - 1], s = step[0][r - 1], rows = 1;
  for (int i = 0; i < r - 1; i++) rows *= extent[i];
  nx_cpu_row0 row = nx_cpu_rows.fill[nx_cpu_width_index(w)];
  int64_t at[NX_MAX_RANK] = {0}, p = d->offset;
  for (int64_t k = 0; k < rows && n > 0; k++) {
    row(n, d->base + p * w, s, e);
    /* The next row: the odometer's last outer axis steps, carrying. */
    int i = r - 2;
    for (; i >= 0 && ++at[i] == extent[i]; i--) {
      at[i] = 0;
      p -= (extent[i] - 1) * step[0][i];
    }
    if (i >= 0) p += step[0][i];
  }
}

/* Scans */

/* Converts rows [t, t + m) of the unit's [W] sequences into the carrier
   at [buf], row i's sequence l at buf + (i W + l)·w, or computes a
   program's there: sequence l is the unit's slice l, or, where [lt] is not
   0, the unit's slice from term l·lt on. */
static void stage_lanes(const plan *q, work *wk, const unit *x, int64_t W,
                        int64_t lt, int64_t t, int64_t m, uint8_t *buf) {
  int64_t rs = q->rs[0], s = (lt ? 0 : x->s) + lt * rs;
  if (q->fused != NULL) {
    eval_in(q->fused, q->carrier, x->p + t * rs, s, W, rs, m, buf, wk->held);
    return;
  }
  nx_cpu_stage_as(q->src, x->p + t * rs, s, rs, W, m, q->carrier, buf, W);
}

/* Terms [t0, t0 + n) of an Arg scan's [W] sequences into their states
   [a], and each result into the destinations. Sequence l is
   the unit's slice l, or, where [lt] is not 0, the unit's slice from term
   l·lt on. The kernel reads the operand and writes the destinations in
   place, a staged operand's terms converted into the carrier a piece at a
   time; an extreme of the operand's dtype goes through a piece of scratch
   and is then copied from the term at its position. */
static void run_states(const plan *q, work *wk, const unit *x, nx_cpu_state *a,
                       int64_t W, int64_t lt, int64_t t0, int64_t n) {
  int64_t rs = q->rs[0], ls = lt ? 0 : x->s, xl = ls + lt * rs;
  int64_t dl = lt ? lt * q->rd : x->sd, d1l = lt ? lt * q->rd1 : x->sd1;
  /* A staged operand's or a program's terms come side by side. */
  int staged = sourced(q);
  int64_t piece = staged ? LANES * ROW / (W * WIDEST) : n;
  uint8_t *buf = wk->terms, *e = wk->extremes;
  for (int64_t t = t0; t < t0 + n; t += piece) {
    int64_t m = t0 + n - t < piece ? t0 + n - t : piece;
    const uint8_t *src = staged ? buf : at(q, x->p + t * rs);
    if (staged) stage_lanes(q, wk, x, W, lt, t, m, buf);
    int64_t xs = staged ? W : rs, xli = staged ? 1 : xl;
    uint8_t *y = q->own ? e : q->dst->base + (x->d + t * q->rd) * q->w;
    int64_t *p = (int64_t *)q->dst1->base + x->d1 + t * q->rd1;
    q->z->rows(a, src, xs, xli, m, W, y, q->own ? W : q->rd, q->own ? 1 : dl,
               p, q->rd1, d1l);
    if (!q->own) continue;
    for (int64_t i = 0; i < m; i++)
      for (int64_t l = 0; l < W; l++)
        own_term(q, x->d + (t + i) * q->rd + l * dl,
                 x->p + l * ls + p[i * q->rd1 + l * d1l] * rs);
  }
}

/* Chunk [c]'s total of the unit's slices, into [v]: slice by slice where
   they run side by side. An Arg scan's, being exact, is its reduction's,
   numbered on to the chunk's end. */
static void total(const plan *q, work *wk, const unit *x, int64_t c,
                  uint8_t *v) {
  int64_t t0 = c * SCAN_CHUNK;
  int64_t n = q->terms - t0 < SCAN_CHUNK ? q->terms - t0 : SCAN_CHUNK;
  int64_t b0 = c * (SCAN_CHUNK / BLOCK), b1 = b0 + SCAN_CHUNK / BLOCK;
  if (b1 > q->blocks) b1 = q->blocks;
  if (!q->slices)
    /* A row of a scan's outputs, in pieces the blocks' path holds. */
    for (int64_t j = 0, most = ROW / q->size; j < x->W; j += most)
      blocks(q, wk, x->p + j * x->s, x->s, x->W - j < most ? x->W - j : most,
             NULL, b0, b1, v + j * q->size);
  else
    for (int64_t j = 0; j < x->W; j++)
      blocks(q, wk, x->p + j * x->s, 0, 1, NULL, b0, b1, v + j * q->size);
  if (q->z != NULL)
    for (int64_t j = 0; j < x->W; j++) ((nx_cpu_state *)v)[j].n = t0 + n;
}

/* Results [t0, t0 + n) of the unit's slices, from the carry [a], which it
   leaves holding the last ones, term t of slice j at
   x + ((t - t0)·st + j·sj)·w: side by side, or a row of outputs at a
   time. */
static void scan_run(const plan *q, const unit *x, uint8_t *a,
                     const uint8_t *x0, int64_t st, int64_t sj, int64_t t0,
                     int64_t n) {
  uint8_t *y = q->dst->base + (x->d + t0 * q->rd) * q->w;
  if (q->slices || x->W == 1) {
    q->f->scan(a, x0, st, sj, y, q->rd, x->sd, n, (int)x->W);
    return;
  }
  for (int64_t t = 0; t < n; t++) {
    uint8_t *row = y + t * q->rd * q->w;
    if (t + SCAN_AHEAD < n && x->W * q->w <= ROW) {
      /* A staged operand's terms are in a buffer the stage just wrote. */
      if (sj == 1 && !sourced(q))
        PREFETCH_ROW(x0 + (t + SCAN_AHEAD) * st * q->w, x->W * q->w, 0);
#if defined(__aarch64__)
      if (x->sd == 1)
        PREFETCH_ROW(row + SCAN_AHEAD * q->rd * q->w, x->W * q->w, 1);
#endif
    }
    q->f->combine(a, x0 + t * st * q->w, sj, x->W);
    if (x->sd == 1) memcpy(row, a, x->W * q->w);
    else
      for (int64_t j = 0; j < x->W; j++)
        memcpy(row + j * x->sd * q->w, a + j * q->w, q->w);
  }
}

/* Whether a scan's chunk 0 has no carry and starts from its first terms:
   a complex product's, given none. */
static int bare(const plan *q) { return q->first && q->cy == NULL; }

/* The carries of the unit's slices into chunk 0, at [a]: the given ones,
   else values of no term. */
static void start(const plan *q, const unit *x, uint8_t *a) {
  if (q->cy == NULL) {
    fill(q, a, x->W);
    return;
  }
  for (int64_t j = 0; j < x->W; j++)
    memcpy(a + j * q->w, q->cy->base + (x->c + j * x->sc) * q->w, q->w);
}

/* Chunk [c]'s results of the unit's slices, from the carry [a], which it
   leaves holding the chunk's last results. A chunk 0 that has no carry
   starts from its first terms. A staged operand's or a program's terms
   come into the carrier a piece at a time. */
static void rescan(const plan *q, work *wk, const unit *x, int64_t c,
                   uint8_t *a) {
  int64_t t0 = c * SCAN_CHUNK;
  int64_t n = q->terms - t0 < SCAN_CHUNK ? q->terms - t0 : SCAN_CHUNK;
  if (q->z != NULL) {
    run_states(q, wk, x, (nx_cpu_state *)a, x->W, 0, t0, n);
    return;
  }
  if (c == 0 && bare(q)) {
    uint8_t *y = q->dst->base + x->d * q->w;
    _Alignas(16) uint8_t tmp[WIDEST];
    for (int64_t j = 0; j < x->W; j++) {
      memcpy(a + j * q->w, element(q, x->p + j * x->s, tmp), q->w);
      memcpy(y + j * x->sd * q->w, a + j * q->w, q->w);
    }
    t0 = 1;
    n--;
  }
  if (!sourced(q)) {
    scan_run(q, x, a, at(q, x->p + t0 * q->rs[0]), q->rs[0], x->s, t0, n);
    return;
  }
  /* Pieces of LANES * ROW bytes, or of two rows where a row is longer
     than ROW. */
  uint8_t *buf = wk->pieces;
  int64_t row = x->W * q->w;
  int64_t piece = (row <= ROW ? LANES * ROW : 2 * SCAN_ROW) / row;
  for (int64_t t = t0; t < t0 + n; t += piece) {
    int64_t m = t0 + n - t < piece ? t0 + n - t : piece;
    stage_terms(q, wk, x->p, x->s, x->W, t, m, buf);
    scan_run(q, x, a, buf, x->W, 1, t, m);
  }
}

/* The carry [a] of [W] slices into the chunk after [c], from [a] into [c]
   and [c]'s total [v]: the total alone where chunk 0 has no carry, as an
   Arg scan's has none. */
static void carry_on(const plan *q, uint8_t *a, const uint8_t *v, int64_t c,
                     int64_t W) {
  if (c == 0 && (bare(q) || q->z != NULL)) memcpy(a, v, W * q->size);
  else if (q->z == NULL) q->f->combine(a, v, 1, W);
  else
    for (int64_t j = 0; j < W; j++)
      q->z->merge((nx_cpu_state *)a + j, (const nx_cpu_state *)v + j);
}

/* One pass per unit: each chunk's results from the carry, which then takes
   the chunk's total. The last chunk's total carries nowhere. */
static void scan_units(const plan *q, work *wk, int64_t lo, int64_t hi) {
  uint8_t *carry = wk->carry, *a = wk->a, *v = wk->v;
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u);
    start(q, &x, carry);
    for (int64_t c = 0; c * SCAN_CHUNK < q->terms; c++) {
      memcpy(a, carry, x.W * q->size);
      rescan(q, wk, &x, c, a);
      if ((c + 1) * SCAN_CHUNK >= q->terms) break;
      total(q, wk, &x, c, v);
      carry_on(q, carry, v, c, x.W);
    }
  }
}

/* Few slices, long axes: every chunk's total in one job, each unit's
   carries in order, every chunk's results in a second job. A unit's group
   is its chunk. */
static void total_units(const plan *q, work *wk, int64_t lo, int64_t hi) {
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u);
    if (x.g < q->groups - 1)
      total(q, wk, &x, x.g, q->scratch + u * q->row * q->size);
  }
}

/* The unit of span [v] of a job over each unit's first [chunks] chunks,
   SLICES of them a span, and the span's first chunk [c] and count [k]. */
static int64_t span(int64_t v, int64_t chunks, int64_t *c, int *k) {
  int64_t spans = (chunks + SLICES - 1) / SLICES;
  *c = v % spans * SLICES;
  *k = (int)(chunks - *c < SLICES ? chunks - *c : SLICES);
  return v / spans;
}

static void rescan_units(const plan *q, work *wk, int64_t lo, int64_t hi) {
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u);
    rescan(q, wk, &x, x.g, q->scratch + u * q->row * q->size);
  }
}

/* [k] whole chunks from [t0] of the slice [x] from their carries [a], side
   by side: in place, or a staged operand's or a program's a piece of each
   at a time, in the carrier. */
static void quad(const plan *q, work *wk, const unit *x, uint8_t *a,
                 int64_t t0, int k) {
  if (q->z != NULL) {
    run_states(q, wk, x, (nx_cpu_state *)a, k, SCAN_CHUNK, t0, SCAN_CHUNK);
    return;
  }
  int64_t rs = q->rs[0];
  uint8_t *y = q->dst->base + (x->d + t0 * q->rd) * q->w;
  if (!sourced(q)) {
    q->f->scan(a, at(q, x->p + t0 * rs), rs, SCAN_CHUNK * rs, y, q->rd,
               SCAN_CHUNK * q->rd, SCAN_CHUNK, k);
    return;
  }
  uint8_t *buf = wk->terms;
  int64_t piece = LANES * ROW / (4 * q->w);
  for (int64_t i = 0; i < SCAN_CHUNK; i += piece) {
    stage_lanes(q, wk, x, k, SCAN_CHUNK, t0 + i, piece, buf);
    q->f->scan(a, buf, k, 1, y + i * q->rd * q->w, q->rd,
               SCAN_CHUNK * q->rd, piece, k);
  }
}

/* One slice's chunks SLICES at a time, side by side, where a unit is one
   slice: their carries lie side by side in the scratch. A short last chunk
   runs alone, and so does a chunk 0 that has no carry. */
static void rescan_quads(const plan *q, work *wk, int64_t lo, int64_t hi) {
  for (int64_t v = lo; v < hi; v++) {
    int64_t c;
    int k;
    int64_t u = span(v, q->groups, &c, &k);
    unit x = unit_of(q, u * q->groups);
    uint8_t *a = q->scratch + (u * q->groups + c) * q->size;
    if (c == 0 && bare(q)) {
      rescan(q, wk, &x, 0, a);
      c = 1;
      a += q->size;
      k--;
    }
    int full = (c + k) * SCAN_CHUNK <= q->terms ? k : k - 1;
    int64_t t0 = c * SCAN_CHUNK;
    if (full > 0) quad(q, wk, &x, a, t0, full);
    if (full < k) rescan(q, wk, &x, c + full, a + full * q->size);
  }
}

static void carries(const plan *q, work *wk, int64_t lo, int64_t hi) {
  int64_t row = q->row * q->size;
  uint8_t *carry = wk->carry, *v = wk->v;
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u * q->groups);
    start(q, &x, carry);
    for (int64_t c = 0; c < q->groups; c++) {
      uint8_t *s = q->scratch + (u * q->groups + c) * row;
      memcpy(v, s, row);
      memcpy(s, carry, row);
      if (c < q->groups - 1) carry_on(q, carry, v, c, x.W);
    }
  }
}

/* Lays out a scan along the axis [axis] of the destinations [d] and [d1]
   and the operand [x]: the axes before it and those after it coalesce
   apart, and with a program's loads, so that each kept axis steps every
   load evenly. */
static void plan_scan(plan *q, const nx_array *d, const nx_array *d1,
                      const nx_array *x, int axis) {
  int nl = q->fused != NULL ? q->fused->nloads : 0;
  int r = x->rank;
  q->nr = 1;
  q->re[0] = x->dim[axis];
  q->rs[0] = x->dim[r + axis];
  q->rd = d->dim[r + axis];
  q->rd1 = d1->dim[r + axis];
  q->x0 = x->offset;
  q->d0 = d->offset;
  q->d1 = d1->offset;
  q->c0 = q->cy != NULL ? q->cy->offset : 0;
  q->nk = 0;
  /* The carries, where given, coalesce with the rest, after the loads. */
  int nc = q->cy != NULL;
  int parts[2][2] = {{0, axis}, {axis + 1, r}};
  int64_t local[4][NX_MAX_RANK];
  int64_t (*step)[NX_MAX_RANK] = q->fused != NULL ? q->fused->rows : local;
  for (int k = 0; k < 2; k++) {
    int64_t extent[NX_MAX_RANK];
    int n = parts[k][1] - parts[k][0];
    for (int i = 0; i < n; i++) {
      int a = parts[k][0] + i;
      extent[i] = x->dim[a];
      step[0][i] = d->dim[r + a];
      step[1][i] = d1->dim[r + a];
      step[2][i] = x->dim[r + a];
      for (int o = 0; o < nl; o++)
        step[3 + o][i] = q->fused->loads[o].dim[r + a];
      if (nc) step[3 + nl][i] = q->cy->dim[r + a];
    }
    n = nx_coalesce_dims(3 + nl + nc, n, extent, step);
    for (int i = 0; i < n; i++)
      keep(q, extent[i], step[2][i], step[0][i], step[1][i],
           nc ? step[3 + nl][i] : 0);
  }
  choose(q, SCAN);
  int64_t chunks = (q->terms + SCAN_CHUNK - 1) / SCAN_CHUNK;
  q->groups = q->outer * q->chunks < UNITS && chunks > 1 ? chunks : 1;
}

/* Whether the destination's shape [y] of rank [yr] is the result's for an
   operand of shape [x] of rank [xr]: [x] without [axes] for a reduction,
   [x] for a scan along its axis. Sets the terms per output and the
   outputs. */
static int fits(task op, const int64_t *x, int xr, const int64_t *y, int yr,
                const int *axes, int naxes, int64_t *terms, int64_t *outputs) {
  int j = 0, k = 0;
  *terms = *outputs = 1;
  for (int i = 0; i < xr; i++) {
    if (j < naxes && axes[j] == i) {
      *terms *= x[i];
      j++;
      if (op == REDUCE) continue;
    } else
      *outputs *= x[i];
    if (k >= yr || y[k++] != x[i]) return 0;
  }
  return j == naxes && k == yr;
}

/* Lays out [q]'s units for [op] of [x] along [axes] into [y] and [y1]. */
static void lay_out(plan *q, task op, const int *axes, int naxes,
                    const nx_array *y, const nx_array *y1, const nx_array *x) {
  if (op == SCAN) {
    plan_scan(q, y, y1, x, axes[0]);
    return;
  }
  /* The loop: the destinations step 0 along the reduced axes; a program's
     loads follow the operand, so that each of the plan's axes steps every
     load evenly. */
  int rank = x->rank, nl = q->fused != NULL ? q->fused->nloads : 0;
  int64_t extent[NX_MAX_RANK], local[3][NX_MAX_RANK];
  int64_t (*step)[NX_MAX_RANK] = q->fused != NULL ? q->fused->rows : local;
  for (int i = 0, k = 0, j = 0; i < rank; i++) {
    extent[i] = x->dim[i];
    step[2][i] = x->dim[rank + i];
    for (int o = 0; o < nl; o++)
      step[3 + o][i] = q->fused->loads[o].dim[rank + i];
    int reduced = j < naxes && axes[j] == i;
    j += reduced;
    step[0][i] = reduced ? 0 : y->dim[y->rank + k];
    step[1][i] = reduced ? 0 : y1->dim[y1->rank + k];
    k += !reduced;
  }
  q->x0 = x->offset;
  q->d0 = y->offset;
  q->d1 = y1->offset;
  plan_reduce(q, nx_coalesce_dims(3 + nl, rank, extent, step), extent, step);
}

/* A reduction's units, [cost] in all, [term] a term: each unit's blocks,
   or each unit's groups of them then the groups combined, each group's
   value a term. */
static void reduce_jobs(plan *q, int64_t units, int64_t bytes, double cost,
                        double term) {
  if (q->groups == 1) {
    units_job(units, bytes, cost, reduce_units, q);
    return;
  }
  int64_t values = units * q->groups * q->row;
  units_job(units * q->groups, bytes, cost, reduce_units, q);
  units_job(units, values * q->size, (double)values * term, finish_units,
            q);
}

/* Whether [op] by [r] of [x] along the [naxes] axes [axes] fits the
   destinations [y] and [y1]: NX_OK, setting the terms per output and the
   outputs, or NX_SHAPE, an extreme of no term included. */
static int checked(task op, int r, const int *axes, int naxes,
                   const nx_array *y, const nx_array *y1, const nx_array *x,
                   int64_t *terms, int64_t *outputs) {
  if (!fits(op, x->dim, x->rank, y->dim, y->rank, axes, naxes, terms,
            outputs) ||
      !fits(op, x->dim, x->rank, y1->dim, y1->rank, axes, naxes, terms,
            outputs))
    return NX_SHAPE;
  int extreme = r == NX_MONOID_MAX || r == NX_MONOID_MIN ||
                r == NX_CPU_ARG_MAX || r == NX_CPU_ARG_MIN;
  if (op == REDUCE && *terms == 0 && *outputs > 0 && extreme) return NX_SHAPE;
  return NX_OK;
}

/* Reduces or scans, as [op] says, [x], of [dt] or staged into it, into
   [y] and [y1] by [r] in the dtype [dt]: a monoid into [y], [y1] being
   [y]; Logsumexp into [y], [y1] being [y], or Arg into the extreme [y]
   and its int64 position [y1]; or reduced, Moments into the mean [y] and
   the variance [y1]. The destinations are of [dt],
   but Arg's positions, and its extreme where that is of [x]'s dtype, its
   term copied. A monoid's scan starts from the carries [cy] where they are
   not NULL. A term costs [term] (term_cost). NX_OK, NX_SHAPE for shapes
   that do not fit, an extreme of no term included, or -1 if host memory
   runs out. Apart, so that the casts reduce_into runs around it take none
   of its plan's frame. */
static __attribute__((noinline)) int reduce_in(
    task op, int r, int dt, const int *axes, int naxes, const nx_array *y,
    const nx_array *y1, const nx_array *x, const nx_array *cy,
    const fused *z, double term) {
  int64_t terms, outputs;
  int e = checked(op, r, axes, naxes, y, y1, x, &terms, &outputs);
  if (e) return e;
  _Atomic int failed = 0;
  int stateful = r >= NX_CPU_MONOIDS, w = nx_cpu_width(dt), p = part_of(dt);
  int arg = r == NX_CPU_ARG_MAX || r == NX_CPU_ARG_MIN;
  const nx_cpu_stateful *st =
      stateful ? &table.stateful[r - NX_CPU_MONOIDS][dt] : NULL;
  plan q = {.f = stateful ? NULL : &table.monoid[r][dt],
            .r = stateful ? &st->value : NULL,
            .z = stateful && op == SCAN ? &st->scan : NULL,
            .w = w,
            .first = r == NX_MONOID_PROD && p != dt,
            .carrier = dt,
            .staged = x->dtype != dt,
            .own = arg && x->dtype != dt && y->dtype == x->dtype,
            .arg = arg,
            .size = stateful ? (int64_t)sizeof(nx_cpu_state) : w,
            .dst = y,
            .dst1 = y1,
            .src = x,
            .fused = z,
            .cy = cy,
            .terms = terms,
            .factor = {-1, -1},
            .failed = &failed};
  if (outputs > 0 && terms == 0 && op == REDUCE) {
    /* The results of no term: the identity, or a state of none's. */
    _Alignas(16) uint8_t e0[WIDEST], e1[WIDEST];
    if (stateful) q.r->finish(&(nx_cpu_state){0}, 0, e0, e1);
    else identity(r, dt, e0);
    fill_dst(y, e0, w);
    if (y1 != y) fill_dst(y1, e1, nx_cpu_width(y1->dtype));
  }
  if (outputs == 0 || terms == 0) return NX_OK;
  if (!stateful) identity(r, dt, q.id);
  int64_t bytes = terms * outputs * w;
  double cost = (double)terms * (double)outputs * term;
  q.threads = nx_cpu_threads(bytes, cost);
  lay_out(&q, op, axes, naxes, y, y1, x);
  int64_t units = q.outer * q.bands * q.chunks;
  if (q.groups > 1)
    q.scratch = malloc((size_t)(units * q.groups * q.row * q.size));
  int center = op == REDUCE && stateful && st->center.block != NULL;
  if (center) q.centers = malloc((size_t)(units * q.row) * sizeof(double));
  if ((q.groups > 1 && q.scratch == NULL) || (center && q.centers == NULL)) {
    free(q.scratch);
    free(q.centers);
    return -1;
  }
  if (center && r == NX_CPU_MOMENTS && q.groups == 1) {
    q.center = &st->center;
    units_job(units, bytes, cost, center_units, &q);
  } else if (center) {
    /* Logsumexp's passes, or an output's groups on several threads, whose
       center needs them all before its value's pass: each pass reads
       every term, half the cost. */
    q.r = &st->center;
    q.centering = 1;
    reduce_jobs(&q, units, bytes, cost / 2, term / 2);
    q.r = &st->value;
    q.centering = 0;
    reduce_jobs(&q, units, bytes, cost / 2, term / 2);
  } else if (op == REDUCE) reduce_jobs(&q, units, bytes, cost, term);
  else if (q.groups == 1) units_job(units, bytes, cost, scan_units, &q);
  else {
    /* The groups' totals, then their rescans from the carries: each pass
       reads every term, half the cost. */
    units_job(units * q.groups, bytes, cost / 2, total_units, &q);
    units_serial(units, carries, &q);
    if (q.row == 1)
      units_job(units * ((q.groups + SLICES - 1) / SLICES), bytes, cost / 2,
                rescan_quads, &q);
    else units_job(units * q.groups, bytes, cost / 2, rescan_units, &q);
  }
  free(q.scratch);
  free(q.centers);
  return failed ? -1 : NX_OK;
}

/* A C-contiguous array of [y]'s shape in the dtype [c], in scratch at
   [t->base] that the caller gives back: 0, or 1 if host memory runs out. */
static int carrier_copy(const nx_array *y, int c, nx_array *t) {
  int r = y->rank, w = nx_cpu_width(c);
  *t = (nx_array){.dtype = c, .bits = 8 * w, .rank = r};
  int64_t n = 1;
  for (int i = r - 1; i >= 0; i--) {
    t->dim[i] = y->dim[i];
    t->dim[r + i] = n;
    n *= y->dim[i];
  }
  t->base = nx_cpu_take((n > 0 ? n : 1) * w);
  return t->base == NULL;
}

/* Where a copy lives: the thread's kept scratch, which the caller gives
   back, or the heap, which it frees. */
typedef enum { KEPT_SCRATCH, HEAP } home;

/* A C-contiguous copy of [x] at [t->base], in [h]: 0, or 1 if host memory
   runs out. */
static int own_copy(const nx_array *x, nx_array *t, home h) {
  int r = x->rank;
  *t = (nx_array){.dtype = x->dtype, .bits = x->bits, .rank = r};
  int64_t n = 1;
  for (int i = r - 1; i >= 0; i--) {
    t->dim[i] = x->dim[i];
    t->dim[r + i] = n;
    n *= x->dim[i];
  }
  /* An operand of no element has no base, as nx_read gives it. */
  if (n == 0) return 0;
  int64_t bytes = (n * x->bits + 7) / 8;
  t->base = h == KEPT_SCRATCH ? nx_cpu_take(bytes) : malloc((size_t)bytes);
  if (t->base == NULL) return 1;
  nx_array pair[2] = {*t, *x};
  nx_loop l;
  nx_coalesce(2, pair, &l);
  nx_cpu_copy_loop(t->base, x->base, x->bits, &l, nx_cpu_width(x->dtype));
  return 0;
}

/* Casts [a] into [b], of one shape, as a cast does, at a cast's cost.
   Apart, so that its loop's 5 KiB are off the frame its caller's jobs run
   under. */
static __attribute__((noinline)) void cast(const nx_array *b,
                                           const nx_array *a) {
  nx_array pair[2] = {*b, *a};
  nx_loop l;
  nx_coalesce(2, pair, &l);
  nx_cpu_cast_loop(b, a, &l, nx_cpu_cast_cost(a->dtype, b->dtype));
}

/* As reduce_in, of [x] of any dtype in its carrier, into destinations of
   any dtype but Arg's int64 positions: where theirs is not the carrier,
   into carrier_copy's arrays, then rounded once into them, but Arg's
   extreme of [x]'s dtype, which reduce_in copies from [x]. A monoid's scan
   starts from the carries [cy], of the carrier, where they are not
   NULL. A term costs [term]. */
static int reduce_into(task op, int r, const int *axes, int naxes,
                       const nx_array *y, const nx_array *y1,
                       const nx_array *x, const nx_array *cy,
                       const fused *z, double term) {
  int c = nx_cpu_carrier(x->dtype);
  int arg = r == NX_CPU_ARG_MAX || r == NX_CPU_ARG_MIN;
  const nx_array *y2[2] = {y, y1}, *d[2] = {y, y1};
  nx_array t[2];
  int copied[2] = {0, 0}, e = NX_OK;
  for (int k = 0; k < 2 && e == NX_OK; k++) {
    int kept = arg && (k == 1 || y->dtype == x->dtype);
    if (k == 1 && y1 == y) d[1] = d[0];
    else if (!kept && y2[k]->dtype != c) {
      if (carrier_copy(y2[k], c, &t[k])) e = -1;
      else {
        d[k] = &t[k];
        copied[k] = 1;
      }
    }
  }
  if (e == NX_OK)
    e = reduce_in(op, r, c, axes, naxes, d[0], d[1], x, cy, z, term);
  for (int k = 0; k < 2; k++) {
    if (!copied[k]) continue;
    if (e == NX_OK) cast(y2[k], &t[k]);
    nx_cpu_give(t[k].base);
  }
  return e;
}

/* Network scans */

/* A scan by Logsumexp or Moments [r]: the outer axes, those but the
   scanned one, coalesced, with the steps of the result (0), Moments'
   variance (1), the result again for Logsumexp, and the operand (2); the
   scanned axis's extent and steps. [failed] is set where a unit's scratch
   could not be allocated. A unit runs [lanes] slices side by side, along
   the outer axis that steps least. By tiles, [per] items a
   unit run [step] (cpu.h), over its states, [stride] bytes from the last
   unit's; [bad] is set where Moments' fast pass met a value not finite. */
typedef struct {
  int r;
  const nx_array *a[3];
  nx_loop outer;
  int64_t len, along[3];
  int64_t rows, chunks; /* outer positions but the last axis; units along it */
  int64_t lanes;        /* slices a unit */
  const fused *fused;   /* a program's terms, or NULL */
  _Atomic int failed;
  nx_cpu_step step;
  int64_t per, full, stride; /* items a unit, its full tiles */
  uint8_t *states;
  _Atomic int bad;
} network;

/* A unit's terms of a program: [w] slices from the virtual position [p],
   [lane] apart, each stepping [along], in the carrier [c], their held
   values through [held]. */
typedef struct {
  const fused *fused;
  int64_t p, lane, along, w;
  int c;
  uint8_t *held;
} slices;

/* Terms [t0, t0 + tn) of the slices, at most a tile of them, a narrow
   float's of two bytes or fewer held. */
static void fill_slices(const void *ctx, int64_t t0, int64_t tn,
                        uint8_t *out) {
  const slices *c = ctx;
  eval_in(c->fused, c->c, c->p + t0 * c->along, c->lane, c->w, c->along, tn,
          out, c->held);
}

/* The positions of unit [u]'s first result, variance and term [at]; its
   slices. */
static int64_t unit_at(const network *q, int64_t u, int64_t *at) {
  const nx_loop *l = &q->outer;
  int last = l->rank - 1;
  int64_t most = q->lanes, inner = l->extent[last];
  int64_t row = u / q->chunks, from = (u % q->chunks) * most;
  for (int o = 0; o < 3; o++) at[o] = l->first[o];
  for (int i = last - 1; i >= 0; i--) {
    int64_t k = row % l->extent[i];
    row /= l->extent[i];
    for (int o = 0; o < 3; o++) at[o] += k * l->step[o][i];
  }
  for (int o = 0; o < 3; o++) at[o] += from * l->step[o][last];
  return inner - from < most ? inner - from : most;
}

/* Unit [u]'s run of [run] in [scratch], a program's held values [held]
   bytes into it, by [span] or whole. */
static void run_unit(network *q, nx_cpu_network run, int64_t u,
                     uint8_t *scratch, int64_t held, nx_cpu_span *span) {
  const nx_loop *l = &q->outer;
  int last = l->rank - 1, w8 = nx_cpu_width(q->a[2]->dtype);
  int64_t at[3], w = unit_at(q, u, at);
  slices c = {q->fused,       at[2], l->step[2][last], q->along[2], w,
              q->a[2]->dtype, scratch + held};
  nx_cpu_terms src = {fill_slices, &c};
  /* A program's operand is virtual, of no buffer. */
  const uint8_t *x = q->fused != NULL ? NULL : q->a[2]->base + at[2] * w8;
  run(x, q->along[2], l->step[2][last], q->a[0]->base + at[0] * w8,
      q->along[0], l->step[0][last], q->a[1]->base + at[1] * w8, q->along[1],
      l->step[1][last], q->len, w, scratch, q->fused != NULL ? &src : NULL,
      span);
}

/* Runs units [lo, hi) whole, or by tiles where [q->per] is not 0: items
   [lo, hi) of [q->per] a unit, those of one unit in one span. */
static void network_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  network *q = ctx;
  const nx_loop *l = &q->outer;
  int64_t inner = l->extent[l->rank - 1], most = q->lanes;
  int64_t lanes = inner < most ? inner : most;
  int64_t levels = nx_cpu_network_levels(q->len);
  int64_t states = (9 * levels + 3 * NX_CPU_NETWORK_TILE) * lanes;
  if (states < NETWORK_ONE) states = NETWORK_ONE;
  /* The pass's states, then a tile of a program's held values, which are
     no wider than the carrier. */
  int64_t held = NX_CPU_NETWORK_TILE * lanes;
  int w8 = nx_cpu_width(q->a[2]->dtype);
  uint8_t *scratch = nx_cpu_take((states + held) * w8);
  if (scratch == NULL) {
    q->failed = 1;
    return;
  }
  int dt = q->a[2]->dtype;
  nx_cpu_network run =
      q->r == NX_CPU_MOMENTS ? table.moments[dt] : table.logsumexp[dt];
  for (int64_t i = lo; i < hi;) {
    if (q->per == 0) {
      run_unit(q, run, i++, scratch, states * w8, NULL);
      continue;
    }
    int64_t u = i / q->per, q0 = i % q->per;
    int64_t q1 = hi - i < q->per - q0 ? q0 + hi - i : q->per;
    /* CARRIES takes all of a unit's full tiles' states at once. */
    nx_cpu_span s = {q->step, q0, q->step == NX_CPU_CARRIES ? q->full : q1,
                     q->states + u * q->stride, 0};
    run_unit(q, run, u, scratch, states * w8, &s);
    if (s.bad) q->bad = 1;
    i += q1 - q0;
  }
  nx_cpu_give(scratch);
}

/* Scans [x] along [axis], [outputs] slices of [terms] terms, by
   Logsumexp or Moments [r] into [y] and Moments' variance [y1], [y1]
   being [y] for Logsumexp, float32 or float64 of [x]'s dtype, all of
   shapes that fit: NX_OK, or -1 if host memory runs out. [x] is a
   program's virtual operand in its carrier where [z] is not NULL, the
   outer axes coalesced with its loads. Apart, so that the casts around
   it run on a small frame.

   Units at most half the job's threads and slices of two tiles or more
   run by tiles, in three jobs: every full tile's state, the network over
   each unit's tiles' states, every tile's results, each tile's states
   made twice. With more units, the threads the tiles add do not pay for
   the jobs' rounds and the states made twice. On fudge,
   scan/host-logsumexp-f32-1M took 3.39 ms on one thread, 0.57 by
   tiles. */
static __attribute__((noinline)) int network_in(
    int r, int axis, int64_t terms, int64_t outputs, const nx_array *y,
    const nx_array *y1, const nx_array *x, const fused *z, double term) {
  network q = {.r = r, .a = {y, y1, x}, .len = terms, .fused = z};
  nx_loop *l = &q.outer;
  int rank = x->rank, k = 0, nl = z != NULL ? z->nloads : 0;
  int64_t local[3][NX_MAX_RANK];
  int64_t (*step)[NX_MAX_RANK] = z != NULL ? z->rows : local;
  for (int o = 0; o < 3; o++) {
    l->first[o] = q.a[o]->offset;
    q.along[o] = q.a[o]->dim[rank + axis];
  }
  for (int i = 0; i < rank; i++) {
    if (i == axis) continue;
    l->extent[k] = x->dim[i];
    for (int o = 0; o < 3; o++) step[o][k] = q.a[o]->dim[rank + i];
    for (int o = 0; o < nl; o++) step[3 + o][k] = z->loads[o].dim[rank + i];
    k++;
  }
  /* A slice alone keeps one outer axis, of extent 1. */
  if (k == 0) {
    l->extent[0] = 1;
    for (int o = 0; o < 3 + nl; o++) step[o][0] = 0;
    k = 1;
  }
  l->rank = nx_coalesce_dims(3 + nl, k, l->extent, step);
  for (int o = 0; o < 3; o++)
    memcpy(l->step[o], step[o], sizeof(int64_t) * (size_t)l->rank);
  int64_t inner = l->extent[l->rank - 1];
  q.rows = outputs / inner;
  /* Units fewer than UNITS run fewer slices each, so that more threads
     share them. */
  q.lanes = NX_CPU_NETWORK_LANES;
  while (q.lanes > NETWORK_LEAST &&
         q.rows * ((inner + q.lanes - 1) / q.lanes) < UNITS)
    q.lanes /= 2;
  q.chunks = (inner + q.lanes - 1) / q.lanes;
  int64_t units = q.rows * q.chunks, w8 = nx_cpu_width(x->dtype);
  int64_t bytes = 3 * terms * outputs * w8, spans = (terms + PT - 1) / PT;
  double cost = (double)terms * (double)outputs * term;
  if (spans < 2 || 2 * units > nx_cpu_threads(bytes, cost)) {
    nx_cpu_job(units, bytes, cost, network_units, &q);
    return q.failed ? -1 : NX_OK;
  }
  int64_t lanes = inner < q.lanes ? inner : q.lanes;
  q.full = terms / PT;
  q.stride = q.full * 3 * lanes * w8;
  q.states = malloc(units * q.stride);
  if (q.states == NULL) return -1;
  /* The full tiles' states and every tile's results each read the terms,
     half the cost; the network scans one state a full tile. */
  nx_cpu_step steps[3] = {NX_CPU_TOTALS, NX_CPU_CARRIES, NX_CPU_RESULTS};
  int64_t per[3] = {q.full, 1, spans};
  double share[3] = {cost / 2, cost * (double)q.full / (double)terms,
                     cost / 2};
  for (int i = 0; i < 3 && !q.failed; i++) {
    q.step = steps[i];
    q.per = per[i];
    nx_cpu_job(units * q.per, bytes, share[i], network_units, &q);
  }
  /* Where Moments' fast pass met a value not finite, every unit runs its
     exact pass. */
  q.step = NX_CPU_EXACT;
  q.per = 1;
  if (!q.failed && q.bad) nx_cpu_job(units, bytes, cost, network_units, &q);
  free(q.states);
  return q.failed ? -1 : NX_OK;
}

/* As network_in, of [x] of any float dtype in its carrier, into
   destinations of any dtype: a narrow [x] cast whole into carrier_copy's
   array first, or a program's terms converted tile by tile, and
   destinations not of the carrier computed into such arrays, then rounded
   once into them. A term costs [term]. NX_OK, NX_SHAPE for shapes that do
   not fit, or -1 if host memory runs out. */
static int network_into(int r, int axis, const nx_array *y,
                        const nx_array *y1, const nx_array *x,
                        const fused *z, double term) {
  int64_t terms, outputs;
  if (!fits(NETWORK, x->dim, x->rank, y->dim, y->rank, &axis, 1, &terms,
            &outputs) ||
      !fits(NETWORK, x->dim, x->rank, y1->dim, y1->rank, &axis, 1, &terms,
            &outputs))
    return NX_SHAPE;
  if (outputs == 0 || terms == 0) return NX_OK;
  int c = nx_cpu_carrier(x->dtype), e = NX_OK;
  int nd = y1 == y ? 1 : 2;
  const nx_array *a[3] = {y, y1, x}, *d[3] = {y, y1, x};
  nx_array t[3];
  int copied[3] = {0, 0, 0};
  for (int k = 0; k < 3 && e == NX_OK; k++) {
    if (k == 1 && nd == 1) {
      d[1] = d[0];
      continue;
    }
    if (a[k]->dtype == c) continue;
    if (k == 2 && z != NULL) {
      t[2] = *x;
      t[2].dtype = c;
      t[2].bits = 8 * nx_cpu_width(c);
      d[2] = &t[2];
      continue;
    }
    if (carrier_copy(a[k], c, &t[k])) {
      e = -1;
      continue;
    }
    if (k == 2) cast(&t[k], x);
    d[k] = &t[k];
    copied[k] = 1;
  }
  if (e == NX_OK)
    e = network_in(r, axis, terms, outputs, d[0], d[1], d[2], z, term);
  for (int k = 0; k < 3; k++) {
    if (!copied[k]) continue;
    if (e == NX_OK && k < 2) cast(a[k], &t[k]);
    nx_cpu_give(t[k].base);
  }
  return e;
}

/* Entries */

/* What a call computes, as [op] says: [r]'s reduction along the [naxes]
   axes [axes], its scan along [axes[0]], or its network's scan along
   [axes[0]]. */
typedef struct {
  task op;
  int r, naxes;
  int axes[NX_MAX_RANK];
} call;

/* What a term of a call takes (cpu.h, Calls and jobs, scalar; M1 Max and
   fudge, 2026-10-14): a reduction's or a
   scan's by sums and products, extremes, Logsumexp, Arg and Moments,
   along adjacent terms or strided ones whose neighbours along a kept axis
   share their lines. Worst float16, float32 or float64 terms: reduced,
   29 (M1), 57 (fudge), 236 (M1), 27 and 70 (M1); scanned, 101, 194, 350,
   504 and 189 (M1). A reduction of few terms to an output (a lane or
   fewer) takes up to 57 a term of a monoid (FEW_TERM_COST, M1 alone). A
   term no neighbour shares its line with adds a line (NX_CPU_LINE); so
   does a stateful reduction's strided or few one. */
enum { BY_SUMS, BY_EXTREMES, BY_LOGSUMEXP, BY_ARG, BY_MOMENTS };
static const double TERM_COST[2][5] = {{64, 128, 512, 64, 128},
                                       {256, 512, 1024, 1024, 512}};
#define FEW_TERM_COST 128
#define DEAREST_TERM 1024 /* TERM_COST's largest */

/* The row of TERM_COST a term of [c] takes. */
static int term_of(const call *c) {
  if (c->r == NX_CPU_MOMENTS) return BY_MOMENTS;
  if (c->r == NX_CPU_LOGSUMEXP) return BY_LOGSUMEXP;
  if (c->r == NX_CPU_ARG_MAX || c->r == NX_CPU_ARG_MIN) return BY_ARG;
  return c->r <= NX_MONOID_PROD ? BY_SUMS : BY_EXTREMES;
}

/* Whether the operand [x], of the shape [s] and rank [r], steps one
   element along axis [i]: a [copied] one, aliased or a program's virtual
   operand, is read C-contiguous. */
static int unit_step(const nx_array *x, const int64_t *s, int r, int copied,
                     int i) {
  if (copied) {
    for (int j = i + 1; j < r; j++)
      if (s[j] != 1) return 0;
    return 1;
  }
  return x->dim[r + i] == 1 || x->dim[r + i] == -1;
}

/* The price (cpu.h, Calls) of a reduction of no term's fill of [d]
   (fill_dst): streaming its bytes, and ROW_COST for each of its coalesced
   rows, 102 measured for uint8 rows of 2, 64 bytes apart (M1 Max alone,
   2026-10-14); or a
   line for each element where a row's elements lie apart. */
#define ROW_COST 256

static double fill_cost(const nx_array *d) {
  int64_t extent[NX_MAX_RANK], step[1][NX_MAX_RANK];
  int r = rows_of(d, extent, step);
  double rows = 1, n = (double)extent[r - 1];
  for (int i = 0; i < r - 1; i++) rows *= (double)extent[i];
  int apart = step[0][r - 1] != 1 && step[0][r - 1] != -1;
  return nx_cpu_bytes(d) + rows * (apart ? n * NX_CPU_LINE : ROW_COST);
}

/* What a term of the call [c] over the operand [x] of the shape [s] and
   rank [r], [copied] or not (unit_step), costs, [extra] more: its jobs'
   cost a term, and its call's. */
static double term_cost(const call *c, const nx_array *x, const int64_t *s,
                        int r, int copied, double extra) {
  double terms = 1;
  int reduced[NX_MAX_RANK] = {0}, shared = 0;
  for (int i = 0; i < c->naxes; i++)
    if (c->axes[i] < r) {
      terms *= (double)s[c->axes[i]];
      reduced[c->axes[i]] = 1;
    }
  for (int i = 0; i < r && !shared; i++)
    shared = !reduced[i] && s[i] >= LANES && unit_step(x, s, r, copied, i);
  int last = c->naxes > 0 ? c->axes[c->naxes - 1] : r;
  int adjacent = last >= r || unit_step(x, s, r, copied, last);
  int few = c->op == REDUCE && terms <= LANES;
  int t = term_of(c), stateful = t != BY_SUMS && t != BY_EXTREMES;
  double term = TERM_COST[c->op != REDUCE][t];
  if (few && !stateful) term = FEW_TERM_COST;
  if ((!adjacent && !shared) || (stateful && (!adjacent || few)))
    term += NX_CPU_LINE;
  return term + extra;
}

/* The cost (cpu.h, Calls and jobs) of the call [c] over [x] as term_cost
   takes it, into the [nd] destinations [d]: its terms, or, without a
   term, the destinations' fills. A call too small to cost
   NX_CPU_RELEASE_BYTES at the dearest term is bounded so, held without
   term_cost's analysis. */
static double call_cost(const call *c, const nx_array *x, const int64_t *s,
                        int r, int copied, double extra, const nx_array *d,
                        int nd) {
  double numel = 1, cost = 0;
  for (int i = 0; i < r; i++) numel *= (double)s[i];
  if (numel == 0) {
    for (int k = 0; k < nd; k++) cost += fill_cost(&d[k]);
    return cost;
  }
  /* The most a term takes: a line and a scanned term's dearest. */
  double most = NX_CPU_LINE + DEAREST_TERM + extra;
  if (numel * most < NX_CPU_RELEASE_BYTES) return numel * most;
  return numel * term_cost(c, x, s, r, copied, extra);
}

/* Whether the carries [cy], where not NULL, fit [c], a monoid's scan into
   [y]: NX_OK, or NX_SHAPE for another call's or unless they have [y]'s
   shape but one along the scan's axis. */
static int carried(const call *c, const nx_array *cy, const nx_array *y) {
  if (cy == NULL) return NX_OK;
  if (c->op != SCAN || c->r > NX_MONOID_MIN) return NX_SHAPE;
  if (cy->rank != y->rank) return NX_SHAPE;
  for (int i = 0; i < y->rank; i++)
    if (cy->dim[i] != (i == c->axes[0] ? 1 : y->dim[i])) return NX_SHAPE;
  return NX_OK;
}

/* reduce.c's code of the Nx_kernel.reduction [r]: a Monoid's monoid,
   NX_CPU_LOGSUMEXP for Logsumexp, NX_CPU_ARG_MAX or NX_CPU_ARG_MIN for an
   Arg, NX_CPU_MOMENTS for Moments. */
static int code_of(value r) {
  if (Is_long(r))
    return Long_val(r) == NX_REDUCTION_LOGSUMEXP ? NX_CPU_LOGSUMEXP
                                                 : NX_CPU_MOMENTS;
  int k = Int_val(Field(r, 0));
  return Tag_val(r) == 0 ? k : NX_CPU_ARG_MAX + k;
}

/* The dtype of the program [vp]'s one load where [vp] is that load's
   identity (Prog.is_identity), else -1. Reads [vp] alone. */
static int identity_of(value vp) {
  value ins = Field(vp, NX_PROG_INS), outs = Field(vp, NX_PROG_OUTS);
  if (Wosize_val(ins) != 1 || Wosize_val(outs) != 1) return -1;
  value node = Field(Field(vp, NX_PROG_NODES), Int_val(Field(outs, 0)));
  return Tag_val(node) == NX_NODE_IN ? Int_val(Field(ins, 0)) : -1;
}

/* Copies the axis [v] into [a]: 0, or 1 for an axis no array has. */
static int axis_of(value v, int *a) {
  intnat x = Long_val(v);
  if (x < 0 || x >= NX_MAX_RANK) return 1;
  *a = (int)x;
  return 0;
}

/* Whether [r] has two destinations: Arg's and Moments'. */
static int two(int r) {
  return r == NX_CPU_ARG_MAX || r == NX_CPU_ARG_MIN || r == NX_CPU_MOMENTS;
}

/* What a call holds on the C heap where it holds any: a program's copy
   and, past NX_MAX_OPERANDS entries, its tables. A custom block owns them,
   so that an exception nx_read_each raises, Out_of_memory or Rig.Lost,
   leaves them to its finalizer. */
typedef struct {
  nx_cpu_prog *g;
  void *tables;
} call_held;

static void release(value owner) {
  call_held *h = Data_custom_val(owner);
  if (h->g != NULL) nx_cpu_prog_free(h->g);
  free(h->tables);
  h->g = NULL;
  h->tables = NULL;
}

/* release, what a count sized freed with the domain lock released where
   [many] (nx_free_released). */
static void drop(value owner, int many) {
  if (!many) {
    release(owner);
    return;
  }
  call_held h = *(call_held *)Data_custom_val(owner);
  *(call_held *)Data_custom_val(owner) = (call_held){NULL, NULL};
  caml_enter_blocking_section_no_pending();
  nx_cpu_prog_free(h.g);
  free(h.tables);
  caml_leave_blocking_section();
}

static struct custom_operations call_ops = {
    "nx.cpu.reduce",            release,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

/* The operands of a call as the door reads them, from the roots [vdsts],
   [vloads] and [vcarry]: the destinations, each of its own dtype or the
   one [dts] names, then the loads, as [g] loads them, then the carry, in
   the carrier of output [k]'s held dtype. Where [g] is NULL, the program
   is the identity of its one load, of the dtype [dt]: the load is of
   [dt], the carry of its carrier. */
typedef struct {
  const value *vdsts, *vloads, *vcarry;
  const int *dts;
  const nx_cpu_prog *g;
  int dt, nd, nl, k;
} operands;

static int operand(void *ctx, int k, nx_operand *in) {
  const operands *o = ctx;
  if (k < o->nd) {
    value y = Field(*o->vdsts, k);
    int dt = o->dts[k] < 0 ? nx_array_dtype(y) : o->dts[k];
    *in = (nx_operand){y, dt, 1};
  } else if (k < o->nd + o->nl) {
    int i = k - o->nd;
    int dt = o->g == NULL ? o->dt : nx_cpu_prog_in(o->g, i);
    *in = (nx_operand){Field(Field(*o->vloads, i), 0), dt, 0};
  } else {
    int held = o->g == NULL ? o->dt : nx_cpu_prog_held(o->g, o->k);
    *in = (nx_operand){Field(*o->vcarry, 0), nx_cpu_carrier(held), 0};
  }
  return NX_OK;
}

/* Whether the operand [x] has the loop [l]'s shape: NX_OK, else
   NX_SHAPE. */
static int shaped(const nx_array *x, const nx_array *l) {
  if (x->rank != l->rank) return NX_SHAPE;
  for (int i = 0; i < x->rank; i++)
    if (x->dim[i] != l->dim[i]) return NX_SHAPE;
  return NX_OK;
}

/* Computes [c] of the program [vp] over the loop of [rank] axes of extents
   [shape], or of its first destination's shape where [rank] is -1, its
   plain loads [vloads], by the reductions [vrs], an array of pairs of an
   Nx_kernel.reduction and an output, or a scan's one pair where [single],
   in turn into the destinations [vdsts]: a reduction's one, or an Arg's
   extreme and int64 positions, a Moments' mean and variance. Each load
   has the loop's shape and every destination is checked before any is
   written, so that a refusal writes nothing.

   A program that is its one load's identity reads that load in place;
   another's terms are virtual, read through the program (Programs) in
   the output's held dtype, from its copy made before the door. Each
   reduction's loop holds its one or two destinations, the virtual
   operand and the loads, whatever the count of reductions or loads,
   which nothing bounds. The door's tables of every destination and load
   live on the stack up to NX_MAX_OPERANDS entries and past it on the
   heap, which a custom block owns with the program's copy; a program's
   loops, in the block of its terms (fused), taken after the door. So the
   identity of one load into few destinations allocates nothing. */
static value run(const call *c, value vp, value vrs, int single, int rank,
                 const int64_t *shape, value vdsts, value vloads,
                 value vcarry) {
  CAMLparam5(vp, vrs, vdsts, vloads, vcarry);
  CAMLlocal1(owner);
  int nl = (int)Wosize_val(Field(vp, NX_PROG_INS));
  int nouts = (int)Wosize_val(Field(vp, NX_PROG_OUTS));
  if ((int)Wosize_val(vloads) != nl) CAMLreturn(Val_int(NX_SHAPE));
  int nd = (int)Wosize_val(vdsts), e = NX_OK;
  int nc = Is_block(vcarry), dt = identity_of(vp);
  int nr = single ? 1 : (int)Wosize_val(vrs), want = 0;
  /* The tables, on the stack up to NX_MAX_OPERANDS entries, else in one
     block of the heap: the door's descriptors and each destination's
     dtype, -1 for its own, then each reduction's code, output and first
     destination. */
  int nt = nr > NX_MAX_OPERANDS ? nr : NX_MAX_OPERANDS;
  int no = nd + nl + nc > NX_MAX_OPERANDS ? nd + nl + nc : NX_MAX_OPERANDS;
  int heap = nt > NX_MAX_OPERANDS || no > NX_MAX_OPERANDS;
  int owned = heap || dt < 0;
  if (owned) {
    owner = caml_alloc_custom(&call_ops, sizeof(call_held), 0, 1);
    *(call_held *)Data_custom_val(owner) = (call_held){NULL, NULL};
  }
  nx_cpu_prog *g = NULL;
  if (dt < 0) {
    g = nx_cpu_prog_copy(&vp);
    if (g == NULL) caml_raise_out_of_memory();
    ((call_held *)Data_custom_val(owner))->g = g;
  }
  nx_array a_stack[NX_MAX_OPERANDS], *a = a_stack;
  int tables[4 * NX_MAX_OPERANDS], *dts = tables;
  if (heap) {
    uint8_t *b = nx_alloc_released((size_t)no * (sizeof *a + sizeof *dts) +
                                   3 * (size_t)nt * sizeof *dts);
    if (b == NULL) {
      release(owner);
      caml_raise_out_of_memory();
    }
    ((call_held *)Data_custom_val(owner))->tables = b;
    a = (nx_array *)b;
    dts = (int *)(b + (size_t)no * sizeof *a);
  }
  int *rs = dts + no, *ks = rs + nt, *first = ks + nt;
  int nodes = g != NULL ? nx_cpu_prog_nodes(g) : 0;
  /* Tables or a program sized by a count, freed with the lock released. */
  int many = heap || nodes > NX_CPU_ITEMS;
  if (nr < 1 || (c->op != REDUCE && nr != 1)) e = NX_SHAPE;
  /* The reductions are read from the root, letting other domains collect
     every NX_CPU_ITEMS of them. */
  for (int i = 0, next = NX_CPU_ITEMS; !e && i < nr; i++) {
    if (i == next) {
      nx_let_collect();
      next += NX_CPU_ITEMS;
    }
    value pair = single ? vrs : Field(vrs, i);
    rs[i] = code_of(Field(pair, 0));
    ks[i] = Int_val(Field(pair, 1));
    first[i] = want;
    if (want < nd) dts[want] = -1;
    if (two(rs[i]) && want + 1 < nd)
      dts[want + 1] =
          rs[i] == NX_CPU_ARG_MAX || rs[i] == NX_CPU_ARG_MIN ? NX_INT64 : -1;
    want += 1 + two(rs[i]);
    if (ks[i] < 0 || ks[i] >= nouts) e = NX_SHAPE;
  }
  if (!e && want != nd) e = NX_SHAPE;
  if (e) {
    if (owned) drop(owner, many);
    CAMLreturn(Val_int(e));
  }
  /* The door: past NX_MAX_OPERANDS operands in runs. */
  operands o = {&vdsts, &vloads, &vcarry, dts, g, dt, nd, nl, ks[0]};
  int n = nd + nl + nc;
  e = nx_read_each(n, operand, &o, a);
  if (e) {
    if (owned) drop(owner, many);
    CAMLreturn(Val_int(e));
  }
  /* A call whose reductions and nodes cost a release's worth to check and
     price does both with the lock released, whatever its work costs. */
  int early = nr + nodes > NX_CPU_ITEMS, released = 0;
  if (early) released = nx_cpu_release(NX_CPU_RELEASE_BYTES);
  if (rank < 0) {
    rank = a[0].rank;
    shape = a[0].dim;
  }
  const nx_array *cy = nc ? &a[nd + nl] : NULL;
  /* Each load has the loop's shape, and every reduction's destinations
     fit it, before any load is copied. */
  nx_array loop = {.rank = rank};
  memcpy(loop.dim, shape, sizeof(int64_t) * (size_t)rank);
  for (int i = 0; !e && i < nl; i++) e = shaped(&a[nd + i], &loop);
  for (int i = 0; !e && i < nr; i++) {
    int64_t terms, outputs;
    const nx_array *y = &a[first[i]];
    e = checked(c->op, rs[i], c->axes, c->naxes, y,
                two(rs[i]) ? y + 1 : y, &loop, &terms, &outputs);
  }
  if (!e && nc) {
    call ci = *c;
    ci.r = rs[0];
    e = carried(&ci, cy, &a[first[0]]);
  }
  if (e) {
    nx_cpu_reacquire(released);
    nx_done(n, a);
    if (owned) drop(owner, many);
    CAMLreturn(Val_int(e));
  }
  /* The call's price (cpu.h, Calls): each reduction's (call_cost), over
     the identity's load as it lies or a program's C-contiguous virtual
     operand, each term also running the program's kinds
     (nx_cpu_prog_cost), and the copies of aliased loads. */
  double per = g != NULL ? nx_cpu_prog_cost(g) : 0, cost = 0;
  for (int i = 0; i < nl; i++)
    if (a[nd + i].alias)
      per += NX_CPU_STRIDED_COST * nx_cpu_width(a[nd + i].dtype);
  int contiguous = g != NULL || a[nd].alias;
  for (int i = 0; i < nr; i++) {
    call ci = *c;
    ci.r = rs[i];
    cost += call_cost(&ci, &a[nd], shape, rank, contiguous, per, &a[first[i]],
                      1 + two(rs[i]));
  }
  if (!early) released = nx_cpu_release(cost);
  /* A load a destination aliases is read from a copy, since a scan reads
     its terms again after it writes their results: for its chunks'
     totals, and in a Moments scan's exact pass. The identity's one copy is
     in the thread's kept scratch; a program's are on the heap, as many as
     its loads, so that the thread's kept buffers hold no more than the
     program's loop, the reductions' two carrier copies and their work,
     four at most. The program's loop is in the kept scratch, taken after
     the door and given back before its end, since nothing raises in
     between. */
  home h = g != NULL ? HEAP : KEPT_SCRATCH;
  nx_array identity_load, *loads = &identity_load;
  int copied = 0, oom = 0;
  fused *z = g != NULL ? nx_cpu_take(fused_bytes(nl)) : NULL;
  if (g != NULL && z == NULL) oom = 1;
  if (z != NULL) {
    z->nloads = nl;
    fused_tables(z, nl);
    loads = z->loads;
  }
  for (; copied < nl && !oom; copied++) {
    nx_array *l = &a[nd + copied];
    loads[copied] = *l;
    if (l->alias) oom = own_copy(l, &loads[copied], h);
  }
  /* A carry a destination aliases, from a copy, as the results may
     overwrite it before it is read. */
  nx_array kept_carry = {0};
  if (!oom && cy != NULL && cy->alias) oom = own_copy(cy, &kept_carry, HEAP);
  if (kept_carry.base != NULL) cy = &kept_carry;
  /* A program's virtual operand is C-contiguous over the loop's shape,
     the loads'. */
  int index = 0;
  nx_array x = {.rank = rank};
  if (!oom && g != NULL) e = nx_cpu_prog_bind(g, loads, rank, shape, &index);
  if (!oom && !e && g != NULL) {
    z->prog = g;
    int64_t past = 1;
    for (int d = rank - 1; d >= 0; d--) {
      x.dim[d] = shape[d];
      x.dim[rank + d] = past;
      past *= x.dim[d];
    }
    fuse(z, &x, index);
  }
  for (int i = 0; !oom && !e && i < nr; i++) {
    const nx_array *y = &a[first[i]], *y1 = two(rs[i]) ? y + 1 : y;
    const nx_array *terms = &loads[0];
    if (g != NULL) {
      x.dtype = nx_cpu_prog_held(g, ks[i]);
      x.bits = nx_dtype_row_of(x.dtype).bits;
      z->k = ks[i];
      terms = &x;
    }
    call ci = *c;
    ci.r = rs[i];
    double term = term_cost(&ci, &a[nd], shape, rank, contiguous, per);
    e = c->op == NETWORK
            ? network_into(rs[i], c->axes[0], y, y1, terms, z, term)
            : reduce_into(c->op, rs[i], c->axes, c->naxes, y, y1, terms, cy,
                          z, term);
  }
  /* A program that could not have its arenas computed nothing. */
  if (!oom && !e && g != NULL && nx_cpu_prog_failed(g)) e = -1;
  /* A copy that failed left its base NULL, which free and nx_cpu_give
     skip. */
  for (int i = 0; i < copied; i++) {
    if (!a[nd + i].alias) continue;
    if (h == HEAP) free(loads[i].base);
    else nx_cpu_give(loads[i].base);
  }
  free(kept_carry.base);
  if (z != NULL) nx_cpu_give(z);
  nx_cpu_reacquire(released);
  nx_done(n, a);
  if (owned) drop(owner, many);
  if (oom || e < 0) caml_raise_out_of_memory();
  CAMLreturn(Val_int(e));
}

/* Reduces the program [vp] over the loop of shape [vshape] along [vaxes],
   its plain [loads], by the reductions [vrs], pairs of an
   Nx_kernel.reduction and an output, in turn into [vdsts]. */
value nx_cpu_reduce(value vp, value vrs, value vshape, value vaxes,
                    value vdsts, value vloads) {
  call c = {.op = REDUCE, .naxes = (int)Wosize_val(vaxes)};
  int rank = (int)Wosize_val(vshape);
  if (c.naxes > NX_MAX_RANK || rank > NX_MAX_RANK) return Val_int(NX_SHAPE);
  int64_t shape[NX_MAX_RANK];
  for (int d = 0; d < rank; d++)
    if ((shape[d] = Long_val(Field(vshape, d))) < 0) return Val_int(NX_SHAPE);
  for (int i = 0; i < c.naxes; i++)
    if (axis_of(Field(vaxes, i), &c.axes[i])) return Val_int(NX_SHAPE);
  return run(&c, vp, vrs, 0, rank, shape, vdsts, vloads, Val_none);
}

value nx_cpu_reduce_byte(value *argv, int argn) {
  (void)argn;
  return nx_cpu_reduce(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}

/* Scans the program [vp] along [vaxis] by the pair [vr] of a reduction and
   an output, Logsumexp and Moments by the network, its plain [loads], into
   [vdsts], over their first one's shape, from the carries [vcarry] where
   it is Some. */
value nx_cpu_scan(value vp, value vr, value vaxis, value vdsts, value vloads,
                  value vcarry) {
  call c = {.op = Is_long(Field(vr, 0)) ? NETWORK : SCAN, .naxes = 1};
  if (axis_of(vaxis, &c.axes[0])) return Val_int(NX_SHAPE);
  return run(&c, vp, vr, 1, -1, NULL, vdsts, vloads, vcarry);
}

value nx_cpu_scan_byte(value *argv, int argn) {
  (void)argn;
  return nx_cpu_scan(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}
