/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Fill, iota, and the kinds of one, two and three operands.

   Each kind of one, two or three operands runs as a row function per dtype,
   rows.c's, at the dtype its operands compute in: their carrier (cpu.h),
   so that a narrow float computes in float32 and rounds once on the store,
   and int4 computes in int8 and keeps its low bits.
   Where moves bits, so a dtype of a byte or more selects in itself, by its
   width alone. The walk hands the rows a block's rows. An operand read in
   place is one already in its compute dtype; the stage brings any other
   input into a slot first, as it does an input whose rows step more than
   one element, and the unstage stores a slot into a destination of
   another dtype. Fill moves bits too, a sub-byte dtype's code into each
   element.

   OCaml passes a kind as its value: a constant constructor is its index
   among the type's constant constructors, a constructor with an argument a
   block of that tag. The enums of nx_kernel.h list the kinds in that order. */

#include <stdatomic.h>
#include <string.h>

#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "cpu.h"

/* A block holds at most NX_CPU_SLOT bytes of its widest operand in its
   carrier, so that an operand staged into a slot fits it and the block's
   rows stay in L1. Blocks of 64 Ki elements where no input is staged ran
   fill-f32-1M and where-f32-1M slower on kimchi (29-37 against 28-33 us,
   44-60 against 44-54): fewer units for the job's threads to share. The
   slots of the operands a block stages, as a kind of narrow dtypes stages
   all of them, hold at most half the L1 together: on fudge's P-cores, of
   48 KiB L1, add-bf16-1M took 45.1 us in three full slots on six of them,
   36.6 in 8 KiB slots. */

static int64_t most(int n, const nx_array *a) {
  int w = 1, staged = 0;
  for (int k = 0; k < n; k++) {
    int c = nx_cpu_carrier(a[k].dtype);
    if (nx_cpu_width(c) > w) w = nx_cpu_width(c);
    staged += c != a[k].dtype;
  }
  int64_t slot = NX_CPU_SLOT / w;
  int64_t staged_bytes = nx_cpu_l1 / 2;
  int64_t budget = staged > 0 ? staged_bytes / (staged * w) : slot;
  return budget < slot ? budget : slot;
}

/* The operands of a kind: its destination and at most three. */
#define OPERANDS 4

typedef nx_cpu_row1 row1;
typedef nx_cpu_row2 row2;
typedef nx_cpu_row3 row3;

/* Blocks */

typedef struct {
  const nx_array *a;
  int n;
  int as[OPERANDS]; /* the dtype each operand computes in */
  row1 f1;
  row2 f2;
  row3 f3;
  const uint8_t *bits; /* a fill's element */
  atomic_int failed;   /* a block's slots could not be taken */
} op;

/* The address of operand [k]'s element at position [p]. */
static inline uint8_t *at(const nx_array *a, int k, int64_t p) {
  return a[k].base + p * (a[k].bits / 8);
}

/* The rows of a plane: operand k's row r at in[k] + r·s1[k] bytes, stepping
   s0[k] elements along it. */
static void each_row(const op *j, int64_t n0, int64_t n1, uint8_t *const *in,
                     const int64_t *s0, const int64_t *s1) {
  for (int64_t r = 0; r < n1; r++) {
    uint8_t *p[OPERANDS];
    for (int k = 0; k < j->n; k++) p[k] = in[k] + r * s1[k];
    if (j->n == 2)
      j->f1(n0, p[0], s0[0], p[1], s0[1]);
    else if (j->n == 3)
      j->f2(n0, p[0], s0[0], p[1], s0[1], p[2], s0[2]);
    else
      j->f3(n0, p[0], s0[0], p[1], s0[1], p[2], s0[2], p[3], s0[3]);
  }
}

/* Whether the rows read or write operand [k] of [b] in place: it is in its
   compute dtype and, for an input, its rows step by one element or none,
   or the block has one row. A transposed input's rows are worth a copy:
   nx_copy_box moves them in block transposes, and the rows then read
   contiguous elements. */
static int direct(const op *j, const nx_cpu_block *b, int k) {
  if (j->a[k].dtype != j->as[k]) return 0;
  return k == 0 || b->s0[k] == 0 || b->s0[k] == 1 || b->n1 == 1;
}

/* Operand [k]'s plane of [p] into [dst], rows [row] bytes apart, in its
   compute dtype: its bits where that is its own, else through the stage. */
static void stage(const op *j, const nx_cpu_block *p, int k, uint8_t *dst,
                  int64_t row) {
  const nx_array *a = &j->a[k];
  if (j->as[k] != a->dtype) {
    nx_cpu_stage(a, p, k, dst, row);
    return;
  }
  nx_copy_box(dst, a->base,
              &(nx_box){{1, p->n1, p->n0},
                        {0, p->at[k]},
                        {{0, row / (a->bits / 8), 1}, {0, p->s1[k], p->s0[k]}}},
              a->bits);
}

/* A block of an operand not read or written in place goes through a slot,
   plane by plane, the slots in the thread's kept scratch: a slot an
   operand, 64 KiB in all, is more than a thread's stack should hold. */
static void staged(const nx_cpu_block *b, op *j) {
  uint8_t(*slot)[NX_CPU_SLOT] = nx_cpu_take(OPERANDS * NX_CPU_SLOT);
  if (slot == NULL) {
    atomic_store(&j->failed, 1);
    return;
  }
  const nx_array *a = j->a;
  nx_cpu_block p = *b;
  p.n2 = 1;
  for (int64_t q = 0; q < b->n2; q++) {
    uint8_t *in[OPERANDS];
    int64_t s0[OPERANDS], s1[OPERANDS];
    for (int k = 0; k < j->n; k++) {
      p.at[k] = b->at[k] + q * b->s2[k];
      if (direct(j, b, k)) {
        in[k] = at(a, k, p.at[k]);
        s0[k] = b->s0[k];
        s1[k] = b->s1[k] * (a[k].bits / 8);
        continue;
      }
      in[k] = slot[k];
      s0[k] = 1;
      s1[k] = b->n0 * nx_cpu_width(j->as[k]);
      if (k > 0) stage(j, &p, k, slot[k], s1[k]);
    }
    each_row(j, b->n0, b->n1, in, s0, s1);
    if (!direct(j, b, 0)) nx_cpu_unstage(&a[0], &p, 0, slot[0], s1[0], j->as[0]);
  }
  nx_cpu_give(slot);
}

static void block(const nx_cpu_block *b, void *ctx) {
  op *j = ctx;
  const nx_array *a = j->a;
  for (int k = 0; k < j->n; k++)
    if (!direct(j, b, k)) {
      staged(b, j);
      return;
    }
  int64_t s1[OPERANDS];
  for (int k = 0; k < j->n; k++) s1[k] = b->s1[k] * (a[k].bits / 8);
  for (int64_t q = 0; q < b->n2; q++) {
    uint8_t *in[OPERANDS];
    for (int k = 0; k < j->n; k++) in[k] = at(a, k, b->at[k] + q * b->s2[k]);
    each_row(j, b->n0, b->n1, in, b->s0, s1);
  }
}

void nx_cpu_fill_block(const nx_array *d, const nx_cpu_block *b,
                       const uint8_t *bits) {
  int w = nx_cpu_width(d->dtype);
  nx_cpu_row0 f = nx_cpu_rows.fill[nx_cpu_width_index(w)];
  for (int64_t q = 0; q < b->n2; q++)
    for (int64_t r = 0; r < b->n1; r++) {
      int64_t p = b->at[0] + q * b->s2[0] + r * b->s1[0];
      if (d->bits >= 8) f(b->n0, d->base + p * w, b->s0[0], bits);
      else if (b->s0[0] == 1)
        nx_sub_fill_run(d->base, d->bits, p, b->n0, bits[0]);
      else
        for (int64_t i = 0; i < b->n0; i++)
          nx_sub_store(d->base, d->bits, p + i * b->s0[0], bits[0]);
    }
}

/* A fill's block. */
static void fill_block(const nx_cpu_block *b, void *ctx) {
  const op *j = ctx;
  nx_cpu_fill_block(&j->a[0], b, j->bits);
}

/* Reads the [n] operands [in] through the door and walks them with [f],
   [per] the cost of an element (nx_cpu_loop_cost). A kind whose loop is
   one run within a block, each operand in its compute dtype and stepping
   by one element or none, and on the one thread its walk takes, runs the
   one row block would run on the walk's one block, on the calling thread,
   without the block: on the M1 Max an
   add of one float32 element spent 35 of its 93 ns between the loop and
   the row. */
static value walk_operands(int n, const nx_operand *in, nx_cpu_block_fn f,
                           op *j, double per) {
  atomic_init(&j->failed, 0);
  nx_array a[OPERANDS];
  nx_loop l;
  int e = nx_read(n, in, a);
  if (e) return Val_int(e);
  j->a = a;
  j->n = n;
  if ((e = nx_coalesce(n, a, &l))) {
    nx_done(n, a);
    return Val_int(e);
  }
  int released = nx_cpu_release(nx_cpu_loop_cost(n, a, &l, per));
  int64_t block_most = most(n, a);
  int run = f == block && l.rank == 1 && l.extent[0] <= block_most;
  for (int k = 0; k < n && run; k++) {
    int64_t step = l.step[k][0];
    run = a[k].dtype == j->as[k] && (step == 0 || step == 1);
  }
  run = run && nx_cpu_walk_threads(n, a, &l, per) == 1;
  if (!run) nx_cpu_walk(n, a, &l, block_most, per, f, j);
  else if (l.extent[0] > 0) {
    uint8_t *p[OPERANDS];
    int64_t s0[OPERANDS] = {0}, s1[OPERANDS] = {0};
    for (int k = 0; k < n; k++) {
      p[k] = at(a, k, l.first[k]);
      s0[k] = l.step[k][0];
    }
    each_row(j, l.extent[0], 1, p, s0, s1);
  }
  nx_cpu_reacquire(released);
  nx_done(n, a);
  if (atomic_load(&j->failed)) caml_raise_out_of_memory();
  return Val_int(e);
}

/* Entries */

/* [k] is a Prog.unary: nx_kernel.h's op1 codes list them from NX_OP1_NEG. */
value nx_cpu_apply1(value k, value vd, value vx) {
  int x = nx_array_dtype(vx), c = nx_cpu_carrier(x);
  int code = NX_OP1_NEG + Int_val(k);
  nx_operand in[2] = {{vd, x, 1}, {vx, x, 0}};
  op j = {.as = {c, c}, .f1 = nx_cpu_rows.op1[code][c]};
  return walk_operands(2, in, block, &j, nx_cpu_kind_cost(1, code, x, c));
}

value nx_cpu_apply2(value k, value vd, value vx, value vy) {
  int x = nx_array_dtype(vx), c = nx_cpu_carrier(x);
  int code = NX_OP2_CODE(k);
  int d = code >= NX_OP2_EQUAL ? NX_BOOL : x;
  nx_operand in[3] = {{vd, d, 1}, {vx, x, 0}, {vy, x, 0}};
  op j = {.as = {nx_cpu_carrier(d), c, c}, .f2 = nx_cpu_rows.op2[code][c]};
  return walk_operands(3, in, block, &j, nx_cpu_kind_cost(2, code, x, c));
}

value nx_cpu_apply3(value k, value vd, value vc, value vx, value vy) {
  int x = nx_array_dtype(vx), c = nx_array_dtype(vc);
  op j;
  if (Int_val(k) == NX_OP3_WHERE) {
    if (c != NX_BOOL && c != NX_BIT) return Val_int(NX_DTYPE);
    /* A sub-byte dtype selects among its codes, one per byte. */
    int s = nx_dtype_row_of(x).bits < 8 ? nx_cpu_carrier(x) : x;
    j = (op){.as = {s, NX_BOOL, s, s},
             .f3 = nx_cpu_rows.where[nx_cpu_width_index(nx_cpu_width(s))]};
  } else {
    int s = nx_cpu_carrier(x);
    j = (op){.as = {s, s, s, s}, .f3 = nx_cpu_rows.fma[s]};
  }
  nx_operand in[4] = {{vd, x, 1}, {vc, c, 0}, {vx, x, 0}, {vy, x, 0}};
  /* A Where's condition stages too where it is a bit. */
  int where = Int_val(k) == NX_OP3_WHERE;
  double per = nx_cpu_kind_cost(3, Int_val(k), x, j.as[2]) +
               (where ? nx_cpu_stage_cost(c, j.as[1]) : 0);
  return walk_operands(4, in, block, &j, per);
}

value nx_cpu_fill(value vbits, value vd) {
  CAMLparam2(vbits, vd);
  int d = nx_array_dtype(vd), w = nx_cpu_width(d);
  if (caml_string_length(vbits) != (size_t)w) CAMLreturn(Val_int(NX_DTYPE));
  /* The door may run OCaml code, which may move the string. */
  uint8_t element[16];
  memcpy(element, String_val(vbits), (size_t)w);
  nx_operand in[1] = {{vd, d, 1}};
  op j = {.bits = element};
  CAMLreturn(walk_operands(1, in, fill_block, &j, w));
}

/* Iota

   Each element is its index along axis [axis], an int64.

   A C-contiguous destination, as a fresh one is, runs as a job over its
   elements in C order: along it the index is constant for [inner]
   elements, the product of the extents past [axis], and counts up to the
   axis's extent and back to 0. Another layout runs on one thread, an
   odometer over its axes. */

typedef struct {
  int64_t *base;
  int64_t numel, inner, ext;
} iota_job;

/* Elements [lo, hi) of a C-contiguous iota. */
static void iota_run(int64_t *d, int64_t lo, int64_t hi, int64_t inner,
                     int64_t ext) {
  int64_t i = lo, v = lo / inner % ext;
  if (inner == 1) {
    /* To the period's end, whole periods, then the rest. */
    int64_t head = (ext - v) % ext < hi - i ? (ext - v) % ext : hi - i;
    for (int64_t k = 0; k < head; k++) d[i + k] = v + k;
    i += head;
    for (; hi - i >= ext; i += ext)
      for (int64_t e = 0; e < ext; e++) d[i + e] = e;
    for (int64_t e = 0; i < hi; i++, e++) d[i] = e;
    return;
  }
  int64_t r = lo % inner;
  while (i < hi) {
    int64_t n = inner - r < hi - i ? inner - r : hi - i;
    for (int64_t k = 0; k < n; k++) d[i + k] = v;
    i += n;
    r = 0;
    if (++v == ext) v = 0;
  }
}

/* An iota of [a], of any layout, along [axis]. */
static void iota_odometer(const nx_array *a, int axis) {
  int r = a->rank;
  int64_t idx[NX_MAX_RANK] = {0}, n = a->dim[r - 1], s = a->dim[2 * r - 1];
  for (;;) {
    int64_t p = a->offset;
    for (int i = 0; i < r - 1; i++) p += idx[i] * a->dim[r + i];
    int64_t *d = (int64_t *)a->base + p;
    if (axis == r - 1)
      for (int64_t i = 0; i < n; i++) d[i * s] = i;
    else
      for (int64_t i = 0; i < n; i++) d[i * s] = idx[axis];
    int i = r - 2;
    while (i >= 0 && ++idx[i] == a->dim[i]) idx[i--] = 0;
    if (i < 0) return;
  }
}

/* Elements of an iota's unit of work. */
#define IOTA_UNIT (16 * 1024)

/* What an iota element takes (cpu.h, Calls and jobs; M1, 2026-10-14): a
   contiguous one, its int64's store (streaming); an odometer's at most
   (scalar), the same loop at float32 having taken up to 124. */
#define IOTA_COST 8
#define ODOMETER_COST 2048

/* Units [lo, hi) of the job: the last one ends at the last element. */
static void iota_body(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const iota_job *j = ctx;
  int64_t end = hi * IOTA_UNIT < j->numel ? hi * IOTA_UNIT : j->numel;
  iota_run(j->base, lo * IOTA_UNIT, end, j->inner, j->ext);
}

/* The external, into an int64 destination: nx_cpu.ml's map calls it for
   a program of one Coord. */
value nx_cpu_iota(value vaxis, value vd) {
  int axis = Int_val(vaxis);
  nx_operand in[1] = {{vd, NX_INT64, 1}};
  nx_array a[1];
  int e = nx_read(1, in, a);
  if (e) return Val_int(e);
  if (axis < 0 || axis >= a->rank) {
    nx_done(1, a);
    return Val_int(NX_SHAPE);
  }
  int64_t numel = 1, inner = 1;
  for (int i = 0; i < a->rank; i++) numel *= a->dim[i];
  for (int i = axis + 1; i < a->rank; i++) inner *= a->dim[i];
  int job = a->flags & NX_CONTIGUOUS;
  double cost = (double)numel * (job ? IOTA_COST : ODOMETER_COST);
  int released = nx_cpu_release(cost);
  if (numel > 0 && job) {
    iota_job j = {(int64_t *)a->base + a->offset, numel, inner, a->dim[axis]};
    int64_t units = (numel + IOTA_UNIT - 1) / IOTA_UNIT;
    nx_cpu_job(units, numel * 8, cost, iota_body, &j);
  } else if (numel > 0)
    iota_odometer(a, axis);
  nx_cpu_reacquire(released);
  nx_done(1, a);
  return Val_int(NX_OK);
}
