/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Fill, and the kinds of one, two and three operands; iota is iota.c's.

   Each kind of one, two or three operands runs as a row function per dtype,
   the target table's (rows.c), at the dtype its operands compute in: their
   carrier (cpu.h), so that a narrow float computes in float32 and rounds
   once on the store, and int4 computes in int8 and keeps its low bits.
   Where moves bits, so a dtype of a byte or more selects in itself, by its
   width alone. The walk hands the rows a block's rows. An operand read in
   place is one already in its compute dtype; the stage brings any other
   input into a slot first, as it does an input whose rows step more than
   one element, and the unstage stores a slot into a destination of
   another dtype. Fill moves bits too, at every dtype of a byte or more.

   OCaml passes a kind as its value: a constant constructor is its index
   among the type's constant constructors, a constructor with an argument a
   block of that tag. The enums of nx_spec.h list the kinds in that order. */

#include <string.h>

#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "cpu.h"
#include "nx_spec.h"

/* A block holds at most NX_CPU_SLOT bytes of its widest operand in its
   carrier, so that an operand staged into a slot fits it and the block's
   rows stay in L1. Blocks of 64 Ki elements where no input is staged ran
   fill-f32-1M and where-f32-1M slower on kimchi (29-37 against 28-33 us,
   44-60 against 44-54): fewer units for the job's threads to share. */
static int64_t most(int n, const nx_array *a) {
  int w = 1;
  for (int k = 0; k < n; k++) {
    int c = nx_cpu_width(nx_cpu_carrier(a[k].dtype));
    if (c > w) w = c;
  }
  return NX_CPU_SLOT / w;
}

typedef nx_cpu_row1 row1;
typedef nx_cpu_row2 row2;
typedef nx_cpu_row3 row3;

/* The index of a width of 1, 2, 4, 8 or 16 bytes in the target table's
   rows by width, or -1. */
static int width_index(int bytes) {
  switch (bytes) {
    case 1: return 0;
    case 2: return 1;
    case 4: return 2;
    case 8: return 3;
    case 16: return 4;
    default: return -1;
  }
}

/* Blocks */

typedef struct {
  const nx_array *a;
  int n;
  int as[NX_MAX_OPERANDS]; /* the dtype each operand computes in */
  row1 f1;
  row2 f2;
  row3 f3;
  const uint8_t *bits; /* a fill's element */
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
    uint8_t *p[NX_MAX_OPERANDS];
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
   plane by plane. Out of line, so that only such a block's call takes the
   slots' stack. */
static __attribute__((noinline)) void staged(const nx_cpu_block *b,
                                             const op *j) {
  _Alignas(64) uint8_t slot[NX_MAX_OPERANDS][NX_CPU_SLOT];
  const nx_array *a = j->a;
  nx_cpu_block p = *b;
  p.n2 = 1;
  for (int64_t q = 0; q < b->n2; q++) {
    uint8_t *in[NX_MAX_OPERANDS];
    int64_t s0[NX_MAX_OPERANDS], s1[NX_MAX_OPERANDS];
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
}

static void block(const nx_cpu_block *b, void *ctx) {
  const op *j = ctx;
  const nx_array *a = j->a;
  for (int k = 0; k < j->n; k++)
    if (!direct(j, b, k)) {
      staged(b, j);
      return;
    }
  int64_t s1[NX_MAX_OPERANDS];
  for (int k = 0; k < j->n; k++) s1[k] = b->s1[k] * (a[k].bits / 8);
  for (int64_t q = 0; q < b->n2; q++) {
    uint8_t *in[NX_MAX_OPERANDS];
    for (int k = 0; k < j->n; k++) in[k] = at(a, k, b->at[k] + q * b->s2[k]);
    each_row(j, b->n0, b->n1, in, b->s0, s1);
  }
}

/* A fill stores its element through the target table's row of its width. */
static void fill_block(const nx_cpu_block *b, void *ctx) {
  const op *j = ctx;
  nx_cpu_row0 f = nx_cpu_table->fill[width_index(j->a[0].bits / 8)];
  for (int64_t q = 0; q < b->n2; q++)
    for (int64_t r = 0; r < b->n1; r++)
      f(b->n0, at(j->a, 0, b->at[0] + q * b->s2[0] + r * b->s1[0]), b->s0[0],
        j->bits);
}

/* Reads the [n] operands [in] through the door and walks them with [f]. */
static value walk_operands(int n, const nx_operand *in, nx_cpu_block_fn f,
                           op *j) {
  nx_array a[NX_MAX_OPERANDS];
  nx_loop l;
  int e = nx_read(n, in, a);
  if (e) return Val_int(e);
  j->a = a;
  j->n = n;
  if (!(e = nx_coalesce(n, a, &l))) nx_cpu_walk(n, a, &l, most(n, a), f, j);
  nx_done(n, a);
  return Val_int(e);
}

/* Entries */

/* [k] is a Prog.unary: nx_spec.h's op1 codes list them from NX_OP1_NEG. */
value nx_cpu_apply1(value k, value vd, value vx) {
  int x = nx_array_dtype(vx), c = nx_cpu_carrier(x);
  row1 f = nx_cpu_table->op1[NX_OP1_NEG + Int_val(k)][c];
  if (f == NULL) return Val_int(NX_DECLINED);
  nx_operand in[2] = {{vd, x, 1}, {vx, x, 0}};
  op j = {.as = {c, c}, .f1 = f};
  return walk_operands(2, in, block, &j);
}

/* nx_spec.h's code of a Prog.op2 value. */
static int op2_code(value k) {
  return Tag_val(k) == 0 ? Int_val(Field(k, 0))
                         : NX_OP2_EQUAL + Int_val(Field(k, 0));
}

value nx_cpu_apply2(value k, value vd, value vx, value vy) {
  int x = nx_array_dtype(vx), c = nx_cpu_carrier(x), code = op2_code(k);
  row2 f = nx_cpu_table->op2[code][c];
  if (f == NULL) return Val_int(NX_DECLINED);
  int d = code >= NX_OP2_EQUAL ? NX_BOOL : x;
  nx_operand in[3] = {{vd, d, 1}, {vx, x, 0}, {vy, x, 0}};
  op j = {.as = {nx_cpu_carrier(d), c, c}, .f2 = f};
  return walk_operands(3, in, block, &j);
}

value nx_cpu_apply3(value k, value vd, value vc, value vx, value vy) {
  int x = nx_array_dtype(vx), c = nx_array_dtype(vc);
  op j;
  if (Int_val(k) == NX_OP3_WHERE) {
    if (c != NX_BOOL && c != NX_BIT) return Val_int(NX_DTYPE);
    /* A sub-byte dtype selects among its codes, one per byte. */
    int s = nx_dtype_row_of(x).bits < 8 ? nx_cpu_carrier(x) : x;
    j = (op){.as = {s, NX_BOOL, s, s},
             .f3 = nx_cpu_table->where[width_index(nx_cpu_width(s))]};
  } else {
    int s = nx_cpu_carrier(x);
    j = (op){.as = {s, s, s, s}, .f3 = nx_cpu_table->fma[s]};
  }
  if (j.f3 == NULL) return Val_int(NX_DECLINED);
  nx_operand in[4] = {{vd, x, 1}, {vc, c, 0}, {vx, x, 0}, {vy, x, 0}};
  return walk_operands(4, in, block, &j);
}

value nx_cpu_fill(value vbits, value vd) {
  CAMLparam2(vbits, vd);
  int d = nx_array_dtype(vd), bits = nx_dtype_row_of(d).bits;
  if (bits < 8 || width_index(bits / 8) < 0) CAMLreturn(Val_int(NX_DECLINED));
  if (caml_string_length(vbits) != (size_t)(bits / 8))
    CAMLreturn(Val_int(NX_DTYPE));
  /* The door may run OCaml code, which may move the string. */
  uint8_t element[16];
  memcpy(element, String_val(vbits), bits / 8);
  nx_operand in[1] = {{vd, d, 1}};
  op j = {.bits = element};
  CAMLreturn(walk_operands(1, in, fill_block, &j));
}
