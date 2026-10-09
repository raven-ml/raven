/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Fill, and the kinds of two and three operands; iota is iota.c's.

   Each kind of two or three operands runs as a row function per dtype, the
   target table's (rows.c). The walk hands the rows a block's rows, and an
   input whose rows step more than one element is staged first. Fill and
   Where move bits, so they run on every dtype of a byte or more by its
   width alone.

   OCaml passes a kind as its value: a constant constructor is its index
   among the type's constant constructors, a constructor with an argument a
   block of that tag. The enums of nx_spec.h list the kinds in that order. */

#include <string.h>

#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "cpu.h"
#include "nx_spec.h"

/* A block holds at most NX_CPU_SLOT bytes of its widest operand, so that
   an input staged into a slot fits it and the block's rows stay in L1.
   Blocks of 64 Ki elements where no input is staged ran fill-f32-1M and
   where-f32-1M slower on kimchi (29-37 against 28-33 us, 44-60 against
   44-54): fewer units for the job's threads to share. */
static int64_t most(int n, const nx_array *a) {
  int w = 1;
  for (int k = 0; k < n; k++)
    if (a[k].bits / 8 > w) w = a[k].bits / 8;
  return NX_CPU_SLOT / w;
}

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

static row3 where_of(int bytes) {
  int i = width_index(bytes);
  return i < 0 ? NULL : nx_cpu_table->where[i];
}

/* Blocks */

typedef struct {
  const nx_array *a;
  int n;
  row2 f2;
  row3 f3;
  const uint8_t *bits; /* a fill's element */
} op;

/* The address of operand [k]'s element at position [p]. */
static inline uint8_t *at(const nx_array *a, int k, int64_t p) {
  return a[k].base + p * (a[k].bits / 8);
}

/* The rows of a plane: operand k's row r at in[k] + r·s1[k] elements of
   its width, stepping s0[k] along it. */
static void each_row(const op *j, int64_t n0, int64_t n1, uint8_t *const *in,
                     const int64_t *s0, const int64_t *s1) {
  const nx_array *a = j->a;
  for (int64_t r = 0; r < n1; r++) {
    uint8_t *p[NX_MAX_OPERANDS];
    for (int k = 0; k < j->n; k++) p[k] = in[k] + r * s1[k] * (a[k].bits / 8);
    if (j->n == 3)
      j->f2(n0, p[0], s0[0], p[1], s0[1], p[2], s0[2]);
    else
      j->f3(n0, p[0], s0[0], p[1], s0[1], p[2], s0[2], p[3], s0[3]);
  }
}

/* An input whose rows step more than one element, as a transposed one's
   do, is copied into a slot first through nx_copy_box's block transposes,
   so the rows read contiguous elements. A block of one row is not staged:
   its gather would read the same strided elements, plus a copy. Out of
   line, so that only a staged block's call takes the slots' stack. */
static __attribute__((noinline)) void staged(const nx_cpu_block *b,
                                             const op *j) {
  _Alignas(64) uint8_t slot[NX_MAX_OPERANDS - 1][NX_CPU_SLOT];
  const nx_array *a = j->a;
  for (int64_t q = 0; q < b->n2; q++) {
    uint8_t *in[NX_MAX_OPERANDS];
    int64_t s0[NX_MAX_OPERANDS], s1[NX_MAX_OPERANDS];
    for (int k = 0; k < j->n; k++) {
      int64_t p = b->at[k] + q * b->s2[k];
      s0[k] = b->s0[k];
      s1[k] = b->s1[k];
      if (k == 0 || s0[k] == 0 || s0[k] == 1) {
        in[k] = at(a, k, p);
        continue;
      }
      nx_copy_box(slot[k - 1], a[k].base,
                  &(nx_box){{1, b->n1, b->n0},
                            {0, p},
                            {{0, b->n0, 1}, {0, b->s1[k], b->s0[k]}}},
                  a[k].bits);
      in[k] = slot[k - 1];
      s0[k] = 1;
      s1[k] = b->n0;
    }
    each_row(j, b->n0, b->n1, in, s0, s1);
  }
}

static void block(const nx_cpu_block *b, void *ctx) {
  const op *j = ctx;
  const nx_array *a = j->a;
  for (int k = 1; k < j->n; k++)
    if (b->s0[k] != 0 && b->s0[k] != 1 && b->n1 > 1) {
      staged(b, j);
      return;
    }
  for (int64_t q = 0; q < b->n2; q++) {
    uint8_t *in[NX_MAX_OPERANDS];
    for (int k = 0; k < j->n; k++) in[k] = at(a, k, b->at[k] + q * b->s2[k]);
    each_row(j, b->n0, b->n1, in, b->s0, b->s1);
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
  if (!(e = nx_coalesce(n, a, &l))) {
    /* A loop of one axis that one block holds, as a small operation's is,
       runs as that block on the calling thread: the walk and the job would
       decide the same at a cost that rivals the block's. */
    int64_t m = most(n, a);
    if (l.rank == 1 && l.extent[0] <= m) {
      nx_cpu_block b = {.n0 = l.extent[0], .n1 = 1, .n2 = 1};
      for (int k = 0; k < n; k++) {
        b.at[k] = l.first[k];
        b.s0[k] = l.step[k][0];
      }
      if (b.n0 > 0) f(&b, j);
    } else
      nx_cpu_walk(n, a, &l, m, f, j);
  }
  nx_done(n, a);
  return Val_int(e);
}

/* Entries */

/* nx_spec.h's code of a Prog.op2 value. */
static int op2_code(value k) {
  return Tag_val(k) == 0 ? Int_val(Field(k, 0))
                         : NX_OP2_EQUAL + Int_val(Field(k, 0));
}

value nx_cpu_apply2(value k, value vd, value vx, value vy) {
  int x = nx_array_dtype(vx), c = op2_code(k);
  row2 f = nx_cpu_table->op2[c][x];
  if (f == NULL) return Val_int(NX_DECLINED);
  int d = c >= NX_OP2_EQUAL ? NX_BOOL : x;
  nx_operand in[3] = {{vd, d, 1}, {vx, x, 0}, {vy, x, 0}};
  op j = {.f2 = f};
  return walk_operands(3, in, block, &j);
}

value nx_cpu_apply3(value k, value vd, value vc, value vx, value vy) {
  int x = nx_array_dtype(vx), c = x;
  row3 f;
  if (Int_val(k) == NX_OP3_WHERE) {
    int bits = nx_dtype_row_of(x).bits;
    f = bits < 8 || nx_array_dtype(vc) == NX_BIT ? NULL : where_of(bits / 8);
    c = NX_BOOL;
  } else
    f = nx_cpu_table->fma[x];
  if (f == NULL) return Val_int(NX_DECLINED);
  nx_operand in[4] = {{vd, x, 1}, {vc, c, 0}, {vx, x, 0}, {vy, x, 0}};
  op j = {.f3 = f};
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
