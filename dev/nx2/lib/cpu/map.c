/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Maps: a scalar program run over blocks.

   A map's loop operands are its outputs, written, then its loads, then one
   coordinate per axis its Coord nodes read: an int64 operand of no buffer
   that steps one along that axis and none along the others, so that
   coalescing keeps the axes it needs and its position at an index is the
   index along its axis. The loop is walked in blocks. Per block, plane by
   plane, each node runs in order over values held in slots on the stack,
   through the target table's rows, as apply.c's kinds do:

   - a load held in its own dtype, its rows contiguous, is read in place,
     and any other is staged into its slot;
   - a constant is read in place with a step of 0;
   - a coordinate fills its slot with its positions;
   - Copy, a Cast into the operand's own dtype and a Bitcast between byte
     wide dtypes read their operand's value;
   - each output's value is stored into its destination.

   A node's value is held as its dtype's element exactly, so that a program
   gives the bits its nodes give run one by one. float16, bfloat16 and the
   float8 dtypes are held as their codes, decoded into float32 for a kind
   and encoded once after it, so Copy, Where and Bitcast move their bits as
   apply does; int4, uint4, float4 and bit are held in their carriers
   (cpu.h), brought back to the dtype after each kind.

   Slots go by liveness: a node's slot is free once its last reader has
   run. A program that holds more than SLOTS values at once, has a padded
   load, needs more operands than a loop holds, or bitcasts a sub-byte
   dtype is declined: its caller runs the nodes one by one. */

#include <stdlib.h>
#include <string.h>

#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "cpu.h"
#include "nx_spec.h"

/* Values a program holds at once, each in a slot of SLOT_BYTES, and the
   slots a node decodes its operands into and computes in before it
   encodes. A block takes at most SLOT_BYTES of its widest element, so that
   all of them, 48 KiB, stay in the L1 of kimchi's performance cores. */
#define SLOTS 8
#define TEMPS 4
#define SLOT_BYTES 4096

typedef uint8_t slot_t[SLOT_BYTES];

/* The dtype a value of [dt] is held in: a byte-wide narrow float's own,
   its codes, else its carrier. */
static int held(int dt) {
  switch (dt) {
    case NX_FLOAT16:
    case NX_BFLOAT16:
    case NX_FLOAT8_E4M3FN:
    case NX_FLOAT8_E5M2: return dt;
    default: return nx_cpu_carrier(dt);
  }
}

static int width(int dt) { return nx_cpu_width(dt); }

/* A node as the interpreter runs it. */
typedef struct {
  int tag, kind, dt;
  int x[3];   /* operand nodes */
  int arity;  /* of the kinds of one to three operands; 0 otherwise */
  int slot;   /* its slot, or -1 for a value read elsewhere */
  int alias;  /* the node whose value it is, or -1 */
  int last;   /* the last node that reads its value; an output's is past all */
  int k;      /* In and Coord: its loop operand */
  int out;    /* the output it is written into in place, or -1 */
  void *f;    /* a kind's row */
  uint8_t bits[16]; /* Const: its element, held */
} step;

typedef struct {
  const nx_array *a;
  int n, nouts, nnodes;
  const step *s;
  const int32_t *outs;
  int direct[NX_MAX_OPERANDS]; /* whether a load may be read in place */
} prog;

/* A value of a plane: row r's element i at p + r·s1 + i·s0·w bytes, w its
   held dtype's width. */
typedef struct {
  uint8_t *p;
  int64_t s0, s1;
} view;

/* [n] elements of [dt] at [d], each its carrier's element at [s] brought to
   [dt]: int4 sign-extended from its low bits, uint4 masked, float4 rounded
   through its code in [t]. */
static void round_held(int dt, const uint8_t *s, uint8_t *d, uint8_t *t,
                       int64_t n) {
  switch (dt) {
    case NX_INT4:
      for (int64_t i = 0; i < n; i++)
        d[i] = (uint8_t)((int8_t)(uint8_t)(s[i] << 4) >> 4);
      return;
    case NX_UINT4:
      for (int64_t i = 0; i < n; i++) d[i] = s[i] & 15;
      return;
    case NX_FLOAT4_E2M1FN:
      nx_cpu_table->convert[NX_FLOAT32][dt](s, t, n);
      nx_cpu_table->convert[dt][NX_FLOAT32](t, d, n);
      return;
    default:
      if (s != d) memcpy(d, s, (size_t)(n * width(dt)));
  }
}

/* Whether load [k] of plane [p] is read in place. */
static int in_place(const prog *j, const nx_cpu_block *p, int k) {
  const nx_array *a = &j->a[k];
  return j->direct[k] && a->dtype == held(a->dtype) && p->s0[k] == 1;
}

/* Where node [i], a node with a slot, computes its value in plane [p]: its
   output's destination where it is written in place there, else its
   slot. */
static view home(const prog *j, const nx_cpu_block *p, slot_t *slot, int i) {
  const step *s = &j->s[i];
  int64_t w = width(held(s->dt));
  int o = s->out;
  if (o >= 0 && p->s0[o] == 1)
    return (view){j->a[o].base + p->at[o] * w, 1, p->s1[o] * w};
  return (view){slot[s->slot], 1, p->n0 * w};
}

/* Node [i]'s value in plane [p]. */
static view value_of(const prog *j, const nx_cpu_block *p, slot_t *slot,
                     int i) {
  const step *s = &j->s[i];
  if (s->alias >= 0) return value_of(j, p, slot, s->alias);
  int64_t w = width(held(s->dt));
  if (s->tag == NX_NODE_CONST) return (view){(uint8_t *)s->bits, 0, 0};
  if (s->tag == NX_NODE_IN && in_place(j, p, s->k)) {
    const nx_array *a = &j->a[s->k];
    return (view){a->base + p->at[s->k] * w, 1, p->s1[s->k] * w};
  }
  return home(j, p, slot, i);
}

/* [v], of the dtype [dt], with contiguous rows: itself, or copied into
   [t]. */
static view runs(view v, int dt, const nx_cpu_block *p, uint8_t *t) {
  if (v.s0 == 1) return v;
  int64_t w = width(dt);
  for (int64_t r = 0; r < p->n1; r++)
    for (int64_t i = 0; i < p->n0; i++)
      memcpy(t + (r * p->n0 + i) * w, v.p + r * v.s1 + i * v.s0 * w,
             (size_t)w);
  return (view){t, 1, p->n0 * w};
}

/* Operand [o] of node [i] in its carrier: decoded into [t] where it is
   held as codes. */
static view wide_of(const prog *j, const nx_cpu_block *p, slot_t *slot, int i,
                    int o, uint8_t *t) {
  int x = j->s[i].x[o];
  int dt = j->s[x].dt, c = nx_cpu_carrier(dt);
  view v = value_of(j, p, slot, x);
  if (held(dt) == c) return v;
  nx_cpu_run decode = nx_cpu_table->convert[dt][c];
  if (v.s0 == 0) {
    decode(v.p, t, 1);
    return (view){t, 0, 0};
  }
  for (int64_t r = 0; r < p->n1; r++)
    decode(v.p + r * v.s1, t + r * p->n0 * width(c), p->n0);
  return (view){t, 1, p->n0 * width(c)};
}

/* Runs node [i] over plane [p] into its slot. */
static void run(const prog *j, const nx_cpu_block *p, slot_t *slot,
                slot_t *tmp, int i) {
  const step *s = &j->s[i];
  int64_t n0 = p->n0, n1 = p->n1;
  int dt = s->dt, c = nx_cpu_carrier(dt), h = held(dt);
  uint8_t *d = slot[s->slot];
  if (s->tag == NX_NODE_IN) {
    if (in_place(j, p, s->k)) return;
    const nx_array *a = &j->a[s->k];
    if (h != a->dtype) {
      nx_cpu_stage(a, p, s->k, d, n0 * width(h));
      return;
    }
    nx_copy_box(d, a->base,
                &(nx_box){{1, n1, n0},
                          {0, p->at[s->k]},
                          {{0, n0, 1}, {0, p->s1[s->k], p->s0[s->k]}}},
                a->bits);
    return;
  }
  if (s->tag == NX_NODE_COORD) {
    int64_t *v = (int64_t *)d;
    for (int64_t r = 0; r < n1; r++)
      for (int64_t e = 0; e < n0; e++)
        v[r * n0 + e] = p->at[s->k] + r * p->s1[s->k] + e * p->s0[s->k];
    return;
  }
  view t = home(j, p, slot, i);
  if (s->tag == NX_NODE_OP1 && s->kind == NX_OP1_CAST) {
    /* The operand in its carrier, converted as a cast does into the dtype's
       elements: in place where the dtype is held as itself, else into
       [tmp[2]], then to its carrier. */
    int xc = nx_cpu_carrier(j->s[s->x[0]].dt);
    view v = runs(wide_of(j, p, slot, i, 0, tmp[0]), xc, p, tmp[1]);
    nx_cpu_run convert = nx_cpu_table->convert[xc][dt];
    if (h == dt) {
      for (int64_t r = 0; r < n1; r++)
        convert(v.p + r * v.s1, t.p + r * t.s1, n0);
      return;
    }
    for (int64_t r = 0; r < n1; r++)
      convert(v.p + r * v.s1, tmp[2] + r * n0 * width(dt), n0);
    if (dt == NX_FLOAT4_E2M1FN)
      nx_cpu_table->convert[dt][NX_FLOAT32](tmp[2], d, n0 * n1);
    else
      round_held(dt, tmp[2], d, NULL, n0 * n1);
    return;
  }
  /* Where selects held values; a kind computes in its operands' carrier,
     in place where the result is held in its carrier. */
  int where = s->tag == NX_NODE_OP3 && s->kind == NX_OP3_WHERE;
  view v[3];
  for (int o = 0; o < s->arity; o++)
    v[o] = where ? value_of(j, p, slot, s->x[o])
                 : wide_of(j, p, slot, i, o, tmp[o]);
  int64_t w = where ? width(h) : width(c);
  view o = where || h == c ? t : (view){tmp[3], 1, n0 * w};
  for (int64_t r = 0; r < n1; r++) {
    uint8_t *y = o.p + r * o.s1;
    uint8_t *x0 = v[0].p + r * v[0].s1;
    if (s->arity == 1) {
      ((nx_cpu_row1)s->f)(n0, y, 1, x0, v[0].s0);
      continue;
    }
    uint8_t *x1 = v[1].p + r * v[1].s1;
    if (s->arity == 2) {
      ((nx_cpu_row2)s->f)(n0, y, 1, x0, v[0].s0, x1, v[1].s0);
      continue;
    }
    ((nx_cpu_row3)s->f)(n0, y, 1, x0, v[0].s0, x1, v[1].s0,
                        v[2].p + r * v[2].s1, v[2].s0);
  }
  if (where) return;
  /* A kind's result in its carrier: encoded where its dtype is held as
     codes, else brought to the dtype in its slot. */
  if (h != c)
    for (int64_t r = 0; r < n1; r++)
      nx_cpu_table->convert[c][dt](tmp[3] + r * n0 * w, t.p + r * t.s1, n0);
  else if (dt != c)
    round_held(dt, d, d, tmp[3], n0 * n1);
}

/* Stores output [o]'s value of plane [p] into its destination. */
static void store(const prog *j, const nx_cpu_block *p, slot_t *slot,
                  slot_t *tmp, int o) {
  const nx_array *a = &j->a[o];
  int dt = a->dtype, h = held(dt), i = j->outs[o];
  if (j->s[i].out == o && p->s0[o] == 1) return;
  view v = value_of(j, p, slot, i);
  if (h == dt) {
    int64_t w = width(dt);
    nx_copy_box(a->base, v.p,
                &(nx_box){{1, p->n1, p->n0},
                          {p->at[o], 0},
                          {{0, p->s1[o], p->s0[o]}, {0, v.s1 / w, v.s0}}},
                a->bits);
    return;
  }
  v = runs(v, h, p, tmp[0]);
  nx_cpu_unstage(a, p, o, v.p, v.s1, h);
}

static void block(const nx_cpu_block *b, void *ctx) {
  const prog *j = ctx;
  _Alignas(64) slot_t slot[SLOTS], tmp[TEMPS];
  nx_cpu_block p = *b;
  p.n2 = 1;
  for (int64_t q = 0; q < b->n2; q++) {
    for (int k = 0; k < j->n; k++) p.at[k] = b->at[k] + q * b->s2[k];
    for (int i = 0; i < j->nnodes; i++)
      if (j->s[i].slot >= 0) run(j, &p, slot, tmp, i);
    for (int o = 0; o < j->nouts; o++) store(j, &p, slot, tmp, o);
  }
}

/* Decoding */

/* The row of the kind [s], over operands of the dtype [xt] after a first
   one of [ct], or NULL where the table has none. */
static void *row_of(const step *s, int ct, int xt) {
  const nx_cpu_target *t = nx_cpu_table;
  int c = nx_cpu_carrier(xt);
  switch (s->tag) {
    case NX_NODE_OP1: return (void *)t->op1[s->kind][c];
    case NX_NODE_OP2: return (void *)t->op2[s->kind][c];
    default: break;
  }
  if (s->kind == NX_OP3_FMA) return (void *)t->fma[c];
  if (ct != NX_BOOL && ct != NX_BIT) return NULL;
  switch (width(held(xt))) {
    case 1: return (void *)t->where[0];
    case 2: return (void *)t->where[1];
    case 4: return (void *)t->where[2];
    case 8: return (void *)t->where[3];
    default: return (void *)t->where[4];
  }
}

/* A constant's element held, from the bits of its dtype's element. */
static void const_of(step *s, const uint8_t *bits) {
  int dt = s->dt;
  memcpy(s->bits, bits, 16);
  if (dt == NX_FLOAT4_E2M1FN)
    nx_cpu_table->convert[dt][NX_FLOAT32](bits, s->bits, 1);
  else if (held(dt) != dt)
    round_held(dt, bits, s->bits, NULL, 1);
}

/* Fills [s] from the program [g], whose loop has [n] operands before its
   coordinates, and answers NX_OK, or NX_DECLINED; on NX_OK [n] counts the
   coordinates too and [axis] gives each one's axis. */
static int decode(const nx_prog *g, step *s, int *n, int *axis) {
  int nn = g->nnodes, nouts = g->nouts, base = *n;
  const int32_t *outs = nx_prog_outs(g);
  int coord[NX_MAX_RANK];
  for (int i = 0; i < NX_MAX_RANK; i++) coord[i] = -1;
  for (int i = 0; i < nn; i++) {
    const nx_prog_node *d = &g->nodes[i];
    step *t = &s[i];
    *t = (step){.tag = d->tag, .kind = d->kind, .dt = d->dtype,
                .x = {d->a, d->b, d->c}, .slot = -1, .alias = -1, .last = i,
                .out = -1};
    switch (d->tag) {
      case NX_NODE_IN: t->k = nouts + d->a; break;
      case NX_NODE_COORD:
        if (coord[d->a] < 0) {
          if (*n == NX_MAX_OPERANDS) return NX_DECLINED;
          axis[*n - base] = d->a;
          coord[d->a] = (*n)++;
        }
        t->k = coord[d->a];
        break;
      case NX_NODE_CONST: const_of(t, d->bits); break;
      case NX_NODE_OP1: {
        int xt = s[d->a].dt;
        t->arity = 1;
        if (d->kind == NX_OP1_COPY || (d->kind == NX_OP1_CAST && xt == t->dt))
          t->alias = d->a;
        else if (d->kind == NX_OP1_BITCAST) {
          if (nx_dtype_row_of(xt).bits < 8) return NX_DECLINED;
          t->alias = d->a;
        } else if (d->kind != NX_OP1_CAST &&
                   (t->f = row_of(t, xt, xt)) == NULL)
          return NX_DECLINED;
        break;
      }
      case NX_NODE_OP2:
        t->arity = 2;
        if ((t->f = row_of(t, s[d->a].dt, s[d->a].dt)) == NULL)
          return NX_DECLINED;
        break;
      default:
        t->arity = 3;
        if ((t->f = row_of(t, s[d->a].dt, s[d->b].dt)) == NULL)
          return NX_DECLINED;
    }
    if (t->alias >= 0 && s[t->alias].alias >= 0) t->alias = s[t->alias].alias;
  }
  /* Liveness: a value lives to its last reader, an output's to the end. */
  for (int i = 0; i < nn; i++) {
    int arity = s[i].tag == NX_NODE_OP1 && s[i].alias >= 0 ? 0 : s[i].arity;
    for (int o = 0; o < arity; o++) {
      int x = s[s[i].x[o]].alias >= 0 ? s[s[i].x[o]].alias : s[i].x[o];
      if (s[x].last < i) s[x].last = i;
    }
    if (s[i].alias >= 0 && s[s[i].alias].last < i) s[s[i].alias].last = i;
  }
  /* An output computed by a kind into its own dtype is written in place in
     its destination, once no load is read after it: a load identical to
     the destination is read before it is written. */
  int ins_end = 0;
  for (int i = 0; i < nn; i++)
    if (s[i].tag == NX_NODE_IN) ins_end = i + 1;
  for (int o = nouts - 1; o >= 0; o--) {
    int x = s[outs[o]].alias >= 0 ? s[outs[o]].alias : outs[o];
    s[x].last = nn;
    if (s[x].arity > 0 && s[x].alias < 0 && x == outs[o] && x >= ins_end &&
        held(s[x].dt) == s[x].dt)
      s[x].out = o;
  }
  int owner[SLOTS];
  for (int k = 0; k < SLOTS; k++) owner[k] = -1;
  for (int i = 0; i < nn; i++) {
    if (s[i].alias >= 0 || s[i].tag == NX_NODE_CONST) continue;
    int free = -1;
    for (int k = 0; k < SLOTS && free < 0; k++)
      if (owner[k] < 0 || s[owner[k]].last < i) free = k;
    if (free < 0) return NX_DECLINED;
    owner[free] = i;
    s[i].slot = free;
  }
  return NX_OK;
}

/* The entry */

/* The coordinate along axis [i], counted from the last, of a loop over
   [a]'s shape: a descriptor coalescing reads, of no buffer. An axis past
   the shape's has the index 0 everywhere. */
static nx_array coordinate(const nx_array *a, int i) {
  nx_array c = {.dtype = NX_INT64, .bits = 64, .rank = a->rank};
  for (int d = 0; d < a->rank; d++) {
    c.dim[d] = a->dim[d];
    c.dim[a->rank + d] = d == a->rank - 1 - i;
  }
  return c;
}

/* The bytes of the widest element [p]'s blocks hold: its operands', its
   nodes' held and carrier forms, a coordinate's. */
static int widest(int n, const nx_array *a, const step *s, int nn) {
  int w = 1;
  for (int k = 0; k < n; k++)
    if (width(nx_cpu_carrier(a[k].dtype)) > w)
      w = width(nx_cpu_carrier(a[k].dtype));
  for (int i = 0; i < nn; i++)
    if (width(nx_cpu_carrier(s[i].dt)) > w) w = width(nx_cpu_carrier(s[i].dt));
  return w;
}

static value run_map(const nx_spec_loop *m, value vdsts, value vops, step *s) {
  const nx_prog *g = nx_spec_loop_prog(m);
  int nouts = g->nouts, nins = g->nins, n = nouts + nins, e;
  int axis[NX_MAX_OPERANDS];
  if ((int)Wosize_val(vdsts) != nouts || (int)Wosize_val(vops) != nins)
    return Val_int(NX_ARITY);
  if (n > NX_MAX_OPERANDS) return Val_int(NX_DECLINED);
  for (int k = 0; k < nins; k++)
    if (nx_spec_loop_pad(m, k) != NULL) return Val_int(NX_DECLINED);
  int loop = n;
  if (decode(g, s, &loop, axis) != NX_OK) return Val_int(NX_DECLINED);
  const int32_t *ins = nx_prog_ins(g), *outs = nx_prog_outs(g);
  nx_operand in[NX_MAX_OPERANDS];
  for (int o = 0; o < nouts; o++)
    in[o] = (nx_operand){Field(Field(vdsts, o), 0), s[outs[o]].dt, 1};
  for (int k = 0; k < nins; k++)
    in[nouts + k] = (nx_operand){Field(Field(vops, k), 0), ins[k], 0};
  nx_array a[NX_MAX_OPERANDS];
  if ((e = nx_read(n, in, a))) return Val_int(e);
  for (int k = n; k < loop; k++) a[k] = coordinate(&a[0], axis[k - n]);
  prog j = {a, loop, nouts, g->nnodes, s, outs, {0}};
  /* A load that is an output too is read through its slot: a later output
     of the plane would read it after an earlier one's store. */
  for (int k = nouts; k < n; k++) {
    j.direct[k] = 1;
    for (int o = 0; o < nouts; o++)
      if (a[o].base != NULL && a[o].base == a[k].base) j.direct[k] = 0;
  }
  nx_loop l;
  if (!(e = nx_coalesce(loop, a, &l)))
    nx_cpu_walk(loop, a, &l, SLOT_BYTES / widest(loop, a, s, g->nnodes), block,
                &j);
  nx_done(n, a);
  return Val_int(e);
}

value nx_cpu_map(value vs, value vdsts, value vops) {
  CAMLparam3(vs, vdsts, vops);
  /* The descriptor, copied: the door may move the string. */
  size_t len = caml_string_length(vs);
  uint8_t *copy = malloc(len);
  if (copy == NULL) caml_raise_out_of_memory();
  memcpy(copy, String_val(vs), len);
  const nx_spec_loop *m = (const nx_spec_loop *)copy;
  int nn = nx_spec_loop_prog(m)->nnodes;
  step *s = malloc(sizeof(step) * (size_t)(nn > 0 ? nn : 1));
  if (s == NULL) {
    free(copy);
    caml_raise_out_of_memory();
  }
  value r = run_map(m, vdsts, vops, s);
  free(s);
  free(copy);
  CAMLreturn(r);
}
