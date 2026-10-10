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
   plane, each node runs in order over values held in slots of an arena on
   the stack, through the target table's rows, as apply.c's kinds do:

   - a load held in its own dtype, its rows contiguous, is read in place,
     unless it is identical to an output, which a node may write first;
     any other is staged into its slot;
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

   Slots go by liveness: a value is held from the node that computes it to
   the last node or output that reads it, and a slot is free once its value
   is no longer held, so that a node never writes a slot it reads. The
   arena is cut into one slot per value held at once, and TEMPS more where
   a node casts or holds a dtype other than its carrier, so that the more
   values a program holds, the fewer elements a block has. A
   program that holds more than HELD values at once, has a padded load,
   needs more operands than a loop holds, or bitcasts a sub-byte dtype is
   declined: its caller runs the nodes one by one. */

#include <stdlib.h>
#include <string.h>

#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "cpu.h"
#include "nx_spec.h"

/* A block's values live in an arena on the stack: a slot for each value
   the program holds at once, at most HELD, and where needed TEMPS
   temporaries, which a node decodes its operands into and computes in
   before it encodes. A
   program that holds HELD values gets LEAST elements of the widest
   carrier, complex128's, a slot; one that holds fewer gets longer blocks,
   up to SLOT bytes a slot. */
#define HELD 256
#define TEMPS 4
#define LEAST 16
#define ARENA ((HELD + TEMPS) * LEAST * 16)

/* The most bytes a slot holds. Slots of half apply.c's NX_CPU_SLOT ran
   faster on both machines: on the M1 Max map-add-f32-1M took 23.6 us in 8
   KiB slots, 33.5 in 16 and 27.6 in 4; on kimchi map-6-f32-1M 52.2 us in 8
   KiB, 57.3 in 16. */
#define SLOT (8 * 1024)

/* The bytes of the cache line a slot starts on. */
#define LINE 64

/* The nodes of a program whose steps live on the stack; a longer one's are
   allocated. */
#define STEPS 32

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
  int alias;  /* the node whose value it is, itself no alias, or -1 */
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
  int32_t outs[NX_MAX_OPERANDS];
  int nheld; /* the values held at once: their slots come first */
  int temps; /* TEMPS where a node casts or is not of its carrier, else 0 */
  int w;     /* the bytes of the widest carrier a block holds */
} prog;

/* A plane of a block, and the arena its values are held in: slot k at
   arena + k·bytes. */
typedef struct {
  nx_cpu_block p;
  uint8_t *arena;
  int64_t bytes;
} plane;

static uint8_t *slot(const plane *f, int k) { return f->arena + k * f->bytes; }

/* Temporary [k] of [f]. */
static uint8_t *temp(const prog *j, const plane *f, int k) {
  return slot(f, j->nheld + k);
}

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

/* Whether load [k] of plane [p] is read in place. A load identical to an
   output is not: a node may write the output before another reads the
   load. */
static int in_place(const prog *j, const nx_cpu_block *p, int k) {
  const nx_array *a = &j->a[k];
  return !a->alias && a->dtype == held(a->dtype) && p->s0[k] == 1;
}

/* Where node [i], a node with a slot, computes its value in plane [f]: its
   output's destination where it is written in place there, else its
   slot. */
static view home(const prog *j, const plane *f, int i) {
  const step *s = &j->s[i];
  const nx_cpu_block *p = &f->p;
  int64_t w = width(held(s->dt));
  int o = s->out;
  if (o >= 0 && p->s0[o] == 1)
    return (view){j->a[o].base + p->at[o] * w, 1, p->s1[o] * w};
  return (view){slot(f, s->slot), 1, p->n0 * w};
}

/* Node [i]'s value in plane [f]. */
static view value_of(const prog *j, const plane *f, int i) {
  const step *s = &j->s[i];
  if (s->alias >= 0) s = &j->s[i = s->alias];
  if (s->tag == NX_NODE_CONST) return (view){(uint8_t *)s->bits, 0, 0};
  if (s->tag == NX_NODE_IN && in_place(j, &f->p, s->k)) {
    const nx_array *a = &j->a[s->k];
    int64_t w = width(a->dtype);
    return (view){a->base + f->p.at[s->k] * w, 1, f->p.s1[s->k] * w};
  }
  return home(j, f, i);
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

/* Operand [o] of node [i] in its carrier: decoded into temporary [o] where
   it is held as codes. */
static view wide_of(const prog *j, const plane *f, int i, int o) {
  const nx_cpu_block *p = &f->p;
  int x = j->s[i].x[o];
  int dt = j->s[x].dt, c = nx_cpu_carrier(dt);
  view v = value_of(j, f, x);
  if (held(dt) == c) return v;
  uint8_t *t = temp(j, f, o);
  nx_cpu_run decode = nx_cpu_table->convert[dt][c];
  if (v.s0 == 0) {
    decode(v.p, t, 1);
    return (view){t, 0, 0};
  }
  for (int64_t r = 0; r < p->n1; r++)
    decode(v.p + r * v.s1, t + r * p->n0 * width(c), p->n0);
  return (view){t, 1, p->n0 * width(c)};
}

/* Runs node [i] over plane [f] into its slot. */
static void run(const prog *j, const plane *f, int i) {
  const step *s = &j->s[i];
  const nx_cpu_block *p = &f->p;
  int64_t n0 = p->n0, n1 = p->n1;
  int dt = s->dt, c = nx_cpu_carrier(dt), h = held(dt);
  uint8_t *d = slot(f, s->slot);
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
  view t = home(j, f, i);
  if (s->tag == NX_NODE_OP1 && s->kind == NX_OP1_CAST) {
    /* The operand in its carrier, converted as a cast does into the dtype's
       elements: in place where the dtype is held as itself, else into the
       third temporary, then to its carrier. */
    int xc = nx_cpu_carrier(j->s[s->x[0]].dt);
    uint8_t *t2 = temp(j, f, 2);
    view v = runs(wide_of(j, f, i, 0), xc, p, temp(j, f, 1));
    nx_cpu_run convert = nx_cpu_table->convert[xc][dt];
    if (h == dt) {
      for (int64_t r = 0; r < n1; r++)
        convert(v.p + r * v.s1, t.p + r * t.s1, n0);
      return;
    }
    for (int64_t r = 0; r < n1; r++)
      convert(v.p + r * v.s1, t2 + r * n0 * width(dt), n0);
    if (dt == NX_FLOAT4_E2M1FN)
      nx_cpu_table->convert[dt][NX_FLOAT32](t2, d, n0 * n1);
    else
      round_held(dt, t2, d, NULL, n0 * n1);
    return;
  }
  /* Where selects held values; a kind computes in its operands' carrier,
     in place where the result is held in its carrier. */
  int where = s->tag == NX_NODE_OP3 && s->kind == NX_OP3_WHERE;
  view v[3];
  for (int o = 0; o < s->arity; o++)
    v[o] = where ? value_of(j, f, s->x[o]) : wide_of(j, f, i, o);
  int64_t w = where ? width(h) : width(c);
  uint8_t *t3 = where || dt == c ? NULL : temp(j, f, 3);
  view o = where || h == c ? t : (view){t3, 1, n0 * w};
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
      nx_cpu_table->convert[c][dt](t3 + r * n0 * w, t.p + r * t.s1, n0);
  else if (dt != c)
    round_held(dt, d, d, t3, n0 * n1);
}

/* Stores output [o]'s value of plane [f] into its destination. */
static void store(const prog *j, const plane *f, int o) {
  const nx_cpu_block *p = &f->p;
  const nx_array *a = &j->a[o];
  int dt = a->dtype, h = held(dt), i = j->outs[o];
  if (j->s[i].out == o && p->s0[o] == 1) return;
  view v = value_of(j, f, i);
  if (h == dt) {
    int64_t w = width(dt);
    nx_copy_box(a->base, v.p,
                &(nx_box){{1, p->n1, p->n0},
                          {p->at[o], 0},
                          {{0, p->s1[o], p->s0[o]}, {0, v.s1 / w, v.s0}}},
                a->bits);
    return;
  }
  v = runs(v, h, p, temp(j, f, 0));
  nx_cpu_unstage(a, p, o, v.p, v.s1, h);
}

static void block(const nx_cpu_block *b, void *ctx) {
  const prog *j = ctx;
  /* The arena, as long as the block needs, on a cache line. */
  int64_t bytes = (b->n0 * b->n1 * j->w + LINE - 1) / LINE * LINE;
  uint8_t room[(j->nheld + j->temps) * bytes + LINE];
  plane f = {*b, room + (LINE - (uintptr_t)room % LINE) % LINE, bytes};
  f.p.n2 = 1;
  /* A row of one element steps nowhere: every operand's is contiguous. */
  if (b->n0 == 1)
    for (int k = 0; k < j->n; k++) f.p.s0[k] = 1;
  for (int64_t q = 0; q < b->n2; q++) {
    for (int k = 0; k < j->n; k++) f.p.at[k] = b->at[k] + q * b->s2[k];
    for (int i = 0; i < j->nnodes; i++)
      if (j->s[i].slot >= 0) run(j, &f, i);
    for (int o = 0; o < j->nouts; o++) store(j, &f, o);
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
   coordinates too, [axis] gives each one's axis and [slots] counts the
   values held at once. */
static int decode(const nx_prog *g, step *s, int *n, int *axis, int *slots) {
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
  /* A value takes the first slot whose value is no longer held: owner[k]
     is slot k's latest value. */
  int owner[HELD];
  *slots = 0;
  for (int i = 0; i < nn; i++) {
    if (s[i].alias >= 0 || s[i].tag == NX_NODE_CONST) continue;
    int k = 0;
    while (k < *slots && s[owner[k]].last >= i) k++;
    if (k == HELD) return NX_DECLINED;
    if (k == *slots) (*slots)++;
    owner[k] = i;
    s[i].slot = k;
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

/* The temporaries the nodes [s] need: TEMPS where one casts, or is of a
   dtype other than its carrier, which it decodes or brings back to its
   dtype through them, else none. */
static int temps(const step *s, int nn) {
  for (int i = 0; i < nn; i++)
    if (s[i].dt != nx_cpu_carrier(s[i].dt) ||
        (s[i].tag == NX_NODE_OP1 && s[i].kind == NX_OP1_CAST &&
         s[i].alias < 0))
      return TEMPS;
  return 0;
}

/* The bytes of the widest carrier the nodes [s] hold: a block's loads,
   outputs and coordinates are nodes. */
static int widest(const step *s, int nn) {
  int w = 1;
  for (int i = 0; i < nn; i++)
    if (width(nx_cpu_carrier(s[i].dt)) > w) w = width(nx_cpu_carrier(s[i].dt));
  return w;
}

/* Runs the program [g] over the [nouts + nins] operands [a] the door read,
   decoding it into [s]. */
static int run_map(const nx_prog *g, nx_array *a, step *s) {
  int n = g->nouts + g->nins, loop = n, slots, e;
  int axis[NX_MAX_OPERANDS];
  if (decode(g, s, &loop, axis, &slots) != NX_OK) return NX_DECLINED;
  for (int k = n; k < loop; k++) a[k] = coordinate(&a[0], axis[k - n]);
  prog j = {a, loop, g->nouts, g->nnodes, s, {0}, slots, temps(s, g->nnodes),
            widest(s, g->nnodes)};
  memcpy(j.outs, nx_prog_outs(g), sizeof(int32_t) * (size_t)g->nouts);
  int64_t bytes = ARENA / (slots + j.temps) / LINE * LINE;
  if (bytes > SLOT) bytes = SLOT;
  nx_loop l;
  if (!(e = nx_coalesce(loop, a, &l)))
    nx_cpu_walk(loop, a, &l, bytes / j.w, block, &j);
  return e;
}

value nx_cpu_map(value vs, value vdsts, value vops) {
  CAMLparam3(vs, vdsts, vops);
  const nx_spec_loop *m = (const nx_spec_loop *)String_val(vs);
  const nx_prog *g = nx_spec_loop_prog(m);
  int nouts = g->nouts, nins = g->nins, n = nouts + nins, e;
  if ((int)Wosize_val(vdsts) != nouts || (int)Wosize_val(vops) != nins)
    CAMLreturn(Val_int(NX_ARITY));
  if (n > NX_MAX_OPERANDS) CAMLreturn(Val_int(NX_DECLINED));
  for (int k = 0; k < nins; k++)
    if (nx_spec_loop_pad(m, k) != NULL) CAMLreturn(Val_int(NX_DECLINED));
  const int32_t *ins = nx_prog_ins(g), *outs = nx_prog_outs(g);
  nx_operand in[NX_MAX_OPERANDS];
  for (int o = 0; o < nouts; o++)
    in[o] = (nx_operand){Field(Field(vdsts, o), 0), g->nodes[outs[o]].dtype, 1};
  for (int k = 0; k < nins; k++)
    in[nouts + k] = (nx_operand){Field(Field(vops, k), 0), ins[k], 0};
  nx_array a[NX_MAX_OPERANDS];
  if ((e = nx_read(n, in, a))) CAMLreturn(Val_int(e));
  /* The door may have run OCaml code, which may move the descriptor. */
  g = nx_spec_loop_prog((const nx_spec_loop *)String_val(vs));
  step local[STEPS], *s = local;
  if (g->nnodes > STEPS &&
      (s = malloc(sizeof(step) * (size_t)g->nnodes)) == NULL) {
    nx_done(n, a);
    caml_raise_out_of_memory();
  }
  e = run_map(g, a, s);
  if (s != local) free(s);
  nx_done(n, a);
  CAMLreturn(Val_int(e));
}
