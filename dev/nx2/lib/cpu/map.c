/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Maps: a scalar program run over blocks.

   A map's loop operands are its outputs, written, then its loads, then,
   where its Coord nodes read it, the index: an int64 operand of no
   buffer, C-contiguous over the loop's shape, whose position at an index
   is that index's place in C order, from which each coordinate is
   divided out. The loop is walked in blocks. Per block, plane by
   plane, each node runs in order over values held in slots of an arena on
   the stack, through the target table's rows, as apply.c's kinds do:

   - a load held in its own dtype whose rows are contiguous is read in
     place;
   - a load held in its own dtype of which the plane reads one element, as
     a broadcast's, is read in place with a step of 0, as a constant is;
   - a load identical to an output is staged, since a node may write the
     output first, and so is any other load: into its slot;
   - a constant is read in place with a step of 0;
   - a coordinate fills its slot from the index's positions;
   - Copy, a Cast into the operand's own dtype and a Bitcast between byte
     wide dtypes read their operand's value, a Bitcast between sub-byte
     ones its codes;
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
   values a program holds, the fewer elements a block has, down to LEAST.
   An arena of more than ARENA bytes, as a program of many values of a
   wide dtype needs, comes from the heap in place of the stack, once per
   block the walk runs at a time: a block puts it back for the next.

   A padded load is first copied padded into C-heap memory (assemble.c),
   which the call holds until it returns, and read from there as a plain
   one. */

#include <stdlib.h>
#include <string.h>

#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "cpu.h"

/* A block's values live in an arena: a slot for each value the program
   holds at once, and where needed TEMPS temporaries, which a node decodes
   its operands into and computes in before it encodes. A program that
   holds HELD values gets LEAST elements of the widest carrier,
   complex128's, a slot in ARENA bytes; one that holds fewer gets longer
   blocks, up to SLOT bytes a slot; one that holds more gets LEAST
   elements of its widest carrier a slot. The arena is on the stack up to
   ARENA bytes, past them from the heap: at a carrier of one byte, up to
   1040 values stay on the stack. */
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
  int k;      /* In: its loop operand */
  int out;    /* the output it is written into in place, or -1 */
  void *f;    /* a kind's row */
  uint8_t bits[16]; /* Const: its element, held */
} step;

typedef struct {
  const nx_array *a;
  int n, nouts, nnodes;
  int index; /* the index's loop operand, or -1 */
  struct heap *heap; /* the arenas past ARENA bytes */
  /* The loop's extents, and the product of those past each axis. */
  int64_t extent[NX_MAX_RANK], inner[NX_MAX_RANK];
  const step *s;
  int32_t outs[NX_MAX_OPERANDS];
  int nheld; /* the values held at once: their slots come first */
  int temps; /* TEMPS where a node casts or is not of its carrier, else 0 */
  int w;     /* the bytes of the widest carrier a block holds */
} prog;

/* A plane of a block, and the arena its values are held in: slot k at
   arena + k·bytes. Where [to] is not NULL, each output [o] with a buffer
   to[o] goes there, C-contiguous, and no other is stored. */
typedef struct {
  nx_cpu_block p;
  uint8_t *arena;
  int64_t bytes;
  uint8_t *const *to;
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

/* [n] values of the 4-bit dtype [dt], held, at [d] from the held values
   [x] of the 4-bit dtype [xt] whose codes they share: through [codes], a
   byte each. */
static void sub_bitcast(int xt, int dt, const uint8_t *x, uint8_t *codes,
                        uint8_t *d, int64_t n) {
  if (xt == NX_FLOAT4_E2M1FN)
    nx_cpu_table->convert[NX_FLOAT32][xt](x, codes, n);
  else
    for (int64_t i = 0; i < n; i++) codes[i] = x[i] & 15;
  if (dt == NX_FLOAT4_E2M1FN)
    nx_cpu_table->convert[dt][NX_FLOAT32](codes, d, n);
  else round_held(dt, codes, d, NULL, n);
}

/* Whether the plane [p] reads one element of operand [k]. */
static int repeated(const nx_cpu_block *p, int k) {
  return p->s0[k] == 0 && (p->s1[k] == 0 || p->n1 == 1);
}

/* Whether load [k] of plane [p] is read in place. A load identical to an
   output is not: a node may write the output before another reads the
   load. */
static int in_place(const prog *j, const nx_cpu_block *p, int k) {
  const nx_array *a = &j->a[k];
  return !a->alias && a->dtype == held(a->dtype) &&
         (p->s0[k] == 1 || repeated(p, k));
}

/* Where node [i], a node with a slot, computes its value in plane [f]: its
   output's destination or buffer where it is written in place there, else
   its slot. */
static view home(const prog *j, const plane *f, int i) {
  const step *s = &j->s[i];
  const nx_cpu_block *p = &f->p;
  int64_t w = width(held(s->dt));
  int o = s->out;
  if (o >= 0 && f->to != NULL && f->to[o] != NULL)
    return (view){f->to[o], 1, p->n0 * w};
  if (o >= 0 && f->to == NULL && p->s0[o] == 1)
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
    uint8_t *e = a->base + f->p.at[s->k] * w;
    if (repeated(&f->p, s->k)) return (view){e, 0, 0};
    return (view){e, 1, f->p.s1[s->k] * w};
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

/* Fills [v] with each element's index along axis [a] of plane [p]: from
   the index's position q at the element, (q / inner) % extent. Along a
   row q steps by s0, a product of the extents past some axis as inner is,
   so one divides the other: the coordinate steps by s0 / inner each
   element, wrapping at the extent, or holds for runs of inner / s0. Each
   run, and each stretch between wraps, is a loop of its own, which
   vectorises. */
static void coords(const prog *j, const nx_cpu_block *p, int a, int64_t *v) {
  int k = j->index;
  int64_t inner = j->inner[a], extent = j->extent[a], s0 = p->s0[k];
  int64_t n0 = p->n0;
  for (int64_t r = 0; r < p->n1; r++) {
    int64_t q = p->at[k] + r * p->s1[k], c = q / inner % extent;
    int64_t *row = v + r * n0;
    if (s0 >= inner) {
      int64_t m = s0 / inner % extent;
      for (int64_t e = 0; e < n0;) {
        int64_t span = m == 0 ? n0 - e : (extent - c + m - 1) / m;
        if (span > n0 - e) span = n0 - e;
        for (int64_t i = 0; i < span; i++) row[e + i] = c + i * m;
        e += span;
        c += span * m - extent;
      }
      continue;
    }
    int64_t run = inner / s0;
    for (int64_t e = 0, left = run - q % inner / s0; e < n0; left = run) {
      int64_t span = left < n0 - e ? left : n0 - e;
      for (int64_t i = 0; i < span; i++) row[e + i] = c;
      e += span;
      if (++c == extent) c = 0;
    }
  }
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
    coords(j, p, s->x[0], (int64_t *)d);
    return;
  }
  if (s->tag == NX_NODE_OP1 && s->kind == NX_OP1_BITCAST) {
    int xt = j->s[s->x[0]].dt;
    view v = runs(value_of(j, f, s->x[0]), held(xt), p, temp(j, f, 1));
    for (int64_t r = 0; r < n1; r++)
      sub_bitcast(xt, dt, v.p + r * v.s1, temp(j, f, 0), d + r * n0 * width(h),
                  n0);
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
  if (f->to != NULL) {
    /* The value's rows, as its view steps them, a constant's or a
       repeated load's included, into the buffer's. */
    if (f->to[o] == NULL || j->s[i].out == o) return;
    view v = value_of(j, f, i);
    int64_t w = width(h);
    nx_copy_box(f->to[o], v.p,
                &(nx_box){{1, p->n1, p->n0},
                          {0, 0},
                          {{0, p->n0, 1}, {0, v.s1 / w, v.s0}}},
                8 * (int)w);
    return;
  }
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

/* Arenas on the heap, of [bytes] each, which a walk's blocks share: a
   block takes a spare one or allocates one, and puts it back, so a walk
   allocates as many as it runs blocks at once; past SPARES spares, an
   arena given back is freed. [failed] is set where one cannot be had. */
#define SPARES 64

struct heap {
  int64_t bytes;
  void *spare[SPARES];
  int failed;
};

static void *arena_take(struct heap *h) {
  for (int i = 0; i < SPARES; i++) {
    void *p = __atomic_exchange_n(&h->spare[i], NULL, __ATOMIC_ACQ_REL);
    if (p != NULL) return p;
  }
  void *p = malloc((size_t)h->bytes);
  if (p == NULL) __atomic_store_n(&h->failed, 1, __ATOMIC_RELAXED);
  return p;
}

static void arena_give(struct heap *h, void *p) {
  for (int i = 0; i < SPARES; i++) {
    void *none = NULL;
    if (__atomic_compare_exchange_n(&h->spare[i], &none, p, 0,
                                    __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
      return;
  }
  free(p);
}

/* Runs [j] over the block [b], its outputs stored into their destinations,
   or into the buffers [to]. */
static void planes(const nx_cpu_block *b, const prog *j, uint8_t *const *to) {
  /* The arena, as long as the block needs, on a cache line: on the stack
     up to ARENA bytes, else from the heap. */
  int64_t bytes = (b->n0 * b->n1 * j->w + LINE - 1) / LINE * LINE;
  int64_t size = (j->nheld + j->temps) * bytes + LINE;
  uint8_t stack[size <= ARENA + LINE ? size : 1];
  uint8_t *room = size <= ARENA + LINE ? stack : arena_take(j->heap);
  if (room == NULL) return;
  plane f = {*b, room + (LINE - (uintptr_t)room % LINE) % LINE, bytes, to};
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
  if (room != stack) arena_give(j->heap, room);
}

static void block(const nx_cpu_block *b, void *ctx) { planes(b, ctx, NULL); }

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

/* A constant's element held, from the bits of its dtype's element in
   [s]. */
static void const_of(step *s) {
  int dt = s->dt;
  uint8_t bits[16];
  memcpy(bits, s->bits, 16);
  if (dt == NX_FLOAT4_E2M1FN)
    nx_cpu_table->convert[dt][NX_FLOAT32](bits, s->bits, 1);
  else if (held(dt) != dt)
    round_held(dt, bits, s->bits, NULL, 1);
}

/* Fills the steps [s], copied from a program of [nn] nodes and outputs
   [outs] over a loop of rank [rank], and answers NX_OK, NX_SHAPE for a
   coordinate past the rank, NX_DECLINED, or -1 where memory runs out; on
   NX_OK [index] says whether a coordinate reads the index and [slots]
   counts the values held at once. */
static int decode(step *s, int nn, int nouts, const int32_t *outs, int rank,
                  int *index, int *slots) {
  *index = 0;
  for (int i = 0; i < nn; i++) {
    step *t = &s[i];
    t->arity = 0;
    t->slot = t->alias = t->out = -1;
    t->last = i;
    t->f = NULL;
    int a = t->x[0];
    switch (t->tag) {
      case NX_NODE_IN: t->k = nouts + a; break;
      case NX_NODE_COORD:
        if (a >= rank) return NX_SHAPE;
        *index = 1;
        break;
      case NX_NODE_CONST: const_of(t); break;
      case NX_NODE_OP1: {
        int xt = s[a].dt;
        t->arity = 1;
        if (t->kind == NX_OP1_COPY || (t->kind == NX_OP1_CAST && xt == t->dt))
          t->alias = a;
        else if (t->kind == NX_OP1_BITCAST) {
          /* A sub-byte bitcast into another dtype computes its codes. */
          if (nx_dtype_row_of(xt).bits >= 8 || xt == t->dt) t->alias = a;
        } else if (t->kind != NX_OP1_CAST &&
                   (t->f = row_of(t, xt, xt)) == NULL)
          return NX_DECLINED;
        break;
      }
      case NX_NODE_OP2:
        t->arity = 2;
        if ((t->f = row_of(t, s[a].dt, s[a].dt)) == NULL) return NX_DECLINED;
        break;
      default:
        t->arity = 3;
        if ((t->f = row_of(t, s[a].dt, s[t->x[1]].dt)) == NULL)
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
  int local[HELD], *owner = local;
  if (nn > HELD && (owner = malloc(sizeof(int) * (size_t)nn)) == NULL)
    return -1;
  *slots = 0;
  for (int i = 0; i < nn; i++) {
    if (s[i].alias >= 0 || s[i].tag == NX_NODE_CONST) continue;
    int k = 0;
    while (k < *slots && s[owner[k]].last >= i) k++;
    if (k == *slots) (*slots)++;
    owner[k] = i;
    s[i].slot = k;
  }
  if (owner != local) free(owner);
  return NX_OK;
}

/* The entry */

/* The index of a loop over [a]'s shape, C-contiguous: a descriptor
   coalescing reads, of no buffer. Fills [extent] and [inner], the
   product of the extents past each axis. */
static nx_array index_of(const nx_array *a, int64_t *extent, int64_t *inner) {
  nx_array c = {.dtype = NX_INT64, .bits = 64, .rank = a->rank};
  int64_t past = 1;
  for (int d = a->rank - 1; d >= 0; d--) {
    extent[d] = c.dim[d] = a->dim[d];
    inner[d] = c.dim[a->rank + d] = past;
    past *= a->dim[d];
  }
  return c;
}

/* What an element of the nodes [s] costs (cpu.h, Calls, scalar): the
   kind each node runs that is no alias, at its operand's dtype, a
   Where's the chosen ones' (nx_cpu_kind_cost). */
static double element_cost(const step *s, int nn) {
  double c = 0;
  for (int i = 0; i < nn; i++)
    if (s[i].arity > 0 && s[i].alias < 0)
      c += nx_cpu_kind_cost(s[i].arity, s[i].kind,
                            s[s[i].x[s[i].arity == 3]].dt);
  return c;
}

/* What an element of the map of the nodes [s] over the [n] operands [a]
   costs: its kinds, and its operands' loads and stores once each: a
   kind's cost counts its own, but a program of no kind, a load copied to
   its output, has none. */
static double element_price(const step *s, int nn, const nx_array *a,
                            int n) {
  double per = element_cost(s, nn);
  for (int k = 0; k < n; k++) per += 2 * width(a[k].dtype);
  return per;
}

/* Whether run_map prices its walk and releases the domain lock for it,
   or runs it as its caller left the lock, its caller's price covering
   the walk. */
enum { OWN_PRICE, CALLERS_PRICE };

/* The nodes of [s] that run a kind's row: each takes about as long as a
   memcpy of the loop's bytes. */
static int64_t kinds(const step *s, int nn) {
  int64_t k = 0;
  for (int i = 0; i < nn; i++) k += s[i].arity > 0 && s[i].alias < 0;
  return k > 0 ? k : 1;
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

/* Runs the program copied into the steps [s], of [nn] nodes and outputs
   [outs], over the [n] operands [a] the door read, its outputs first. */
/* Decodes the program copied into the steps [s], of [nn] nodes and outputs
   [outs], into [j] over the [n] operands [a], its outputs first and of the
   loop's shape, the index after them where a coordinate reads it, its
   arenas past ARENA bytes from [heap]; sets [most], the elements a block
   holds. */
static int setup(prog *j, step *s, int nn, int nouts, const int32_t *outs,
                 nx_array *a, int n, struct heap *heap, int64_t *most) {
  int index, slots, e;
  if ((e = decode(s, nn, nouts, outs, a[0].rank, &index, &slots))) return e;
  *j = (prog){.a = a, .n = n, .nouts = nouts, .nnodes = nn, .index = -1,
              .heap = heap, .s = s, .nheld = slots,
              .temps = temps(s, nn), .w = widest(s, nn)};
  if (index) {
    a[n] = index_of(&a[0], j->extent, j->inner);
    j->index = j->n++;
  }
  memcpy(j->outs, outs, sizeof(int32_t) * (size_t)nouts);
  /* A program of constants alone holds nothing in the arena: its blocks
     take SLOT bytes, as one of few slots does. */
  int held = slots + j->temps;
  int64_t bytes = held > 0 ? ARENA / held / LINE * LINE : SLOT;
  if (bytes > SLOT) bytes = SLOT;
  if (bytes < LEAST * j->w) bytes = LEAST * j->w;
  /* A block's arena takes at most its slots of [bytes] on a cache line. */
  heap->bytes = held * ((bytes + LINE - 1) / LINE * LINE) + LINE;
  *most = bytes / j->w;
  return NX_OK;
}

/* Runs the program copied into the steps [s], of [nn] nodes and outputs
   [outs], over the [n] operands [a] the door read, its outputs first,
   priced as [pricing] says. */
static int run_map(step *s, int nn, int nouts, const int32_t *outs,
                   nx_array *a, int n, int pricing) {
  struct heap heap = {0};
  prog j;
  int64_t most;
  /* A program whose nodes' analysis costs a release's worth is analysed
     with the lock released, whatever its walk costs. */
  int early = pricing == OWN_PRICE && nn > NX_CPU_ITEMS, released = 0;
  if (early) released = nx_cpu_release(NX_CPU_RELEASE_BYTES);
  int e = setup(&j, s, nn, nouts, outs, a, n, &heap, &most);
  nx_loop l;
  if (!e && !(e = nx_coalesce(j.n, a, &l))) {
    if (pricing == OWN_PRICE && !early)
      released = nx_cpu_release(
          nx_cpu_loop_cost(j.n, &l, element_price(s, nn, a, j.n)));
    nx_cpu_walk(j.n, a, &l, most, kinds(s, nn), block, &j);
  }
  if (pricing == OWN_PRICE) nx_cpu_reacquire(released);
  for (int i = 0; i < SPARES; i++) free(heap.spare[i]);
  return heap.failed ? -1 : e;
}

/* Copies the program at the root [vp], a Prog.t, into the steps [s], one
   per node: its tag, kind, dtype, operands and a constant's bits,
   zero-padded; and its operands' dtypes into [ins] and its outputs into
   [outs]. Prog.v checked every index it follows. It lets other domains
   collect every NX_CPU_ITEMS nodes, reading the program again from its
   root after each. */
static void copy_prog(const value *vp, step *s, int *ins, int32_t *outs) {
  value vins = Field(*vp, NX_PROG_INS), vouts = Field(*vp, NX_PROG_OUTS);
  mlsize_t next = NX_CPU_ITEMS;
  for (mlsize_t k = 0; k < Wosize_val(vins); k++)
    ins[k] = Int_val(Field(vins, k));
  for (mlsize_t o = 0; o < Wosize_val(vouts); o++)
    outs[o] = Int_val(Field(vouts, o));
  mlsize_t nn = Wosize_val(Field(*vp, NX_PROG_NODES));
  for (mlsize_t i = 0; i < nn; i++) {
    if (i == next) {
      nx_let_collect();
      next += NX_CPU_ITEMS;
    }
    value d = Field(Field(*vp, NX_PROG_NODES), i);
    step *t = &s[i];
    memset(t, 0, sizeof *t);
    t->tag = Tag_val(d);
    switch (t->tag) {
      case NX_NODE_IN:
        t->x[0] = Int_val(Field(d, NX_IN_OPERAND));
        t->dt = ins[t->x[0]];
        break;
      case NX_NODE_COORD:
        t->x[0] = Int_val(Field(d, NX_COORD_AXIS));
        t->dt = NX_INT64;
        break;
      case NX_NODE_CONST: {
        value b = Field(d, NX_CONST_BITS);
        t->dt = Int_val(Field(d, NX_CONST_DTYPE));
        memcpy(t->bits, String_val(b), caml_string_length(b));
        break;
      }
      case NX_NODE_OP1:
        t->kind = NX_OP1_CODE(Field(d, NX_OP1_KIND));
        t->dt = Int_val(Field(d, NX_OP1_DTYPE));
        t->x[0] = Int_val(Field(d, NX_OP1_OPERAND));
        break;
      case NX_NODE_OP2:
        t->kind = NX_OP2_CODE(Field(d, NX_OP2_KIND));
        for (int o = 0; o < 2; o++)
          t->x[o] = Int_val(Field(d, NX_OP2_OPERANDS + o));
        t->dt = t->kind >= NX_OP2_EQUAL ? NX_BOOL : s[t->x[0]].dt;
        break;
      default:
        t->kind = Int_val(Field(d, NX_OP3_KIND));
        for (int o = 0; o < 3; o++)
          t->x[o] = Int_val(Field(d, NX_OP3_OPERANDS + o));
        t->dt = s[t->x[1]].dt;
    }
  }
}

/* Programs (cpu.h) */

struct nx_cpu_prog {
  step local[STEPS], *s;
  int nn, nouts, nins;
  int ins[NX_MAX_OPERANDS];
  int32_t outs[NX_MAX_OPERANDS];
  /* The outputs, of no buffer, the loads and the index, as a map's
     operands. */
  nx_array a[NX_MAX_OPERANDS];
  prog j;
  struct heap heap;
  int64_t most;
};

nx_cpu_prog *nx_cpu_prog_copy(const value *vp) {
  int nn = (int)Wosize_val(Field(*vp, NX_PROG_NODES));
  nx_cpu_prog *g = calloc(1, sizeof *g);
  if (g == NULL) return NULL;
  g->s = g->local;
  if (nn > STEPS && (g->s = malloc(sizeof(step) * (size_t)nn)) == NULL) {
    free(g);
    return NULL;
  }
  g->nn = nn;
  g->nouts = (int)Wosize_val(Field(*vp, NX_PROG_OUTS));
  g->nins = (int)Wosize_val(Field(*vp, NX_PROG_INS));
  copy_prog(vp, g->s, g->ins, g->outs);
  return g;
}

void nx_cpu_prog_free(nx_cpu_prog *g) {
  if (g == NULL) return;
  for (int i = 0; i < SPARES; i++) free(g->heap.spare[i]);
  if (g->s != g->local) free(g->s);
  free(g);
}

int nx_cpu_prog_nodes(const nx_cpu_prog *g) { return g->nn; }
int nx_cpu_prog_ins(const nx_cpu_prog *g) { return g->nins; }
int nx_cpu_prog_in(const nx_cpu_prog *g, int k) { return g->ins[k]; }
int nx_cpu_prog_outs(const nx_cpu_prog *g) { return g->nouts; }
int nx_cpu_prog_out(const nx_cpu_prog *g, int o) {
  return g->s[g->outs[o]].dt;
}

int nx_cpu_prog_held(const nx_cpu_prog *g, int o) {
  return held(nx_cpu_prog_out(g, o));
}

int nx_cpu_prog_failed(const nx_cpu_prog *g) {
  return __atomic_load_n(&g->heap.failed, __ATOMIC_RELAXED);
}

int nx_cpu_prog_bind(nx_cpu_prog *g, const nx_array *a, int rank,
                     const int64_t *shape, int *index) {
  if (g->nouts + g->nins >= NX_MAX_OPERANDS) return NX_ARITY;
  for (int o = 0; o < g->nouts; o++) {
    int dt = nx_cpu_prog_out(g, o);
    nx_array *y = &g->a[o];
    memset(y, 0, sizeof *y);
    y->dtype = dt;
    y->bits = nx_dtype_row_of(dt).bits;
    y->rank = rank;
    for (int d = 0; d < rank; d++) y->dim[d] = shape[d];
  }
  for (int k = 0; k < g->nins; k++) g->a[g->nouts + k] = a[k];
  int e = setup(&g->j, g->s, g->nn, g->nouts, g->outs, g->a,
                g->nouts + g->nins, &g->heap, &g->most);
  if (!e) *index = g->j.index >= 0;
  return e;
}

int64_t nx_cpu_prog_kinds(const nx_cpu_prog *g) { return kinds(g->s, g->nn); }

double nx_cpu_prog_cost(const nx_cpu_prog *g) {
  return element_cost(g->s, g->nn);
}

void nx_cpu_prog_terms(const nx_cpu_prog *g, const nx_cpu_block *b,
                       uint8_t *const *to) {
  const prog *j = &g->j;
  int loads = g->nins + (j->index >= 0);
  nx_cpu_block c = {.n2 = 1};
  for (int o = 0; o < g->nouts; o++) c.s0[o] = 1;
  for (int k = 0; k < loads; k++) {
    c.s0[g->nouts + k] = b->s0[k];
    c.s1[g->nouts + k] = b->s1[k];
  }
  int64_t n0 = b->n0, n1 = b->n1, most = g->most;
  /* Pieces a block holds: whole rows, or pieces of one. */
  int64_t rows = n0 <= most ? most / n0 : 1, cut = n0 <= most ? n0 : most;
  uint8_t *t[NX_MAX_OPERANDS];
  for (int64_t r = 0; r < n1; r += rows)
    for (int64_t i = 0; i < n0; i += cut) {
      c.n0 = n0 - i < cut ? n0 - i : cut;
      c.n1 = n1 - r < rows ? n1 - r : rows;
      for (int k = 0; k < loads; k++)
        c.at[g->nouts + k] = b->at[k] + r * b->s1[k] + i * b->s0[k];
      for (int o = 0; o < g->nouts; o++)
        t[o] = to[o] == NULL
                   ? NULL
                   : to[o] + (r * n0 + i) * width(nx_cpu_prog_held(g, o));
      planes(&c, j, t);
    }
}

/* What nx_cpu_map holds on the C heap where it holds any: a long
   program's steps and its padded loads' pads. A custom block owns them,
   so that an exception nx_read raises, Out_of_memory or Rig.Lost, leaves
   them to its finalizer. A call of a short program and no padded load
   allocates none. */
typedef struct {
  step *s;
  nx_cpu_pad *pads;
} map_held;

static void map_release(value owner) {
  map_held *h = Data_custom_val(owner);
  free(h->s);
  free(h->pads);
  h->s = NULL;
  h->pads = NULL;
}

/* map_release, a long program's steps freed with the domain lock released
   (nx_free_released). */
static void map_drop(value owner, int long_program) {
  map_held *h = Data_custom_val(owner);
  void *steps = h->s;
  h->s = NULL;
  free(h->pads);
  h->pads = NULL;
  if (long_program) nx_free_released(steps);
  else free(steps);
}

static struct custom_operations map_ops = {
    "nx.cpu.map",               map_release,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

/* The paddings of a map's loads, copied out of OCaml before the door:
   [padded[k]] where load k is padded, with its pad at [pads[k]] and its
   fill's bits. */
typedef struct {
  int padded[NX_MAX_OPERANDS];
  nx_cpu_pad *pads;
  uint8_t fill[NX_MAX_OPERANDS][16];
} paddings;

/* Runs the map over the [n] operands [a] the door read, its loads from
   [nouts], each padded one through its padded copy, the copies'
   descriptors in the thread's kept scratch. Answers -1 if host memory
   runs out. */
static int run_padded(step *s, int nn, int nouts, const int32_t *outs,
                      nx_array *a, int n, const paddings *p) {
  if (p->pads == NULL) return run_map(s, nn, nouts, outs, a, n, OWN_PRICE);
  /* One descriptor more than the operands: the index's, which setup
     writes after them where the program reads a coordinate. */
  nx_array *run = nx_cpu_take((n + 1) * (int64_t)sizeof(nx_array));
  if (run == NULL) return -1;
  /* A program whose nodes' analysis costs a release's worth is priced
     with the lock released, as run_map analyses it. */
  int early = nn > NX_CPU_ITEMS, released = 0;
  if (early) released = nx_cpu_release(NX_CPU_RELEASE_BYTES);
  /* The price of the padded copies and the walk over them, each element
     strided: each copy has the loop's shape, the destinations', and its
     load's dtype; the walk's element is priced as run_map prices it, with
     an index's loads and stores. */
  double copies = 0, per = element_price(s, nn, a, n) + 2 * sizeof(int64_t);
  for (int k = 0; k < n - nouts; k++)
    if (p->padded[k]) copies += nx_cpu_width(a[nouts + k].dtype);
  if (!early)
    released = nx_cpu_release(nx_cpu_elements(&a[0]) * (copies + per) *
                              NX_CPU_STRIDED_COST);
  int copied = nouts, e = 0;
  memcpy(run, a, sizeof(nx_array) * (size_t)n);
  for (; copied < n && !e; copied++) {
    int k = copied - nouts;
    if (p->padded[k] &&
        nx_cpu_unpad(&a[copied], &p->pads[k], p->fill[k], &run[copied]))
      e = -1;
  }
  if (!e) e = run_map(s, nn, nouts, outs, run, n, CALLERS_PRICE);
  /* A copy that failed left its base NULL: free skips it. */
  for (int i = nouts; i < copied; i++)
    if (p->padded[i - nouts]) free(run[i].base);
  nx_cpu_give(run);
  nx_cpu_reacquire(released);
  return e;
}

/* The external: the program [vp], the destinations [vdsts], the loads
   [vloads] and their paddings [vpads], each Some (pad, fill's bits) for a
   padded load, or none where no load is padded. A load's array is its
   first field, Plain's or Padded's. The program and the paddings are
   copied before the door, which can run OCaml code: the steps of a long
   program and the pads on the heap the custom block owns. */
value nx_cpu_map(value vp, value vdsts, value vloads, value vpads) {
  CAMLparam4(vp, vdsts, vloads, vpads);
  CAMLlocal1(owner);
  int nouts = (int)Wosize_val(Field(vp, NX_PROG_OUTS));
  int nins = (int)Wosize_val(Field(vp, NX_PROG_INS));
  int nn = (int)Wosize_val(Field(vp, NX_PROG_NODES)), n = nouts + nins, e;
  if ((int)Wosize_val(vdsts) != nouts || (int)Wosize_val(vloads) != nins)
    CAMLreturn(Val_int(NX_ARITY));
  /* Prog.v holds a program to two operands fewer than a loop: room for
     the index. */
  if (n >= NX_MAX_OPERANDS) CAMLreturn(Val_int(NX_ARITY));
  int pads = Wosize_val(vpads) > 0, owned = pads || nn > STEPS;
  if (owned) {
    owner = caml_alloc_custom(&map_ops, sizeof(map_held), 0, 1);
    *(map_held *)Data_custom_val(owner) = (map_held){NULL, NULL};
  }
  paddings p;
  p.pads = NULL;
  if (pads) {
    p.pads = malloc((size_t)nins * sizeof *p.pads);
    if (p.pads == NULL) caml_raise_out_of_memory();
    ((map_held *)Data_custom_val(owner))->pads = p.pads;
    for (int k = 0; k < nins; k++) {
      value v = Field(vpads, k);
      p.padded[k] = Is_block(v);
      if (p.padded[k] &&
          nx_cpu_padding_of(Field(v, 0), Field(Field(vloads, k), 0),
                            &p.pads[k], p.fill[k])) {
        map_release(owner);
        CAMLreturn(Val_int(NX_SHAPE));
      }
    }
  }
  step local[STEPS], *s = local;
  if (nn > STEPS) {
    s = malloc(sizeof(step) * (size_t)nn);
    if (s == NULL) caml_raise_out_of_memory();
    ((map_held *)Data_custom_val(owner))->s = s;
  }
  int ins[NX_MAX_OPERANDS];
  int32_t outs[NX_MAX_OPERANDS];
  copy_prog(&vp, s, ins, outs);
  nx_operand in[NX_MAX_OPERANDS];
  for (int o = 0; o < nouts; o++)
    in[o] = (nx_operand){Field(vdsts, o), s[outs[o]].dt, 1};
  for (int k = 0; k < nins; k++)
    in[nouts + k] = (nx_operand){Field(Field(vloads, k), 0), ins[k], 0};
  nx_array a[NX_MAX_OPERANDS];
  if (!(e = nx_read(n, in, a))) {
    e = run_padded(s, nn, nouts, outs, a, n, &p);
    nx_done(n, a);
  }
  if (owned) map_drop(owner, nn > NX_CPU_ITEMS);
  if (e < 0) caml_raise_out_of_memory();
  CAMLreturn(Val_int(e));
}
