/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_array.h"
#include "nx_contract_view_fill.h"
#include "nx_prog.h"

_Static_assert(NX_OP1_COUNT == 31 && NX_OP2_COUNT == 18 && NX_OP3_COUNT == 2,
               "one code per kind");

/* The bits nx_dtype.h's store of [x] writes into an element of the float
   dtype [dt] of at most 16 bits. */
intnat nx_kernel_narrow_bits(intnat dt, double x) {
  return (intnat)nx_double_to_bits((int)dt, x);
}

value nx_kernel_narrow_bits_byte(value dt, value x) {
  return Val_long(nx_kernel_narrow_bits(Long_val(dt), Double_val(x)));
}

_Static_assert(NX_VIEW_DST == 3 && NX_VIEW_CONTRACTED == 3,
               "operand and axis indices");

/* nx_contract_view_fill, in one pass over the pairs [vbatch] and
   [vcontracting] and the layouts of [y], [a], [b] and the init [i], an
   option, into [w]: the paired axes marked in a mask per side, each
   operand's axes of each group gathered, every group's extents checked
   before any merges, then each group coalesced (nx_coalesce_dims) into
   its runs and written. It reads each pair once, so another domain's write
   to the arrays cannot change what it checked. */
enum { A = NX_VIEW_A, B = NX_VIEW_B, I = NX_VIEW_INIT, Y = NX_VIEW_DST };
enum {
  BATCH = NX_VIEW_BATCH,
  ROW = NX_VIEW_ROW,
  COLUMN = NX_VIEW_COLUMN,
  CONTRACTED = NX_VIEW_CONTRACTED
};

typedef nx_layout layout;

/* Groups the [count] axes of group [g] over the [n] operands [ops], whose
   axes are [axes][op][g], into its runs in [r] (nx_contract_view_fill.h):
   one run where there are at most one axis, as nx_coalesce_dims writes it,
   an absent or single axis of extent 1 stepping 0, as does an empty one.
   The operands not in [ops] step 0. */
static void group(nx_contract_runs *r, int g, int count, int n,
                  const int *ops, const layout *l,
                  int8_t axes[4][4][NX_MAX_RANK]) {
  int64_t ext[NX_MAX_RANK], st[4][NX_MAX_RANK];
  int runs = 1;
  if (count <= 1) {
    ext[0] = count ? l[ops[0]].dim[axes[ops[0]][g][0]] : 1;
    for (int p = 0; p < n; p++) {
      const layout *lp = &l[ops[p]];
      st[p][0] = ext[0] > 1 ? lp->dim[lp->rank + axes[ops[p]][g][0]] : 0;
    }
  } else {
    for (int k = 0; k < count; k++) {
      ext[k] = l[ops[0]].dim[axes[ops[0]][g][k]];
      for (int p = 0; p < n; p++) {
        const layout *lp = &l[ops[p]];
        st[p][k] = lp->dim[lp->rank + axes[ops[p]][g][k]];
      }
    }
    runs = nx_coalesce_dims(n, count, ext, st);
  }
  r->runs[g] = runs;
  for (int o = A; o <= Y; o++)
    for (int k = 0; k < runs; k++) r->stride[o][g][k] = 0;
  for (int k = 0; k < runs; k++) {
    r->extent[g][k] = ext[k];
    for (int p = 0; p < n; p++) r->stride[ops[p]][g][k] = st[p][k];
  }
}

/* Whether the operands [ops] agree on group [g]'s [count] extents. */
static int fits(int g, int count, int n, const int *ops, const layout *l,
                int8_t axes[4][4][NX_MAX_RANK]) {
  for (int p = 1; p < n; p++)
    for (int k = 0; k < count; k++)
      if (l[ops[p]].dim[axes[ops[p]][g][k]] !=
          l[ops[0]].dim[axes[ops[0]][g][k]])
        return 0;
  return 1;
}

/* Whether operand [o] spans group [g]: batch all, rows a, init and dst,
   columns b, init and dst, contracted a and b. */
static const int spans[4][4] = {
    [A] = {[BATCH] = 1, [ROW] = 1, [CONTRACTED] = 1},
    [B] = {[BATCH] = 1, [COLUMN] = 1, [CONTRACTED] = 1},
    [I] = {[BATCH] = 1, [ROW] = 1, [COLUMN] = 1},
    [Y] = {[BATCH] = 1, [ROW] = 1, [COLUMN] = 1}};

/* The view where each group holds [count[g]] of at most one axis, as a
   product of matrices' groups do: each group's extent is its first
   operand's (b's for the columns, a's for the others), which every other
   operand's must equal, and each operand steps along its axis, 0 over at
   most one element, as fits and group check and write them, without
   their lists. On the M1 Max it took an 8³ product's fill from 38 to 18
   ns. */
static int single(nx_contract_view *w, const layout *l, int init,
                  const int *count, int8_t axes[4][4][NX_MAX_RANK]) {
  int64_t ext[4];
  for (int g = BATCH; g <= CONTRACTED; g++) {
    int o = g == COLUMN ? B : A;
    ext[g] = count[g] ? l[o].dim[axes[o][g][0]] : 1;
  }
  for (int o = A; o <= Y; o++)
    for (int g = BATCH; g <= CONTRACTED; g++)
      if ((o != I || init) && count[g] && spans[o][g] &&
          l[o].dim[axes[o][g][0]] != ext[g])
        return NX_SHAPE;
  memset(w, 0, sizeof *w);
  for (int o = A; o <= Y; o++) {
    if (o == I && !init) continue;
    w->offset[o] = l[o].offset;
    for (int g = BATCH; g <= CONTRACTED; g++)
      if (count[g] && spans[o][g] && ext[g] > 1)
        w->stride[o][g] = l[o].dim[l[o].rank + axes[o][g][0]];
  }
  for (int g = BATCH; g <= CONTRACTED; g++) w->extent[g] = ext[g];
  return NX_OK;
}

/* A contraction's pairs read and checked: the layouts, each operand's
   axes by group, the axes in each group and whether it has an init. */
typedef struct {
  layout l[4];
  int8_t axes[4][4][NX_MAX_RANK];
  int count[4];
  int init;
} pairs;

/* Checks the [nb] batch pairs then the [nc] contracting pairs [ps], a's
   axis then b's, against the layouts in [q], and gathers each operand's
   axes by group: NX_OK, or NX_SHAPE where they do not fit the layouts'
   ranks or an axis is in two pairs. */
static int pairs_of(pairs *q, int nb, int nc, const int (*ps)[2]) {
  layout *l = q->l;
  if (nb + nc > l[A].rank || nb + nc > l[B].rank) return NX_SHAPE;
  int fa = l[A].rank - nb - nc, fb = l[B].rank - nb - nc;
  int ry = nb + fa + fb;
  uint64_t na = 0, nbm = 0;
  for (int k = 0; k < nb + nc; k++) {
    int ia = ps[k][0], ib = ps[k][1];
    if (ia < 0 || ia >= l[A].rank || ib < 0 || ib >= l[B].rank ||
        na >> ia & 1 || nbm >> ib & 1)
      return NX_SHAPE;
    na |= (uint64_t)1 << ia;
    nbm |= (uint64_t)1 << ib;
    int g = k < nb ? BATCH : CONTRACTED, at = k < nb ? k : k - nb;
    q->axes[A][g][at] = (int8_t)ia;
    q->axes[B][g][at] = (int8_t)ib;
  }
  if (l[Y].rank != ry || (q->init && l[I].rank != ry)) return NX_SHAPE;
  for (int ax = 0, n = 0; ax < l[A].rank; ax++)
    if (!(na >> ax & 1)) q->axes[A][ROW][n++] = ax;
  for (int ax = 0, n = 0; ax < l[B].rank; ax++)
    if (!(nbm >> ax & 1)) q->axes[B][COLUMN][n++] = ax;
  for (int o = I; o <= Y; o++) {
    for (int k = 0; k < nb; k++) q->axes[o][BATCH][k] = k;
    for (int k = 0; k < fa; k++) q->axes[o][ROW][k] = nb + k;
    for (int k = 0; k < fb; k++) q->axes[o][COLUMN][k] = nb + fa + k;
  }
  q->count[BATCH] = nb;
  q->count[ROW] = fa;
  q->count[COLUMN] = fb;
  q->count[CONTRACTED] = nc;
  return NX_OK;
}

/* Reads the pairs [vbatch] and [vcontracting] and the layouts of [y], [a],
   [b] and the init [i] into [q] (pairs_of). */
static int read_pairs(pairs *q, value vbatch, value vcontracting, value y,
                      value a, value b, value i) {
  int nb = (int)Wosize_val(vbatch), nc = (int)Wosize_val(vcontracting);
  layout *l = q->l;
  q->init = Is_block(i);
  l[A].rank = nx_array_layout(a, l[A].dim, &l[A].offset);
  l[B].rank = nx_array_layout(b, l[B].dim, &l[B].offset);
  l[Y].rank = nx_array_layout(y, l[Y].dim, &l[Y].offset);
  l[I].rank =
      q->init ? nx_array_layout(Field(i, 0), l[I].dim, &l[I].offset) : 0;
  if (nb + nc > l[A].rank || nb + nc > l[B].rank) return NX_SHAPE;
  int ps[NX_MAX_RANK][2];
  for (int k = 0; k < nb + nc; k++) {
    value p = Field(k < nb ? vbatch : vcontracting, k < nb ? k : k - nb);
    intnat ia = Long_val(Field(p, 0)), ib = Long_val(Field(p, 1));
    /* Out of an int's range is out of the ranks'. */
    ps[k][0] = ia < 0 || ia > NX_MAX_RANK ? -1 : (int)ia;
    ps[k][1] = ib < 0 || ib > NX_MAX_RANK ? -1 : (int)ib;
  }
  return pairs_of(q, nb, nc, ps);
}

/* Groups [q] into [r]: NX_OK, or NX_SHAPE where the operands' extents
   differ within a group. Each group's operands: batch over all, rows over
   a, init and dst, columns over b, init and dst, contracted over a and
   b. */
static int runs_of(nx_contract_runs *r, pairs *q) {
  int batch[4] = {A, B, Y, I}, row[3] = {A, Y, I}, column[3] = {B, Y, I};
  int contracted[2] = {A, B};
  int nbatch = 3 + q->init, nrow = 2 + q->init;
  const int *c = q->count;
  /* Every group fits before any merges: a misfit is refused even behind a
     group that does not merge. */
  if (!fits(BATCH, c[BATCH], nbatch, batch, q->l, q->axes) ||
      !fits(ROW, c[ROW], nrow, row, q->l, q->axes) ||
      !fits(COLUMN, c[COLUMN], nrow, column, q->l, q->axes) ||
      !fits(CONTRACTED, c[CONTRACTED], 2, contracted, q->l, q->axes))
    return NX_SHAPE;
  group(r, BATCH, c[BATCH], nbatch, batch, q->l, q->axes);
  group(r, ROW, c[ROW], nrow, row, q->l, q->axes);
  group(r, COLUMN, c[COLUMN], nrow, column, q->l, q->axes);
  group(r, CONTRACTED, c[CONTRACTED], 2, contracted, q->l, q->axes);
  for (int o = A; o <= Y; o++)
    r->offset[o] = o == I && !q->init ? 0 : q->l[o].offset;
  return NX_OK;
}

/* The view or the runs of the pairs read into [q] (nx_contract_fill). */
static int fill_of(nx_contract_view *w, nx_contract_runs *r, pairs *q) {
  const int *c = q->count;
  if (c[BATCH] <= 1 && c[ROW] <= 1 && c[COLUMN] <= 1 && c[CONTRACTED] <= 1)
    return single(w, q->l, q->init, c, q->axes);
  int e;
  if ((e = runs_of(r, q))) return e;
  for (int g = BATCH; g <= CONTRACTED; g++)
    if (r->runs[g] != 1) return NX_DECLINED;
  memset(w, 0, sizeof *w);
  for (int o = A; o <= Y; o++) {
    w->offset[o] = r->offset[o];
    for (int g = BATCH; g <= CONTRACTED; g++)
      w->stride[o][g] = r->stride[o][g][0];
  }
  for (int g = BATCH; g <= CONTRACTED; g++) w->extent[g] = r->extent[g][0];
  return NX_OK;
}

int nx_contract_fill(nx_contract_view *w, nx_contract_runs *r, value vbatch,
                     value vcontracting, value y, value a, value b,
                     value i) {
  pairs q;
  int e = read_pairs(&q, vbatch, vcontracting, y, a, b, i);
  if (e) return e;
  return fill_of(w, r, &q);
}

int nx_contract_fill_layouts(nx_contract_view *w, nx_contract_runs *r,
                             int nb, int nc, const int (*ps)[2],
                             const nx_layout *y, const nx_layout *a,
                             const nx_layout *b, const nx_layout *i) {
  pairs q;
  q.l[A] = *a;
  q.l[B] = *b;
  q.l[Y] = *y;
  q.init = i != NULL;
  q.l[I].rank = 0;
  if (i) q.l[I] = *i;
  int e = pairs_of(&q, nb, nc, ps);
  if (e) return e;
  return fill_of(w, r, &q);
}

int nx_contract_view_fill(nx_contract_view *w, value vbatch,
                          value vcontracting, value y, value a, value b,
                          value i) {
  nx_contract_runs r;
  return nx_contract_fill(w, &r, vbatch, vcontracting, y, a, b, i);
}

/* Expressions */

/* Prog.E.t's layout (prog.ml): In of operand and dtype, Coord of axis,
   Const of dtype and bits, then Op1, Op2 and Op3 of kind and dtype, whose
   operands, tag - 2 of them, follow from field 2. */
enum { NX_E_IN, NX_E_COORD, NX_E_CONST, NX_E_OP1, NX_E_OP2, NX_E_OP3 };

static int nx_operands(value e) {
  return Tag_val(e) >= NX_E_OP1 ? (int)Tag_val(e) - 2 : 0;
}

/* A node's structure: its tag, its fields as ints (an operand, an axis, a
   dtype's code, a kind's code), its operands' nodes, and a constant's bits. */
typedef struct {
  int32_t tag, x0, x1, ops[3];
  value bits;
} nx_shape;

/* A block being walked, and the nodes of the operands walked so far. */
typedef struct {
  value e;
  int next;
  int32_t ops[3];
} nx_frame;

/* Each buffer starts at 128 nodes and grows eightfold: a large program
   pays for few fresh buffers and rehashes. */
enum { NX_FIRST = 128, NX_GROWTH = 8 };

/* A walk's state, all of it outside the OCaml heap: a table from block
   addresses to their nodes; the nodes' shapes, and a table from shapes to
   nodes, both open addressing grown at half load; and the stack of blocks
   being walked. */
typedef struct {
  value *keys;
  int32_t *vals;
  size_t mask, used;
  nx_shape *shapes;
  size_t nodes, shapes_cap;
  int32_t *slots;
  size_t slots_mask;
  nx_frame *stack;
  size_t depth, stack_cap;
  int32_t *outs;
} nx_walk;

static void nx_walk_free(nx_walk *w) {
  free(w->keys);
  free(w->vals);
  free(w->shapes);
  free(w->slots);
  free(w->stack);
  free(w->outs);
}

static void nx_walk_fail(nx_walk *w) {
  nx_walk_free(w);
  caml_raise_out_of_memory();
}

/* Grows the capacity [*cap] of [*p] NX_GROWTH times, of elements of [size] bytes. */
static void nx_walk_grow(nx_walk *w, void **p, size_t *cap, size_t size) {
  void *q = malloc(NX_GROWTH * *cap * size);
  if (q == NULL) nx_walk_fail(w);
  memcpy(q, *p, *cap * size);
  free(*p);
  *p = q;
  *cap *= NX_GROWTH;
}

/* The slot of hash [h] in a table of [mask + 1] slots, a power of two:
   the top bits of Fibonacci hashing, one multiply. */
static size_t nx_index(uint64_t h, size_t mask) {
  return (size_t)((h * 0x9E3779B97F4A7C15ull) >> __builtin_clzll(mask));
}

static size_t nx_slot(value v, size_t mask) {
  return nx_index((uint64_t)v, mask);
}

static int32_t nx_find(const nx_walk *w, value v) {
  for (size_t i = nx_slot(v, w->mask);; i = (i + 1) & w->mask) {
    if (w->keys[i] == 0) return -1;
    if (w->keys[i] == v) return w->vals[i];
  }
}

static void nx_put(value *keys, int32_t *vals, size_t mask, value v,
                   int32_t k) {
  size_t i = nx_slot(v, mask);
  while (keys[i] != 0) i = (i + 1) & mask;
  keys[i] = v;
  vals[i] = k;
}

static void nx_insert(nx_walk *w, value v, int32_t k) {
  if (2 * (w->used + 1) > w->mask + 1) {
    size_t mask = NX_GROWTH * (w->mask + 1) - 1;
    value *keys = calloc(mask + 1, sizeof(value));
    int32_t *vals = malloc((mask + 1) * sizeof(int32_t));
    if (keys == NULL || vals == NULL) {
      free(keys);
      free(vals);
      nx_walk_fail(w);
    }
    for (size_t i = 0; i <= w->mask; i++)
      if (w->keys[i] != 0) nx_put(keys, vals, mask, w->keys[i], w->vals[i]);
    free(w->keys);
    free(w->vals);
    w->keys = keys;
    w->vals = vals;
    w->mask = mask;
  }
  nx_put(w->keys, w->vals, w->mask, v, k);
  w->used++;
}

/* The shape of the block of [f], whose operands are walked. */
static nx_shape nx_shape_of(const nx_frame *f) {
  value e = f->e, k = Field(e, 0);
  nx_shape s = {(int32_t)Tag_val(e), 0, 0, {-1, -1, -1}, 0};
  switch (s.tag) {
  case NX_E_IN:
    s.x0 = (int32_t)Long_val(k);
    s.x1 = (int32_t)Long_val(Field(e, 1));
    break;
  case NX_E_COORD:
    s.x0 = (int32_t)Long_val(k);
    break;
  case NX_E_CONST:
    s.x0 = (int32_t)Long_val(k);
    s.bits = Field(e, 1);
    break;
  case NX_E_OP1:
    s.x0 = NX_OP1_CODE(k);
    s.x1 = (int32_t)Long_val(Field(e, 1));
    break;
  case NX_E_OP2:
    s.x0 = NX_OP2_CODE(k);
    break;
  default:
    s.x0 = (int32_t)Long_val(k);
  }
  for (int i = 0; i < nx_operands(e); i++) s.ops[i] = f->ops[i];
  return s;
}

/* The shape's fields packed into three words, mixed by independent
   multiplies; a constant's bits, of at most 16 bytes, into a fourth. */
static uint64_t nx_hash(const nx_shape *s) {
  uint64_t a = (uint64_t)(uint32_t)s->tag | (uint64_t)(uint32_t)s->x0 << 8 |
               (uint64_t)(uint32_t)s->x1 << 40;
  uint64_t b = (uint64_t)(uint32_t)s->ops[0] |
               (uint64_t)(uint32_t)s->ops[1] << 32;
  uint64_t c = (uint32_t)s->ops[2];
  uint64_t h = a * 0xC2B2AE3D27D4EB4Full + b * 0x165667B19E3779F9ull +
               c * 0x27D4EB2F165667C5ull;
  if (s->bits != 0) {
    uint64_t q[2] = {0, 0};
    mlsize_t n = caml_string_length(s->bits);
    memcpy(q, String_val(s->bits), n < sizeof q ? n : sizeof q);
    h += q[0] * 0x9FB21C651E98DF25ull + q[1] * 0xFF51AFD7ED558CCDull;
  }
  return h ^ (h >> 29);
}

static int nx_same(const nx_shape *a, const nx_shape *b) {
  if (a->tag != b->tag || a->x0 != b->x0 || a->x1 != b->x1 ||
      a->ops[0] != b->ops[0] || a->ops[1] != b->ops[1] ||
      a->ops[2] != b->ops[2])
    return 0;
  if (a->bits == 0) return 1;
  mlsize_t n = caml_string_length(a->bits);
  return n == caml_string_length(b->bits) &&
         memcmp(String_val(a->bits), String_val(b->bits), n) == 0;
}

/* The node of [s]: the node of that shape, made now if there is none. */
static int32_t nx_node(nx_walk *w, const nx_shape *s) {
  uint64_t h = nx_hash(s);
  size_t i = nx_index(h, w->slots_mask);
  for (; w->slots[i] != 0; i = (i + 1) & w->slots_mask)
    if (nx_same(&w->shapes[w->slots[i] - 1], s)) return w->slots[i] - 1;
  if (w->nodes == w->shapes_cap)
    nx_walk_grow(w, (void **)&w->shapes, &w->shapes_cap, sizeof(nx_shape));
  w->shapes[w->nodes] = *s;
  w->slots[i] = (int32_t)++w->nodes;
  if (2 * w->nodes > w->slots_mask + 1) {
    size_t mask = NX_GROWTH * (w->slots_mask + 1) - 1;
    int32_t *slots = calloc(mask + 1, sizeof(int32_t));
    if (slots == NULL) nx_walk_fail(w);
    for (size_t n = 0; n < w->nodes; n++) {
      size_t j = nx_index(nx_hash(&w->shapes[n]), mask);
      while (slots[j] != 0) j = (j + 1) & mask;
      slots[j] = (int32_t)n + 1;
    }
    free(w->slots);
    w->slots = slots;
    w->slots_mask = mask;
  }
  return (int32_t)w->nodes - 1;
}

/* Walks [root], not met before, and is its node. */
static int32_t nx_walk_root(nx_walk *w, value root) {
  int32_t k = -1;
  w->stack[w->depth++] = (nx_frame){root, 0, {-1, -1, -1}};
  while (w->depth > 0) {
    nx_frame *f = &w->stack[w->depth - 1];
    if (f->next < nx_operands(f->e)) {
      value c = Field(f->e, 2 + f->next);
      int32_t j = nx_find(w, c);
      if (j >= 0) {
        f->ops[f->next++] = j;
        continue;
      }
      if (w->depth == w->stack_cap)
        nx_walk_grow(w, (void **)&w->stack, &w->stack_cap, sizeof(nx_frame));
      w->stack[w->depth++] = (nx_frame){c, 0, {-1, -1, -1}};
      continue;
    }
    nx_shape s = nx_shape_of(f);
    k = nx_node(w, &s);
    nx_insert(w, f->e, k);
    if (--w->depth > 0) {
      nx_frame *p = &w->stack[w->depth - 1];
      p->ops[p->next++] = k;
    }
  }
  return k;
}

/* Prog.walk: no OCaml allocation runs until the nodes are copied out, so
   no collection moves a block while the tables hold its address. */
value nx_kernel_prog_walk(value outs) {
  CAMLparam1(outs);
  CAMLlocal1(r);
  size_t nouts = Wosize_val(outs);
  nx_walk w = {calloc(2 * NX_FIRST, sizeof(value)),
               malloc(2 * NX_FIRST * sizeof(int32_t)), 2 * NX_FIRST - 1, 0,
               malloc(NX_FIRST * sizeof(nx_shape)), 0, NX_FIRST,
               calloc(2 * NX_FIRST, sizeof(int32_t)), 2 * NX_FIRST - 1,
               malloc(NX_FIRST * sizeof(nx_frame)), 0, NX_FIRST,
               malloc((nouts > 0 ? nouts : 1) * sizeof(int32_t))};
  if (!w.keys || !w.vals || !w.shapes || !w.slots || !w.stack || !w.outs)
    nx_walk_fail(&w);
  for (size_t o = 0; o < nouts; o++) {
    value root = Field(outs, o);
    int32_t k = nx_find(&w, root);
    w.outs[o] = k >= 0 ? k : nx_walk_root(&w, root);
  }
  /* Six words per node, then the constants' bits, each padded to whole
     words. */
  size_t words = 2 + nouts + 6 * w.nodes;
  for (size_t n = 0; n < w.nodes; n++)
    if (w.shapes[n].bits != 0)
      words += (caml_string_length(w.shapes[n].bits) + 3) / 4;
  /* The walk is over: allocation may move blocks from here on, so the
     nodes, bits included, are copied out before it. */
  int32_t *buf = malloc(words * sizeof(int32_t));
  if (buf == NULL) nx_walk_fail(&w);
  int32_t *at = buf, *bits = buf + 2 + nouts + 6 * w.nodes;
  *at++ = (int32_t)w.nodes;
  *at++ = (int32_t)nouts;
  for (size_t o = 0; o < nouts; o++) *at++ = w.outs[o];
  for (size_t n = 0; n < w.nodes; n++) {
    const nx_shape *s = &w.shapes[n];
    int32_t rec[6] = {s->tag, s->x0, s->x1, s->ops[0], s->ops[1], s->ops[2]};
    if (s->bits != 0) {
      mlsize_t len = caml_string_length(s->bits);
      rec[2] = (int32_t)(bits - buf);
      rec[3] = (int32_t)len;
      memset(bits, 0, (len + 3) / 4 * sizeof(int32_t));
      memcpy(bits, String_val(s->bits), len);
      bits += (len + 3) / 4;
    }
    memcpy(at, rec, sizeof rec);
    at += 6;
  }
  nx_walk_free(&w);
  r = caml_alloc_initialized_string(words * sizeof(int32_t), (char *)buf);
  free(buf);
  CAMLreturn(r);
}
