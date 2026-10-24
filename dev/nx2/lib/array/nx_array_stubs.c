/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <stdlib.h>
#include <string.h>

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/callback.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>

#include "nx_array.h"
#include "nx_layout.h"
#include "rig.h"

#ifndef FLAT_FLOAT_ARRAY
#error "nx_array writes float arrays as flat arrays of doubles"
#endif

/* An array is the record { dtype; layout; buffer }, in this order: the
   dtype an immediate whose value is its code, the layout a record
   (nx_layout.h). This file is the only C that reads it. */
enum { ARRAY_DTYPE, ARRAY_LAYOUT, ARRAY_BUFFER };

/* Layouts */

/* Copies the layout [v] into [a]'s rank, flags, offset and dims: the record
   lives in the OCaml heap, which moves. Layout's constructors make every
   layout well formed (strides of its shape's length, at most
   NX_MAX_RANK), so nothing here checks it again. */
static void read_layout(value v, nx_array *a) {
  value shape = Field(v, NX_LAYOUT_SHAPE);
  value strides = Field(v, NX_LAYOUT_STRIDES);
  int r = (int)Wosize_val(shape);
  a->rank = r;
  a->offset = Long_val(Field(v, NX_LAYOUT_OFFSET));
  a->flags = (int)Long_val(Field(v, NX_LAYOUT_FLAGS));
  for (int i = 0; i < r; i++) {
    a->dim[i] = Long_val(Field(shape, i));
    a->dim[r + i] = Long_val(Field(strides, i));
  }
}

/* The door */

int nx_array_dtype(value v) { return (int)Long_val(Field(v, ARRAY_DTYPE)); }

int nx_array_layout(value v, int64_t *dim, int64_t *offset) {
  value l = Field(v, ARRAY_LAYOUT);
  value shape = Field(l, NX_LAYOUT_SHAPE);
  value strides = Field(l, NX_LAYOUT_STRIDES);
  int r = (int)Wosize_val(shape);
  for (int i = 0; i < r; i++) {
    dim[i] = Long_val(Field(shape, i));
    dim[r + i] = Long_val(Field(strides, i));
  }
  *offset = Long_val(Field(l, NX_LAYOUT_OFFSET));
  return r;
}

static int claim_code(enum rig_claim c) {
  switch (c) {
    case RIG_CLAIMED:
    case RIG_WAIT:
    case RIG_LOST: return NX_OK;
    case RIG_DEAD: return NX_DEAD;
    case RIG_EXCLUSIVE: return NX_EXCLUSIVE;
    case RIG_READ_ONLY: return NX_READ_ONLY;
  }
  caml_fatal_error("nx_read: rig_buffer_claim answered %d", (int)c);
}

/* Where an operand's bytes lie: [lo, hi) bytes past its buffer's first,
   from its layout's span. */
typedef struct {
  int64_t lo, hi;
} where;

/* Whether the operands [a] and [b], both with elements, share a byte. Two
   the host addresses compare host addresses; others ask rig, as
   Rig.Buffer.overlaps of the views of their bytes would answer, which sees
   a borrow and the memory it maps, and an io memory and a bigarray over
   its pages. A layout lies within its buffer (Nx_array.v), so rig's bounds
   hold; were one past it, the pair would count as sharing. */
static int meet(const nx_array *a, const where *wa, const nx_array *b,
                const where *wb) {
  if (a->base != NULL && b->base != NULL) {
    intptr_t x = (intptr_t)a->base, y = (intptr_t)b->base;
    return x + wa->lo < y + wb->hi && y + wb->lo < x + wa->hi;
  }
  if ((size_t)wa->hi > rig_buffer_bytes(a->buffer) ||
      (size_t)wb->hi > rig_buffer_bytes(b->buffer))
    return 1;
  return rig_buffers_overlap(a->buffer, wa->lo, wa->hi - wa->lo, b->buffer,
                             wb->lo, wb->hi - wb->lo);
}

/* Whether [a] and [b], which share a byte, are identical: one width, and
   every index at one byte: equal extents and strides, and one first bit,
   the bits of its byte compared, the byte by host address or by rig. */
static int identical(const nx_array *a, const nx_array *b) {
  if (a->bits != b->bits || a->rank != b->rank) return 0;
  for (int i = 0; i < 2 * a->rank; i++)
    if (a->dim[i] != b->dim[i]) return 0;
  int64_t fa = a->offset * a->bits, fb = b->offset * b->bits;
  if (fa % 8 != fb % 8) return 0;
  if (a->base != NULL && b->base != NULL)
    return (intptr_t)a->base + fa / 8 == (intptr_t)b->base + fb / 8;
  if ((size_t)(fa / 8) >= rig_buffer_bytes(a->buffer) ||
      (size_t)(fb / 8) >= rig_buffer_bytes(b->buffer))
    return 0;
  return rig_buffers_overlap(a->buffer, fa / 8, 1, b->buffer, fb / 8, 1);
}

/* Releases the claims of the first [n] operands [a] made, and answers [e]: a
   refusal after some claims. */
static int refuse_claimed(int n, const nx_array *a, int e) {
  for (int k = 0; k < n; k++) rig_buffer_release(a[k].buffer);
  return e;
}

/* The door's checks, which claim every operand or none, for [host] kernels
   or for a kernel that submits device work: [w] holds n operands' byte
   ranges. */
static int checks(int n, const nx_operand *in, nx_array *out, where *w,
                  int host) {
  for (int k = 0; k < n; k++) {
    value v = in[k].array;
    nx_array *a = &out[k];
    a->dtype = nx_array_dtype(v);
    if (a->dtype != in[k].dtype) return NX_DTYPE;
    a->bits = nx_dtype_row_of(a->dtype).bits;
    read_layout(Field(v, ARRAY_LAYOUT), a);
    a->buffer = Field(v, ARRAY_BUFFER);
    a->alias = 0;
  }
  /* Every operand is claimed before anything else is checked: the claim
     checks under it that the buffer lives, so a dead operand is refused as
     dead whatever else is wrong with it, and before any wait, so a wait
     runs under every claim and no donation on another domain consumes an
     operand while the call waits. The claims go in runs of up to
     NX_MAX_OPERANDS, gathered on the stack, and a run refused releases the
     runs before it. */
  value buffer[NX_MAX_OPERANDS];
  enum rig_access access[NX_MAX_OPERANDS];
  enum rig_claim answer[NX_MAX_OPERANDS];
  for (int at = 0; at < n; at += NX_MAX_OPERANDS) {
    int m = n - at < NX_MAX_OPERANDS ? n - at : NX_MAX_OPERANDS;
    for (int j = 0; j < m; j++) {
      buffer[j] = out[at + j].buffer;
      access[j] = in[at + j].written ? RIG_READ_WRITE : RIG_READ;
    }
    int refused = rig_buffers_claim(m, buffer, access, answer);
    if (refused >= 0)
      return refuse_claimed(at, out, claim_code(answer[refused]));
    for (int j = 0; j < m; j++) out[at + j].wait = answer[j];
  }
  /* Each operand's bytes, read once with its layout's span. */
  for (int k = 0; k < n; k++) {
    nx_array *a = &out[k];
    if (in[k].written && !(a->flags & NX_DISTINCT))
      return refuse_claimed(n, out, NX_NOT_DISTINCT);
    if (a->flags & NX_EMPTY) {
      a->base = NULL;
      w[k].lo = w[k].hi = 0;
      continue;
    }
    a->base = rig_buffer_host(a->buffer);
    if (host && a->base == NULL) return refuse_claimed(n, out, NX_NOT_HOST);
    value l = Field(in[k].array, ARRAY_LAYOUT);
    w[k].lo = Long_val(Field(l, NX_LAYOUT_LO)) * a->bits / 8;
    w[k].hi = (Long_val(Field(l, NX_LAYOUT_HI)) * a->bits + 7) / 8;
  }
  /* A written operand shares bytes with no other but a read operand
     identical to it, which is marked its alias. Identity is of bytes, and
     two memories can hold the same bytes, such as a host borrow of a
     device's memory and a host buffer over the borrow's bigarray: each
     operand keeps its own claim and wait, for its own memory's work and
     loss. Read operands may share bytes, so only pairs with a written
     operand are compared. */
  for (int k = 0; k < n; k++) {
    if (!in[k].written || w[k].lo == w[k].hi) continue;
    for (int j = 0; j < n; j++) {
      if (j == k || out[j].alias || w[j].lo == w[j].hi) continue;
      if (!meet(&out[k], &w[k], &out[j], &w[j])) continue;
      if (in[j].written || !identical(&out[k], &out[j]))
        return refuse_claimed(n, out, NX_OVERLAP);
      out[j].alias = 1;
    }
  }
  return NX_OK;
}

/* checks, its byte ranges on the stack up to NX_MAX_OPERANDS operands,
   beyond on the heap, allocated before any claim. */
static int admit(int n, const nx_operand *in, nx_array *out, int host) {
  where local[NX_MAX_OPERANDS], *w = local;
  if (n > NX_MAX_OPERANDS) {
    w = malloc((size_t)n * sizeof *w);
    if (w == NULL) caml_raise_out_of_memory();
  }
  int e = checks(n, in, out, w, host);
  if (w != local) free(w);
  return e;
}

/* Unlinks the roots of the [n] operands [a], pushed in order through
   a[0].local, under any pushed since. */
static void unlink_roots(int n, nx_array *a) {
  struct caml__roots_block **p = a[0].local;
  while (*p != &a[n - 1].roots) {
    if (*p == NULL)
      caml_fatal_error("nx_done: the descriptors' roots were popped before "
                       "nx_done");
    p = &(*p)->next;
  }
  *p = a[0].roots.next;
}

/* Many operands

   Past NX_MAX_OPERANDS operands the door works in runs of that many, and
   releases the domain lock and takes it back between two runs
   (nx_let_collect), so that no stretch it holds the lock grows with the
   count. A run first reads its operands and links their buffers as roots,
   as a release may move OCaml values; then each run claims its operands;
   then the checks that read C data alone run with the lock released, the
   tables sized by the count are freed released, and a wait or a raise
   goes in runs too. The refusals keep nx_read's order: the dtypes, then
   the claims, then the rest. */

void nx_let_collect(void) {
  caml_enter_blocking_section_no_pending();
  caml_leave_blocking_section();
}

void nx_free_released(void *p) {
  if (p == NULL) return;
  caml_enter_blocking_section_no_pending();
  free(p);
  caml_leave_blocking_section();
}

/* The least page size of the hosts nx runs on: a write every this many
   bytes reaches each page. */
#define PAGE_BYTES 4096

void *nx_alloc_released(size_t n) {
  uint8_t *p = malloc(n);
  if (p == NULL) return NULL;
  caml_enter_blocking_section_no_pending();
  for (size_t i = 0; i < n; i += PAGE_BYTES) p[i] = 0;
  p[n - 1] = 0;
  caml_leave_blocking_section();
  return p;
}

/* The end of the run of operands from [at], of [n]. */
static int run_end(int at, int n) {
  return n - at < NX_MAX_OPERANDS ? n : at + NX_MAX_OPERANDS;
}

/* Releases the claims of the [n] operands [a], collecting between runs. */
static void release_runs(int n, const nx_array *a) {
  for (int at = 0; at < n; at = run_end(at, n)) {
    if (at > 0) nx_let_collect();
    for (int k = at; k < run_end(at, n); k++) rig_buffer_release(a[k].buffer);
  }
}

/* Links the buffers of [a][at, end) as local roots on top of [*roots]. */
static void push_roots(nx_array *a, int at, int end,
                       struct caml__roots_block **roots) {
  struct caml__roots_block *top = *roots;
  for (int k = at; k < end; k++) {
    a[k].roots.next = top;
    a[k].roots.ntables = 1;
    a[k].roots.nitems = 1;
    a[k].roots.tables[0] = &a[k].buffer;
    top = &a[k].roots;
  }
  *roots = top;
}

/* Reads, in runs, the [n] operands [operand] gives into [in], [out] and
   [w]: the dtype checked, the layout, the byte range and the buffer, which
   it links as a root through out[0].local. Answers NX_OK, or NX_DTYPE or
   [operand]'s refusal with no root linked. */
static int read_runs(int n, int (*operand)(void *, int, nx_operand *),
                     void *ctx, nx_operand *in, nx_array *out, where *w) {
  out[0].local = &CAML_LOCAL_ROOTS;
  for (int at = 0; at < n; at = run_end(at, n)) {
    if (at > 0) nx_let_collect();
    for (int k = at; k < run_end(at, n); k++) {
      int e = operand(ctx, k, &in[k]);
      nx_array *a = &out[k];
      value v = in[k].array;
      if (!e && (a->dtype = nx_array_dtype(v)) != in[k].dtype) e = NX_DTYPE;
      if (e) {
        if (at > 0) unlink_roots(at, out);
        return e;
      }
      a->bits = nx_dtype_row_of(a->dtype).bits;
      read_layout(Field(v, ARRAY_LAYOUT), a);
      a->buffer = Field(v, ARRAY_BUFFER);
      a->alias = 0;
      value l = Field(v, ARRAY_LAYOUT);
      w[k].lo = w[k].hi = 0;
      if (!(a->flags & NX_EMPTY)) {
        w[k].lo = Long_val(Field(l, NX_LAYOUT_LO)) * a->bits / 8;
        w[k].hi = (Long_val(Field(l, NX_LAYOUT_HI)) * a->bits + 7) / 8;
      }
    }
    push_roots(out, at, run_end(at, n), out[0].local);
  }
  return NX_OK;
}

/* Claims, in runs, the [n] operands read_runs read, each one's wait and
   host address recorded: NX_OK, [*waits] set if one waits and [*lost] to
   the first a lost device holds, if one does; or a refusal with none
   claimed and every root unlinked. */
static int claim_runs(int n, const nx_operand *in, nx_array *out, int *waits,
                      int *lost) {
  value buffer[NX_MAX_OPERANDS];
  enum rig_access access[NX_MAX_OPERANDS];
  enum rig_claim answer[NX_MAX_OPERANDS];
  for (int at = 0; at < n; at = run_end(at, n)) {
    int m = run_end(at, n) - at;
    nx_let_collect();
    for (int j = 0; j < m; j++) {
      buffer[j] = out[at + j].buffer;
      access[j] = in[at + j].written ? RIG_READ_WRITE : RIG_READ;
    }
    int refused = rig_buffers_claim(m, buffer, access, answer);
    if (refused >= 0) {
      release_runs(at, out);
      unlink_roots(n, out);
      return claim_code(answer[refused]);
    }
    for (int j = 0; j < m; j++) {
      nx_array *a = &out[at + j];
      a->wait = answer[j];
      *waits |= a->wait;
      if (answer[j] == RIG_LOST && *lost > at + j) *lost = at + j;
      a->base = a->flags & NX_EMPTY ? NULL : rig_buffer_host(a->buffer);
    }
  }
  return NX_OK;
}

/* The checks of claimed operands that read C data alone, as nx_read
   orders them: a written operand distinct, a [host] one's memory the
   host's. */
static int faults(int n, const nx_operand *in, const nx_array *out,
                  int host) {
  for (int k = 0; k < n; k++) {
    if (in[k].written && !(out[k].flags & NX_DISTINCT)) return NX_NOT_DISTINCT;
    if (host && !(out[k].flags & NX_EMPTY) && out[k].base == NULL)
      return NX_NOT_HOST;
  }
  return NX_OK;
}

/* A written operand's bytes, by host address. */
typedef struct {
  intptr_t lo, hi;
  int k;
} span;

static int by_lo(const void *x, const void *y) {
  intptr_t a = ((const span *)x)->lo, b = ((const span *)y)->lo;
  return (a > b) - (a < b);
}

/* nx_read's overlap check of [n] claimed host operands, by address, in
   O(n log n) for [n] of any size: [s] holds n spans. The written spans are
   sorted; two that meet overlap, and a read operand meets at most one,
   which it is identical to. Reads C data alone. */
static int host_overlaps(int n, const nx_operand *in, nx_array *out,
                         const where *w, span *s) {
  int ns = 0;
  for (int k = 0; k < n; k++) {
    if (!in[k].written || w[k].lo == w[k].hi) continue;
    intptr_t b = (intptr_t)out[k].base;
    s[ns++] = (span){b + w[k].lo, b + w[k].hi, k};
  }
  qsort(s, (size_t)ns, sizeof *s, by_lo);
  for (int i = 1; i < ns; i++)
    if (s[i].lo < s[i - 1].hi) return NX_OVERLAP;
  for (int j = 0; j < n; j++) {
    if (in[j].written || w[j].lo == w[j].hi) continue;
    intptr_t lo = (intptr_t)out[j].base + w[j].lo;
    intptr_t hi = (intptr_t)out[j].base + w[j].hi;
    /* The first written span past [lo]: the spans are disjoint, so sorted
       by their ends too. */
    int a = 0, b = ns;
    while (a < b) {
      int c = a + (b - a) / 2;
      if (s[c].hi <= lo) a = c + 1;
      else b = c;
    }
    if (a == ns || s[a].lo >= hi) continue;
    if ((a + 1 < ns && s[a + 1].lo < hi) || !identical(&out[s[a].k], &out[j]))
      return NX_OVERLAP;
    out[j].alias = 1;
  }
  return NX_OK;
}

/* The overlap check of [n] claimed operands wherever their memory lies,
   held, as rig compares memory off the host: it collects every
   NX_MAX_OPERANDS steps of its loops, compared or skipped. */
static int collecting_overlaps(int n, const nx_operand *in, nx_array *out,
                               const where *w) {
  int64_t steps = 0;
  for (int k = 0; k < n; k++) {
    if (++steps % NX_MAX_OPERANDS == 0) nx_let_collect();
    if (!in[k].written || w[k].lo == w[k].hi) continue;
    for (int j = 0; j < n; j++) {
      if (++steps % NX_MAX_OPERANDS == 0) nx_let_collect();
      if (j == k || out[j].alias || w[j].lo == w[j].hi) continue;
      if (!meet(&out[k], &w[k], &out[j], &w[j])) continue;
      if (in[j].written || !identical(&out[k], &out[j])) return NX_OVERLAP;
      out[j].alias = 1;
    }
  }
  return NX_OK;
}

/* The exception [r] holds. OCaml 5.5's public caml_result interface has
   no accessor for it that does not raise it, and it must stay rooted
   while the claims are released in runs: it reads the field, so that a
   change to the field fails the build. */
static value exception_of(caml_result r) { return r.data; }

/* Waits under the claims, in runs, for the device work of the [n]
   operands [out] read for [in]. If a wait raises, it releases every claim
   in runs, holding the exception as a root, unlinks the roots, frees
   [scratch] and raises. */
static void waits_runs(int n, const nx_operand *in, nx_array *out,
                       void *scratch) {
  CAMLparam0();
  CAMLlocal1(exn);
  for (int at = 0; at < n; at = run_end(at, n)) {
    if (at > 0) nx_let_collect();
    for (int k = at; k < run_end(at, n); k++) {
      if (!out[k].wait) continue;
      caml_result r = rig_buffer_wait(
          out[k].buffer, in[k].written ? RIG_READ_WRITE : RIG_READ);
      if (!caml_result_is_exception(r)) continue;
      exn = exception_of(r);
      release_runs(n, out);
      /* The frame goes before the roots under it: no collection runs
         until the raise. */
      CAMLdrop;
      unlink_roots(n, out);
      free(scratch);
      caml_raise(exn);
    }
  }
  CAMLreturn0;
}

int nx_read_each(int n, int (*operand)(void *ctx, int k, nx_operand *in),
                 void *ctx, nx_array *out) {
  if (n <= 0) return NX_OK;
  /* The operands, their byte ranges and the spans of the overlap check, on
     the heap, allocated before any claim. */
  size_t one = sizeof(nx_operand) + sizeof(where) + sizeof(span);
  uint8_t *b = nx_alloc_released((size_t)n * one);
  if (b == NULL) caml_raise_out_of_memory();
  nx_operand *in = (nx_operand *)b;
  where *w = (where *)(in + n);
  span *s = (span *)(w + n);
  int waiting = 0, lost = n;
  int e = read_runs(n, operand, ctx, in, out, w);
  if (e) {
    nx_free_released(b);
    return e;
  }
  if ((e = claim_runs(n, in, out, &waiting, &lost))) {
    nx_free_released(b);
    return e;
  }
  caml_enter_blocking_section_no_pending();
  e = faults(n, in, out, 1);
  if (!e) e = host_overlaps(n, in, out, w, s);
  caml_leave_blocking_section();
  if (e) {
    release_runs(n, out);
    unlink_roots(n, out);
    nx_free_released(b);
    return e;
  }
  if (waiting) waits_runs(n, in, out, b);
  nx_free_released(b);
  return NX_OK;
}

int nx_read(int n, const nx_operand *in, nx_array *out) {
  if (n <= 0) return NX_OK;
  int e = admit(n, in, out, 1);
  if (e) return e;
  int waits = 0;
  for (int k = 0; k < n; k++) waits |= out[k].wait;
  /* Each buffer becomes a local root, so that a kernel that allocates or
     releases the domain lock keeps it reachable and finds it again in
     nx_done. They are chained here and linked at once: where the runtime
     has no thread-local variables to share (macOS), each access to the
     domain's state is a function call. */
  struct caml__roots_block **roots = &CAML_LOCAL_ROOTS, *top = *roots;
  /* nx_done unlinks through the same pointer: one lookup a call. */
  out[0].local = roots;
  for (int k = 0; k < n; k++) {
    out[k].roots.next = top;
    out[k].roots.ntables = 1;
    out[k].roots.nitems = 1;
    out[k].roots.tables[0] = &out[k].buffer;
    top = &out[k].roots;
  }
  *roots = top;
  if (!waits) return NX_OK;
  /* The wait runs OCaml code, which may move each buffer: it is read from
     its root. */
  for (int k = 0; k < n; k++) {
    if (!out[k].wait) continue;
    caml_result r = rig_buffer_wait(
        out[k].buffer, in[k].written ? RIG_READ_WRITE : RIG_READ);
    if (caml_result_is_exception(r)) {
      /* The exception is no root: the claims end without a release of
         the lock, which nx_done makes past NX_MAX_OPERANDS. */
      for (int j = 0; j < n; j++) rig_buffer_release(out[j].buffer);
      unlink_roots(n, out);
      (void)caml_get_value_or_raise(r);
    }
  }
  return NX_OK;
}

void nx_done(int n, nx_array *a) {
  if (n <= 0) return;
  if (n <= NX_MAX_OPERANDS)
    for (int k = 0; k < n; k++) rig_buffer_release(a[k].buffer);
  else release_runs(n, a);
  unlink_roots(n, a);
}

/* The OCaml door (Nx_array.door) admits through admit, in C, so that no
   OCaml code, and so no asynchronous exception, runs between two claims.
   [written] and [read] are OCaml arrays of Nx_array.any, which is unboxed:
   an any is its array's block. */

static value nth_array(value written, value read, mlsize_t k) {
  mlsize_t nw = Wosize_val(written);
  return k < nw ? Field(written, k) : Field(read, k - nw);
}

static void release_first(value written, value read, mlsize_t n) {
  for (mlsize_t k = 0; k < n; k++)
    rig_buffer_release(Field(nth_array(written, read, k), ARRAY_BUFFER));
}

/* Admits every array of [written] for writing and of [read] for reading, of
   any dtype and wherever its memory lies, and waits for nothing: NX_OK with
   every array claimed, or a refusal's code with none. A read array
   identical to a written one is claimed too: claims are counted, so two of
   one memory do not conflict. If an array is a lost device's memory, or
   follows a lost device's work, it raises Rig.Lost, as the wait for its
   access does at once, holding no claim. */
/* The door's arrays at the roots [written] and [read], as read_runs reads
   them. */
typedef struct {
  const value *written, *read;
} door_arrays;

static int door_operand(void *ctx, int k, nx_operand *in) {
  const door_arrays *d = ctx;
  value v = nth_array(*d->written, *d->read, (mlsize_t)k);
  *in = (nx_operand){v, nx_array_dtype(v),
                     (mlsize_t)k < Wosize_val(*d->written)};
  return NX_OK;
}

/* Releases the claims of the first [n] arrays of the roots [written] and
   [read], collecting between runs. */
static void release_first_runs(const value *written, const value *read,
                               mlsize_t n) {
  for (mlsize_t at = 0; at < n; at += NX_MAX_OPERANDS) {
    if (at > 0) nx_let_collect();
    mlsize_t end = n - at < NX_MAX_OPERANDS ? n : at + NX_MAX_OPERANDS;
    for (mlsize_t k = at; k < end; k++)
      rig_buffer_release(Field(nth_array(*written, *read, k), ARRAY_BUFFER));
  }
}

/* nx_array_admit past NX_MAX_OPERANDS arrays, in runs (Many operands). */
static value admit_many(value written, value read) {
  CAMLparam2(written, read);
  CAMLlocal1(exn);
  int n = (int)(Wosize_val(written) + Wosize_val(read));
  uint8_t *b = nx_alloc_released(
      (size_t)n * (sizeof(nx_operand) + sizeof(nx_array) + sizeof(where)));
  if (b == NULL) caml_raise_out_of_memory();
  nx_array *out = (nx_array *)b;
  where *w = (where *)(out + n);
  nx_operand *in = (nx_operand *)(w + n);
  door_arrays d = {&written, &read};
  int waiting = 0, lost = n;
  int e = read_runs(n, door_operand, &d, in, out, w);
  if (!e && !(e = claim_runs(n, in, out, &waiting, &lost))) {
    caml_enter_blocking_section_no_pending();
    e = faults(n, in, out, 0);
    caml_leave_blocking_section();
    if (!e) e = collecting_overlaps(n, in, out, w);
    if (e) release_runs(n, out);
    /* The claims stay until release_all; the roots end here. */
    unlink_roots(n, out);
  }
  nx_free_released(b);
  if (e || lost == n) CAMLreturn(Val_int(e));
  value a = nth_array(written, read, (mlsize_t)lost);
  caml_result r = rig_buffer_wait(Field(a, ARRAY_BUFFER),
                                  (mlsize_t)lost < Wosize_val(written)
                                      ? RIG_READ_WRITE
                                      : RIG_READ);
  if (caml_result_is_exception(r)) {
    exn = exception_of(r);
    release_first_runs(&written, &read, (mlsize_t)n);
    caml_raise(exn);
  }
  CAMLreturn(Val_int(NX_OK));
}

value nx_array_admit(value written, value read) {
  if (Wosize_val(written) + Wosize_val(read) > NX_MAX_OPERANDS)
    return admit_many(written, read);
  CAMLparam2(written, read);
  mlsize_t n = Wosize_val(written) + Wosize_val(read);
  /* The operands, their descriptors and their byte ranges, on the stack:
     admit_many takes more than NX_MAX_OPERANDS. */
  nx_operand in[NX_MAX_OPERANDS];
  nx_array out[NX_MAX_OPERANDS];
  where w[NX_MAX_OPERANDS];
  for (mlsize_t k = 0; k < n; k++) {
    value v = nth_array(written, read, k);
    in[k] = (nx_operand){v, nx_array_dtype(v), k < Wosize_val(written)};
  }
  int e = checks((int)n, in, out, w, 0);
  mlsize_t lost = n;
  for (mlsize_t k = 0; !e && k < n && lost == n; k++)
    if (out[k].wait == RIG_LOST) lost = k;
  if (e || lost == n) CAMLreturn(Val_int(e));
  value a = nth_array(written, read, lost);
  caml_result r = rig_buffer_wait(Field(a, ARRAY_BUFFER),
                                  lost < Wosize_val(written) ? RIG_READ_WRITE
                                                             : RIG_READ);
  if (caml_result_is_exception(r)) {
    release_first(written, read, n);
    (void)caml_get_value_or_raise(r);
  }
  CAMLreturn(Val_int(NX_OK));
}

value nx_array_release_all(value written, value read) {
  release_first(written, read, Wosize_val(written) + Wosize_val(read));
  return Val_unit;
}

/* nx_array_release_all past NX_MAX_OPERANDS arrays, collecting between
   runs. */
value nx_array_release_many(value written, value read) {
  CAMLparam2(written, read);
  release_first_runs(&written, &read, Wosize_val(written) + Wosize_val(read));
  CAMLreturn(Val_unit);
}

/* Coalescing */

/* The coalescer, written once: it reads axis i's extent at [ein][i] and
   operand k's step at [sin][k][i], and writes the merged loop into
   [extent] and [step], which may be [ein] and [sin] themselves. Axis i
   lands at [out] or merges into out - 1, both at most i: the loop
   rewrites only what it has read. Inlined at both callers: through a call,
   nx_coalesce read the descriptors into its loop first, and a door's loop
   over strided operands cost about 7 ns more on the M1 Max. A merged
   extent is a product of one layout's extents: at most NX_MAX_NUMEL. */
static inline __attribute__((always_inline)) int coalesce(
    int n, int rank, const int64_t *ein, const int64_t *const *sin,
    int64_t *extent, int64_t (*step)[NX_MAX_RANK]) {
  int out = 0;
  for (int i = 0; i < rank; i++) {
    const int64_t d = ein[i];
    if (d == 0) {
      extent[0] = 0;
      for (int k = 0; k < n; k++) step[k][0] = 0;
      return 1;
    }
    if (d == 1) continue;
    /* The axis joins the previous one if every operand's previous stride
       is this stride times this extent: the two axes are one run. */
    int joins = out > 0;
    for (int k = 0; k < n && joins; k++)
      joins = step[k][out - 1] == sin[k][i] * d;
    if (joins) {
      extent[out - 1] *= d;
      for (int k = 0; k < n; k++) step[k][out - 1] = sin[k][i];
    } else {
      extent[out] = d;
      for (int k = 0; k < n; k++) step[k][out] = sin[k][i];
      out++;
    }
  }
  if (out == 0) {
    extent[0] = 1;
    for (int k = 0; k < n; k++) step[k][0] = 0;
    out = 1;
  }
  return out;
}

int nx_coalesce(int n, const nx_array *a, nx_loop *l) {
  if (n < 1 || n > NX_MAX_OPERANDS) return NX_SHAPE;
  int r = a[0].rank;
  for (int k = 1; k < n; k++) {
    if (a[k].rank != r) return NX_SHAPE;
    for (int i = 0; i < r; i++)
      if (a[k].dim[i] != a[0].dim[i]) return NX_SHAPE;
  }
  for (int k = 0; k < n; k++) l->first[k] = a[k].offset;
  /* Operands all in C order merge into one run: the loop is their number
     of elements, in steps of one. */
  int contiguous = 1;
  for (int k = 0; k < n; k++) contiguous &= (a[k].flags & NX_CONTIGUOUS) != 0;
  if (contiguous) {
    int64_t e = 1;
    for (int i = 0; i < r; i++) e *= a[0].dim[i];
    l->rank = 1;
    l->extent[0] = e;
    for (int k = 0; k < n; k++) l->step[k][0] = e > 1;
    return NX_OK;
  }
  const int64_t *steps[NX_MAX_OPERANDS];
  for (int k = 0; k < n; k++) steps[k] = a[k].dim + r;
  l->rank = coalesce(n, r, a[0].dim, steps, l->extent, l->step);
  return NX_OK;
}

int nx_coalesce_dims(int n, int rank, int64_t *extent,
                     int64_t (*step)[NX_MAX_RANK]) {
  const int64_t *steps[NX_MAX_OPERANDS];
  for (int k = 0; k < n; k++) steps[k] = step[k];
  return coalesce(n, rank, extent, steps, extent, step);
}

/* Calls [run] on each run of the loop [l] over [n] operands: the positions
   [at] of each operand's first element, and the run's length; the run's
   steps are the loop's innermost. It is inlined at each call, where [run]
   is known, so that [run] is inlined into the odometer: clang at -O3
   otherwise calls it through the pointer, once per run. */
static inline __attribute__((always_inline)) void walk(
    int n, const nx_loop *l, void *ctx,
    void (*run)(void *ctx, const int64_t *at, int64_t len)) {
  int r = l->rank;
  int64_t len = l->extent[r - 1];
  if (len == 0) return;
  int64_t idx[NX_MAX_RANK] = {0};
  int64_t at[NX_MAX_OPERANDS];
  for (int k = 0; k < n; k++) at[k] = l->first[k];
  for (;;) {
    run(ctx, at, len);
    /* The odometer over the outer axes. */
    int i = r - 2;
    for (; i >= 0; i--) {
      for (int k = 0; k < n; k++) at[k] += l->step[k][i];
      if (++idx[i] < l->extent[i]) break;
      for (int k = 0; k < n; k++) at[k] -= l->step[k][i] * l->extent[i];
      idx[i] = 0;
    }
    if (i < 0) return;
  }
}

/* Layout.coalesce: coalesces the layouts [ls] into [out], an int array of
   1 + rank + n·(1 + rank) words: the rank, the extents, then per operand
   its first position and steps. Answers a code. */
value nx_array_coalesce(value ls, value out) {
  int n = (int)Wosize_val(ls);
  nx_array a[NX_MAX_OPERANDS];
  nx_loop l;
  if (n < 1 || n > NX_MAX_OPERANDS) return Val_int(NX_SHAPE);
  for (int k = 0; k < n; k++) read_layout(Field(ls, k), &a[k]);
  int e = nx_coalesce(n, a, &l);
  if (e) return Val_int(e);
  int r = l.rank;
  /* [out] is an int array: immediates need no write barrier. */
  Field(out, 0) = Val_long(r);
  for (int i = 0; i < r; i++) Field(out, 1 + i) = Val_long(l.extent[i]);
  for (int k = 0; k < n; k++) {
    mlsize_t at = 1 + r + k * (1 + r);
    Field(out, at) = Val_long(l.first[k]);
    for (int i = 0; i < r; i++)
      Field(out, at + 1 + i) = Val_long(l.step[k][i]);
  }
  return Val_int(NX_OK);
}

/* Elements

   Loads and stores of one element at position [p] from [base]. A float
   dtype's element is a double; an integer's, a boolean's or a narrow
   float's bits are an int64_t, sign-extended for signed integers. A complex
   element is two components, [part] 0 the real one. */

static double load_float(const uint8_t *base, int dt, int64_t p, int part) {
  switch (dt) {
    case NX_FLOAT64: {
      double x;
      memcpy(&x, base + 8 * p, 8);
      return x;
    }
    case NX_FLOAT32: {
      float x;
      memcpy(&x, base + 4 * p, 4);
      return x;
    }
    case NX_FLOAT16:
    case NX_BFLOAT16: {
      uint16_t c;
      memcpy(&c, base + 2 * p, 2);
      return nx_bits_to_float(dt, c);
    }
    case NX_FLOAT8_E4M3FN:
    case NX_FLOAT8_E5M2: return nx_bits_to_float(dt, base[p]);
    case NX_FLOAT4_E2M1FN: return nx_bits_to_float(dt, nx_sub_load(base, 4, p));
    case NX_COMPLEX128: return load_float(base, NX_FLOAT64, 2 * p + part, 0);
    default: return load_float(base, NX_FLOAT32, 2 * p + part, 0);
  }
}

static int64_t load_int(const uint8_t *base, int dt, int64_t p) {
  switch (dt) {
    case NX_INT64:
    case NX_UINT64: {
      int64_t x;
      memcpy(&x, base + 8 * p, 8);
      return x;
    }
    case NX_INT32: {
      int32_t x;
      memcpy(&x, base + 4 * p, 4);
      return x;
    }
    case NX_UINT32: {
      uint32_t x;
      memcpy(&x, base + 4 * p, 4);
      return x;
    }
    case NX_INT16: {
      int16_t x;
      memcpy(&x, base + 2 * p, 2);
      return x;
    }
    case NX_UINT16: {
      uint16_t x;
      memcpy(&x, base + 2 * p, 2);
      return x;
    }
    case NX_INT8: return (int8_t)base[p];
    case NX_UINT8: return base[p];
    case NX_INT4: return ((int32_t)(nx_sub_load(base, 4, p) << 28)) >> 28;
    case NX_UINT4: return nx_sub_load(base, 4, p);
    case NX_BOOL: return base[p] != 0;
    default: return nx_sub_load(base, 1, p); /* bit */
  }
}

static void store_float(uint8_t *base, int dt, int64_t p, int part,
                        double x) {
  switch (dt) {
    case NX_FLOAT64: memcpy(base + 8 * p, &x, 8); return;
    case NX_FLOAT32: {
      float f = (float)x;
      memcpy(base + 4 * p, &f, 4);
      return;
    }
    case NX_FLOAT16:
    case NX_BFLOAT16: {
      uint16_t c = (uint16_t)nx_double_to_bits(dt, x);
      memcpy(base + 2 * p, &c, 2);
      return;
    }
    case NX_FLOAT8_E4M3FN:
    case NX_FLOAT8_E5M2: base[p] = (uint8_t)nx_double_to_bits(dt, x); return;
    case NX_FLOAT4_E2M1FN:
      nx_sub_store(base, 4, p, (uint32_t)nx_double_to_bits(dt, x));
      return;
    case NX_COMPLEX128:
      store_float(base, NX_FLOAT64, 2 * p + part, 0, x);
      return;
    default: store_float(base, NX_FLOAT32, 2 * p + part, 0, x); return;
  }
}

static void store_int(uint8_t *base, int dt, int64_t p, int64_t x) {
  switch (dt) {
    case NX_INT64:
    case NX_UINT64: memcpy(base + 8 * p, &x, 8); return;
    case NX_INT32:
    case NX_UINT32: {
      uint32_t c = (uint32_t)x;
      memcpy(base + 4 * p, &c, 4);
      return;
    }
    case NX_INT16:
    case NX_UINT16: {
      uint16_t c = (uint16_t)x;
      memcpy(base + 2 * p, &c, 2);
      return;
    }
    case NX_INT8:
    case NX_UINT8:
    case NX_BOOL: base[p] = (uint8_t)x; return;
    case NX_INT4:
    case NX_UINT4: nx_sub_store(base, 4, p, (uint32_t)x); return;
    default: nx_sub_store(base, 1, p, (uint32_t)x); return; /* bit */
  }
}

intnat nx_array_host(value b) {
  void *p = rig_buffer_host(b);
  return p == NULL ? -1 : (intnat)(intptr_t)p;
}

value nx_array_host_byte(value b) { return Val_long(nx_array_host(b)); }

/* The byte [i] of the host buffer [b], loaded atomically: other threads may
   store elements of a sub-byte array in it. */
intnat nx_array_load_byte(value b, intnat i) {
  const uint8_t *p = rig_buffer_host(b);
  return __atomic_load_n(p + i, __ATOMIC_RELAXED);
}

value nx_array_load_byte_byte(value b, value i) {
  return Val_long(nx_array_load_byte(b, Long_val(i)));
}

/* get: [v]'s element at position [vp], which its layout reaches, read
   through the door and boxed as its dtype's OCaml values are. A refusal
   raises Nx_array's Refused with its code, holding no claim. The element
   is loaded under the claim and boxed after it, so that an allocation that
   raises holds no claim. */
value nx_array_get(value v, value vp) {
  CAMLparam1(v);
  CAMLlocal1(x);
  nx_operand in = {v, nx_array_dtype(v), 0};
  nx_array a;
  int e = nx_read(1, &in, &a);
  if (e) {
    static const value *refused = NULL;
    if (refused == NULL) refused = caml_named_value("Nx_array.Refused");
    caml_raise_with_arg(*refused, Val_int(e));
  }
  int dt = a.dtype;
  int64_t p = Long_val(vp), i = 0;
  double re = 0, im = 0;
  int kind = nx_dtype_row_of(dt).kind;
  if (kind == NX_KIND_FLOAT || kind == NX_KIND_COMPLEX) {
    re = load_float(a.base, dt, p, 0);
    if (kind == NX_KIND_COMPLEX) im = load_float(a.base, dt, p, 1);
  } else
    i = load_int(a.base, dt, p);
  nx_done(1, &a);
  switch (kind) {
    case NX_KIND_FLOAT: x = caml_copy_double(re); break;
    case NX_KIND_COMPLEX:
      x = caml_alloc_small(2 * Double_wosize, Double_array_tag);
      Store_double_flat_field(x, 0, re);
      Store_double_flat_field(x, 1, im);
      break;
    default:
      if (dt == NX_INT64 || dt == NX_UINT64) x = caml_copy_int64(i);
      else if (dt == NX_INT32 || dt == NX_UINT32)
        x = caml_copy_int32((int32_t)i);
      else if (dt == NX_BOOL || dt == NX_BIT) x = Val_bool(i != 0);
      else x = Val_long(i);
  }
  CAMLreturn(x);
}

/* set: stores [x] at [v]'s position [vp], which its layout reaches, through
   the door, and answers its code. [x] is read before the door, whose wait
   may move it. */
value nx_array_set(value v, value vp, value x) {
  CAMLparam1(v);
  int dt = nx_array_dtype(v);
  int kind = nx_dtype_row_of(dt).kind;
  double re = 0, im = 0;
  int64_t i = 0;
  if (kind == NX_KIND_FLOAT) re = Double_val(x);
  else if (kind == NX_KIND_COMPLEX) {
    re = Double_flat_field(x, 0);
    im = Double_flat_field(x, 1);
  } else if (dt == NX_INT64 || dt == NX_UINT64) i = Int64_val(x);
  else if (dt == NX_INT32 || dt == NX_UINT32) i = Int32_val(x);
  else if (dt == NX_BOOL || dt == NX_BIT) i = Bool_val(x);
  else i = Long_val(x);
  nx_operand in = {v, dt, 1};
  nx_array a;
  int e = nx_read(1, &in, &a);
  if (e) CAMLreturn(Val_int(e));
  int64_t p = Long_val(vp);
  if (kind == NX_KIND_FLOAT || kind == NX_KIND_COMPLEX) {
    store_float(a.base, dt, p, 0, re);
    if (kind == NX_KIND_COMPLEX) store_float(a.base, dt, p, 1, im);
  } else
    store_int(a.base, dt, p, i);
  nx_done(1, &a);
  CAMLreturn(Val_int(NX_OK));
}

/* Bulk access */

enum { TO_FLAT, TO_IMMEDIATE, TO_BOXED };

/* How a dtype's values lie in an OCaml array: flat doubles, immediates, or
   boxes. */
static int representation(int dt) {
  switch (nx_dtype_row_of(dt).kind) {
    case NX_KIND_FLOAT: return TO_FLAT;
    case NX_KIND_COMPLEX: return TO_BOXED;
    default:
      return nx_dtype_row_of(dt).bits >= 32 ? TO_BOXED : TO_IMMEDIATE;
  }
}

/* float16 runs from and to doubles. arm64 widens float16 with FCVT and
   FCVTL, exactly, and narrows a double to __fp16 in one rounding, as the
   codecs do, a vector at a time; elsewhere the runs are the codecs'
   loops. */
#if defined(__aarch64__)
typedef __fp16 __attribute__((may_alias)) fp16;

static void f16_to_double(const uint16_t *src, double *dst, size_t n) {
  const fp16 *h = (const fp16 *)src;
  for (size_t i = 0; i < n; i++) dst[i] = (double)h[i];
}

static void double_to_f16(const double *src, uint16_t *dst, size_t n) {
  fp16 *h = (fp16 *)dst;
  for (size_t i = 0; i < n; i++) h[i] = (__fp16)src[i];
}
#else
static void f16_to_double(const uint16_t *src, double *dst, size_t n) {
  for (size_t i = 0; i < n; i++) dst[i] = nx_f16_to_float(src[i]);
}

static void double_to_f16(const double *src, uint16_t *dst, size_t n) {
  for (size_t i = 0; i < n; i++) dst[i] = nx_double_to_f16(src[i]);
}
#endif

typedef struct {
  const nx_array *a;
  const nx_loop *l;
  value out;
  int64_t k; /* the next element of [out] */
} to_ctx;

/* Writes a run of flat or immediate elements: no allocation. */
static void to_run(void *ctx, const int64_t *at, int64_t len) {
  to_ctx *c = ctx;
  const nx_array *a = c->a;
  int dt = a->dtype;
  int64_t step = c->l->step[0][c->l->rank - 1];
  int64_t p = at[0];
  if (representation(dt) == TO_IMMEDIATE) {
    /* Immediates need no write barrier. */
    for (int64_t j = 0; j < len; j++, p += step)
      Field(c->out, c->k++) = Val_long(load_int(a->base, dt, p));
    return;
  }
  double *dst = (double *)Op_val(c->out) + c->k;
  c->k += len;
  /* A loop per format, the format's decoder inlined in it. */
#define DECODE(T, decode)                                                 \
  do {                                                                    \
    const T *src = (const T *)a->base + p;                                \
    for (int64_t j = 0; j < len; j++) dst[j] = decode(src[j * step]);     \
  } while (0)
  switch (dt) {
    case NX_FLOAT64: DECODE(double, (double)); return;
    case NX_FLOAT32: DECODE(float, (double)); return;
    case NX_FLOAT16:
      if (step == 1)
        f16_to_double((const uint16_t *)a->base + p, dst, (size_t)len);
      else DECODE(uint16_t, nx_f16_to_float);
      return;
    case NX_BFLOAT16: DECODE(uint16_t, nx_bf16_to_float); return;
    case NX_FLOAT8_E4M3FN: DECODE(uint8_t, nx_e4m3fn_to_float); return;
    case NX_FLOAT8_E5M2: DECODE(uint8_t, nx_e5m2_to_float); return;
    default: /* float4: elements share bytes */
      for (int64_t j = 0; j < len; j++, p += step)
        dst[j] = load_float(a->base, dt, p, 0);
      return;
  }
#undef DECODE
}

/* to_array: reads [v]'s elements, in C order of indices, into [out], an
   OCaml array of their number, flat floats or immediates. It allocates
   nothing past the door's wait; elements that box go through
   nx_array_to_bigarray. */
value nx_array_to_array(value v, value out) {
  CAMLparam1(out);
  nx_operand in = {v, nx_array_dtype(v), 0};
  nx_array a;
  nx_loop l;
  if (representation(in.dtype) == TO_BOXED) CAMLreturn(Val_int(NX_DTYPE));
  int e = nx_read(1, &in, &a);
  if (e) CAMLreturn(Val_int(e));
  if (!(e = nx_coalesce(1, &a, &l))) {
    to_ctx c = {&a, &l, out, 0};
    walk(1, &l, &c, to_run);
  }
  nx_done(1, &a);
  CAMLreturn(Val_int(e));
}

typedef struct {
  const nx_array *a;
  const nx_loop *l;
  value values;
  int64_t k;
} of_ctx;

static void of_run(void *ctx, const int64_t *at, int64_t len) {
  of_ctx *c = ctx;
  const nx_array *a = c->a;
  int dt = a->dtype;
  int64_t step = c->l->step[0][c->l->rank - 1];
  int64_t p = at[0], k = c->k;
  c->k += len;
  switch (representation(dt)) {
    case TO_FLAT: {
      const double *src = (const double *)Op_val(c->values) + k;
      /* A loop per format, the format's encoder inlined in it. */
#define ENCODE(T, encode)                                                 \
  do {                                                                    \
    T *dst = (T *)a->base + p;                                            \
    for (int64_t j = 0; j < len; j++) dst[j * step] = encode(src[j]);     \
  } while (0)
      switch (dt) {
        case NX_FLOAT64: ENCODE(double, (double)); return;
        case NX_FLOAT32: ENCODE(float, (float)); return;
        case NX_FLOAT16:
          if (step == 1)
            double_to_f16(src, (uint16_t *)a->base + p, (size_t)len);
          else ENCODE(uint16_t, nx_double_to_f16);
          return;
        case NX_BFLOAT16: ENCODE(uint16_t, nx_double_to_bf16); return;
        case NX_FLOAT8_E4M3FN: ENCODE(uint8_t, nx_double_to_e4m3fn); return;
        case NX_FLOAT8_E5M2: ENCODE(uint8_t, nx_double_to_e5m2); return;
        default: /* float4: elements share bytes */
          for (int64_t j = 0; j < len; j++, p += step)
            store_float(a->base, dt, p, 0, src[j]);
          return;
      }
#undef ENCODE
    }
    case TO_IMMEDIATE:
      for (int64_t j = 0; j < len; j++, p += step)
        store_int(a->base, dt, p, Long_val(Field(c->values, k + j)));
      return;
    default:
      for (int64_t j = 0; j < len; j++, p += step) {
        value x = Field(c->values, k + j);
        if (dt == NX_INT32 || dt == NX_UINT32)
          store_int(a->base, dt, p, Int32_val(x));
        else if (dt == NX_INT64 || dt == NX_UINT64)
          store_int(a->base, dt, p, Int64_val(x));
        else {
          store_float(a->base, dt, p, 0, Double_flat_field(x, 0));
          store_float(a->base, dt, p, 1, Double_flat_field(x, 1));
        }
      }
      return;
  }
}

/* of_array: writes [values], an OCaml array of [v]'s number of elements, in
   C order of indices. It allocates nothing past the door's wait. */
value nx_array_of_array(value v, value values) {
  CAMLparam1(values);
  nx_operand in = {v, nx_array_dtype(v), 1};
  nx_array a;
  nx_loop l;
  int e = nx_read(1, &in, &a);
  if (e) CAMLreturn(Val_int(e));
  if (!(e = nx_coalesce(1, &a, &l))) {
    of_ctx c = {&a, &l, values, 0};
    walk(1, &l, &c, of_run);
  }
  nx_done(1, &a);
  CAMLreturn(Val_int(e));
}

/* copy: gathers [src]'s elements into [dst], a fresh contiguous array of
   its dtype and shape, bits for bits: one walk over coalesced runs, each
   run a memcpy where both operands' elements are adjacent, else element by
   element. The host's strided copy, tiled and threaded, is nx.cpu's. */

typedef struct {
  const nx_array *a; /* dst, src */
  const nx_loop *l;
} copy_ctx;

/* Rows of a copy are read ahead: a row that starts apart from the one before,
   as a box's rows do in a wider array, is a stream the hardware prefetcher has
   not seen, and its copy would wait on memory. On the M1, 4 MiB in rows of
   1 KiB, 4 KiB apart, gather in 470 us without it and in 104 us reading two
   rows ahead, where one memcpy of the 4 MiB takes 90. Only a row's first 4 KiB
   are asked for: the prefetcher follows a longer row once it has started. A
   read ahead is a hint: past the buffer's end it loads nothing and faults
   nothing. */
#define READ_AHEAD_ROWS 2
#define READ_AHEAD_BYTES 4096
#define CACHE_LINE 64

/* Asks for the first bytes, at most [READ_AHEAD_BYTES], of the [n] at
   [base + off] in cache. The address is formed as an integer, since it may
   lie past the buffer. */
static inline __attribute__((always_inline)) void read_ahead(
    const uint8_t *base, int64_t off, int64_t n) {
  if (n > READ_AHEAD_BYTES) n = READ_AHEAD_BYTES;
  for (int64_t b = 0; b < n; b += CACHE_LINE)
    __builtin_prefetch((const void *)((uintptr_t)base + (uintptr_t)(off + b)));
}

/* Copies [n] elements of [w] bytes, the [k]th from [s + k·bs] to
   [d + k·bd]. Called with a constant [w], the copies are loads and stores. */
static inline __attribute__((always_inline)) void strided(
    uint8_t *d, int64_t bd, const uint8_t *s, int64_t bs, int64_t n,
    size_t w) {
  for (int64_t k = 0; k < n; k++, d += bd, s += bs) memcpy(d, s, w);
}

/* The [k] elements of [bits] bits from position [p] of [s], [step] apart,
   packed LSB first. */
static inline uint8_t pack(const uint8_t *s, int bits, int64_t p, int64_t step,
                           int k) {
  uint8_t x = 0;
  for (int j = 0; j < k; j++, p += step)
    x |= (uint8_t)(nx_sub_load(s, bits, p) << (j * bits));
  return x;
}

__attribute__((noinline)) void nx_sub_copy_run(uint8_t *d, int64_t pd,
                                               const uint8_t *s, int64_t ps,
                                               int64_t ss, int64_t len,
                                               int bits) {
  int per = 8 / bits;
  int head = (int)((per - pd % per) % per);
  if (head > len) head = (int)len;
  if (head > 0) {
    int at = (int)(pd % per) * bits;
    uint8_t mask = (uint8_t)(((1u << (head * bits)) - 1) << at);
    nx_sub_put(d + pd / per, mask,
               (uint8_t)(pack(s, bits, ps, ss, head) << at));
    pd += head;
    ps += head * ss;
    len -= head;
  }
  int64_t whole = len / per;
  uint8_t *db = d + pd / per;
  if (whole > 0 && ss == 1) {
    int64_t bit = ps * bits;
    int sh = (int)(bit & 7);
    const uint8_t *sb = s + (bit >> 3);
    if (sh == 0)
      memcpy(db, sb, (size_t)whole);
    else {
      /* Byte i takes the top of sb[i] and the bottom of sb[i + 1]. sb[0]
         and sb[whole] hold elements outside the run, which other threads
         may store: they are loaded atomically. */
      uint8_t first = __atomic_load_n(sb, __ATOMIC_RELAXED);
      uint8_t last = __atomic_load_n(sb + whole, __ATOMIC_RELAXED);
      if (whole == 1)
        db[0] = (uint8_t)((first >> sh) | (last << (8 - sh)));
      else {
        db[0] = (uint8_t)((first >> sh) | (sb[1] << (8 - sh)));
        int64_t i = 1;
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        /* Eight bytes at a time: the word at sb + i, shifted, and the
           bottom of the word after it. */
        if (i + 17 < whole) {
          uint64_t lo, hi;
          memcpy(&lo, sb + i, 8);
          for (; i + 17 < whole; i += 8, lo = hi) {
            memcpy(&hi, sb + i + 8, 8);
            uint64_t w = (lo >> sh) | (hi << (64 - sh));
            memcpy(db + i, &w, 8);
          }
        }
#endif
        for (; i < whole - 1; i++)
          db[i] = (uint8_t)((sb[i] >> sh) | (sb[i + 1] << (8 - sh)));
        db[whole - 1] = (uint8_t)((sb[whole - 1] >> sh) | (last << (8 - sh)));
      }
    }
  } else
    for (int64_t i = 0; i < whole; i++)
      db[i] = pack(s, bits, ps + i * per * ss, ss, per);
  pd += whole * per;
  ps += whole * per * ss;
  len -= whole * per;
  if (len > 0)
    nx_sub_put(d + pd / per, (uint8_t)((1u << (len * bits)) - 1),
             pack(s, bits, ps, ss, (int)len));
}

/* A run of the gather, the source's run [READ_AHEAD_ROWS] steps along the
   next axis out read ahead where its elements are adjacent and the runs lie
   apart: the walk's next runs are most often those. */
static void copy_run(void *ctx, const int64_t *at, int64_t len) {
  copy_ctx *c = ctx;
  int r = c->l->rank;
  const nx_array *s = &c->a[1];
  if (r > 1 && c->l->step[1][r - 1] == 1 && c->l->step[1][r - 2] != len) {
    int64_t ahead = at[1] + READ_AHEAD_ROWS * c->l->step[1][r - 2];
    read_ahead(s->base, ahead * s->bits / 8, (len * s->bits + 7) / 8);
  }
  int bits = s->bits;
  int64_t dc = c->l->step[0][r - 1], sc = c->l->step[1][r - 1];
  uint8_t *d = c->a[0].base;
  if (bits < 8) {
    if (dc == 1) {
      nx_sub_copy_run(d, at[0], s->base, at[1], sc, len, bits);
      return;
    }
    int64_t pd = at[0], ps = at[1];
    for (int64_t j = 0; j < len; j++, pd += dc, ps += sc)
      nx_sub_store(d, bits, pd, nx_sub_load(s->base, bits, ps));
    return;
  }
  int64_t w = bits / 8;
  d += at[0] * w;
  const uint8_t *src = s->base + at[1] * w;
  if (dc == 1 && sc == 1) {
    memcpy(d, src, (size_t)(len * w));
    return;
  }
  switch (w) {
    case 1: strided(d, dc, src, sc, len, 1); return;
    case 2: strided(d, 2 * dc, src, 2 * sc, len, 2); return;
    case 4: strided(d, 4 * dc, src, 4 * sc, len, 4); return;
    case 8: strided(d, 8 * dc, src, 8 * sc, len, 8); return;
    default: strided(d, 16 * dc, src, 16 * sc, len, 16); return;
  }
}

/* The gather: copies the elements of a[1] into a[0], operands of one dtype
   read through the door. Answers NX_SHAPE if their shapes differ. */
static int gather(nx_array a[2]) {
  nx_loop l;
  int e = nx_coalesce(2, a, &l);
  if (e) return e;
  copy_ctx c = {a, &l};
  walk(2, &l, &c, copy_run);
  return NX_OK;
}

value nx_array_copy(value dst, value src) {
  int dt = nx_array_dtype(src);
  nx_operand in[2] = {{dst, dt, 1}, {src, dt, 0}};
  nx_array a[2];
  int e = nx_read(2, in, a);
  if (e) return Val_int(e);
  e = gather(a);
  nx_done(2, a);
  return Val_int(e);
}

/* to_bigarray: gathers [v]'s elements, in C order of indices, into [out], a
   bigarray of their number and width, bit for bit. to_array boxes elements
   from there once the claim is released, so that an allocation that raises
   holds no claim. */
value nx_array_to_bigarray(value v, value out) {
  CAMLparam1(out);
  nx_operand in = {v, nx_array_dtype(v), 0};
  nx_array a[2];
  int e = nx_read(1, &in, &a[1]);
  if (e) CAMLreturn(Val_int(e));
  /* [out] as a C-contiguous array of [v]'s dtype and shape. */
  nx_array *d = &a[0];
  int r = a[1].rank;
  d->base = Caml_ba_data_val(out);
  d->dtype = a[1].dtype;
  d->bits = a[1].bits;
  d->rank = r;
  d->flags = NX_CONTIGUOUS | NX_DISTINCT;
  d->offset = 0;
  int64_t stride = 1;
  for (int i = r - 1; i >= 0; i--) {
    d->dim[i] = a[1].dim[i];
    d->dim[r + i] = stride;
    stride *= a[1].dim[i];
  }
  e = gather(a);
  nx_done(1, &a[1]);
  CAMLreturn(Val_int(e));
}
