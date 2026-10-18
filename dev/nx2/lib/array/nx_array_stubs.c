/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <stdlib.h>
#include <string.h>

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

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

/* Whether the descriptors [a] and [b], both with elements, are identical:
   one width, and every index at one byte: equal extents and strides, and
   one first bit. */
static int identical(const nx_array *a, const nx_array *b) {
  if (a->bits != b->bits || a->rank != b->rank) return 0;
  for (int i = 0; i < 2 * a->rank; i++)
    if (a->dim[i] != b->dim[i]) return 0;
  return (int64_t)(intptr_t)a->base * 8 + a->offset * a->bits ==
         (int64_t)(intptr_t)b->base * 8 + b->offset * b->bits;
}

/* Releases the claims of the first [n] operands [a] made, and answers [e]: a
   refusal after some claims. */
static int refuse_claimed(int n, const nx_array *a, int e) {
  for (int k = 0; k < n; k++) rig_buffer_release(a[k].buffer);
  return e;
}

/* nx_read's checks, which claim every operand or none: [span] holds 2n
   words for the operands' bytes. */
static int admit(int n, const nx_operand *in, nx_array *out, int64_t *span) {
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
    /* A loss's wait raises it. */
    for (int j = 0; j < m; j++) out[at + j].wait = answer[j] != RIG_CLAIMED;
  }
  /* Each operand's bytes, as host addresses, read once with its layout's
     span: [first, last) at span[2k], span[2k + 1]. */
  for (int k = 0; k < n; k++) {
    nx_array *a = &out[k];
    int64_t *first = &span[2 * k], *last = &span[2 * k + 1];
    if (in[k].written && !(a->flags & NX_DISTINCT))
      return refuse_claimed(n, out, NX_NOT_DISTINCT);
    if (a->flags & NX_EMPTY) {
      a->base = NULL;
      *first = *last = 0;
      continue;
    }
    if ((a->base = rig_buffer_host(a->buffer)) == NULL)
      return refuse_claimed(n, out, NX_NOT_HOST);
    value l = Field(in[k].array, ARRAY_LAYOUT);
    int64_t base = (int64_t)(intptr_t)a->base;
    *first = base + Long_val(Field(l, NX_LAYOUT_LO)) * a->bits / 8;
    *last = base + (Long_val(Field(l, NX_LAYOUT_HI)) * a->bits + 7) / 8;
  }
  /* A written operand shares bytes with no other but a read operand
     identical to it, which is marked its alias. Identity is of bytes, and
     two memories can hold the same bytes, such as a host borrow of a
     device's memory and a host buffer over the borrow's bigarray: each
     operand keeps its own claim and wait, for its own memory's work and
     loss. Read operands may share bytes, so only pairs with a written
     operand are compared. */
  for (int k = 0; k < n; k++) {
    int64_t fk = span[2 * k], lk = span[2 * k + 1];
    if (!in[k].written || fk == lk) continue;
    for (int j = 0; j < n; j++) {
      if (j == k || out[j].alias) continue;
      if (!(span[2 * j] < lk && fk < span[2 * j + 1])) continue;
      if (in[j].written || !identical(&out[k], &out[j]))
        return refuse_claimed(n, out, NX_OVERLAP);
      out[j].alias = 1;
    }
  }
  return NX_OK;
}

int nx_read(int n, const nx_operand *in, nx_array *out) {
  if (n <= 0) return NX_OK;
  /* The spans live on the stack up to NX_MAX_OPERANDS operands, beyond on
     the heap, allocated before any claim. */
  int64_t local[2 * NX_MAX_OPERANDS], *span = local;
  if (n > NX_MAX_OPERANDS) {
    span = malloc(2 * (size_t)n * sizeof *span);
    if (span == NULL) caml_raise_out_of_memory();
  }
  int e = admit(n, in, out, span);
  if (span != local) free(span);
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
      nx_done(n, out);
      (void)caml_get_value_or_raise(r);
    }
  }
  return NX_OK;
}

void nx_done(int n, nx_array *a) {
  if (n <= 0) return;
  for (int k = 0; k < n; k++) rig_buffer_release(a[k].buffer);
  /* Unlink the roots nx_read pushed, under any pushed since. */
  struct caml__roots_block **p = a[0].local;
  while (*p != &a[n - 1].roots) {
    if (*p == NULL)
      caml_fatal_error("nx_done: the descriptors' roots were popped before "
                       "nx_done");
    p = &(*p)->next;
  }
  *p = a[0].roots.next;
}

/* The OCaml door (Nx_array.door) claims through these, in C, so that no
   OCaml code, and so no asynchronous exception, runs between two claims.
   [written] and [read] are OCaml arrays of Nx_array.any, which is unboxed:
   an any is its array's block. */

static value any_buffer(value anys, mlsize_t k) {
  return Field(Field(anys, k), ARRAY_BUFFER);
}

/* The buffer of the [k]th array of [written], then [read]. */
static value nth_buffer(value written, value read, mlsize_t k) {
  mlsize_t nw = Wosize_val(written);
  return k < nw ? any_buffer(written, k) : any_buffer(read, k - nw);
}

static void release_first(value written, value read, mlsize_t n) {
  for (mlsize_t k = 0; k < n; k++)
    rig_buffer_release(nth_buffer(written, read, k));
}

/* Claims every array of [written] for writing and of [read] for reading, or
   none, and waits for nothing. Answers the refusal's code, with nothing
   claimed; or NX_OK, every array claimed, plus 256 times one more than the
   index of the first array whose claim answered RIG_LOST, 0 for none: a
   lost device is behind it, which the door probes for its exception. The
   index counts [written], then [read]. A read array identical to a written
   one is claimed too: claims are counted, so two of one memory do not
   conflict. It claims in runs, as nx_read does. */
intnat nx_array_claim_all(value written, value read) {
  mlsize_t nw = Wosize_val(written), n = nw + Wosize_val(read);
  value buffer[NX_MAX_OPERANDS];
  enum rig_access access[NX_MAX_OPERANDS];
  enum rig_claim answer[NX_MAX_OPERANDS];
  mlsize_t lost = 0;
  for (mlsize_t at = 0; at < n; at += NX_MAX_OPERANDS) {
    int m = n - at < NX_MAX_OPERANDS ? (int)(n - at) : NX_MAX_OPERANDS;
    for (int j = 0; j < m; j++) {
      buffer[j] = nth_buffer(written, read, at + j);
      access[j] = at + j < nw ? RIG_READ_WRITE : RIG_READ;
    }
    int refused = rig_buffers_claim(m, buffer, access, answer);
    if (refused >= 0) {
      release_first(written, read, at);
      return claim_code(answer[refused]);
    }
    for (int j = 0; j < m && lost == 0; j++)
      if (answer[j] == RIG_LOST) lost = at + j + 1;
  }
  return NX_OK + 256 * (intnat)lost;
}

value nx_array_claim_all_byte(value written, value read) {
  return Val_long(nx_array_claim_all(written, read));
}

value nx_array_release_all(value written, value read) {
  release_first(written, read, Wosize_val(written) + Wosize_val(read));
  return Val_unit;
}

/* Claims [b]'s memory for [access], RIG_READ or RIG_READ_WRITE as the
   constructors of Rig.access, and waits for nothing: NX_OK with the claim
   held, which nx_array_end_claim ends, plus 256 if the access must wait first
   (RIG_WAIT or RIG_LOST); or a refusal's code with none. */
intnat nx_array_claim(value b, value v_access) {
  enum rig_claim c = rig_buffer_claim(b, (enum rig_access)Int_val(v_access));
  int e = claim_code(c);
  if (e) return e;
  return c == RIG_CLAIMED ? NX_OK : NX_OK + 256;
}

value nx_array_claim_byte(value b, value v_access) {
  return Val_long(nx_array_claim(b, v_access));
}

/* Ends a claim nx_array_claim took on [b]. */
value nx_array_end_claim(value b) {
  rig_buffer_release(b);
  return Val_unit;
}

/* Whether the arrays [a] and [b] reach a common byte. Their buffers' first
   bytes lie at their host addresses where the host addresses both, and
   otherwise at [pa] and [pb] of one memory. */
value nx_array_shares(value a, value b, intnat pa, intnat pb) {
  value la = Field(a, ARRAY_LAYOUT), lb = Field(b, ARRAY_LAYOUT);
  if ((Long_val(Field(la, NX_LAYOUT_FLAGS)) |
       Long_val(Field(lb, NX_LAYOUT_FLAGS))) &
      NX_EMPTY)
    return Val_false;
  uint8_t *ha = rig_buffer_host(Field(a, ARRAY_BUFFER));
  uint8_t *hb = rig_buffer_host(Field(b, ARRAY_BUFFER));
  if (ha != NULL && hb != NULL) {
    pa = (intnat)(intptr_t)ha;
    pb = (intnat)(intptr_t)hb;
  }
  /* A span's end is at most NX_MAX_NUMEL: times 128 bits, it fits. */
  int64_t bits_a = nx_dtype_row_of(nx_array_dtype(a)).bits;
  int64_t bits_b = nx_dtype_row_of(nx_array_dtype(b)).bits;
  int64_t first_a = pa + Long_val(Field(la, NX_LAYOUT_LO)) * bits_a / 8;
  int64_t last_a = pa + (Long_val(Field(la, NX_LAYOUT_HI)) * bits_a + 7) / 8;
  int64_t first_b = pb + Long_val(Field(lb, NX_LAYOUT_LO)) * bits_b / 8;
  int64_t last_b = pb + (Long_val(Field(lb, NX_LAYOUT_HI)) * bits_b + 7) / 8;
  return Val_bool(first_a < last_b && first_b < last_a);
}

value nx_array_shares_byte(value a, value b, value pa, value pb) {
  return nx_array_shares(a, b, Long_val(pa), Long_val(pb));
}

/* Whether the arrays [a] and [b], which share a byte, are identical: one
   width, equal extents and strides, and one first bit, their buffers placed
   as nx_array_shares places them. */
value nx_array_identical(value a, value b, intnat pa, intnat pb) {
  value la = Field(a, ARRAY_LAYOUT), lb = Field(b, ARRAY_LAYOUT);
  int64_t bits = nx_dtype_row_of(nx_array_dtype(a)).bits;
  if (bits != nx_dtype_row_of(nx_array_dtype(b)).bits) return Val_false;
  value sa = Field(la, NX_LAYOUT_SHAPE), sb = Field(lb, NX_LAYOUT_SHAPE);
  value ta = Field(la, NX_LAYOUT_STRIDES), tb = Field(lb, NX_LAYOUT_STRIDES);
  mlsize_t r = Wosize_val(sa);
  if (Wosize_val(sb) != r) return Val_false;
  for (mlsize_t i = 0; i < r; i++)
    if (Field(sa, i) != Field(sb, i) || Field(ta, i) != Field(tb, i))
      return Val_false;
  uint8_t *ha = rig_buffer_host(Field(a, ARRAY_BUFFER));
  uint8_t *hb = rig_buffer_host(Field(b, ARRAY_BUFFER));
  if (ha != NULL && hb != NULL) {
    pa = (intnat)(intptr_t)ha;
    pb = (intnat)(intptr_t)hb;
  }
  return Val_bool(pa * 8 + Long_val(Field(la, NX_LAYOUT_OFFSET)) * bits ==
                  pb * 8 + Long_val(Field(lb, NX_LAYOUT_OFFSET)) * bits);
}

value nx_array_identical_byte(value a, value b, value pa, value pb) {
  return nx_array_identical(a, b, Long_val(pa), Long_val(pb));
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
  if (n < 1 || n > NX_MAX_OPERANDS) return NX_ARITY;
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
  if (n < 1 || n > NX_MAX_OPERANDS) return Val_int(NX_ARITY);
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

/* Element access from OCaml, on a buffer the caller claimed and waited
   for, at a position its layout reaches. */

intnat nx_array_host(value b) {
  void *p = rig_buffer_host(b);
  return p == NULL ? -1 : (intnat)(intptr_t)p;
}

double nx_array_get_float(value b, intnat dt, intnat p, intnat part) {
  return load_float(rig_buffer_host(b), (int)dt, p, (int)part);
}

int64_t nx_array_get_int(value b, intnat dt, intnat p) {
  return load_int(rig_buffer_host(b), (int)dt, p);
}

value nx_array_set_float(value b, intnat dt, intnat p, intnat part,
                         double x) {
  store_float(rig_buffer_host(b), (int)dt, p, (int)part, x);
  return Val_unit;
}

value nx_array_set_int(value b, intnat dt, intnat p, int64_t x) {
  store_int(rig_buffer_host(b), (int)dt, p, x);
  return Val_unit;
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

value nx_array_get_float_byte(value b, value dt, value p, value part) {
  return caml_copy_double(
      nx_array_get_float(b, Long_val(dt), Long_val(p), Long_val(part)));
}

value nx_array_get_int_byte(value b, value dt, value p) {
  return caml_copy_int64(nx_array_get_int(b, Long_val(dt), Long_val(p)));
}

value nx_array_set_float_byte(value b, value dt, value p, value part,
                              value x) {
  return nx_array_set_float(b, Long_val(dt), Long_val(p), Long_val(part),
                            Double_val(x));
}

value nx_array_set_int_byte(value b, value dt, value p, value x) {
  return nx_array_set_int(b, Long_val(dt), Long_val(p), Int64_val(x));
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
