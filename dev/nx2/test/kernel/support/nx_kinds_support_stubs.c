/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_kinds.h's kinds for test_kinds and sweep_kinds, through the per-target
   loops: every call runs the loop nx.cpu would run on this CPU, or a named
   target's. f32 operands and results travel as their bits, f64 ones as
   OCaml floats, integers as int64 values. nx_kinds_support_sweep releases
   the runtime; the other stubs take milliseconds and keep it. */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>

#include "nx_kinds.h"
#include "nx_kinds_strata.h"
#include "nx_kinds_support.h"

/* The loops of the target nx.cpu would pick on this CPU. */
static const nx_kinds_loops *nx_kinds_loops_best(void) {
#if defined(__x86_64__)
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
    return nx_kinds_loops_v3;
#endif
  return nx_kinds_loops_base;
}

static const nx_real_loop *nx_kinds_real_loop(const nx_kinds_loops *l,
                                              const char *name) {
  for (int i = 0; i < NX_REAL_LOOPS; i++)
    if (strcmp(l->real[i].name, name) == 0) return &l->real[i];
  return NULL;
}

static const nx_int_loop *nx_kinds_int_loop(const nx_kinds_loops *l,
                                            const char *name) {
  for (int i = 0; i < NX_INT_LOOPS; i++)
    if (strcmp(l->ints[i].name, name) == 0) return &l->ints[i];
  return NULL;
}

static const nx_real_loop *real_loop(const nx_kinds_loops *l, value name) {
  const nx_real_loop *k = nx_kinds_real_loop(l, String_val(name));
  if (k == NULL) caml_invalid_argument(String_val(name));
  return k;
}

static const nx_kinds_loops *target(value name) {
  const char *t = String_val(name);
  if (strcmp(t, "base") == 0) return nx_kinds_loops_base;
  if (strcmp(t, "v3") == 0 && nx_kinds_loops_v3 != NULL) return nx_kinds_loops_v3;
  caml_invalid_argument(t);
}

/* [targets ()]: the targets this CPU runs, the best last. */
value nx_kinds_support_targets(value unit) {
  CAMLparam1(unit);
  CAMLlocal2(l, cell);
  l = Val_emptylist;
  const char *names[2] = {"v3", "base"};
  for (int i = 0; i < 2; i++) {
    if (i == 0 && nx_kinds_loops_best() != nx_kinds_loops_v3) continue;
    cell = caml_alloc(2, 0);
    Store_field(cell, 0, caml_copy_string(names[i]));
    Store_field(cell, 1, l);
    l = cell;
  }
  CAMLreturn(l);
}

/* Scalar calls */

value nx_kinds_support_f32(value name, value args) {
  const nx_real_loop *k = real_loop(nx_kinds_loops_best(), name);
  if ((int)Wosize_val(args) != k->arity) caml_invalid_argument(String_val(name));
  float x[3] = {0, 0, 0}, y;
  for (int i = 0; i < k->arity; i++)
    x[i] = nx_bits_float((uint32_t)Long_val(Field(args, i)));
  k->f32(&x[0], &x[1], &x[2], &y, 1);
  return Val_long(nx_float_bits(y));
}

value nx_kinds_support_f64(value name, value args) {
  CAMLparam2(name, args);
  const nx_real_loop *k = real_loop(nx_kinds_loops_best(), name);
  if ((int)(Wosize_val(args) / Double_wosize) != k->arity)
    caml_invalid_argument(String_val(name));
  double x[3] = {0, 0, 0}, y;
  for (int i = 0; i < k->arity; i++) x[i] = Double_flat_field(args, i);
  k->f64(&x[0], &x[1], &x[2], &y, 1);
  CAMLreturn(caml_copy_double(y));
}

/* The integer kind [name] at [ty]: the result's int64, sign-extended from
   i32 and zero-extended from u32. */
value nx_kinds_support_int(value ty, value name, value args) {
  CAMLparam3(ty, name, args);
  const nx_int_loop *k = nx_kinds_int_loop(nx_kinds_loops_best(), String_val(name));
  if (k == NULL || (int)Wosize_val(args) != k->arity)
    caml_invalid_argument(String_val(name));
  uint64_t x[3] = {0, 0, 0}, y[4];
  for (int i = 0; i < k->arity; i++) x[i] = (uint64_t)Int64_val(Field(args, i));
  k->run(&x[0], &x[1], &x[2], y, 1);
  const char *t = String_val(ty);
  int64_t r;
  if (!strcmp(t, "i32")) r = (int32_t)(uint32_t)y[0];
  else if (!strcmp(t, "u32")) r = (int64_t)(uint32_t)y[1];
  else if (!strcmp(t, "i64")) r = (int64_t)y[2];
  else if (!strcmp(t, "u64")) r = (int64_t)y[3];
  else caml_invalid_argument(t);
  CAMLreturn(caml_copy_int64(r));
}

/* Errors of f32 results against the C library's f64 functions

   The error of y against the double r rounded correctly to binary32, in
   ordered-bit ranks with -0 and +0 one rank apart. r's own error is far
   below binary32's: where r lies within 2^-44 of it from a midpoint
   between two floats, either float could be the correctly rounded one, and
   the smaller error counts. A NaN against a number and a zero of the wrong
   sign count as 2^31. */

#define WRONG ((int64_t)1 << 31)

/* How f(-x) relates to f(x). */
enum sym { NONE, ODD, EVEN };

static const struct {
  const char *name;
  double (*ref)(double);
  enum sym sym;
} references[] = {
    {"exp", exp, NONE},    {"exp2", exp2, NONE},   {"expm1", expm1, NONE},
    {"log", log, NONE},    {"log2", log2, NONE},   {"log1p", log1p, NONE},
    {"sin", sin, ODD},     {"cos", cos, EVEN},     {"tan", tan, ODD},
    {"asin", asin, ODD},   {"acos", acos, NONE},   {"atan", atan, ODD},
    {"sinh", sinh, ODD},   {"cosh", cosh, EVEN},   {"tanh", tanh, ODD},
    {"erf", erf, ODD}};

static int find_reference(value name) {
  for (int i = 0; i < (int)(sizeof references / sizeof *references); i++)
    if (strcmp(references[i].name, String_val(name)) == 0) return i;
  caml_invalid_argument(String_val(name));
}

static int64_t rank32(float f) {
  uint32_t u = nx_float_bits(f);
  return (u >> 31) ? -(int64_t)(u & 0x7FFFFFFFu) - 1 : (int64_t)u;
}

static int64_t error32(float y, double r, int *ambiguous) {
  float c = (float)r;
  if (nx_float_bits(y) == nx_float_bits(c)) return 0;
  if (y != y || r != r) return (y != y && r != r) ? 0 : WRONG;
  if (y == 0 && c == 0) return WRONG;
  int64_t e = llabs(rank32(y) - rank32(c));
  if (isfinite(c) && r != (double)c) {
    float n = nextafterf(c, (double)c < r ? INFINITY : -INFINITY);
    double mid = ((double)c + (double)n) / 2;
    if (fabs(r - mid) <= fabs(r) * 0x1p-44 + 0x1p-1060) {
      *ambiguous = 1;
      int64_t e2 = llabs(rank32(y) - rank32(n));
      e = e2 < e ? e2 : e;
    }
  }
  return e;
}

/* The worst points a check finds: the largest errors, and among equal
   ones at most two to a binade, ordered by a scramble of the pattern, so
   that a kind with many points at its maximum records points spread over
   its range. Nx_kinds_support.merge orders them the same way. */
#define WORST 64

typedef struct {
  int64_t error[WORST];
  uint32_t arg[WORST];
  int n;
  long ambiguous;
} worst;

static int before(int64_t e, uint32_t p, int64_t e2, uint32_t p2) {
  return e != e2 ? e > e2 : p * 0x9E3779B1u < p2 * 0x9E3779B1u;
}

static void note(worst *w, int64_t e, uint32_t p) {
  if (e == 0) return;
  if (w->n == WORST && !before(e, p, w->error[WORST - 1], w->arg[WORST - 1]))
    return;
  /* a third tie in p's binade replaces the later of the two there */
  int ties = 0, later = -1;
  for (int j = 0; j < w->n; j++)
    if (w->error[j] == e && w->arg[j] >> 23 == p >> 23 && ++ties == 2)
      later = j;
  if (later >= 0) {
    if (!before(e, p, e, w->arg[later])) return;
    memmove(&w->error[later], &w->error[later + 1],
            (size_t)(w->n - later - 1) * sizeof *w->error);
    memmove(&w->arg[later], &w->arg[later + 1],
            (size_t)(w->n - later - 1) * sizeof *w->arg);
    w->n--;
  }
  int i = w->n < WORST ? w->n++ : WORST - 1;
  for (; i > 0 && before(e, p, w->error[i - 1], w->arg[i - 1]); i--) {
    w->error[i] = w->error[i - 1];
    w->arg[i] = w->arg[i - 1];
  }
  w->error[i] = e;
  w->arg[i] = p;
}

/* Checks the f32 kind at the n patterns p, noting errors in w. A
   symmetric kind is checked at each pattern's magnitude against the
   reference, and at the negated magnitude against its value there, bit for
   bit, so one pattern covers both signs. */
#define BLOCK 4096

static void check32(const nx_real_loop *k, int ref, const uint32_t *p, long n,
                    worst *w) {
  double (*f)(double) = references[ref].ref;
  enum sym sym = references[ref].sym;
  static _Thread_local float x[BLOCK], nx[BLOCK], y[BLOCK], m[BLOCK];
  for (long base = 0; base < n; base += BLOCK) {
    int b = (int)(n - base < BLOCK ? n - base : BLOCK);
    for (int j = 0; j < b; j++) {
      x[j] = nx_bits_float(p[base + j]);
      if (sym != NONE) x[j] = nx_abs_bits_f32(x[j]);
      nx[j] = -x[j];
    }
    k->f32(x, NULL, NULL, y, b);
    if (sym != NONE) {
      k->f32(nx, NULL, NULL, m, b);
      for (int j = 0; j < b; j++) {
        float want = sym == ODD ? -y[j] : y[j];
        int same = nx_float_bits(m[j]) == nx_float_bits(want) ||
                   (m[j] != m[j] && y[j] != y[j]);
        if (!same) note(w, WRONG, nx_float_bits(nx[j]));
      }
    }
    for (int j = 0; j < b; j++) {
      double r = f((double)x[j]);
      if (nx_float_bits(y[j]) == nx_float_bits((float)r)) continue;
      int amb = 0;
      note(w, error32(y[j], r, &amb), nx_float_bits(x[j]));
      w->ambiguous += amb;
    }
  }
}

static value worst_value(const worst *w) {
  CAMLparam0();
  CAMLlocal3(res, pts, pair);
  pts = caml_alloc(w->n, 0);
  for (int i = 0; i < w->n; i++) {
    pair = caml_alloc_tuple(2);
    Store_field(pair, 0, Val_long(w->error[i]));
    Store_field(pair, 1, Val_long(w->arg[i]));
    Store_field(pts, i, pair);
  }
  res = caml_alloc_tuple(2);
  Store_field(res, 0, pts);
  Store_field(res, 1, Val_long(w->ambiguous));
  CAMLreturn(res);
}

/* [sweep kind lo hi step]: the f32 transcendental [kind] at the patterns
   lo, lo + step, ... below hi, on the best target, without the runtime.
   A symmetric kind takes only the positive patterns, each with its
   negation, and checks nothing at a negative one; a step dividing 2^31
   keeps the negations in the set. */
value nx_kinds_support_sweep(value name, value lo, value hi, value step) {
  CAMLparam4(name, lo, hi, step);
  const nx_real_loop *k = real_loop(nx_kinds_loops_best(), name);
  int ref = find_reference(name);
  uint64_t a = (uint64_t)Long_val(lo), b = (uint64_t)Long_val(hi);
  uint64_t d = (uint64_t)Long_val(step);
  if (d == 0) caml_invalid_argument("Nx_kinds_support.sweep_range: step 0");
  if (references[ref].sym != NONE && b > 0x80000000u) b = 0x80000000u;
  worst w = {.n = 0, .ambiguous = 0};
  caml_enter_blocking_section();
  static _Thread_local uint32_t p[BLOCK];
  for (uint64_t u = a; u < b;) {
    long n = 0;
    for (; u < b && n < BLOCK; u += d) p[n++] = (uint32_t)u;
    check32(k, ref, p, n, &w);
  }
  caml_leave_blocking_section();
  CAMLreturn(worst_value(&w));
}

/* [points kind ps]: the f32 transcendental [kind] at the patterns ps. */
value nx_kinds_support_points(value name, value ps) {
  CAMLparam2(name, ps);
  const nx_real_loop *k = real_loop(nx_kinds_loops_best(), name);
  int ref = find_reference(name);
  long n = (long)Wosize_val(ps);
  uint32_t *p = malloc((size_t)(n ? n : 1) * sizeof *p);
  if (p == NULL) caml_raise_out_of_memory();
  for (long i = 0; i < n; i++) p[i] = (uint32_t)Long_val(Field(ps, i));
  worst w = {.n = 0, .ambiguous = 0};
  check32(k, ref, p, n, &w);
  free(p);
  CAMLreturn(worst_value(&w));
}

/* The strata */

static uint64_t *strata[2];
static uint8_t *regions[2];
static size_t strata_n[2];

/* The f32 (f64 = 0) or f64 strata, built at the first use. */
static size_t strata_of(int f64) {
  if (strata[f64] == NULL) {
    nx_strata s = {NULL, NULL, 0, 0};
    size_t n = nx_strata_real(&s, f64);
    s.bits = malloc(n * sizeof *s.bits);
    s.region = malloc(n);
    if (s.bits == NULL || s.region == NULL) caml_raise_out_of_memory();
    s.n = 0;
    s.cap = n;
    nx_strata_real(&s, f64);
    strata[f64] = s.bits;
    regions[f64] = s.region;
    strata_n[f64] = n;
  }
  return strata_n[f64];
}

/* [strata kind]: the f32 transcendental [kind] at the f32 strata, by
   region: each region's size and its worst points. */
value nx_kinds_support_strata(value name) {
  CAMLparam1(name);
  CAMLlocal3(res, row, w);
  const nx_real_loop *k = real_loop(nx_kinds_loops_best(), name);
  int ref = find_reference(name);
  size_t n = strata_of(0);
  res = caml_alloc(NX_REGION_COUNT, 0);
  uint32_t *p = malloc(n * sizeof *p);
  if (p == NULL) caml_raise_out_of_memory();
  for (int r = 0; r < NX_REGION_COUNT; r++) {
    long m = 0;
    for (size_t i = 0; i < n; i++)
      if (regions[0][i] == r) p[m++] = (uint32_t)strata[0][i];
    worst wr = {.n = 0, .ambiguous = 0};
    check32(k, ref, p, m, &wr);
    w = worst_value(&wr);
    row = caml_alloc_tuple(3);
    Store_field(row, 0, caml_copy_string(nx_region_names[r]));
    Store_field(row, 1, Val_long(m));
    Store_field(row, 2, w);
    Store_field(res, r, row);
  }
  free(p);
  CAMLreturn(res);
}

/* Digests

   [digest target kind ty]: the digest of the kind at [ty] ("f32", "f64"
   or "int") over its strata on [target], as 16 hex digits. Kinds of two
   and three operands take every pair (or triple) of nx_strata_specials,
   then 2^18 tuples drawn from the strata; integer kinds take 2^18 drawn
   pairs and every pair of their extremes. */

#define DRAWS (1 << 18)
#define SPECIALS ((int)NX_STRATA_COUNT(nx_strata_specials))

static uint64_t digest_real(const nx_real_loop *k, int f64) {
  size_t n = strata_of(f64), m;
  uint64_t h = NX_FNV_START;
  size_t s3 = (size_t)SPECIALS * SPECIALS * (k->arity == 3 ? SPECIALS : 1);
  m = k->arity == 1 ? n : s3 + DRAWS;
  size_t bytes = f64 ? 8 : 4;
  char *x = malloc(m * bytes), *z = malloc(m * bytes), *w = malloc(m * bytes);
  char *y = malloc(m * bytes);
  if (!x || !z || !w || !y) caml_raise_out_of_memory();
  for (size_t i = 0; i < m; i++) {
    uint64_t a, b, c;
    if (k->arity == 1) {
      a = strata[f64][i];
      b = c = 0;
    } else if (i < s3) {
      a = nx_strata_bits(nx_strata_specials[i % SPECIALS], f64);
      b = nx_strata_bits(nx_strata_specials[(i / SPECIALS) % SPECIALS], f64);
      c = nx_strata_bits(nx_strata_specials[i / ((size_t)SPECIALS * SPECIALS)], f64);
    } else {
      a = strata[f64][nx_strata_mix(3 * i) % n];
      b = strata[f64][nx_strata_mix(3 * i + 1) % n];
      c = strata[f64][nx_strata_mix(3 * i + 2) % n];
    }
    if (f64) {
      memcpy(x + 8 * i, &a, 8);
      memcpy(z + 8 * i, &b, 8);
      memcpy(w + 8 * i, &c, 8);
    } else {
      uint32_t a32 = (uint32_t)a, b32 = (uint32_t)b, c32 = (uint32_t)c;
      memcpy(x + 4 * i, &a32, 4);
      memcpy(z + 4 * i, &b32, 4);
      memcpy(w + 4 * i, &c32, 4);
    }
  }
  if (f64) {
    k->f64((double *)x, (double *)z, (double *)w, (double *)y, (long)m);
    for (size_t i = 0; i < m; i++) h = nx_fnv(h, nx_digest_f64(((double *)y)[i]));
  } else {
    k->f32((float *)x, (float *)z, (float *)w, (float *)y, (long)m);
    for (size_t i = 0; i < m; i++) h = nx_fnv(h, nx_digest_f32(((float *)y)[i]));
  }
  free(x);
  free(z);
  free(w);
  free(y);
  return h;
}

static const uint64_t int_extremes[] = {
    0, 1, UINT64_MAX, 2, UINT64_MAX - 1, UINT64_C(0x7FFFFFFF),
    UINT64_C(0x80000000), UINT64_C(0xFFFFFFFF), UINT64_C(0x7FFFFFFFFFFFFFFF),
    UINT64_C(0x8000000000000000), UINT64_C(0xFFFFFFFF80000000), 31, 32, 63};

#define EXTREMES ((int)(sizeof int_extremes / sizeof *int_extremes))

static uint64_t digest_int(const nx_int_loop *k) {
  size_t s = (size_t)EXTREMES * EXTREMES * (k->arity == 3 ? EXTREMES : 1);
  size_t m = s + DRAWS;
  uint64_t *x = malloc(m * 8), *z = malloc(m * 8), *w = malloc(m * 8);
  uint64_t *y = malloc(m * 32);
  if (!x || !z || !w || !y) caml_raise_out_of_memory();
  for (size_t i = 0; i < m; i++) {
    if (i < s) {
      x[i] = int_extremes[i % EXTREMES];
      z[i] = int_extremes[(i / EXTREMES) % EXTREMES];
      w[i] = int_extremes[i / ((size_t)EXTREMES * EXTREMES)];
    } else {
      /* a quarter of the draws small, so that powers and divisions matter */
      uint64_t a = nx_strata_mix(3 * i), b = nx_strata_mix(3 * i + 1);
      x[i] = (i & 3) == 0 ? (uint64_t)((int64_t)(a % 41) - 20) : a;
      z[i] = (i & 3) == 1 ? (uint64_t)((int64_t)(b % 41) - 20) : b;
      w[i] = nx_strata_mix(3 * i + 2);
    }
  }
  k->run(x, z, w, y, (long)m);
  uint64_t h = NX_FNV_START;
  for (size_t i = 0; i < 4 * m; i++) h = nx_fnv(h, y[i]);
  free(x);
  free(z);
  free(w);
  free(y);
  return h;
}

value nx_kinds_support_digest(value tgt, value name, value ty) {
  CAMLparam3(tgt, name, ty);
  const nx_kinds_loops *l = target(tgt);
  const char *t = String_val(ty);
  uint64_t h;
  if (!strcmp(t, "int")) {
    const nx_int_loop *k = nx_kinds_int_loop(l, String_val(name));
    if (k == NULL) caml_invalid_argument(String_val(name));
    h = digest_int(k);
  } else if (!strcmp(t, "f32") || !strcmp(t, "f64")) {
    h = digest_real(real_loop(l, name), !strcmp(t, "f64"));
  } else {
    caml_invalid_argument(t);
  }
  char s[17];
  for (int i = 0; i < 16; i++) s[i] = "0123456789abcdef"[(h >> (60 - 4 * i)) & 15];
  s[16] = 0;
  CAMLreturn(caml_copy_string(s));
}

/* The narrow floats

   [narrow kind dt]: the f32 transcendental [kind] at every code of the
   narrow float dtype [dt], computed in f32 and rounded once to [dt],
   against the C library's f64 function rounded once to [dt]: the largest
   error in the dtype's ordered ranks, and a code where it occurs. NaN is at
   no distance from NaN. */

static uint32_t narrow_of_double(int dt, double r) {
  switch (dt) {
  case NX_FLOAT16: return nx_double_to_f16(r);
  case NX_BFLOAT16: return nx_double_to_bf16(r);
  case NX_FLOAT8_E4M3FN: return nx_double_to_e4m3fn(r);
  case NX_FLOAT8_E5M2: return nx_double_to_e5m2(r);
  default: return nx_double_to_e2m1fn(r);
  }
}

static uint32_t narrow_of_float(int dt, float r) {
  switch (dt) {
  case NX_FLOAT16: return nx_float_to_f16(r);
  case NX_BFLOAT16: return nx_float_to_bf16(r);
  case NX_FLOAT8_E4M3FN: return nx_float_to_e4m3fn(r);
  case NX_FLOAT8_E5M2: return nx_float_to_e5m2(r);
  default: return nx_float_to_e2m1fn(r);
  }
}

value nx_kinds_support_narrow(value name, value code) {
  CAMLparam2(name, code);
  CAMLlocal1(res);
  const nx_real_loop *k = real_loop(nx_kinds_loops_best(), name);
  double (*f)(double) = references[find_reference(name)].ref;
  int dt = (int)Long_val(code);
  int bits = nx_dtype_row_of(dt).bits;
  if (nx_dtype_row_of(dt).kind != NX_KIND_FLOAT || bits > 16)
    caml_invalid_argument("Nx_kinds_support.narrow: not a narrow float");
  uint32_t n = 1u << bits, sign = n >> 1;
  float *x = malloc(n * sizeof *x), *y = malloc(n * sizeof *y);
  if (!x || !y) caml_raise_out_of_memory();
  for (uint32_t c = 0; c < n; c++) x[c] = nx_bits_to_float(dt, c);
  k->f32(x, NULL, NULL, y, (long)n);
  long worst = 0, arg = 0;
  for (uint32_t c = 0; c < n; c++) {
    uint32_t got = narrow_of_float(dt, y[c]);
    uint32_t want = narrow_of_double(dt, f((double)x[c]));
    float gf = nx_bits_to_float(dt, got), wf = nx_bits_to_float(dt, want);
    long e;
    if (gf != gf || wf != wf) {
      e = (gf != gf && wf != wf) ? 0 : (long)WRONG;
    } else {
      long rg = (got & sign) ? -(long)(got & (sign - 1)) - 1 : (long)got;
      long rw = (want & sign) ? -(long)(want & (sign - 1)) - 1 : (long)want;
      e = labs(rg - rw);
    }
    if (e > worst) {
      worst = e;
      arg = (long)c;
    }
  }
  free(x);
  free(y);
  res = caml_alloc_tuple(2);
  Store_field(res, 0, Val_long(worst));
  Store_field(res, 1, Val_long(arg));
  CAMLreturn(res);
}

/* f32 pow and atan2

   [binary kind seed n]: the f32 kind at n pairs against the C library's
   f64 function: a sixth drawn from all patterns, the rest from strata
   where the kind is hard. pow: x within 2^-20 of 1 with |y| up to 2^30;
   y log2 x near the overflow at 128 and the underflows at -126 and
   -149.5; x near sqrt(2) 2^k, where log's reduction switches; y log2 x
   near k + 1/2, where exp2's does. atan2: ratios near 1, where atan's
   reduction switches, tiny against huge, and operands of every binade.
   The draws follow [seed]. Returns the largest error and a pair where it
   occurs. */

static float uniform_f32(uint64_t *s, float lo, float hi) {
  *s = nx_strata_mix(*s);
  return lo + (hi - lo) * (float)((*s >> 40) * 0x1p-24);
}

static float pattern_f32(uint64_t *s) {
  *s = nx_strata_mix(*s);
  return nx_bits_float((uint32_t)*s);
}

value nx_kinds_support_binary(value name, value seed, value count) {
  CAMLparam3(name, seed, count);
  CAMLlocal1(res);
  const nx_real_loop *k = real_loop(nx_kinds_loops_best(), name);
  int is_pow = !strcmp(String_val(name), "pow");
  if (!is_pow && strcmp(String_val(name), "atan2"))
    caml_invalid_argument(String_val(name));
  long n = Long_val(count);
  uint64_t s = (uint64_t)Long_val(seed);
  float *x = malloc((size_t)n * 4), *z = malloc((size_t)n * 4), *y = malloc((size_t)n * 4);
  if (!x || !z || !y) caml_raise_out_of_memory();
  for (long i = 0; i < n; i++) {
    float a, b;
    int stratum = (int)(i % 6);
    if (stratum == 0) {
      a = pattern_f32(&s);
      b = pattern_f32(&s);
    } else if (is_pow && stratum == 1) {
      a = 1.0f + uniform_f32(&s, -0x1p-20f, 0x1p-20f);
      b = uniform_f32(&s, -1.0f, 1.0f) * 0x1p30f;
    } else if (is_pow && stratum == 4) {
      int k = (int)((s >> 5) % 250) - 125;
      a = nx_bits_float(nx_float_bits(ldexpf(1.41421356f, k)) +
                        (uint32_t)((s >> 13) % 9) - 4u);
      b = uniform_f32(&s, -8.0f, 8.0f);
    } else if (is_pow && stratum == 5) {
      a = uniform_f32(&s, 0.5f, 100.0f);
      float t = (float)((int)((s >> 9) % 260) - 140) + 0.5f +
                uniform_f32(&s, -0x1p-12f, 0x1p-12f);
      b = (float)(t / log2((double)a));
    } else if (is_pow) {
      static const float targets[3] = {128.0f, -126.0f, -149.5f};
      a = uniform_f32(&s, 1.0001f, 1000.0f);
      if (s & 1) a = 1.0f / a;
      float t = targets[(s >> 7) % 3] + uniform_f32(&s, -1.0f, 1.0f);
      b = (float)(t / log2((double)a));
    } else if (stratum == 1 || stratum == 4) {
      a = uniform_f32(&s, -100.0f, 100.0f);
      b = a * (1.0f + uniform_f32(&s, -0x1p-10f, 0x1p-10f));
    } else {
      int ea = (int)((s >> 3) % 250) - 125, eb = (int)((s >> 11) % 250) - 125;
      a = ldexpf(uniform_f32(&s, 1.0f, 2.0f), ea);
      b = ldexpf(uniform_f32(&s, -2.0f, 2.0f), eb);
    }
    x[i] = a;
    z[i] = b;
  }
  k->f32(x, z, NULL, y, n);
  int64_t worst = 0;
  long arg = 0;
  for (long i = 0; i < n; i++) {
    double r = is_pow ? pow((double)x[i], (double)z[i]) : atan2((double)x[i], (double)z[i]);
    int amb = 0;
    int64_t e = error32(y[i], r, &amb);
    if (e > worst) {
      worst = e;
      arg = i;
    }
  }
  res = caml_alloc_tuple(3);
  Store_field(res, 0, Val_long(worst));
  Store_field(res, 1, Val_long(nx_float_bits(x[arg])));
  Store_field(res, 2, Val_long(nx_float_bits(z[arg])));
  free(x);
  free(z);
  free(y);
  CAMLreturn(res);
}

value nx_kinds_support_threefry(value counter, value key) {
  CAMLparam2(counter, key);
  uint64_t r = nx_threefry_u64((uint64_t)Int64_val(counter), (uint64_t)Int64_val(key));
  CAMLreturn(caml_copy_int64((int64_t)r));
}

/* [run kind x z y]: y.{i} <- kind x.{i} (z.{i}) on the best target, x, z
   and y float32 or float64 bigarrays of one length; the bench's rows. */
value nx_kinds_support_run(value name, value x, value z, value y) {
  const nx_real_loop *k = real_loop(nx_kinds_loops_best(), name);
  long n = (long)Caml_ba_array_val(x)->dim[0];
  if ((Caml_ba_array_val(x)->flags & CAML_BA_KIND_MASK) == CAML_BA_FLOAT64)
    k->f64(Caml_ba_data_val(x), Caml_ba_data_val(z), NULL, Caml_ba_data_val(y), n);
  else
    k->f32(Caml_ba_data_val(x), Caml_ba_data_val(z), NULL, Caml_ba_data_val(y), n);
  return Val_unit;
}
