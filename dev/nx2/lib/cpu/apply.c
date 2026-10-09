/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Kinds of no, two and three operands.

   Each kind runs as a row function per dtype, which loads each operand's
   element into its compute type, computes the kind as nx_kinds.h does, and
   stores the result in the destination's dtype: 8- and 16-bit integers
   compute in the 32-bit type of their signedness and wrap on the store,
   booleans compute in uint32 from 0 or 1. The walk hands the row functions
   rows of a block; rows of contiguous elements take a loop the compiler
   vectorises. A dtype with no row function, a narrow float, complex or
   sub-byte one, is declined. Fill and Where move bits, so they run on every
   dtype of a byte or more by its width alone.

   OCaml passes a kind as its value: a constant constructor is its index
   among the type's constant constructors, a constructor with an argument a
   block of that tag. The enums of nx_spec.h list the kinds in that order. */

#include <string.h>

#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "cpu.h"
#include "nx_kinds.h"
#include "nx_spec.h"

/* Elements of a block: at 16 Ki a block's call costs next to nothing
   beside its loads and stores, and its rows stay in L1. */
#define MOST (16 * 1024)

typedef void (*row2)(int64_t n, uint8_t *d, int64_t sd, const uint8_t *x,
                     int64_t sx, const uint8_t *y, int64_t sy);

typedef void (*row3)(int64_t n, uint8_t *d, int64_t sd, const uint8_t *c,
                     int64_t sc, const uint8_t *x, int64_t sx,
                     const uint8_t *y, int64_t sy);

/* Rows */

/* A row of [F] over two operands of the storage type [T], loaded by [LD]
   into the compute type, stored as [R]. */
#define BIN(NAME, T, R, LD, F)                                               \
  static void NAME(int64_t n, uint8_t *d_, int64_t sd, const uint8_t *x_,   \
                   int64_t sx, const uint8_t *y_, int64_t sy) {             \
    R *d = (R *)d_;                                                          \
    const T *x = (const T *)x_, *y = (const T *)y_;                          \
    if (sd == 1 && sx == 1 && sy == 1)                                       \
      for (int64_t i = 0; i < n; i++) d[i] = (R)F(LD(x[i]), LD(y[i]));       \
    else                                                                     \
      for (int64_t i = 0; i < n; i++)                                        \
        d[i * sd] = (R)F(LD(x[i * sx]), LD(y[i * sy]));                      \
  }

#define FMA(NAME, T, LD, F)                                                  \
  static void NAME(int64_t n, uint8_t *d_, int64_t sd, const uint8_t *a_,   \
                   int64_t sa, const uint8_t *b_, int64_t sb,                \
                   const uint8_t *c_, int64_t sc) {                          \
    T *d = (T *)d_;                                                          \
    const T *a = (const T *)a_, *b = (const T *)b_, *c = (const T *)c_;      \
    if (sd == 1 && sa == 1 && sb == 1 && sc == 1)                            \
      for (int64_t i = 0; i < n; i++)                                        \
        d[i] = (T)F(LD(a[i]), LD(b[i]), LD(c[i]));                           \
    else                                                                     \
      for (int64_t i = 0; i < n; i++)                                        \
        d[i * sd] = (T)F(LD(a[i * sa]), LD(b[i * sb]), LD(c[i * sc]));       \
  }

/* The kinds every dtype of a class takes, at the compute suffix [S]. */
#define COMPARES(D, T, LD, S)                                                \
  BIN(equal_##D, T, uint8_t, LD, nx_equal_##S)                               \
  BIN(not_equal_##D, T, uint8_t, LD, nx_not_equal_##S)                       \
  BIN(less_##D, T, uint8_t, LD, nx_less_##S)                                 \
  BIN(less_equal_##D, T, uint8_t, LD, nx_less_equal_##S)                     \
  BIN(maximum_##D, T, T, LD, nx_maximum_##S)                                 \
  BIN(minimum_##D, T, T, LD, nx_minimum_##S)

#define ARITH(D, T, LD, S)                                                   \
  COMPARES(D, T, LD, S)                                                      \
  BIN(add_##D, T, T, LD, nx_add_##S)                                         \
  BIN(sub_##D, T, T, LD, nx_sub_##S)                                         \
  BIN(mul_##D, T, T, LD, nx_mul_##S)                                         \
  BIN(mod_##D, T, T, LD, nx_mod_##S)                                         \
  BIN(pow_##D, T, T, LD, nx_pow_##S)                                         \
  FMA(fma_##D, T, LD, nx_fma_##S)

#define FLOATS(D, T, S)                                                      \
  ARITH(D, T, , S)                                                           \
  BIN(fdiv_##D, T, T, , nx_fdiv_##S)                                         \
  BIN(atan2_##D, T, T, , nx_atan2_##S)

#define INTS(D, T, CT, S)                                                    \
  ARITH(D, T, (CT), S)                                                       \
  BIN(idiv_##D, T, T, (CT), nx_idiv_##S)                                     \
  BIN(and_##D, T, T, (CT), nx_and_##S)                                       \
  BIN(or_##D, T, T, (CT), nx_or_##S)                                         \
  BIN(xor_##D, T, T, (CT), nx_xor_##S)

/* A boolean is 1 where its byte is not zero. */
#define BOOL_LD(v) ((uint32_t)((v) != 0))

FLOATS(f32, float, f32)
FLOATS(f64, double, f64)
INTS(i8, int8_t, int32_t, i32)
INTS(i16, int16_t, int32_t, i32)
INTS(i32, int32_t, int32_t, i32)
INTS(i64, int64_t, int64_t, i64)
INTS(u8, uint8_t, uint32_t, u32)
INTS(u16, uint16_t, uint32_t, u32)
INTS(u32, uint32_t, uint32_t, u32)
INTS(u64, uint64_t, uint64_t, u64)
COMPARES(b, uint8_t, BOOL_LD, u32)
BIN(and_b, uint8_t, uint8_t, BOOL_LD, nx_and_u32)
BIN(or_b, uint8_t, uint8_t, BOOL_LD, nx_or_u32)
BIN(xor_b, uint8_t, uint8_t, BOOL_LD, nx_xor_u32)
BIN(threefry_u64, uint64_t, uint64_t, , nx_threefry_u64)

/* The tables, by kind then dtype: NULL where the case is declined. */

#define CMP_ROWS(DT, D)                                                      \
  [NX_OP2_EQUAL][DT] = equal_##D, [NX_OP2_NOT_EQUAL][DT] = not_equal_##D,    \
  [NX_OP2_LESS][DT] = less_##D, [NX_OP2_LESS_EQUAL][DT] = less_equal_##D,    \
  [NX_OP2_MAXIMUM][DT] = maximum_##D, [NX_OP2_MINIMUM][DT] = minimum_##D

#define ARITH_ROWS(DT, D)                                                    \
  CMP_ROWS(DT, D), [NX_OP2_ADD][DT] = add_##D, [NX_OP2_SUB][DT] = sub_##D,   \
  [NX_OP2_MUL][DT] = mul_##D, [NX_OP2_MOD][DT] = mod_##D,                    \
  [NX_OP2_POW][DT] = pow_##D

#define FLOAT_ROWS(DT, D)                                                    \
  ARITH_ROWS(DT, D), [NX_OP2_FDIV][DT] = fdiv_##D,                           \
  [NX_OP2_ATAN2][DT] = atan2_##D

#define INT_ROWS(DT, D)                                                      \
  ARITH_ROWS(DT, D), [NX_OP2_IDIV][DT] = idiv_##D,                           \
  [NX_OP2_AND][DT] = and_##D, [NX_OP2_OR][DT] = or_##D,                      \
  [NX_OP2_XOR][DT] = xor_##D

static const row2 rows2[NX_OP2_COUNT][NX_DTYPE_COUNT] = {
    FLOAT_ROWS(NX_FLOAT32, f32),
    FLOAT_ROWS(NX_FLOAT64, f64),
    INT_ROWS(NX_INT8, i8),
    INT_ROWS(NX_INT16, i16),
    INT_ROWS(NX_INT32, i32),
    INT_ROWS(NX_INT64, i64),
    INT_ROWS(NX_UINT8, u8),
    INT_ROWS(NX_UINT16, u16),
    INT_ROWS(NX_UINT32, u32),
    INT_ROWS(NX_UINT64, u64),
    CMP_ROWS(NX_BOOL, b),
    [NX_OP2_AND][NX_BOOL] = and_b,
    [NX_OP2_OR][NX_BOOL] = or_b,
    [NX_OP2_XOR][NX_BOOL] = xor_b,
    [NX_OP2_THREEFRY][NX_UINT64] = threefry_u64,
};

static const row3 fmas[NX_DTYPE_COUNT] = {
    [NX_FLOAT32] = fma_f32, [NX_FLOAT64] = fma_f64, [NX_INT8] = fma_i8,
    [NX_INT16] = fma_i16,   [NX_INT32] = fma_i32,   [NX_INT64] = fma_i64,
    [NX_UINT8] = fma_u8,    [NX_UINT16] = fma_u16,  [NX_UINT32] = fma_u32,
    [NX_UINT64] = fma_u64,
};

/* Where selects bits: one row per width, its condition a boolean. */
#define WHERE(W, T)                                                          \
  static void where_##W(int64_t n, uint8_t *d_, int64_t sd,                 \
                        const uint8_t *c, int64_t sc, const uint8_t *x_,     \
                        int64_t sx, const uint8_t *y_, int64_t sy) {         \
    T *d = (T *)d_;                                                          \
    const T *x = (const T *)x_, *y = (const T *)y_;                          \
    if (sd == 1 && sc == 1 && sx == 1 && sy == 1)                            \
      for (int64_t i = 0; i < n; i++) d[i] = c[i] ? x[i] : y[i];             \
    else                                                                     \
      for (int64_t i = 0; i < n; i++)                                        \
        d[i * sd] = c[i * sc] ? x[i * sx] : y[i * sy];                       \
  }

typedef struct {
  uint64_t lo, hi;
} w16;

WHERE(1, uint8_t)
WHERE(2, uint16_t)
WHERE(4, uint32_t)
WHERE(8, uint64_t)
WHERE(16, w16)

static row3 where_of(int bytes) {
  switch (bytes) {
    case 1: return where_1;
    case 2: return where_2;
    case 4: return where_4;
    case 8: return where_8;
    case 16: return where_16;
    default: return NULL;
  }
}

/* Blocks */

typedef struct {
  const nx_array *a;
  int n;
  row2 f2;
  row3 f3;
  const uint8_t *bits; /* a fill's element */
} job;

/* The address of operand [k]'s element at position [p]. */
static inline uint8_t *at(const nx_array *a, int k, int64_t p) {
  return a[k].base + p * (a[k].bits / 8);
}

static void block(const nx_cpu_block *b, void *ctx) {
  const job *j = ctx;
  const nx_array *a = j->a;
  for (int64_t q = 0; q < b->n2; q++)
    for (int64_t r = 0; r < b->n1; r++) {
      int64_t p[NX_MAX_OPERANDS];
      for (int k = 0; k < j->n; k++)
        p[k] = b->at[k] + q * b->s2[k] + r * b->s1[k];
      if (j->n == 3)
        j->f2(b->n0, at(a, 0, p[0]), b->s0[0], at(a, 1, p[1]), b->s0[1],
              at(a, 2, p[2]), b->s0[2]);
      else
        j->f3(b->n0, at(a, 0, p[0]), b->s0[0], at(a, 1, p[1]), b->s0[1],
              at(a, 2, p[2]), b->s0[2], at(a, 3, p[3]), b->s0[3]);
    }
}

/* A fill stores its element by its width. */
#define FILL(W, T)                                                           \
  static void fill_##W(const nx_cpu_block *b, void *ctx) {                   \
    const job *j = ctx;                                                      \
    T v;                                                                     \
    memcpy(&v, j->bits, sizeof v);                                           \
    int64_t s = b->s0[0];                                                    \
    for (int64_t q = 0; q < b->n2; q++)                                      \
      for (int64_t r = 0; r < b->n1; r++) {                                  \
        T *d = (T *)at(j->a, 0, b->at[0] + q * b->s2[0] + r * b->s1[0]);     \
        if (s == 1)                                                          \
          for (int64_t i = 0; i < b->n0; i++) d[i] = v;                      \
        else                                                                 \
          for (int64_t i = 0; i < b->n0; i++) d[i * s] = v;                  \
      }                                                                      \
  }

FILL(1, uint8_t)
FILL(2, uint16_t)
FILL(4, uint32_t)
FILL(8, uint64_t)
FILL(16, w16)

static nx_cpu_block_fn fill_of(int bytes) {
  switch (bytes) {
    case 1: return fill_1;
    case 2: return fill_2;
    case 4: return fill_4;
    case 8: return fill_8;
    case 16: return fill_16;
    default: return NULL;
  }
}

/* Reads the [n] operands [in] through the door and walks them with [f]. */
static value run(int n, const nx_operand *in, nx_cpu_block_fn f, job *j) {
  nx_array a[NX_MAX_OPERANDS];
  nx_loop l;
  int e = nx_read(n, in, a);
  if (e) return Val_int(e);
  j->a = a;
  j->n = n;
  if (!(e = nx_coalesce(n, a, &l))) nx_cpu_walk(n, a, &l, MOST, f, j);
  nx_done(n, a);
  return Val_int(e);
}

/* Entries */

/* nx_spec.h's code of an Prog.op2 value. */
static int op2_code(value k) {
  return Tag_val(k) == 0 ? Int_val(Field(k, 0))
                         : NX_OP2_EQUAL + Int_val(Field(k, 0));
}

value nx_cpu_apply2(value k, value vd, value vx, value vy) {
  int x = nx_array_dtype(vx), c = op2_code(k);
  row2 f = rows2[c][x];
  if (f == NULL) return Val_int(NX_DECLINED);
  int d = c >= NX_OP2_EQUAL ? NX_BOOL : x;
  nx_operand in[3] = {{vd, d, 1}, {vx, x, 0}, {vy, x, 0}};
  job j = {.f2 = f};
  return run(3, in, block, &j);
}

value nx_cpu_apply3(value k, value vd, value vc, value vx, value vy) {
  int x = nx_array_dtype(vx), c = x;
  row3 f;
  if (Int_val(k) == NX_OP3_WHERE) {
    int bits = nx_dtype_row_of(x).bits;
    f = bits < 8 || nx_array_dtype(vc) == NX_BIT ? NULL : where_of(bits / 8);
    c = NX_BOOL;
  } else
    f = fmas[x];
  if (f == NULL) return Val_int(NX_DECLINED);
  nx_operand in[4] = {{vd, x, 1}, {vc, c, 0}, {vx, x, 0}, {vy, x, 0}};
  job j = {.f3 = f};
  return run(4, in, block, &j);
}

value nx_cpu_fill(value vbits, value vd) {
  CAMLparam2(vbits, vd);
  int d = nx_array_dtype(vd), bits = nx_dtype_row_of(d).bits;
  if (bits < 8) CAMLreturn(Val_int(NX_DECLINED));
  if (caml_string_length(vbits) != (size_t)(bits / 8))
    CAMLreturn(Val_int(NX_DTYPE));
  /* The door may run OCaml code, which may move the string. */
  uint8_t element[16];
  memcpy(element, String_val(vbits), bits / 8);
  nx_operand in[1] = {{vd, d, 1}};
  job j = {.bits = element};
  CAMLreturn(run(1, in, fill_of(bits / 8), &j));
}

/* Iota: each element is its index along axis [axis], cast to the dtype. The
   walk merges axes, so iota counts its own: an odometer over the axes
   outside the innermost, rows along it. */
#define IOTA(D, T)                                                           \
  static void iota_##D(const nx_array *a, int axis) {                        \
    int r = a->rank;                                                         \
    int64_t idx[NX_MAX_RANK] = {0}, n = a->dim[r - 1], s = a->dim[2 * r - 1]; \
    for (int i = 0; i < r; i++)                                              \
      if (a->dim[i] == 0) return;                                            \
    for (;;) {                                                               \
      int64_t p = a->offset;                                                 \
      for (int i = 0; i < r - 1; i++) p += idx[i] * a->dim[r + i];           \
      T *d = (T *)a->base + p;                                               \
      if (axis == r - 1)                                                     \
        for (int64_t i = 0; i < n; i++) d[i * s] = (T)i;                     \
      else                                                                   \
        for (int64_t i = 0; i < n; i++) d[i * s] = (T)idx[axis];             \
      int i = r - 2;                                                         \
      while (i >= 0 && ++idx[i] == a->dim[i]) idx[i--] = 0;                  \
      if (i < 0) return;                                                     \
    }                                                                        \
  }

IOTA(f32, float)
IOTA(f64, double)
IOTA(i8, int8_t)
IOTA(i16, int16_t)
IOTA(i32, int32_t)
IOTA(i64, int64_t)
IOTA(u8, uint8_t)
IOTA(u16, uint16_t)
IOTA(u32, uint32_t)
IOTA(u64, uint64_t)

typedef void (*iota_fn)(const nx_array *a, int axis);

static const iota_fn iotas[NX_DTYPE_COUNT] = {
    [NX_FLOAT32] = iota_f32, [NX_FLOAT64] = iota_f64, [NX_INT8] = iota_i8,
    [NX_INT16] = iota_i16,   [NX_INT32] = iota_i32,   [NX_INT64] = iota_i64,
    [NX_UINT8] = iota_u8,    [NX_UINT16] = iota_u16,  [NX_UINT32] = iota_u32,
    [NX_UINT64] = iota_u64,
};

value nx_cpu_iota(value vaxis, value vd) {
  int d = nx_array_dtype(vd), axis = Int_val(vaxis);
  iota_fn f = iotas[d];
  if (f == NULL) return Val_int(NX_DECLINED);
  nx_operand in[1] = {{vd, d, 1}};
  nx_array a[1];
  int e = nx_read(1, in, a);
  if (e) return Val_int(e);
  if (axis < 0 || axis >= a->rank)
    e = NX_SHAPE;
  else
    f(a, axis);
  nx_done(1, a);
  return Val_int(e);
}
