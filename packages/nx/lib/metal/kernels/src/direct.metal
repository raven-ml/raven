/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/


/* Direct maps: a map whose tape is a direct form (kernels.h,
   NX_LOOP_DIRECT) runs as loop_map_direct, or loop_map_direct_tiled
   where it takes tiles, 4 elements a thread in registers, with no
   planes and no tape: up to three loads, each element as it is, a truth
   or decoded from a narrow float, one kind, a Where or neither, the
   result as it is, wrapped or encoded, and one store. Its
   loads and its store, over arrays in registers, place and read elements
   as the tape's (tape.h), and each of its steps calls the function the
   tape's instruction does, so its bits are the tape's.

   A unit apart from loop.metal, so that its calls of the kinds leave the
   tape kernels' inlining as it is. It holds no planes: its group memory
   holds its stages alone (tape.h, Tiles), or 16 bytes where it has none,
   since a kernel's argument is bound, at a length of a multiple of 16.
   It states no bound on its groups'
   threads, as loop.metal's kernels do not.

   Its table is a direct map's (kernels.h, NX_DIRECT_SLOTS): each
   operand's descriptor in a slot of its own and its buffer in the
   parameters' word of that slot, so that a thread finds each at a fixed
   place, not through the loads' table. */

#include "tape.h"

static_assert(NX_DIRECT_SLOTS + 4 * NX_DIRECT_STRIDE + 4 <= NX_LOOP_HOLDS,
              "a direct map's slots and tables are held");
static_assert(4 <= NX_LOOP_BUFFERS, "a direct map's slots have buffers");

/* Slot [k]'s descriptor: 0 the destination's, 1 + j operand j's. */
__attribute__((always_inline)) static inline constant u64 *
direct_slot(thread const loop &c, uint k) {
  return c.w->held + NX_DIRECT_SLOTS + k * NX_DIRECT_STRIDE;
}

/* Whether element [p0] of slot [k]'s buffer, of [size] bytes, lies on
   [bytes]: tape.h's on, of the slot's buffer word. */
__attribute__((always_inline)) static inline bool
direct_on(thread const loop &c, uint k, uint p0, uint size, uint bytes) {
  const u64 a = ((constant u64 *)c.w)[NX_LOOP_BUFFER + k];
  return ((a + u64(p0) * size) & (bytes - 1)) == 0;
}

#define EACH4 _Pragma("unroll") for (uint v = 0; v < 4; v++)
#define ROLLED4                                                                \
  _Pragma("clang loop unroll(disable)") for (uint v = 0; v < 4; v++)

/* An operand's held words [y] as the op takes them, by [how]. */
__attribute__((always_inline)) static inline void
direct_in(uint how, thread uint *y) {
  switch (how) {
  case NX_DIRECT_TRUTH: EACH4 y[v] = y[v] != 0 ? 1u : 0u; return;
  case NX_DIRECT_BF16:
    EACH4 y[v] = as_type<uint>(nx_bf16_to_float(ushort(y[v])));
    return;
  case NX_DIRECT_F16:
    EACH4 y[v] = as_type<uint>(nx_metal_f16_to_float(ushort(y[v])));
    return;
  }
}

/* An 8-bit operand's held words [y] as the op takes them, by [how]: as
   they are or as truths, the hows of a map of bytes (NX_LOOP_BITS8). */
__attribute__((always_inline)) static inline void
direct_truth(uint how, thread uint *y) {
  if (how == NX_DIRECT_TRUTH) EACH4 y[v] = y[v] != 0 ? 1u : 0u;
}

/* The op's results [y] as the store takes them, by [how]. */
__attribute__((always_inline)) static inline void
direct_out(uint how, int dt, thread uint *y) {
  switch (how) {
  case NX_DIRECT_WRAP: EACH4 y[v] = uint(of_raw(dt, y[v])); return;
  case NX_DIRECT_BF16:
    EACH4 y[v] = nx_float_to_bf16(as_type<float>(y[v]));
    return;
  case NX_DIRECT_F16:
    EACH4 y[v] = nx_metal_float_to_f16(as_type<float>(y[v]));
    return;
  }
}

#define DK1(S, T, U, n)                                                        \
  case (CT_##S << 7) + NX_OP1_##U:                                             \
    EACH4 r[v] = as_type<uint>(nx_##n##_##S(as_type<T>(a[v])));               \
    return;
#define DK2(S, T, U, n)                                                        \
  case (CT_##S << 7) + NX_LOOP_BINARY + NX_OP2_##U:                            \
    EACH4 r[v] =                                                               \
        as_type<uint>(nx_##n##_##S(as_type<T>(a[v]), as_type<T>(b[v])));       \
    return;
#define DKC(S, T, U, n)                                                        \
  case (CT_##S << 7) + NX_LOOP_BINARY + NX_OP2_##U:                            \
    EACH4 r[v] =                                                               \
        nx_##n##_##S(as_type<T>(a[v]), as_type<T>(b[v])) != 0 ? 1u : 0u;       \
    return;

/* A long kind's elements one at a time, in a rolled loop, as the tape
   runs its costly kinds (tape.h, S1 and S2): 4 copies of a long function
   would set the registers of the whole kernel, every op's. */
#define DS1(S, T, U, n)                                                        \
  case (CT_##S << 7) + NX_OP1_##U:                                             \
    ROLLED4 r[v] = as_type<uint>(nx_##n##_##S(as_type<T>(a[v])));             \
    return;
#define DS2(S, T, U, n)                                                        \
  case (CT_##S << 7) + NX_LOOP_BINARY + NX_OP2_##U:                            \
    ROLLED4 r[v] =                                                             \
        as_type<uint>(nx_##n##_##S(as_type<T>(a[v]), as_type<T>(b[v])));       \
    return;
#define DK3(S, T, U, n)                                                        \
  case (CT_##S << 7) + NX_LOOP_TERNARY + NX_OP3_##U:                           \
    EACH4 r[v] = as_type<uint>(                                                \
        nx_##n##_##S(as_type<T>(a[v]), as_type<T>(b[v]), as_type<T>(e[v])));   \
    return;
/* REAL1S in two, as measured on an M1 Max: a kind runs unrolled where
   that gains on its own map and costs no other kind's. Rolling all of
   REAL1S takes exp over 1M elements from 2.21 ms to 2.40. Unrolled, expm1
   and log1p over 1M take 2.75 and 3.25 ms against 2.59 and 3.06 rolled,
   and t_add 8192^2 30.6 ms against 29.9; fdiv takes 2.43 ms against 1.95,
   so REAL2S stays rolled. Each of REAL1S_SHORT rolled alone takes longer
   over 1M, and no other kind's map gains: recip 2.34 ms against 2.11,
   sqrt 2.37 against 2.19, exp2 2.39 against 2.27, log 2.56 against 2.45,
   log2 2.73 against 2.57. The narrow fmas run unrolled: rolled, a
   bfloat16 fma over 2M elements takes 5.17 ms against 3.54, and t_add
   34.3 ms. */
#define REAL1S_SHORT(X, S, T)                                                  \
  X(S, T, RECIP, recip) X(S, T, SQRT, sqrt) X(S, T, EXP, exp)                  \
  X(S, T, EXP2, exp2) X(S, T, LOG, log) X(S, T, LOG2, log2)
#define REAL1S_LONG(X, S, T)                                                   \
  X(S, T, LOG1P, log1p) X(S, T, EXPM1, expm1) X(S, T, SIN, sin)                \
  X(S, T, COS, cos) X(S, T, TAN, tan) X(S, T, ASIN, asin)                      \
  X(S, T, ACOS, acos) X(S, T, ATAN, atan) X(S, T, SINH, sinh)                  \
  X(S, T, COSH, cosh) X(S, T, TANH, tanh) X(S, T, ERF, erf)
/* The set of a list's kinds, a bit a code: the two halves' sets must make
   REAL1S's, and a kind in both is a duplicate case label below. */
static_assert(NX_OP1_COUNT <= 64, "an op1's code is a bit of 64");
#define DIRECT_BIT(S, T, U, n) | (1ull << NX_OP1_##U)
static_assert((0 REAL1S(DIRECT_BIT, f32, float)) ==
                  (0 REAL1S_SHORT(DIRECT_BIT, f32, float)
                       REAL1S_LONG(DIRECT_BIT, f32, float)),
              "REAL1S_SHORT and REAL1S_LONG split REAL1S");

#define DIRECT_REAL(S, T)                                                      \
  REAL1F(DK1, S, T) REAL1S_SHORT(DK1, S, T) REAL1S_LONG(DS1, S, T)            \
  REAL2F(DK2, S, T) REAL2S(DS2, S, T) COMPARE(DKC, S, T) DK3(S, T, FMA, fma)
#define DIRECT_INT(S, T)                                                       \
  INT1F(DK1, S, T) INT1S(DS1, S, T) INT2F(DK2, S, T) INT2S(DS2, S, T)          \
  COMPARE(DKC, S, T) DK3(S, T, FMA, fma)

/* The kind [op] of the word kinds (tape.h's word_kind) of [a], [b] and
   [e] into [r]. */
__attribute__((always_inline)) static inline void
direct_kind(int op, thread const uint *a, thread const uint *b,
            thread const uint *e, thread uint *r) {
  switch (op) {
    DIRECT_REAL(f32, float) DIRECT_INT(i32, int) DIRECT_INT(u32, uint)
  case (NX_LOOP_F32 << 7) + NX_LOOP_FMA_NARROW:
    EACH4 r[v] = as_type<uint>(nx_fma_narrow(
        as_type<float>(a[v]), as_type<float>(b[v]), as_type<float>(e[v])));
    return;
  case (NX_LOOP_F32 << 7) + NX_LOOP_FMA_BF16:
    EACH4 r[v] = as_type<uint>(nx_fma_bf16(
        as_type<float>(a[v]), as_type<float>(b[v]), as_type<float>(e[v])));
    return;
  }
}

/* Direct32's kinds in two kernels (kernels.h, NX_DIRECT_LONG_REAL1): the
   cheap kinds, the narrow fmas among them, and the long ones. */
#define DIRECT_REAL_CHEAP(S, T)                                                \
  REAL1F(DK1, S, T) REAL2F(DK2, S, T) COMPARE(DKC, S, T) DK3(S, T, FMA, fma)
#define DIRECT_INT_CHEAP(S, T)                                                 \
  INT1F(DK1, S, T) INT2F(DK2, S, T) COMPARE(DKC, S, T) DK3(S, T, FMA, fma)
#define DIRECT_REAL_LONG(S, T)                                                 \
  REAL1S_SHORT(DK1, S, T) REAL1S_LONG(DS1, S, T) REAL2S(DS2, S, T)
#define DIRECT_INT_LONG(S, T) INT1S(DS1, S, T) INT2S(DS2, S, T)

#define DIRECT_BIT2(S, T, U, n) | (1u << NX_OP2_##U)
static_assert((0 REAL1S(DIRECT_BIT, f32, float)) == NX_DIRECT_LONG_REAL1,
              "NX_DIRECT_LONG_REAL1 is REAL1S");
static_assert((0 REAL2S(DIRECT_BIT2, f32, float)) == NX_DIRECT_LONG_REAL2,
              "NX_DIRECT_LONG_REAL2 is REAL2S");
static_assert((0 INT1S(DIRECT_BIT, i32, int)) == NX_DIRECT_LONG_INT1,
              "NX_DIRECT_LONG_INT1 is INT1S");
static_assert((0 INT2S(DIRECT_BIT2, i32, int)) == NX_DIRECT_LONG_INT2,
              "NX_DIRECT_LONG_INT2 is INT2S");

__attribute__((always_inline)) static inline void
direct_kind_cheap(int op, thread const uint *a, thread const uint *b,
                  thread const uint *e, thread uint *r) {
  switch (op) {
    DIRECT_REAL_CHEAP(f32, float)
    DIRECT_INT_CHEAP(i32, int) DIRECT_INT_CHEAP(u32, uint)
  case (NX_LOOP_F32 << 7) + NX_LOOP_FMA_NARROW:
    EACH4 r[v] = as_type<uint>(nx_fma_narrow(
        as_type<float>(a[v]), as_type<float>(b[v]), as_type<float>(e[v])));
    return;
  case (NX_LOOP_F32 << 7) + NX_LOOP_FMA_BF16:
    EACH4 r[v] = as_type<uint>(nx_fma_bf16(
        as_type<float>(a[v]), as_type<float>(b[v]), as_type<float>(e[v])));
    return;
  }
}

__attribute__((always_inline)) static inline void
direct_kind_long(int op, thread const uint *a, thread const uint *b,
                 thread const uint *e, thread uint *r) {
  switch (op) {
    DIRECT_REAL_LONG(f32, float)
    DIRECT_INT_LONG(i32, int) DIRECT_INT_LONG(u32, uint)
  }
}

/* A quad's elements of [d] as T into [y]: one vector where the quad lies
   on its size, as the tape's read_quad (tape.h). */
template <class T, class T4>
__attribute__((always_inline)) static inline void
direct_quad_in(thread const loop &c, uint k, lanes x, thread uint *y) {
  constant u64 *d = direct_slot(c, k);
  device const T *e = (device const T *)buffer_at(c, int(k));
  const uint p0 = uint(place(c, d, x.L0));
  const long st = last_stride(c, d);
  if (st == 0) {
    const uint z = x.n > 0 ? uint(e[p0]) : 0u;
    EACH4 y[v] = z;
    return;
  }
  if (st == 1 && x.n == 4 && direct_on(c, k, p0, sizeof(T), 4 * sizeof(T))) {
    const T4 q = *(device const T4 *)(e + p0);
    y[0] = uint(q.x);
    y[1] = uint(q.y);
    y[2] = uint(q.z);
    y[3] = uint(q.w);
    return;
  }
  EACH4 y[v] = v < x.n ? uint(e[p0 + uint(long(v) * st)]) : 0u;
}

/* The positions of plain operand [d]'s elements at the thread's 4
   indices, into [at], as the tape's positions (tape.h). */
__attribute__((always_inline)) static inline void
direct_positions(thread const loop &c, constant u64 *d, lanes x,
                 thread u64 *at) {
  const u64 p0 = place(c, d, x.L0);
  const int rank = int(hd(c, NX_LOOP_RANK));
  const u64 span = u64(x.n > 0 ? x.n - 1 : 0) * x.step;
  if (rank > 0 && coordinate(c, x.L0, rank - 1) + span < c.t[rank - 1]) {
    const uint st = uint(x.step * d[2 + rank - 1]);
    EACH4 at[v] = uint(p0) + v * st;
    return;
  }
  EACH4 at[v] = v < x.n ? place(c, d, index_of(x, v)) : 0;
}

/* A direct map's codec: none, or the narrow float every load and its
   destination hold (NX_LOOP_BITS16), by its how (kernels.h,
   NX_DIRECT_BF16). */
enum { CODEC_NONE = 0 };

/* The bits of the plain operand of slot [k]'s elements, of whole bytes
   of at most 32 bits, at the thread's indices, into [y], an element of
   no index taking zeros: the tape's load (tape.h) without its planes. A
   kernel of a codec [C] reads 16 bits, in quads; one of bytes ([b8]) 8
   bits, in quads. */
template <int C>
__attribute__((always_inline)) static inline void
direct_bits(thread const loop &c, uint k, lanes x, thread uint *y) {
  constant u64 *d = direct_slot(c, k);
  const uint bits =
      c.b32 ? 32 : C ? 16 : c.b8 ? 8 : bits_of(int(d[0] & 0xff));
  if (c.tiled && stage_of(d) != 0) {
    staged_bits(c, stage_of(d), y);
    return;
  }
  if (c.b32 || C || c.b8 || x.quad) {
    switch (bits) {
    case 32: direct_quad_in<uint, uint4>(c, k, x, y); return;
    case 16: direct_quad_in<ushort, ushort4>(c, k, x, y); return;
    default: direct_quad_in<uchar, uchar4>(c, k, x, y); return;
    }
  }
  device const uchar *b = buffer_at(c, int(k));
  u64 at[4];
  direct_positions(c, d, x, at);
  switch (bits) {
  case 32: EACH4 y[v] = v < x.n ? AT(device const uint, b, at[v]) : 0u; return;
  case 16:
    EACH4 y[v] = v < x.n ? uint(AT(device const ushort, b, at[v])) : 0u;
    return;
  default:
    EACH4 y[v] = v < x.n ? uint(AT(device const uchar, b, at[v])) : 0u;
  }
}

/* The form's operand [j]: its load's held words into [y]. */
template <int C>
__attribute__((always_inline)) static inline void
direct_read(thread const loop &c, lanes x, uint j, thread uint *y) {
  direct_bits<C>(c, 1 + j, x, y);
  if (c.b32 || C) return;
  const int dt = int(direct_slot(c, 1 + j)[0] & 0xff);
  if (c.b8) {
    if (dt == NX_INT8) EACH4 y[v] = uint(int(char(uchar(y[v]))));
    return;
  }
  EACH4 y[v] = held_word(dt, y[v]);
}

/* A quad's words [y] into [d] as T: one vector where the quad lies on
   its size, as the tape's write_quad (tape.h). */
template <class T, class T4>
__attribute__((always_inline)) static inline void
direct_quad_out(thread const loop &c, lanes x, thread const uint *y) {
  constant u64 *d = direct_slot(c, 0);
  device T *e = (device T *)buffer_at(c, 0);
  const uint p0 = uint(place(c, d, x.L0));
  const long st = last_stride(c, d);
  if (st == 1 && x.n == 4 && direct_on(c, 0, p0, sizeof(T), 4 * sizeof(T))) {
    *(device T4 *)(e + p0) = T4(T(y[0]), T(y[1]), T(y[2]), T(y[3]));
    return;
  }
  EACH4 if (v < x.n) e[p0 + uint(long(v) * st)] = T(y[v]);
}

/* The words [y] at the thread's indices into the plain destination, of
   whole bytes of at most 32 bits: the tape's store without its planes. */
template <int C>
__attribute__((always_inline)) static inline void
direct_write(thread const loop &c, lanes x, thread const uint *y) {
  constant u64 *d = direct_slot(c, 0);
  const uint bits =
      c.b32 ? 32 : C ? 16 : c.b8 ? 8 : bits_of(int(d[0] & 0xff));
  if (c.b32 || C || c.b8 || x.quad) {
    switch (bits) {
    case 32: direct_quad_out<uint, uint4>(c, x, y); return;
    case 16: direct_quad_out<ushort, ushort4>(c, x, y); return;
    default: direct_quad_out<uchar, uchar4>(c, x, y); return;
    }
  }
  device uchar *b = buffer_at(c, 0);
  switch (bits) {
  case 32:
    EACH4 if (v < x.n) AT(device uint, b, place(c, d, index_of(x, v))) = y[v];
    return;
  case 16:
    EACH4 if (v < x.n) AT(device ushort, b, place(c, d, index_of(x, v))) =
        ushort(y[v]);
    return;
  default:
    EACH4 if (v < x.n) AT(device uchar, b, place(c, d, index_of(x, v))) =
        uchar(y[v]);
  }
}

__attribute__((always_inline)) static inline uint
direct_how(u64 f, uint j) {
  return uint(f >> (NX_DIRECT_OPERANDS + 4 * j + 2)) & 3;
}

/* The kinds a direct kernel holds: every kind, or direct32's cheap
   kinds or its long ones. */
enum { DIRECT_ALL, DIRECT_CHEAP, DIRECT_LONG };

/* A codec's float32 of a code [y], and its code of a float32: the tape's
   decode and encode (tape.h). */
template <int C> static inline uint codec_in(uint y) {
  return as_type<uint>(C == NX_DIRECT_BF16 ? nx_bf16_to_float(ushort(y))
                                           : nx_metal_f16_to_float(ushort(y)));
}
template <int C> static inline uint codec_out(uint y) {
  const float f = as_type<float>(y);
  return C == NX_DIRECT_BF16 ? uint(nx_float_to_bf16(f))
                             : uint(nx_metal_float_to_f16(f));
}

/* A direct map's thread: inlined into each kernel, so that the kernel's
   constant view (loop_of), its kinds [K] and its codec [C] drop the
   paths it does not take. */
template <int K, int C = CODEC_NONE>
__attribute__((always_inline)) static inline void
direct_map(thread const loop &c, uint gid) {
  const u64 first = hd(c, NX_LOOP_FIRST);
  const u64 mode = hd(c, NX_LOOP_MODE);
  const bool quad = (mode & NX_LOOP_QUADS) != 0;
  const u64 f = hd(c, NX_LOOP_DIRECT);
  const uint m = uint(f >> NX_DIRECT_COUNT) & 3;
  uint a[4], b[4] = {0, 0, 0, 0}, e[4] = {0, 0, 0, 0}, r[4];
  lanes x;
  if (c.tiled) {
    const tile t = tile_of(c, gid);
    if (!t.live) return;
    stage(c, t);
    x = tile_lanes(c, t);
  } else {
    /* Here alone: above the branch, the group's first index would be
       shared with tile_of's, and move in the tiled kernels' code. */
    const uint n = uint(hd(c, NX_LOOP_NUMEL));
    const uint base = (uint(first) + gid) * c.E;
    x = c.b32 || C || c.b8 || quad ? lanes32_of(base, 4 * c.me, 1, n, 4)
                                   : lanes32_of(base, c.me, TB, n, 4);
  }
  x.quad = c.b32 || C || c.b8 || quad;
  if (x.n == 0) return;
  direct_read<C>(c, x, 0, a);
  if (m > 1) direct_read<C>(c, x, 1, b);
  if (m > 2) direct_read<C>(c, x, 2, e);
  if (C) {
    EACH4 a[v] = codec_in<C>(a[v]);
    if (m > 1) EACH4 b[v] = codec_in<C>(b[v]);
    if (m > 2) EACH4 e[v] = codec_in<C>(e[v]);
  } else if (c.b8) {
    direct_truth(direct_how(f, 0), a);
    direct_truth(direct_how(f, 1), b);
    direct_truth(direct_how(f, 2), e);
  } else if (!c.b32) {
    direct_in(direct_how(f, 0), a);
    direct_in(direct_how(f, 1), b);
    direct_in(direct_how(f, 2), e);
  }
  const int op = int(f & 0xffff);
  if (K == DIRECT_LONG) direct_kind_long(op, a, b, e, r);
  else if (op == NX_LOOP_WHERE1) EACH4 r[v] = a[v] != 0 ? b[v] : e[v];
  else if (op == NX_LOOP_CAST) EACH4 r[v] = a[v];
  else if (K == DIRECT_CHEAP) direct_kind_cheap(op, a, b, e, r);
  else direct_kind(op, a, b, e, r);
  if (C) EACH4 r[v] = codec_out<C>(r[v]);
  else if (!c.b32 && !c.b8)
    direct_out(uint(f >> NX_DIRECT_RESULT) & 3,
               int(f >> NX_DIRECT_WRAP_DT) & 0xff, r);
  direct_write<C>(c, x, r);
}

/* A direct map's view: its table is held (kernels.h, NX_DIRECT_SLOTS),
   so that its walk is read at a fixed place. */
__attribute__((always_inline)) static inline loop
direct_of(constant loop_words &w, threadgroup uint *pl, uint me) {
  loop c = loop_of(w, pl, me, 4, false);
  c.t = w.held;
  return c;
}

[[kernel]] void loop_map_direct(constant loop_words &w [[buffer(0)]],
                                threadgroup uint *pl [[threadgroup(0)]],
                                uint gid [[threadgroup_position_in_grid]],
                                uint me [[thread_index_in_threadgroup]]) {
  direct_map<DIRECT_ALL>(direct_of(w, pl, me), gid);
}

/* A direct map of tiles: a kernel of its own, so that its path costs no
   other direct map. */
[[kernel]] void
loop_map_direct_tiled(constant loop_words &w [[buffer(0)]],
                      threadgroup uint *pl [[threadgroup(0)]],
                      uint gid [[threadgroup_position_in_grid]],
                      uint me [[thread_index_in_threadgroup]]) {
  loop c = direct_of(w, pl, me);
  c.tiled = true;
  direct_map<DIRECT_ALL>(c, gid);
}

/* Direct maps of 32-bit lanes (NX_LOOP_BITS32): each load and the
   destination 32 bits wide, each element taken and the result given as
   it is, the threads in quads. Kernels of their own, since the widths'
   switches and the conversions cost a kernel that holds them about twice
   a 32-bit map's time, though no element takes them. Their kinds split by
   cost (kernels.h, NX_DIRECT_LONG_REAL1): on an M1 Max a kernel of every
   kind adds 1M float32 in 21.09 us and takes their exp in 21.36, one of
   the cheap kinds adds them in 20.21, and one of the long kinds takes
   their exp in 19.67.

   Their positions are the other direct kernels': the loop is not wide,
   so a position lies below 2^32, and place, direct_quad_in and the
   tiles' positions compute it in wrapping 32-bit unsigned words, which
   index the operand's elements, so a position in [2^31, 2^32) is read
   as the unsigned position it is. */
[[kernel]] void
loop_map_direct32(constant loop_words &w [[buffer(0)]],
                  threadgroup uint *pl [[threadgroup(0)]],
                  uint gid [[threadgroup_position_in_grid]],
                  uint me [[thread_index_in_threadgroup]]) {
  loop c = direct_of(w, pl, me);
  c.b32 = true;
  direct_map<DIRECT_CHEAP>(c, gid);
}

[[kernel]] void
loop_map_direct32_tiled(constant loop_words &w [[buffer(0)]],
                        threadgroup uint *pl [[threadgroup(0)]],
                        uint gid [[threadgroup_position_in_grid]],
                        uint me [[thread_index_in_threadgroup]]) {
  loop c = direct_of(w, pl, me);
  c.tiled = true;
  c.b32 = true;
  direct_map<DIRECT_CHEAP>(c, gid);
}

[[kernel]] void
loop_map_direct32_long(constant loop_words &w [[buffer(0)]],
                       threadgroup uint *pl [[threadgroup(0)]],
                       uint gid [[threadgroup_position_in_grid]],
                       uint me [[thread_index_in_threadgroup]]) {
  loop c = direct_of(w, pl, me);
  c.b32 = true;
  direct_map<DIRECT_LONG>(c, gid);
}

[[kernel]] void
loop_map_direct32_long_tiled(constant loop_words &w [[buffer(0)]],
                             threadgroup uint *pl [[threadgroup(0)]],
                             uint gid [[threadgroup_position_in_grid]],
                             uint me [[thread_index_in_threadgroup]]) {
  loop c = direct_of(w, pl, me);
  c.tiled = true;
  c.b32 = true;
  direct_map<DIRECT_LONG>(c, gid);
}

/* Direct maps of one narrow float (NX_LOOP_BITS16): each load and the
   destination a bfloat16's or a float16's, each element decoded and the
   result encoded by that float's codec, the op at float32, the threads
   in quads. A kernel for each codec, as direct32's for 32-bit lanes: a
   codec chosen at run time keeps the generic body's cost. Their kinds
   split as direct32's. */
#define DIRECT16(name, K, C, T)                                                \
  [[kernel]] void name(constant loop_words &w [[buffer(0)]],                   \
                       threadgroup uint *pl [[threadgroup(0)]],                \
                       uint gid [[threadgroup_position_in_grid]],              \
                       uint me [[thread_index_in_threadgroup]]) {              \
    loop c = direct_of(w, pl, me);                                             \
    c.tiled = T;                                                               \
    direct_map<K, C>(c, gid);                                                  \
  }
DIRECT16(loop_map_direct16_bf16, DIRECT_CHEAP, NX_DIRECT_BF16, false)
DIRECT16(loop_map_direct16_bf16_tiled, DIRECT_CHEAP, NX_DIRECT_BF16, true)
DIRECT16(loop_map_direct16_bf16_long, DIRECT_LONG, NX_DIRECT_BF16, false)
DIRECT16(loop_map_direct16_bf16_long_tiled, DIRECT_LONG, NX_DIRECT_BF16, true)
DIRECT16(loop_map_direct16_f16, DIRECT_CHEAP, NX_DIRECT_F16, false)
DIRECT16(loop_map_direct16_f16_tiled, DIRECT_CHEAP, NX_DIRECT_F16, true)
DIRECT16(loop_map_direct16_f16_long, DIRECT_LONG, NX_DIRECT_F16, false)
DIRECT16(loop_map_direct16_f16_long_tiled, DIRECT_LONG, NX_DIRECT_F16, true)

/* Direct maps of bytes (NX_LOOP_BITS8): each load and the destination 8
   bits wide, each element taken as it is, an int8's sign-extended, or as
   a truth, one of direct32's cheap kinds at 32 bits, and the result's low
   byte stored, which is its wrap into the destination's dtype; the
   threads in quads. Kernels of their own, as direct32's: in the generic
   body, whose widths and conversions are chosen at run time, an add of
   two uint8 tiles of 8192 x 8192 takes 2.89 ms on an M1 Max, where a copy
   of its bytes takes 0.53. A long kind keeps loop_map_direct. */
[[kernel]] void
loop_map_direct8(constant loop_words &w [[buffer(0)]],
                 threadgroup uint *pl [[threadgroup(0)]],
                 uint gid [[threadgroup_position_in_grid]],
                 uint me [[thread_index_in_threadgroup]]) {
  loop c = direct_of(w, pl, me);
  c.b8 = true;
  direct_map<DIRECT_CHEAP>(c, gid);
}

[[kernel]] void
loop_map_direct8_tiled(constant loop_words &w [[buffer(0)]],
                       threadgroup uint *pl [[threadgroup(0)]],
                       uint gid [[threadgroup_position_in_grid]],
                       uint me [[thread_index_in_threadgroup]]) {
  loop c = direct_of(w, pl, me);
  c.tiled = true;
  c.b8 = true;
  direct_map<DIRECT_CHEAP>(c, gid);
}

/* Casts by name

   A map of one plain load cast into a dtype of 8, 16, 32 or 64 bits but
   complex64 runs as the kernel of its source's class and its target's
   store width (kernels.h, NX_METAL_CAST_KERNELS), 4 elements a thread,
   lanes as loop_map_direct's. The kernel's load and store widths are
   constants, and the target, read from the destination's descriptor, is
   uniform over the launch. A source of a class of its own computes
   tape.h's cast in a branch a target, each of constants. A byte or
   nibble source takes its result from a table of its codes: one load
   where a decode and an encode cost dozens of operations a code. Where
   the host finds both operands contiguous over the walk's one axis, each
   from an element on its quad's size (NX_LOOP_FLAT), a quad's position
   is its first index past the operand's first position, and each quad
   one vector. */

static_assert(NX_CAST_BYTES >= NX_DTYPE_COUNT, "the classes' codes are no dtype's");

template <class E> struct quad_of;
template <> struct quad_of<uint> { typedef uint4 t; };
template <> struct quad_of<int> { typedef int4 t; };
template <> struct quad_of<ushort> { typedef ushort4 t; };
template <> struct quad_of<short> { typedef short4 t; };
template <> struct quad_of<uchar> { typedef uchar4 t; };
template <> struct quad_of<char> { typedef char4 t; };
template <> struct quad_of<ulong> { typedef ulong4 t; };
template <> struct quad_of<long> { typedef long4 t; };

/* The store of a width of bits. */
template <int W> struct store_of;
template <> struct store_of<8> { typedef uchar t; };
template <> struct store_of<16> { typedef ushort t; };
template <> struct store_of<32> { typedef uint t; };
template <> struct store_of<64> { typedef ulong t; };

/* Code tables

   A byte or nibble source's codes, each cast into each target of a
   width, computed by the compiler by nx.cpu's rules (nx_dtype.h): each
   code's value is exact in float32, and in float16 and bfloat16 too, so
   a float target's bits follow from the value's fields; an integer
   source wraps into an integer target, a float source truncates toward
   zero and saturates, NaN to 0; into bool, nonzero, -0 false; into a
   float8, rounded to nearest even, e4m3fn saturating at 448 and e5m2
   past 57344 to infinity, a NaN quiet with its sign. For example
   e4m3fn's 0x01, 2^-9, is float16's 0x1800, and uint8's 200 is int8's
   -56. */

/* The float32 bits of code [c] of the float8 of [m] fraction bits and
   bias [bias], or of e4m3fn where [e4m3]. A NaN's are the decode's:
   quiet, with its sign and, for e5m2, its payload's top bit. */
constexpr uint f8_bits(uint c, uint m, uint bias, bool e4m3) {
  const uint sign = (c >> 7) << 31, q = c & 0x7f;
  const uint exp = q >> m, frac = q & ((1u << m) - 1);
  if (e4m3 && q == 0x7f) return sign | 0x7fc00000u;
  if (!e4m3 && exp == (1u << (7 - m)) - 1) {
    if (frac == 0) return sign | 0x7f800000u;
    return sign | 0x7fc00000u | ((frac & 1) << 21);
  }
  if (exp > 0) return sign | ((exp + 127 - bias) << 23) | (frac << (23 - m));
  if (frac == 0) return sign;
  /* A subnormal frac·2^(1-bias-m), its top bit at k. */
  uint k = 0;
  while ((2u << k) <= frac) k++;
  return sign | ((k + 128 - bias - m) << 23) | ((frac - (1u << k)) << (23 - k));
}

/* The float32 bits of e2m1fn's code [c], its low 4 bits: a 2-bit
   exponent of bias 1 and one fraction bit, no infinity and no NaN. */
constexpr uint e2m1_bits(uint c) {
  return f8_bits(((c & 8) << 4) | (c & 7), 1, 1, false);
}

/* The float32 bits of the integer [v], of magnitude below 2^24. */
constexpr uint f32_of_int(long v) {
  if (v == 0) return 0;
  const uint sign = v < 0 ? 0x80000000u : 0;
  const ulong a = ulong(v < 0 ? -v : v);
  uint k = 0;
  while ((2ul << k) <= a) k++;
  return sign | ((k + 127) << 23) | uint((a - (1ul << k)) << (23 - k));
}

/* float32 bits [u] in float16, for a value float16 holds exactly. */
constexpr ushort f16_of_exact(uint u) {
  const uint sign = (u >> 16) & 0x8000, a = u & 0x7fffffffu;
  if (a > 0x7f800000u) return ushort(sign | 0x7e00 | ((a >> 13) & 0x3ff));
  if (a == 0x7f800000u) return ushort(sign | 0x7c00);
  if (a == 0) return ushort(sign);
  const int e = int(a >> 23) - 112;
  const uint frac = (a & 0x7fffff) >> 13;
  if (e > 0) return ushort(sign | (uint(e) << 10) | frac);
  return ushort(sign | ((0x400 | frac) >> uint(1 - e)));
}

/* float32 bits [u] in bfloat16, for a value bfloat16 holds exactly. */
constexpr ushort bf16_of_exact(uint u) {
  const uint a = u & 0x7fffffffu;
  return ushort(a > 0x7f800000u ? (u >> 16) | 0x40 : u >> 16);
}

/* The magnitude code of float32's magnitude [u], not NaN, in the format
   of [m] fraction bits and bias [bias], to nearest, ties to even: past
   the largest finite code where [u] overflows (nx_mini_round). */
constexpr uint mini_round(uint u, uint m, uint bias) {
  if (u >= (128 - bias) << 23) {
    const uint odd = (u >> (23 - m)) & 1;
    return (u - ((127 - bias) << 23) + (1u << (22 - m)) - 1 + odd) >> (23 - m);
  }
  /* Below the least normal: multiples of the least subnormal,
     2^(1-bias-m), of a value M·2^(e-150). */
  const uint e = u >> 23;
  if (e == 0) return 0;
  const ulong M = (u & 0x7fffff) | 0x800000;
  const uint sh = 151 - bias - m - e;
  if (sh > 25) return 0;
  const ulong q = M >> sh, rest = M & ((1ul << sh) - 1), mid = 1ul << (sh - 1);
  return uint(q + ((rest > mid || (rest == mid && (q & 1))) ? 1 : 0));
}

constexpr uint e4m3_of(uint f) {
  const uint sign = (f >> 24) & 0x80, a = f & 0x7fffffffu;
  if (a > 0x7f800000u) return sign | 0x7f;
  const uint q = mini_round(a, 3, 7);
  return sign | (q > 0x7e ? 0x7e : q);
}

constexpr uint e5m2_of(uint f) {
  const uint sign = (f >> 24) & 0x80, a = f & 0x7fffffffu;
  if (a > 0x7f800000u) return sign | 0x7e | ((a >> 21) & 1);
  const uint q = mini_round(a, 2, 15);
  return sign | (q > 0x7c ? 0x7c : q);
}

/* A code's value: an integer [i], or float32 bits [f]; [f] holds an
   integer's value too. */
struct code_value {
  bool integer;
  long i;
  uint f;
};

constexpr code_value int_value(long i) { return {true, i, f32_of_int(i)}; }
constexpr code_value float_value(uint f) { return {false, 0, f}; }

/* Code [c] of byte or nibble source [s]. */
constexpr code_value value_of(int s, uint c) {
  switch (s) {
  case NX_UINT8: return int_value(long(c));
  case NX_INT8: return int_value(long(c) - (c >= 128 ? 256 : 0));
  case NX_BOOL: return int_value(c != 0 ? 1 : 0);
  case NX_FLOAT8_E4M3FN: return float_value(f8_bits(c, 3, 7, true));
  case NX_FLOAT8_E5M2: return float_value(f8_bits(c, 2, 15, false));
  case NX_UINT4: return int_value(long(c));
  case NX_INT4: return int_value(long(c) - (c >= 8 ? 16 : 0));
  default: return float_value(e2m1_bits(c));
  }
}

/* The truncation of float32 bits [f], not NaN, saturated to [lo, hi],
   as its bits. */
constexpr ulong trunc_sat(uint f, long lo, ulong hi) {
  const uint e = (f >> 23) & 0xff;
  if (e < 127) return 0;
  const ulong M = (f & 0x7fffff) | 0x800000;
  const bool big = e >= 127 + 64;
  const ulong mag = big ? ~0ul : e >= 150 ? M << (e - 150) : M >> (150 - e);
  if ((f >> 31) == 0) return big || mag > hi ? hi : mag;
  if (lo == 0) return 0;
  return big || mag > ulong(-(lo + 1)) + 1 ? ulong(lo) : ulong(-long(mag));
}

/* [v] into an integer of [bits] bits and bounds [lo, hi], as its bits:
   wrapped from an integer, truncated and saturated from a float, NaN to
   0. */
constexpr ulong int_into(code_value v, long lo, ulong hi, uint bits) {
  const ulong mask = bits == 64 ? ~0ul : (1ul << bits) - 1;
  if (v.integer) return ulong(v.i) & mask;
  if ((v.f & 0x7fffffffu) > 0x7f800000u) return 0;
  return trunc_sat(v.f, lo, hi) & mask;
}

/* [v] as the bits of target [t]. */
constexpr ulong encode(code_value v, int t) {
  switch (t) {
  case NX_FLOAT32: return v.f;
  case NX_FLOAT16: return f16_of_exact(v.f);
  case NX_BFLOAT16: return bf16_of_exact(v.f);
  case NX_FLOAT8_E4M3FN: return e4m3_of(v.f);
  case NX_FLOAT8_E5M2: return e5m2_of(v.f);
  case NX_BOOL: return v.integer ? v.i != 0 : (v.f & 0x7fffffffu) != 0;
  case NX_INT8: return int_into(v, -128, 127, 8);
  case NX_UINT8: return int_into(v, 0, 255, 8);
  case NX_INT16: return int_into(v, -32768, 32767, 16);
  case NX_UINT16: return int_into(v, 0, 65535, 16);
  case NX_INT32: return int_into(v, -2147483648l, 2147483647ul, 32);
  case NX_UINT32: return int_into(v, 0, 4294967295ul, 32);
  case NX_INT64:
    return int_into(v, -0x7fffffffffffffffl - 1, 0x7ffffffffffffffful, 64);
  default: return int_into(v, 0, ~0ul, 64); /* uint64 */
  }
}

#define CAST_ONE(x) +1
#define CAST_CODE(x) x,

/* The codes of a class's sources and of a width's targets, in order. */
constexpr int class_source(int k, int i) {
  const int bytes[] = {NX_CAST_BYTE_SOURCES(CAST_CODE)};
  const int nibbles[] = {NX_CAST_NIBBLE_SOURCES(CAST_CODE)};
  return k == NX_CAST_BYTES ? bytes[i] : nibbles[i];
}
constexpr int width_target(int w, int i) {
  const int t8[] = {NX_CAST_TARGETS_8(CAST_CODE)};
  const int t16[] = {NX_CAST_TARGETS_16(CAST_CODE)};
  const int t32[] = {NX_CAST_TARGETS_32(CAST_CODE)};
  const int t64[] = {NX_CAST_TARGETS_64(CAST_CODE)};
  return w == 8 ? t8[i] : w == 16 ? t16[i] : w == 32 ? t32[i] : t64[i];
}

template <int K> struct class_of;
template <> struct class_of<NX_CAST_BYTES> {
  enum { sources = 0 NX_CAST_BYTE_SOURCES(CAST_ONE), codes = 256 };
};
template <> struct class_of<NX_CAST_NIBBLES> {
  enum { sources = 0 NX_CAST_NIBBLE_SOURCES(CAST_ONE), codes = 16 };
};
template <int W> struct width_of;
template <> struct width_of<8> { enum { targets = 0 NX_CAST_TARGETS_8(CAST_ONE) }; };
template <> struct width_of<16> { enum { targets = 0 NX_CAST_TARGETS_16(CAST_ONE) }; };
template <> struct width_of<32> { enum { targets = 0 NX_CAST_TARGETS_32(CAST_ONE) }; };
template <> struct width_of<64> { enum { targets = 0 NX_CAST_TARGETS_64(CAST_ONE) }; };

/* A class's codes into a width's targets: entry (s·targets + t)·codes + c
   is code c of source s cast into target t. */
template <int K, int W> struct code_table {
  typedef typename store_of<W>::t R;
  enum { S = class_of<K>::sources, T = width_of<W>::targets, C = class_of<K>::codes };
  R v[S * T * C];
};

template <int K, int W> constexpr code_table<K, W> code_table_of() {
  typedef code_table<K, W> tb;
  tb t{};
  for (int s = 0; s < tb::S; s++)
    for (int u = 0; u < tb::T; u++)
      for (uint c = 0; c < uint(tb::C); c++)
        t.v[(s * tb::T + u) * tb::C + c] = typename tb::R(
            encode(value_of(class_source(K, s), c), width_target(W, u)));
  return t;
}

/* Each dtype's index among its class's sources and its width's
   targets. */
struct cast_index {
  uchar source[NX_DTYPE_COUNT], target[NX_DTYPE_COUNT];
};

constexpr cast_index cast_index_of() {
  cast_index x{};
  for (int i = 0; i < 0 NX_CAST_BYTE_SOURCES(CAST_ONE); i++)
    x.source[class_source(NX_CAST_BYTES, i)] = uchar(i);
  for (int i = 0; i < 0 NX_CAST_NIBBLE_SOURCES(CAST_ONE); i++)
    x.source[class_source(NX_CAST_NIBBLES, i)] = uchar(i);
  for (int i = 0; i < 0 NX_CAST_TARGETS_8(CAST_ONE); i++) x.target[width_target(8, i)] = uchar(i);
  for (int i = 0; i < 0 NX_CAST_TARGETS_16(CAST_ONE); i++) x.target[width_target(16, i)] = uchar(i);
  for (int i = 0; i < 0 NX_CAST_TARGETS_32(CAST_ONE); i++) x.target[width_target(32, i)] = uchar(i);
  for (int i = 0; i < 0 NX_CAST_TARGETS_64(CAST_ONE); i++) x.target[width_target(64, i)] = uchar(i);
  return x;
}

constant cast_index cast_indices = cast_index_of();

/* Each class's table into each width. */
template <int K, int W> struct table_of;
#define CAST_TABLE(c, K, W)                                                    \
  constant code_table<K, W> c##_##W = code_table_of<K, W>();                   \
  template <> struct table_of<K, W> {                                          \
    static inline constant typename store_of<W>::t *at() { return c##_##W.v; } \
  };
#define CAST_TABLES(c, K)                                                      \
  CAST_TABLE(c, K, 8) CAST_TABLE(c, K, 16) CAST_TABLE(c, K, 32)                \
  CAST_TABLE(c, K, 64)
CAST_TABLES(bytes, NX_CAST_BYTES)
CAST_TABLES(nibbles, NX_CAST_NIBBLES)

/* Computed casts */

/* An element's value as tape.h holds it: an integer of at most 32 bits
   extended to 32 by its sign, a wider element as its bits. */
template <class E> static inline u64 held_bits(E y) {
  return sizeof(E) == 8 ? u64(y) : u64(uint(int(y)));
}

/* The elements [y] of [S] as the bits of [T], one of width [W]'s
   targets, into [r]: one branch on [T] for the four, each arm tape.h's
   cast of constants. A branch per element made a cast of float32 at
   1 Mi 1.4 times as slow. */
#define CAST_TO(t)                                                             \
  case t:                                                                      \
    EACH4 r[v] = R(cast(S, t, held_of(held_bits<E>(y[v]), 0)).lo);            \
    return;
template <int S, int W, class E>
__attribute__((always_inline)) static inline void
cast_to(thread const E *y, int T, thread typename store_of<W>::t *r) {
  typedef typename store_of<W>::t R;
  if (W == 8)
    switch (T) { NX_CAST_TARGETS_8(CAST_TO) }
  else if (W == 16)
    switch (T) { NX_CAST_TARGETS_16(CAST_TO) }
  else if (W == 32)
    switch (T) { NX_CAST_TARGETS_32(CAST_TO) }
  else
    switch (T) { NX_CAST_TARGETS_64(CAST_TO) }
  EACH4 r[v] = R(0);
}

/* The results [r] of a kernel of class [K] into width [W] from the
   source [s]'s elements [y] into target [T]: by its table where the
   class has one, else computed. */
template <int K, int W,
          bool = K == NX_CAST_BYTES || K == NX_CAST_NIBBLES>
struct cast_by {
  typedef typename store_of<W>::t R;
  template <class E>
  __attribute__((always_inline)) static inline void
  run(thread const E *y, int, int T, thread R *r) {
    cast_to<K, W>(y, T, r);
  }
};
template <int K, int W> struct cast_by<K, W, true> {
  typedef typename store_of<W>::t R;
  typedef code_table<K, W> tb;
  template <class E>
  __attribute__((always_inline)) static inline void
  run(thread const E *y, int s, int T, thread R *r) {
    constant R *t = table_of<K, W>::at() +
                    (cast_indices.source[s] * uint(tb::T) +
                     cast_indices.target[T]) * uint(tb::C);
    EACH4 r[v] = t[uint(y[v])];
  }
};

/* Loads and stores */

/* A quad's or a thread's lanes' elements of [d] as E into [y]: one
   vector where the quad lies on its size, as direct_quad_in. */
template <class E>
__attribute__((always_inline)) static inline void
cast_in(thread const loop &c, constant u64 *d, lanes x, thread E *y) {
  device const E *e = (device const E *)buffer_of(c, d);
  if (x.quad) {
    typedef typename quad_of<E>::t E4;
    const uint p0 = uint(place(c, d, x.L0));
    const long st = last_stride(c, d);
    if (st == 0) {
      const E z = x.n > 0 ? e[p0] : E(0);
      EACH4 y[v] = z;
      return;
    }
    if (st == 1 && x.n == 4 && on(c, d, p0, sizeof(E), 4 * sizeof(E))) {
      const E4 q = *(device const E4 *)(e + p0);
      y[0] = q.x;
      y[1] = q.y;
      y[2] = q.z;
      y[3] = q.w;
      return;
    }
    EACH4 y[v] = v < x.n ? e[p0 + uint(long(v) * st)] : E(0);
    return;
  }
  u64 at[4];
  direct_positions(c, d, x, at);
  EACH4 y[v] = v < x.n ? e[uint(at[v])] : E(0);
}

/* The elements of [d], of [B] bits, 1 or 4, at the thread's indices into
   [y]: element p at bit phase + B·p of the buffer, low bits first. A
   quad of 4-bit elements whose 16 bits lie on 2 bytes is one load. */
template <uint B>
__attribute__((always_inline)) static inline void
sub_in(thread const loop &c, constant u64 *d, lanes x, thread uchar *y) {
  device const uchar *b = buffer_of(c, d);
  const u64 ph = phase_of(d);
  const uint mask = (1u << B) - 1;
  if (x.quad) {
    const u64 p0 = place(c, d, x.L0);
    const long st = last_stride(c, d);
    const u64 bit = ph + B * p0;
    if (B == 4 && st == 1 && x.n == 4 && (bit & 15) == 0) {
      const uint q = *(device const ushort *)(b + (bit >> 3));
      EACH4 y[v] = uchar((q >> (4 * v)) & 15);
      return;
    }
    EACH4 {
      const u64 at = ph + B * (p0 + u64(long(v) * st));
      y[v] = v < x.n ? uchar((b[at >> 3] >> (at & 7)) & mask) : uchar(0);
    }
    return;
  }
  u64 at[4];
  direct_positions(c, d, x, at);
  EACH4 {
    const u64 p = ph + B * at[v];
    y[v] = v < x.n ? uchar((b[p >> 3] >> (p & 7)) & mask) : uchar(0);
  }
}

/* The results [r] into [d] as R: one vector where the quad lies on its
   size, as direct_quad_out. */
template <class R>
__attribute__((always_inline)) static inline void
cast_out(thread const loop &c, constant u64 *d, lanes x, thread const R *r) {
  device R *e = (device R *)buffer_of(c, d);
  if (x.quad) {
    typedef typename quad_of<R>::t R4;
    const uint p0 = uint(place(c, d, x.L0));
    const long st = last_stride(c, d);
    if (st == 1 && x.n == 4 && on(c, d, p0, sizeof(R), 4 * sizeof(R))) {
      *(device R4 *)(e + p0) = R4(r[0], r[1], r[2], r[3]);
      return;
    }
    EACH4 if (v < x.n) e[p0 + uint(long(v) * st)] = r[v];
    return;
  }
  EACH4 if (v < x.n) e[uint(place(c, d, index_of(x, v)))] = r[v];
}

/* A flat quad's source elements from index [L0] into [y]: one vector, a
   nibble source's 16 bits on 2 bytes. Indices are 32-bit, as a map of
   quads is never wide; a nibble's bit place, 4 bits an index, passes
   2^32 from 2^30 elements, so it is 64-bit. */
template <int K, class E>
__attribute__((always_inline)) static inline void
cast_flat(thread const loop &c, constant u64 *ds, lanes x, thread E *y) {
  if (K == NX_CAST_NIBBLES) {
    const u64 bit = phase_of(ds) + 4 * (ds[1] + x.L0);
    const uint q = *(device const ushort *)(buffer_of(c, ds) + (bit >> 3));
    EACH4 y[v] = E((q >> (4 * v)) & 15);
    return;
  }
  typedef typename quad_of<E>::t E4;
  device const E *e = (device const E *)buffer_of(c, ds) + uint(ds[1]);
  const E4 q = *(device const E4 *)(e + uint(x.L0));
  y[0] = q.x;
  y[1] = q.y;
  y[2] = q.z;
  y[3] = q.w;
}

/* The source's elements at the thread's indices, by quads or lanes. */
template <int K, class E>
__attribute__((always_inline)) static inline void
cast_load(thread const loop &c, constant u64 *ds, lanes x, thread E *y) {
  if (K == NX_CAST_NIBBLES) {
    sub_in<4>(c, ds, x, (thread uchar *)y);
    return;
  }
  if (K == NX_BIT) {
    sub_in<1>(c, ds, x, (thread uchar *)y);
    return;
  }
  cast_in<E>(c, ds, x, y);
}

template <int K, class E, int W>
__attribute__((always_inline)) static inline void
cast_map(thread const loop &c, uint gid) {
  typedef typename store_of<W>::t R;
  const uint n = uint(hd(c, NX_LOOP_NUMEL));
  const uint base = (uint(hd(c, NX_LOOP_FIRST)) + gid) * c.E;
  const u64 mode = hd(c, NX_LOOP_MODE);
  const bool quad = (mode & NX_LOOP_QUADS) != 0;
  const bool flat = (mode & NX_LOOP_FLAT) != 0;
  lanes x = quad ? lanes32_of(base, 4 * c.me, 1, n, 4)
                 : lanes32_of(base, c.me, TB, n, 4);
  x.quad = quad;
  if (x.n == 0) return;
  constant u64 *ds = operand(c, NX_LOOP_LOADS, 0);
  constant u64 *dd = operand(c, NX_LOOP_DSTS, 0);
  const int s = int(ds[0] & 0xff);
  const int T = int(dd[0] & 0xff);
  E y[4];
  R r[4];
  /* Each path loads, casts and stores without a join: paths that merged
     before the cast made a cast of float8 at 1 Mi 1.1 times as slow. */
  if (flat) {
    typedef typename quad_of<R>::t R4;
    cast_flat<K>(c, ds, x, y);
    cast_by<K, W>::run(y, s, T, r);
    device R *o = (device R *)buffer_of(c, dd) + uint(dd[1]);
    *(device R4 *)(o + uint(x.L0)) = R4(r[0], r[1], r[2], r[3]);
    return;
  }
  cast_load<K>(c, ds, x, y);
  cast_by<K, W>::run(y, s, T, r);
  cast_out<R>(c, dd, x, r);
}

#define CAST_KERNEL(X, name, K, E, W)                                          \
  [[kernel]] void name(constant loop_words &w [[buffer(0)]],                   \
                       uint gid [[threadgroup_position_in_grid]],              \
                       uint me [[thread_index_in_threadgroup]]) {              \
    cast_map<K, E, W>(loop_of(w, nullptr, me, 4, false), gid);                 \
  }
NX_METAL_CAST_KERNELS(CAST_KERNEL, 0)

/* Flat direct maps of 32-bit lanes (kernels.h, NX_DIRECT_PLACES): a
   direct32 map whose walk is one axis and none of whose slots is placed
   by its descriptor. Each slot's element of index i lies i elements past
   its buffer's word, or at that word, so a thread reads no descriptor
   and no stride, and a quad lies on its size where its slot's place says
   so. Kernels of their own, so that no path of theirs costs a placed
   map, and none of a placed map's costs them. Their positions are below
   2^32, as direct32's. The helpers of every direct kernel's thread, here
   and in tape.h, are always inlined: with more callers in the unit, the
   inliner put place and coordinate out of line in the placed kernels. */

/* Slot [k]'s place in the form [f]. */
__attribute__((always_inline)) static inline uint
flat_place(u64 f, uint k) {
  return uint(f >> (NX_DIRECT_PLACES + 2 * k)) & 3;
}

/* Slot [k]'s words at the thread's indices [x], of which there is at
   least one, into [y], an element of no index taking zeros. */
__attribute__((always_inline)) static inline void
flat_in(thread const loop &c, uint k, uint place, lanes x, thread uint *y) {
  device const uint *e = (device const uint *)buffer_at(c, int(k));
  if (place == NX_DIRECT_FIRST) {
    const uint z = e[0];
    EACH4 y[v] = z;
    return;
  }
  const uint p0 = uint(x.L0);
  if (place == NX_DIRECT_FLAT_ON && x.n == 4) {
    const uint4 q = *(device const uint4 *)(e + p0);
    y[0] = q.x;
    y[1] = q.y;
    y[2] = q.z;
    y[3] = q.w;
    return;
  }
  EACH4 y[v] = v < x.n ? e[p0 + v] : 0u;
}

/* The words [y] at the thread's indices [x] into the destination, flat. */
__attribute__((always_inline)) static inline void
flat_out(thread const loop &c, uint place, lanes x, thread const uint *y) {
  device uint *e = (device uint *)buffer_at(c, 0);
  const uint p0 = uint(x.L0);
  if (place == NX_DIRECT_FLAT_ON && x.n == 4) {
    *(device uint4 *)(e + p0) = uint4(y[0], y[1], y[2], y[3]);
    return;
  }
  EACH4 if (v < x.n) e[p0 + v] = y[v];
}

template <int K>
__attribute__((always_inline)) static inline void
flat_map(thread const loop &c, uint gid) {
  const uint n = uint(hd(c, NX_LOOP_NUMEL));
  const uint base = (uint(hd(c, NX_LOOP_FIRST)) + gid) * c.E;
  const u64 f = hd(c, NX_LOOP_DIRECT);
  const uint m = uint(f >> NX_DIRECT_COUNT) & 3;
  uint a[4], b[4] = {0, 0, 0, 0}, e[4] = {0, 0, 0, 0}, r[4];
  const lanes x = lanes32_of(base, 4 * c.me, 1, n, 4);
  if (x.n == 0) return;
  flat_in(c, 1, flat_place(f, 1), x, a);
  if (m > 1) flat_in(c, 2, flat_place(f, 2), x, b);
  if (m > 2) flat_in(c, 3, flat_place(f, 3), x, e);
  const int op = int(f & 0xffff);
  if (K == DIRECT_LONG) direct_kind_long(op, a, b, e, r);
  else if (op == NX_LOOP_WHERE1) EACH4 r[v] = a[v] != 0 ? b[v] : e[v];
  else if (op == NX_LOOP_CAST) EACH4 r[v] = a[v];
  else direct_kind_cheap(op, a, b, e, r);
  flat_out(c, flat_place(f, 0), x, r);
}

[[kernel]] void
loop_map_direct32_flat(constant loop_words &w [[buffer(0)]],
                       threadgroup uint *pl [[threadgroup(0)]],
                       uint gid [[threadgroup_position_in_grid]],
                       uint me [[thread_index_in_threadgroup]]) {
  loop c = direct_of(w, pl, me);
  c.b32 = true;
  flat_map<DIRECT_CHEAP>(c, gid);
}

[[kernel]] void
loop_map_direct32_long_flat(constant loop_words &w [[buffer(0)]],
                            threadgroup uint *pl [[threadgroup(0)]],
                            uint gid [[threadgroup_position_in_grid]],
                            uint me [[thread_index_in_threadgroup]]) {
  loop c = direct_of(w, pl, me);
  c.b32 = true;
  flat_map<DIRECT_LONG>(c, gid);
}
