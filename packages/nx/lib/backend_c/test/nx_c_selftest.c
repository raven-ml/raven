/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_selftest.c — engine self-test.

   Coalescing and the iteration ladder must be validated before any real kernel
   family lands, so this registers trivial f64 kernels (negate/sum/argmax/cumsum)
   through the REAL dispatch tables and drivers in nx_c_engine.c and exercises the
   layouts a kernel would otherwise be needed to reach: contiguous, transposed,
   broadcast, offset, empty, rank-32 all-ones, a >16M parallel split, scan slice
   independence, argreduce tie/empty/cap. It reports the failure count to OCaml;
   details go to stderr. It is linked only into the backend-local test executable.

   This file deliberately does NOT include caml/fail.h or caml/threads.h: it
   never raises and never hands off the runtime lock (the drivers it calls own
   that). caml_nx_c_selftest runs with the runtime lock held, as any CAMLprim
   does. */

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_c_engine.h"

/* Test hooks exported from nx_c_engine.c. */
int nx_c_selftest_coalesce_rank(const nx_c_ndarray *ops, int nop,
                               const int64_t *elem_size, int64_t *total);
int nx_c_selftest_ncores(void);
void nx_c_selftest_seek2(int nk, const int64_t *shape, const int64_t *s_in,
                        const int64_t *s_out, int64_t idx, char *base,
                        int64_t *in_off, int64_t *out_off);
int nx_c_selftest_worker_indices(int nth, int64_t total, int gate, int *out_max,
                                int *out_partition_ok, int *out_pool_workers);

/* Small parallel probe used on both sides of an OCaml [Unix.fork]. The OCaml
   test owns process creation so the OCaml 5 runtime performs its own fork
   bookkeeping; this primitive checks only the backend pool's child lifecycle. */
CAMLprim value caml_nx_c_selftest_parallel_probe(value unit) {
  CAMLparam1(unit);
  int max_worker = -1;
  int partition_ok = 0;
  int distinct = nx_c_selftest_worker_indices(2, 1024, 0, &max_worker,
                                              &partition_ok, NULL);
  CAMLreturn(Val_int(partition_ok && distinct >= 1 && max_worker < 2 ? 0 : 1));
}

/* ── Trivial f64 kernels ───────────────────────────────────────────────────
   Each honors the nx_c.h ABI; ctx is unused (these ops take no parameters). */

static void st_neg_f64(char *const *ptrs, const int64_t *steps, int64_t n,
                       void *ctx) {
  (void)ctx;
  char *o = ptrs[0];
  const char *a = ptrs[1];
  int64_t so = steps[0], sa = steps[1];
  for (int64_t i = 0; i < n; i++)
    *(double *)(o + i * so) = -*(const double *)(a + i * sa);
}

static void st_sum_init_f64(nx_c_acc *acc, void *ctx) {
  (void)ctx;
  acc->d = 0.0;
}
static void st_sum_step_f64(nx_c_acc *acc, const char *in, int64_t in_step,
                            int64_t n, void *ctx) {
  (void)ctx;
  double s = acc->d;
  for (int64_t i = 0; i < n; i++) s += *(const double *)(in + i * in_step);
  acc->d = s;
}
static void st_sum_fini_f64(char *out, const nx_c_acc *acc, void *ctx) {
  (void)ctx;
  *(double *)out = acc->d;
}
static void st_sum_stream_f64(void *accs, const char *in, int64_t in_step,
                              int64_t n, int first, void *ctx) {
  (void)ctx;
  double *a = (double *)accs;
  if (first)
    for (int64_t j = 0; j < n; j++) a[j] = *(const double *)(in + j * in_step);
  else
    for (int64_t j = 0; j < n; j++) a[j] += *(const double *)(in + j * in_step);
}
static void st_sum_scatter_f64(char *out, int64_t out_step, const void *accs,
                               int64_t n, void *ctx) {
  (void)ctx;
  const double *a = (const double *)accs;
  for (int64_t j = 0; j < n; j++) *(double *)(out + j * out_step) = a[j];
}

/* argmax: NaN wins, first index wins on ties (agrees with reduce_max). */
static void st_argmax_step_f64(nx_c_arg_acc *acc, const char *in,
                               int64_t in_step, int64_t n, void *ctx) {
  (void)ctx;
  for (int64_t i = 0; i < n; i++) {
    double v = *(const double *)(in + i * in_step);
    if (acc->index < 0) {
      acc->value.d = v;
      acc->index = i;
      continue;
    }
    double best = acc->value.d;
    if ((v > best) || (isnan(v) && !isnan(best))) {
      acc->value.d = v;
      acc->index = i;
    }
  }
}

static void st_cumsum_init_f64(nx_c_acc *state, void *ctx) {
  (void)ctx;
  state->d = 0.0;
}
static void st_cumsum_step_f64(char *out, int64_t out_step, const char *in,
                               int64_t in_step, int64_t n, nx_c_acc *state,
                               void *ctx) {
  (void)ctx;
  double run = state->d;
  for (int64_t i = 0; i < n; i++) {
    run += *(const double *)(in + i * in_step);
    *(double *)(out + i * out_step) = run;
  }
  state->d = run;
}

/* map2: out = a + b (ptrs[0]=out, [1]=a, [2]=b). */
static void st_add_f64(char *const *ptrs, const int64_t *steps, int64_t n,
                       void *ctx) {
  (void)ctx;
  char *o = ptrs[0];
  const char *a = ptrs[1];
  const char *b = ptrs[2];
  int64_t so = steps[0], sa = steps[1], sb = steps[2];
  for (int64_t i = 0; i < n; i++)
    *(double *)(o + i * so) =
        *(const double *)(a + i * sa) + *(const double *)(b + i * sb);
}

/* map3 where: out = cond ? a : b, with a real bool (uint8) condition operand
   (ptrs[0]=out f64, [1]=cond bool, [2]=a f64, [3]=b f64). Exercises a 4-operand
   map AND differing element sizes (cond 1 byte, the rest 8). */
static void st_where_f64(char *const *ptrs, const int64_t *steps, int64_t n,
                         void *ctx) {
  (void)ctx;
  char *o = ptrs[0];
  const char *c = ptrs[1];
  const char *a = ptrs[2];
  const char *b = ptrs[3];
  int64_t so = steps[0], sc = steps[1], sa = steps[2], sb = steps[3];
  for (int64_t i = 0; i < n; i++)
    *(double *)(o + i * so) = (*(const uint8_t *)(c + i * sc) != 0)
                                  ? *(const double *)(a + i * sa)
                                  : *(const double *)(b + i * sb);
}

/* cast-like: narrow an 8-byte double input to a 4-byte float output. Exercises
   the per-operand elem_size path (in byte-step 8, out byte-step 4). */
static void st_narrow_f64(char *const *ptrs, const int64_t *steps, int64_t n,
                          void *ctx) {
  (void)ctx;
  char *o = ptrs[0];
  const char *a = ptrs[1];
  int64_t so = steps[0], sa = steps[1];
  for (int64_t i = 0; i < n; i++)
    *(float *)(o + i * so) = (float)*(const double *)(a + i * sa);
}

/* Worker-tracking context for the parallelism proof: the neg kernel records the
   distinct OS threads that ran a chunk. Test kernels may synchronize; production
   kernels stay pure. */
typedef struct {
  pthread_mutex_t m;
  pthread_t ids[64];
  int distinct;
} st_worker_set;

static void st_neg_probe_f64(char *const *ptrs, const int64_t *steps, int64_t n,
                             void *ctx) {
  char *o = ptrs[0];
  const char *a = ptrs[1];
  int64_t so = steps[0], sa = steps[1];
  for (int64_t i = 0; i < n; i++)
    *(double *)(o + i * so) = -*(const double *)(a + i * sa);
  st_worker_set *w = ctx;
  pthread_t self = pthread_self();
  pthread_mutex_lock(&w->m);
  int seen = 0;
  for (int k = 0; k < w->distinct; k++)
    if (pthread_equal(w->ids[k], self)) {
      seen = 1;
      break;
    }
  if (!seen && w->distinct < 64) w->ids[w->distinct++] = self;
  pthread_mutex_unlock(&w->m);
}

static const nx_c_map_table st_neg_table = {.fn = {[NX_C_DTYPE_f64] = st_neg_f64}};
static const nx_c_map_table st_add_table = {.fn = {[NX_C_DTYPE_f64] = st_add_f64}};
static const nx_c_map_table st_where_table = {
    .fn = {[NX_C_DTYPE_f64] = st_where_f64}};
static const nx_c_map_table st_narrow_table = {
    .fn = {[NX_C_DTYPE_f64] = st_narrow_f64}};
static const nx_c_map_table st_neg_probe_table = {
    .fn = {[NX_C_DTYPE_f64] = st_neg_probe_f64}};
static const nx_c_fold_table st_sum_table = {
    .init = {[NX_C_DTYPE_f64] = st_sum_init_f64},
    .step = {[NX_C_DTYPE_f64] = st_sum_step_f64},
    .fini = {[NX_C_DTYPE_f64] = st_sum_fini_f64},
};
static const nx_c_stream_table st_sum_stream_table = {
    .stream = {[NX_C_DTYPE_f64] = st_sum_stream_f64},
    .scatter = {[NX_C_DTYPE_f64] = st_sum_scatter_f64},
};
static const nx_c_arg_table st_argmax_table = {
    .step = {[NX_C_DTYPE_f64] = st_argmax_step_f64}};
static const nx_c_scan_table st_cumsum_table = {
    .init = {[NX_C_DTYPE_f64] = st_cumsum_init_f64},
    .step = {[NX_C_DTYPE_f64] = st_cumsum_step_f64},
};

/* A real macro-generated map stub, driven end-to-end from OCaml by
   engine_test.ml: it exercises the funnel (nx_c_ndarray_of_value -> dtype ->
   nx_c_map_run -> raise) and pins the FFI record slot order. */
NX_C_MAP1_STUB(echo_neg, "echo_neg", st_neg_table, NX_C_COST_BANDWIDTH)

/* Reduction-funnel stubs, driven end-to-end from OCaml by engine_test.ml: they
   exercise nx_c_squeeze_out (keepdims=true kept-axis rebuild + mask dup-detect)
   and the reduction funnels' raise paths, which the direct-driver self-tests
   cannot reach. Hand-written (not macro) because reduction stubs carry op-
   specific axis parameters — exactly the thin marshal a family stub would do. */
CAMLprim value caml_nx_c_echo_sum(value vout, value vin, value vaxes) {
  CAMLparam3(vout, vin, vaxes);
  nx_c_fold_funnel("echo_sum", &st_sum_table, &st_sum_stream_table, NX_C_COST_COMPUTE, vout, vin, vaxes,
                  false, NULL);
  CAMLreturn(Val_unit);
}
CAMLprim value caml_nx_c_echo_argmax(value vout, value vin, value vaxis) {
  CAMLparam3(vout, vin, vaxis);
  nx_c_argreduce_funnel("echo_argmax", &st_argmax_table, NX_C_COST_COMPUTE, vout,
                       vin, Int_val(vaxis), NULL);
  CAMLreturn(Val_unit);
}

/* ── Fixtures ──────────────────────────────────────────────────────────────*/

static nx_c_ndarray nd(void *data, int ndim, const int64_t *shape,
                      const int64_t *strides, int64_t offset) {
  nx_c_ndarray a;
  a.data = data;
  a.ndim = ndim;
  a.offset = offset;
  for (int i = 0; i < ndim; i++) {
    a.shape[i] = shape[i];
    a.strides[i] = strides[i];
  }
  return a;
}

static int g_fails;
static void check(int cond, const char *name) {
  if (!cond) {
    fprintf(stderr, "engine self-test FAIL: %s\n", name);
    g_fails++;
  }
}

/* Statuses are static strings, but two identical literals in different
   translation units need not be the same object — compare by content, never by
   pointer. NX_C_OK (NULL) still compares by identity. */
static int status_is(nx_c_status s, nx_c_status want) {
  return s != NULL && want != NULL && strcmp(s, want) == 0;
}

static const int64_t E8[NX_C_MAX_OPERANDS] = {8, 8, 8, 8};

/* ── Map: contiguous 1-D ───────────────────────────────────────────────────*/
static void t_contiguous_1d(void) {
  int64_t n = 1000;
  double *in = malloc(n * sizeof(double));
  double *out = malloc(n * sizeof(double));
  for (int64_t i = 0; i < n; i++) in[i] = (double)(i + 1);
  int64_t sh[1] = {n}, st[1] = {1};
  nx_c_ndarray ops[2] = {nd(out, 1, sh, st, 0), nd(in, 1, sh, st, 0)};
  int64_t total = 0;
  check(nx_c_selftest_coalesce_rank(ops, 2, E8, &total) == 1 && total == n,
        "contiguous 1-D coalesces to rank 1");
  nx_c_status s = nx_c_map_run(&st_neg_table, NX_C_DTYPE_f64, 1, ops, E8,
                             NX_C_COST_BANDWIDTH, NULL);
  check(s == NX_C_OK, "contiguous 1-D status");
  int ok = 1;
  for (int64_t i = 0; i < n; i++)
    if (out[i] != -(double)(i + 1)) ok = 0;
  check(ok, "contiguous 1-D values");
  free(in);
  free(out);
}

/* ── Map: transposed 2-D (strided, must NOT coalesce) ──────────────────────*/
static void t_transposed_2d(void) {
  int64_t R = 17, C = 13;
  double *in = malloc(R * C * sizeof(double));
  double *out = malloc(R * C * sizeof(double));
  for (int64_t k = 0; k < R * C; k++) in[k] = (double)(k + 1);
  /* logical [R,C] as a transpose view: element (i,j) at i*1 + j*R */
  int64_t sh[2] = {R, C}, st[2] = {1, R};
  nx_c_ndarray ops[2] = {nd(out, 2, sh, st, 0), nd(in, 2, sh, st, 0)};
  check(nx_c_selftest_coalesce_rank(ops, 2, E8, NULL) == 2,
        "transposed 2-D stays rank 2");
  nx_c_status s = nx_c_map_run(&st_neg_table, NX_C_DTYPE_f64, 1, ops, E8,
                             NX_C_COST_BANDWIDTH, NULL);
  check(s == NX_C_OK, "transposed 2-D status");
  int ok = 1;
  for (int64_t i = 0; i < R; i++)
    for (int64_t j = 0; j < C; j++)
      if (out[i + j * R] != -in[i + j * R]) ok = 0;
  check(ok, "transposed 2-D values");
  free(in);
  free(out);
}

/* ── Map: coalescing merges contiguous multi-D ─────────────────────────────*/
static void t_coalesce_merge(void) {
  int64_t A = 8, B = 16;
  double *in = malloc(A * B * sizeof(double));
  double *out = malloc(A * B * sizeof(double));
  for (int64_t k = 0; k < A * B; k++) in[k] = (double)(k + 1);
  int64_t sh[2] = {A, B}, st[2] = {B, 1};
  nx_c_ndarray ops[2] = {nd(out, 2, sh, st, 0), nd(in, 2, sh, st, 0)};
  int64_t total = 0;
  check(nx_c_selftest_coalesce_rank(ops, 2, E8, &total) == 1 && total == A * B,
        "contiguous 2-D coalesces to rank 1");
  nx_c_status s = nx_c_map_run(&st_neg_table, NX_C_DTYPE_f64, 1, ops, E8,
                             NX_C_COST_BANDWIDTH, NULL);
  check(s == NX_C_OK, "coalesce-merge status");
  int ok = 1;
  for (int64_t k = 0; k < A * B; k++)
    if (out[k] != -(double)(k + 1)) ok = 0;
  check(ok, "coalesce-merge values");
  free(in);
  free(out);
}

/* ── Map: broadcast scalar input (0-stride) ────────────────────────────────*/
static void t_broadcast(void) {
  int64_t M = 6, N = 9;
  double scalar = 3.5;
  double *out = malloc(M * N * sizeof(double));
  int64_t osh[2] = {M, N}, ost[2] = {N, 1};
  int64_t ist[2] = {0, 0};
  nx_c_ndarray ops[2] = {nd(out, 2, osh, ost, 0), nd(&scalar, 2, osh, ist, 0)};
  check(nx_c_selftest_coalesce_rank(ops, 2, E8, NULL) == 1,
        "broadcast scalar coalesces to rank 1");
  nx_c_status s = nx_c_map_run(&st_neg_table, NX_C_DTYPE_f64, 1, ops, E8,
                             NX_C_COST_BANDWIDTH, NULL);
  check(s == NX_C_OK, "broadcast status");
  int ok = 1;
  for (int64_t k = 0; k < M * N; k++)
    if (out[k] != -scalar) ok = 0;
  check(ok, "broadcast values");
  free(out);
}

/* ── Map: nonzero offset base ──────────────────────────────────────────────*/
static void t_offset(void) {
  int64_t n = 40;
  double *inbuf = malloc((n + 5) * sizeof(double));
  double *outbuf = malloc((n + 3) * sizeof(double));
  for (int64_t i = 0; i < n + 5; i++) inbuf[i] = (double)(i + 1);
  for (int64_t i = 0; i < n + 3; i++) outbuf[i] = 999.0; /* sentinel */
  int64_t sh[1] = {n}, st[1] = {1};
  nx_c_ndarray ops[2] = {nd(outbuf, 1, sh, st, 3), nd(inbuf, 1, sh, st, 5)};
  nx_c_status s = nx_c_map_run(&st_neg_table, NX_C_DTYPE_f64, 1, ops, E8,
                             NX_C_COST_BANDWIDTH, NULL);
  check(s == NX_C_OK, "offset status");
  int ok = (outbuf[0] == 999.0 && outbuf[1] == 999.0 && outbuf[2] == 999.0);
  for (int64_t i = 0; i < n; i++)
    if (outbuf[3 + i] != -inbuf[5 + i]) ok = 0;
  check(ok, "offset values and untouched prefix");
  free(inbuf);
  free(outbuf);
}

/* ── Map: empty tensor ─────────────────────────────────────────────────────*/
static void t_empty(void) {
  double dummy = 1.0;
  int64_t sh[3] = {3, 0, 4}, st[3] = {0, 4, 1};
  nx_c_ndarray ops[2] = {nd(&dummy, 3, sh, st, 0), nd(&dummy, 3, sh, st, 0)};
  nx_c_status s = nx_c_map_run(&st_neg_table, NX_C_DTYPE_f64, 1, ops, E8,
                             NX_C_COST_BANDWIDTH, NULL);
  check(s == NX_C_OK, "empty tensor is a no-op");
}

/* ── Map: rank-32 all-size-1 ───────────────────────────────────────────────*/
static void t_rank32_ones(void) {
  double in = 2.0, out = 0.0;
  int64_t sh[NX_C_MAX_NDIM], st[NX_C_MAX_NDIM];
  for (int i = 0; i < NX_C_MAX_NDIM; i++) {
    sh[i] = 1;
    st[i] = 0;
  }
  nx_c_ndarray ops[2] = {nd(&out, NX_C_MAX_NDIM, sh, st, 0),
                        nd(&in, NX_C_MAX_NDIM, sh, st, 0)};
  int64_t total = 0;
  check(nx_c_selftest_coalesce_rank(ops, 2, E8, &total) == 1 && total == 1,
        "rank-32 all-ones coalesces to one element");
  nx_c_status s = nx_c_map_run(&st_neg_table, NX_C_DTYPE_f64, 1, ops, E8,
                             NX_C_COST_BANDWIDTH, NULL);
  check(s == NX_C_OK && out == -2.0, "rank-32 all-ones value");
}

/* ── Map: >16M parallel split — full coverage AND proof workers ran ─────────*/
static void t_parallel(void) {
  int64_t n = 17LL * 1024 * 1024; /* > 16M to cross the BANDWIDTH serial floor */
  double *buf = malloc(n * sizeof(double));
  if (!buf) {
    check(0, "parallel allocation");
    return;
  }
  for (int64_t i = 0; i < n; i++) buf[i] = (double)(i + 1);
  int64_t sh[1] = {n}, st[1] = {1};
  nx_c_ndarray ops[2] = {nd(buf, 1, sh, st, 0), nd(buf, 1, sh, st, 0)}; /* in-place */
  int cores = nx_c_selftest_ncores();
  st_worker_set w;
  pthread_mutex_init(&w.m, NULL);
  w.distinct = 0;
  nx_c_status s = nx_c_map_run(&st_neg_probe_table, NX_C_DTYPE_f64, 1, ops, E8,
                             NX_C_COST_BANDWIDTH, &w);
  pthread_mutex_destroy(&w.m);
  check(s == NX_C_OK, "parallel status");
  int ok = 1;
  for (int64_t i = 0; i < n; i++)
    if (buf[i] != -(double)(i + 1)) {
      ok = 0;
      break;
    }
  check(ok, "parallel full coverage (no dropped chunk)");
  /* The map driver actually fanned the run across the pool, not a silent serial
     fallback: the kernel recorded the distinct threads that ran a chunk. */
  if (cores <= 1)
    check(w.distinct == 1, "single-core map runs on one thread");
  else
    check(w.distinct > 1, "map driver fans out across worker threads");
  free(buf);
}

/* ── Fold: sum over an axis, over all axes, and over an empty axis ──────────*/
static void t_fold(void) {
  int64_t R = 4, C = 5;
  double *in = malloc(R * C * sizeof(double));
  for (int64_t i = 0; i < R; i++)
    for (int64_t j = 0; j < C; j++) in[i * C + j] = (double)(i * 10 + j + 1);
  int64_t ish[2] = {R, C}, ist[2] = {C, 1};
  nx_c_ndarray tin = nd(in, 2, ish, ist, 0);

  /* reduce axis 1 -> out[R] */
  double *out = malloc(R * sizeof(double));
  int64_t osh[1] = {R}, ost[1] = {1};
  nx_c_ndarray tout = nd(out, 1, osh, ost, 0);
  int rax[1] = {1};
  nx_c_status s = nx_c_fold_run(&st_sum_table, &st_sum_stream_table, NX_C_DTYPE_f64, &tin, 8, &tout, 8,
                              rax, 1, false, NX_C_COST_COMPUTE, NULL);
  check(s == NX_C_OK, "fold axis status");
  int ok = 1;
  for (int64_t i = 0; i < R; i++) {
    double want = 0;
    for (int64_t j = 0; j < C; j++) want += in[i * C + j];
    if (out[i] != want) ok = 0;
  }
  check(ok, "fold axis values");

  /* reduce all axes -> scalar */
  double scal = 0;
  nx_c_ndarray tscal = nd(&scal, 0, NULL, NULL, 0);
  int rax2[2] = {0, 1};
  s = nx_c_fold_run(&st_sum_table, &st_sum_stream_table, NX_C_DTYPE_f64, &tin, 8, &tscal, 8, rax2, 2,
                   false, NX_C_COST_COMPUTE, NULL);
  double total = 0;
  for (int64_t k = 0; k < R * C; k++) total += in[k];
  check(s == NX_C_OK && scal == total, "fold all-axes scalar");
  free(out);
  free(in);

  /* empty reduced axis stores the identity (0) */
  double *out2 = malloc(3 * sizeof(double));
  for (int i = 0; i < 3; i++) out2[i] = 111.0;
  double edummy = 0;
  int64_t esh[2] = {3, 0}, est[2] = {0, 1};
  nx_c_ndarray tein = nd(&edummy, 2, esh, est, 0);
  int64_t o2sh[1] = {3}, o2st[1] = {1};
  nx_c_ndarray teout = nd(out2, 1, o2sh, o2st, 0);
  int rax3[1] = {1};
  s = nx_c_fold_run(&st_sum_table, &st_sum_stream_table, NX_C_DTYPE_f64, &tein, 8, &teout, 8, rax3, 1,
                   false, NX_C_COST_COMPUTE, NULL);
  int okz = (s == NX_C_OK && out2[0] == 0 && out2[1] == 0 && out2[2] == 0);
  check(okz, "fold empty axis stores identity (sum, has identity)");

  /* Same empty extent, but a no-identity op (max/min): rejected before any
     kernel runs, so the -inf/INT64_MIN sentinel can never be stored. */
  for (int i = 0; i < 3; i++) out2[i] = 111.0;
  s = nx_c_fold_run(&st_sum_table, &st_sum_stream_table, NX_C_DTYPE_f64, &tein, 8, &teout, 8, rax3, 1,
                   true, NX_C_COST_COMPUTE, NULL);
  check(status_is(s, NX_C_ERR_EMPTY_REDUCE) && out2[0] == 111.0,
        "fold empty axis rejected for no-identity op (nothing written)");
  free(out2);
}

/* ── Argreduce: first-index tie, NaN-wins, empty rejection, INT32 cap ───────*/
static void t_argreduce(void) {
  double tie[4] = {1, 3, 3, 2};
  int32_t idx = -99;
  int64_t sh[1] = {4}, st[1] = {1};
  nx_c_ndarray tin = nd(tie, 1, sh, st, 0);
  nx_c_ndarray tout = nd(&idx, 0, NULL, NULL, 0);
  nx_c_status s = nx_c_argreduce_run(&st_argmax_table, NX_C_DTYPE_f64, &tin, 8,
                                   &tout, 0, NX_C_COST_COMPUTE, NULL);
  check(s == NX_C_OK && idx == 1, "argmax first-index tie");

  double nanv[4] = {1, NAN, 3, NAN};
  idx = -99;
  nx_c_ndarray tin2 = nd(nanv, 1, sh, st, 0);
  s = nx_c_argreduce_run(&st_argmax_table, NX_C_DTYPE_f64, &tin2, 8, &tout, 0,
                        NX_C_COST_COMPUTE, NULL);
  check(s == NX_C_OK && idx == 1, "argmax NaN wins first index");

  /* empty axis rejected before any work */
  double e = 0;
  int64_t esh[1] = {0}, est[1] = {1};
  nx_c_ndarray tein = nd(&e, 1, esh, est, 0);
  s = nx_c_argreduce_run(&st_argmax_table, NX_C_DTYPE_f64, &tein, 8, &tout, 0,
                        NX_C_COST_COMPUTE, NULL);
  check(status_is(s, NX_C_ERR_EMPTY_REDUCE), "argmax empty axis rejected");

  /* INT32_MAX cap checked without materializing the axis */
  check(status_is(nx_c_argreduce_validate((int64_t)INT32_MAX + 1),
                  NX_C_ERR_ARGREDUCE_CAP),
        "argmax axis > INT32_MAX rejected");
  check(nx_c_argreduce_validate(1024) == NX_C_OK, "argmax normal axis accepted");
}

/* ── Scan: slice independence along contiguous and strided axes ─────────────*/
static void t_scan(void) {
  int64_t S = 5, L = 7;
  double *in = malloc(S * L * sizeof(double));
  double *out = malloc(S * L * sizeof(double));
  for (int64_t i = 0; i < S; i++)
    for (int64_t j = 0; j < L; j++) in[i * L + j] = (double)(i * 10 + j + 1);
  int64_t sh[2] = {S, L}, stc[2] = {L, 1};
  nx_c_ndarray tin = nd(in, 2, sh, stc, 0);
  nx_c_ndarray tout = nd(out, 2, sh, stc, 0);

  /* axis 1: contiguous within slice */
  nx_c_status s = nx_c_scan_run(&st_cumsum_table, NX_C_DTYPE_f64, &tin, 8, &tout, 8,
                              1, NX_C_COST_COMPUTE, NULL);
  int ok = (s == NX_C_OK);
  for (int64_t i = 0; i < S; i++) {
    double run = 0;
    for (int64_t j = 0; j < L; j++) {
      run += in[i * L + j];
      if (out[i * L + j] != run) ok = 0;
    }
  }
  check(ok, "scan axis 1 (contiguous slices)");

  /* axis 0: strided within slice (stride L) */
  s = nx_c_scan_run(&st_cumsum_table, NX_C_DTYPE_f64, &tin, 8, &tout, 8, 0,
                   NX_C_COST_COMPUTE, NULL);
  ok = (s == NX_C_OK);
  for (int64_t j = 0; j < L; j++) {
    double run = 0;
    for (int64_t i = 0; i < S; i++) {
      run += in[i * L + j];
      if (out[i * L + j] != run) ok = 0;
    }
  }
  check(ok, "scan axis 0 (strided slices)");
  free(in);
  free(out);
}

/* ── Dispatch: NULL slot becomes a status, not a call ──────────────────────*/
static void t_dispatch_null(void) {
  double a = 1, b = 0;
  int64_t sh[1] = {1}, st[1] = {1};
  nx_c_ndarray ops[2] = {nd(&b, 1, sh, st, 0), nd(&a, 1, sh, st, 0)};
  /* f32 slot is empty in st_neg_table -> unsupported dtype */
  nx_c_status s = nx_c_map_run(&st_neg_table, NX_C_DTYPE_f32, 1, ops, E8,
                             NX_C_COST_BANDWIDTH, NULL);
  check(status_is(s, NX_C_ERR_UNSUPPORTED_DTYPE),
        "null dispatch slot -> unsupported");
  /* packed dtype -> packed status */
  s = nx_c_map_run(&st_neg_table, NX_C_DTYPE_i4, 1, ops, E8, NX_C_COST_BANDWIDTH,
                  NULL);
  check(status_is(s, NX_C_ERR_PACKED), "packed dtype -> packed status");
}

/* ── Negative strides: flipped map input and order-sensitive flipped scan ───*/
static void t_negative_stride(void) {
  int64_t n = 64;
  double *in = malloc(n * sizeof(double));
  double *out = malloc(n * sizeof(double));
  for (int64_t i = 0; i < n; i++) in[i] = (double)(i + 1);
  /* reversed view: element i is at physical n-1-i (stride -1, offset n-1) */
  int64_t sh[1] = {n}, sr[1] = {-1}, sc[1] = {1};
  nx_c_ndarray ops[2] = {nd(out, 1, sh, sc, 0), nd(in, 1, sh, sr, n - 1)};
  nx_c_status s = nx_c_map_run(&st_neg_table, NX_C_DTYPE_f64, 1, ops, E8,
                             NX_C_COST_BANDWIDTH, NULL);
  int ok = (s == NX_C_OK);
  for (int64_t i = 0; i < n; i++)
    if (out[i] != -in[n - 1 - i]) ok = 0;
  check(ok, "negative-stride map input");

  /* cumsum over the reversed view: out[i] = sum_{k<=i} in[n-1-k] */
  nx_c_ndarray tin = nd(in, 1, sh, sr, n - 1);
  nx_c_ndarray tout = nd(out, 1, sh, sc, 0);
  s = nx_c_scan_run(&st_cumsum_table, NX_C_DTYPE_f64, &tin, 8, &tout, 8, 0,
                   NX_C_COST_COMPUTE, NULL);
  ok = (s == NX_C_OK);
  double run = 0;
  for (int64_t i = 0; i < n; i++) {
    run += in[n - 1 - i];
    if (out[i] != run) ok = 0;
  }
  check(ok, "negative-stride scan (order sensitive)");
  free(in);
  free(out);
}

/* ── map2 / map3 (multi-operand merge rule, broadcast among 3+ operands) ────*/
static void t_map2_map3(void) {
  int64_t M = 6, N = 7;
  double *a = malloc(M * N * sizeof(double));
  double *b = malloc(M * N * sizeof(double));
  double *out = malloc(M * N * sizeof(double));
  for (int64_t k = 0; k < M * N; k++) {
    a[k] = (double)(k + 1);
    b[k] = (double)(2 * k + 1);
  }
  int64_t sh[2] = {M, N}, stc[2] = {N, 1};
  /* map2 add, all contiguous -> coalesces to rank 1 across 3 operands */
  nx_c_ndarray o2[3] = {nd(out, 2, sh, stc, 0), nd(a, 2, sh, stc, 0),
                       nd(b, 2, sh, stc, 0)};
  int64_t e3[3] = {8, 8, 8};
  nx_c_status s = nx_c_map_run(&st_add_table, NX_C_DTYPE_f64, 2, o2, e3,
                             NX_C_COST_BANDWIDTH, NULL);
  int ok = (s == NX_C_OK);
  for (int64_t k = 0; k < M * N; k++)
    if (out[k] != a[k] + b[k]) ok = 0;
  check(ok, "map2 add");

  /* map2 with b broadcast along axis 1: b stride {1,0}. The merge rule must
     block the merge for the non-broadcast operands where b cannot follow. */
  double *bcol = malloc(M * sizeof(double));
  for (int64_t i = 0; i < M; i++) bcol[i] = (double)(100 + i);
  int64_t bst[2] = {1, 0};
  nx_c_ndarray o2b[3] = {nd(out, 2, sh, stc, 0), nd(a, 2, sh, stc, 0),
                        nd(bcol, 2, sh, bst, 0)};
  s = nx_c_map_run(&st_add_table, NX_C_DTYPE_f64, 2, o2b, e3, NX_C_COST_BANDWIDTH,
                  NULL);
  ok = (s == NX_C_OK);
  for (int64_t i = 0; i < M; i++)
    for (int64_t j = 0; j < N; j++)
      if (out[i * N + j] != a[i * N + j] + bcol[i]) ok = 0;
  check(ok, "map2 add with broadcast column");

  /* map3 where with a real bool (1-byte) condition: exercises nop=4 and
     differing element sizes (cond 1 byte, the f64 operands 8). */
  uint8_t *cond = malloc(M * N);
  for (int64_t k = 0; k < M * N; k++) cond[k] = (k % 3 == 0) ? 1 : 0;
  int64_t cst[2] = {N, 1}; /* contiguous bool strides (elements) */
  nx_c_ndarray o3[4] = {nd(out, 2, sh, stc, 0), nd(cond, 2, sh, cst, 0),
                       nd(a, 2, sh, stc, 0), nd(b, 2, sh, stc, 0)};
  int64_t e4[4] = {8, 1, 8, 8};
  s = nx_c_map_run(&st_where_table, NX_C_DTYPE_f64, 3, o3, e4, NX_C_COST_BANDWIDTH,
                  NULL);
  ok = (s == NX_C_OK);
  for (int64_t k = 0; k < M * N; k++)
    if (out[k] != (cond[k] ? a[k] : b[k])) ok = 0;
  check(ok, "map3 where (bool cond, differing elem_size)");
  free(cond);
  free(bcol);
  free(a);
  free(b);
  free(out);
}

/* ── Differing element sizes (cast path): narrow f64 -> f32 ─────────────────*/
static void t_narrow(void) {
  int64_t n = 100;
  double *in = malloc(n * sizeof(double));
  float *out = malloc(n * sizeof(float));
  for (int64_t i = 0; i < n; i++) in[i] = (double)(i + 1) + 0.25;
  int64_t sh[1] = {n}, st[1] = {1};
  nx_c_ndarray ops[2] = {nd(out, 1, sh, st, 0), nd(in, 1, sh, st, 0)};
  int64_t elem[2] = {4, 8}; /* out f32 = 4 bytes, in f64 = 8 bytes */
  nx_c_status s = nx_c_map_run(&st_narrow_table, NX_C_DTYPE_f64, 1, ops, elem,
                             NX_C_COST_BANDWIDTH, NULL);
  int ok = (s == NX_C_OK);
  for (int64_t i = 0; i < n; i++)
    if (out[i] != (float)in[i]) ok = 0;
  check(ok, "cast narrow f64->f32 (per-operand elem_size)");
  free(in);
  free(out);
}

/* ── Multi-dim parallel fold: exercises nx_c_seek2 at a mid-nest start ───────*/
static void t_parallel_fold(void) {
  int64_t A = 512, B = 256, C = 4; /* A*B = 131072 outputs > 64K -> parallel */
  double *in = malloc(A * B * C * sizeof(double));
  double *out = malloc(A * B * sizeof(double));
  for (int64_t k = 0; k < A * B * C; k++) in[k] = (double)((k % 97) + 1);
  int64_t ish[3] = {A, B, C}, ist[3] = {B * C, C, 1};
  int64_t osh[2] = {A, B}, ost[2] = {B, 1};
  nx_c_ndarray tin = nd(in, 3, ish, ist, 0);
  nx_c_ndarray tout = nd(out, 2, osh, ost, 0);
  int rax[1] = {2};
  nx_c_status s = nx_c_fold_run(&st_sum_table, &st_sum_stream_table, NX_C_DTYPE_f64, &tin, 8, &tout, 8,
                              rax, 1, false, NX_C_COST_COMPUTE, NULL);
  int ok = (s == NX_C_OK);
  for (int64_t i = 0; i < A && ok; i++)
    for (int64_t j = 0; j < B; j++) {
      double want = 0;
      for (int64_t c = 0; c < C; c++) want += in[(i * B + j) * C + c];
      if (out[i * B + j] != want) {
        ok = 0;
        break;
      }
    }
  check(ok, "multi-dim parallel fold (seek at mid-nest)");
  free(in);
  free(out);
}

/* ── Fold reduced-axis swap: min-stride reduced axis is not innermost ───────*/
static void t_fold_swap(void) {
  int64_t A = 40, B = 30;
  double *in = malloc(A * B * sizeof(double));
  /* column-major view: element (i,j) at i*1 + j*A, so axis 0 has the small
     stride. Reducing {0,1} makes axis 0 (min stride) the run -> forces the
     stride swap that the C-contiguous tests never trigger. */
  for (int64_t k = 0; k < A * B; k++) in[k] = (double)((k % 31) + 1);
  int64_t ish[2] = {A, B}, ist[2] = {1, A};
  nx_c_ndarray tin = nd(in, 2, ish, ist, 0);
  double scal = 0;
  nx_c_ndarray tout = nd(&scal, 0, NULL, NULL, 0);
  int rax[2] = {0, 1};
  nx_c_status s = nx_c_fold_run(&st_sum_table, &st_sum_stream_table, NX_C_DTYPE_f64, &tin, 8, &tout, 8,
                              rax, 2, false, NX_C_COST_COMPUTE, NULL);
  double want = 0;
  for (int64_t k = 0; k < A * B; k++) want += in[k];
  check(s == NX_C_OK && scal == want, "fold reduced-axis stride swap");
  free(in);
}

/* ── Streaming fold: a kept axis out-contiguities the reduced axis ──────────
   Reducing axis 0 of a C-contiguous matrix (kept axis 1 is the contiguous lane,
   reduced axis 0 has the large stride) takes the driver's streaming path. A
   single-panel case (rank 2) covers the lane + scratch; a multi-panel case
   (rank 3, reduce the middle axis) covers the panel odometer and per-thread
   scratch reuse. Both must match the per-output reference exactly. */
static void t_fold_stream(void) {
  /* rank-2: [R,C] reduce axis 0 -> out[C]; lane = axis 1 (stride 1). */
  int64_t R = 33, C = 40;
  double *in = malloc(R * C * sizeof(double));
  for (int64_t k = 0; k < R * C; k++) in[k] = (double)((k % 17) - 8);
  int64_t ish[2] = {R, C}, ist[2] = {C, 1};
  nx_c_ndarray tin = nd(in, 2, ish, ist, 0);
  double *out = malloc(C * sizeof(double));
  int64_t osh[1] = {C}, ost[1] = {1};
  nx_c_ndarray tout = nd(out, 1, osh, ost, 0);
  int rax0[1] = {0};
  nx_c_status s = nx_c_fold_run(&st_sum_table, &st_sum_stream_table, NX_C_DTYPE_f64,
                              &tin, 8, &tout, 8, rax0, 1, false,
                              NX_C_COST_COMPUTE, NULL);
  int ok = (s == NX_C_OK);
  for (int64_t j = 0; j < C && ok; j++) {
    double want = 0;
    for (int64_t i = 0; i < R; i++) want += in[i * C + j];
    if (out[j] != want) ok = 0;
  }
  check(ok, "streaming fold axis 0 (single panel)");
  free(out);
  free(in);

  /* rank-3: [P,M,C] reduce axis 1 -> out[P,C]; lane = axis 2 (stride 1), the
     panels are axis 0. C-contiguous, so the middle reduced axis has the large
     stride C and streaming is chosen. */
  int64_t P = 6, M = 20, C3 = 12;
  double *in3 = malloc(P * M * C3 * sizeof(double));
  for (int64_t k = 0; k < P * M * C3; k++) in3[k] = (double)((k % 23) - 11);
  int64_t ish3[3] = {P, M, C3}, ist3[3] = {M * C3, C3, 1};
  nx_c_ndarray tin3 = nd(in3, 3, ish3, ist3, 0);
  double *out3 = malloc(P * C3 * sizeof(double));
  int64_t osh3[2] = {P, C3}, ost3[2] = {C3, 1};
  nx_c_ndarray tout3 = nd(out3, 2, osh3, ost3, 0);
  int rax1[1] = {1};
  s = nx_c_fold_run(&st_sum_table, &st_sum_stream_table, NX_C_DTYPE_f64, &tin3, 8,
                   &tout3, 8, rax1, 1, false, NX_C_COST_COMPUTE, NULL);
  ok = (s == NX_C_OK);
  for (int64_t p = 0; p < P && ok; p++)
    for (int64_t j = 0; j < C3; j++) {
      double want = 0;
      for (int64_t m = 0; m < M; m++) want += in3[(p * M + m) * C3 + j];
      if (out3[p * C3 + j] != want) {
        ok = 0;
        break;
      }
    }
  check(ok, "streaming fold middle axis (multi-panel)");
  free(out3);
  free(in3);

  /* Scratch-cap fallback: reduce axis 0 of [2, N] with N chosen so the streaming
     accumulator scratch (N * sizeof(nx_c_acc)) exceeds NX_C_FOLD_STREAM_SCRATCH_CAP
     — the streaming decision must fall back to the per-output path (which
     allocates nothing) rather than attempt a multi-hundred-MB malloc, and still
     produce correct results. N = 4.2M -> ~67 MB would-be scratch > 64 MB cap. */
  int64_t N = 4200000;
  double *inl = malloc(2 * N * sizeof(double));
  double *outl = malloc(N * sizeof(double));
  for (int64_t j = 0; j < N; j++) {
    inl[j] = (double)(j % 100);          /* row 0 */
    inl[N + j] = (double)((j % 100) + 1); /* row 1 */
  }
  int64_t ishl[2] = {2, N}, istl[2] = {N, 1};
  nx_c_ndarray tinl = nd(inl, 2, ishl, istl, 0);
  int64_t oshl[1] = {N}, ostl[1] = {1};
  nx_c_ndarray toutl = nd(outl, 1, oshl, ostl, 0);
  int raxl[1] = {0};
  s = nx_c_fold_run(&st_sum_table, &st_sum_stream_table, NX_C_DTYPE_f64, &tinl, 8,
                   &toutl, 8, raxl, 1, false, NX_C_COST_COMPUTE, NULL);
  ok = (s == NX_C_OK);
  for (int64_t j = 0; j < N && ok; j++)
    if (outl[j] != (double)((j % 100) * 2 + 1)) ok = 0;
  check(ok, "streaming fold falls back over scratch cap ([2, large] axis 0)");
  free(outl);
  free(inl);
}

/* ── Multi-slice parallel scan: exercises nx_c_seek2 at a mid-nest start ─────*/
static void t_parallel_scan(void) {
  int64_t S = 100000, L = 4; /* S slices, total 400000 > 64K -> parallel */
  double *in = malloc(S * L * sizeof(double));
  double *out = malloc(S * L * sizeof(double));
  for (int64_t k = 0; k < S * L; k++) in[k] = (double)((k % 13) + 1);
  int64_t sh[2] = {S, L}, stc[2] = {L, 1};
  nx_c_ndarray tin = nd(in, 2, sh, stc, 0);
  nx_c_ndarray tout = nd(out, 2, sh, stc, 0);
  nx_c_status s = nx_c_scan_run(&st_cumsum_table, NX_C_DTYPE_f64, &tin, 8, &tout, 8,
                              1, NX_C_COST_COMPUTE, NULL);
  int ok = (s == NX_C_OK);
  for (int64_t i = 0; i < S && ok; i++) {
    double run = 0;
    for (int64_t j = 0; j < L; j++) {
      run += in[i * L + j];
      if (out[i * L + j] != run) {
        ok = 0;
        break;
      }
    }
  }
  check(ok, "multi-slice parallel scan (seek at mid-nest)");
  free(in);
  free(out);
}

/* ── Exported nx_c_parallel_for: claim-dispatch contract ────────────────────*/
static void t_worker_indices(void) {
  int cores = nx_c_selftest_ncores();
  int nth = cores < 4 ? cores : 4;
  int gate = cores > 1;
  int pool_workers = 0;

  /* Element-granular range: drives the multi-chunk claim loop. The PRIME total
     keeps the cut off exact-division points for every nchunks the policy can
     pick here (nth * 8 for nth in [2,4]: 100003 mod 16/24/32 != 0), so a
     floor->ceil off-by-one in nx_c_chunk yields a detectably different
     partition — a round total cuts identically under that mutant and slips
     through (review finding). The contract is bounds + exact partition; which
     worker ids appear is a race, so fan-out is forced by the probe's
     two-arrival gate, and asserted only when the pool REALLY has >= 2 threads
     (all worker spawns failing legally degrades dispatch to serial). */
  int max_worker = -99;
  int partition_ok = 0;
  int distinct = nx_c_selftest_worker_indices(nth, 100003, gate, &max_worker,
                                             &partition_ok, &pool_workers);
  check(partition_ok,
        "claimed chunks partition [0,total), worker ids in [0,nth)");
  if (cores > 1 && pool_workers > 1)
    check(distinct >= 2, "claim dispatch fans out across >= 2 pool threads");
  else
    check(distinct == 1 && max_worker == 0,
          "serial/degraded parallel_for stays on worker 0");

  /* Unit-granular tail shape: 9 units on up to 4 workers — chunks == units, so
     the odd unit out goes to whichever thread frees first. Width-1 chunks make
     this row blind to cut mutants by construction; the prime rows either side
     carry the mutation teeth, this one checks the claim protocol. */
  int max2 = -99;
  int part2 = 0;
  int distinct2 =
      nx_c_selftest_worker_indices(nth, 9, gate, &max2, &part2, NULL);
  check(part2, "unit-granular claim partitions 9 units, worker ids in range");
  if (cores > 1 && pool_workers > 1)
    check(distinct2 >= 2, "tail units are claimed by >= 2 pool threads");
  else
    check(distinct2 == 1, "serial/degraded unit claim stays on worker 0");

  /* Small prime total ABOVE the chunk cap (37 > nth * 8 for nth <= 4): width-1
     and width-2 chunks mix, so the cut is inexact at small scale too — the
     tail shape a cut mutant CAN corrupt, unlike the 9-unit row. */
  int part3 = 0;
  (void)nx_c_selftest_worker_indices(nth, 37, gate, NULL, &part3, NULL);
  check(part3, "prime 37-unit claim partitions exactly (inexact small cuts)");
}

/* ── Direct seek2 positioning across a 3-D nest, including mid-nest starts ──*/
static void t_seek2_direct(void) {
  int64_t shape[3] = {4, 5, 3};
  int64_t s_in[3] = {200, 40, 8};  /* distinct byte strides */
  int64_t s_out[3] = {120, 24, 8}; /* different from s_in */
  char base[8192];                 /* covers the max offset (< 800) */
  int ok = 1;
  for (int64_t idx = 0; idx < 4 * 5 * 3; idx++) {
    int64_t in_off = -1, out_off = -1;
    nx_c_selftest_seek2(3, shape, s_in, s_out, idx, base, &in_off, &out_off);
    int64_t k = idx % 3, j = (idx / 3) % 5, i = idx / 15;
    if (in_off != i * 200 + j * 40 + k * 8) ok = 0;
    if (out_off != i * 120 + j * 24 + k * 8) ok = 0;
  }
  check(ok, "seek2 direct positioning (mid-nest chunk starts)");
}

/* ── Driver-side precondition verification (adjudication condition B) ───────*/
static void t_precondition_checks(void) {
  double buf[6] = {1, 2, 3, 4, 5, 6};
  double out[6] = {0};
  int64_t sh2[2] = {2, 3}, st2[2] = {3, 1};
  nx_c_ndarray in2 = nd(buf, 2, sh2, st2, 0);

  /* fold: unsorted axes rejected */
  int64_t osh[1] = {3}, ost[1] = {1};
  nx_c_ndarray o1 = nd(out, 1, osh, ost, 0);
  int bad[2] = {1, 0};
  nx_c_status s = nx_c_fold_run(&st_sum_table, &st_sum_stream_table, NX_C_DTYPE_f64, &in2, 8, &o1, 8, bad,
                              2, false, NX_C_COST_COMPUTE, NULL);
  check(status_is(s, NX_C_ERR_AXES), "fold rejects unsorted axes");

  /* fold: out rank disagreement rejected (rank-2 out for a rank-1 result) */
  int64_t o2sh[2] = {2, 3}, o2st[2] = {3, 1};
  nx_c_ndarray o2 = nd(out, 2, o2sh, o2st, 0);
  int rax[1] = {1};
  s = nx_c_fold_run(&st_sum_table, &st_sum_stream_table, NX_C_DTYPE_f64, &in2, 8, &o2, 8, rax, 1, false,
                   NX_C_COST_COMPUTE, NULL);
  check(status_is(s, NX_C_ERR_OUT_RANK), "fold rejects wrong output rank");

  /* argreduce: axis out of range rejected */
  int32_t idx[3] = {0};
  nx_c_ndarray oi = nd(idx, 1, osh, ost, 0);
  s = nx_c_argreduce_run(&st_argmax_table, NX_C_DTYPE_f64, &in2, 8, &oi, 5,
                        NX_C_COST_COMPUTE, NULL);
  check(status_is(s, NX_C_ERR_AXIS), "argreduce rejects out-of-range axis");

  /* scan: axis out of range rejected */
  nx_c_ndarray so = nd(out, 2, sh2, st2, 0);
  s = nx_c_scan_run(&st_cumsum_table, NX_C_DTYPE_f64, &in2, 8, &so, 8, 9,
                   NX_C_COST_COMPUTE, NULL);
  check(status_is(s, NX_C_ERR_AXIS), "scan rejects out-of-range axis");

  /* map: too many operands rejected */
  nx_c_ndarray many[5] = {in2, in2, in2, in2, in2};
  int64_t e5[5] = {8, 8, 8, 8, 8};
  s = nx_c_map_run(&st_neg_table, NX_C_DTYPE_f64, 4, many, e5, NX_C_COST_BANDWIDTH,
                  NULL);
  check(status_is(s, NX_C_ERR_ARITY), "map rejects arity > NX_C_MAX_OPERANDS");

  /* map: aliased (0-stride) output rejected */
  int64_t ash[1] = {4}, ast0[1] = {0}, ast1[1] = {1};
  nx_c_ndarray al[2] = {nd(out, 1, ash, ast0, 0), nd(buf, 1, ash, ast1, 0)};
  s = nx_c_map_run(&st_neg_table, NX_C_DTYPE_f64, 1, al, E8, NX_C_COST_BANDWIDTH,
                  NULL);
  check(status_is(s, NX_C_ERR_OUT_ALIASED), "map rejects aliased output");
}

CAMLprim value caml_nx_c_selftest(value unit) {
  CAMLparam1(unit);
  g_fails = 0;
  t_contiguous_1d();
  t_transposed_2d();
  t_coalesce_merge();
  t_broadcast();
  t_offset();
  t_empty();
  t_rank32_ones();
  t_parallel();
  t_fold();
  t_argreduce();
  t_scan();
  t_dispatch_null();
  t_negative_stride();
  t_map2_map3();
  t_narrow();
  t_parallel_fold();
  t_parallel_scan();
  t_seek2_direct();
  t_worker_indices();
  t_fold_swap();
  t_fold_stream();
  t_precondition_checks();
  CAMLreturn(Val_int(g_fails));
}
