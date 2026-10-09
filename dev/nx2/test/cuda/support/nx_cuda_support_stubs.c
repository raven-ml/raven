/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The CUDA suite's and bench's C: the harness's cubin, tables of entries,
   and a fill that runs a sequence of record runs. CUDA's functions are
   those the device's capability finds, bound once. Every stub holds the
   runtime: none blocks. */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "harness.h"
#include "nx_cuda.h"

/* The harness's cubin, embedded as nx.cuda embeds its own */

/* Defines [name] and [name]_end around the bytes of the file [file], a
   string, embedded in read-only data at build time. */
#define STR_(x) #x
#define STR(x) STR_(x)
#if defined(__APPLE__)
#define SECTION ".const"
#define SYMBOL(s) "_" s
#else
#define SECTION ".section .rodata"
#define SYMBOL(s) s
#endif
#define EMBED(name, file)                                                      \
  __asm__(SECTION "\n.balign 16\n.globl " SYMBOL(#name) "\n"                   \
          SYMBOL(#name) ":\n.incbin \"" file "\"\n.globl "                     \
          SYMBOL(#name "_end") "\n" SYMBOL(#name "_end")                       \
          ":\n.text\n");                                                       \
  extern const char name[], name##_end[];

EMBED(nx_harness_cubin, STR(NX_HARNESS_CUBIN))

value nx_cuda_support_cubin(value unit) {
  (void)unit;
  return caml_alloc_initialized_string(nx_harness_cubin_end - nx_harness_cubin,
                                       nx_harness_cubin);
}

value nx_cuda_support_kernels(value unit) {
  static const char *names[] = {
#define NAME(name) #name,
      NX_HARNESS_KERNELS(NAME)
#undef NAME
      NULL};
  (void)unit;
  return caml_copy_string_array(names);
}

/* The floors' threads per block and floor_read's vectors per thread, as
   harness.h fixes them. */
value nx_cuda_support_floors(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(r);
  r = caml_alloc_tuple(3);
  Store_field(r, 0, Val_int(NX_COPY_THREADS));
  Store_field(r, 1, Val_int(NX_READ_THREADS));
  Store_field(r, 2, Val_int(NX_READ_VECS));
  CAMLreturn(r);
}

/* nx.cuda */

/* The library's cubin for the architecture [v_arch], if it has one. */
value nx_cuda_support_library(value v_arch) {
  CAMLparam1(v_arch);
  CAMLlocal1(s);
  size_t len;
  const char *c = nx_cuda_cubin(Int_val(v_arch), &len);
  if (c == NULL) CAMLreturn(Val_none);
  s = caml_alloc_initialized_string(len, c);
  CAMLreturn(caml_alloc_some(s));
}

value nx_cuda_support_library_kernels(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(r);
  r = caml_alloc(NX_CUDA_KERNEL_COUNT, 0);
  for (int i = 0; i < NX_CUDA_KERNEL_COUNT; i++)
    Store_field(r, i, caml_copy_string(nx_cuda_kernel_names[i]));
  CAMLreturn(r);
}

static nx_cuda_operand operand_of(value v) {
  nx_cuda_operand o;
  memset(&o, 0, sizeof o);
  o.address = (uint64_t)Long_val(Field(v, 0));
  o.dtype = Int_val(Field(v, 1));
  o.rank = (int)Wosize_val(Field(v, 2));
  for (int i = 0; i < o.rank; i++) {
    o.dim[i] = Long_val(Field(Field(v, 2), i));
    o.dim[o.rank + i] = Long_val(Field(Field(v, 3), i));
  }
  return o;
}

static value bytes(size_t n);

/* A contraction as the plan reads it. */
struct call {
  nx_cuda_contract_in in;
  nx_cuda_operand ops[4];
};

/* The call of a contraction: [v_ops] a, b, init if [v_init], y, each its
   address, dtype, shape and strides; [v_batch] and [v_contracting] pairs
   of axes as flat int arrays; [v_acc] the accumulator. */
value nx_cuda_support_call(value v_ops, value v_batch, value v_contracting,
                           value v_acc, value v_init) {
  CAMLparam5(v_ops, v_batch, v_contracting, v_acc, v_init);
  CAMLlocal1(v);
  v = bytes(sizeof(struct call));
  struct call *c = Caml_ba_data_val(v);
  for (mlsize_t i = 0; i < Wosize_val(v_ops); i++)
    c->ops[i] = operand_of(Field(v_ops, i));
  c->in.nbatch = (int)Wosize_val(v_batch) / 2;
  for (int i = 0; i < c->in.nbatch; i++)
    c->in.batch[i][0] = Int_val(Field(v_batch, 2 * i)),
    c->in.batch[i][1] = Int_val(Field(v_batch, 2 * i + 1));
  c->in.ncontracting = (int)Wosize_val(v_contracting) / 2;
  for (int i = 0; i < c->in.ncontracting; i++)
    c->in.contracting[i][0] = Int_val(Field(v_contracting, 2 * i)),
    c->in.contracting[i][1] = Int_val(Field(v_contracting, 2 * i + 1));
  c->in.acc = Int_val(v_acc);
  c->in.init = Bool_val(v_init);
  CAMLreturn(v);
}

/* The plan of the call [v_call]: Some (records, scratch bytes, launches),
   or None if it declines; Out_of_memory if the host's memory runs out. */
value nx_cuda_support_plan(value v_call) {
  CAMLparam1(v_call);
  CAMLlocal2(r, s);
  struct call *c = Caml_ba_data_val(v_call);
  nx_cuda_records rs = {NULL, 0, 0};
  size_t scratch = 0;
  int launches = nx_cuda_plan_contract(&c->in, c->ops, 89, &rs, &scratch);
  if (launches == NX_OUT_OF_MEMORY) {
    free(rs.bytes);
    caml_raise_out_of_memory();
  }
  if (launches == NX_NOT_COMPUTED) {
    free(rs.bytes);
    CAMLreturn(Val_none);
  }
  s = caml_alloc_initialized_string(rs.len, (const char *)rs.bytes);
  free(rs.bytes);
  r = caml_alloc_tuple(3);
  Store_field(r, 0, s);
  Store_field(r, 1, Val_long((intnat)scratch));
  Store_field(r, 2, Val_int(launches));
  CAMLreturn(caml_alloc_some(r));
}

/* The launches of the call [v_call]'s plan, into records kept from one
   call to the next: the planner's cost alone. */
value nx_cuda_support_plan_only(value v_call) {
  static nx_cuda_records rs = {NULL, 0, 0};
  struct call *c = Caml_ba_data_val(v_call);
  size_t scratch;
  rs.len = 0;
  return Val_int(nx_cuda_plan_contract(&c->in, c->ops, 89, &rs, &scratch));
}

/* The records [v_r] with their scratch at [v_base]. */
value nx_cuda_support_rebase(value v_r, value v_base) {
  CAMLparam2(v_r, v_base);
  CAMLlocal1(r);
  size_t len = caml_string_length(v_r);
  /* v_r is read after the allocation, which may move it. */
  r = caml_alloc_string(len);
  memcpy(Bytes_val(r), String_val(v_r), len);
  nx_cuda_rebase((unsigned char *)Bytes_val(r), len, (uint64_t)Long_val(v_base));
  CAMLreturn(r);
}

/* CUDA, as the capability finds it */

static nx_cuda_launch_fn launch_kernel;
static int (*get_attribute)(int *, int, int);

/* Binds cuLaunchKernel and cuDeviceGetAttribute, in this order. */
value nx_cuda_support_bind(value v_f) {
  launch_kernel = (nx_cuda_launch_fn)Nativeint_val(Field(v_f, 0));
  get_attribute = (void *)Nativeint_val(Field(v_f, 1));
  return Val_unit;
}

/* CUDA device 0's attribute [v_a]. */
value nx_cuda_support_attribute(value v_a) {
  int a = 0;
  if (get_attribute(&a, Int_val(v_a), 0) != 0) caml_failwith("attribute");
  return Val_int(a);
}

/* The entries of the functions [v_funcs]. Never freed: a process makes one
   per image. */
value nx_cuda_support_entries(value v_funcs) {
  int n = (int)Wosize_val(v_funcs);
  nx_cuda_entries *e = malloc(sizeof *e + n * sizeof(void *));
  if (e == NULL) caml_raise_out_of_memory();
  e->launch = launch_kernel;
  e->count = (uint32_t)n;
  for (int i = 0; i < n; i++) e->funcs[i] = (void *)Long_val(Field(v_funcs, i));
  return caml_copy_nativeint((intnat)e);
}

/* Records */

/* The run nx_cuda_add appends the launches [v_ls] to, in order: each a
   pair of the kernel, grid, block, shared bytes, address count and scratch
   mask, an int array of eight, and the parameters. */
value nx_cuda_support_record(value v_ls) {
  CAMLparam1(v_ls);
  CAMLlocal1(r);
  nx_cuda_records rs = {NULL, 0, 0};
  int rc = 0;
  for (mlsize_t i = 0; i < Wosize_val(v_ls) && rc == 0; i++) {
    value l = Field(Field(v_ls, i), 0), ps = Field(Field(v_ls, i), 1);
#define L(i) ((uint32_t)Long_val(Field(l, i)))
    uint32_t grid[3] = {L(1), L(2), L(3)}, block[3] = {L(4), 1, 1};
    rc = nx_cuda_add(&rs, L(0), grid, block, L(5), String_val(ps),
                     (uint32_t)caml_string_length(ps), L(6), L(7));
#undef L
  }
  if (rc == 0)
    r = caml_alloc_initialized_string(rs.len, (const char *)rs.bytes);
  free(rs.bytes);
  if (rc == -1) caml_invalid_argument("nx_cuda_add refused a record");
  if (rc != 0) caml_raise_out_of_memory();
  CAMLreturn(r);
}

/* Fills */

static value bytes(size_t n) {
  value v = caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT, 1, NULL,
                               (intnat)n);
  memset(Caml_ba_data_val(v), 0, n);
  return v;
}

/* An nx_cuda_run of the records [v_records], launched with the entries
   [v_entries], the records copied after it. */
value nx_cuda_support_run(value v_entries, value v_records) {
  CAMLparam2(v_entries, v_records);
  CAMLlocal1(v);
  size_t len = caml_string_length(v_records);
  v = bytes(sizeof(nx_cuda_run) + len);
  nx_cuda_run *r = Caml_ba_data_val(v);
  unsigned char *records = (unsigned char *)(r + 1);
  memcpy(records, String_val(v_records), len);
  *r = (nx_cuda_run){(const nx_cuda_entries *)Nativeint_val(v_entries),
                     records, len};
  CAMLreturn(v);
}

value nx_cuda_support_fill(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)nx_cuda_fill);
}
