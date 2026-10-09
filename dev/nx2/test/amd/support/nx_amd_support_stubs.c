/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The AMD suite's and bench's C: the harness's code object, tables of
   dispatches, runs of records, and nx.amd's plans. Every stub holds the
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
#include "nx_amd.h"

#define STR_(x) #x
#define STR(x) STR_(x)

#if defined(__APPLE__)
#define SECTION ".const"
#define SYMBOL(s) "_" s
#else
#define SECTION ".section .rodata"
#define SYMBOL(s) s
#endif

/* The harness's code object */

__asm__(SECTION "\n"
        ".balign 16\n"
        ".globl " SYMBOL("nx_harness_co") "\n"
        SYMBOL("nx_harness_co") ":\n"
        ".incbin \"" STR(NX_HARNESS_CO) "\"\n"
        ".globl " SYMBOL("nx_harness_co_end") "\n"
        SYMBOL("nx_harness_co_end") ":\n"
        ".text\n");

extern const char nx_harness_co[], nx_harness_co_end[];

value nx_amd_support_code_object(value unit) {
  (void)unit;
  return caml_alloc_initialized_string(nx_harness_co_end - nx_harness_co,
                                       nx_harness_co);
}

value nx_amd_support_kernels(value unit) {
  static const char *names[] = {
#define NAME(name) #name,
      NX_HARNESS_KERNELS(NAME)
#undef NAME
      NULL};
  (void)unit;
  return caml_copy_string_array(names);
}

/* The workgroup sizes harness.h fixes: generate's and the probes', the
   floors', floor_read's vectors per work-item, and the hog's. */
value nx_amd_support_sizes(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(r);
  r = caml_alloc_tuple(5);
  Store_field(r, 0, Val_int(NX_THREADS));
  Store_field(r, 1, Val_int(NX_COPY_THREADS));
  Store_field(r, 2, Val_int(NX_READ_THREADS));
  Store_field(r, 3, Val_int(NX_READ_VECS));
  Store_field(r, 4, Val_int(NX_HOG_THREADS));
  CAMLreturn(r);
}

/* Dispatches */

/* The entries of the dispatches [v_ds], placed with the capability's
   [v_place] and [v_segment]: each dispatch a pair of its words and an int
   array of its holes, args then threads and groups by dimension, and the
   bytes of arguments its kernel reads.
   Never freed: a process makes one per image. */
value nx_amd_support_entries(value v_place, value v_segment, value v_ds) {
  int n = (int)Wosize_val(v_ds);
  size_t words = 0;
  for (int i = 0; i < n; i++)
    words += caml_string_length(Field(Field(v_ds, i), 0)) / 4;
  nx_amd_entries *e =
      malloc(sizeof *e + n * sizeof(nx_amd_dispatch) + 4 * words);
  if (e == NULL) caml_raise_out_of_memory();
  e->place = (void *)Nativeint_val(v_place);
  e->segment = (void *)Nativeint_val(v_segment);
  e->count = (uint32_t)n;
  uint32_t *at = (uint32_t *)&e->kernels[n];
  for (int i = 0; i < n; i++) {
    value w = Field(Field(v_ds, i), 0), h = Field(Field(v_ds, i), 1);
    nx_amd_dispatch *d = &e->kernels[i];
    d->words = at;
    d->n = (uint32_t)(caml_string_length(w) / 4);
    memcpy(at, String_val(w), 4 * d->n);
    at += d->n;
    d->args = (uint32_t)Long_val(Field(h, 0));
    for (int k = 0; k < 3; k++) {
      d->threads[k] = (uint32_t)Long_val(Field(h, 1 + k));
      d->groups[k] = (uint32_t)Long_val(Field(h, 4 + k));
    }
    d->kernarg = (uint32_t)Long_val(Field(h, 7));
  }
  return caml_copy_nativeint((intnat)e);
}

/* Records */

/* The run nx_amd_add appends the launches [v_ls] to, in order: each a pair
   of the kernel, groups, threads along X, address count and scratch mask,
   an int array of seven, and the parameters. */
value nx_amd_support_record(value v_ls) {
  CAMLparam1(v_ls);
  CAMLlocal1(r);
  nx_amd_records rs = {NULL, 0, 0};
  int rc = 0;
  for (mlsize_t i = 0; i < Wosize_val(v_ls) && rc == 0; i++) {
    value l = Field(Field(v_ls, i), 0), ps = Field(Field(v_ls, i), 1);
#define L(i) ((uint32_t)Long_val(Field(l, i)))
    uint32_t groups[3] = {L(1), L(2), L(3)}, threads[3] = {L(4), 1, 1};
    rc = nx_amd_add(&rs, L(0), groups, threads, String_val(ps),
                    (uint32_t)caml_string_length(ps), L(5), L(6));
#undef L
  }
  if (rc == 0)
    r = caml_alloc_initialized_string(rs.len, (const char *)rs.bytes);
  free(rs.bytes);
  if (rc == -1) caml_invalid_argument("nx_amd_add refused a record");
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

/* An nx_amd_run of the records [v_records], placed with the entries
   [v_entries], the records copied after it, with the ring words and
   segment bytes its fill takes, as nx_amd_size gives them. */
value nx_amd_support_run(value v_entries, value v_records) {
  CAMLparam2(v_entries, v_records);
  CAMLlocal2(v, r);
  size_t len = caml_string_length(v_records);
  v = bytes(sizeof(nx_amd_run) + len);
  nx_amd_run *run = Caml_ba_data_val(v);
  unsigned char *records = (unsigned char *)(run + 1);
  memcpy(records, String_val(v_records), len);
  *run = (nx_amd_run){(const nx_amd_entries *)Nativeint_val(v_entries),
                      records, len};
  uint64_t words, n;
  if (nx_amd_size(run, &words, &n) != 0)
    caml_invalid_argument("nx_amd_size refused a record");
  r = caml_alloc_tuple(3);
  Store_field(r, 0, v);
  Store_field(r, 1, Val_long(words));
  Store_field(r, 2, Val_long(n));
  CAMLreturn(r);
}

value nx_amd_support_fill(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)nx_amd_fill);
}

/* nx.amd */

/* The library's code object for the processor [v_arch], if it has one. */
value nx_amd_support_library(value v_arch) {
  CAMLparam1(v_arch);
  CAMLlocal1(s);
  size_t len;
  const char *c = nx_amd_code_object(Int_val(v_arch), &len);
  if (c == NULL) CAMLreturn(Val_none);
  s = caml_alloc_initialized_string(len, c);
  CAMLreturn(caml_alloc_some(s));
}

value nx_amd_support_library_kernels(value unit) {
  const char *names[NX_AMD_KERNEL_COUNT + 1] = {NULL};
  memcpy(names, nx_amd_kernel_names, sizeof nx_amd_kernel_names);
  (void)unit;
  return caml_copy_string_array(names);
}

static nx_amd_operand operand_of(value v) {
  nx_amd_operand o;
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

/* A contraction as the plan reads it. */
struct call {
  nx_amd_contract_in in;
  nx_amd_operand ops[4];
};

/* The call of a contraction: [v_ops] a, b, init if [v_init], y, each its
   address, dtype, shape and strides; [v_batch] and [v_contracting] pairs
   of axes as flat int arrays; [v_acc] the accumulator. */
value nx_amd_support_call(value v_ops, value v_batch, value v_contracting,
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

/* The processor the suite's code object is for. */
#define ARCH 1201

/* The plan of the call [v_call]: Some (records, scratch bytes, launches),
   or None if it declines; Out_of_memory if the host's memory runs out. */
value nx_amd_support_plan(value v_call) {
  CAMLparam1(v_call);
  CAMLlocal2(r, s);
  struct call *c = Caml_ba_data_val(v_call);
  nx_amd_records rs = {NULL, 0, 0};
  size_t scratch = 0;
  int launches = nx_amd_plan_contract(&c->in, c->ops, ARCH, &rs, &scratch);
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
value nx_amd_support_plan_only(value v_call) {
  static nx_amd_records rs = {NULL, 0, 0};
  struct call *c = Caml_ba_data_val(v_call);
  size_t scratch;
  rs.len = 0;
  return Val_int(nx_amd_plan_contract(&c->in, c->ops, ARCH, &rs, &scratch));
}

/* The records [v_r] with their scratch at [v_base]. */
value nx_amd_support_rebase(value v_r, value v_base) {
  CAMLparam2(v_r, v_base);
  CAMLlocal1(r);
  size_t len = caml_string_length(v_r);
  r = caml_alloc_initialized_string(len, String_val(v_r));
  nx_amd_rebase((unsigned char *)Bytes_val(r), len, (uint64_t)Long_val(v_base));
  CAMLreturn(r);
}
