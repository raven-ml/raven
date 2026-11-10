/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The cubins, embedded: one per architecture the library computes on.

   The kernels need a driver of CUDA 13 (R580) or later: nvcc 13.4 builds
   their cubins, which a driver of the same major version runs, newer
   minor versions included, since they carry no PTX to compile. */

#include <string.h>

#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

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

EMBED(nx_cuda_sm_89, STR(NX_CUDA_CUBIN))

/* The cubin of the architecture [arch], as "sm_89", or None. */
CAMLprim value nx_cuda_cubin(value arch) {
  CAMLparam1(arch);
  CAMLlocal1(bytes);
  if (strcmp(String_val(arch), "sm_89") != 0) CAMLreturn(Val_none);
  bytes = caml_alloc_initialized_string(nx_cuda_sm_89_end - nx_cuda_sm_89,
                                        nx_cuda_sm_89);
  CAMLreturn(caml_alloc_some(bytes));
}

/* Kinds */

#include "nx_kernel.h"
#include "kernels.h"

/* The code of the kind of [node], an Op1, Op2 or Op3 of Nx_kernel.Prog, as
   loop.cu's kind ops read it (kernels.h); -1 for a leaf. */
CAMLprim value nx_cuda_kind(value node) {
  switch (Tag_val(node)) {
  case NX_NODE_OP1: return Val_int(NX_OP1_CODE(Field(node, NX_OP1_KIND)));
  case NX_NODE_OP2:
    return Val_int(NX_LOOP_BINARY + NX_OP2_CODE(Field(node, NX_OP2_KIND)));
  case NX_NODE_OP3:
    return Val_int(NX_LOOP_TERNARY + Long_val(Field(node, NX_OP3_KIND)));
  default: return Val_int(-1);
  }
}

/* Layouts */

/* kernels.h's functions the host plans with, of OCaml ints: NX_GEMV_STAGE
   and the spectral routines' layouts. */
CAMLprim value nx_cuda_gemv_stage(value t) {
  return Val_long(NX_GEMV_STAGE(Long_val(t)));
}

CAMLprim value nx_cuda_sp_eigh_per(value n) {
  return Val_long(nx_sp_eigh_per(Int_val(n)));
}

CAMLprim value nx_cuda_sp_eigh_smem(value n, value t, value r) {
  return Val_long(nx_sp_eigh_smem(Int_val(n), Int_val(t), Int_val(r)));
}

CAMLprim value nx_cuda_sp_svd_per(value m, value n, value t, value r) {
  return Val_long(
      nx_sp_svd_per(Int_val(m), Int_val(n), Int_val(t), Int_val(r)));
}

CAMLprim value nx_cuda_sp_svd_smem(value m, value n, value t, value r) {
  return Val_long(
      nx_sp_svd_smem(Int_val(m), Int_val(n), Int_val(t), Int_val(r)));
}

CAMLprim value nx_cuda_sp_td_per(value n, value t, value r) {
  return Val_long(nx_sp_td_per(Int_val(n), Int_val(t), Int_val(r)));
}

CAMLprim value nx_cuda_sp_dc_per(value n, value r) {
  return Val_long(nx_sp_dc_per(Int_val(n), Int_val(r)));
}

CAMLprim value nx_cuda_sp_eig_per(value n, value t, value r) {
  return Val_long(nx_sp_eig_per(Int_val(n), Int_val(t), Int_val(r)));
}

/* nx_sp_bd_layout's struct, its bytes as the host lays it out. */
CAMLprim value nx_cuda_sp_bd_layout(value m, value k, value gc, value t,
                                    value r) {
  nx_sp_bd l = nx_sp_bd_layout(Long_val(m), Int_val(k), Int_val(gc),
                               Int_val(t), Int_val(r));
  return caml_alloc_initialized_string(sizeof l, (const char *)&l);
}
