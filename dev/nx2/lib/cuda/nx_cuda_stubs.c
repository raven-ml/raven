/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The cubins, embedded: one per architecture the library computes on; and
   the descriptor readers of fold.ml.

   The kernels need a driver of CUDA 13 (R580) or later: nvcc 13.4 builds
   their cubins, which a driver of the same major version runs, newer
   minor versions included, since they carry no PTX to compile. */

#include <string.h>

#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_spec.h"

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

EMBED(nx_cuda_sm_89, STR(NX_CUDA_KERNELS_DIR) "/sm_89.cubin")

/* The cubin of the architecture [arch], as "sm_89", or None. */
CAMLprim value nx_cuda_cubin(value arch) {
  CAMLparam1(arch);
  CAMLlocal1(bytes);
  if (strcmp(String_val(arch), "sm_89") != 0) CAMLreturn(Val_none);
  bytes = caml_alloc_initialized_string(nx_cuda_sm_89_end - nx_cuda_sm_89,
                                        nx_cuda_sm_89);
  CAMLreturn(caml_alloc_some(bytes));
}

/* Descriptors of reductions and scans, as fold.ml plans them. */

/* The monoid of [l] where it is the case fold.cu computes: one Sum, Prod,
   Max or Min of the program [In 0]'s output into its operand's dtype, over
   one plain load; else -1. */
static int core(const nx_spec_loop *l) {
  if (l->nloads != 1 || l->nreductions != 1 || nx_spec_loop_pad(l, 0))
    return -1;
  const nx_prog *p = nx_spec_loop_prog(l);
  if (p->nins != 1 || p->nnodes != 1 || p->nouts != 1) return -1;
  if (p->nodes[0].tag != NX_NODE_IN || p->nodes[0].a != 0) return -1;
  const nx_spec_reduction *r = nx_spec_loop_reductions(l);
  if (r->kind > NX_MIN || r->output != 0 || r->dtype != nx_prog_ins(p)[0])
    return -1;
  return r->kind;
}

#define LOOP(v) ((const nx_spec_loop *)String_val(v))

intnat nx_cuda_core(value s) { return core(LOOP(s)); }
value nx_cuda_core_byte(value s) { return Val_long(nx_cuda_core(s)); }

/* The dtype code of the program's operand. */
intnat nx_cuda_core_dtype(value s) {
  return nx_prog_ins(nx_spec_loop_prog(LOOP(s)))[0];
}
value nx_cuda_core_dtype_byte(value s) {
  return Val_long(nx_cuda_core_dtype(s));
}

intnat nx_cuda_naxes(value s) { return LOOP(s)->naxes; }
value nx_cuda_naxes_byte(value s) { return Val_long(nx_cuda_naxes(s)); }

intnat nx_cuda_axis(value s, intnat i) {
  return nx_spec_loop_axes(LOOP(s))[i];
}
value nx_cuda_axis_byte(value s, value i) {
  return Val_long(nx_cuda_axis(s, Long_val(i)));
}
