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
