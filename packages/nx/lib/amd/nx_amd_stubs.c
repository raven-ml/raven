/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The archive of nx.amd's code objects (kernels/kernels.bin), included by the
   assembler, and its bytes as OCaml strings. The section and the symbols'
   spelling are the object format's: ELF, Mach-O or COFF. */

#include <string.h>

#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#define STR_(x) #x
#define STR(x) STR_(x)

#if defined(__APPLE__)
#define SECTION ".const"
#define SYMBOL(s) "_" s
#elif defined(_WIN32)
#define SECTION ".section .rdata,\"dr\""
#define SYMBOL(s) s
#else
#define SECTION ".section .rodata"
#define SYMBOL(s) s
#endif

__asm__(SECTION "\n"
        ".balign 16\n"
        ".globl " SYMBOL("nx_amd_kernels") "\n"
        SYMBOL("nx_amd_kernels") ":\n"
        ".incbin \"" STR(NX_AMD_KERNELS) "\"\n"
        ".globl " SYMBOL("nx_amd_kernels_end") "\n"
        SYMBOL("nx_amd_kernels_end") ":\n"
        ".text\n");

extern const char nx_amd_kernels[], nx_amd_kernels_end[];

/* [caml_nx_amd_kernels_length ()] is the archive's length in bytes. */
value caml_nx_amd_kernels_length(value unit) {
  (void)unit;
  return Val_long(nx_amd_kernels_end - nx_amd_kernels);
}

/* [caml_nx_amd_kernels_sub off len] is the [len] bytes of the archive from
   [off], which the caller keeps inside it. */
value caml_nx_amd_kernels_sub(value off, value len) {
  CAMLparam2(off, len);
  CAMLlocal1(s);
  s = caml_alloc_string(Long_val(len));
  memcpy(Bytes_val(s), nx_amd_kernels + Long_val(off), Long_val(len));
  CAMLreturn(s);
}
