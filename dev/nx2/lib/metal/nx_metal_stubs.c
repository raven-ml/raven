/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx.metal's metallib (kernels/embedded.metallib), included by the
   assembler. The section and the symbols' spelling are the object
   format's: ELF, Mach-O or COFF. */

#include <caml/alloc.h>
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
        ".globl " SYMBOL("nx_metal_lib") "\n"
        SYMBOL("nx_metal_lib") ":\n"
        ".incbin \"" STR(NX_METAL_METALLIB) "\"\n"
        ".globl " SYMBOL("nx_metal_lib_end") "\n"
        SYMBOL("nx_metal_lib_end") ":\n"
        ".text\n");

extern const char nx_metal_lib[], nx_metal_lib_end[];

/* The metallib, one for every Apple GPU. */
value nx_metal_metallib(value unit) {
  (void)unit;
  return caml_alloc_initialized_string(nx_metal_lib_end - nx_metal_lib,
                                       nx_metal_lib);
}
