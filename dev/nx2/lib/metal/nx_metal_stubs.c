/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx.metal's metallib (kernels/kernels.metallib), included by the
   assembler, and its kernels' names. The section and the symbols'
   spelling are the object format's: ELF, Mach-O or COFF. */

#define _GNU_SOURCE

#include "nx_metal.h"

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

const char *nx_metal_metallib(size_t *len) {
  *len = (size_t)(nx_metal_lib_end - nx_metal_lib);
  return nx_metal_lib;
}

const char *const nx_metal_kernel_names[NX_METAL_KERNEL_COUNT] = {
#define NX_METAL_NAME(name) #name,
    NX_METAL_KERNELS(NX_METAL_NAME)
#undef NX_METAL_NAME
};
