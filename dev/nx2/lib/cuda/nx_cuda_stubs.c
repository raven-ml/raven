/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The cubins, embedded: one per architecture the library computes on. */

#include "nx_cuda.h"

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

const char *nx_cuda_cubin(int arch, size_t *len) {
  switch (arch) {
  case 89: *len = (size_t)(nx_cuda_sm_89_end - nx_cuda_sm_89); return nx_cuda_sm_89;
  }
  *len = 0;
  return NULL;
}
