/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The code objects, embedded: one per processor the library computes on. */

#include "nx_amd.h"

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

EMBED(nx_amd_gfx1201, STR(NX_AMD_KERNELS_DIR) "/gfx1201.co")

const char *nx_amd_code_object(int arch, size_t *len) {
  switch (arch) {
  case 1201: *len = (size_t)(nx_amd_gfx1201_end - nx_amd_gfx1201); return nx_amd_gfx1201;
  }
  *len = 0;
  return NULL;
}
