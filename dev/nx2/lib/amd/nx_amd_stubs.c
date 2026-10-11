/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The code objects, embedded: one per processor the library computes on. */

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

EMBED(nx_amd_gfx1201, STR(NX_AMD_KERNELS_DIR) "/gfx1201.co")

/* The code object of the processor [arch], as "gfx1201", or None. */
CAMLprim value nx_amd_code_object(value arch) {
  CAMLparam1(arch);
  CAMLlocal1(bytes);
  if (strcmp(String_val(arch), "gfx1201") != 0) CAMLreturn(Val_none);
  bytes = caml_alloc_initialized_string(nx_amd_gfx1201_end - nx_amd_gfx1201,
                                        nx_amd_gfx1201);
  CAMLreturn(caml_alloc_some(bytes));
}
