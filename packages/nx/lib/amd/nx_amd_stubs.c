/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The code object, embedded: gfx12-generic's, which LLVM's AMDGPU backend
   runs on gfx1200 and gfx1201. */

#include <string.h>

#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_kernel.h"

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

EMBED(nx_amd_gfx12_generic, STR(NX_AMD_CODE_OBJECT))

/* The code object for the processor [arch], as "gfx1201", or None. */
CAMLprim value nx_amd_code_object(value arch) {
  CAMLparam1(arch);
  CAMLlocal1(bytes);
  const char *a = String_val(arch);
  if (strcmp(a, "gfx1200") != 0 && strcmp(a, "gfx1201") != 0)
    CAMLreturn(Val_none);
  bytes = caml_alloc_initialized_string(
      nx_amd_gfx12_generic_end - nx_amd_gfx12_generic, nx_amd_gfx12_generic);
  CAMLreturn(caml_alloc_some(bytes));
}

/* The code of the kind of [node], an Op1, Op2 or Op3 of Nx_kernel.Prog, as
   nx_kernel.h numbers it in its arity; -1 for a leaf. */
CAMLprim value nx_amd_kind(value node) {
  switch (Tag_val(node)) {
  case NX_NODE_OP1: return Val_int(NX_OP1_CODE(Field(node, NX_OP1_KIND)));
  case NX_NODE_OP2: return Val_int(NX_OP2_CODE(Field(node, NX_OP2_KIND)));
  case NX_NODE_OP3: return Val_int(Long_val(Field(node, NX_OP3_KIND)));
  default: return Val_int(-1);
  }
}
