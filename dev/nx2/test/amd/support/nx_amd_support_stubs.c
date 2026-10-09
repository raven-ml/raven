/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The AMD suite's and bench's C: the harness's code object, its kernels'
   names and the workgroup sizes harness.h fixes. */

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "harness.h"

#define STR_(x) #x
#define STR(x) STR_(x)

#if defined(__APPLE__)
#define SECTION ".const"
#define SYMBOL(s) "_" s
#else
#define SECTION ".section .rodata"
#define SYMBOL(s) s
#endif

/* The harness's code object */

__asm__(SECTION "\n"
        ".balign 16\n"
        ".globl " SYMBOL("nx_harness_co") "\n"
        SYMBOL("nx_harness_co") ":\n"
        ".incbin \"" STR(NX_HARNESS_CO) "\"\n"
        ".globl " SYMBOL("nx_harness_co_end") "\n"
        SYMBOL("nx_harness_co_end") ":\n"
        ".text\n");

extern const char nx_harness_co[], nx_harness_co_end[];

value nx_amd_support_code_object(value unit) {
  (void)unit;
  return caml_alloc_initialized_string(nx_harness_co_end - nx_harness_co,
                                       nx_harness_co);
}

value nx_amd_support_kernels(value unit) {
  static const char *names[] = {
#define NAME(name) #name,
      NX_HARNESS_KERNELS(NAME)
#undef NAME
      NULL};
  (void)unit;
  return caml_copy_string_array(names);
}

/* The workgroup sizes harness.h fixes: generate's and the probes', the
   floors', floor_read's vectors per work-item, and the hog's. */
value nx_amd_support_sizes(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(r);
  r = caml_alloc_tuple(5);
  Store_field(r, 0, Val_int(NX_THREADS));
  Store_field(r, 1, Val_int(NX_COPY_THREADS));
  Store_field(r, 2, Val_int(NX_READ_THREADS));
  Store_field(r, 3, Val_int(NX_READ_VECS));
  Store_field(r, 4, Val_int(NX_HOG_THREADS));
  CAMLreturn(r);
}
