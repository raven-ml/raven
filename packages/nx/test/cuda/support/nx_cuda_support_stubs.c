/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The CUDA suite's and bench's C: the harness's cubin, its kernels' names,
   the floors' constants, and the device's attributes through the function
   the device's capability finds, bound once. */

#include <stdint.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "harness.h"

/* The harness's cubin, embedded as nx.cuda embeds its own */

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

EMBED(nx_harness_cubin, STR(NX_HARNESS_CUBIN))

value nx_cuda_support_cubin(value unit) {
  (void)unit;
  return caml_alloc_initialized_string(nx_harness_cubin_end - nx_harness_cubin,
                                       nx_harness_cubin);
}

value nx_cuda_support_kernels(value unit) {
  static const char *names[] = {
#define NAME(name) #name,
      NX_HARNESS_KERNELS(NAME)
#undef NAME
      NULL};
  (void)unit;
  return caml_copy_string_array(names);
}

/* The floors' threads per block and floor_read's vectors per thread, as
   harness.h fixes them. */
value nx_cuda_support_floors(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(r);
  r = caml_alloc_tuple(3);
  Store_field(r, 0, Val_int(NX_COPY_THREADS));
  Store_field(r, 1, Val_int(NX_READ_THREADS));
  Store_field(r, 2, Val_int(NX_READ_VECS));
  CAMLreturn(r);
}

/* CUDA, as the capability finds it */

static int (*get_attribute)(int *, int, int);

/* Binds cuDeviceGetAttribute. */
value nx_cuda_support_bind(value v_f) {
  get_attribute = (void *)Nativeint_val(v_f);
  return Val_unit;
}

/* CUDA device 0's attribute [v_a]. */
value nx_cuda_support_attribute(value v_a) {
  int a = 0;
  if (get_attribute(&a, Int_val(v_a), 0) != 0) caml_failwith("attribute");
  return Val_int(a);
}
