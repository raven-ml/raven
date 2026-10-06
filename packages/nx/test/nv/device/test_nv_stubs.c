/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Host loads and stores of the words a channel's writer touches. */

#include <caml/alloc.h>
#include <caml/mlvalues.h>
#include <stdint.h>

#define Addr_val(v) ((void *)Nativeint_val(v))

value test_nv_load64(value addr) {
  return caml_copy_int64(
      __atomic_load_n((uint64_t *)Addr_val(addr), __ATOMIC_ACQUIRE));
}

value test_nv_store64(value addr, value v) {
  __atomic_store_n((uint64_t *)Addr_val(addr), (uint64_t)Int64_val(v),
                   __ATOMIC_SEQ_CST);
  return Val_unit;
}

value test_nv_store32(value addr, value v) {
  __atomic_store_n((uint32_t *)Addr_val(addr), (uint32_t)Long_val(v),
                   __ATOMIC_SEQ_CST);
  return Val_unit;
}
