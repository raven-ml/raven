/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <caml/alloc.h>
#include <caml/mlvalues.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

/* AddressSanitizer aborts on an allocation larger than it supports, where the
   system returns null: the tests of buffers no device can allocate need the
   null, which nx.device turns into Out_of_memory. ASAN_OPTIONS still overrides
   this default. */
#if defined(__SANITIZE_ADDRESS__)
#define TEST_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define TEST_ASAN 1
#endif
#endif
#ifdef TEST_ASAN
const char *__asan_default_options(void) {
  return "allocator_may_return_null=1";
}
#endif

/* Stores [v] into the word at [addr], as a device signals its timeline. */
value test_nx_device_signal(value addr, value v) {
  atomic_store_explicit((_Atomic uint64_t *)Nativeint_val(addr),
                        (uint64_t)Long_val(v), memory_order_release);
  return Val_unit;
}

/* Copies [n] bytes from [src] to [dst], as a device's copy engine does. */
value test_nx_device_memmove(value dst, value src, value n) {
  memmove((void *)Nativeint_val(dst), (const void *)Nativeint_val(src),
          (size_t)Long_val(n));
  return Val_unit;
}

#include "nx_device.h"

/* The host address of a buffer as C code reads it. */
value test_nx_device_buffer_host(value b) {
  return caml_copy_nativeint((intnat)nx_device_buffer_host(b));
}

/* Whether a buffer is live as C code checks it: its memory was not consumed
   since it was made. */
value test_nx_device_buffer_live(value b) {
  return Val_bool(nx_device_buffer_live(b));
}
