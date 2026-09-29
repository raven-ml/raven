/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sched.h>
#include <time.h>
#endif

/* Copies at least this large release the runtime while they run. */
#define NX_DEVICE_BLOCKING_BYTES (1 << 16)

intnat caml_nx_device_bigarray_address(value ba) {
  return (intnat)Caml_ba_data_val(ba);
}

value caml_nx_device_bigarray_address_byte(value ba) {
  return caml_copy_nativeint(caml_nx_device_bigarray_address(ba));
}

value caml_nx_device_memmove(intnat dst, intnat src, intnat n) {
  if (n >= NX_DEVICE_BLOCKING_BYTES) {
    caml_release_runtime_system();
    memmove((void *)dst, (const void *)src, (size_t)n);
    caml_acquire_runtime_system();
  } else {
    memmove((void *)dst, (const void *)src, (size_t)n);
  }
  return Val_unit;
}

value caml_nx_device_memmove_byte(value dst, value src, value n) {
  return caml_nx_device_memmove(Nativeint_val(dst), Nativeint_val(src),
                                Long_val(n));
}

extern value caml_ba_sub(value vb, value vofs, value vlen);

/* [v_len] elements of kind [v_kind] from byte [v_offset] of [v_src]. The
   header comes from [caml_ba_sub] over the whole of [v_src], so it joins
   [v_src]'s storage, which lives as long as any array over it. Its data,
   length and kind are rewritten, and flags the runtime does not define are
   cleared. */
value caml_nx_device_bigarray_view(value v_src, value v_kind, value v_offset,
                                   value v_len) {
  CAMLparam2(v_src, v_kind);
  CAMLlocal1(view);
  view = caml_ba_sub(v_src, Val_long(0),
                     Val_long(Caml_ba_array_val(v_src)->dim[0]));
  struct caml_ba_array *b = Caml_ba_array_val(view);
  b->data = (char *)b->data + Long_val(v_offset);
  b->flags = (b->flags & (CAML_BA_LAYOUT_MASK | CAML_BA_MANAGED_MASK)) |
             Int_val(v_kind);
  b->dim[0] = Long_val(v_len);
  CAMLreturn(view);
}

/* The timeline's words are read and written by other threads and by devices:
   every access is atomic. */

int64_t caml_nx_device_load_u64(intnat addr) {
  return (int64_t)atomic_load_explicit((_Atomic uint64_t *)addr,
                                       memory_order_acquire);
}

value caml_nx_device_load_u64_byte(value addr) {
  return caml_copy_int64(caml_nx_device_load_u64(Nativeint_val(addr)));
}

value caml_nx_device_store_u64(intnat addr, int64_t v) {
  atomic_store_explicit((_Atomic uint64_t *)addr, (uint64_t)v,
                        memory_order_release);
  return Val_unit;
}

value caml_nx_device_store_u64_byte(value addr, value v) {
  return caml_nx_device_store_u64(Nativeint_val(addr), Int64_val(v));
}

static int64_t now_ms(void) {
#ifdef _WIN32
  return (int64_t)GetTickCount64();
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
#endif
}

static void yield(void) {
#ifdef _WIN32
  SwitchToThread();
#else
  sched_yield();
#endif
}

/* Waits until the word at [addr] reaches [v]. The timeout restarts whenever
   the word moves: a device that makes progress is not hung. */
intnat caml_nx_device_wait_u64(intnat addr, int64_t v, intnat timeout_ms) {
  _Atomic uint64_t *word = (_Atomic uint64_t *)addr;
  uint64_t target = (uint64_t)v;
  uint64_t seen = atomic_load_explicit(word, memory_order_acquire);
  if (seen >= target) return 1;
  int signaled = 0;
  caml_release_runtime_system();
  int64_t start = now_ms();
  for (;;) {
    uint64_t now = atomic_load_explicit(word, memory_order_acquire);
    if (now >= target) {
      signaled = 1;
      break;
    }
    if (now != seen) {
      seen = now;
      start = now_ms();
    } else if (now_ms() - start > timeout_ms) {
      break;
    }
    yield();
  }
  caml_acquire_runtime_system();
  return signaled;
}

value caml_nx_device_wait_u64_byte(value addr, value v, value timeout_ms) {
  return Val_long(caml_nx_device_wait_u64(Nativeint_val(addr), Int64_val(v),
                                          Long_val(timeout_ms)));
}
