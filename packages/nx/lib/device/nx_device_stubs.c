/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#define _GNU_SOURCE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <errno.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nx_device.h"

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#else
#include <dlfcn.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#endif

#if defined(__APPLE__) && defined(__aarch64__)
#include <pthread.h>
#endif

/* Copies at least this large release the runtime while they run. */
#define NX_DEVICE_BLOCKING_BYTES (1 << 16)

/* The bytes of the host's heap that live buffers hold. A buffer reserves its
   bytes against the host's budget and holds a token, a custom block whose
   finaliser returns them: the collector runs it where it frees the block,
   with no OCaml function to call. */
static _Atomic intnat heap_bytes;

intnat caml_nx_device_heap_bytes(value unit) {
  (void)unit;
  return atomic_load_explicit(&heap_bytes, memory_order_relaxed);
}

value caml_nx_device_heap_bytes_byte(value unit) {
  return Val_long(caml_nx_device_heap_bytes(unit));
}

value caml_nx_device_heap_reserve(intnat n, intnat budget) {
  intnat held = atomic_load_explicit(&heap_bytes, memory_order_relaxed);
  do {
    if (n > budget - held) return Val_false;
  } while (!atomic_compare_exchange_weak_explicit(
      &heap_bytes, &held, held + n, memory_order_relaxed,
      memory_order_relaxed));
  return Val_true;
}

value caml_nx_device_heap_reserve_byte(value n, value budget) {
  return caml_nx_device_heap_reserve(Long_val(n), Long_val(budget));
}

value caml_nx_device_heap_return(intnat n) {
  atomic_fetch_sub_explicit(&heap_bytes, n, memory_order_relaxed);
  return Val_unit;
}

value caml_nx_device_heap_return_byte(value n) {
  return caml_nx_device_heap_return(Long_val(n));
}

static void heap_token_finalize(value v) {
  caml_nx_device_heap_return(*(intnat *)Data_custom_val(v));
}

static struct custom_operations heap_token_ops = {
    "nx.device.heap_token",     heap_token_finalize,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

/* A token that returns [v_n] reserved bytes once it is collected. */
value caml_nx_device_heap_token(value v_n) {
  value v = caml_alloc_custom(&heap_token_ops, sizeof(intnat), 0, 1);
  *(intnat *)Data_custom_val(v) = Long_val(v_n);
  return v;
}

/* [v_n] bytes of the heap on a page, as a [char] bigarray that frees them, or
   [None] where the C library aligns nothing that [free] releases (Windows). */
value caml_nx_device_heap_aligned(value v_page, value v_n) {
  CAMLparam2(v_page, v_n);
#ifdef _WIN32
  (void)v_page;
  (void)v_n;
  CAMLreturn(Val_none);
#else
  CAMLlocal1(ba);
  void *data = NULL;
  if (posix_memalign(&data, (size_t)Long_val(v_page), (size_t)Long_val(v_n)))
    caml_raise_out_of_memory();
  intnat dim = Long_val(v_n);
  ba = caml_ba_alloc(CAML_BA_CHAR | CAML_BA_C_LAYOUT | CAML_BA_MANAGED, 1,
                     data, &dim);
  CAMLreturn(caml_alloc_some(ba));
#endif
}

/* The [v_n] bytes at [v_addr] as a [char] bigarray that owns nothing: memory a
   device holds, which the host addresses. */
value caml_nx_device_external_bytes(value v_addr, value v_n) {
  intnat dim = Long_val(v_n);
  return caml_ba_alloc(CAML_BA_CHAR | CAML_BA_C_LAYOUT | CAML_BA_EXTERNAL, 1,
                       (void *)Nativeint_val(v_addr), &dim);
}

value caml_nx_device_page_size(value unit) {
  (void)unit;
#ifdef _WIN32
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return Val_long(info.dwPageSize);
#else
  return Val_long(sysconf(_SC_PAGESIZE));
#endif
}

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

/* Gives the managed array [b] the proxy that its sub-arrays share, if it has
   none. The runtime makes it at the first sub without synchronization; here it
   is made at most once, whatever the domains that view [b] at once. Its first
   reference is [b]'s own, as the runtime counts it. */
static void ensure_proxy(struct caml_ba_array *b) {
  _Atomic(struct caml_ba_proxy *) *slot =
      (_Atomic(struct caml_ba_proxy *) *)&b->proxy;
  if ((b->flags & CAML_BA_MANAGED_MASK) == CAML_BA_EXTERNAL ||
      atomic_load_explicit(slot, memory_order_acquire) != NULL)
    return;
  struct caml_ba_proxy *proxy = malloc(sizeof *proxy);
  if (proxy == NULL) caml_raise_out_of_memory();
  atomic_store_explicit(&proxy->refcount, 1, memory_order_relaxed);
  proxy->data = b->data;
  proxy->size = b->flags & CAML_BA_MAPPED_FILE ? caml_ba_byte_size(b) : 0;
  struct caml_ba_proxy *none = NULL;
  if (!atomic_compare_exchange_strong_explicit(
          slot, &none, proxy, memory_order_acq_rel, memory_order_acquire))
    free(proxy);
}

/* [v_len] elements of kind [v_kind] from byte [v_offset] of [v_src]. The
   header comes from [caml_ba_sub] over the whole of [v_src], so it joins
   [v_src]'s storage, which lives as long as any array over it. Its data,
   length and kind are rewritten, and flags the runtime does not define are
   cleared. */
value caml_nx_device_bigarray_view(value v_src, value v_kind, value v_offset,
                                   value v_len) {
  CAMLparam2(v_src, v_kind);
  CAMLlocal1(view);
  ensure_proxy(Caml_ba_array_val(v_src));
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

intnat caml_nx_device_now_ms(value unit) {
  (void)unit;
  return (intnat)now_ms();
}

value caml_nx_device_now_ms_byte(value unit) {
  return Val_long(caml_nx_device_now_ms(unit));
}

intnat caml_nx_device_now_ns(value unit) {
  (void)unit;
  return (intnat)nx_device_now_ns();
}

value caml_nx_device_now_ns_byte(value unit) {
  return Val_long(caml_nx_device_now_ns(unit));
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

/* Host programs */

/* Code is never writable and executable at once. Elsewhere than arm64 macOS,
   it is mapped writable, filled, then made executable. arm64 macOS maps
   MAP_JIT memory, whose write protection each thread lifts for itself. */

static void code_fail(const char *what) {
  char msg[256];
#ifdef _WIN32
  snprintf(msg, sizeof msg, "Nx_device.Program.load: %s: error %lu", what,
           (unsigned long)GetLastError());
#else
  snprintf(msg, sizeof msg, "Nx_device.Program.load: %s: %s", what,
           strerror(errno));
#endif
  caml_failwith(msg);
}

value caml_nx_device_code_alloc(value v_size) {
  size_t size = (size_t)Long_val(v_size);
#ifdef _WIN32
  void *p = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  if (p == NULL) code_fail("no executable memory");
#else
  int flags = MAP_PRIVATE | MAP_ANON;
  int prot = PROT_READ | PROT_WRITE;
#if defined(__APPLE__) && defined(__aarch64__)
  flags |= MAP_JIT;
  prot |= PROT_EXEC;
#endif
  void *p = mmap(NULL, size, prot, flags, -1, 0);
  if (p == MAP_FAILED) code_fail("no executable memory");
#endif
  return caml_copy_nativeint((intnat)p);
}

value caml_nx_device_code_install(value v_addr, value v_code) {
  char *p = (char *)Nativeint_val(v_addr);
  size_t n = caml_string_length(v_code);
#ifdef _WIN32
  DWORD old;
  memcpy(p, Bytes_val(v_code), n);
  if (!VirtualProtect(p, n, PAGE_EXECUTE_READ, &old))
    code_fail("cannot make the code executable");
  FlushInstructionCache(GetCurrentProcess(), p, n);
#else
#if defined(__APPLE__) && defined(__aarch64__)
  pthread_jit_write_protect_np(0);
  memcpy(p, Bytes_val(v_code), n);
  pthread_jit_write_protect_np(1);
#else
  memcpy(p, Bytes_val(v_code), n);
  if (mprotect(p, n, PROT_READ | PROT_EXEC) != 0)
    code_fail("cannot make the code executable");
#endif
  __builtin___clear_cache(p, p + n);
#endif
  return Val_unit;
}

value caml_nx_device_code_free(value v_addr, value v_size) {
  void *p = (void *)Nativeint_val(v_addr);
#ifdef _WIN32
  (void)v_size;
  VirtualFree(p, 0, MEM_RELEASE);
#else
  munmap(p, (size_t)Long_val(v_size));
#endif
  return Val_unit;
}

/* Compiler builtins. The compiler lowers a conversion the target cannot do
   inline, such as float32 to bfloat16 on x86_64 without AVX512-BF16, to a call
   into its runtime library, even where the source converts nothing: a 16-bit
   float merged across a branch is widened to float32 and rounded back. A
   loaded object has no such library, and a host's copy, if any, may take
   another calling convention: the object's is System V on x86_64, even on
   Windows. These are the host's copies, with the object's convention.

   A 16-bit float travels in the low bits of a floating-point register, where
   a float's bits start, so it is passed and returned as a float holding its
   bits. Rounding is to nearest, ties to even. A NaN keeps the high bits of its
   payload, and its lowest kept bit is set if a dropped bit was: it stays a
   NaN, and a widened value comes back with its bits. */

#if defined(_WIN32) && defined(__x86_64__)
#define NX_DEVICE_OBJECT_ABI __attribute__((sysv_abi))
#else
#define NX_DEVICE_OBJECT_ABI
#endif

static uint32_t float_bits(float x) {
  uint32_t b;
  memcpy(&b, &x, sizeof b);
  return b;
}

static float bits_float(uint32_t b) {
  float x;
  memcpy(&x, &b, sizeof x);
  return x;
}

static NX_DEVICE_OBJECT_ABI float truncsfbf2(float x) {
  uint32_t b = float_bits(x);
  if ((b & 0x7fffffffu) > 0x7f800000u)
    b |= (b & 0xffffu) ? 0x10000u : 0;
  else
    b += 0x7fffu + ((b >> 16) & 1u);
  return bits_float(b >> 16);
}

static NX_DEVICE_OBJECT_ABI float truncsfhf2(float x) {
  uint32_t b = float_bits(x);
  uint32_t sign = (b >> 16) & 0x8000u, abs = b & 0x7fffffffu;
  uint32_t h;
  if (abs > 0x7f800000u) {
    h = 0x7c00u | ((abs >> 13) & 0x3ffu) | ((abs & 0x1fffu) ? 1u : 0);
  } else if (abs >= 0x477ff000u) {
    h = 0x7c00u;
  } else if (abs >= 0x38800000u) {
    uint32_t r = abs - 0x38000000u;
    h = (r + 0xfffu + ((r >> 13) & 1u)) >> 13;
  } else if (abs >= 0x33000000u) {
    uint32_t m = (abs & 0x7fffffu) | 0x800000u, shift = 126u - (abs >> 23);
    uint32_t rest = m & ((1u << shift) - 1u), tie = 1u << (shift - 1u);
    h = m >> shift;
    h += rest > tie || (rest == tie && (h & 1u));
  } else {
    h = 0;
  }
  return bits_float(sign | h);
}

static NX_DEVICE_OBJECT_ABI float extendhfsf2(float x) {
  uint32_t h = float_bits(x) & 0xffffu;
  uint32_t sign = (h & 0x8000u) << 16, m = h & 0x3ffu;
  int e = (h >> 10) & 0x1f;
  if (e == 0x1f) return bits_float(sign | 0x7f800000u | m << 13);
  if (e == 0 && m == 0) return bits_float(sign);
  if (e == 0) {
    for (e = 1; !(m & 0x400u); e--) m <<= 1;
    m &= 0x3ffu;
  }
  return bits_float(sign | (uint32_t)(e + 112) << 23 | m << 13);
}

static const struct {
  const char *name;
  NX_DEVICE_OBJECT_ABI float (*fn)(float);
} builtins[] = {
    {"__truncsfbf2", truncsfbf2},
    {"__truncsfhf2", truncsfhf2},
    {"__extendhfsf2", extendhfsf2},
};

/* The address of [name]: the host's copy of a compiler builtin, else the
   definition in the libraries the process loaded, which hold the C and math
   libraries, then in the compiler's runtime. [0] if none defines it. */
value caml_nx_device_symbol(value v_name) {
  const char *name = String_val(v_name);
  void *a = NULL;
  for (size_t i = 0; i < sizeof builtins / sizeof builtins[0]; i++)
    if (strcmp(name, builtins[i].name) == 0)
      return caml_copy_nativeint((intnat)builtins[i].fn);
#ifdef _WIN32
  HANDLE modules = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, 0);
  if (modules != INVALID_HANDLE_VALUE) {
    MODULEENTRY32 m;
    m.dwSize = sizeof m;
    for (BOOL more = Module32First(modules, &m); more && a == NULL;
         more = Module32Next(modules, &m))
      a = (void *)GetProcAddress(m.hModule, name);
    CloseHandle(modules);
  }
#else
  a = dlsym(RTLD_DEFAULT, name);
#ifdef __linux__
  /* Loads run with the host taken, one at a time. */
  static void *rt = NULL;
  if (a == NULL && rt == NULL) rt = dlopen("libgcc_s.so.1", RTLD_LAZY);
  if (a == NULL && rt != NULL) a = dlsym(rt, name);
#endif
#endif
  return caml_copy_nativeint((intnat)a);
}

/* Runs [f(buffers, values)] with the runtime released. The buffers' addresses
   and the values are read first, into memory the collector does not move: from
   Nx_device.Buffer.t values, or from (address, size) pairs when [addresses]. */
#define NX_DEVICE_CALL_WORDS 32

static value call(value v_entry, value v_buffers, value v_values,
                  int addresses) {
  CAMLparam3(v_entry, v_buffers, v_values);
  mlsize_t nb = Wosize_val(v_buffers), nv = Wosize_val(v_values);
  void *small_b[NX_DEVICE_CALL_WORDS];
  int64_t small_v[NX_DEVICE_CALL_WORDS];
  void **b = small_b;
  int64_t *v = small_v;
  if (nb > NX_DEVICE_CALL_WORDS) b = malloc(nb * sizeof *b);
  if (nv > NX_DEVICE_CALL_WORDS) v = malloc(nv * sizeof *v);
  if (b == NULL || v == NULL) {
    if (b != small_b) free(b);
    if (v != small_v) free(v);
    caml_raise_out_of_memory();
  }
  for (mlsize_t i = 0; i < nb; i++)
    b[i] = addresses
               ? (void *)Nativeint_val(Field(Field(v_buffers, i), 0))
               : nx_device_buffer_host(Field(v_buffers, i));
  for (mlsize_t i = 0; i < nv; i++) v[i] = (int64_t)Long_val(Field(v_values, i));
  void (*f)(void **, const int64_t *) =
      (void (*)(void **, const int64_t *))Nativeint_val(v_entry);
  caml_release_runtime_system();
  f(b, v);
  caml_acquire_runtime_system();
  if (b != small_b) free(b);
  if (v != small_v) free(v);
  CAMLreturn(Val_unit);
}

value caml_nx_device_call(value v_entry, value v_buffers, value v_values) {
  return call(v_entry, v_buffers, v_values, 0);
}

/* As [caml_nx_device_call], given each buffer as an (address, size) pair. */
value caml_nx_device_call_addresses(value v_entry, value v_buffers,
                                    value v_values) {
  return call(v_entry, v_buffers, v_values, 1);
}
