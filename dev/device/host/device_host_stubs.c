/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Code memory, the process's symbols, and the calls of linked code. A
   failing system call returns its error as a negative integer, -errno or
   -GetLastError (), which error_message renders. Every stub but the call
   holds the runtime: none of them blocks. */

#define _GNU_SOURCE

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#if defined(_WIN32)
#include <windows.h>
/* After windows.h, which it needs. */
#include <tlhelp32.h>
#else
#include <dlfcn.h>
#include <errno.h>
#include <sys/mman.h>
#endif

#if defined(__APPLE__) && defined(__aarch64__)
#include <pthread.h>
#endif

#include "nx_pool.h"

/* The host */

/* The host's ELF machine: EM_X86_64, EM_AARCH64, or 0 for another. */
value caml_device_host_machine(value unit) {
  (void)unit;
#if defined(__x86_64__) || defined(_M_X64)
  return Val_int(62);
#elif defined(__aarch64__) || defined(_M_ARM64)
  return Val_int(183);
#else
  return Val_int(0);
#endif
}

value caml_device_host_error_message(value v_error) {
#if defined(_WIN32)
  char msg[256];
  DWORD n =
      FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                     NULL, (DWORD)Long_val(v_error), 0, msg, sizeof msg, NULL);
  while (n > 0 && (msg[n - 1] == '\n' || msg[n - 1] == '\r')) n--;
  msg[n] = '\0';
  return caml_copy_string(msg);
#else
  return caml_copy_string(strerror((int)Long_val(v_error)));
#endif
}

/* Instruction streams

   A core that runs code another core wrote must synchronize its instruction
   stream with it: the writer's cache maintenance reaches every core, but
   only a context synchronization (isb) makes a core refetch, and a pool
   worker takes none between jobs. [installed] counts the installs of code;
   before each call of linked code a thread synchronizes if code was
   installed since it last did, which costs a load when none was. The
   writer's cache maintenance completes before its count is published. x86
   keeps its instruction stream coherent with stores. */

#if defined(__aarch64__)
static _Atomic uint64_t installed;
static _Thread_local uint64_t synchronized;
#endif

static inline void synchronize(void) {
#if defined(__aarch64__)
  uint64_t n = atomic_load_explicit(&installed, memory_order_acquire);
  if (n == synchronized) return;
  __asm__ volatile("isb" ::: "memory");
  synchronized = n;
#endif
}

/* Code memory

   A program's code lives in a mapping of its own, which a custom block owns
   and unmaps when the collector frees it. The mapping is never writable and
   executable at once. On arm64 macOS it is MAP_JIT memory, mapped
   read-write-execute, whose write protection each thread lifts for itself;
   elsewhere it is mapped read-write, written once, then made read-execute. A
   block whose mapping failed holds no memory and the error. */

typedef struct {
  char *base;
  size_t size;
  intnat error;
} mapping;

#define Mapping_val(v) ((mapping *)Data_custom_val(v))

static void finalize_mapping(value v) {
  mapping *m = Mapping_val(v);
  if (m->base == NULL) return;
#if defined(_WIN32)
  VirtualFree(m->base, 0, MEM_RELEASE);
#else
  munmap(m->base, m->size);
#endif
}

static struct custom_operations mapping_ops = {
    "device_host.mapping",      finalize_mapping,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default,
};

/* A mapping of [size] bytes, writable until installed. The block comes
   first, so that the mapping is never left without an owner. */
value caml_device_host_map(value v_size) {
  CAMLparam1(v_size);
  CAMLlocal1(v);
  size_t size = (size_t)Long_val(v_size);
  v = caml_alloc_custom_mem(&mapping_ops, sizeof(mapping), size);
  mapping *m = Mapping_val(v);
  *m = (mapping){NULL, size, 0};
#if defined(_WIN32)
  m->base = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  if (m->base == NULL) m->error = -(intnat)GetLastError();
#else
  int prot = PROT_READ | PROT_WRITE, flags = MAP_PRIVATE | MAP_ANON;
#if defined(__APPLE__) && defined(__aarch64__)
  prot |= PROT_EXEC;
  flags |= MAP_JIT;
#endif
  void *p = mmap(NULL, size, prot, flags, -1, 0);
  if (p == MAP_FAILED)
    m->error = -errno;
  else
    m->base = p;
#endif
  CAMLreturn(v);
}

/* The address of [mapping], or its error. */
value caml_device_host_base(value v_mapping) {
  mapping *m = Mapping_val(v_mapping);
  return Val_long(m->base ? (intnat)m->base : m->error);
}

#if defined(__APPLE__) && defined(__aarch64__)
/* On arm64 macOS the system lifts this thread's write protection around
   write_code alone. Under the jit-write-allowlist entitlement it runs only
   the callbacks an image listed when the process started, and ends the
   process for any other. The system reads an image's list up to its first
   NULL, so this is the one list of the image that links this file. */
typedef struct {
  char *to;
  const char *from;
  size_t n;
} code_write;

static int write_code(void *ctx) {
  code_write *w = ctx;
  memcpy(w->to, w->from, w->n);
  return 0;
}

PTHREAD_JIT_WRITE_ALLOW_CALLBACKS_NP(write_code);
#endif

/* Writes [bytes] at the start of [mapping], makes it executable and
   synchronizes the instruction caches with it: 0, or the error. It is the
   only writer of code memory. */
value caml_device_host_install(value v_mapping, value v_bytes) {
  mapping *m = Mapping_val(v_mapping);
  size_t n = caml_string_length(v_bytes);
#if defined(_WIN32)
  DWORD old;
  memcpy(m->base, Bytes_val(v_bytes), n);
  if (!VirtualProtect(m->base, m->size, PAGE_EXECUTE_READ, &old))
    return Val_long(-(intnat)GetLastError());
  FlushInstructionCache(GetCurrentProcess(), m->base, n);
#else
#if defined(__APPLE__) && defined(__aarch64__)
  code_write w = {m->base, (const char *)Bytes_val(v_bytes), n};
  pthread_jit_write_with_callback_np(write_code, &w);
#else
  memcpy(m->base, Bytes_val(v_bytes), n);
  if (mprotect(m->base, m->size, PROT_READ | PROT_EXEC) != 0)
    return Val_long(-errno);
#endif
  __builtin___clear_cache(m->base, m->base + n);
#endif
#if defined(__aarch64__)
  atomic_fetch_add_explicit(&installed, 1, memory_order_release);
#endif
  return Val_long(0);
}

/* Calls

   Linked code follows System V on x86_64, Windows included, whose own
   convention differs: a function pointer to it, and device_host_call, carry
   the attribute there. A compiler without GNU attributes would call it with
   the wrong convention. */

#if defined(_WIN64) && (defined(__x86_64__) || defined(_M_X64))
#if !defined(__GNUC__)
#error "calling linked code on x86_64 Windows needs __attribute__((sysv_abi))"
#endif
#define SYSV __attribute__((sysv_abi))
#else
#define SYSV
#endif

typedef void(SYSV *program)(void **, const int64_t *);

/* The copies of the values that a split's calls take, on the stack when they
   fit. Each worker's copy takes whole cache lines, of 128 bytes on the hosts
   with the largest, so that no two workers write one line. */
#define SMALL_WORDS 1024
#define LINE_WORDS 16

static size_t stride_of(int64_t n) {
  return ((size_t)n + LINE_WORDS - 1) / LINE_WORDS * LINE_WORDS;
}

typedef struct {
  program f;
  void **buffers;
  int64_t *copies; /* a copy of the values per worker, [stride] apart */
  size_t stride;
  int64_t lo, hi;
} job;

static void body(int64_t first, int64_t last, int worker, void *ctx) {
  job *j = ctx;
  int64_t *v = j->copies + (size_t)worker * j->stride;
  v[j->lo] = first;
  v[j->hi] = last;
  synchronize();
  j->f(j->buffers, v);
}

/* The threads a split of {extent, blocks, lo, hi} runs on: one copy of the
   values each. The pool's blocks are at most [extent], and its threads at
   most its blocks. */
static int threads_of(const int64_t *split) {
  int64_t t = nx_pool_performance_cores();
  if (t > split[1]) t = split[1];
  if (t > split[0]) t = split[0] > 0 ? split[0] : 1;
  return (int)t;
}

/* [f] split by [split] on [threads] threads, each the pool's worker of its
   copy of the [n] values in [copies]. The first copy holds the values; the
   others are filled here. */
static void run_split(program f, void **buffers, int64_t *copies, int64_t n,
                      const int64_t *split, int threads) {
  size_t stride = stride_of(n);
  for (int w = 1; w < threads; w++)
    memcpy(copies + (size_t)w * stride, copies, (size_t)n * sizeof *copies);
  job j = {f, buffers, copies, stride, split[2], split[3]};
  nx_pool_run(threads, split[0], split[1], body, &j);
}

/* A split call of linked code. Its copies come from the stack when they fit,
   else from malloc, else from the stack for as many threads as fit: at
   least one, since n <= SMALL_WORDS. It is out of line, so that an unsplit
   call reserves no stack for the copies. */
static __attribute__((noinline)) void call_split(program f, void **buffers,
                                                 const int64_t *values,
                                                 int64_t n,
                                                 const int64_t *split) {
  int threads = threads_of(split);
  size_t stride = stride_of(n);
  _Alignas(128) int64_t small[SMALL_WORDS];
  int64_t *copies = small;
  if ((size_t)threads * stride > SMALL_WORDS) {
    copies = malloc((size_t)threads * stride * sizeof *copies);
    if (copies == NULL) {
      copies = small;
      threads = (int)(SMALL_WORDS / stride);
    }
  }
  memcpy(copies, values, (size_t)n * sizeof *copies);
  run_split(f, buffers, copies, n, split, threads);
  if (copies != small) free(copies);
}

/* The entry of linked code. */
SYSV void device_host_call(program f, void **buffers, const int64_t *values,
                           int64_t n, const int64_t *split) {
  if (split != NULL) {
    call_split(f, buffers, values, n, split);
    return;
  }
  synchronize();
  f(buffers, values);
}

/* Calls the program at [entry] on [buffers] and [values], split by [split]
   if it is [Some], with the runtime released. The buffers' addresses and the
   copies of the values are in C memory, reserved before the release. [code]
   is the program's mapping, which the root keeps mapped while it runs. */
value caml_device_host_call(value v_entry, value v_code, value v_buffers,
                            value v_values, value v_split) {
  CAMLparam5(v_entry, v_code, v_buffers, v_values, v_split);
  size_t nb = Wosize_val(v_buffers), n = Wosize_val(v_values);
  int64_t split[4], *s = NULL;
  int threads = 1;
  size_t words = n;
  if (Is_some(v_split)) {
    for (int i = 0; i < 4; i++)
      split[i] = Long_val(Field(Some_val(v_split), i));
    s = split;
    threads = threads_of(s);
    words = (size_t)threads * stride_of((int64_t)n);
  }
  void *small_b[SMALL_WORDS], **buffers = small_b;
  _Alignas(128) int64_t small_v[SMALL_WORDS];
  int64_t *copies = small_v;
  if (nb > SMALL_WORDS) buffers = malloc(nb * sizeof *buffers);
  if (words > SMALL_WORDS) copies = malloc(words * sizeof *copies);
  if (buffers == NULL || copies == NULL) {
    if (buffers != small_b) free(buffers);
    if (copies != small_v) free(copies);
    caml_raise_out_of_memory();
  }
  for (size_t i = 0; i < nb; i++)
    buffers[i] = (void *)Long_val(Field(v_buffers, i));
  for (size_t i = 0; i < n; i++) copies[i] = Long_val(Field(v_values, i));
  program f = (program)Long_val(v_entry);
  caml_release_runtime_system();
  if (s == NULL) {
    synchronize();
    f(buffers, copies);
  } else {
    run_split(f, buffers, copies, (int64_t)n, s, threads);
  }
  caml_acquire_runtime_system();
  if (buffers != small_b) free(buffers);
  if (copies != small_v) free(copies);
  CAMLreturn(Val_unit);
}

value caml_device_host_workers(value unit) {
  (void)unit;
  return Val_int(nx_pool_performance_cores());
}

/* Symbols */

/* The address of the symbol [name] that linked code refers to and does not
   define: device_host_call, else the definition in the libraries the process
   loaded into its global scope; 0 if none defines it. */
value caml_device_host_symbol(value v_name) {
  const char *name = String_val(v_name);
  if (strcmp(name, "device_host_call") == 0)
    return Val_long((intnat)device_host_call);
#if defined(_WIN32)
  void *a = NULL;
  HANDLE modules = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, 0);
  if (modules == INVALID_HANDLE_VALUE) return Val_long(0);
  MODULEENTRY32 m;
  m.dwSize = sizeof m;
  for (BOOL more = Module32First(modules, &m); more && a == NULL;
       more = Module32Next(modules, &m))
    a = (void *)GetProcAddress(m.hModule, name);
  CloseHandle(modules);
  return Val_long((intnat)a);
#else
  return Val_long((intnat)dlsym(RTLD_DEFAULT, name));
#endif
}
