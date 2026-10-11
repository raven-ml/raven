/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The host's code: its memory, the process's symbols, and the host's
   driver, which runs the launches of linked code. A failing system call
   returns its error as a negative integer, -errno or -GetLastError (),
   which error_message renders. Every stub holds the runtime: none of them
   blocks. */

#define _GNU_SOURCE

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

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

#include "rig_pool.h"
#include "rig_stubs.h"

/* The host */

value caml_rig_host_error_message(value v_error) {
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

void rig_code_sync(void) { synchronize(); }

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
    "rig.code",      finalize_mapping,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default,
};

/* A mapping of [size] bytes, writable until installed. The block comes
   first, so that the mapping is never left without an owner. */
value caml_rig_host_map(value v_size) {
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
value caml_rig_host_base(value v_mapping) {
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
value caml_rig_host_install(value v_mapping, value v_bytes) {
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

/* Launches

   Linked code follows System V on x86_64, Windows included, whose own
   convention differs: a function pointer to it carries the attribute
   there. A compiler without GNU attributes would call it with the wrong
   convention. */

#if defined(_WIN64) && (defined(__x86_64__) || defined(_M_X64))
#if !defined(__GNUC__)
#error "calling linked code on x86_64 Windows needs __attribute__((sysv_abi))"
#endif
#define SYSV __attribute__((sysv_abi))
#else
#define SYSV
#endif

typedef void(SYSV *entry)(const void *, int64_t, int64_t);

/* Loaded code need not have Clang function-type metadata before its entry.
   That probe can read outside the mapping; other sanitizer checks apply. */
#if defined(__clang__)
__attribute__((no_sanitize("function")))
#endif
static inline void invoke(entry f, const void *params, int64_t first,
                          int64_t end) {
  synchronize();
  f(params, first, end);
}

/* A launch's entry and its parameters, which every range shares. */
typedef struct {
  entry f;
  const void *params;
} job;

static void body(int64_t first, int64_t end, int worker, void *ctx) {
  (void)worker;
  job *j = ctx;
  invoke(j->f, j->params, first, end);
}

/* The most threads a launch has: past it, a thread's index and the end of a
   range would not fit an int64_t. */
#define MOST_THREADS (UINT64_C(1) << 62)

/* The threads of [b]'s grid, or 0 for a grid with an empty axis or more
   than MOST_THREADS threads. [*groups] is its groups, or 0 where the threads
   are 0. */
static uint64_t threads_of(const struct rig_block *b, uint64_t *groups) {
  uint64_t g = 1, t = 1, n;
  *groups = 0;
  for (int i = 0; i < 3; i++) {
    if (__builtin_mul_overflow(g, (uint64_t)b->groups[i], &g) ||
        __builtin_mul_overflow(t, (uint64_t)b->threads[i], &t))
      return 0;
  }
  if (__builtin_mul_overflow(g, t, &n) || n == 0 || n > MOST_THREADS)
    return 0;
  *groups = g;
  return n;
}

static const struct rig_block *block_of(const struct rig_part *p,
                                        const uint8_t *args) {
  return (const struct rig_block *)(const void *)(args + p->launch.args.block);
}

/* The host runs launches with no shared memory over a grid of threads that
   fits threads_of. */
static int host_room(void *self, const struct rig_part *parts, int n,
                     void *scratch, const uint8_t *args) {
  (void)self;
  (void)scratch;
  for (int i = 0; i < n; i++) {
    const struct rig_part *p = &parts[i];
    if (p->kind != RIG_LAUNCH) return RIG_NEVER;
    const struct rig_block *b = block_of(p, args);
    uint64_t groups;
    if (b->shared != 0 || threads_of(b, &groups) == 0) return RIG_NEVER;
  }
  return RIG_FITS;
}

/* Runs launch [p] over its grid: the pool cuts its threads into one chunk
   per group and runs them on at most a thread per performance core. Every
   range shares one copy of the parameters, refs placed. */
static void launch(const struct rig_part *p, const uint8_t *args,
                   const uint64_t *slots) {
  _Alignas(16) uint8_t params[RIG_PARAMS];
  rig_params(params, &p->launch.args, args, slots);
  uint64_t groups, n = threads_of(block_of(p, args), &groups);
  job j = {(entry)(uintptr_t)p->launch.code, params};
  int64_t threads = rig_pool_performance_cores();
  if ((uint64_t)threads > groups) threads = (int64_t)groups;
  rig_pool_run((int)threads, (int64_t)n, (int64_t)groups, body, &j);
}

/* The host's state: its driver and its word. The submitting thread runs the
   launches in order, then makes [v] observable, so every value is reached
   when its submit returns. */
static struct {
  const struct rig_driver *driver;
  _Atomic uint64_t word;
} host;

static int host_submit(void *self, uint64_t v, const struct rig_wait *waits,
                       int nwaits, const struct rig_part *parts, int nparts,
                       void *scratch, const uint8_t *args,
                       const uint64_t *slots, int nslots, uint64_t *times,
                       const char **failure) {
  (void)self;
  (void)scratch;
  (void)waits;
  (void)nwaits;
  (void)nslots;
  (void)failure;
  if (times != NULL) times[0] = rig_now_ns();
  for (int i = 0; i < nparts; i++) launch(&parts[i], args, slots);
  if (times != NULL) times[1] = rig_now_ns();
  atomic_store_explicit(&host.word, v, memory_order_release);
  return RIG_COMMITTED;
}

/* The work ran at its hand-over: nothing is left to commit. */
static int host_commit(void *self, uint64_t v, const char **failure) {
  (void)self;
  (void)v;
  (void)failure;
  return RIG_OK;
}

static const struct rig_driver host_driver = {host_room, host_submit,
                                              host_commit};

void *rig_host_state(void) {
  host.driver = &host_driver;
  return &host;
}

_Atomic uint64_t *rig_host_word(void) { return &host.word; }

value caml_rig_host_word(value unit) {
  (void)unit;
  return Val_long((intnat)&host.word);
}

/* Symbols */

/* The address of the symbol [name] that linked code refers to and does not
   define: its definition in the libraries the process loaded into its global
   scope; 0 if none defines it. */
value caml_rig_host_symbol(value v_name) {
  const char *name = String_val(v_name);
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
