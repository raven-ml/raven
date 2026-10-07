/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Code memory, the process's symbols, and the calls of linked code. A
   failing system call returns its error as a negative integer, -errno or
   -GetLastError (), which error_message renders. Every stub but the call
   holds the runtime: none of them blocks. */

#define _GNU_SOURCE

#include <stdint.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <errno.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(__APPLE__) && defined(__aarch64__)
#include <pthread.h>
#endif

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

value caml_device_host_page_size(value unit) {
  (void)unit;
#if defined(_WIN32)
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return Val_long(info.dwPageSize);
#else
  return Val_long(sysconf(_SC_PAGESIZE));
#endif
}

value caml_device_host_error_message(value v_error) {
  CAMLparam1(v_error);
#if defined(_WIN32)
  char msg[256];
  DWORD n =
      FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                     NULL, (DWORD)Long_val(v_error), 0, msg, sizeof msg, NULL);
  while (n > 0 && (msg[n - 1] == '\n' || msg[n - 1] == '\r')) n--;
  msg[n] = '\0';
  CAMLreturn(caml_copy_string(msg));
#else
  CAMLreturn(caml_copy_string(strerror((int)Long_val(v_error))));
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

/* Writes [bytes] at the start of [mapping], makes it executable and
   synchronizes the instruction caches with it: 0, or the error. It is the
   only writer of code memory. On arm64 macOS the write window is this
   thread's and closes before it returns. */
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
  pthread_jit_write_protect_np(0);
  memcpy(m->base, Bytes_val(v_bytes), n);
  pthread_jit_write_protect_np(1);
#else
  memcpy(m->base, Bytes_val(v_bytes), n);
  if (mprotect(m->base, m->size, PROT_READ | PROT_EXEC) != 0)
    return Val_long(-errno);
#endif
  __builtin___clear_cache(m->base, m->base + n);
#endif
  return Val_long(0);
}
