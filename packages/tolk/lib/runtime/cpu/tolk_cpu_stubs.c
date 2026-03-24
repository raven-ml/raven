#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#if defined(_WIN32)
#include <malloc.h>
#else
#include <alloca.h>
#endif
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#include <dlfcn.h>
#include <sys/mman.h>
#endif

/* MAP_ANON is the BSD spelling of MAP_ANONYMOUS; older systems may only define one. */
#if !defined(_WIN32)
#ifndef MAP_ANON
#define MAP_ANON MAP_ANONYMOUS
#endif

#ifndef MAP_JIT
#define MAP_JIT 0x0800
#endif
#endif

static void *jit_alloc_executable(size_t size) {
#if defined(_WIN32)
  return VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
#else
  int flags = MAP_PRIVATE | MAP_ANON;
#if defined(__APPLE__)
  flags |= MAP_JIT;
#endif
  void *mem = mmap(NULL, size, PROT_READ | PROT_WRITE | PROT_EXEC, flags, -1, 0);
  if (mem == MAP_FAILED) {
    return NULL;
  }
  return mem;
#endif
}

CAMLprim value caml_tolk_cpu_alloc(value v_size) {
  CAMLparam1(v_size);
  size_t size = (size_t)Long_val(v_size);
  void *ptr = calloc(1, size);
  if (ptr == NULL) {
    caml_failwith("cpu_alloc failed");
  }
  CAMLreturn(caml_copy_nativeint((intnat)ptr));
}

CAMLprim value caml_tolk_cpu_free(value v_ptr) {
  CAMLparam1(v_ptr);
  void *ptr = (void *)Nativeint_val(v_ptr);
  free(ptr);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_cpu_copyin(value v_ptr, value v_bytes) {
  CAMLparam2(v_ptr, v_bytes);
  void *ptr = (void *)Nativeint_val(v_ptr);
  size_t len = (size_t)caml_string_length(v_bytes);
  memcpy(ptr, Bytes_val(v_bytes), len);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_cpu_copyout(value v_bytes, value v_ptr) {
  CAMLparam2(v_bytes, v_ptr);
  void *ptr = (void *)Nativeint_val(v_ptr);
  size_t len = (size_t)caml_string_length(v_bytes);
  memcpy(Bytes_val(v_bytes), ptr, len);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_cpu_jit_alloc(value v_size) {
  CAMLparam1(v_size);
  size_t size = (size_t)Long_val(v_size);
  void *mem = jit_alloc_executable(size);
  if (mem == NULL) {
    caml_failwith("jit_alloc failed");
  }
  CAMLreturn(caml_copy_nativeint((intnat)mem));
}

CAMLprim value caml_tolk_cpu_jit_free(value v_ptr, value v_size) {
  CAMLparam2(v_ptr, v_size);
  void *ptr = (void *)Nativeint_val(v_ptr);
  size_t size = (size_t)Long_val(v_size);
#if defined(_WIN32)
  VirtualFree(ptr, 0, MEM_RELEASE);
#else
  munmap(ptr, size);
#endif
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_cpu_jit_write(value v_ptr, value v_bytes) {
  CAMLparam2(v_ptr, v_bytes);
  void *ptr = (void *)Nativeint_val(v_ptr);
  size_t len = (size_t)caml_string_length(v_bytes);
#if defined(__APPLE__)
  pthread_jit_write_protect_np(0);
#endif
  memcpy(ptr, Bytes_val(v_bytes), len);
#if defined(__APPLE__)
  pthread_jit_write_protect_np(1);
#endif
#if defined(_WIN32)
  FlushInstructionCache(GetCurrentProcess(), ptr, len);
#else
  __builtin___clear_cache((char *)ptr, (char *)ptr + len);
#endif
  CAMLreturn(Val_unit);
}

/* Tinygrad passes each buffer address and scalar value as individual C function
   arguments (varargs-style via ctypes). Tolk uses a fixed two-pointer convention:
   fn(const uint64_t *bufs, const int64_t *vals). This simplifies the FFI to a
   constant-arity call regardless of kernel argument count. The C and LLVM IR
   renderers must generate code that reads from these arrays (bufs[i], vals[j])
   rather than named parameters. */
CAMLprim value caml_tolk_cpu_jit_call(value v_entry, value v_bufs, value v_vals) {
  CAMLparam3(v_entry, v_bufs, v_vals);
  size_t nbufs = (size_t)Wosize_val(v_bufs);
  size_t nvals = (size_t)Wosize_val(v_vals);
  uint64_t *bufs = nbufs > 0 ? (uint64_t *)alloca(sizeof(uint64_t) * nbufs) : NULL;
  int64_t *vals = nvals > 0 ? (int64_t *)alloca(sizeof(int64_t) * nvals) : NULL;
  for (size_t i = 0; i < nbufs; ++i) {
    bufs[i] = (uint64_t)Nativeint_val(Field(v_bufs, i));
  }
  for (size_t i = 0; i < nvals; ++i) {
    vals[i] = (int64_t)Int64_val(Field(v_vals, i));
  }
  void (*fn)(const uint64_t *, const int64_t *) =
      (void (*)(const uint64_t *, const int64_t *))Nativeint_val(v_entry);
  caml_release_runtime_system();
  fn(bufs, vals);
  caml_acquire_runtime_system();
  CAMLreturn(Val_unit);
}

static void *try_dlopen(const char *name) {
#if defined(_WIN32)
  void *handle = (void *)LoadLibraryA(name);
  if (handle != NULL) {
    return handle;
  }
  char buf[256];
  if (snprintf(buf, sizeof(buf), "%s.dll", name) > 0) {
    handle = (void *)LoadLibraryA(buf);
    if (handle != NULL) {
      return handle;
    }
  }
  if (snprintf(buf, sizeof(buf), "lib%s.dll", name) > 0) {
    handle = (void *)LoadLibraryA(buf);
    if (handle != NULL) {
      return handle;
    }
  }
  return NULL;
#else
  void *handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
  if (handle != NULL) {
    return handle;
  }
  char buf[256];
  if (snprintf(buf, sizeof(buf), "lib%s.so", name) > 0) {
    handle = dlopen(buf, RTLD_LAZY | RTLD_LOCAL);
    if (handle != NULL) {
      return handle;
    }
  }
  if (snprintf(buf, sizeof(buf), "lib%s.dylib", name) > 0) {
    handle = dlopen(buf, RTLD_LAZY | RTLD_LOCAL);
    if (handle != NULL) {
      return handle;
    }
  }
  return NULL;
#endif
}

CAMLprim value caml_tolk_cpu_jit_link_symbol(value v_libs, value v_sym) {
  CAMLparam2(v_libs, v_sym);
  const char *sym = String_val(v_sym);
  void *addr = NULL;

  size_t nlibs = (size_t)Wosize_val(v_libs);
#if defined(_WIN32)
  if (nlibs > 0) {
    for (size_t i = 0; i < nlibs && addr == NULL; ++i) {
      const char *lib = String_val(Field(v_libs, i));
      void *handle = try_dlopen(lib);
      if (handle != NULL) {
        addr = (void *)GetProcAddress((HMODULE)handle, sym);
      }
    }
  }
#else
  if (nlibs > 0) {
    for (size_t i = 0; i < nlibs && addr == NULL; ++i) {
      const char *lib = String_val(Field(v_libs, i));
      void *handle = try_dlopen(lib);
      if (handle != NULL) {
        addr = dlsym(handle, sym);
      }
    }
  }
#endif

  if (addr == NULL) {
    caml_failwith("link_symbol failed");
  }
  CAMLreturn(caml_copy_nativeint((intnat)addr));
}
