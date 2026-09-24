#include <caml/alloc.h>
#include <caml/bigarray.h>
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
#include <time.h>

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
#if defined(_WIN32)
  /* A whole reservation is released at once; the size was fixed at alloc. */
  VirtualFree(ptr, 0, MEM_RELEASE);
#else
  munmap(ptr, (size_t)Long_val(v_size));
#endif
  CAMLreturn(Val_unit);
}

/* Kernel timings rank BEAM candidates, so they need a monotonic clock finer
   than the wall clock: on Windows gettimeofday moves in millisecond steps and
   ties every fast kernel at zero. Nanoseconds since an arbitrary origin. */
CAMLprim value caml_tolk_cpu_monotonic_ns(value unit) {
  (void)unit;
#if defined(_WIN32)
  static LARGE_INTEGER freq;
  LARGE_INTEGER now;
  if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&now);
  int64_t whole = now.QuadPart / freq.QuadPart;
  int64_t rest = now.QuadPart % freq.QuadPart;
  return Val_long(
      (intnat)(whole * 1000000000LL + rest * 1000000000LL / freq.QuadPart));
#else
  struct timespec ts;
#if defined(__APPLE__)
  /* Darwin's adjusted monotonic clock rounds to microseconds, which can time
     a synchronous kernel at zero. Use the raw high-resolution counter. */
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
  clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
  return Val_long((intnat)ts.tv_sec * 1000000000LL + ts.tv_nsec);
#endif
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
#if defined(_WIN32)
    addr = (void *)GetProcAddress(GetModuleHandle(NULL), sym);
#else
    addr = dlsym(RTLD_DEFAULT, sym);
#endif
  }
  if (addr == NULL) {
    caml_failwith("link_symbol failed");
  }
  CAMLreturn(caml_copy_nativeint((intnat)addr));
}
