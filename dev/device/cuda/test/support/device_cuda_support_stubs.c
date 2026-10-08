/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Fills, as compiled code would write them, CUDA's own view of host
   memory, the machine's GPU lock and host memory. CUDA's functions are
   those the device's capability finds, bound once. A fill's argument is C
   memory that its custom block frees. Every stub but the lock's holds the
   runtime: none blocks. */

#define _GNU_SOURCE

#include <errno.h>
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
#include <caml/threads.h>

#if defined(_WIN32)
#include <windows.h>
#define CUDAAPI __stdcall
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#define CUDAAPI
#endif

#include "device_cuda.h"

#define Ptr_val(v) ((void *)Nativeint_val(v))
#define Addr_val(v) ((void *)Long_val(v))

typedef int CUresult;
typedef void *CUcontext;

/* CUDA, as the capability finds it */

static CUresult(CUDAAPI *launch_kernel)(void *, unsigned int, unsigned int,
                                        unsigned int, unsigned int,
                                        unsigned int, unsigned int,
                                        unsigned int, void *, void **,
                                        void **);
static CUresult(CUDAAPI *get_current)(CUcontext *);
static CUresult(CUDAAPI *retain)(CUcontext *, int);
static CUresult(CUDAAPI *push)(CUcontext);
static CUresult(CUDAAPI *pop)(CUcontext *);
static CUresult(CUDAAPI *device_pointer)(uint64_t *, void *, unsigned int);
static CUresult(CUDAAPI *get_attribute)(int *, int, int);
static CUresult(CUDAAPI *memcpy_async)(uint64_t, uint64_t, size_t, void *);
static CUresult(CUDAAPI *memcpy_dtoh)(void *, uint64_t, size_t);
static CUresult(CUDAAPI *memcpy_htod)(uint64_t, const void *, size_t);
static CUresult(CUDAAPI *host_register)(void *, size_t, unsigned int);
static CUresult(CUDAAPI *host_unregister)(void *);

/* Binds cuLaunchKernel, cuCtxGetCurrent, cuDevicePrimaryCtxRetain,
   cuCtxPushCurrent_v2, cuCtxPopCurrent_v2, cuMemHostGetDevicePointer_v2,
   cuDeviceGetAttribute, cuMemcpyAsync, cuMemcpyDtoH_v2, cuMemcpyHtoD_v2,
   cuMemHostRegister_v2 and cuMemHostUnregister, in this order. */
value device_cuda_test_bind(value v_f) {
  launch_kernel = Ptr_val(Field(v_f, 0));
  get_current = Ptr_val(Field(v_f, 1));
  retain = Ptr_val(Field(v_f, 2));
  push = Ptr_val(Field(v_f, 3));
  pop = Ptr_val(Field(v_f, 4));
  device_pointer = Ptr_val(Field(v_f, 5));
  get_attribute = Ptr_val(Field(v_f, 6));
  memcpy_async = Ptr_val(Field(v_f, 7));
  memcpy_dtoh = Ptr_val(Field(v_f, 8));
  memcpy_htod = Ptr_val(Field(v_f, 9));
  host_register = Ptr_val(Field(v_f, 10));
  host_unregister = Ptr_val(Field(v_f, 11));
  return Val_unit;
}

/* The calling thread's current context. */
value device_cuda_test_current(value unit) {
  CUcontext c = NULL;
  (void)unit;
  if (get_current(&c) != 0) c = NULL;
  return caml_copy_nativeint((intnat)c);
}

/* Makes the primary context of CUDA's device 0 current above the
   thread's own. */
static void push_primary(void) {
  static CUcontext context = NULL;
  if (context == NULL && retain(&context, 0) != 0) caml_failwith("retain");
  if (push(context) != 0) caml_failwith("push");
}

/* Whether CUDA holds the host memory at [v_p] page-locked and mapped. */
value device_cuda_test_locked(value v_p) {
  CUcontext popped;
  uint64_t d = 0;
  push_primary();
  CUresult s = device_pointer(&d, Addr_val(v_p), 0);
  pop(&popped);
  return Val_bool(s == 0);
}

/* Page-locks the [v_n] bytes of host memory at [v_p] for every device and
   maps them (CU_MEMHOST_PORTABLE | CU_MEMHOST_DEVICEMAP), as another library
   would. */
value device_cuda_test_register(value v_p, value v_n) {
  CUcontext popped;
  push_primary();
  CUresult s = host_register(Addr_val(v_p), Long_val(v_n), 0x3);
  pop(&popped);
  if (s != 0) caml_failwith("cuMemHostRegister");
  return Val_unit;
}

value device_cuda_test_unregister(value v_p) {
  CUcontext popped;
  push_primary();
  CUresult s = host_unregister(Addr_val(v_p));
  pop(&popped);
  if (s != 0) caml_failwith("cuMemHostUnregister");
  return Val_unit;
}

/* The [v_n] bytes of GPU memory at [v_a], copied by CUDA once every stream
   of the context is done with them. */
value device_cuda_test_read_gpu(value v_a, value v_n) {
  CAMLparam2(v_a, v_n);
  CAMLlocal1(r);
  CUcontext popped;
  size_t n = Long_val(v_n);
  char *buf = malloc(n);
  if (buf == NULL) caml_raise_out_of_memory();
  push_primary();
  CUresult s = memcpy_dtoh(buf, (uint64_t)Nativeint_val(v_a), n);
  pop(&popped);
  if (s != 0) {
    free(buf);
    caml_failwith("cuMemcpyDtoH");
  }
  r = caml_alloc_initialized_string(n, buf);
  free(buf);
  CAMLreturn(r);
}

value device_cuda_test_write_gpu(value v_a, value v_s) {
  CUcontext popped;
  push_primary();
  CUresult s = memcpy_htod((uint64_t)Nativeint_val(v_a), String_val(v_s),
                           caml_string_length(v_s));
  pop(&popped);
  if (s != 0) caml_failwith("cuMemcpyHtoD");
  return Val_unit;
}

/* CUDA device 0's attribute [v_a]. */
value device_cuda_test_attribute(value v_a) {
  int a = 0;
  if (get_attribute(&a, Int_val(v_a), 0) != 0) caml_failwith("attribute");
  return Val_int(a);
}

/* The GPU lock */

/* One try at the exclusive lock of the file [v_path], which the process
   then holds until it exits. A missing file is made writable by every user
   of the machine. Once taken, the file names [v_holder] and the process's
   id, for the processes that wait. Answers [0] once the process holds the
   lock, [-1] after a nap of 100 ms if another process holds it, or the
   errno of a failing call. Releases the runtime for the nap. */
value device_cuda_test_lock(value v_path, value v_holder) {
#if defined(_WIN32)
  (void)v_path;
  (void)v_holder;
  return Val_int(ENOSYS);
#else
  /* The descriptor that holds the lock once taken. The suites take it from
     one domain. */
  static int held = -1;
  if (held >= 0) return Val_int(0);
  const char *path = String_val(v_path);
  int fd = open(path, O_RDWR | O_CLOEXEC);
  /* O_EXCL: Linux refuses O_CREAT on another user's file in /tmp
     (fs.protected_regular). */
  if (fd < 0 && errno == ENOENT) {
    fd = open(path, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
    if (fd < 0 && errno == EEXIST) fd = open(path, O_RDWR | O_CLOEXEC);
    else if (fd >= 0 && fchmod(fd, 0666) != 0) {
      int e = errno;
      close(fd);
      return Val_int(e);
    }
  }
  if (fd < 0) return Val_int(errno);
  if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
    int e = errno;
    close(fd);
    if (e != EWOULDBLOCK) return Val_int(e);
    struct timespec nap = {0, 100 * 1000 * 1000};
    caml_release_runtime_system();
    nanosleep(&nap, NULL);
    caml_acquire_runtime_system();
    return Val_int(-1);
  }
  char note[1024] = "";
  snprintf(note, sizeof note, "%s, pid %ld\n", String_val(v_holder),
           (long)getpid());
  size_t len = strlen(note);
  if (ftruncate(fd, 0) != 0 || pwrite(fd, note, len, 0) != (ssize_t)len) {
    int e = errno;
    close(fd);
    return Val_int(e);
  }
  held = fd;
  return Val_int(0);
#endif
}

/* Host memory */

value device_cuda_test_page_size(value unit) {
  (void)unit;
#if defined(_WIN32)
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return Val_long(info.dwPageSize);
#else
  return Val_long(sysconf(_SC_PAGESIZE));
#endif
}

/* [v_n] zeroed bytes from a page, writable, or read-only if
   [v_read_only]. */
value device_cuda_test_pages(value v_n, value v_read_only) {
  size_t n = Long_val(v_n);
#if defined(_WIN32)
  void *p = VirtualAlloc(NULL, n, MEM_COMMIT | MEM_RESERVE,
                         Bool_val(v_read_only) ? PAGE_READONLY
                                               : PAGE_READWRITE);
  if (p == NULL) caml_raise_out_of_memory();
#else
  int prot = Bool_val(v_read_only) ? PROT_READ : PROT_READ | PROT_WRITE;
  void *p = mmap(NULL, n, prot, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) caml_raise_out_of_memory();
#endif
  return Val_long((intnat)p);
}

value device_cuda_test_free_pages(value v_p, value v_n) {
#if defined(_WIN32)
  (void)v_n;
  VirtualFree(Addr_val(v_p), 0, MEM_RELEASE);
#else
  munmap(Addr_val(v_p), Long_val(v_n));
#endif
  return Val_unit;
}

value device_cuda_test_get64(value v_p) {
  _Atomic uint64_t *p = Addr_val(v_p);
  return Val_long((intnat)atomic_load_explicit(p, memory_order_acquire));
}

value device_cuda_test_set64(value v_p, value v_x) {
  _Atomic uint64_t *p = Addr_val(v_p);
  atomic_store_explicit(p, (uint64_t)Long_val(v_x), memory_order_release);
  return Val_unit;
}

value device_cuda_test_read(value v_p, value v_n) {
  CAMLparam2(v_p, v_n);
  CAMLlocal1(s);
  s = caml_alloc_string(Long_val(v_n));
  memcpy(Bytes_val(s), Addr_val(v_p), Long_val(v_n));
  CAMLreturn(s);
}

value device_cuda_test_write(value v_p, value v_s) {
  memcpy(Addr_val(v_p), String_val(v_s), caml_string_length(v_s));
  return Val_unit;
}

/* Fills */

#define Arg_val(v) (*(void **)Data_custom_val(v))

static void finalize_arg(value v) { free(Arg_val(v)); }

static struct custom_operations arg_ops = {
    "device_cuda_test.arg",     finalize_arg,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default,
};

/* A block owning [n] zeroed bytes of C memory, the argument of a fill. */
static value arg(size_t n) {
  void *p = calloc(1, n);
  if (p == NULL) caml_raise_out_of_memory();
  value v = caml_alloc_custom(&arg_ops, sizeof(void *), 0, 1);
  Arg_val(v) = p;
  return v;
}

value device_cuda_test_arg_address(value v_arg) {
  return caml_copy_nativeint((intnat)Arg_val(v_arg));
}

/* A fill that returns [code]. */
struct failing {
  int code;
};

static int failing(void *queue, void *arg, uint64_t v) {
  (void)queue, (void)v;
  return ((struct failing *)arg)->code;
}

value device_cuda_test_failing(value v_code) {
  value v = arg(sizeof(struct failing));
  ((struct failing *)Arg_val(v))->code = Int_val(v_code);
  return v;
}

value device_cuda_test_failing_fill(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)failing);
}

/* A fill that launches [count] times the kernel [f] over [grid] blocks of
   [block] threads with the two 64-bit parameters [a] and [b], and records
   the context current while it ran. */
struct launch {
  void *f;
  unsigned int grid, block;
  int count;
  uint64_t a, b;
  CUcontext seen;
};

static int launch(void *queue, void *arg, uint64_t v) {
  struct launch *l = arg;
  void *params[2] = {&l->a, &l->b};
  (void)v;
  if (get_current(&l->seen) != 0) l->seen = NULL;
  for (int i = 0; i < l->count; i++) {
    CUresult s = launch_kernel(l->f, l->grid, 1, 1, l->block, 1, 1, 0, queue,
                               params, NULL);
    if (s != 0) return s;
  }
  return 0;
}

value device_cuda_test_launch(value v_f, value v_grid, value v_block,
                              value v_count, value v_a, value v_b) {
  value v = arg(sizeof(struct launch));
  struct launch *l = Arg_val(v);
  l->f = (void *)Long_val(v_f);
  l->grid = (unsigned int)Long_val(v_grid);
  l->block = (unsigned int)Long_val(v_block);
  l->count = Int_val(v_count);
  l->a = (uint64_t)Long_val(v_a);
  l->b = (uint64_t)Long_val(v_b);
  return v;
}

value device_cuda_test_launch_byte(value *argv, int argn) {
  (void)argn;
  return device_cuda_test_launch(argv[0], argv[1], argv[2], argv[3], argv[4],
                                 argv[5]);
}

value device_cuda_test_launch_fill(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)launch);
}

value device_cuda_test_seen(value v_arg) {
  return caml_copy_nativeint((intnat)((struct launch *)Arg_val(v_arg))->seen);
}

/* What device_cuda_room answers for one fill on queue [v_queue], with one
   ring word iff [v_words], [v_units] ring units, [v_bytes] segment bytes and
   the indices [v_after]. */
value device_cuda_test_room(value v_self, value v_queue, value v_words,
                            value v_units, value v_bytes, value v_after) {
  static const uint32_t word = 0;
  int after[8];
  struct nx_part p;
  memset(&p, 0, sizeof p);
  p.queue = Int_val(v_queue);
  p.words = Bool_val(v_words) ? &word : NULL;
  p.n = Bool_val(v_words) ? 1 : 0;
  p.fill = failing;
  p.ring_units = Long_val(v_units);
  p.segment_bytes = Long_val(v_bytes);
  p.nafter = (int)Wosize_val(v_after);
  if (p.nafter > 8) caml_invalid_argument("device_cuda_test_room");
  for (int i = 0; i < p.nafter; i++) after[i] = Int_val(Field(v_after, i));
  p.after = after;
  return Val_int(device_cuda_room(Ptr_val(v_self), &p, 1));
}

value device_cuda_test_room_byte(value *argv, int argn) {
  (void)argn;
  return device_cuda_test_room(argv[0], argv[1], argv[2], argv[3], argv[4],
                               argv[5]);
}

/* A fill that runs the kernel [spin] for [ns] nanoseconds, then copies [n]
   bytes from [src] to [dst]: a copy that starts late. */
struct delayed {
  void *spin;
  uint64_t flag, ns, dst, src, n;
};

static int delayed(void *queue, void *arg, uint64_t v) {
  struct delayed *d = arg;
  void *params[2] = {&d->flag, &d->ns};
  (void)v;
  CUresult s =
      launch_kernel(d->spin, 1, 1, 1, 1, 1, 1, 0, queue, params, NULL);
  if (s != 0) return s;
  return memcpy_async(d->dst, d->src, d->n, queue);
}

value device_cuda_test_delayed(value v_spin, value v_flag, value v_ns,
                               value v_dst, value v_src, value v_n) {
  value v = arg(sizeof(struct delayed));
  struct delayed *d = Arg_val(v);
  d->spin = (void *)Long_val(v_spin);
  d->flag = (uint64_t)Long_val(v_flag);
  d->ns = (uint64_t)Long_val(v_ns);
  d->dst = (uint64_t)Long_val(v_dst);
  d->src = (uint64_t)Long_val(v_src);
  d->n = (uint64_t)Long_val(v_n);
  return v;
}

value device_cuda_test_delayed_byte(value *argv, int argn) {
  (void)argn;
  return device_cuda_test_delayed(argv[0], argv[1], argv[2], argv[3], argv[4],
                                  argv[5]);
}

value device_cuda_test_delayed_fill(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)delayed);
}
