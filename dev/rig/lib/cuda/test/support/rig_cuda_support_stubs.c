/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Fills, as compiled code would write them, CUDA's own view of host
   memory, the machine's GPU lock and host memory. CUDA's functions are
   those the device's capability finds, bound once. A fill's argument is a
   bigarray's C memory. Every stub but the lock's and the submit's holds the
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
#include <caml/bigarray.h>
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

#include "rig_cuda.h"

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
value rig_cuda_test_bind(value v_f) {
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
value rig_cuda_test_current(value unit) {
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
value rig_cuda_test_locked(value v_p) {
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
value rig_cuda_test_register(value v_p, value v_n) {
  CUcontext popped;
  push_primary();
  CUresult s = host_register(Addr_val(v_p), Long_val(v_n), 0x3);
  pop(&popped);
  if (s != 0) caml_failwith("cuMemHostRegister");
  return Val_unit;
}

value rig_cuda_test_unregister(value v_p) {
  CUcontext popped;
  push_primary();
  CUresult s = host_unregister(Addr_val(v_p));
  pop(&popped);
  if (s != 0) caml_failwith("cuMemHostUnregister");
  return Val_unit;
}

/* The [v_n] bytes of GPU memory at [v_a], copied by CUDA once every stream
   of the context is done with them. */
value rig_cuda_test_read_gpu(value v_a, value v_n) {
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

value rig_cuda_test_write_gpu(value v_a, value v_s) {
  CUcontext popped;
  push_primary();
  CUresult s = memcpy_htod((uint64_t)Nativeint_val(v_a), String_val(v_s),
                           caml_string_length(v_s));
  pop(&popped);
  if (s != 0) caml_failwith("cuMemcpyHtoD");
  return Val_unit;
}

/* CUDA device 0's attribute [v_a]. */
value rig_cuda_test_attribute(value v_a) {
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
value rig_cuda_test_lock(value v_path, value v_holder) {
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

value rig_cuda_test_page_size(value unit) {
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
value rig_cuda_test_pages(value v_n, value v_read_only) {
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

value rig_cuda_test_free_pages(value v_p, value v_n) {
#if defined(_WIN32)
  (void)v_n;
  VirtualFree(Addr_val(v_p), 0, MEM_RELEASE);
#else
  munmap(Addr_val(v_p), Long_val(v_n));
#endif
  return Val_unit;
}

value rig_cuda_test_get64(value v_p) {
  _Atomic uint64_t *p = Addr_val(v_p);
  return Val_long((intnat)atomic_load_explicit(p, memory_order_acquire));
}

value rig_cuda_test_set64(value v_p, value v_x) {
  _Atomic uint64_t *p = Addr_val(v_p);
  atomic_store_explicit(p, (uint64_t)Long_val(v_x), memory_order_release);
  return Val_unit;
}

value rig_cuda_test_read(value v_p, value v_n) {
  CAMLparam2(v_p, v_n);
  CAMLlocal1(s);
  s = caml_alloc_string(Long_val(v_n));
  memcpy(Bytes_val(s), Addr_val(v_p), Long_val(v_n));
  CAMLreturn(s);
}

value rig_cuda_test_write(value v_p, value v_s) {
  memcpy(Addr_val(v_p), String_val(v_s), caml_string_length(v_s));
  return Val_unit;
}

/* Fills */

#define Arg_val(v) Caml_ba_data_val(v)

/* [n] zeroed bytes of C memory as a bigarray, the argument of a fill: the
   suite hands it to rig as a host buffer. */
static value arg(size_t n) {
  value v = caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT, 1, NULL,
                               (intnat)n);
  memset(Caml_ba_data_val(v), 0, n);
  return v;
}

/* A fill that returns [code]. */
struct failing {
  int code;
};

static int failing(void *queue, void *arg, uint64_t v) {
  (void)queue, (void)v;
  return ((struct failing *)arg)->code;
}

value rig_cuda_test_failing(value v_code) {
  value v = arg(sizeof(struct failing));
  ((struct failing *)Arg_val(v))->code = Int_val(v_code);
  return v;
}

value rig_cuda_test_failing_fill(value unit) {
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

value rig_cuda_test_launch(value v_f, value v_grid, value v_block,
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

value rig_cuda_test_launch_byte(value *argv, int argn) {
  (void)argn;
  return rig_cuda_test_launch(argv[0], argv[1], argv[2], argv[3], argv[4],
                                 argv[5]);
}

value rig_cuda_test_launch_fill(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)launch);
}

value rig_cuda_test_seen(value v_arg) {
  return caml_copy_nativeint((intnat)((struct launch *)Arg_val(v_arg))->seen);
}

/* What rig_cuda_room answers for one fill on queue [v_queue], with one
   ring word iff [v_words], [v_units] ring units, [v_bytes] segment bytes and
   the indices [v_after]. */
value rig_cuda_test_room(value v_self, value v_queue, value v_words,
                            value v_units, value v_bytes, value v_after) {
  static const uint32_t word = 0;
  int after[8];
  struct rig_part p;
  memset(&p, 0, sizeof p);
  p.queue = Int_val(v_queue);
  p.words = Bool_val(v_words) ? &word : NULL;
  p.n = Bool_val(v_words) ? 1 : 0;
  p.fill = failing;
  p.ring_units = Long_val(v_units);
  p.segment_bytes = Long_val(v_bytes);
  p.nafter = (int)Wosize_val(v_after);
  if (p.nafter > 8) caml_invalid_argument("rig_cuda_test_room");
  for (int i = 0; i < p.nafter; i++) after[i] = Int_val(Field(v_after, i));
  p.after = after;
  return Val_int(rig_cuda_room(Ptr_val(v_self), &p, 1));
}

value rig_cuda_test_room_byte(value *argv, int argn) {
  (void)argn;
  return rig_cuda_test_room(argv[0], argv[1], argv[2], argv[3], argv[4],
                               argv[5]);
}

/* What rig_cuda_submit answers for the copies [v_copies] as the value
   [v_v], after the waits [v_waits], an address and a value each: [None] for
   RIG_OK, [Some why] for RIG_FAILED. A copy is the ints queue, dst, src and
   bytes, then its [after] indices. Releases the runtime: the submit may
   block. */
value rig_cuda_test_copies(value v_self, value v_v, value v_waits,
                              value v_copies) {
  CAMLparam4(v_self, v_v, v_waits, v_copies);
  CAMLlocal1(why);
  int nw = (int)(Wosize_val(v_waits) / 2), np = (int)Wosize_val(v_copies);
  size_t nafter = 0;
  for (int i = 0; i < np; i++) nafter += Wosize_val(Field(v_copies, i)) - 4;
  size_t size = nw * sizeof(struct rig_wait) + np * sizeof(struct rig_part) +
                nafter * sizeof(int);
  char *mem = calloc(1, size + 1);
  if (mem == NULL) caml_raise_out_of_memory();
  struct rig_wait *w = (struct rig_wait *)mem;
  struct rig_part *p = (struct rig_part *)(w + nw);
  int *after = (int *)(p + np);
  for (int i = 0; i < nw; i++) {
    w[i].kind = RIG_WORD;
    w[i].at = (uint64_t)Long_val(Field(v_waits, 2 * i));
    w[i].value = (uint64_t)Long_val(Field(v_waits, 2 * i + 1));
  }
  for (int i = 0; i < np; i++) {
    value c = Field(v_copies, i);
    p[i].queue = Int_val(Field(c, 0));
    p[i].copy_dst = (uint64_t)Long_val(Field(c, 1));
    p[i].copy_src = (uint64_t)Long_val(Field(c, 2));
    p[i].copy_bytes = (uint64_t)Long_val(Field(c, 3));
    p[i].after = after;
    p[i].nafter = (int)Wosize_val(c) - 4;
    for (int j = 0; j < p[i].nafter; j++) *after++ = Int_val(Field(c, 4 + j));
  }
  void *self = Ptr_val(v_self);
  uint64_t v = (uint64_t)Long_val(v_v);
  const char *failure = NULL;
  caml_release_runtime_system();
  int rc = rig_cuda_submit(self, v, w, nw, p, np, NULL, 0, &failure);
  caml_acquire_runtime_system();
  free(mem);
  if (rc == RIG_OK) CAMLreturn(Val_none);
  why = caml_copy_string(failure);
  CAMLreturn(caml_alloc_some(why));
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

value rig_cuda_test_delayed(value v_spin, value v_flag, value v_ns,
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

value rig_cuda_test_delayed_byte(value *argv, int argn) {
  (void)argn;
  return rig_cuda_test_delayed(argv[0], argv[1], argv[2], argv[3], argv[4],
                                  argv[5]);
}

value rig_cuda_test_delayed_fill(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)delayed);
}
