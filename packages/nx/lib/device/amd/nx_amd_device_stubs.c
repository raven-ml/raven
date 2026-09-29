/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#ifdef __linux__
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "kfd_ioctl.h"
#endif

#ifdef __linux__
static void fail_errno(const char *what, int e) {
  char msg[512];
  snprintf(msg, sizeof msg, "%s: %s", what, strerror(e));
  caml_failwith(msg);
}
#else
static void no_linux(void) { caml_failwith("KFD needs Linux"); }
#define LINUX_ONLY(...)                                                        \
  do {                                                                         \
    no_linux();                                                                \
    return Val_unit;                                                           \
  } while (0)
#endif

/* Files and mappings */

value caml_nx_amd_open(value path) {
  CAMLparam1(path);
#ifdef __linux__
  int fd = open(String_val(path), O_RDWR | O_CLOEXEC);
  if (fd < 0) fail_errno(String_val(path), errno);
  CAMLreturn(Val_int(fd));
#else
  (void)path;
  no_linux();
  CAMLreturn(Val_unit);
#endif
}

value caml_nx_amd_close(value fd) {
#ifdef __linux__
  close(Int_val(fd));
#else
  (void)fd;
#endif
  return Val_unit;
}

/* Reserves [n] addresses, inaccessible, anywhere. */
value caml_nx_amd_reserve(value n) {
#ifdef __linux__
  void *p = mmap(NULL, Long_val(n), PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (p == MAP_FAILED) fail_errno("reserving GPU addresses", errno);
  return caml_copy_nativeint((intnat)p);
#else
  (void)n;
  LINUX_ONLY();
#endif
}

/* [n] bytes of shared anonymous memory, readable and writable. */
value caml_nx_amd_anon(value n) {
#ifdef __linux__
  void *p = mmap(NULL, Long_val(n), PROT_READ | PROT_WRITE,
                 MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) fail_errno("allocating host memory for the GPU", errno);
  return caml_copy_nativeint((intnat)p);
#else
  (void)n;
  LINUX_ONLY();
#endif
}

/* Maps [n] bytes of [fd] from [off] at [at], or anywhere if [at] is 0. */
value caml_nx_amd_map(value fd, value at, value n, value off) {
#ifdef __linux__
  void *want = (void *)Nativeint_val(at);
  void *p = mmap(want, Long_val(n), PROT_READ | PROT_WRITE,
                 MAP_SHARED | (want ? MAP_FIXED : 0), Int_val(fd),
                 (off_t)Int64_val(off));
  if (p == MAP_FAILED) fail_errno("mapping GPU memory", errno);
  return caml_copy_nativeint((intnat)p);
#else
  (void)fd;
  (void)at;
  (void)n;
  (void)off;
  LINUX_ONLY();
#endif
}

value caml_nx_amd_unmap(value a, value n) {
#ifdef __linux__
  munmap((void *)Nativeint_val(a), Long_val(n));
#else
  (void)a;
  (void)n;
#endif
  return Val_unit;
}

/* KFD ioctls. An interrupted call is retried, as the kernel asks. Those that
   allocate, pin, map or create queues can take long, and run with the runtime
   released: their arguments are C locals. */

#ifdef __linux__
static int kfd(int fd, unsigned long req, void *arg) {
  int r;
  do r = ioctl(fd, req, arg);
  while (r < 0 && (errno == EINTR || errno == EAGAIN));
  return r < 0 ? errno : 0;
}

static int kfd_blocking(int fd, unsigned long req, void *arg) {
  caml_release_runtime_system();
  int e = kfd(fd, req, arg);
  caml_acquire_runtime_system();
  return e;
}
#endif

value caml_nx_kfd_version(value fd) {
  CAMLparam1(fd);
  CAMLlocal1(r);
#ifdef __linux__
  struct kfd_ioctl_get_version_args a = {0};
  int e = kfd(Int_val(fd), AMDKFD_IOC_GET_VERSION, &a);
  if (e) fail_errno("reading the KFD version", e);
  r = caml_alloc_tuple(2);
  Store_field(r, 0, Val_int(a.major_version));
  Store_field(r, 1, Val_int(a.minor_version));
#else
  (void)fd;
  no_linux();
#endif
  CAMLreturn(r);
}

value caml_nx_kfd_acquire_vm(value fd, value drm, value gpu) {
#ifdef __linux__
  struct kfd_ioctl_acquire_vm_args a = {.drm_fd = Int_val(drm),
                                        .gpu_id = Int_val(gpu)};
  int e = kfd(Int_val(fd), AMDKFD_IOC_ACQUIRE_VM, &a);
  if (e) fail_errno("acquiring the GPU's address space", e);
  return Val_unit;
#else
  (void)fd;
  (void)drm;
  (void)gpu;
  LINUX_ONLY();
#endif
}

value caml_nx_kfd_runtime_enable(value fd) {
#ifdef __linux__
  struct kfd_ioctl_runtime_enable_args a = {0};
  int e = kfd(Int_val(fd), AMDKFD_IOC_RUNTIME_ENABLE, &a);
  if (e && e != EBUSY) fail_errno("enabling the KFD runtime", e);
  return Val_unit;
#else
  (void)fd;
  LINUX_ONLY();
#endif
}

/* Allocates memory; [Ok (handle, mmap_offset)] or [Error errno]. */
value caml_nx_kfd_alloc(value fd, value gpu, value va, value size, value flags,
                        value mmap_offset) {
  CAMLparam5(fd, gpu, va, size, flags);
  CAMLxparam1(mmap_offset);
  CAMLlocal2(r, t);
#ifdef __linux__
  struct kfd_ioctl_alloc_memory_of_gpu_args a = {
      .va_addr = (uint64_t)Nativeint_val(va),
      .size = (uint64_t)Long_val(size),
      .mmap_offset = (uint64_t)Int64_val(mmap_offset),
      .gpu_id = (uint32_t)Int_val(gpu),
      .flags = (uint32_t)Long_val(flags)};
  int e = kfd_blocking(Int_val(fd), AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &a);
  if (e) {
    r = caml_alloc(1, 1);
    Store_field(r, 0, Val_int(e));
  } else {
    t = caml_alloc_tuple(2);
    Store_field(t, 0, caml_copy_int64((int64_t)a.handle));
    Store_field(t, 1, caml_copy_int64((int64_t)a.mmap_offset));
    r = caml_alloc(1, 0);
    Store_field(r, 0, t);
  }
#else
  (void)fd;
  (void)gpu;
  (void)va;
  (void)size;
  (void)flags;
  (void)mmap_offset;
  no_linux();
#endif
  CAMLreturn(r);
}

value caml_nx_kfd_alloc_byte(value *argv, int argn) {
  (void)argn;
  return caml_nx_kfd_alloc(argv[0], argv[1], argv[2], argv[3], argv[4],
                           argv[5]);
}

value caml_nx_kfd_free(value fd, value handle) {
#ifdef __linux__
  struct kfd_ioctl_free_memory_of_gpu_args a = {
      .handle = (uint64_t)Int64_val(handle)};
  int e = kfd_blocking(Int_val(fd), AMDKFD_IOC_FREE_MEMORY_OF_GPU, &a);
  if (e) fail_errno("freeing GPU memory", e);
  return Val_unit;
#else
  (void)fd;
  (void)handle;
  LINUX_ONLY();
#endif
}

/* Maps or unmaps [handle] on the GPU [gpu]; the errno, 0 on success. */
value caml_nx_kfd_map(value fd, value handle, value gpu, value map) {
#ifdef __linux__
  uint32_t ids[1] = {(uint32_t)Int_val(gpu)};
  struct kfd_ioctl_map_memory_to_gpu_args a = {
      .handle = (uint64_t)Int64_val(handle),
      .device_ids_array_ptr = (uint64_t)(uintptr_t)ids,
      .n_devices = 1};
  int e = kfd_blocking(Int_val(fd),
                       Bool_val(map) ? AMDKFD_IOC_MAP_MEMORY_TO_GPU
                                     : AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU,
                       &a);
  if (!e && a.n_success != 1) e = EIO;
  return Val_int(e);
#else
  (void)fd;
  (void)handle;
  (void)gpu;
  (void)map;
  LINUX_ONLY();
#endif
}

/* Creates an event; its id. [page] is the handle of the event page, given to
   the first event of the process. */
value caml_nx_kfd_create_event(value fd, value type, value page) {
#ifdef __linux__
  struct kfd_ioctl_create_event_args a = {
      .event_page_offset = (uint64_t)Int64_val(page),
      .event_type = (uint32_t)Int_val(type),
      .auto_reset = Int_val(type) == KFD_IOC_EVENT_SIGNAL};
  int e = kfd(Int_val(fd), AMDKFD_IOC_CREATE_EVENT, &a);
  if (e) fail_errno("creating a KFD event", e);
  return Val_int(a.event_id);
#else
  (void)fd;
  (void)type;
  (void)page;
  LINUX_ONLY();
#endif
}

/* Creates a queue; (queue id, doorbell offset). */
value caml_nx_kfd_create_queue(value fd, value args) {
  CAMLparam2(fd, args);
  CAMLlocal1(r);
#ifdef __linux__
  struct kfd_ioctl_create_queue_args a = {
      .ring_base_address = (uint64_t)Nativeint_val(Field(args, 0)),
      .ring_size = (uint32_t)Long_val(Field(args, 1)),
      .gpu_id = (uint32_t)Int_val(Field(args, 2)),
      .queue_type = (uint32_t)Int_val(Field(args, 3)),
      .queue_percentage = KFD_MAX_QUEUE_PERCENTAGE,
      .queue_priority = 7,
      .eop_buffer_address = (uint64_t)Nativeint_val(Field(args, 4)),
      .eop_buffer_size = (uint64_t)Long_val(Field(args, 5)),
      .ctx_save_restore_address = (uint64_t)Nativeint_val(Field(args, 6)),
      .ctx_save_restore_size = (uint32_t)Long_val(Field(args, 7)),
      .ctl_stack_size = (uint32_t)Long_val(Field(args, 8)),
      .write_pointer_address = (uint64_t)Nativeint_val(Field(args, 9)),
      .read_pointer_address = (uint64_t)Nativeint_val(Field(args, 10))};
  int e = kfd_blocking(Int_val(fd), AMDKFD_IOC_CREATE_QUEUE, &a);
  if (e) fail_errno("creating a KFD queue", e);
  r = caml_alloc_tuple(2);
  Store_field(r, 0, Val_int(a.queue_id));
  Store_field(r, 1, caml_copy_int64((int64_t)a.doorbell_offset));
#else
  (void)fd;
  (void)args;
  no_linux();
#endif
  CAMLreturn(r);
}

/* Waits at most [ms] for the signal, memory-exception and hardware-exception
   events [ids], with the runtime released, and reports an exception of the
   GPU [gpu]: [""] if there is none. */
value caml_nx_kfd_wait(value fd, value ids, value gpu, value ms) {
  CAMLparam4(fd, ids, gpu, ms);
#ifdef __linux__
  struct kfd_event_data ev[3];
  memset(ev, 0, sizeof ev);
  for (int i = 0; i < 3; i++) ev[i].event_id = Int_val(Field(ids, i));
  struct kfd_ioctl_wait_events_args a = {
      .events_ptr = (uint64_t)(uintptr_t)ev,
      .num_events = 3,
      .wait_for_all = 0,
      .timeout = (uint32_t)Int_val(ms)};
  int kfd_fd = Int_val(fd);
  uint32_t me = (uint32_t)Int_val(gpu);
  caml_release_runtime_system();
  int e = kfd(kfd_fd, AMDKFD_IOC_WAIT_EVENTS, &a);
  caml_acquire_runtime_system();
  if (e) fail_errno("waiting for KFD events", e);
  char msg[512] = "";
  struct kfd_hsa_memory_exception_data *m = &ev[1].memory_exception_data;
  struct kfd_hsa_hw_exception_data *h = &ev[2].hw_exception_data;
  if (m->gpu_id == me)
    snprintf(msg, sizeof msg,
             "memory fault at 0x%llx (not present %u, read-only %u, "
             "no execute %u, imprecise %u, error type %u)",
             (unsigned long long)m->va, m->failure.NotPresent,
             m->failure.ReadOnly, m->failure.NoExecute, m->failure.imprecise,
             m->ErrorType);
  else if (h->gpu_id == me)
    snprintf(msg, sizeof msg,
             "hardware exception (reset type %u, reset cause %u, memory "
             "lost %u)",
             h->reset_type, h->reset_cause, h->memory_lost);
  CAMLreturn(caml_copy_string(msg));
#else
  (void)fd;
  (void)ids;
  (void)gpu;
  (void)ms;
  no_linux();
  CAMLreturn(Val_unit);
#endif
}

/* Milliseconds of a monotonic clock. */
intnat caml_nx_amd_now_ms(value unit) {
  (void)unit;
#ifdef _WIN32
  return (intnat)GetTickCount64();
#else
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (intnat)t.tv_sec * 1000 + t.tv_nsec / 1000000;
#endif
}

value caml_nx_amd_now_ms_byte(value unit) {
  return Val_long(caml_nx_amd_now_ms(unit));
}

value caml_nx_amd_linux(value unit) {
  (void)unit;
#ifdef __linux__
  return Val_true;
#else
  return Val_false;
#endif
}
