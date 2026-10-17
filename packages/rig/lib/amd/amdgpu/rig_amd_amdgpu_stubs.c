/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The system calls the amdgpu path makes: Linux's amdgpu driver's files,
   their ioctls, packed in the structures of kfd_ioctl.h and amdgpu_drm.h so
   that the compiler lays them out, and the process's mappings of their
   memory. The structures come from pinned excerpts of the headers
   (gen/headers, made by gen/gen.py), so the build machine's headers, which
   may predate them, play no part. A descriptor is an int. A stub raises
   Unix.Unix_error for the kernel's refusal. A request may wait, so each
   releases the runtime; the other stubs hold it.

   Off Linux every stub raises ENOSYS, so an open fails at its first
   file. */

#define _GNU_SOURCE

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <caml/unixsupport.h>

#include "rig_amd.h"

#define Ptr_val(v) ((void *)Long_val(v))

#ifdef __linux__
#include <fcntl.h>
#include <linux/types.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "gen/headers/drm.h"
#include "gen/headers/amdgpu_drm.h"
#include "gen/headers/kfd_ioctl.h"

/* [(r, v)]: a result and a 64-bit value. */
static value with64(long r, uint64_t v) {
  CAMLparam0();
  CAMLlocal2(t, boxed);
  boxed = caml_copy_int64(v);
  t = caml_alloc_tuple(2);
  Store_field(t, 0, Val_long(r));
  Store_field(t, 1, boxed);
  CAMLreturn(t);
}

/* Opens the device file [v_path]: its descriptor. A GPU's first open
   initialises it, so this releases the runtime. */
value caml_rig_amd_amdgpu_open(value v_path) {
  CAMLparam1(v_path);
  char *path = caml_stat_strdup(String_val(v_path));
  caml_release_runtime_system();
  int fd = open(path, O_RDWR | O_CLOEXEC);
  int e = errno;
  caml_acquire_runtime_system();
  caml_stat_free(path);
  if (fd < 0) caml_unix_error(e, "open", v_path);
  CAMLreturn(Val_int(fd));
}

value caml_rig_amd_amdgpu_close(value v_fd) {
  close(Int_val(v_fd));
  return Val_unit;
}

/* Requests */

/* Makes the request [request] of [fd] with [arg], again while it is
   interrupted or refused for now, as the kernel asks: 0, or -errno.
   Releases the runtime: a wait blocks, and [arg] is not the OCaml heap's. */
static int call(int fd, unsigned long request, void *arg) {
  int r;
  caml_release_runtime_system();
  do r = ioctl(fd, request, arg);
  while (r < 0 && (errno == EINTR || errno == EAGAIN));
  int e = errno;
  caml_acquire_runtime_system();
  return r < 0 ? -e : 0;
}

/* Raises Unix.Unix_error for the -errno [e] a request answered. */
static void check(int e) {
  if (e < 0) caml_unix_error(-e, "ioctl", Nothing);
}

/* KFD's version, major * 1000 + minor. */
value caml_rig_amd_amdgpu_version(value v_kfd) {
  struct kfd_ioctl_get_version_args a = {0};
  check(call(Int_val(v_kfd), AMDKFD_IOC_GET_VERSION, &a));
  return Val_int((int)(a.major_version * 1000 + a.minor_version));
}

/* Gives the process the address space of GPU [v_gpu], whose render node is
   open as [v_drm]. */
value caml_rig_amd_amdgpu_acquire_vm(value v_kfd, value v_drm, value v_gpu) {
  struct kfd_ioctl_acquire_vm_args a = {.drm_fd = Int_val(v_drm),
                                        .gpu_id = Int_val(v_gpu)};
  check(call(Int_val(v_kfd), AMDKFD_IOC_ACQUIRE_VM, &a));
  return Val_unit;
}

value caml_rig_amd_amdgpu_runtime_enable(value v_kfd) {
  struct kfd_ioctl_runtime_enable_args a = {0};
  check(call(Int_val(v_kfd), AMDKFD_IOC_RUNTIME_ENABLE, &a));
  return Val_unit;
}

/* The flags of the memory type's constructors, in order: GPU memory, GPU
   memory the host maps through the BAR, system memory the kernel driver
   owns, the process's own memory, and the page of registers the driver
   remaps. */
#define SHARED                                                           \
  (KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED | \
   KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC)
static const uint32_t memories[] = {
    KFD_IOC_ALLOC_MEM_FLAGS_VRAM,
    KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC,
    KFD_IOC_ALLOC_MEM_FLAGS_GTT | SHARED,
    KFD_IOC_ALLOC_MEM_FLAGS_USERPTR | SHARED,
    KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP};
#define USERPTR 3
#define MMIO 4

/* Allocates [v_bytes] bytes of the memory kind [v_m] for GPU [v_gpu] at
   the address [v_va]: [(handle, mmap_offset)]. All but the registers are
   executable. */
value caml_rig_amd_amdgpu_alloc(value v_kfd, value v_gpu, value v_va,
                                value v_bytes, value v_m) {
  int m = Int_val(v_m);
  struct kfd_ioctl_alloc_memory_of_gpu_args a = {
      .va_addr = Long_val(v_va),
      .size = Long_val(v_bytes),
      .gpu_id = Int_val(v_gpu),
      .flags = memories[m] | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
               KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE |
               (m == MMIO ? 0 : KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE)};
  /* The process's own memory is named by its address. */
  if (m == USERPTR) a.mmap_offset = a.va_addr;
  check(call(Int_val(v_kfd), AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &a));
  return with64((long)a.handle, a.mmap_offset);
}

value caml_rig_amd_amdgpu_free(value v_kfd, value v_handle) {
  struct kfd_ioctl_free_memory_of_gpu_args a = {.handle = Long_val(v_handle)};
  check(call(Int_val(v_kfd), AMDKFD_IOC_FREE_MEMORY_OF_GPU, &a));
  return Val_unit;
}

/* Maps [v_handle] for GPU [v_gpu], or unmaps it if [v_map] is false: the
   GPUs it reached. KFD maps and unmaps with the same layout. */
value caml_rig_amd_amdgpu_map_gpu(value v_kfd, value v_handle, value v_gpu,
                                  value v_map) {
  uint32_t gpu = Int_val(v_gpu);
  struct kfd_ioctl_map_memory_to_gpu_args a = {
      .handle = Long_val(v_handle),
      .device_ids_array_ptr = (uintptr_t)&gpu,
      .n_devices = 1};
  int e = Bool_val(v_map)
              ? call(Int_val(v_kfd), AMDKFD_IOC_MAP_MEMORY_TO_GPU, &a)
              : call(Int_val(v_kfd), AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &a);
  check(e);
  return Val_int((int)a.n_success);
}

/* The event types of the event type's constructors, in order. */
static const uint32_t events[] = {KFD_IOC_EVENT_SIGNAL, KFD_IOC_EVENT_MEMORY,
                                  KFD_IOC_EVENT_HW_EXCEPTION};

/* Makes an event of kind [v_k] on the event page whose handle is [v_page],
   or 0 for events after the page's first: its id. A signal
   event resets as a wait takes it. */
value caml_rig_amd_amdgpu_event(value v_kfd, value v_k, value v_page) {
  struct kfd_ioctl_create_event_args a = {.event_page_offset = Long_val(v_page),
                                          .event_type = events[Int_val(v_k)],
                                          .auto_reset = Int_val(v_k) == 0};
  check(call(Int_val(v_kfd), AMDKFD_IOC_CREATE_EVENT, &a));
  return Val_long((long)a.event_id);
}

value caml_rig_amd_amdgpu_destroy_event(value v_kfd, value v_id) {
  struct kfd_ioctl_destroy_event_args a = {.event_id = Int_val(v_id)};
  check(call(Int_val(v_kfd), AMDKFD_IOC_DESTROY_EVENT, &a));
  return Val_unit;
}

static int reset_event(int kfd, uint32_t id) {
  struct kfd_ioctl_reset_event_args a = {.event_id = id};
  return call(kfd, AMDKFD_IOC_RESET_EVENT, &a);
}

/* The queue types of the queue type's constructors, in order. */
static const uint32_t queues[] = {KFD_IOC_QUEUE_TYPE_COMPUTE,
                                  KFD_IOC_QUEUE_TYPE_COMPUTE_AQL,
                                  KFD_IOC_QUEUE_TYPE_SDMA};

/* The priority of the path's queues, of 0 to 15. */
#define QUEUE_PRIORITY 7

/* Makes a queue of kind [v_k] on GPU [v_gpu] from its ring, end-of-pipe
   buffer and context save area (addresses and bytes), its control stack's
   bytes, and the addresses of its write and read positions:
   [(id, doorbell_offset)]. */
value caml_rig_amd_amdgpu_queue(value v_kfd, value v_k, value v_gpu,
                                value v_ring, value v_ring_bytes, value v_eop,
                                value v_eop_bytes, value v_save,
                                value v_save_bytes, value v_ctl_stack,
                                value v_write, value v_read) {
  struct kfd_ioctl_create_queue_args a = {
      .ring_base_address = Long_val(v_ring),
      .write_pointer_address = Long_val(v_write),
      .read_pointer_address = Long_val(v_read),
      .ring_size = Long_val(v_ring_bytes),
      .gpu_id = Int_val(v_gpu),
      .queue_type = queues[Int_val(v_k)],
      .queue_percentage = KFD_MAX_QUEUE_PERCENTAGE,
      .queue_priority = QUEUE_PRIORITY,
      .eop_buffer_address = Long_val(v_eop),
      .eop_buffer_size = Long_val(v_eop_bytes),
      .ctx_save_restore_address = Long_val(v_save),
      .ctx_save_restore_size = Long_val(v_save_bytes),
      .ctl_stack_size = Long_val(v_ctl_stack)};
  check(call(Int_val(v_kfd), AMDKFD_IOC_CREATE_QUEUE, &a));
  return with64((long)a.queue_id, a.doorbell_offset);
}

value caml_rig_amd_amdgpu_queue_byte(value *argv, int argn) {
  (void)argn;
  return caml_rig_amd_amdgpu_queue(argv[0], argv[1], argv[2], argv[3], argv[4],
                                   argv[5], argv[6], argv[7], argv[8], argv[9],
                                   argv[10], argv[11]);
}

value caml_rig_amd_amdgpu_destroy_queue(value v_kfd, value v_id) {
  struct kfd_ioctl_destroy_queue_args a = {.queue_id = Int_val(v_id)};
  check(call(Int_val(v_kfd), AMDKFD_IOC_DESTROY_QUEUE, &a));
  return Val_unit;
}

/* The most events a wait waits for: a signal and the two exception
   events. */
#define MOST_EVENTS 3

/* Waits at most [v_ms] for any of the events [v_ids], a signal, then a
   memory and a hardware exception event, or the two exception events
   alone. Is [Some f] for a fault of GPU [v_me] an exception event reports,
   [f] a fault, else [None], allocating nothing.

   An exception event another GPU's fault set does not reset itself, and
   would end every later wait at once: the wait resets it. */
value caml_rig_amd_amdgpu_wait(value v_kfd, value v_ids, value v_ms,
                               value v_me) {
  CAMLparam0();
  CAMLlocal1(f);
  int kfd = Int_val(v_kfd), n = Wosize_val(v_ids);
  uint32_t me = Int_val(v_me);
  if (n < 2 || n > MOST_EVENTS)
    caml_invalid_argument("kfd_wait: 2 or 3 events");
  struct kfd_event_data ev[MOST_EVENTS];
  memset(ev, 0, sizeof ev);
  for (int i = 0; i < n; i++) ev[i].event_id = Int_val(Field(v_ids, i));
  struct kfd_ioctl_wait_events_args a = {
      .events_ptr = (uintptr_t)ev, .num_events = n, .timeout = Int_val(v_ms)};
  check(call(kfd, AMDKFD_IOC_WAIT_EVENTS, &a));
  const struct kfd_hsa_memory_exception_data *m =
      &ev[n - 2].memory_exception_data;
  const struct kfd_hsa_hw_exception_data *h = &ev[n - 1].hw_exception_data;
  if (m->gpu_id != 0 && m->gpu_id != me) reset_event(kfd, ev[n - 2].event_id);
  if (h->gpu_id != 0 && h->gpu_id != me) reset_event(kfd, ev[n - 1].event_id);
  if (m->gpu_id == me) {
    f = caml_alloc_small(6, 0);
    Field(f, 0) = Val_long((intnat)m->va);
    Field(f, 1) = Val_int(m->failure.NotPresent);
    Field(f, 2) = Val_int(m->failure.ReadOnly);
    Field(f, 3) = Val_int(m->failure.NoExecute);
    Field(f, 4) = Val_int(m->failure.imprecise);
    Field(f, 5) = Val_int(m->ErrorType);
  } else if (h->gpu_id == me) {
    f = caml_alloc_small(3, 1);
    Field(f, 0) = Val_int(h->reset_type);
    Field(f, 1) = Val_int(h->reset_cause);
    Field(f, 2) = Val_int(h->memory_lost);
  } else
    CAMLreturn(Val_none);
  CAMLreturn(caml_alloc_some(f));
}

/* The render node */

/* The GPU's clock in kHz and its active compute units, 16 masks
   engine-major as cu_bitmap lays them out: [(clock, cus)]. */
value caml_rig_amd_amdgpu_device_info(value v_drm) {
  CAMLparam0();
  CAMLlocal2(t, cus);
  struct drm_amdgpu_info_device d;
  memset(&d, 0, sizeof d);
  struct drm_amdgpu_info a = {.return_pointer = (uintptr_t)&d,
                              .return_size = sizeof d,
                              .query = AMDGPU_INFO_DEV_INFO};
  check(call(Int_val(v_drm), DRM_IOCTL_AMDGPU_INFO, &a));
  int n = sizeof d.cu_bitmap / sizeof d.cu_bitmap[0][0];
  cus = caml_alloc_tuple(n);
  for (int i = 0; i < n; i++)
    Store_field(cus, i, Val_long(d.cu_bitmap[i / 4][i % 4]));
  t = caml_alloc_tuple(2);
  Store_field(t, 0, Val_long((long)d.gpu_counter_freq));
  Store_field(t, 1, cus);
  CAMLreturn(t);
}

/* Holds the GPU in its stable power state with a new context on the render
   node [v_drm], until the process closes it. */
value caml_rig_amd_amdgpu_stable_pstate(value v_drm) {
  int drm = Int_val(v_drm);
  union drm_amdgpu_ctx c = {.in = {.op = AMDGPU_CTX_OP_ALLOC_CTX}};
  check(call(drm, DRM_IOCTL_AMDGPU_CTX, &c));
  uint32_t id = c.out.alloc.ctx_id;
  c = (union drm_amdgpu_ctx){.in = {.op = AMDGPU_CTX_OP_SET_STABLE_PSTATE,
                                    .flags = AMDGPU_CTX_STABLE_PSTATE_STANDARD,
                                    .ctx_id = id}};
  int e = call(drm, DRM_IOCTL_AMDGPU_CTX, &c);
  if (e < 0) {
    c = (union drm_amdgpu_ctx){
        .in = {.op = AMDGPU_CTX_OP_FREE_CTX, .ctx_id = id}};
    call(drm, DRM_IOCTL_AMDGPU_CTX, &c);
  }
  check(e);
  return Val_unit;
}

/* Memory */

/* [n] addresses, inaccessible, anywhere: their address. */
value caml_rig_amd_amdgpu_reserve(value v_n) {
  void *p = mmap(NULL, (size_t)Long_val(v_n), PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (p == MAP_FAILED) caml_unix_error(errno, "mmap", Nothing);
  return Val_long((intnat)p);
}

/* Maps [n] bytes of [fd] from the offset [off] at [at], or anywhere if
   [at] is 0: the address. */
value caml_rig_amd_amdgpu_map(value v_fd, value v_at, value v_n, value v_off) {
  void *want = Ptr_val(v_at);
  void *p = mmap(want, (size_t)Long_val(v_n), PROT_READ | PROT_WRITE,
                 MAP_SHARED | (want ? MAP_FIXED : 0), Int_val(v_fd),
                 (off_t)Int64_val(v_off));
  if (p == MAP_FAILED) caml_unix_error(errno, "mmap", Nothing);
  return Val_long((intnat)p);
}

value caml_rig_amd_amdgpu_unmap(value v_at, value v_n) {
  munmap(Ptr_val(v_at), (size_t)Long_val(v_n));
  return Val_unit;
}

/* Stores [v] at the host address [at], after every store before it. */
value caml_rig_amd_amdgpu_store64(value v_at, value v_v) {
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  *(volatile uint64_t *)Ptr_val(v_at) = (uint64_t)Long_val(v_v);
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  return Val_unit;
}

/* The GPU's clock

   AMDKFD_IOC_GET_CLOCK_COUNTERS reads the GPU's clock counter, the counter
   of the queues' timestamps, which ROCr converts its dispatches' times by
   (amdgpu_amdkfd_get_gpu_clock_counter, GpuAgent::TranslateTime). KFD
   samples CLOCK_MONOTONIC_RAW after the counter read returns
   (kfd_ioctl_get_clock_counters, ktime_get_raw_ns). The gap between the
   two is one-sided, the host sample later, usually one register read, and
   unbounded: interrupts may run between them, and the task may be
   preempted before the sample. The width below does not include that
   gap. The reader shifts the sample to rig's
   clock by the two clocks' difference, from a reading of
   CLOCK_MONOTONIC_RAW between two of rig's clock right after the call;
   their interval is the read's width. */

/* A GPU's reader: KFD's file and the GPU's id there. */
struct counters {
  struct rig_amd_clock clock;
  int kfd;
  uint32_t gpu;
};

static int read_counters(struct rig_amd_clock *c, uint64_t *ticks,
                         uint64_t *host, uint64_t *width) {
  struct counters *k = (struct counters *)c;
  struct kfd_ioctl_get_clock_counters_args a;
  memset(&a, 0, sizeof a);
  a.gpu_id = k->gpu;
  int r;
  do r = ioctl(k->kfd, AMDKFD_IOC_GET_CLOCK_COUNTERS, &a);
  while (r < 0 && errno == EINTR);
  if (r < 0 || a.gpu_clock_counter == 0) return -1;
  struct timespec raw;
  uint64_t before = rig_now_ns();
  clock_gettime(CLOCK_MONOTONIC_RAW, &raw);
  uint64_t after = rig_now_ns();
  uint64_t raw_ns = (uint64_t)raw.tv_sec * 1000000000u + (uint64_t)raw.tv_nsec;
  *ticks = a.gpu_clock_counter;
  *host = a.cpu_clock_counter + (before + (after - before) / 2 - raw_ns);
  *width = after - before;
  return 0;
}

/* A reader of the clock of the GPU [v_gpu] through KFD's file [v_kfd]: the
   address of its struct rig_amd_clock, which lives for the process. Raises
   [Out_of_memory] without memory for it. */
value caml_rig_amd_amdgpu_clock(value v_kfd, value v_gpu) {
  struct counters *k = malloc(sizeof *k);
  if (k == NULL) caml_raise_out_of_memory();
  k->clock.read = read_counters;
  k->kfd = Int_val(v_kfd);
  k->gpu = (uint32_t)Long_val(v_gpu);
  return Val_long((intnat)k);
}

#else

#define NONE(name, ...)                                                   \
  value caml_rig_amd_amdgpu_##name(__VA_ARGS__) {                         \
    caml_unix_error(ENOSYS, #name, Nothing);                             \
  }

#define UNUSED __attribute__((unused))
NONE(open, value a UNUSED)
NONE(close, value a UNUSED)
NONE(version, value a UNUSED)
NONE(acquire_vm, value a UNUSED, value b UNUSED, value c UNUSED)
NONE(runtime_enable, value a UNUSED)
NONE(alloc, value a UNUSED, value b UNUSED, value c UNUSED, value d UNUSED,
     value e UNUSED)
NONE(free, value a UNUSED, value b UNUSED)
NONE(map_gpu, value a UNUSED, value b UNUSED, value c UNUSED, value d UNUSED)
NONE(event, value a UNUSED, value b UNUSED, value c UNUSED)
NONE(destroy_event, value a UNUSED, value b UNUSED)
NONE(queue_byte, value *a UNUSED, int b UNUSED)
NONE(queue, value a UNUSED, value b UNUSED, value c UNUSED, value d UNUSED,
     value e UNUSED, value f UNUSED, value g UNUSED, value h UNUSED,
     value i UNUSED, value j UNUSED, value k UNUSED, value l UNUSED)
NONE(destroy_queue, value a UNUSED, value b UNUSED)
NONE(wait, value a UNUSED, value b UNUSED, value c UNUSED, value d UNUSED)
NONE(device_info, value a UNUSED)
NONE(stable_pstate, value a UNUSED)
NONE(reserve, value a UNUSED)
NONE(map, value a UNUSED, value b UNUSED, value c UNUSED, value d UNUSED)
NONE(unmap, value a UNUSED, value b UNUSED)
NONE(store64, value a UNUSED, value b UNUSED)
NONE(clock, value a UNUSED, value b UNUSED)

#endif
