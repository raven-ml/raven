/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Linux's amdgpu driver: KFD's compute interface (/dev/kfd) and the render
   node's queries. A stub answers the kernel's refusal as -errno, or 0, and
   writes what it reads into a [bytes] or [int array] its caller gives; no
   stub builds an OCaml value but a string. An interrupted request is made
   again, as the kernel asks. Only [wait] releases the runtime; the others
   are short.

   Off Linux every stub answers -ENOSYS; the library calls none there. */

#define _GNU_SOURCE

#include <errno.h>
#include <stdint.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

value caml_rig_amd_amdgpu_strerror(value v_e) {
  return caml_copy_string(strerror(Int_val(v_e)));
}

#ifdef __linux__
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "amdgpu_drm.h"
#include "kfd_ioctl.h"

#define Ptr_val(v) ((void *)Long_val(v))

static int request(int fd, unsigned long req, void *arg) {
  int r;
  do r = ioctl(fd, req, arg);
  while (r < 0 && (errno == EINTR || errno == EAGAIN));
  return r < 0 ? -errno : 0;
}

static void put64(value b, int i, uint64_t v) {
  memcpy(Bytes_val(b) + 8 * i, &v, 8);
}

value caml_rig_amd_amdgpu_linux(value unit) {
  (void)unit;
  return Val_true;
}

/* Files and mappings */

value caml_rig_amd_amdgpu_open(value v_path) {
  int fd = open(String_val(v_path), O_RDWR | O_CLOEXEC);
  return Val_int(fd < 0 ? -errno : fd);
}

value caml_rig_amd_amdgpu_close(value v_fd) {
  close(Int_val(v_fd));
  return Val_unit;
}

/* [n] addresses, inaccessible, anywhere: their address, or -errno. */
value caml_rig_amd_amdgpu_reserve(value v_n) {
  void *p = mmap(NULL, (size_t)Long_val(v_n), PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  return Val_long(p == MAP_FAILED ? -errno : (intnat)p);
}

/* Maps [n] bytes of [fd] from the offset in the first 8 bytes of [b] at
   [at], or anywhere if [at] is 0: the address, or -errno. */
value caml_rig_amd_amdgpu_map(value v_fd, value v_at, value v_n, value v_b) {
  void *want = Ptr_val(v_at);
  uint64_t off;
  memcpy(&off, Bytes_val(v_b), 8);
  void *p = mmap(want, (size_t)Long_val(v_n), PROT_READ | PROT_WRITE,
                 MAP_SHARED | (want ? MAP_FIXED : 0), Int_val(v_fd), (off_t)off);
  return Val_long(p == MAP_FAILED ? -errno : (intnat)p);
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

/* KFD */

/* The interface's version, major * 1000 + minor, or -errno. */
value caml_rig_amd_amdgpu_version(value v_fd) {
  struct kfd_ioctl_get_version_args a = {0};
  int e = request(Int_val(v_fd), AMDKFD_IOC_GET_VERSION, &a);
  return Val_int(e ? e : (int)(a.major_version * 1000 + a.minor_version));
}

value caml_rig_amd_amdgpu_acquire_vm(value v_fd, value v_drm, value v_gpu) {
  struct kfd_ioctl_acquire_vm_args a = {.drm_fd = (uint32_t)Int_val(v_drm),
                                        .gpu_id = (uint32_t)Int_val(v_gpu)};
  return Val_int(request(Int_val(v_fd), AMDKFD_IOC_ACQUIRE_VM, &a));
}

value caml_rig_amd_amdgpu_runtime_enable(value v_fd) {
  struct kfd_ioctl_runtime_enable_args a = {0};
  int e = request(Int_val(v_fd), AMDKFD_IOC_RUNTIME_ENABLE, &a);
  return Val_int(e == -EBUSY ? 0 : e);
}

/* The kinds of memory the path allocates, in Rig_amd_amdgpu's order:
   GPU memory, GPU memory the host maps through the BAR, system memory the
   kernel driver owns, the process's own memory, and the page of
   registers the driver remaps. */
static const uint32_t kinds[] = {
    KFD_IOC_ALLOC_MEM_FLAGS_VRAM,
    KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC,
    KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
        KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED | KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC,
    KFD_IOC_ALLOC_MEM_FLAGS_USERPTR | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
        KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED | KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC,
    KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP,
};

/* Allocates [n] bytes of kind [k] for GPU [gpu] at the address [va]: 0,
   with the handle and the offset to map it at in [b], or -errno. */
value caml_rig_amd_amdgpu_alloc(value v_fd, value v_gpu, value v_va,
                                   value v_n, value v_k, value v_b) {
  int k = Int_val(v_k);
  uint32_t flags = kinds[k] | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                   KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE;
  if (k != 4) flags |= KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE;
  struct kfd_ioctl_alloc_memory_of_gpu_args a = {
      .va_addr = (uint64_t)Long_val(v_va),
      .size = (uint64_t)Long_val(v_n),
      .mmap_offset = k == 3 ? (uint64_t)Long_val(v_va) : 0,
      .gpu_id = (uint32_t)Int_val(v_gpu),
      .flags = flags};
  int e = request(Int_val(v_fd), AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &a);
  if (e) return Val_int(e);
  put64(v_b, 0, a.handle);
  put64(v_b, 1, a.mmap_offset);
  return Val_int(0);
}

value caml_rig_amd_amdgpu_alloc_byte(value *argv, int argn) {
  (void)argn;
  return caml_rig_amd_amdgpu_alloc(argv[0], argv[1], argv[2], argv[3],
                                      argv[4], argv[5]);
}

value caml_rig_amd_amdgpu_free(value v_fd, value v_handle) {
  struct kfd_ioctl_free_memory_of_gpu_args a = {
      .handle = (uint64_t)Long_val(v_handle)};
  return Val_int(request(Int_val(v_fd), AMDKFD_IOC_FREE_MEMORY_OF_GPU, &a));
}

/* Maps or unmaps [handle] for GPU [gpu]. */
value caml_rig_amd_amdgpu_map_gpu(value v_fd, value v_handle, value v_gpu,
                                     value v_map) {
  uint32_t ids[1] = {(uint32_t)Int_val(v_gpu)};
  struct kfd_ioctl_map_memory_to_gpu_args a = {
      .handle = (uint64_t)Long_val(v_handle),
      .device_ids_array_ptr = (uint64_t)(uintptr_t)ids,
      .n_devices = 1};
  int e = request(Int_val(v_fd),
                  Bool_val(v_map) ? AMDKFD_IOC_MAP_MEMORY_TO_GPU
                                  : AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU,
                  &a);
  return Val_int(e == 0 && a.n_success != 1 ? -EIO : e);
}

/* Events, in Rig_amd_amdgpu's order: signal, memory exception, hardware
   exception. */
static const uint32_t events[] = {KFD_IOC_EVENT_SIGNAL, KFD_IOC_EVENT_MEMORY,
                                  KFD_IOC_EVENT_HW_EXCEPTION};

/* A new event of kind [k], on the event page whose handle is [page] (the
   process's first signal event names it, the others 0): its id, or -errno.
   A signal event resets as a wait takes it. */
value caml_rig_amd_amdgpu_event(value v_fd, value v_k, value v_page) {
  int k = Int_val(v_k);
  struct kfd_ioctl_create_event_args a = {
      .event_page_offset = (uint64_t)Long_val(v_page),
      .event_type = events[k],
      .auto_reset = k == 0};
  int e = request(Int_val(v_fd), AMDKFD_IOC_CREATE_EVENT, &a);
  return Val_long(e ? e : (intnat)a.event_id);
}

value caml_rig_amd_amdgpu_destroy_event(value v_fd, value v_id) {
  struct kfd_ioctl_destroy_event_args a = {.event_id = (uint32_t)Int_val(v_id)};
  return Val_int(request(Int_val(v_fd), AMDKFD_IOC_DESTROY_EVENT, &a));
}

static const uint32_t queue_types[] = {KFD_IOC_QUEUE_TYPE_COMPUTE,
                                       KFD_IOC_QUEUE_TYPE_COMPUTE_AQL,
                                       KFD_IOC_QUEUE_TYPE_SDMA};

/* A queue of type [k] (PM4, AQL, SDMA) from the ints of [q]: GPU, ring,
   ring bytes, EOP buffer and its bytes, context save area and its bytes,
   control stack bytes, write and read positions. 0, with the queue's id and
   its doorbell's offset in [b], or -errno. */
value caml_rig_amd_amdgpu_queue(value v_fd, value v_k, value v_q,
                                   value v_b) {
#define Q(i) Long_val(Field(v_q, i))
  struct kfd_ioctl_create_queue_args a = {
      .gpu_id = (uint32_t)Q(0),
      .ring_base_address = (uint64_t)Q(1),
      .ring_size = (uint32_t)Q(2),
      .eop_buffer_address = (uint64_t)Q(3),
      .eop_buffer_size = (uint64_t)Q(4),
      .ctx_save_restore_address = (uint64_t)Q(5),
      .ctx_save_restore_size = (uint32_t)Q(6),
      .ctl_stack_size = (uint32_t)Q(7),
      .write_pointer_address = (uint64_t)Q(8),
      .read_pointer_address = (uint64_t)Q(9),
      .queue_type = queue_types[Int_val(v_k)],
      .queue_percentage = KFD_MAX_QUEUE_PERCENTAGE,
      .queue_priority = 7};
#undef Q
  int e = request(Int_val(v_fd), AMDKFD_IOC_CREATE_QUEUE, &a);
  if (e) return Val_int(e);
  put64(v_b, 0, a.queue_id);
  put64(v_b, 1, a.doorbell_offset);
  return Val_int(0);
}

value caml_rig_amd_amdgpu_destroy_queue(value v_fd, value v_id) {
  struct kfd_ioctl_destroy_queue_args a = {.queue_id = (uint32_t)Int_val(v_id)};
  return Val_int(request(Int_val(v_fd), AMDKFD_IOC_DESTROY_QUEUE, &a));
}

/* Waits at most [ms] for the events [ids] (signal, memory, hardware), with
   the runtime released. 0 if none reports a fault of GPU [gpu]; 1, a
   memory fault, with its address, not-present, read-only, no-execute,
   imprecise and error type in [r]; 2, a hardware exception, with its reset
   type, reset cause and lost memory in [r]; or -errno. */
value caml_rig_amd_amdgpu_wait(value v_fd, value v_ids, value v_gpu,
                                  value v_ms, value v_r) {
  CAMLparam5(v_fd, v_ids, v_gpu, v_ms, v_r);
  struct kfd_event_data ev[3];
  memset(ev, 0, sizeof ev);
  for (int i = 0; i < 3; i++) ev[i].event_id = (uint32_t)Int_val(Field(v_ids, i));
  struct kfd_ioctl_wait_events_args a = {.events_ptr = (uint64_t)(uintptr_t)ev,
                                         .num_events = 3,
                                         .wait_for_all = 0,
                                         .timeout = (uint32_t)Int_val(v_ms)};
  int fd = Int_val(v_fd);
  uint32_t me = (uint32_t)Int_val(v_gpu);
  caml_release_runtime_system();
  int e = request(fd, AMDKFD_IOC_WAIT_EVENTS, &a);
  caml_acquire_runtime_system();
  if (e) CAMLreturn(Val_int(e));
  struct kfd_hsa_memory_exception_data *m = &ev[1].memory_exception_data;
  struct kfd_hsa_hw_exception_data *h = &ev[2].hw_exception_data;
  if (m->gpu_id == me) {
    intnat f[] = {(intnat)m->va, m->failure.NotPresent, m->failure.ReadOnly,
                  m->failure.NoExecute, m->failure.imprecise, m->ErrorType};
    for (int i = 0; i < 6; i++) Store_field(v_r, i, Val_long(f[i]));
    CAMLreturn(Val_int(1));
  }
  if (h->gpu_id == me) {
    intnat f[] = {h->reset_type, h->reset_cause, h->memory_lost};
    for (int i = 0; i < 3; i++) Store_field(v_r, i, Val_long(f[i]));
    CAMLreturn(Val_int(2));
  }
  CAMLreturn(Val_int(0));
}

/* The render node */

/* The GPU's clock in kHz, its active compute units into [cus] (16 ints,
   engine-major as cu_bitmap lays them out), or -errno. */
value caml_rig_amd_amdgpu_device_info(value v_drm, value v_cus) {
  struct drm_amdgpu_info_device info;
  memset(&info, 0, sizeof info);
  struct drm_amdgpu_info q = {.return_pointer = (uint64_t)(uintptr_t)&info,
                              .return_size = sizeof info,
                              .query = AMDGPU_INFO_DEV_INFO};
  int e = request(Int_val(v_drm), DRM_IOCTL_AMDGPU_INFO, &q);
  if (e) return Val_long(e);
  for (int i = 0; i < 16; i++)
    Store_field(v_cus, i, Val_long(info.cu_bitmap[i / 4][i % 4]));
  return Val_long(info.gpu_counter_freq);
}

/* Holds the GPU in its stable power state with a new context on the
   render node, until the process closes it. */
value caml_rig_amd_amdgpu_stable_power(value v_drm) {
  int fd = Int_val(v_drm);
  union drm_amdgpu_ctx c;
  memset(&c, 0, sizeof c);
  c.in.op = AMDGPU_CTX_OP_ALLOC_CTX;
  int e = request(fd, DRM_IOCTL_AMDGPU_CTX, &c);
  if (e) return Val_int(e);
  uint32_t id = c.out.alloc.ctx_id;
  memset(&c, 0, sizeof c);
  c.in.op = AMDGPU_CTX_OP_SET_STABLE_PSTATE;
  c.in.flags = AMDGPU_CTX_STABLE_PSTATE_STANDARD;
  c.in.ctx_id = id;
  e = request(fd, DRM_IOCTL_AMDGPU_CTX, &c);
  if (e) {
    memset(&c, 0, sizeof c);
    c.in.op = AMDGPU_CTX_OP_FREE_CTX;
    c.in.ctx_id = id;
    request(fd, DRM_IOCTL_AMDGPU_CTX, &c);
  }
  return Val_int(e);
}

#else

#define NONE(name, ...)                                                        \
  value caml_rig_amd_amdgpu_##name(__VA_ARGS__) { return Val_int(-ENOSYS); }

value caml_rig_amd_amdgpu_linux(value unit) {
  (void)unit;
  return Val_false;
}

#define UNUSED __attribute__((unused))
NONE(open, value a UNUSED)
NONE(close, value a UNUSED)
NONE(reserve, value a UNUSED)
NONE(map, value a UNUSED, value b UNUSED, value c UNUSED, value d UNUSED)
NONE(unmap, value a UNUSED, value b UNUSED)
NONE(store64, value a UNUSED, value b UNUSED)
NONE(version, value a UNUSED)
NONE(acquire_vm, value a UNUSED, value b UNUSED, value c UNUSED)
NONE(runtime_enable, value a UNUSED)
NONE(alloc, value a UNUSED, value b UNUSED, value c UNUSED, value d UNUSED,
     value e UNUSED, value f UNUSED)
NONE(alloc_byte, value *a UNUSED, int b UNUSED)
NONE(free, value a UNUSED, value b UNUSED)
NONE(map_gpu, value a UNUSED, value b UNUSED, value c UNUSED, value d UNUSED)
NONE(event, value a UNUSED, value b UNUSED, value c UNUSED)
NONE(destroy_event, value a UNUSED, value b UNUSED)
NONE(queue, value a UNUSED, value b UNUSED, value c UNUSED, value d UNUSED)
NONE(destroy_queue, value a UNUSED, value b UNUSED)
NONE(wait, value a UNUSED, value b UNUSED, value c UNUSED, value d UNUSED,
     value e UNUSED)
NONE(device_info, value a UNUSED, value b UNUSED)
NONE(stable_power, value a UNUSED)

#endif
