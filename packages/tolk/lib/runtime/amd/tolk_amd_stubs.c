/*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*/

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* KFD ioctls */

#ifdef __linux__

#include <caml/threads.h>
#include <sys/ioctl.h>

#include "kfd_ioctl.h"

static void raise_errno(const char *what) {
  char buf[256];
  snprintf(buf, sizeof(buf), "%s: %s", what, strerror(errno));
  caml_failwith(buf);
}

/* The OCaml side hard-codes the driver ABI constants; pin them, and every
   argument struct layout, against the vendored header. */
_Static_assert(sizeof(struct kfd_ioctl_get_version_args) == 8, "kfd ABI");
_Static_assert(sizeof(struct kfd_ioctl_acquire_vm_args) == 8, "kfd ABI");
_Static_assert(sizeof(struct kfd_ioctl_runtime_enable_args) == 16, "kfd ABI");
_Static_assert(sizeof(struct kfd_ioctl_alloc_memory_of_gpu_args) == 40,
               "kfd ABI");
_Static_assert(sizeof(struct kfd_ioctl_free_memory_of_gpu_args) == 8,
               "kfd ABI");
_Static_assert(sizeof(struct kfd_ioctl_map_memory_to_gpu_args) == 24,
               "kfd ABI");
_Static_assert(sizeof(struct kfd_ioctl_unmap_memory_from_gpu_args) == 24,
               "kfd ABI");
_Static_assert(sizeof(struct kfd_ioctl_create_event_args) == 32, "kfd ABI");
_Static_assert(sizeof(struct kfd_ioctl_wait_events_args) == 24, "kfd ABI");
_Static_assert(sizeof(struct kfd_ioctl_create_queue_args) == 96, "kfd ABI");
_Static_assert(sizeof(struct kfd_memory_exception_failure) == 16, "kfd ABI");
_Static_assert(sizeof(struct kfd_hsa_memory_exception_data) == 32, "kfd ABI");
_Static_assert(sizeof(struct kfd_hsa_hw_exception_data) == 16, "kfd ABI");
_Static_assert(sizeof(struct kfd_event_data) == 48, "kfd ABI");
_Static_assert(KFD_IOC_ALLOC_MEM_FLAGS_VRAM == (1 << 0), "kfd ABI");
_Static_assert(KFD_IOC_ALLOC_MEM_FLAGS_GTT == (1 << 1), "kfd ABI");
_Static_assert(KFD_IOC_ALLOC_MEM_FLAGS_USERPTR == (1 << 2), "kfd ABI");
_Static_assert(KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED == (1 << 25), "kfd ABI");
_Static_assert(KFD_IOC_ALLOC_MEM_FLAGS_COHERENT == (1 << 26), "kfd ABI");
_Static_assert(KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE == (1 << 28), "kfd ABI");
_Static_assert(KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC == (1 << 29), "kfd ABI");
_Static_assert(KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE == (1 << 30), "kfd ABI");
_Static_assert((uint32_t)KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE == (1u << 31),
               "kfd ABI");
_Static_assert(KFD_IOC_QUEUE_TYPE_COMPUTE == 0x0, "kfd ABI");
_Static_assert(KFD_IOC_QUEUE_TYPE_SDMA == 0x1, "kfd ABI");
_Static_assert(KFD_IOC_EVENT_SIGNAL == 0, "kfd ABI");
_Static_assert(KFD_IOC_EVENT_HW_EXCEPTION == 3, "kfd ABI");
_Static_assert(KFD_IOC_EVENT_MEMORY == 8, "kfd ABI");
_Static_assert(KFD_MAX_QUEUE_PERCENTAGE == 100, "kfd ABI");

static int kfd_ioctl(int fd, unsigned long req, void *arg) {
  int r;
  do r = ioctl(fd, req, arg);
  while (r < 0 && errno == EINTR);
  return r;
}

CAMLprim value caml_tolk_kfd_get_version(value v_fd) {
  CAMLparam1(v_fd);
  CAMLlocal1(res);
  struct kfd_ioctl_get_version_args a = {0};
  if (kfd_ioctl(Int_val(v_fd), AMDKFD_IOC_GET_VERSION, &a) < 0)
    raise_errno("AMDKFD_IOC_GET_VERSION");
  res = caml_alloc_tuple(2);
  Store_field(res, 0, Val_long(a.major_version));
  Store_field(res, 1, Val_long(a.minor_version));
  CAMLreturn(res);
}

CAMLprim value caml_tolk_kfd_acquire_vm(value v_fd, value v_drm_fd,
                                        value v_gpu_id) {
  CAMLparam3(v_fd, v_drm_fd, v_gpu_id);
  struct kfd_ioctl_acquire_vm_args a = {0};
  a.drm_fd = (uint32_t)Long_val(v_drm_fd);
  a.gpu_id = (uint32_t)Long_val(v_gpu_id);
  if (kfd_ioctl(Int_val(v_fd), AMDKFD_IOC_ACQUIRE_VM, &a) < 0)
    raise_errno("AMDKFD_IOC_ACQUIRE_VM");
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_kfd_runtime_enable(value v_fd, value v_mode_mask) {
  CAMLparam2(v_fd, v_mode_mask);
  struct kfd_ioctl_runtime_enable_args a = {0};
  a.mode_mask = (uint32_t)Long_val(v_mode_mask);
  if (kfd_ioctl(Int_val(v_fd), AMDKFD_IOC_RUNTIME_ENABLE, &a) < 0)
    raise_errno("AMDKFD_IOC_RUNTIME_ENABLE");
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_kfd_alloc_memory_of_gpu(value v_fd, value v_va,
                                                 value v_size, value v_gpu_id,
                                                 value v_flags,
                                                 value v_mmap_offset) {
  CAMLparam5(v_fd, v_va, v_size, v_gpu_id, v_flags);
  CAMLxparam1(v_mmap_offset);
  CAMLlocal2(res, payload);
  struct kfd_ioctl_alloc_memory_of_gpu_args a = {0};
  a.va_addr = (uint64_t)Nativeint_val(v_va);
  a.size = (uint64_t)Long_val(v_size);
  a.gpu_id = (uint32_t)Long_val(v_gpu_id);
  a.flags = (uint32_t)Long_val(v_flags);
  a.mmap_offset = (uint64_t)Int64_val(v_mmap_offset);
  if (kfd_ioctl(Int_val(v_fd), AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &a) < 0) {
    /* EINVAL and ENOMEM come back as values: the caller knows the request
       and turns them into actionable messages. */
    if (errno != EINVAL && errno != ENOMEM)
      raise_errno("AMDKFD_IOC_ALLOC_MEMORY_OF_GPU");
    res = caml_alloc(1, 1); /* Error */
    Store_field(res, 0, Val_int(errno == EINVAL ? 0 : 1));
    CAMLreturn(res);
  }
  payload = caml_alloc_tuple(2);
  Store_field(payload, 0, caml_copy_int64((int64_t)a.handle));
  Store_field(payload, 1, caml_copy_int64((int64_t)a.mmap_offset));
  res = caml_alloc(1, 0); /* Ok */
  Store_field(res, 0, payload);
  CAMLreturn(res);
}

CAMLprim value caml_tolk_kfd_free_memory_of_gpu(value v_fd, value v_handle) {
  CAMLparam2(v_fd, v_handle);
  struct kfd_ioctl_free_memory_of_gpu_args a = {0};
  a.handle = (uint64_t)Int64_val(v_handle);
  if (kfd_ioctl(Int_val(v_fd), AMDKFD_IOC_FREE_MEMORY_OF_GPU, &a) < 0)
    raise_errno("AMDKFD_IOC_FREE_MEMORY_OF_GPU");
  CAMLreturn(Val_unit);
}

static value kfd_map_or_unmap(value v_fd, value v_handle, value v_gpu_ids,
                              unsigned long req, const char *what) {
  CAMLparam3(v_fd, v_handle, v_gpu_ids);
  uint32_t ids[16];
  size_t n = Wosize_val(v_gpu_ids);
  if (n == 0 || n > 16) caml_invalid_argument(what);
  for (size_t i = 0; i < n; i++)
    ids[i] = (uint32_t)Long_val(Field(v_gpu_ids, i));
  struct kfd_ioctl_map_memory_to_gpu_args a = {0};
  a.handle = (uint64_t)Int64_val(v_handle);
  a.device_ids_array_ptr = (uint64_t)(uintptr_t)ids;
  a.n_devices = (uint32_t)n;
  if (kfd_ioctl(Int_val(v_fd), req, &a) < 0) raise_errno(what);
  if (a.n_success != n) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s: %u of %zu devices succeeded", what,
             a.n_success, n);
    caml_failwith(buf);
  }
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_kfd_map_memory_to_gpu(value v_fd, value v_handle,
                                               value v_gpu_ids) {
  return kfd_map_or_unmap(v_fd, v_handle, v_gpu_ids,
                          AMDKFD_IOC_MAP_MEMORY_TO_GPU,
                          "AMDKFD_IOC_MAP_MEMORY_TO_GPU");
}

CAMLprim value caml_tolk_kfd_unmap_memory_from_gpu(value v_fd, value v_handle,
                                                   value v_gpu_ids) {
  return kfd_map_or_unmap(v_fd, v_handle, v_gpu_ids,
                          AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU,
                          "AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU");
}

CAMLprim value caml_tolk_kfd_create_event(value v_fd, value v_page_offset,
                                          value v_event_type,
                                          value v_auto_reset) {
  CAMLparam4(v_fd, v_page_offset, v_event_type, v_auto_reset);
  CAMLlocal1(res);
  struct kfd_ioctl_create_event_args a = {0};
  a.event_page_offset = (uint64_t)Int64_val(v_page_offset);
  a.event_type = (uint32_t)Long_val(v_event_type);
  a.auto_reset = (uint32_t)Long_val(v_auto_reset);
  if (kfd_ioctl(Int_val(v_fd), AMDKFD_IOC_CREATE_EVENT, &a) < 0)
    raise_errno("AMDKFD_IOC_CREATE_EVENT");
  res = caml_alloc_tuple(2);
  Store_field(res, 0, Val_long(a.event_id));
  Store_field(res, 1, Val_long(a.event_slot_index));
  CAMLreturn(res);
}

CAMLprim value caml_tolk_kfd_wait_events(value v_fd, value v_queue_id,
                                         value v_mem_id, value v_hw_id,
                                         value v_timeout_ms) {
  CAMLparam5(v_fd, v_queue_id, v_mem_id, v_hw_id, v_timeout_ms);
  CAMLlocal4(res, memf, hwf, tmp);
  /* The event-data entries are a union; which member the driver fills is
     fixed by each event's type, so the layout is read positionally. */
  struct kfd_event_data evs[3];
  memset(evs, 0, sizeof(evs));
  evs[0].event_id = (uint32_t)Long_val(v_queue_id);
  evs[1].event_id = (uint32_t)Long_val(v_mem_id);
  evs[2].event_id = (uint32_t)Long_val(v_hw_id);
  struct kfd_ioctl_wait_events_args a = {0};
  a.events_ptr = (uint64_t)(uintptr_t)evs;
  a.num_events = 3;
  a.wait_for_all = 0;
  a.timeout = (uint32_t)Long_val(v_timeout_ms);
  int fd = Int_val(v_fd);
  int r;
  caml_release_runtime_system();
  do r = ioctl(fd, AMDKFD_IOC_WAIT_EVENTS, &a);
  while (r < 0 && errno == EINTR);
  caml_acquire_runtime_system();
  if (r < 0) raise_errno("AMDKFD_IOC_WAIT_EVENTS");
  memf = Val_none;
  if (evs[1].memory_exception_data.gpu_id != 0) {
    tmp = caml_alloc_tuple(5);
    Store_field(tmp, 0,
                caml_copy_int64((int64_t)evs[1].memory_exception_data.va));
    Store_field(
        tmp, 1,
        Val_long(evs[1].memory_exception_data.failure.NotPresent));
    Store_field(tmp, 2,
                Val_long(evs[1].memory_exception_data.failure.ReadOnly));
    Store_field(tmp, 3,
                Val_long(evs[1].memory_exception_data.failure.NoExecute));
    Store_field(tmp, 4,
                Val_long(evs[1].memory_exception_data.failure.imprecise));
    memf = caml_alloc_some(tmp);
  }
  hwf = Val_none;
  if (evs[2].hw_exception_data.gpu_id != 0) {
    tmp = caml_alloc_tuple(4);
    Store_field(tmp, 0, Val_long(evs[2].hw_exception_data.reset_type));
    Store_field(tmp, 1, Val_long(evs[2].hw_exception_data.reset_cause));
    Store_field(tmp, 2, Val_long(evs[2].hw_exception_data.memory_lost));
    Store_field(tmp, 3, Val_long(evs[2].hw_exception_data.gpu_id));
    hwf = caml_alloc_some(tmp);
  }
  res = caml_alloc_tuple(2);
  Store_field(res, 0, memf);
  Store_field(res, 1, hwf);
  CAMLreturn(res);
}

CAMLprim value caml_tolk_kfd_create_queue(
    value v_fd, value v_ring_base, value v_ring_size, value v_gpu_id,
    value v_queue_type, value v_queue_percentage, value v_queue_priority,
    value v_eop_addr, value v_eop_size, value v_cwsr_addr, value v_cwsr_size,
    value v_ctl_stack_size, value v_wptr, value v_rptr) {
  CAMLparam5(v_fd, v_ring_base, v_ring_size, v_gpu_id, v_queue_type);
  CAMLxparam5(v_queue_percentage, v_queue_priority, v_eop_addr, v_eop_size,
              v_cwsr_addr);
  CAMLxparam4(v_cwsr_size, v_ctl_stack_size, v_wptr, v_rptr);
  CAMLlocal1(res);
  struct kfd_ioctl_create_queue_args a = {0};
  a.ring_base_address = (uint64_t)Nativeint_val(v_ring_base);
  a.ring_size = (uint32_t)Long_val(v_ring_size);
  a.gpu_id = (uint32_t)Long_val(v_gpu_id);
  a.queue_type = (uint32_t)Long_val(v_queue_type);
  a.queue_percentage = (uint32_t)Long_val(v_queue_percentage);
  a.queue_priority = (uint32_t)Long_val(v_queue_priority);
  a.eop_buffer_address = (uint64_t)Nativeint_val(v_eop_addr);
  a.eop_buffer_size = (uint64_t)Long_val(v_eop_size);
  a.ctx_save_restore_address = (uint64_t)Nativeint_val(v_cwsr_addr);
  a.ctx_save_restore_size = (uint32_t)Long_val(v_cwsr_size);
  a.ctl_stack_size = (uint32_t)Long_val(v_ctl_stack_size);
  a.write_pointer_address = (uint64_t)Nativeint_val(v_wptr);
  a.read_pointer_address = (uint64_t)Nativeint_val(v_rptr);
  if (kfd_ioctl(Int_val(v_fd), AMDKFD_IOC_CREATE_QUEUE, &a) < 0)
    raise_errno("AMDKFD_IOC_CREATE_QUEUE");
  res = caml_alloc_tuple(3);
  Store_field(res, 0, caml_copy_int64((int64_t)a.doorbell_offset));
  Store_field(res, 1, caml_copy_nativeint((intnat)a.read_pointer_address));
  Store_field(res, 2, caml_copy_nativeint((intnat)a.write_pointer_address));
  CAMLreturn(res);
}

#else /* !__linux__ */

static value kfd_unavailable(void) {
  caml_failwith("AMD runtime requires Linux");
  return Val_unit; /* unreachable */
}

CAMLprim value caml_tolk_kfd_get_version(value v_fd) {
  (void)v_fd;
  return kfd_unavailable();
}

CAMLprim value caml_tolk_kfd_acquire_vm(value v_fd, value v_drm_fd,
                                        value v_gpu_id) {
  (void)v_fd;
  (void)v_drm_fd;
  (void)v_gpu_id;
  return kfd_unavailable();
}

CAMLprim value caml_tolk_kfd_runtime_enable(value v_fd, value v_mode_mask) {
  (void)v_fd;
  (void)v_mode_mask;
  return kfd_unavailable();
}

CAMLprim value caml_tolk_kfd_alloc_memory_of_gpu(value v_fd, value v_va,
                                                 value v_size, value v_gpu_id,
                                                 value v_flags,
                                                 value v_mmap_offset) {
  (void)v_fd;
  (void)v_va;
  (void)v_size;
  (void)v_gpu_id;
  (void)v_flags;
  (void)v_mmap_offset;
  return kfd_unavailable();
}

CAMLprim value caml_tolk_kfd_free_memory_of_gpu(value v_fd, value v_handle) {
  (void)v_fd;
  (void)v_handle;
  return kfd_unavailable();
}

CAMLprim value caml_tolk_kfd_map_memory_to_gpu(value v_fd, value v_handle,
                                               value v_gpu_ids) {
  (void)v_fd;
  (void)v_handle;
  (void)v_gpu_ids;
  return kfd_unavailable();
}

CAMLprim value caml_tolk_kfd_unmap_memory_from_gpu(value v_fd, value v_handle,
                                                   value v_gpu_ids) {
  (void)v_fd;
  (void)v_handle;
  (void)v_gpu_ids;
  return kfd_unavailable();
}

CAMLprim value caml_tolk_kfd_create_event(value v_fd, value v_page_offset,
                                          value v_event_type,
                                          value v_auto_reset) {
  (void)v_fd;
  (void)v_page_offset;
  (void)v_event_type;
  (void)v_auto_reset;
  return kfd_unavailable();
}

CAMLprim value caml_tolk_kfd_wait_events(value v_fd, value v_queue_id,
                                         value v_mem_id, value v_hw_id,
                                         value v_timeout_ms) {
  (void)v_fd;
  (void)v_queue_id;
  (void)v_mem_id;
  (void)v_hw_id;
  (void)v_timeout_ms;
  return kfd_unavailable();
}

CAMLprim value caml_tolk_kfd_create_queue(
    value v_fd, value v_ring_base, value v_ring_size, value v_gpu_id,
    value v_queue_type, value v_queue_percentage, value v_queue_priority,
    value v_eop_addr, value v_eop_size, value v_cwsr_addr, value v_cwsr_size,
    value v_ctl_stack_size, value v_wptr, value v_rptr) {
  (void)v_fd;
  (void)v_ring_base;
  (void)v_ring_size;
  (void)v_gpu_id;
  (void)v_queue_type;
  (void)v_queue_percentage;
  (void)v_queue_priority;
  (void)v_eop_addr;
  (void)v_eop_size;
  (void)v_cwsr_addr;
  (void)v_cwsr_size;
  (void)v_ctl_stack_size;
  (void)v_wptr;
  (void)v_rptr;
  return kfd_unavailable();
}

#endif /* __linux__ */

CAMLprim value caml_tolk_kfd_alloc_memory_of_gpu_bc(value *argv, int argn) {
  (void)argn;
  return caml_tolk_kfd_alloc_memory_of_gpu(argv[0], argv[1], argv[2], argv[3],
                                           argv[4], argv[5]);
}

CAMLprim value caml_tolk_kfd_create_queue_bc(value *argv, int argn) {
  (void)argn;
  return caml_tolk_kfd_create_queue(argv[0], argv[1], argv[2], argv[3],
                                    argv[4], argv[5], argv[6], argv[7],
                                    argv[8], argv[9], argv[10], argv[11],
                                    argv[12], argv[13]);
}
