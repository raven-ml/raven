/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The system calls the amdgpu path makes: Linux's amdgpu driver's files,
   their ioctls, whose parameters Request packs, and the process's mappings
   of their memory. A stub answers the kernel's refusal as -errno, or a
   non-negative result. A stub whose comment says it releases the runtime
   does so; the others hold it.

   Off Linux every stub answers -ENOSYS; the library calls none there. */

#define _GNU_SOURCE

#include <errno.h>
#include <stdint.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#define Ptr_val(v) ((void *)Long_val(v))

value caml_rig_amd_amdgpu_strerror(value v_e) {
  return caml_copy_string(strerror(Int_val(v_e)));
}

value caml_rig_amd_amdgpu_address(value v_params) {
  return Val_long((intnat)Caml_ba_data_val(v_params));
}

#ifdef __linux__
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

value caml_rig_amd_amdgpu_linux(value unit) {
  (void)unit;
  return Val_true;
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
  CAMLreturn(Val_int(fd < 0 ? -e : fd));
}

value caml_rig_amd_amdgpu_close(value v_fd) {
  close(Int_val(v_fd));
  return Val_unit;
}

/* Runs the ioctl [v_request] on [v_fd] with [v_params]' bytes, which may
   point to other bytes the caller keeps alive, making an interrupted or
   refused-for-now request again, as the kernel asks. Releases the runtime:
   a wait blocks, and [v_params] stays rooted, so another domain's
   collection cannot free the bytes the kernel writes. */
value caml_rig_amd_amdgpu_ioctl(value v_fd, value v_request,
                                value v_params) {
  CAMLparam1(v_params);
  int fd = Int_val(v_fd);
  unsigned long request = (unsigned long)Long_val(v_request);
  void *arg = Caml_ba_data_val(v_params);
  int r;
  caml_release_runtime_system();
  do r = ioctl(fd, request, arg);
  while (r < 0 && (errno == EINTR || errno == EAGAIN));
  int e = errno;
  caml_acquire_runtime_system();
  CAMLreturn(Val_int(r < 0 ? -e : 0));
}

/* [n] addresses, inaccessible, anywhere: their address, or -errno. */
value caml_rig_amd_amdgpu_reserve(value v_n) {
  void *p = mmap(NULL, (size_t)Long_val(v_n), PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  return Val_long(p == MAP_FAILED ? -errno : (intnat)p);
}

/* Maps [n] bytes of [fd] from the offset [off] at [at], or anywhere if
   [at] is 0: the address, or -errno. */
value caml_rig_amd_amdgpu_map(value v_fd, value v_at, value v_n,
                              value v_off) {
  void *want = Ptr_val(v_at);
  void *p = mmap(want, (size_t)Long_val(v_n), PROT_READ | PROT_WRITE,
                 MAP_SHARED | (want ? MAP_FIXED : 0), Int_val(v_fd),
                 (off_t)Int64_val(v_off));
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

value caml_rig_amd_amdgpu_pid(value unit) {
  (void)unit;
  return Val_long(getpid());
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
NONE(ioctl, value a UNUSED, value b UNUSED, value c UNUSED)
NONE(reserve, value a UNUSED)
NONE(map, value a UNUSED, value b UNUSED, value c UNUSED, value d UNUSED)
NONE(unmap, value a UNUSED, value b UNUSED)
NONE(store64, value a UNUSED, value b UNUSED)
NONE(pid, value a UNUSED)

#endif
