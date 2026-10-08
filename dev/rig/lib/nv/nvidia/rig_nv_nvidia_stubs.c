/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The system calls the kernel path makes: NVIDIA's device files, their
   ioctls, and the process's mappings at the addresses the GPU uses.

   A stub returns a non-negative result, or errno negated; the OCaml side
   says what it was doing. Linux only: elsewhere every call answers
   -ENOSYS. A stub whose comment says it releases the runtime does so; the
   others hold it. */

#define _GNU_SOURCE

#include <errno.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#define PTR(v) ((void *)Long_val(v))

/* Whether [v_errno] says the process or the system has no file left. */
value caml_rig_nv_nvidia_no_file(value v_errno) {
#if defined(EMFILE) && defined(ENFILE)
  return Val_bool(Int_val(v_errno) == EMFILE || Int_val(v_errno) == ENFILE);
#else
  (void)v_errno;
  return Val_false;
#endif
}

value caml_rig_nv_nvidia_strerror(value v_errno) {
  return caml_copy_string(strerror(Int_val(v_errno)));
}

value caml_rig_nv_nvidia_address(value v_params) {
  return Val_long((intnat)Caml_ba_data_val(v_params));
}

#ifdef __linux__
#include <fcntl.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

/* Opens the device file [v_path]: its descriptor. A GPU's first open
   initialises it, so this releases the runtime. */
value caml_rig_nv_nvidia_open(value v_path) {
  CAMLparam1(v_path);
  char *path = caml_stat_strdup(String_val(v_path));
  caml_release_runtime_system();
  int fd = open(path, O_RDWR | O_CLOEXEC);
  int e = errno;
  caml_acquire_runtime_system();
  caml_stat_free(path);
  CAMLreturn(Val_long(fd < 0 ? -e : fd));
}

value caml_rig_nv_nvidia_close(value v_fd) {
  close(Int_val(v_fd));
  return Val_unit;
}

/* Runs the ioctl [v_request] on [v_fd] with [v_params]' bytes, which may
   point to other bytes the caller keeps alive, retrying an interrupted
   call. Releases the runtime: the RM may take time to answer, and
   [v_params] stays rooted, so another domain's collection cannot free the
   bytes the kernel writes. */
value caml_rig_nv_nvidia_ioctl(value v_fd, value v_request, value v_params) {
  CAMLparam1(v_params);
  int fd = Int_val(v_fd);
  unsigned long request = (unsigned long)Long_val(v_request);
  void *arg = Caml_ba_data_val(v_params);
  int r;
  caml_release_runtime_system();
  do r = ioctl(fd, request, arg);
  while (r < 0 && errno == EINTR);
  int e = errno;
  caml_acquire_runtime_system();
  CAMLreturn(Val_long(r < 0 ? -e : 0));
}

/* Maps [v_n] bytes at [v_at], readable, writable and shared: of [v_fd] from
   its start, or anonymous memory if [v_fd] is negative. Releases the
   runtime. */
value caml_rig_nv_nvidia_map(value v_fd, value v_at, value v_n) {
  int fd = Int_val(v_fd);
  void *at = PTR(v_at);
  size_t n = (size_t)Long_val(v_n);
  int flags = MAP_SHARED | MAP_FIXED | (fd < 0 ? MAP_ANONYMOUS : 0);
  caml_release_runtime_system();
  void *p = mmap(at, n, PROT_READ | PROT_WRITE, flags, fd, 0);
  int e = errno;
  caml_acquire_runtime_system();
  return Val_long(p == MAP_FAILED ? -e : 0);
}

/* Reserves the [v_n] bytes at [v_at], inaccessible, unless something is
   mapped there. */
value caml_rig_nv_nvidia_reserve(value v_at, value v_n) {
  void *at = PTR(v_at);
  size_t n = (size_t)Long_val(v_n);
  void *p = mmap(at, n, PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE |
                     MAP_FIXED_NOREPLACE,
                 -1, 0);
  if (p == MAP_FAILED) return Val_long(-errno);
  /* A kernel before 4.17 reads the flag as a hint. */
  if (p != at) {
    munmap(p, n);
    return Val_long(-EEXIST);
  }
  return Val_long(0);
}

/* Gives back the reservation of the [v_n] bytes at [v_at]. */
value caml_rig_nv_nvidia_unreserve(value v_at, value v_n) {
  return Val_long(munmap(PTR(v_at), (size_t)Long_val(v_n)) == 0 ? 0 : -errno);
}

/* Returns the [v_n] bytes at [v_at] to the reservation they came from. */
value caml_rig_nv_nvidia_unmap(value v_at, value v_n) {
  void *p = mmap(PTR(v_at), (size_t)Long_val(v_n), PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1,
                 0);
  return Val_long(p == MAP_FAILED ? -errno : 0);
}

#else

value caml_rig_nv_nvidia_open(value v_path) {
  CAMLparam1(v_path);
  CAMLreturn(Val_long(-ENOSYS));
}

value caml_rig_nv_nvidia_close(value v_fd) {
  (void)v_fd;
  return Val_unit;
}

value caml_rig_nv_nvidia_ioctl(value v_fd, value v_request, value v_params) {
  (void)v_fd;
  (void)v_request;
  (void)v_params;
  return Val_long(-ENOSYS);
}

value caml_rig_nv_nvidia_map(value v_fd, value v_at, value v_n) {
  (void)v_fd;
  (void)v_at;
  (void)v_n;
  return Val_long(-ENOSYS);
}

value caml_rig_nv_nvidia_reserve(value v_at, value v_n) {
  (void)v_at;
  (void)v_n;
  return Val_long(-ENOSYS);
}

value caml_rig_nv_nvidia_unmap(value v_at, value v_n) {
  (void)v_at;
  (void)v_n;
  return Val_long(-ENOSYS);
}

value caml_rig_nv_nvidia_unreserve(value v_at, value v_n) {
  (void)v_at;
  (void)v_n;
  return Val_long(-ENOSYS);
}

#endif
