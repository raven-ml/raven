/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The system calls of the uverbs path: the verbs file, its ioctl, its
   mappings, and its event files.

   A stub returns a non-negative result, or errno negated; the OCaml side
   says what it was doing. Linux only: elsewhere every call answers -ENOSYS.
   The ioctl and the wait release the runtime; the others are short. */

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

value caml_rig_mlx5_uverbs_strerror(value v_errno) {
  return caml_copy_string(strerror(Int_val(v_errno)));
}

value caml_rig_mlx5_uverbs_address(value v_params) {
  return Val_long((intnat)Caml_ba_data_val(v_params));
}

#ifdef __linux__
#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

value caml_rig_mlx5_uverbs_open(value v_path) {
  int fd = open(String_val(v_path), O_RDWR | O_CLOEXEC);
  return Val_long(fd < 0 ? -errno : fd);
}

value caml_rig_mlx5_uverbs_close(value v_fd) {
  close(Int_val(v_fd));
  return Val_unit;
}

/* Makes reads of [v_fd] answer -EAGAIN when it holds nothing. */
value caml_rig_mlx5_uverbs_nonblock(value v_fd) {
  int fd = Int_val(v_fd);
  int flags = fcntl(fd, F_GETFL);
  if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0)
    return Val_long(-errno);
  return Val_long(0);
}

/* Runs the verbs ioctl [v_request] on [v_fd] with [v_params], whose
   attributes may name other bytes the caller keeps alive, retrying an
   interrupted call. Releases the runtime: the kernel may take long, such as
   when it pins memory, and [v_params] stays rooted. */
value caml_rig_mlx5_uverbs_ioctl(value v_fd, value v_request, value v_params) {
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

/* Maps the [v_n] bytes at offset [v_off] of [v_fd], shared, readable and
   writable: their address. */
value caml_rig_mlx5_uverbs_map(value v_fd, value v_off, value v_n) {
  void *p = mmap(NULL, (size_t)Long_val(v_n), PROT_READ | PROT_WRITE,
                 MAP_SHARED, Int_val(v_fd), (off_t)Long_val(v_off));
  return Val_long(p == MAP_FAILED ? -errno : (intnat)p);
}

value caml_rig_mlx5_uverbs_unmap(value v_at, value v_n) {
  munmap(PTR(v_at), (size_t)Long_val(v_n));
  return Val_unit;
}

value caml_rig_mlx5_uverbs_page_size(value unit) {
  (void)unit;
  return Val_long(sysconf(_SC_PAGESIZE));
}

/* Waits until [v_a] or [v_b] is readable, at most [v_ms] milliseconds: bit
   0 set if [v_a] is, bit 1 if [v_b] is. Releases the runtime. */
value caml_rig_mlx5_uverbs_poll(value v_a, value v_b, value v_ms) {
  struct pollfd fds[2] = {{.fd = Int_val(v_a), .events = POLLIN},
                          {.fd = Int_val(v_b), .events = POLLIN}};
  int ms = Int_val(v_ms);
  caml_release_runtime_system();
  int r = poll(fds, 2, ms);
  int e = errno;
  caml_acquire_runtime_system();
  if (r < 0) return Val_long(e == EINTR ? 0 : -e);
  return Val_long((fds[0].revents ? 1 : 0) | (fds[1].revents ? 2 : 0));
}

/* Reads what [v_fd] holds into [v_params]: the bytes read, or -errno. */
value caml_rig_mlx5_uverbs_read(value v_fd, value v_params) {
  ssize_t r = read(Int_val(v_fd), Caml_ba_data_val(v_params),
                   Caml_ba_array_val(v_params)->dim[0]);
  return Val_long(r < 0 ? -errno : r);
}

#else

value caml_rig_mlx5_uverbs_open(value v_path) {
  (void)v_path;
  return Val_long(-ENOSYS);
}

value caml_rig_mlx5_uverbs_close(value v_fd) {
  (void)v_fd;
  return Val_unit;
}

value caml_rig_mlx5_uverbs_nonblock(value v_fd) {
  (void)v_fd;
  return Val_long(-ENOSYS);
}

value caml_rig_mlx5_uverbs_ioctl(value v_fd, value v_request, value v_params) {
  (void)v_fd;
  (void)v_request;
  (void)v_params;
  return Val_long(-ENOSYS);
}

value caml_rig_mlx5_uverbs_map(value v_fd, value v_off, value v_n) {
  (void)v_fd;
  (void)v_off;
  (void)v_n;
  return Val_long(-ENOSYS);
}

value caml_rig_mlx5_uverbs_unmap(value v_at, value v_n) {
  (void)v_at;
  (void)v_n;
  return Val_unit;
}

value caml_rig_mlx5_uverbs_page_size(value unit) {
  (void)unit;
  return Val_long(-ENOSYS);
}

value caml_rig_mlx5_uverbs_poll(value v_a, value v_b, value v_ms) {
  (void)v_a;
  (void)v_b;
  (void)v_ms;
  return Val_long(-ENOSYS);
}

value caml_rig_mlx5_uverbs_read(value v_fd, value v_params) {
  (void)v_fd;
  (void)v_params;
  return Val_long(-ENOSYS);
}

#endif
