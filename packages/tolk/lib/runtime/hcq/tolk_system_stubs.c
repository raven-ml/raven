/*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*/

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __linux__
#include <linux/vfio.h>
#include <poll.h>
#include <stdint.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#endif

/* Linux-only mmap flags. Zero values disable a flag, so portable callers can
   OR them in unconditionally; the header values are preferred and the
   fallbacks match the generic Linux ABI. */
#ifdef __linux__
#ifndef MAP_LOCKED
#define MAP_LOCKED 0x2000
#endif
#ifndef MAP_POPULATE
#define MAP_POPULATE 0x8000
#endif
#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000
#endif
#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif
#else
#ifndef MAP_LOCKED
#define MAP_LOCKED 0
#endif
#ifndef MAP_POPULATE
#define MAP_POPULATE 0
#endif
#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0
#endif
#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0
#endif
#endif

static void raise_errno(const char *what) {
  char buf[256];
  snprintf(buf, sizeof(buf), "%s: %s", what, strerror(errno));
  caml_failwith(buf);
}

CAMLprim value caml_tolk_system_constants(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(v);
  v = caml_alloc_tuple(9);
  Store_field(v, 0, Val_int(O_WRONLY));
  Store_field(v, 1, Val_int(O_CREAT));
  Store_field(v, 2, Val_int(O_SYNC));
  Store_field(v, 3, Val_int(MAP_LOCKED));
  Store_field(v, 4, Val_int(MAP_POPULATE));
  Store_field(v, 5, Val_int(MAP_HUGETLB));
  Store_field(v, 6, Val_int(MAP_FIXED_NOREPLACE));
  Store_field(v, 7, Val_long(sysconf(_SC_PAGESIZE)));
#ifdef __linux__
  Store_field(v, 8, Val_true);
#else
  Store_field(v, 8, Val_false);
#endif
  CAMLreturn(v);
}

/* open(2) with an explicit creation mode; close-on-exec is always added and
   the mode is applied with fchmod so it does not depend on the process
   umask. */
CAMLprim value caml_tolk_system_open_mode(value v_path, value v_flags,
                                          value v_mode) {
  CAMLparam3(v_path, v_flags, v_mode);
  int fd = open(String_val(v_path), Int_val(v_flags) | O_CLOEXEC,
                (mode_t)Int_val(v_mode));
  if (fd < 0) raise_errno(String_val(v_path));
  if ((Int_val(v_flags) & O_CREAT) &&
      fchmod(fd, (mode_t)Int_val(v_mode)) != 0) {
    int e = errno;
    close(fd);
    errno = e;
    raise_errno(String_val(v_path));
  }
  CAMLreturn(Val_int(fd));
}

CAMLprim value caml_tolk_system_flock_try(value v_fd) {
  int r;
  do r = flock(Int_val(v_fd), LOCK_EX | LOCK_NB);
  while (r != 0 && errno == EINTR);
  return Val_bool(r == 0);
}

CAMLprim value caml_tolk_system_mlock(value v_addr, value v_size) {
  return Val_bool(
      mlock((void *)Nativeint_val(v_addr), (size_t)Long_val(v_size)) == 0);
}

CAMLprim value caml_tolk_system_madvise_dontfork(value v_addr, value v_size) {
#ifdef __linux__
  if (madvise((void *)Nativeint_val(v_addr), (size_t)Long_val(v_size),
              MADV_DONTFORK) != 0)
    raise_errno("madvise");
  return Val_unit;
#else
  (void)v_addr;
  (void)v_size;
  caml_failwith("madvise(MADV_DONTFORK) requires Linux");
  return Val_unit; /* unreachable */
#endif
}

/* The copies must keep the OCaml runtime lock held: the bytes value may move
   under the GC otherwise. */

CAMLprim value caml_tolk_system_pread(value v_fd, value v_buf, value v_pos,
                                      value v_len, value v_off) {
  CAMLparam5(v_fd, v_buf, v_pos, v_len, v_off);
  ssize_t r;
  do
    r = pread(Int_val(v_fd), Bytes_val(v_buf) + Long_val(v_pos),
              (size_t)Long_val(v_len), (off_t)Int64_val(v_off));
  while (r < 0 && errno == EINTR);
  if (r < 0) raise_errno("pread");
  CAMLreturn(Val_long(r));
}

CAMLprim value caml_tolk_system_pwrite(value v_fd, value v_buf, value v_pos,
                                       value v_len, value v_off) {
  CAMLparam5(v_fd, v_buf, v_pos, v_len, v_off);
  ssize_t r;
  do
    r = pwrite(Int_val(v_fd), Bytes_val(v_buf) + Long_val(v_pos),
               (size_t)Long_val(v_len), (off_t)Int64_val(v_off));
  while (r < 0 && errno == EINTR);
  if (r < 0) raise_errno("pwrite");
  CAMLreturn(Val_long(r));
}

CAMLprim value caml_tolk_system_write(value v_fd, value v_buf, value v_len) {
  CAMLparam3(v_fd, v_buf, v_len);
  ssize_t r;
  do r = write(Int_val(v_fd), Bytes_val(v_buf), (size_t)Long_val(v_len));
  while (r < 0 && errno == EINTR);
  if (r < 0) raise_errno("write");
  CAMLreturn(Val_long(r));
}

CAMLprim value caml_tolk_system_readlink(value v_path) {
  CAMLparam1(v_path);
  char buf[4096];
  ssize_t r = readlink(String_val(v_path), buf, sizeof(buf) - 1);
  if (r < 0) raise_errno(String_val(v_path));
  buf[r] = '\0';
  CAMLreturn(caml_copy_string(buf));
}

/* VFIO interrupt plumbing (Linux only). Thin marshallers: the ioctl
   numbers and struct layout come from <linux/vfio.h>. */

CAMLprim value caml_tolk_system_eventfd(value v_initval) {
#ifdef __linux__
  int fd = eventfd((unsigned int)Int_val(v_initval), EFD_CLOEXEC);
  if (fd < 0) raise_errno("eventfd");
  return Val_int(fd);
#else
  (void)v_initval;
  caml_failwith("eventfd requires Linux");
  return Val_unit; /* unreachable */
#endif
}

/* Blocks up to the timeout, so the OCaml runtime is released. */
CAMLprim value caml_tolk_system_poll_in(value v_fd, value v_timeout_ms) {
#ifdef __linux__
  struct pollfd p = {.fd = Int_val(v_fd), .events = POLLIN, .revents = 0};
  int timeout = Int_val(v_timeout_ms);
  int r;
  caml_release_runtime_system();
  do r = poll(&p, 1, timeout);
  while (r < 0 && errno == EINTR);
  caml_acquire_runtime_system();
  if (r < 0) raise_errno("poll");
  return Val_bool(r > 0);
#else
  (void)v_fd;
  (void)v_timeout_ms;
  caml_failwith("poll requires Linux");
  return Val_unit; /* unreachable */
#endif
}

CAMLprim value caml_tolk_system_eventfd_drain(value v_fd) {
#ifdef __linux__
  uint64_t counter;
  ssize_t r;
  do r = read(Int_val(v_fd), &counter, sizeof(counter));
  while (r < 0 && errno == EINTR);
  if (r < 0) raise_errno("eventfd read");
  return Val_unit;
#else
  (void)v_fd;
  caml_failwith("eventfd requires Linux");
  return Val_unit; /* unreachable */
#endif
}

CAMLprim value caml_tolk_system_vfio_check_extension(value v_fd) {
#ifdef __linux__
  if (ioctl(Int_val(v_fd), VFIO_CHECK_EXTENSION,
            (unsigned long)VFIO_NOIOMMU_IOMMU) < 0)
    raise_errno("VFIO_CHECK_EXTENSION");
  return Val_unit;
#else
  (void)v_fd;
  caml_failwith("vfio requires Linux");
  return Val_unit; /* unreachable */
#endif
}

CAMLprim value caml_tolk_system_vfio_group_set_container(value v_group,
                                                         value v_container) {
#ifdef __linux__
  int container_fd = Int_val(v_container);
  if (ioctl(Int_val(v_group), VFIO_GROUP_SET_CONTAINER, &container_fd) < 0)
    raise_errno("VFIO_GROUP_SET_CONTAINER");
  return Val_unit;
#else
  (void)v_group;
  (void)v_container;
  caml_failwith("vfio requires Linux");
  return Val_unit; /* unreachable */
#endif
}

CAMLprim value caml_tolk_system_vfio_set_iommu(value v_fd) {
#ifdef __linux__
  return Val_bool(ioctl(Int_val(v_fd), VFIO_SET_IOMMU,
                        (unsigned long)VFIO_NOIOMMU_IOMMU) == 0);
#else
  (void)v_fd;
  caml_failwith("vfio requires Linux");
  return Val_unit; /* unreachable */
#endif
}

CAMLprim value caml_tolk_system_vfio_group_get_device_fd(value v_group,
                                                         value v_pcibus) {
#ifdef __linux__
  CAMLparam2(v_group, v_pcibus);
  int fd = ioctl(Int_val(v_group), VFIO_GROUP_GET_DEVICE_FD,
                 String_val(v_pcibus));
  if (fd < 0) raise_errno("VFIO_GROUP_GET_DEVICE_FD");
  CAMLreturn(Val_int(fd));
#else
  (void)v_group;
  (void)v_pcibus;
  caml_failwith("vfio requires Linux");
  return Val_unit; /* unreachable */
#endif
}

/* Routes MSI vector 0 to the eventfd: one vfio_irq_set frame with the
   descriptor as its payload. */
CAMLprim value caml_tolk_system_vfio_set_irq_eventfd(value v_dev,
                                                     value v_eventfd) {
#ifdef __linux__
  struct {
    struct vfio_irq_set set;
    int fd;
  } irqs;
  memset(&irqs, 0, sizeof(irqs));
  irqs.set.argsz = sizeof(irqs);
  irqs.set.flags = VFIO_IRQ_SET_DATA_EVENTFD | VFIO_IRQ_SET_ACTION_TRIGGER;
  irqs.set.index = VFIO_PCI_MSI_IRQ_INDEX;
  irqs.set.start = 0;
  irqs.set.count = 1;
  irqs.fd = Int_val(v_eventfd);
  if (ioctl(Int_val(v_dev), VFIO_DEVICE_SET_IRQS, &irqs) < 0)
    raise_errno("VFIO_DEVICE_SET_IRQS");
  return Val_unit;
#else
  (void)v_dev;
  (void)v_eventfd;
  caml_failwith("vfio requires Linux");
  return Val_unit; /* unreachable */
#endif
}
