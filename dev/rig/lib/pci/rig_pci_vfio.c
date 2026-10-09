/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* VFIO's requests, whose parameters Vfio_request packs, over descriptors
   that are Unix.file_descr, an int on Linux, and the eventfd its interrupts
   signal. A refused request raises Unix.Unix_error with its errno, so the
   caller names what to change. A request may wait, as opening a function or
   resetting it does, so each runs with the runtime released. Elsewhere
   every request raises Unix_error ENOSYS. */

#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>

#define CAML_NAME_SPACE
#include <caml/bigarray.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <caml/unixsupport.h>

value caml_rig_pci_vfio_address(value v_params) {
  return Val_long((intnat)Caml_ba_data_val(v_params));
}

#ifdef __linux__
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <unistd.h>

/* The ioctl [v_request] on [v_fd] with the argument [v_arg], an integer or
   the address of memory the caller keeps alive: its result. */
value caml_rig_pci_vfio_ioctl(value v_fd, value v_request, value v_arg) {
  int fd = Int_val(v_fd);
  unsigned long request = (unsigned long)Long_val(v_request);
  unsigned long arg = (unsigned long)Long_val(v_arg);
  caml_release_runtime_system();
  int r = ioctl(fd, request, arg);
  int e = errno;
  caml_acquire_runtime_system();
  if (r < 0) caml_unix_error(e, "ioctl", Nothing);
  return Val_int(r);
}

/* An eventfd an interrupt signals. */
value caml_rig_pci_eventfd(value unit) {
  (void)unit;
  int fd = eventfd(0, EFD_CLOEXEC);
  if (fd < 0) caml_uerror("eventfd", Nothing);
  return Val_int(fd);
}

/* Waits at most [ms] for the eventfd [efd], with the runtime released. A
   poll a signal interrupts reports no interrupt: callers poll in a loop. */
value caml_rig_pci_wait(value efd, value ms) {
  struct pollfd p = {.fd = Int_val(efd), .events = POLLIN};
  int timeout = Int_val(ms);
  caml_release_runtime_system();
  int r = poll(&p, 1, timeout);
  uint64_t count;
  if (r > 0 && read(p.fd, &count, sizeof count) < 0) r = 0;
  caml_acquire_runtime_system();
  return Val_bool(r > 0);
}

#else

/* Without Linux every request raises Unix_error ENOSYS, and no interrupt
   comes. */

value caml_rig_pci_vfio_ioctl(value v_fd, value v_request, value v_arg) {
  (void)v_fd;
  (void)v_request;
  (void)v_arg;
  caml_unix_error(ENOSYS, "ioctl", Nothing);
}

value caml_rig_pci_eventfd(value unit) {
  (void)unit;
  caml_unix_error(ENOSYS, "eventfd", Nothing);
}

value caml_rig_pci_wait(value efd, value ms) {
  (void)efd;
  (void)ms;
  return Val_false;
}
#endif
