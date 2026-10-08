/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The clock of a machine's poll loop, and the numbers of its devices. */

#define _GNU_SOURCE
#include <sys/types.h>
#include <time.h>
#ifdef __linux__
#include <sys/sysmacros.h>
#endif

#define CAML_NAME_SPACE
#include <caml/mlvalues.h>

/* Monotonic nanoseconds. Holds the runtime. */
intnat caml_rig_pci_now_ns(value unit) {
  (void)unit;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (intnat)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

value caml_rig_pci_now_ns_byte(value unit) {
  return Val_long(caml_rig_pci_now_ns(unit));
}

/* The number [stat] gives the device [major:minor]. Windows numbers no
   device: -1, which no file's number is. Holds the runtime. */
intnat caml_rig_pci_makedev(intnat major, intnat minor) {
#ifdef _WIN32
  (void)major;
  (void)minor;
  return -1;
#else
  return (intnat)makedev((unsigned)major, (unsigned)minor);
#endif
}

value caml_rig_pci_makedev_byte(value major, value minor) {
  return Val_long(caml_rig_pci_makedev(Long_val(major), Long_val(minor)));
}
