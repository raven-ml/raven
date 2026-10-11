/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The numbers of a machine's devices. */

#define _GNU_SOURCE
#include <sys/types.h>
#ifdef __linux__
#include <sys/sysmacros.h>
#endif

#define CAML_NAME_SPACE
#include <caml/mlvalues.h>

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
