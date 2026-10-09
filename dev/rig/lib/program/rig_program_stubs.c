/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <stdint.h>
#include <caml/mlvalues.h>
#include "rig.h"

/* The host address of buffer [v_b]'s first byte, 0 if the host does not
   address its memory. */
value caml_rig_program_host(value v_b) {
  return Val_long((intnat)(uintptr_t)rig_buffer_host(v_b));
}
