/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Read claims around a kernel (kernels.ml).

   A kernel claims its operands' memory for reading while it runs, so that no
   compiled call on another domain donates it meanwhile. A kernel the host
   runs also waits for the work that wrote the memory; a device's kernel is
   ordered by its device. */

#include <stdio.h>

#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "rig.h"

/* Claims [b]'s memory for reading; with [wait], also waits for its last
   write. Raises Invalid_argument for dead or exclusive memory, and Rig.Lost
   where the memory depends on a lost device, holding no claim then. */
static void claim(value b, int wait) {
  CAMLparam1(b);
  switch (rig_buffer_claim(b, RIG_READ)) {
  case RIG_CLAIMED:
    break;
  case RIG_WAIT:
    if (!wait) break;
    /* fall through */
  case RIG_LOST: {
    caml_result r = rig_buffer_wait(b, RIG_READ);
    if (caml_result_is_exception(r)) {
      rig_buffer_release(b);
      caml_get_value_or_raise(r);
    }
    break;
  }
  case RIG_DEAD: {
    char why[256];
    snprintf(why, sizeof why, "%s", rig_buffer_why(b));
    caml_invalid_argument(why);
  }
  case RIG_EXCLUSIVE:
  case RIG_READ_ONLY:
    caml_invalid_argument(
        "Nx: storage is held exclusive by a consuming call");
  }
  CAMLreturn0;
}

value caml_nx_claim(value b) {
  claim(b, 0);
  return Val_unit;
}

value caml_nx_claim_host(value b) {
  claim(b, 1);
  return Val_unit;
}

value caml_nx_release(value b) {
  rig_buffer_release(b);
  return Val_unit;
}
