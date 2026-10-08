/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* One C template, which the tests load and fill. */

#include <stdint.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/mlvalues.h>

#include <rig_packet.h>

static struct rig_template t;

value test_packet_template(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&t);
}

/* The template's words filled from the arguments, as little-endian bytes
   on a little-endian host. */
value test_packet_fill(value v_a0, value v_a1, value v_a2) {
  uint64_t args[RIG_TEMPLATE_ARGS] = {(uint64_t)Int64_val(v_a0),
                                      (uint64_t)Int64_val(v_a1),
                                      (uint64_t)Int64_val(v_a2)};
  uint32_t w[RIG_TEMPLATE_WORDS];
  int n = rig_fill(&t, args, w);
  value s = caml_alloc_string(4 * (mlsize_t)n);
  memcpy(Bytes_val(s), w, 4 * (size_t)n);
  return s;
}
