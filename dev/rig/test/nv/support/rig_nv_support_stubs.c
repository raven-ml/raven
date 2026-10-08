/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Patterns of bytes in host memory, written and checked in C. Every stub
   holds the runtime: none blocks. */

#include <stdint.h>

#define CAML_NAME_SPACE
#include <caml/mlvalues.h>

#define Ptr_val(v) ((void *)Long_val(v))

/* Byte [i] of the pattern of [seed]. */
static uint8_t pattern_byte(uint64_t seed, uint64_t i) {
  uint64_t h = (i + (seed << 40)) * 0x9E3779B97F4A7C1ULL;
  return (uint8_t)((h ^ (h >> 29)) >> 17);
}

value rig_nv_test_pattern(value v_p, value v_n, value v_seed) {
  uint8_t *p = Ptr_val(v_p);
  uint64_t n = Long_val(v_n), seed = Long_val(v_seed);
  for (uint64_t i = 0; i < n; i++) p[i] = pattern_byte(seed, i);
  return Val_unit;
}

value rig_nv_test_mismatch(value v_p, value v_n, value v_seed) {
  const uint8_t *p = Ptr_val(v_p);
  uint64_t n = Long_val(v_n), seed = Long_val(v_seed);
  for (uint64_t i = 0; i < n; i++)
    if (p[i] != pattern_byte(seed, i)) return Val_long(i);
  return Val_long(-1);
}
