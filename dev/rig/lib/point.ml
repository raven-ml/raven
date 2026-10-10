(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A point is a 63-bit pattern, [rig_stubs.h]'s: from index 32768 on it is a
   negative int, so it is compared with 0 alone, never ordered as an int. *)
let value_bits = 47
let max_value = (1 lsl value_bits) - 1
let max_index = 65_535
let make index v = (index lsl value_bits) lor v
let index p = p lsr value_bits
let value p = p land max_value
