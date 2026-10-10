(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* CR: Treat points as 63-bit patterns throughout. From device index 32768,
   Memory.iter_write's p > 0 skips writes, and caml_rig_done's Long_val
   sign-extends the point into an absent device reported complete. Use
   Unsigned_long_val in done and run_collect, and test nonzero in iter_write.
   Give stamps_get an end marker from the unused index-zero encodings:
   -1 is a valid maximal point. *)
let value_bits = 47
let max_value = (1 lsl value_bits) - 1
let max_index = 65_535
let make index v = (index lsl value_bits) lor v
let index p = p lsr value_bits
let value p = p land max_value
