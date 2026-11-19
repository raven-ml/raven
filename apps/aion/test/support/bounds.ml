(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* The float32 just above [x]: the next bit pattern away from zero above it,
   toward zero below it, and the least subnormal above either zero. *)
let up x =
  if x = 0. then Int32.float_of_bits 1l
  else
    let b = Int32.bits_of_float x in
    Int32.float_of_bits (if x > 0. then Int32.succ b else Int32.pred b)

let rec move k x =
  if k = 0 then x
  else if k > 0 then move (k - 1) (up x)
  else move (k + 1) (-.up (-.x))

let around k y = (move (-k) (f32 y), move k (f32 y))
let through k f (lo, hi) = (move (-k) (f32 (f lo)), move k (f32 (f hi)))
let inside (lo, hi) x = lo <= x && x <= hi
