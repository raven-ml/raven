(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let step = 64

let rec plus sum shift x =
  let t = sum +. Float.ldexp x (-shift) in
  if Float.is_finite t then (t, shift)
  else plus (Float.ldexp sum (-step)) (shift + step) x

let rec add sums shifts k x =
  let t = sums.(k) +. Float.ldexp x (-shifts.(k)) in
  if Float.is_finite t then sums.(k) <- t
  else begin
    sums.(k) <- Float.ldexp sums.(k) (-step);
    shifts.(k) <- shifts.(k) + step;
    add sums shifts k x
  end

(* The mean of finite values is finite, so clamping a rounding past the largest
   float only brings it closer. *)
let get sum shift n =
  let m = Float.ldexp (sum /. Float.of_int n) shift in
  Float.min Float.max_float (Float.max (-.Float.max_float) m)
