(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { unscaled : int64; scale : int }

let max_unscaled = 999_999_999_999_999_999L

let v ~unscaled ~scale =
  if scale < 0 || scale > 18 then
    invalid_arg (Printf.sprintf "Decimal.v: scale %d is not in [0;18]" scale);
  if
    Int64.compare unscaled max_unscaled > 0
    || Int64.compare unscaled (Int64.neg max_unscaled) < 0
  then
    invalid_arg
      (Printf.sprintf "Decimal.v: %Ld has more than 18 digits" unscaled);
  { unscaled; scale }

let unscaled d = d.unscaled
let scale d = d.scale

let pow10 =
  let p = Array.make 19 1L in
  for k = 1 to 18 do
    p.(k) <- Int64.mul 10L p.(k - 1)
  done;
  p

(* [d0] and [d1] are compared at the scale of the finer one without multiplying,
   which could overflow: with [s0 <= s1], [u0 * 10^k] against [u1] is [u0]
   against the floor [q] of [u1 / 10^k], then [0] against the remainder. *)
let rec compare d0 d1 =
  if d0.scale > d1.scale then -compare d1 d0
  else
    let p = pow10.(d1.scale - d0.scale) in
    let q = Int64.div d1.unscaled p and r = Int64.rem d1.unscaled p in
    let q, r =
      if Int64.compare r 0L < 0 then (Int64.pred q, Int64.add r p) else (q, r)
    in
    match Int64.compare d0.unscaled q with
    | 0 -> if Int64.equal r 0L then 0 else -1
    | c -> c

let equal d0 d1 = compare d0 d1 = 0

let pp ppf d =
  if d.scale = 0 then Format.fprintf ppf "%Ld" d.unscaled
  else
    let sign = if Int64.compare d.unscaled 0L < 0 then "-" else "" in
    let u = Int64.abs d.unscaled and p = pow10.(d.scale) in
    Format.fprintf ppf "%s%Ld.%0*Ld" sign (Int64.div u p) d.scale
      (Int64.rem u p)
