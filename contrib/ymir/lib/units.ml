(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Ymir_units

let strf = Printf.sprintf
let parsec = Unit.(int 648000 / pi * astronomical_unit)
let julian_year = Unit.(int 31557600 * second)

(* Magnitudes *)

let exponent_error fn s =
  invalid_arg (strf "%s: the exponent of 10 for %S leaves int" fn s)

(* [parse fn s] is [(negative, m, k)] for the decimal [s] = ±m 10^k, an optional
   [-] then {!Unit.decimal}'s grammar, zero allowed. *)
let parse fn s =
  let negative = String.length s > 0 && s.[0] = '-' in
  let digits = if negative then String.sub s 1 (String.length s - 1) else s in
  match Unit.scan_decimal digits with
  | Ok (m, k) -> (negative, m, k)
  | Error Not_decimal -> invalid_arg (strf "%s: %S is not a decimal" fn s)
  | Error Mantissa_past ->
      invalid_arg (strf "%s: %S has a mantissa of 2^62 or more" fn s)
  | Error (Exponent_past _) -> exponent_error fn s

(* [add], [mul] and [pow] raise where the exact result leaves int, or is
   [min_int], so that negating it is exact. *)
let add fn s a b =
  let r = a + b in
  if (a >= 0 = (b >= 0) && r >= 0 <> (a >= 0)) || r = min_int then
    exponent_error fn s
  else r

let mul fn s a b =
  if a <> 0 && b > max_int / a then exponent_error fn s else a * b

let pow fn s b k =
  let r = ref 1 in
  for _ = 1 to k do
    r := mul fn s !r b
  done;
  !r

(* [strip p m] is [(m', a)] with [m = m' p^a] and [m'] prime to [p], for [m >
   0]. *)
let strip p m =
  let m = ref m and a = ref 0 in
  while !m mod p = 0 do
    m := !m / p;
    incr a
  done;
  (!m, !a)

(* [pogson fn s base] is 10^(−2v/5) [base] for the decimal [v] of [s]: the unit
   of magnitude [v] on the scale whose magnitude 0 is [base].

   With v = ±m 10^k, −2v/5 is ∓m 2^(1 + k) 5^(k − 1). Moving m's factors 2 and 5
   into those powers leaves the exponent n/d in lowest terms: d holds only 2s
   and 5s, and n neither where d has it. *)
let pogson fn s base =
  let negative, m, k = parse fn s in
  if m = 0 then base
  else begin
    let m, a = strip 2 m in
    let m, b = strip 5 m in
    let p2 = add fn s (add fn s 1 k) a and p5 = add fn s (add fn s k (-1)) b in
    (* [up p e] is p^e for e > 0, and [down p e] p^-e for e < 0 *)
    let up p e = pow fn s p (max e 0) and down p e = pow fn s p (max (-e) 0) in
    let n = mul fn s m (mul fn s (up 2 p2) (up 5 p5)) in
    let d = mul fn s (down 2 p2) (down 5 p5) in
    let n = if negative then n else -n in
    Unit.((root d (int 10) ** n) * base)
  end

(* Oke & Gunn (1983), ApJ 266, 713, §II: AB = −2.5 log f_ν − 48.60, f_ν in erg
   cm^-2 s^-1 Hz^-1, so AB magnitude 0 is 10^(−48.60/2.5) of that unit. The sign
   is PhotDM 1.1's eq. 12, f_ν = 10^(−0.4 (m_AB + 48.6)). *)
let ab_offset = "48.6"
let erg = Unit.(decimal "1e-7" * joule)
let cgs_flux_density = Unit.(erg / second / (centi metre ** 2) / hertz)
let maggy = pogson "Units.maggy" ab_offset cgs_flux_density
let zero_point m = pogson "Units.zero_point" m maggy
