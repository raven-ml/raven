(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* References for the kinds' bounds, shared by the tests of nx_kinds.h and of
   the kernels: distances in ordered-bit ranks, the goldens of gen/kinds.py,
   each value correctly rounded with its remainder, and double-doubles. *)

(* Distances in ordered-bit ranks, -0 and +0 one apart; NaN is at no distance
   from NaN and infinitely far from a number. *)

let rank64 x =
  let b = Int64.bits_of_float x in
  if Int64.compare b 0L < 0 then Int64.(sub (neg (logand b max_int)) 1L) else b

let ulps64 a b =
  if Float.is_nan a || Float.is_nan b then
    if Float.is_nan a && Float.is_nan b then 0 else max_int
  else
    let d = Int64.abs (Int64.sub (rank64 a) (rank64 b)) in
    if Int64.compare d (Int64.of_int max_int) >= 0 then max_int
    else Int64.to_int d

(* The rows of golden/kinds/<kind>.golden, from dev/nx2/test/kernel:
   operands, then the value correctly rounded and its remainder. *)
let golden kind =
  let ic = open_in (Filename.concat "golden/kinds" (kind ^ ".golden")) in
  let rows = ref [] in
  (try
     while true do
       rows := input_line ic :: !rows
     done
   with End_of_file -> close_in ic);
  List.rev !rows
  |> List.filter (fun l -> l <> "" && l.[0] <> '#')
  |> List.tl
  |> List.map (fun l ->
      String.split_on_char '\t' l |> List.map float_of_string |> Array.of_list)

(* The exact values below are double-doubles, hi + lo. *)
let two_sum a b =
  let s = a +. b in
  let bb = s -. a in
  (s, a -. (s -. bb) +. (b -. bb))

let two_prod a b =
  let p = a *. b in
  (p, Float.fma a b (-.p))

let dd_add (ah, al) (bh, bl) =
  let s, e = two_sum ah bh in
  two_sum s (e +. al +. bl)

let dd_neg (h, l) = (-.h, -.l)

let dd_mul (ah, al) (bh, bl) =
  let p, e = two_prod ah bh in
  two_sum p (e +. (ah *. bl) +. (al *. bh))

let dd_div ((ah, _) as a) ((bh, _) as b) =
  let q1 = ah /. bh in
  let r = dd_add a (dd_neg (dd_mul b (q1, 0.))) in
  two_sum q1 (fst r /. bh)

(* The exact product and quotient of (ar + ai i) and (br + bi i) *)
let exact_mul (ar, ai) (br, bi) =
  ( dd_add (two_prod ar br) (dd_neg (two_prod ai bi)),
    dd_add (two_prod ar bi) (two_prod ai br) )

let exact_div (ar, ai) (br, bi) =
  let den = dd_add (two_prod br br) (two_prod bi bi) in
  ( dd_div (dd_add (two_prod ar br) (two_prod ai bi)) den,
    dd_div (dd_add (two_prod ai br) (dd_neg (two_prod ar bi))) den )

(* |z' - z| / |z| in units of [u]. *)
let normwise u (re, im) ((zr, _) as r, ((zi, _) as i)) =
  let d x z = fst (dd_add (x, 0.) (dd_neg z)) in
  Float.hypot (d re r) (d im i) /. Float.hypot zr zi /. u

let round_f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* The rows of golden/kinds/complex-<kind>.golden: operands, then each part
   of the value correctly rounded and its remainder. *)
let complex_golden kind = golden ("complex-" ^ kind)

(* Whether a row's operands are binary32's and its value is in the range
   c64's bounds hold in. *)
let single_row args (re, im) =
  Array.for_all (fun a -> round_f32 a = a) args
  && Float.is_finite (round_f32 (Float.hypot (fst re) (fst im)))
  && Float.hypot (fst re) (fst im) >= 0x1p-100

let golden_value row n = ((row.(n), row.(n + 1)), (row.(n + 2), row.(n + 3)))
