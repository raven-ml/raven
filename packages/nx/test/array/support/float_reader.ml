(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type format = { precision : int; emin : int }

(* Naturals *)

(* Naturals in limbs of 24 bits, least significant first. [mul a m c] is
   [a m + c] for [m] and [c] below 2^24, so every product stays below 2^48. *)
let mul a m c =
  let carry = ref c in
  let r =
    Array.map
      (fun l ->
        let t = (l * m) + !carry in
        carry := t lsr 24;
        t land 0xFF_FFFF)
      a
  in
  if !carry = 0 then r else Array.append r [| !carry |]

let rec times a ~twos ~tens =
  if twos > 0 then times (mul a (1 lsl min twos 23) 0) ~twos:(twos - 23) ~tens
  else if tens > 0 then times (mul a 10 0) ~twos ~tens:(tens - 1)
  else a

let compare_nat a b =
  let top a =
    let n = ref (Array.length a) in
    while !n > 0 && a.(!n - 1) = 0 do
      decr n
    done;
    !n
  in
  let n = top a and m = top b in
  if n <> m then Int.compare n m
  else
    let i = ref (n - 1) in
    while !i >= 0 && a.(!i) = b.(!i) do
      decr i
    done;
    if !i < 0 then 0 else Int.compare a.(!i) b.(!i)

(* [compare_dec (d, e) n q] compares [d 10^e] with [n 2^q], for [n] below
   2^62, by multiplying both by [10^-e] and [2^-q] where those are whole. *)
let compare_dec (d, e) n q =
  let digit a c = mul a 10 (Char.code c - Char.code '0') in
  let dec = String.fold_left digit [| 0 |] d in
  let bin = [| n land 0xFF_FFFF; (n lsr 24) land 0xFF_FFFF; n lsr 48 |] in
  compare_nat
    (times dec ~twos:(max 0 (-q)) ~tens:(max 0 e))
    (times bin ~twos:(max 0 q) ~tens:(max 0 (-e)))

(* Reading *)

(* [|x| = c 2^q] with [q] at least the subnormals' exponent [qmin]. *)
let decode f x =
  let x = Float.abs x in
  let qmin = f.emin - f.precision + 1 in
  let q =
    if x = 0. then qmin else max (snd (Float.frexp x) - f.precision) qmin
  in
  (Float.to_int (Float.ldexp x (-q)), q, qmin)

let compare de x =
  let c, q, _ = decode { precision = 53; emin = -1022 } x in
  compare_dec de c q

(* The reals that read as [c 2^q] lie between the midpoints
   [(2c - 1) 2^(q-1)] and [(2c + 1) 2^(q-1)], or from [(4c - 1) 2^(q-2)] at
   the bottom of a binade above the subnormals, the ends included when [c] is
   even. *)
let reads f x de =
  let c, q, qmin = decode f x in
  let inside o = o < 0 || (o = 0 && c land 1 = 0) in
  let above =
    if c = 0 then true
    else if c = 1 lsl (f.precision - 1) && q > qmin then
      inside (-compare_dec de ((4 * c) - 1) (q - 2))
    else inside (-compare_dec de ((2 * c) - 1) (q - 1))
  in
  above && inside (compare_dec de ((2 * c) + 1) (q - 1))
