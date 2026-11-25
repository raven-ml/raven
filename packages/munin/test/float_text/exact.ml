(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The exact reference of the float suites: whether a decimal reads as a float,
   decided in integers. [float_of_string] is the C library's [strtod], which
   Windows' C runtime does not promise to round correctly, and the laws must
   hold on every system. *)

module F = Munin_float_text.Float_text

(* Naturals in limbs of 24 bits, least significant first. [mul a m c] is [a m +
   c] for [m] and [c] below 2^24, so every product stays below 2^48. *)
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

let rec pow10 k = if k = 0 then 1 else 10 * pow10 (k - 1)

(* [times a ~twos ~tens] is [a 2^twos 10^tens], 2^23 or 10^7 a step, both below
   2^24. *)
let rec times a ~twos ~tens =
  if twos > 0 then times (mul a (1 lsl min twos 23) 0) ~twos:(twos - 23) ~tens
  else if tens > 0 then
    let k = min tens 7 in
    times (mul a (pow10 k) 0) ~twos ~tens:(tens - k)
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

(* [compare_dec (d, e) n q] compares [d 10^e] with [n 2^q], for [n] below 2^62,
   by multiplying both by [10^-e] and [2^-q] where those are whole. *)
let compare_dec (d, e) n q =
  let digit a c = mul a 10 (Char.code c - Char.code '0') in
  let dec = String.fold_left digit [| 0 |] d in
  let bin = [| n land 0xFF_FFFF; (n lsr 24) land 0xFF_FFFF; n lsr 48 |] in
  compare_nat
    (times dec ~twos:(max 0 (-q)) ~tens:(max 0 e))
    (times bin ~twos:(max 0 q) ~tens:(max 0 (-e)))

(* [decode f x] is [(c, q, qmin)] with [|x| = c 2^q], [q] at least the
   subnormals' exponent [qmin]. *)
let decode (f : F.format) x =
  let x = Float.abs x in
  let qmin = f.emin - f.precision + 1 in
  let q =
    if x = 0. then qmin else max (snd (Float.frexp x) - f.precision) qmin
  in
  (Float.to_int (Float.ldexp x (-q)), q, qmin)

(* [reads f x de] is whether the decimal [de] reads as [|x|] in [f], to nearest,
   ties to even. [|x| = c 2^q]; the reals that read as it lie between the
   midpoints [(2c - 1) 2^(q-1)] and [(2c + 1) 2^(q-1)], or from [(4c - 1)
   2^(q-2)] at the bottom of a binade above the subnormals, the ends included
   when [c] is even. *)
let reads (f : F.format) x de =
  let c, q, qmin = decode f x in
  let inside o = o < 0 || (o = 0 && c land 1 = 0) in
  let above =
    if c = 0 then true
    else if c = 1 lsl (f.precision - 1) && q > qmin then
      inside (-compare_dec de ((4 * c) - 1) (q - 2))
    else inside (-compare_dec de ((2 * c) - 1) (q - 1))
  in
  above && inside (compare_dec de ((2 * c) + 1) (q - 1))

(* [overflows de] is whether [de] is at or past the binary64 overflow threshold,
   the midpoint [(2^54 - 1) 2^970] between the greatest float and 2^1024, which
   reads as an infinity since the greatest float is odd. *)
let overflows de = compare_dec de ((1 lsl 54) - 1) 970 >= 0

(* [decimal s] is the sign and the decimal of the finite number [s] writes, as
   [(d, e)] without leading or trailing zeros, [("0", 0)] for zero. *)
let decimal s =
  let neg = String.length s > 0 && s.[0] = '-' in
  let s = if neg then String.sub s 1 (String.length s - 1) else s in
  let m, x =
    match String.index_opt (String.lowercase_ascii s) 'e' with
    | None -> (s, 0)
    | Some i ->
        let t = String.sub s (i + 1) (String.length s - i - 1) in
        let t =
          if String.length t > 0 && t.[0] = '+' then
            String.sub t 1 (String.length t - 1)
          else t
        in
        (String.sub s 0 i, int_of_string t)
  in
  let frac =
    match String.index_opt m '.' with
    | None -> 0
    | Some i -> String.length m - i - 1
  in
  let d = String.concat "" (String.split_on_char '.' m) in
  let n = String.length d in
  let first = ref 0 and last = ref (n - 1) in
  while !first < n && d.[!first] = '0' do
    incr first
  done;
  while !last >= !first && d.[!last] = '0' do
    decr last
  done;
  if !first > !last then (neg, ("0", 0))
  else
    (neg, (String.sub d !first (!last - !first + 1), x - frac + (n - 1 - !last)))

(* [pow5 k] is the digits of 5^k, by schoolbook multiplication in base 10. *)
let pow5 k =
  let a = Array.make ((7 * k / 10) + 2) 0 in
  a.(0) <- 1;
  for _ = 1 to k do
    let carry = ref 0 in
    for i = 0 to Array.length a - 1 do
      let t = (a.(i) * 5) + !carry in
      a.(i) <- t mod 10;
      carry := t / 10
    done
  done;
  let top = ref (Array.length a - 1) in
  while a.(!top) = 0 do
    decr top
  done;
  String.init (!top + 1) (fun i -> Char.chr (48 + a.(!top - i)))
