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

(* a b + c rounded once, to nearest with ties to even. Float.fma rounds
   some subnormal results twice under MinGW, so a result below the least
   normal magnitude, or one whose product is within 2^54 of it, is computed
   from its exact value in integers. *)

(* Magnitudes as arrays of 30-bit limbs, least significant first. *)
module Big = struct
  let limb = 30
  let mask = (1 lsl limb) - 1
  let of_int n = [| n land mask; (n lsr limb) land mask |]

  let normal a =
    let n = ref (Array.length a) in
    while !n > 0 && a.(!n - 1) = 0 do
      decr n
    done;
    Array.sub a 0 !n

  let get a i = if i < Array.length a then a.(i) else 0

  let mul a b =
    let r = Array.make (Array.length a + Array.length b + 1) 0 in
    for i = 0 to Array.length a - 1 do
      let carry = ref 0 in
      for j = 0 to Array.length b - 1 do
        let t = r.(i + j) + (a.(i) * b.(j)) + !carry in
        r.(i + j) <- t land mask;
        carry := t lsr limb
      done;
      r.(i + Array.length b) <- r.(i + Array.length b) + !carry
    done;
    normal r

  let shift_left a k =
    let q = k / limb and s = k mod limb in
    let r = Array.make (Array.length a + q + 1) 0 in
    for i = 0 to Array.length a - 1 do
      let v = a.(i) lsl s in
      r.(i + q) <- r.(i + q) lor (v land mask);
      r.(i + q + 1) <- v lsr limb
    done;
    normal r

  let compare a b =
    let a = normal a and b = normal b in
    let n = Array.length a in
    if n <> Array.length b then Int.compare n (Array.length b)
    else
      let rec go i =
        if i < 0 then 0
        else if a.(i) <> b.(i) then Int.compare a.(i) b.(i)
        else go (i - 1)
      in
      go (n - 1)

  let add a b =
    let n = max (Array.length a) (Array.length b) + 1 in
    let r = Array.make n 0 and carry = ref 0 in
    for i = 0 to n - 1 do
      let t = get a i + get b i + !carry in
      r.(i) <- t land mask;
      carry := t lsr limb
    done;
    normal r

  (* a - b, where a >= b. *)
  let sub a b =
    let r = Array.make (Array.length a) 0 and borrow = ref 0 in
    for i = 0 to Array.length a - 1 do
      let t = a.(i) - get b i - !borrow in
      borrow := if t < 0 then 1 else 0;
      r.(i) <- t + (!borrow lsl limb)
    done;
    normal r

  let bit a i = (get a (i / limb) lsr (i mod limb)) land 1 = 1

  let bits a =
    let a = normal a in
    let n = Array.length a in
    if n = 0 then 0
    else
      let top = ref a.(n - 1) and k = ref 0 in
      while !top <> 0 do
        top := !top lsr 1;
        incr k
      done;
      ((n - 1) * limb) + !k

  (* Bits [lo] to [lo + n - 1] of [a] as an integer, n <= 54. *)
  let field a lo n =
    let r = ref 0 in
    for i = n - 1 downto 0 do
      r := (!r lsl 1) lor if bit a (lo + i) then 1 else 0
    done;
    !r

  (* A finite nonzero x as its sign, its significand as an integer below 2^53
     and the exponent of the significand's last bit. *)
  let parts x =
    let e = max (snd (Float.frexp x) - 53) (-1074) in
    (x < 0., of_int (Float.to_int (Float.ldexp (Float.abs x) (-e))), e)

  (* [n] 2^[e] rounded once to a float of its sign [neg]: 53 bits, fewer
     below 2^-1022. *)
  let round neg n e =
    let l = bits n in
    if l = 0 then 0.
    else
      let last = max (l + e - 53) (-1074) in
      let k = last - e in
      let m =
        if k <= 0 then field n 0 l
        else
          let m = field n k (l - k) in
          let rest = ref false in
          for i = 0 to k - 2 do
            if bit n i then rest := true
          done;
          if bit n (k - 1) && (!rest || m land 1 = 1) then m + 1 else m
      in
      let r = Float.ldexp (Float.of_int m) (max last e) in
      if neg then -.r else r
end

let exact_fma a b c =
  if not (Float.is_finite a && Float.is_finite b) then (a *. b) +. c
  else if not (Float.is_finite c) then c
  else if a = 0. || b = 0. then (a *. b) +. c
  else
    let an, am, ae = Big.parts a and bn, bm, be = Big.parts b in
    let pn = an <> bn and p = Big.mul am bm and pe = ae + be in
    let cn, cm, ce = if c = 0. then (pn, [||], pe) else Big.parts c in
    let e = min pe ce in
    let p = Big.shift_left p (pe - e) and q = Big.shift_left cm (ce - e) in
    if pn = cn then Big.round pn (Big.add p q) e
    else if Big.compare p q >= 0 then Big.round pn (Big.sub p q) e
    else Big.round cn (Big.sub q p) e

let fma a b c =
  let r = Float.fma a b c in
  if Float.abs r >= Float.min_float && Float.abs (a *. b) >= 0x1p-968 then r
  else exact_fma a b c
