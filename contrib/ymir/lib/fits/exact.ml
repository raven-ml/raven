(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Floats and the decimals that read as them, decided exactly. The C library
   rounds some decimals wrong both ways: under mingw-w64, msvcrt's printf keeps
   17 correct digits and rounds from those, so [%.15e] of 2^-1024 gives
   5.562684646268004e-309 for 5.562684646268003e-309, and its strtod reads
   3.839223843572815e-239 as 2^-792, one float above the nearest. *)

(* Naturals *)

(* Little-endian limbs of [bits] bits. *)
let bits = 24
let mask = (1 lsl bits) - 1

let of_int n =
  let rec go n acc =
    if n = 0 then List.rev acc else go (n lsr bits) ((n land mask) :: acc)
  in
  Array.of_list (go n [])

(* [a * k + c] for [k] and [c] below 2^30. *)
let mul_add a k c =
  let n = Array.length a in
  let r = Array.make (n + 2) 0 and carry = ref c in
  for i = 0 to n - 1 do
    let v = (a.(i) * k) + !carry in
    r.(i) <- v land mask;
    carry := v lsr bits
  done;
  r.(n) <- !carry land mask;
  r.(n + 1) <- !carry lsr bits;
  r

let mul a k = mul_add a k 0

(* 5^0 to 5^12: a limb times 5^12 fits an int. *)
let fives =
  let a = Array.make 13 1 in
  for k = 1 to 12 do
    a.(k) <- 5 * a.(k - 1)
  done;
  a

let rec pow5 a q =
  if q > 12 then pow5 (mul a fives.(12)) (q - 12) else mul a fives.(q)

let pow2 a p =
  let a = Array.append (Array.make (p / bits) 0) a in
  mul a (1 lsl (p mod bits))

let top a =
  let i = ref (Array.length a - 1) in
  while !i >= 0 && a.(!i) = 0 do
    decr i
  done;
  !i

let compare_nat a b =
  let ta = top a and tb = top b in
  if ta <> tb then Int.compare ta tb
  else
    let rec go i =
      if i < 0 then 0
      else if a.(i) <> b.(i) then Int.compare a.(i) b.(i)
      else go (i - 1)
    in
    go ta

(* Numbers [n 2^p 5^q] *)

let compare (n1, p1, q1) (n2, p2, q2) =
  let p = Int.min p1 p2 and q = Int.min q1 q2 in
  let side n pn qn = pow2 (pow5 n (qn - q)) (pn - p) in
  compare_nat (side n1 p1 q1) (side n2 p2 q2)

(* [m 10^e]. *)
let decimal m e = (m, e, e)

(* The nonnegative finite float [x] as [n 2^p], [2^p] its ulp: the floats
   next to it are [(n ± 1) 2^p], but for the closer one below a power of
   two. *)
let parts x =
  let b = Int64.bits_of_float x in
  let field = Int64.to_int (Int64.shift_right_logical b 52) land 0x7ff in
  let frac = Int64.to_int (Int64.logand b 0xf_ffff_ffff_ffffL) in
  if field = 0 then (frac, -1074) else (frac lor (1 lsl 52), field - 1075)

let exact x =
  let n, p = parts x in
  (of_int n, p, 0)

(* The midpoints between [x] and the floats below and above it. *)
let below x =
  let n, p = parts x in
  if n = 1 lsl 52 && p > -1074 then (of_int ((4 * n) - 1), p - 2, 0)
  else (of_int ((2 * n) - 1), p - 1, 0)

let above x =
  let n, p = parts x in
  (of_int ((2 * n) + 1), p - 1, 0)

let even x = fst (parts x) land 1 = 0

(* Whether [d] lies beyond the midpoint [m] next to [x], on the side [sign]
   gives (-1 below [x], 1 above): there it reads as the neighbour, and at the
   midpoint too when [x]'s significand is odd. *)
let past ~sign d m x =
  let c = sign * compare d m in
  c > 0 || (c = 0 && not (even x))

(* Text *)

(* [split t] is [(negative, digits, e)] with the real text [t] (FITS 4.0
   §4.2.4) equal to [digits 10^e], [digits] without leading or trailing
   zeros, and zero as [(negative, "", 0)]; [None] when the exponent does not
   fit an int. *)
let split t =
  let n = String.length t in
  let negative = n > 0 && t.[0] = '-' in
  let i = if n > 0 && (t.[0] = '-' || t.[0] = '+') then 1 else 0 in
  let is_exponent c = c = 'E' || c = 'D' || c = 'e' || c = 'd' in
  let mantissa_end =
    let j = ref i in
    while !j < n && not (is_exponent t.[!j]) do
      incr j
    done;
    !j
  in
  let exponent =
    if mantissa_end = n then Some 0
    else
      let s = String.sub t (mantissa_end + 1) (n - mantissa_end - 1) in
      int_of_string_opt
        (if s <> "" && s.[0] = '+' then String.sub s 1 (String.length s - 1)
         else s)
  in
  match exponent with
  | None -> None
  | Some exponent ->
      let whole, frac =
        let mantissa = String.sub t i (mantissa_end - i) in
        match String.index_opt mantissa '.' with
        | None -> (mantissa, "")
        | Some j ->
            ( String.sub mantissa 0 j,
              String.sub mantissa (j + 1) (String.length mantissa - j - 1) )
      in
      let digits = whole ^ frac in
      let e = exponent - String.length frac in
      let l = ref 0 and r = ref (String.length digits) in
      while !l < !r && digits.[!l] = '0' do
        incr l
      done;
      while !r > !l && digits.[!r - 1] = '0' do
        decr r
      done;
      if !l = !r then Some (negative, "", 0)
      else
        Some
          ( negative,
            String.sub digits !l (!r - !l),
            e + (String.length digits - !r) )

(* Reading *)

(* 10^0 to 10^22, each a float. *)
let tens =
  let a = Array.make 23 1. in
  for k = 1 to 22 do
    a.(k) <- 10. *. a.(k - 1)
  done;
  a

(* [read digits e] is the float nearest [digits 10^e], ties to even, or
   [None] past the largest float. Up to 15 digits and 10^±22, both are floats
   and one product or quotient rounds once. Otherwise the C library's reading,
   a few floats away at most, moves to the nearest. *)
let read digits e =
  let n = String.length digits in
  if digits = "" || e + n < -324 then Some 0.
  else if e + n - 1 > 308 then None
  else if n <= 15 && Int.abs e <= 22 then
    let m = float_of_int (int_of_string digits) in
    Some (if e >= 0 then m *. tens.(e) else m /. tens.(-e))
  else
    let d =
      decimal
        (String.fold_left (fun a c -> mul_add a 10 (Char.code c - 48)) [||]
           digits)
        e
    in
    let rec go x =
      if x > 0. && past ~sign:(-1) d (below x) x then go (Float.pred x)
      else if past ~sign:1 d (above x) x then
        if x = Float.max_float then None else go (Float.succ x)
      else Some x
    in
    let near = float_of_string (digits ^ "e" ^ string_of_int e) in
    go (if Float.is_finite near then near else Float.max_float)

(* Writing *)

(* Whether [d] reads as the nonnegative float [x]. *)
let reads d x =
  ((x = 0. || not (past ~sign:(-1) d (below x) x))
  && not (past ~sign:1 d (above x) x))

(* [shortest x] is [(m, e)], [m 10^e] the decimal of the fewest significant
   digits that reads as the positive finite [x], the nearest of them, ties to
   even. *)
let shortest x =
  let v = exact x in
  let rec go p =
    (* [m 10^e], [x] to [p] digits by the C library, near [x]; [lo 10^e] and
       [(lo + 1) 10^e] bracket it, and a [p]-digit decimal reads as [x] only
       if one of them does. Seventeen digits read as any float. *)
    let t = Printf.sprintf "%.*e" (p - 1) x in
    let i = String.index t 'e' in
    let m =
      int_of_string
        (String.concat "" (String.split_on_char '.' (String.sub t 0 i)))
    in
    let e =
      int_of_string (String.sub t (i + 1) (String.length t - i - 1)) - (p - 1)
    in
    let at m = decimal (of_int m) e in
    let rec bracket lo =
      if compare (at lo) v > 0 then bracket (lo - 1)
      else if compare (at (lo + 1)) v <= 0 then bracket (lo + 1)
      else lo
    in
    let lo = bracket m in
    match List.filter (fun c -> reads (at c) x) [ lo; lo + 1 ] with
    | [ c ] -> (c, e)
    | [ lo; hi ] ->
        let c = compare v (of_int ((2 * lo) + 1), e - 1, e) in
        if c < 0 || (c = 0 && lo land 1 = 0) then (lo, e) else (hi, e)
    | _ when p < 17 -> go (p + 1)
    | _ -> invalid_arg (Printf.sprintf "Exact.shortest %h" x)
  in
  go 1
