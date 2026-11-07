(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Exact decimals *)

(* A decimal [(d, e)] is the natural of digits [d], most significant first,
   times [10^e], without leading or trailing zero; zero is [("", 0)]. A float [n
   2^k] is [n 2^k] or [n 5^-k 10^k], computed in little-endian limbs of
   [10^9]. *)

let base = 1_000_000_000

let normalize d e =
  let n = String.length d in
  let lead = ref 0 and stop = ref n in
  while !lead < n && d.[!lead] = '0' do
    incr lead
  done;
  while !stop > !lead && d.[!stop - 1] = '0' do
    decr stop
  done;
  if !lead = !stop then ("", 0)
  else (String.sub d !lead (!stop - !lead), e + n - !stop)

(* [times a k] is the limbs [a] times [k], below [2^31]: each product and carry
   stays below [2^61], and the last carry takes up to two limbs. *)
let times a k =
  let n = Array.length a in
  let r = Array.make (n + 2) 0 and carry = ref 0 in
  for i = 0 to n - 1 do
    let t = (a.(i) * k) + !carry in
    r.(i) <- t mod base;
    carry := t / base
  done;
  r.(n) <- !carry mod base;
  r.(n + 1) <- !carry / base;
  let top = ref (n + 2) in
  while !top > 0 && r.(!top - 1) = 0 do
    decr top
  done;
  Array.sub r 0 !top

(* [power b k] is [b^k] for an [int]. *)
let rec power b k = if k = 0 then 1 else b * power b (k - 1)

(* [dyadic n k] is the decimal of [n 2^k], [n] at least [0]. 2^30 and 5^13 are
   the greatest powers below 2^31. *)
let dyadic n k =
  let rec limbs n = if n = 0 then [] else (n mod base) :: limbs (n / base) in
  let a = ref (Array.of_list (limbs n)) in
  let b, chunk = if k >= 0 then (2, 30) else (5, 13) in
  let left = ref (Int.abs k) in
  while !left > 0 do
    let j = Int.min chunk !left in
    a := times !a (power b j);
    left := !left - j
  done;
  let a = !a in
  let n = Array.length a in
  let b = Bytes.create (9 * n) in
  for i = 0 to n - 1 do
    let v = ref a.(i) in
    for j = 1 to 9 do
      Bytes.unsafe_set b
        ((9 * (n - i)) - j)
        (Char.unsafe_chr (48 + (!v mod 10)));
      v := !v / 10
    done
  done;
  normalize (Bytes.unsafe_to_string b) (Int.min k 0)

(* [decimal x] is the decimal of [|x|], [x] finite. *)
let decimal x =
  if x = 0. then ("", 0)
  else
    let f, e = Float.frexp (Float.abs x) in
    let m = ref (Float.to_int (Float.ldexp f 53)) and k = ref (e - 53) in
    while !m land 1 = 0 do
      m := !m lsr 1;
      incr k
    done;
    dyadic !m !k

let compare_decimal (d0, e0) (d1, e1) =
  match (d0, d1) with
  | "", "" -> 0
  | "", _ -> -1
  | _, "" -> 1
  | _ ->
      let m0 = String.length d0 + e0 and m1 = String.length d1 + e1 in
      if m0 <> m1 then Int.compare m0 m1 else String.compare d0 d1

(* [reads_back x] is whether a decimal rounds to the positive finite [x], ties
   to even: whether it lies between the midpoints to [x]'s neighbours, the ends
   included when [x] is even. *)
let reads_back x =
  let _, e = Float.frexp x in
  let u = Int.max (e - 53) (-1074) in
  let m = Float.to_int (Float.ldexp x (-u)) in
  let even = m land 1 = 0 in
  let lo =
    if m = 1 lsl 52 && u > -1074 then dyadic ((4 * m) - 1) (u - 2)
    else dyadic ((2 * m) - 1) (u - 1)
  in
  let hi = dyadic ((2 * m) + 1) (u - 1) in
  fun c ->
    let l = compare_decimal c lo and h = compare_decimal c hi in
    (l > 0 || (l = 0 && even)) && (h < 0 || (h = 0 && even))

(* Reading *)

(* The C library's strtod, behind [float_of_string], is not correctly rounded
   everywhere: under mingw-w64 it reads 3.839223843572815e-239 one float above
   its nearest. A reading is kept when it is exact by construction (Clinger's
   fast path: an integer of at most 2^53 times or over a power of ten that is
   itself exact), and otherwise checked against the text's exact decimal, moving
   to the float the decimal rounds to. *)

(* JSON's number: [-]int[.digits][(e|E)[+-]digits], [int] a zero alone or digits
   not starting with one. *)

let is_exp c = c = 'e' || c = 'E'

let digits_end s i stop =
  let i = ref i in
  while !i < stop && s.[!i] >= '0' && s.[!i] <= '9' do
    incr i
  done;
  !i

let number_end s i stop =
  let i = if i < stop && s.[i] = '-' then i + 1 else i in
  let j = if i < stop && s.[i] = '0' then i + 1 else digits_end s i stop in
  let j =
    if j = i then -1
    else if j < stop && s.[j] = '.' then
      let k = digits_end s (j + 1) stop in
      if k = j + 1 then -1 else k
    else j
  in
  if j >= 0 && j < stop && is_exp s.[j] then
    let k = j + 1 in
    let k = if k < stop && (s.[k] = '+' || s.[k] = '-') then k + 1 else k in
    let l = digits_end s k stop in
    if l = k then -1 else l
  else j

(* [exponent s i stop] is the exponent whose sign or first digit is at [i], in a
   number [number_end] took. *)
let exponent s i stop =
  let sign = if s.[i] = '-' then -1 else 1 in
  let i = if s.[i] = '-' || s.[i] = '+' then i + 1 else i in
  let v = ref 0 in
  for k = i to stop - 1 do
    (* Beyond any float's range either way; keep it from overflowing. *)
    if !v < 100_000 then v := (10 * !v) + Char.code s.[k] - 48
  done;
  sign * !v

(* [parse s] is the sign and the decimal of the number [s] writes, or [None]. *)
let parse s =
  let n = String.length s in
  if n = 0 || number_end s 0 n <> n then None
  else
    let neg = s.[0] = '-' in
    let digits = Buffer.create 24 and frac = ref 0 and point = ref false in
    let i = ref (if neg then 1 else 0) in
    while !i < n && not (is_exp s.[!i]) do
      if s.[!i] = '.' then point := true
      else begin
        Buffer.add_char digits s.[!i];
        if !point then incr frac
      end;
      incr i
    done;
    let e = if !i < n then exponent s (!i + 1) n else 0 in
    Some (neg, normalize (Buffer.contents digits) (e - !frac))

(* The powers of ten a float holds exactly, [10^0] to [10^22]. *)
let exact_tens =
  Array.init 23 (fun k -> Float.of_string ("1e" ^ string_of_int k))

(* [fast (d, e)] is the float nearest [(d, e)] when one rounding gives it. *)
let fast (d, e) =
  if d = "" then Some 0.
  else if String.length d > 16 || e < -22 || e > 22 then None
  else
    let m = int_of_string d in
    if m > 1 lsl 53 then None
    else
      let m = Float.of_int m in
      Some (if e >= 0 then m *. exact_tens.(e) else m /. exact_tens.(-e))

(* [side c b] is negative, zero or positive as the decimal [c] rounds below, to
   or above the nonnegative finite float of bits [b]. *)
let side c b =
  let y = Int64.float_of_bits b in
  if y = 0. then
    (* Zero is even, so a decimal half the least subnormal rounds to it. *)
    if compare_decimal c (dyadic 1 (-1075)) <= 0 then 0 else 1
  else if reads_back y c then 0
  else compare_decimal c (decimal y)

let max_bits = Int64.bits_of_float Float.max_float

(* [nearest c x] is the nonnegative float [c] rounds to, ties to even, searched
   from [x]: doubling steps bracket it, and halving the bracket finds it. An [x]
   that is not finite, as the C library reads a negative decimal beyond the
   greatest float, starts from the greatest float. *)
let nearest c x =
  let start =
    if Float.is_finite x then Int64.bits_of_float (Float.abs x) else max_bits
  in
  match side c start with
  | 0 -> Int64.float_of_bits start
  | dir -> (
      let step b k = if dir > 0 then Int64.add b k else Int64.sub b k in
      let clamp b = Int64.max 0L (Int64.min max_bits b) in
      (* [lo] is past on [dir]'s side, [hi] beyond the answer or at it. *)
      let rec gallop lo k =
        let b = clamp (step start k) in
        let s = side c b in
        if s = 0 then `Found b
        else if s = dir then
          if b = max_bits && dir > 0 then `Inf
          else if b = 0L then `Found 0L
          else gallop b (Int64.mul k 2L)
        else `Bracket (lo, b)
      in
      let rec bisect lo hi =
        let mid = Int64.add lo (Int64.div (Int64.sub hi lo) 2L) in
        if mid = lo || mid = hi then hi
        else
          match side c mid with
          | 0 -> mid
          | s when s = dir -> bisect mid hi
          | _ -> bisect lo mid
      in
      match gallop start 1L with
      | `Inf -> Float.infinity
      | `Found b -> Int64.float_of_bits b
      | `Bracket (lo, hi) -> Int64.float_of_bits (bisect lo hi))

let read s =
  match parse s with
  | None -> None
  | Some (neg, c) ->
      let x =
        match fast c with
        | Some x -> x
        | None ->
            (* Beyond these the value is an infinity or a zero whatever its
               digits. *)
            let d, e = c in
            let mag = String.length d + e in
            if mag > 310 then Float.infinity
            else if mag < -324 then 0.
            else nearest c (float_of_string s)
      in
      Some (if neg then Float.neg x else x)

(* Reading without allocating

   Most numbers a store holds have at most 18 significant digits [m] and a
   decimal exponent [e] from [-22] to [22]: [m] fits an [int] and [10^|e|] is a
   float exactly. Such a number is read with no exact decimal. A candidate from
   two roundings lies within a few floats of the answer, and whether [m 10^e]
   rounds to a float [x] is the sign of [m 10^e] less each midpoint around [x].
   That sign is the sign of a sum of a few floats, each exact by an error-free
   product (an fma) or a power of two; Shewchuk's Grow-Expansion ("Adaptive
   Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates",
   1997, section 2.6) gives the sign of such a sum exactly. *)

let max_digits = 18
let max_ten = 22

(* [sum_sign a n] is the sign of the exact sum of the first [n] elements of [a],
   which it overwrites. Adding each element to the expansion of those before it
   by error-free sums leaves a nonoverlapping expansion of increasing magnitude,
   zeros aside, whose sign is that of its last nonzero component. *)
let sum_sign a n =
  for i = 1 to n - 1 do
    let q = ref (Float.Array.unsafe_get a i) in
    for j = 0 to i - 1 do
      let b = Float.Array.unsafe_get a j in
      let s = !q +. b in
      let bv = s -. !q in
      Float.Array.unsafe_set a j (!q -. (s -. bv) +. (b -. bv));
      q := s
    done;
    Float.Array.unsafe_set a i !q
  done;
  let sign = ref 0 and k = ref (n - 1) in
  while !sign = 0 && !k >= 0 do
    let c = Float.Array.unsafe_get a !k in
    if c > 0. then sign := 1 else if c < 0. then sign := -1;
    decr k
  done;
  !sign

(* [above m e x h a] is the sign of [m 10^e - (x + h)], for [m] below 2^60 and
   [h] a power of two or its negation. [m] is split in two floats, each exact.
   Each product is stored before it is used, so that no compiler fuses it into a
   sum. *)
let above m e x h a =
  let t = Array.unsafe_get exact_tens (Int.abs e) in
  let mh = Float.of_int (m land lnot 0xFF)
  and ml = Float.of_int (m land 0xFF) in
  if e >= 0 then begin
    (* m t - x - h *)
    Float.Array.unsafe_set a 0 (mh *. t);
    Float.Array.unsafe_set a 1 (Float.fma mh t (-.Float.Array.unsafe_get a 0));
    Float.Array.unsafe_set a 2 (ml *. t);
    Float.Array.unsafe_set a 3 (Float.fma ml t (-.Float.Array.unsafe_get a 2));
    Float.Array.unsafe_set a 4 (-.x);
    Float.Array.unsafe_set a 5 (-.h);
    sum_sign a 6
  end
  else begin
    (* The sign of m / t - x - h is that of m - x t - h t. *)
    Float.Array.unsafe_set a 0 mh;
    Float.Array.unsafe_set a 1 ml;
    Float.Array.unsafe_set a 2 (x *. t);
    Float.Array.unsafe_set a 3 (Float.fma x t (-.Float.Array.unsafe_get a 2));
    Float.Array.unsafe_set a 2 (-.Float.Array.unsafe_get a 2);
    Float.Array.unsafe_set a 3 (-.Float.Array.unsafe_get a 3);
    Float.Array.unsafe_set a 4 (-.(h *. t));
    sum_sign a 5
  end

(* [direction m e x a] is [0] if [m 10^e] rounds to the positive normal [x],
   ties to even, and otherwise the side its float lies on. *)
let direction m e x a =
  let even = Int64.logand (Int64.bits_of_float x) 1L = 0L in
  let hi = above m e x ((Float.succ x -. x) /. 2.) a in
  if hi > 0 || (hi = 0 && not even) then 1
  else
    let lo = above m e x (-.(x -. Float.pred x) /. 2.) a in
    if lo < 0 || (lo = 0 && not even) then -1 else 0

(* [nearest_int m e] is the float nearest [m 10^e], [m] from 2^53 to 2^60, or
   [nan] if a few steps do not find it. *)
let nearest_int m e =
  let t = Array.unsafe_get exact_tens (Int.abs e) in
  let x = ref (if e >= 0 then Float.of_int m *. t else Float.of_int m /. t) in
  let a = Float.Array.create 6 in
  let dir = ref (direction m e !x a) and steps = ref 0 in
  while !dir <> 0 && !steps < 4 do
    x := if !dir > 0 then Float.succ !x else Float.pred !x;
    dir := direction m e !x a;
    incr steps
  done;
  if !dir = 0 then !x else Float.nan

(* [quick s pos len] is the number [s] writes from [pos] for [len] bytes when it
   is in the range above, and [nan] otherwise: text that is not a number, more
   digits or a larger exponent. Its loops take no closure, so that their state
   stays in registers. *)
let quick s pos len =
  let stop = pos + len in
  if len = 0 || number_end s pos stop <> stop then Float.nan
  else begin
    let neg = s.[pos] = '-' in
    let i = ref (if neg then pos + 1 else pos) in
    (* [m] holds the significant digits, [shift] the fraction's digits. *)
    let m = ref 0 and count = ref 0 and shift = ref 0 in
    let point = ref false and ok = ref true in
    while !i < stop && not (is_exp (String.unsafe_get s !i)) do
      let c = String.unsafe_get s !i in
      if c = '.' then point := true
      else begin
        let d = Char.code c - 48 in
        if !m > 0 || d > 0 then
          if !count < max_digits then begin
            m := (10 * !m) + d;
            incr count
          end
          else ok := false;
        if !point then incr shift
      end;
      incr i
    done;
    if not !ok then Float.nan
    else begin
      let e =
        ref ((if !i < stop then exponent s (!i + 1) stop else 0) - !shift)
      in
      while !m > 0 && !m mod 10 = 0 do
        m := !m / 10;
        incr e
      done;
      let m = !m and e = !e in
      let x =
        if m = 0 then 0.
        else if e < -max_ten || e > max_ten then Float.nan
        else if m <= 1 lsl 53 then
          (* Clinger's fast path: one rounding of exact operands. *)
          let t = Array.unsafe_get exact_tens (Int.abs e) in
          if e >= 0 then Float.of_int m *. t else Float.of_int m /. t
        else nearest_int m e
      in
      if neg then Float.neg x else x
    end
  end

let of_substring s pos len =
  if pos < 0 || len < 0 || pos > String.length s - len then
    invalid_arg "Float_read.of_substring";
  let x = quick s pos len in
  if Float.is_nan x then read (String.sub s pos len) else Some x

let of_string s = of_substring s 0 (String.length s)
