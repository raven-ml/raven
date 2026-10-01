(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Exact decimal numbers, and how the kit writes them. *)

(* [{ neg; digits; exp }] is [±digits × 10^exp]. [digits] has no leading and no
   trailing zero and is empty for zero, which is never negative and has [exp =
   0]. Each number has one representation. *)
type t = { neg : bool; digits : string; exp : int }

let zero = { neg = false; digits = ""; exp = 0 }
let is_zero d = String.length d.digits = 0

let make neg digits exp =
  let n = String.length digits in
  let i = ref 0 in
  while !i < n && String.unsafe_get digits !i = '0' do
    incr i
  done;
  let j = ref (n - 1) in
  while !j >= !i && String.unsafe_get digits !j = '0' do
    decr j
  done;
  if !j < !i then zero
  else
    {
      neg;
      digits = String.sub digits !i (!j - !i + 1);
      exp = exp + (n - 1 - !j);
    }

let abs d = { d with neg = false }
let shift k d = if is_zero d then d else { d with exp = d.exp + k }
let pow10 neg k = { neg; digits = "1"; exp = k }

(* [mag d] is the position of the first digit of the nonzero [d], the [e] of
   [10^e <= |d| < 10^(e+1)]. *)
let mag d = String.length d.digits - 1 + d.exp

(* Binary values *)

let limb = 1_000_000_000

(* [mul_small a len k] multiplies the [len] little-endian limbs of [a] by [k <=
   5^13] in place and is the new number of limbs. *)
let mul_small a len k =
  let carry = ref 0 in
  for i = 0 to len - 1 do
    let v = (Array.unsafe_get a i * k) + !carry in
    Array.unsafe_set a i (v mod limb);
    carry := v / limb
  done;
  let len = ref len in
  while !carry > 0 do
    a.(!len) <- !carry mod limb;
    carry := !carry / limb;
    incr len
  done;
  !len

let pow5_13 = 1_220_703_125

let rec of_binary neg m e =
  if m = 0 then zero
  else if e < 0 && m land 1 = 0 then of_binary neg (m lsr 1) (e + 1)
  else
    let n = Int.abs e in
    let a = Array.make (((19 + n) / 9) + 3) 0 in
    a.(0) <- m mod limb;
    a.(1) <- m / limb mod limb;
    a.(2) <- m / limb / limb;
    let len = ref 3 in
    if e >= 0 then begin
      for _ = 1 to n / 30 do
        len := mul_small a !len (1 lsl 30)
      done;
      len := mul_small a !len (1 lsl (n mod 30))
    end
    else begin
      for _ = 1 to n / 13 do
        len := mul_small a !len pow5_13
      done;
      let r = ref 1 in
      for _ = 1 to n mod 13 do
        r := !r * 5
      done;
      len := mul_small a !len !r
    end;
    let b = Bytes.create (9 * !len) in
    for i = 0 to !len - 1 do
      let v = ref a.(!len - 1 - i) in
      for k = 8 downto 0 do
        Bytes.unsafe_set b ((9 * i) + k) (Char.unsafe_chr (48 + (!v mod 10)));
        v := !v / 10
      done
    done;
    make neg (Bytes.unsafe_to_string b) (Int.min e 0)

let of_float x =
  if x = 0. then zero
  else
    let f, e = Float.frexp (Float.abs x) in
    of_binary (x < 0.) (Float.to_int (Float.ldexp f 53)) (e - 53)

(* Comparing *)

let compare_mag a b =
  match (is_zero a, is_zero b) with
  | true, true -> 0
  | true, false -> -1
  | false, true -> 1
  | false, false ->
      let ma = mag a and mb = mag b in
      if ma <> mb then Int.compare ma mb
      else
        let la = String.length a.digits and lb = String.length b.digits in
        let n = Int.min la lb in
        let rec loop i =
          if i = n then Int.compare la lb
          else
            let c = Char.compare a.digits.[i] b.digits.[i] in
            if c <> 0 then c else loop (i + 1)
        in
        loop 0

(* [padded d e] is the digits of [|d|] written down to position [e <= d.exp]. *)
let padded d e = d.digits ^ String.make (d.exp - e) '0'

(* Rounding *)

type direction = Nearest | Toward_zero | Away_from_zero

(* [increment s] is the natural number [s] plus one, in decimal. *)
let increment s =
  let b = Bytes.of_string s in
  let rec loop i =
    if i < 0 then "1" ^ Bytes.unsafe_to_string b
    else if Bytes.get b i = '9' then begin
      Bytes.set b i '0';
      loop (i - 1)
    end
    else begin
      Bytes.set b i (Char.unsafe_chr (Char.code (Bytes.get b i) + 1));
      Bytes.unsafe_to_string b
    end
  in
  loop (String.length s - 1)

let round dir pos d =
  if is_zero d || pos <= d.exp then d
  else
    let keep = d.exp + String.length d.digits - pos in
    if keep < 0 then
      match dir with Away_from_zero -> pow10 d.neg pos | _ -> zero
    else
      (* The digits below [pos] are not all zero: the last digit is not. *)
      let up =
        match dir with
        | Toward_zero -> false
        | Away_from_zero -> true
        | Nearest -> d.digits.[keep] >= '5'
      in
      let kept = String.sub d.digits 0 keep in
      make d.neg (if up then increment kept else kept) pos

(* Binary formats *)

type format = { precision : int; emin : int; max : float; saturates : bool }

let binary64 =
  { precision = 53; emin = -1022; max = Float.max_float; saturates = false }

let round_half_even s =
  let f = Float.floor s in
  let r = s -. f in
  if r > 0.5 then f +. 1.
  else if r < 0.5 then f
  else if Float.rem f 2. = 0. then f
  else f +. 1.

(* [round_to_format f x] is the finite [x] rounded to [f], ties to even. *)
let round_to_format f x =
  let _, e = Float.frexp x in
  let u = Int.max (e - f.precision) (f.emin - f.precision + 1) in
  let y = Float.ldexp (round_half_even (Float.ldexp (Float.abs x) (-u))) u in
  let y =
    if y <= f.max then y else if f.saturates then f.max else Float.infinity
  in
  Float.copy_sign y x

(* [rounds_to f c y] is [true] iff the decimal [c], zero or of the sign of [y],
   rounds to [y], a finite value of [f], in [f], with ties to even. *)
let rounds_to f c y =
  let u_min = f.emin - f.precision + 1 in
  if y = 0. then compare_mag c (of_binary false 1 (u_min - 1)) <= 0
  else
    let y = Float.abs y in
    let _, e = Float.frexp y in
    let u = Int.max (e - f.precision) u_min in
    let m = Float.to_int (Float.ldexp y (-u)) in
    let even = m land 1 = 0 in
    let lo =
      if m = 1 lsl (f.precision - 1) && u > u_min then
        of_binary false ((4 * m) - 1) (u - 2)
      else of_binary false ((2 * m) - 1) (u - 1)
    in
    let above_lo =
      let k = compare_mag c lo in
      k > 0 || (even && k = 0)
    in
    above_lo
    && ((y = f.max && f.saturates)
       ||
       let k = compare_mag c (of_binary false ((2 * m) + 1) (u - 1)) in
       k < 0 || (even && k = 0))

(* Floats of decimal multiples *)

(* Natural numbers in little-endian limbs of 30 bits, without a zero limb last;
   [[||]] is zero. *)

let limb_bits = 30
let limb_mask = (1 lsl limb_bits) - 1

let nat n =
  let rec limbs n =
    if n = 0 then [] else (n land limb_mask) :: limbs (n lsr limb_bits)
  in
  Array.of_list (limbs n)

let nat_trim a =
  let n = ref (Array.length a) in
  while !n > 0 && a.(!n - 1) = 0 do
    decr n
  done;
  Array.sub a 0 !n

let nat_mul a b =
  let la = Array.length a and lb = Array.length b in
  let r = Array.make (la + lb + 1) 0 in
  for i = 0 to la - 1 do
    (* Each sum stays below [2^30 + 2^60 + 2^32], within an [int]. *)
    let carry = ref 0 in
    for j = 0 to lb - 1 do
      let t = r.(i + j) + (a.(i) * b.(j)) + !carry in
      r.(i + j) <- t land limb_mask;
      carry := t lsr limb_bits
    done;
    let k = ref (i + lb) in
    while !carry > 0 do
      let t = r.(!k) + !carry in
      r.(!k) <- t land limb_mask;
      carry := t lsr limb_bits;
      incr k
    done
  done;
  nat_trim r

let nat_shift a s =
  let q = s / limb_bits and r = s mod limb_bits in
  let la = Array.length a in
  let res = Array.make (la + q + 1) 0 in
  for i = 0 to la - 1 do
    let v = a.(i) lsl r in
    res.(i + q) <- res.(i + q) lor (v land limb_mask);
    res.(i + q + 1) <- v lsr limb_bits
  done;
  nat_trim res

let nat_compare a b =
  let la = Array.length a and lb = Array.length b in
  if la <> lb then Int.compare la lb
  else
    let rec loop i =
      if i < 0 then 0
      else
        let c = Int.compare a.(i) b.(i) in
        if c <> 0 then c else loop (i - 1)
    in
    loop (la - 1)

let pow5_13_nat = nat pow5_13

let nat_pow5 k =
  let r = ref (nat 1) in
  for _ = 1 to k / 13 do
    r := nat_mul !r pow5_13_nat
  done;
  let rest = ref 1 in
  for _ = 1 to k mod 13 do
    rest := !rest * 5
  done;
  nat_mul !r (nat !rest)

(* [nearest n k] is the float nearest [n × 10^k], ties to even. [near], within a
   few floats of it, is moved to it: [n × 10^k] is compared exactly, in binary,
   with the midpoints between [near] and its neighbours. *)
let nearest n k near =
  if n = 0 then 0.
  else
    let m = nat (Int.abs n) and p5 = nat_pow5 (Int.abs k) in
    let big = if k >= 0 then nat_mul m p5 else p5 in
    (* [vs d f] compares [|n| × 10^k] with [d × 2^f], [d >= 1]. *)
    let vs d f =
      let e = f - k in
      let l, r = if k >= 0 then (big, nat d) else (m, nat_mul (nat d) big) in
      if e >= 0 then nat_compare l (nat_shift r e)
      else nat_compare (nat_shift l (-e)) r
    in
    (* [place y] is negative, zero or positive as [|n| × 10^k] rounds below [y
       >= 0], to it, or above it. *)
    let place y =
      if y = 0. then
        (* No integer below 2^62 has the factor 5^324 a tie needs here. *)
        if vs 1 (-1075) <= 0 then 0 else 1
      else
        let mant, e =
          if y < Float.min_float then (Float.to_int (Float.ldexp y 1074), -1074)
          else
            let f, e = Float.frexp y in
            (Float.to_int (Float.ldexp f 53), e - 53)
        in
        let even = mant land 1 = 0 in
        let lo =
          if mant = 1 lsl 52 && e > -1074 then vs ((4 * mant) - 1) (e - 2)
          else vs ((2 * mant) - 1) (e - 1)
        in
        if lo < 0 || (lo = 0 && not even) then -1
        else
          let hi = vs ((2 * mant) + 1) (e - 1) in
          if hi > 0 || (hi = 0 && not even) then 1 else 0
    in
    let rec fix y =
      match place y with
      | 0 -> y
      | c when c < 0 -> fix (Float.pred y)
      | _ -> if y = Float.max_float then Float.infinity else fix (Float.succ y)
    in
    let near = Float.abs near in
    let y = fix (if Float.is_finite near then near else Float.max_float) in
    if n < 0 then -.y else y

(* Writing *)

type notation = Plain | Exponent | Si | Percent
type precision = Decimals of int | Significant of int

let superscript_digits = [| "⁰"; "¹"; "²"; "³"; "⁴"; "⁵"; "⁶"; "⁷"; "⁸"; "⁹" |]

let superscript n =
  let s = string_of_int (Int.abs n) in
  let b = Buffer.create 16 in
  if n < 0 then Buffer.add_string b "\u{207B}";
  String.iter
    (fun c -> Buffer.add_string b superscript_digits.(Char.code c - 48))
    s;
  Buffer.contents b

(* [grouped locale s] is the integer digits [s] separated into the groups of
   [locale]. *)
let grouped locale s =
  let sep = Locale.group locale in
  let n = String.length s in
  let rec cuts acc pos sizes =
    let size, rest =
      match sizes with
      | [ g ] -> (g, sizes)
      | g :: gs -> (g, gs)
      | [] -> assert false
    in
    if pos - size <= 0 then 0 :: acc
    else cuts ((pos - size) :: acc) (pos - size) rest
  in
  let starts = cuts [] n (Locale.grouping locale) in
  let b = Buffer.create 16 in
  let rec write = function
    | s0 :: (s1 :: _ as rest) ->
        Buffer.add_string b (String.sub s s0 (s1 - s0));
        Buffer.add_string b sep;
        write rest
    | [ s0 ] -> Buffer.add_string b (String.sub s s0 (n - s0))
    | [] -> ()
  in
  write starts;
  Buffer.contents b

(* [fixed decimals d] is the integer digits and the [decimals] fraction digits
   of [|d|], a multiple of [10^-decimals]. *)
let fixed decimals d =
  if is_zero d then ("0", String.make decimals '0')
  else if d.exp >= 0 then (padded d 0, String.make decimals '0')
  else
    let k = -d.exp and len = String.length d.digits in
    let pad = String.make (decimals - k) '0' in
    if len > k then
      (String.sub d.digits 0 (len - k), String.sub d.digits (len - k) k ^ pad)
    else ("0", String.make (k - len) '0' ^ d.digits ^ pad)

(* [positional locale ~group ~trim decimals d] writes [d], a multiple of
   [10^-decimals], in positional digits. *)
let positional locale ~group ~trim decimals d =
  let int, frac = fixed decimals d in
  let frac =
    if not trim then frac
    else
      let n = ref (String.length frac) in
      while !n > 0 && frac.[!n - 1] = '0' do
        decr n
      done;
      String.sub frac 0 !n
  in
  let int = if group then grouped locale int else int in
  let sign = if d.neg then Locale.minus locale else "" in
  if frac = "" then sign ^ int
  else String.concat "" [ sign; int; Locale.decimal locale; frac ]

(* [rounded p d] is [d] rounded to the precision [p] and the number of decimals
   to write it with. *)
let rounded p d =
  match p with
  | Decimals n -> (round Nearest (-n) d, n)
  | Significant n ->
      if is_zero d then (d, n - 1)
      else
        let r = round Nearest (mag d - n + 1) d in
        (r, Int.max 0 (n - 1 - mag r))

let prefixes =
  [|
    "q";
    "r";
    "y";
    "z";
    "a";
    "f";
    "p";
    "n";
    "\u{00B5}";
    "m";
    "";
    "k";
    "M";
    "G";
    "T";
    "P";
    "E";
    "Z";
    "Y";
    "R";
    "Q";
  |]

let floor_div a b = if a >= 0 then a / b else -((b - 1 - a) / b)
let ten = pow10 false 1
let thousand = pow10 false 3

let write locale ~group ~trim notation p d =
  match notation with
  | Plain ->
      let r, n = rounded p d in
      positional locale ~group ~trim n r
  | Percent ->
      let r, n = rounded p (shift 2 d) in
      positional locale ~group ~trim n r ^ "%"
  | Exponent when is_zero d ->
      let r, n = rounded p d in
      positional locale ~group ~trim n r
  | Exponent ->
      let mantissa_decimals =
        match p with Decimals n -> n | Significant n -> n - 1
      in
      let mantissa e = round Nearest (-mantissa_decimals) (shift (-e) d) in
      let e = mag d in
      let e, m =
        let m = mantissa e in
        if compare_mag m ten >= 0 then (e + 1, mantissa (e + 1)) else (e, m)
      in
      let digits = positional locale ~group ~trim mantissa_decimals (abs m) in
      let sign = if m.neg then Locale.minus locale else "" in
      let power = "10" ^ superscript e in
      if digits = "1" then sign ^ power
      else String.concat "" [ sign; digits; "×"; power ]
  | Si when is_zero d ->
      let r, n = rounded p d in
      positional locale ~group ~trim n r
  | Si ->
      let quotient k =
        let q = shift (-3 * k) d in
        let r, n = rounded p q in
        (k, r, n)
      in
      let k = Int.max (-10) (Int.min 10 (floor_div (mag d) 3)) in
      let k, r, n =
        let ((_, r, _) as q) = quotient k in
        if k < 10 && compare_mag r thousand >= 0 then quotient (k + 1) else q
      in
      positional locale ~group ~trim n r ^ prefixes.(k + 10)
