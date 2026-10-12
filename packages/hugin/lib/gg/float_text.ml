(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Exact decimals *)

(* A decimal [(d, e)] is the natural of digits [d], most significant first,
   times [10^e], without leading or trailing zero; zero is [("", 0)]. A float
   [n 2^k] is [n 2^k] or [n 5^-k 10^k], computed in little-endian limbs of
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

(* [times a k] is the limbs [a] times [k], below [2^31]: each product and
   carry stays below [2^61], and the last carry takes up to two limbs. *)
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
      Bytes.unsafe_set b ((9 * (n - i)) - j) (Char.unsafe_chr (48 + (!v mod 10)));
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

(* [increment d] is the decimal digits [d] plus one. *)
let increment d =
  let b = Bytes.of_string d in
  let rec carry i =
    if i < 0 then "1" ^ Bytes.to_string b
    else if Bytes.get b i = '9' then begin
      Bytes.set b i '0';
      carry (i - 1)
    end
    else begin
      Bytes.set b i (Char.chr (Char.code (Bytes.get b i) + 1));
      Bytes.to_string b
    end
  in
  carry (String.length d - 1)

(* [round n (d, e)] is [(d, e)] rounded to its first [n] digits, ties to even;
   [n] may be [0] or less, which rounds below the first digit. *)
let round n (d, e) =
  let len = String.length d in
  if len <= n then (d, e)
  else if n < 0 then ("", 0)
  else
    let head = String.sub d 0 n and tail = String.sub d n (len - n) in
    (* [tail] has no trailing zero: it is half a unit exactly iff it is [5]. *)
    let odd = n > 0 && (Char.code head.[n - 1] - 48) land 1 = 1 in
    let up = tail.[0] > '5' || (tail.[0] = '5' && (len - n > 1 || odd)) in
    normalize (if up then increment head else head) (e + len - n)

(* Writing *)

let special x =
  if Float.is_nan x then Some "nan"
  else if x = Float.infinity then Some "inf"
  else if x = Float.neg_infinity then Some "-inf"
  else None

let sign x = if Float.sign_bit x then "-" else ""

(* [layout p (d, e)] writes the nonzero decimal [(d, e)] of at most [p]
   digits as [%.*g] does. *)
let layout p (d, e) =
  let n = String.length d in
  let x = n - 1 + e in
  if x < -4 || x >= p then
    let fraction = if n > 1 then "." ^ String.sub d 1 (n - 1) else "" in
    Printf.sprintf "%c%se%c%02d" d.[0] fraction
      (if x < 0 then '-' else '+')
      (Int.abs x)
  else if e >= 0 then d ^ String.make e '0'
  else if x >= 0 then String.sub d 0 (x + 1) ^ "." ^ String.sub d (x + 1) (-e)
  else "0." ^ String.make (-x - 1) '0' ^ d

let g ?(precision = 6) x =
  match special x with
  | Some s -> s
  | None ->
      if x = 0. then sign x ^ "0"
      else
        let p = Int.max 1 precision in
        sign x ^ layout p (round p (decimal x))

let pp ppf x = Format.pp_print_string ppf (g x)

let f decimals x =
  match special x with
  | Some s -> s
  | None ->
      let decimals = Int.max 0 decimals in
      let ((d, e) as v) = decimal x in
      let d, e = round (String.length d + e + decimals) v in
      (* The value times [10^decimals], as an integer of at least [decimals +
         1] digits. *)
      let scaled = if d = "" then "0" else d ^ String.make (e + decimals) '0' in
      let scaled =
        String.make (Int.max 0 (decimals + 1 - String.length scaled)) '0'
        ^ scaled
      in
      let int = String.length scaled - decimals in
      sign x ^ String.sub scaled 0 int
      ^ if decimals = 0 then "" else "." ^ String.sub scaled int decimals

(* [reads_back x] is whether a decimal rounds to the positive finite [x], ties
   to even: whether it lies between the midpoints to [x]'s neighbours, the
   ends included when [x] is even. *)
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

let exact x =
  match special x with
  | Some s -> s
  | None ->
      if x = 0. then sign x ^ "0"
      else
        let v = decimal x and reads = reads_back (Float.abs x) in
        let rec first p =
          let c = round p v in
          if p >= 17 || reads c then layout p c else first (p + 1)
        in
        sign x ^ first 15

let pp_exact ppf x = Format.pp_print_string ppf (exact x)
