(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Decimal digits of floats, exact on every system.

   A finite float is c 2^q for integers c and q, so its value, and the ends of
   the interval of reals that read back as it, are exact decimals: c 2^q is
   c 5^-q 10^q when q < 0. This module writes them as decimal naturals and
   then only cuts, compares and increments digits. Nothing rounds on the way,
   so every answer is the exact one on every system. The float operations
   are frexp, ldexp, fmod, and in [limbs] a subtraction and a division whose
   results are integers below 2^53, all exact, and no table is needed.

   A natural is an array of limbs, least significant first, perhaps with
   zero limbs on top. Limbs are in base 10^9 where [int] has 63 bits and in
   base 10^4 where it has 31 or 32, as under js_of_ocaml: a product of two
   limbs, or of a limb and a factor below [room], plus two limbs, stays
   inside [int]. The answers do not depend on the base.

   Every function reads [|x|]. Callers take the sign from [Float.sign_bit x],
   so that [-0.] and a negative number that rounds to zero keep it:
   [decimals 2 (-0.001)] is [("0", 0)], which C writes [-0.00]. *)

type format = { precision : int; emin : int }

let binary64 = { precision = 53; emin = -1022 }
let binary32 = { precision = 24; emin = -126 }

(* Naturals *)

(* A primitive, so that the compiler folds the base to a constant and divides
   by multiplying. *)
external int_size : unit -> int = "%int_size"

let wide = int_size () >= 63
let digits = if wide then 9 else 4
let base = if wide then 1_000_000_000 else 10_000
let room = if wide then 1 lsl 30 else 1 lsl 16

(* [top a] is the number of limbs of [a] up to its last nonzero one. *)
let top a =
  let n = ref (Array.length a) in
  while !n > 0 && a.(!n - 1) = 0 do
    decr n
  done;
  !n

(* [pow b k] is [b^k], [b] 2 or 5, multiplied in place by factors below
   [room]. [b^k] has at most [0.7 k + 1] digits. *)
let pow b k =
  let a = Array.make ((((7 * k) / 10) + 1) / digits + 2) 0 in
  a.(0) <- 1;
  let live = ref 1 and k = ref k in
  while !k > 0 do
    let m = ref b and j = ref 1 in
    while !j < !k && !m * b < room do
      m := !m * b;
      incr j
    done;
    let carry = ref 0 in
    for i = 0 to !live - 1 do
      let t = (a.(i) * !m) + !carry in
      a.(i) <- t mod base;
      carry := t / base
    done;
    while !carry > 0 do
      a.(!live) <- !carry mod base;
      carry := !carry / base;
      incr live
    done;
    k := !k - !j
  done;
  a

(* [scale a m] is [a m], [m] below [room]: the last carry is below [m], so
   two limbs hold it. *)
let scale a m =
  let n = Array.length a in
  let r = Array.make (n + 2) 0 and carry = ref 0 in
  for i = 0 to n - 1 do
    let t = (a.(i) * m) + !carry in
    r.(i) <- t mod base;
    carry := t / base
  done;
  r.(n) <- !carry mod base;
  r.(n + 1) <- !carry / base;
  r

let mul a b =
  let na = top a and nb = top b in
  let r = Array.make (na + nb + 1) 0 in
  for j = 0 to nb - 1 do
    let carry = ref 0 in
    for i = 0 to na - 1 do
      let t = r.(i + j) + (a.(i) * b.(j)) + !carry in
      r.(i + j) <- t mod base;
      carry := t / base
    done;
    r.(na + j) <- !carry
  done;
  r

let compare a b =
  let n = top a in
  if n <> top b then Int.compare n (top b)
  else begin
    let i = ref (n - 1) in
    while !i > 0 && a.(!i) = b.(!i) do
      decr i
    done;
    if n = 0 then 0 else Int.compare a.(!i) b.(!i)
  end

(* [trunc a i] is [a] without its limbs below [i]. *)
let trunc a i =
  let r = Array.copy a in
  Array.fill r 0 (Int.min i (Array.length a)) 0;
  r

(* [bump a i d] is [a + d base^i], [d] from 0 to [base]. *)
let bump a i d =
  let r = Array.make (Int.max (Array.length a) i + 2) 0 in
  Array.blit a 0 r 0 (Array.length a);
  let i = ref i and carry = ref d in
  while !carry > 0 do
    let t = r.(!i) + !carry in
    r.(!i) <- t mod base;
    carry := t / base;
    incr i
  done;
  r

let limb a i = if i < Array.length a then a.(i) else 0

(* [ndigits a] is the number of decimal digits of the nonzero [a]. *)
let ndigits a =
  let rec width l = if l < 10 then 1 else 1 + width (l / 10) in
  let n = top a in
  (digits * (n - 1)) + width a.(n - 1)

(* [text a e] is the nonzero decimal [a 10^e] as its digits, without leading
   or trailing zeros, and the exponent of the last one. Only the limbs from
   the lowest nonzero one are written. *)
let text a e =
  let n = top a and low = ref 0 in
  while a.(!low) = 0 do
    incr low
  done;
  let len = digits * (n - !low) in
  let b = Bytes.create len in
  for i = !low to n - 1 do
    let l = ref a.(i) in
    for j = 1 to digits do
      Bytes.set b
        (len - (digits * (i - !low)) - j)
        (Char.chr (48 + (!l mod 10)));
      l := !l / 10
    done
  done;
  let first = ref 0 and last = ref (len - 1) in
  while Bytes.get b !first = '0' do
    incr first
  done;
  while Bytes.get b !last = '0' do
    decr last
  done;
  ( Bytes.sub_string b !first (!last - !first + 1),
    e + (digits * !low) + len - 1 - !last )

(* [align a e p] is [(a', i, e')], [a 10^e] written as [a' 10^e'] with the
   unit of limb [i] at [10^p] and a limb below it, for [p > e]. *)
let align a e p =
  let j = digits - ((p - e) mod digits) in
  let rec ten j = if j = 0 then 1 else 10 * ten (j - 1) in
  (scale a (ten j), (p - e + j) / digits, e - j)

(* Floats *)

(* [decode f x] is [(c, q)] with [|x| = c 2^q], [c] below [2^precision] and
   [q] at least the subnormals' exponent, [c] as a float, which holds it
   exactly. A precision past 53 would let [c] pass 2^53, where [limbs]'s float
   arithmetic rounds. *)
let decode f x =
  if f.precision < 1 || f.precision > 53 then
    invalid_arg "Float_text: precision outside 1 to 53";
  if not (Float.is_finite x) then invalid_arg "Float_text: not finite";
  let x = Float.abs x in
  let q =
    Int.max (snd (Float.frexp x) - f.precision) (f.emin - f.precision + 1)
  in
  let c = Float.ldexp x (-q) in
  if not (Float.is_integer c) then
    invalid_arg "Float_text: not a value of the format";
  (c, q)

(* [limbs c d] is [4c + d], [c] an integer below 2^53 and [d] from 0 to 3,
   read from the float by exact remainders. [base] is a multiple of 4, so the
   low limb of [4c] is at most [base - 4] and adding [d] carries nothing. *)
let limbs c d =
  let a = Array.make 6 0 and c = ref c and i = ref 0 in
  while !c > 0. do
    let r = Float.rem !c (Float.of_int base) in
    a.(!i) <- Float.to_int r;
    c := (!c -. r) /. Float.of_int base;
    incr i
  done;
  let a = scale a 4 in
  a.(0) <- a.(0) + d;
  a

(* [quarter q] is [(u, e)] with [2^(q - 2) = u 10^e]. *)
let quarter q = if q >= 2 then (pow 2 (q - 2), 0) else (pow 5 (2 - q), q - 2)

(* Shortest *)

(* The reals that read back as v = c 2^q, to nearest, ties to even, lie
   between the midpoints to its neighbours, the ends included when c is even:
   from (4c - 2) 2^(q-2) to (4c + 2) 2^(q-2), or from (4c - 1) 2^(q-2) when
   v is the least of its binade above the subnormals, whose neighbour below
   is half as far. Let 10^k <= hi - lo < 10^(k+1) and s = floor(v / 10^k).

   The interval holds at most one multiple of 10^(k+1); when s >= 10, such a
   multiple has fewer digits than any other decimal in it, so it is the
   answer when there is one. Otherwise the interval holds a multiple of 10^k
   (its width is 10^k or more, and equals it only for q = 0, where its ends
   are not integers), and s 10^k <= v < (s + 1) 10^k, so if any multiple of
   10^k at or below v is in it, s 10^k is, and above, (s + 1) 10^k. The
   answer is whichever of the two is in it, the nearer to v when both are,
   the even one at a tie. A decimal in the interval whose last digit is below
   10^k has more digits than either, unless it is one digit d 10^(k-1) below
   10^k. Then s < 10, since s >= 10 would put it more than 9 10^k below v,
   past the half-width, which is under 5 10^k. And s >= 1, since the interval
   is no wider than v, so s 10^k lies between it and v: inside, and
   nearer. *)
let shortest f x =
  let c, q = decode f x in
  if c = 0. then ("0", 0)
  else
    let u, e = quarter q in
    let closer =
      c = Float.ldexp 1. (f.precision - 1) && q > f.emin - f.precision + 1
    in
    let k = e + ndigits (scale u (if closer then 3 else 4)) - 1 in
    let u, i, e = align u e k in
    let lo = mul u (limbs (c -. 1.) (if closer then 3 else 2)) in
    let v = mul u (limbs c 0) and hi = mul u (limbs c 2) in
    let even = Float.rem c 2. = 0. in
    let inside t =
      let l = compare lo t and h = compare t hi in
      (l < 0 || (l = 0 && even)) && (h < 0 || (h = 0 && even))
    in
    let d = trunc v i in
    let up = bump d i 1 in
    let below = Array.copy d in
    below.(i) <- d.(i) - (d.(i) mod 10);
    (* [below] is 0 when s < 10, and 0 is never inside, since lo > 0. *)
    let t =
      if inside below then below
      else if top below > 0 && inside (bump below i 10) then bump below i 10
      else if not (inside up) then d
      else if not (inside d) then up
      else
        let m = compare v (bump d (i - 1) (base / 2)) in
        if m < 0 || (m = 0 && d.(i) mod 2 = 0) then d else up
    in
    text t e

(* Rounding *)

(* [exact x] is [(v, e)] with [|x| = v 10^e]. *)
let exact x =
  let c, q = decode binary64 x in
  let u, e = quarter q in
  (mul u (limbs c 0), e)

(* [round v e p] is the nonzero [v 10^e] rounded to a multiple of [10^p], to
   nearest, ties to even, for [e < p <= e + ndigits v]. *)
let round v e p =
  let v, i, e = align v e p in
  let d = trunc v i in
  let m = compare v (bump d (i - 1) (base / 2)) in
  let r = if m > 0 || (m = 0 && limb d i mod 2 = 1) then bump d i 1 else d in
  if top r = 0 then ("0", 0) else text r e

(* [n] is compared before any arithmetic on it, so that [max_int] and
   [min_int] cannot wrap: [e] and [ndigits v] stay within a few thousand. *)
let significant n x =
  if n < 1 then invalid_arg "Float_text.significant: fewer than one digit";
  let v, e = exact x in
  if top v = 0 then ("0", 0)
  else if n >= ndigits v then text v e
  else round v e (e + ndigits v - n)

(* Past [e + ndigits v], the value is under a tenth of [10^-n] and rounds to
   0. *)
let decimals n x =
  let v, e = exact x in
  if top v = 0 then ("0", 0)
  else if n >= -e then text v e
  else if n < -(e + ndigits v) then ("0", 0)
  else round v e (-n)
