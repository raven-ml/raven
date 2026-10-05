(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type base = Prime of int | Pi

type error =
  | Zero
  | Subnormal
  | Overflow
  | Not_integer
  | Out_of_range
  | Too_wide
  | Boolean

let budget = 1 lsl 16

exception Wide

let check n = if Nat.bit_length n > budget then raise_notrace Wide else n

let ceil_shift n s =
  let q = Nat.shift_right n s in
  if Nat.low_bits_zero n s then q else Nat.add q Nat.one

let ceil_div n d =
  let q, r = Nat.div_rem n d in
  if Nat.is_zero r then q else Nat.add q Nat.one

let floor_div a b = if a >= 0 then a / b else -((-a + b - 1) / b)

let int_bits n =
  let rec loop n k = if n = 0 then k else loop (n lsr 1) (k + 1) in
  loop n 0

(* Float formats. [max] is the largest finite value; a format without infinity
   (float8_e4m3) has its own. *)

type format = { precision : int; emin : int; max : float }

let float16 = { precision = 11; emin = -14; max = 65504. }
let float32 = { precision = 24; emin = -126; max = Float.ldexp 0xffffffp0 104 }
let float64 = { precision = 53; emin = -1022; max = Float.max_float }
let bfloat16 = { precision = 8; emin = -126; max = Float.ldexp 0xffp0 120 }
let float8_e4m3 = { precision = 4; emin = -6; max = 448. }
let float8_e5m2 = { precision = 3; emin = -14; max = 57344. }
let emax f = snd (Float.frexp f.max) - 1

let classify f r =
  if r = 0. then Error Zero
  else if r < Float.ldexp 1. f.emin then Error Subnormal
  else if r > f.max then Error Overflow
  else Ok r

(* [round_ratio f n d z] is [n / d * 2^z] rounded to [f] with ties to even,
   subnormals, and an exponent unbounded above. The result is a float64, which
   holds it exactly; past float64's range it is infinity. *)
let round_ratio f n d z =
  let a = Nat.bit_length n - Nat.bit_length d in
  let at_least_2a =
    if a >= 0 then Nat.compare n (Nat.shift_left d a) >= 0
    else Nat.compare (Nat.shift_left n (-a)) d >= 0
  in
  let exp = (if at_least_2a then a else a - 1) + z in
  let u = max exp f.emin - f.precision + 1 in
  let s = z - u in
  let num, den =
    if s >= 0 then (Nat.shift_left n s, d) else (n, Nat.shift_left d (-s))
  in
  let q, r = Nat.div_rem num den in
  let q = Option.get (Nat.to_int q) in
  let c = Nat.compare (Nat.shift_left r 1) den in
  let q = if c > 0 || (c = 0 && q land 1 = 1) then q + 1 else q in
  Float.ldexp (Float.of_int q) u

(* Enclosures: [{ lo; hi; e }] bounds a positive real [x] with [lo * 2^e <= x <=
   hi * 2^e] and [lo >= 1]. Each operation keeps about [k] significant bits and
   rounds its bounds outwards. *)

type enclosure = { lo : Nat.t; hi : Nat.t; e : int }

let trunc k x =
  let s = Nat.bit_length x.lo - k in
  if s <= 0 then x
  else { lo = Nat.shift_right x.lo s; hi = ceil_shift x.hi s; e = x.e + s }

let mul k a b =
  trunc k { lo = Nat.mul a.lo b.lo; hi = Nat.mul a.hi b.hi; e = a.e + b.e }

let rec pow k a n =
  if n = 1 then a
  else
    let h = pow k (mul k a a) (n / 2) in
    if n land 1 = 1 then mul k h a else h

let inv k a =
  let m = Nat.bit_length a.hi + k in
  let p = check (Nat.shift_left Nat.one m) in
  { lo = fst (Nat.div_rem p a.hi); hi = ceil_div p a.lo; e = -a.e - m }

(* The radicand is scaled to about [k * n] bits so that the root keeps [k]. *)
let root k a n =
  if n > budget then raise_notrace Wide;
  let need = max 0 ((k * n) - Nat.bit_length a.lo) in
  let e = floor_div (a.e - need) n in
  let r = a.e - (n * e) in
  let lo = check (Nat.shift_left a.lo r)
  and hi = check (Nat.shift_left a.hi r) in
  let hi' = Nat.root hi n in
  let hi' = if Nat.equal (Nat.pow hi' n) hi then hi' else Nat.add hi' Nat.one in
  { lo = Nat.root lo n; hi = hi'; e }

let ratio k n d =
  let m = k + Nat.bit_length d - Nat.bit_length n in
  let n, d =
    if m >= 0 then (Nat.shift_left n m, d) else (n, Nat.shift_left d (-m))
  in
  let q, r = Nat.div_rem n d in
  { lo = q; hi = (if Nat.is_zero r then q else Nat.add q Nat.one); e = -m }

(* π by Machin's formula, π = 16 arctan(1/5) - 4 arctan(1/239), in fixed point
   with [guard] extra bits. Each series term is floored twice and is within 3 of
   its true value, and the series stops at a term below 2, which bounds the rest
   of the alternating series; the error budget [err] counts both. *)

let guard = 32

let machin t =
  let w = t + guard in
  let arctan_inv x =
    let x2 = x * x in
    let power = ref (Nat.div_int (Nat.shift_left Nat.one w) x) in
    let pos = ref Nat.zero and neg = ref Nat.zero and k = ref 0 in
    while not (Nat.is_zero !power) do
      let term = Nat.div_int !power ((2 * !k) + 1) in
      if !k land 1 = 0 then pos := Nat.add !pos term
      else neg := Nat.add !neg term;
      power := Nat.div_int !power x2;
      incr k
    done;
    (Nat.sub !pos !neg, !k)
  in
  let a5, k5 = arctan_inv 5 and a239, k239 = arctan_inv 239 in
  let s = Nat.sub (Nat.mul (Nat.of_int 16) a5) (Nat.mul (Nat.of_int 4) a239) in
  let err = Nat.of_int ((16 * ((3 * k5) + 2)) + (4 * ((3 * k239) + 2))) in
  {
    lo = Nat.shift_right (Nat.sub s err) guard;
    hi = ceil_shift (Nat.add s err) guard;
    e = -t;
  }

(* π to [pi_bits] bits, which covers every float format's first attempts. *)
let pi_bits = 1100
let pi_table = machin pi_bits

let pi_bounds t =
  if t > budget then raise_notrace Wide;
  if t > pi_bits then machin t
  else
    let s = pi_bits - t in
    {
      lo = Nat.shift_right pi_table.lo s;
      hi = ceil_shift pi_table.hi s;
      e = -t;
    }

let pi_pow k num den =
  let t = k + int_bits (abs num) + 8 in
  let x = pow t (pi_bounds t) (abs num) in
  let x = if num < 0 then inv t x else x in
  if den = 1 then trunc k x else root k x den

let prime_root k p num den =
  let bits = Float.of_int num *. Float.log2 (Float.of_int p) in
  if den > budget || bits > Float.of_int budget then raise_notrace Wide;
  let x = check (Nat.pow (Nat.of_int p) num) in
  root k { lo = x; hi = x; e = 0 } den

(* A number split for evaluation: [2^two * num / den * Π roots * π^pi], where
   [num] and [den] are odd naturals, and [roots] holds each prime with a
   fractional exponent part [f/d], [0 < f < d]. *)

type parts = {
  two : int;
  num : Nat.t;
  den : Nat.t;
  roots : (int * int * int) list;
  pi : int * int;
}

let log2_pi = Float.log2 Float.pi
let log2_of = function Prime p -> Float.log2 (Float.of_int p) | Pi -> log2_pi

(* [log2_bounds v] bounds [log2 v] from float arithmetic, widened by [2^-40] of
   the terms' magnitudes and [2^-20]. [round_float]'s cut-offs sit a binade
   beyond the thresholds they guard, so any error below 1 is safe. *)
let log2_bounds v =
  let est, mag =
    List.fold_left
      (fun (est, mag) (b, num, den) ->
        let x = Float.of_int num /. Float.of_int den *. log2_of b in
        (est +. x, mag +. Float.abs x))
      (0., 0.) v
  in
  let err = (mag *. 0x1p-40) +. 0x1p-20 in
  (est -. err, est +. err)

(* [natural_of ps] is the product of [ps]'s powers, or [Wide] when its estimated
   width is past the budget. *)
let natural_of ps =
  let bits =
    List.fold_left
      (fun acc (p, k) -> acc +. (Float.of_int k *. Float.log2 (Float.of_int p)))
      0. ps
  in
  if bits > Float.of_int (budget + 1) then raise_notrace Wide;
  List.fold_left
    (fun acc (p, k) -> check (Nat.mul acc (Nat.pow (Nat.of_int p) k)))
    Nat.one ps

let split v =
  let two = ref 0 and num = ref [] and den = ref [] and roots = ref [] in
  let pi = ref (0, 1) in
  List.iter
    (fun (b, n, d) ->
      match b with
      | Pi -> pi := (n, d)
      | Prime p ->
          let q = floor_div n d in
          let f = n - (q * d) in
          if f <> 0 then roots := (p, f, d) :: !roots;
          if p = 2 then two := q
          else if q > 0 then num := (p, q) :: !num
          else if q < 0 then den := (p, -q) :: !den)
    v;
  {
    two = !two;
    num = natural_of !num;
    den = natural_of !den;
    roots = List.rev !roots;
    pi = !pi;
  }

let enclose k p =
  let x = ratio k p.num p.den in
  let x =
    List.fold_left (fun x (q, f, d) -> mul k x (prime_root k q f d)) x p.roots
  in
  let x = match p.pi with 0, _ -> x | n, d -> mul k x (pi_pow k n d) in
  { x with e = x.e + p.two }

(* An irrational number is never a rounding boundary, so doubling the precision
   of its enclosure ends once both bounds round alike; rounding is monotone, so
   the number rounds as they do. *)
let rec refine f p k =
  if k > budget then raise_notrace Wide;
  let x = enclose k p in
  let lo = round_ratio f x.lo Nat.one x.e
  and hi = round_ratio f x.hi Nat.one x.e in
  if lo > f.max then Error Overflow
  else if hi = 0. then Error Zero
  else if lo = hi then classify f lo
  else if lo > 0. && hi < Float.ldexp 1. f.emin then Error Subnormal
  else refine f p (2 * k)

let round_float f v =
  let lo2, hi2 = log2_bounds v in
  if lo2 > Float.of_int (emax f + 2) then Error Overflow
  else if hi2 < Float.of_int (f.emin - f.precision - 1) then Error Zero
  else
    match split v with
    | exception Wide -> Error Too_wide
    | { roots = []; pi = 0, _; num; den; two } ->
        classify f (round_ratio f num den two)
    | p -> ( try refine f p (f.precision + 30) with Wide -> Error Too_wide)

(* Integers *)

let round_int max v =
  let power = function Prime p, n, 1 when n > 0 -> Some (p, n) | _ -> None in
  let ps = List.filter_map power v in
  if List.compare_lengths ps v <> 0 then Error Not_integer
  else
    let lo2, _ = log2_bounds v in
    if lo2 > 65. then Error Out_of_range
    else
      let n = natural_of ps in
      if Nat.compare n max > 0 then Error Out_of_range else Ok n

let max_bits bits = Nat.sub (Nat.shift_left Nat.one bits) Nat.one
let to_int n = Option.get (Nat.to_int n)

let round : type a b.
    (a, b) Nx_dtype.t -> (base * int * int) list -> (a, error) result =
 fun d v ->
  let complex f =
    Result.map (fun re -> { Complex.re; im = 0. }) (round_float f v)
  in
  let small bits = Result.map to_int (round_int (max_bits bits) v) in
  let int32 bits =
    Result.map (fun n -> Int32.of_int (to_int n)) (round_int (max_bits bits) v)
  in
  let int64 bits = Result.map Nat.to_int64 (round_int (max_bits bits) v) in
  match d with
  | Float16 -> round_float float16 v
  | Float32 -> round_float float32 v
  | Float64 -> round_float float64 v
  | BFloat16 -> round_float bfloat16 v
  | Float8_e4m3 -> round_float float8_e4m3 v
  | Float8_e5m2 -> round_float float8_e5m2 v
  | Complex64 -> complex float32
  | Complex128 -> complex float64
  | Int4 -> small 3
  | UInt4 -> small 4
  | Int8 -> small 7
  | UInt8 -> small 8
  | Int16 -> small 15
  | UInt16 -> small 16
  | Int32 -> int32 31
  | UInt32 -> int32 32
  | Int64 -> int64 63
  | UInt64 -> int64 64
  | Bool | Bit -> Error Boolean
