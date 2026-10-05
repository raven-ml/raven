(* Correctly rounded references

   [exp2], [log2] and [sin] of a float, computed on integers in fixed point with
   [prec] bits of fraction and rounded once to a float type. A value [n *
   2^-prec] errs by a few units of [2^-prec], far below any rounding of a float
   type: the result is correctly rounded unless the exact value lies within
   [2^-200] of a tie, which no input of these functions reaches. *)

open Tolk
module Z = Bigint

let prec = 256
let one = Z.shift_left Z.one prec
let mul a b = Z.shift_right (Z.mul a b) prec
let div a b = Z.div (Z.shift_left a prec) b

(* [fixed x] is [x], finite, in fixed point; bits below [2^-prec] are lost. *)
let fixed x =
  let f, e = Float.frexp x in
  let m = Z.of_float (Float.ldexp f 53) and e = e - 53 + prec in
  if e >= 0 then Z.shift_left m e else Z.div m (Z.shift_left Z.one (-e))

(* [sum term] is the sum of [term i], from [i = 0] until a term vanishes. *)
let sum term =
  let rec go i acc =
    let t = term i in
    if Z.equal t Z.zero then acc else go (i + 1) (Z.add acc t)
  in
  go 0 Z.zero

(* [atanh_series t] is [atanh t] for [|t| <= 1/3], at [mul]'s precision. *)
let atanh_series t =
  let t2 = mul t t in
  let power = ref t in
  sum (fun i ->
      let p = !power in
      power := mul p t2;
      Z.div p (Z.of_int ((2 * i) + 1)))

let ln2 = lazy (Z.shift_left (atanh_series (Z.div one (Z.of_int 3))) 1)

(* [taylor ~step ~alternate ~start z] is the sum of [z^k / k!] over [k = start,
   start + step, ...], with signs alternating from [+] if [alternate]. *)
let taylor ?(step = 1) ?(alternate = false) ~start z =
  let t = ref one and k = ref 0 in
  let next () =
    incr k;
    t := Z.div (mul !t z) (Z.of_int !k)
  in
  while !k < start do
    next ()
  done;
  sum (fun i ->
      let v = !t in
      for _ = 1 to step do
        next ()
      done;
      if alternate && i land 1 = 1 then Z.neg v else v)

(* [round dt n p] is [n * 2^-p] rounded to nearest, ties to even, in the float
   type [dt]: an infinity past its greatest value, its subnormals below its
   least normal. *)
let round dt n p =
  let eb, mb = Dtype.finfo dt in
  let emax = (1 lsl (eb - 1)) - 1 in
  let emin = 1 - emax in
  let negative = Z.sign n < 0 and a = Z.abs n in
  if Z.equal a Z.zero then 0.
  else
    let e = Z.numbits a - 1 - p in
    let q = Int.max e emin - mb in
    let drop = q + p in
    let m =
      if drop <= 0 then Z.shift_left a (-drop)
      else
        let m = Z.shift_right a drop in
        let rest = Z.sub a (Z.shift_left m drop) in
        let half = Z.shift_left Z.one (drop - 1) in
        let c = Z.compare rest half in
        if c > 0 || (c = 0 && Z.to_int (Z.logand m Z.one) = 1) then Z.succ m
        else m
    in
    let v = Float.ldexp (Z.to_float m) q in
    let v = if v >= Float.ldexp 1. (emax + 1) then Float.infinity else v in
    if negative then -.v else v

let exp2 dt x =
  if Float.is_nan x then Float.nan
  else if x >= 1100. then Float.infinity
  else if x <= -1200. then 0.
  else
    let k = Float.floor x in
    let z = mul (fixed (x -. k)) (Lazy.force ln2) in
    round dt (taylor ~start:0 z) (prec - int_of_float k)

let log2 dt x =
  if Float.is_nan x || x < 0. then Float.nan
  else if x = 0. then Float.neg_infinity
  else if x = Float.infinity then x
  else
    let f, e = Float.frexp x in
    let m = fixed (2. *. f) in
    let t = div (Z.sub m one) (Z.add m one) in
    let ln_m = Z.shift_left (atanh_series t) 1 in
    let v =
      Z.add (Z.shift_left (Z.of_int (e - 1)) prec) (div ln_m (Lazy.force ln2))
    in
    round dt v prec

(* pi to [wide] bits, by Machin's formula: enough that [x mod pi/2] keeps [prec]
   bits for a float64's greatest [x]. *)
let wide = prec + 1200

let pi =
  lazy
    (let w = Z.shift_left Z.one wide in
     let atan_inv n =
       let n2 = Z.of_int (n * n) in
       let power = ref (Z.div w (Z.of_int n)) in
       sum (fun i ->
           let p = !power in
           power := Z.div p n2;
           let t = Z.div p (Z.of_int ((2 * i) + 1)) in
           if i land 1 = 1 then Z.neg t else t)
     in
     Z.sub
       (Z.mul (Z.of_int 16) (atan_inv 5))
       (Z.mul (Z.of_int 4) (atan_inv 239)))

let sin dt x =
  if not (Float.is_finite x) then Float.nan
  else if Float.abs x < 0x1p-30 then x
  else
    let f, e = Float.frexp (Float.abs x) in
    let m = Z.of_float (Float.ldexp f 53) and e = e - 53 + wide in
    let a =
      if e >= 0 then Z.shift_left m e else Z.div m (Z.shift_left Z.one (-e))
    in
    let half_pi = Z.shift_right (Lazy.force pi) 1 in
    let k = Z.div a half_pi in
    let r = Z.shift_right (Z.sub a (Z.mul k half_pi)) (wide - prec) in
    let s = taylor ~step:2 ~alternate:true ~start:1 r in
    let c = taylor ~step:2 ~alternate:true ~start:0 r in
    let v =
      match Z.to_int (Z.logand k (Z.of_int 3)) with
      | 0 -> s
      | 1 -> c
      | 2 -> Z.neg s
      | _ -> Z.neg c
    in
    round dt (if x < 0. then Z.neg v else v) prec

(* Distance

   The units in the last place between two values of a float type, counted on
   its ordered line: [-0.] lies one below [0.], an infinity one past the
   greatest value. A NaN is at no distance from a NaN, and at [max_int] from any
   number. *)

let rank dt v =
  let w = 8 * Dtype.itemsize dt in
  let unsigned : Dtype.t =
    match w with 16 -> Uint16 | 32 -> Uint32 | _ -> Uint64
  in
  match Dtype.bitcast dt unsigned (`Float v) with
  | `Int bits ->
      let magnitude = Z.extract bits 0 (w - 1) in
      if Float.sign_bit v then Z.sub Z.minus_one magnitude else magnitude
  | _ -> invalid_arg "a float's bits"

let ulps dt expected actual =
  if Float.is_nan expected || Float.is_nan actual then
    if Float.is_nan expected && Float.is_nan actual then 0 else max_int
  else
    let d = Z.abs (Z.sub (rank dt expected) (rank dt actual)) in
    if Z.fits_int d then Z.to_int d else max_int
