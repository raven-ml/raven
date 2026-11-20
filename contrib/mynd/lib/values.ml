(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Image

(* Conversion *)

(* What a dtype's elements stand for: the fraction itself, or a code of [s]
   value bits, [m = 2ˢ - 1] its largest. *)
type num = Fraction | Code of { signed : bool; s : int }

let num (type v r) fn (dt : (v, r) Nx.dtype) =
  match Nx.Dtype.kind dt with
  | Float -> Fraction
  | Unsigned -> Code { signed = false; s = Nx.Dtype.bits dt }
  | Signed -> Code { signed = true; s = Nx.Dtype.bits dt - 1 }
  | Complex | Boolean ->
      invalid_argf "%s: %s holds no fraction" fn (Nx.Dtype.name dt)

let largest s = Float.ldexp 1. s -. 1.
let mask s = if s = 64 then -1L else Int64.pred (Int64.shift_left 1L s)

(* [k / m]. A code [k] and [m] of up to 16 bits are exact in float32, whose
   quotient rounds once. A narrower [dt] rounds it again, which lands where one
   rounding does when the quotient is never within float32's rounding of a
   midpoint of [dt]: [|k / m - c| >= 1 / (m 2ᵖ⁺¹)] for a midpoint [c] of [p + 1]
   bits, above 2⁻²⁴ for codes of up to 8 bits and every narrow float. Wider
   codes into a narrow float divide at float64, and so do codes past 16 bits. *)
let of_code (type b d v r) (dt : (float, b) Nx.dtype) ~signed ~s
    (x : (v, r, d) Nx.t) : (float, b, d) Nx.t =
  let quotient (type w) (wide : (float, w) Nx.dtype) =
    let q = Nx.div_s (Nx.cast wide x) (largest s) in
    Nx.cast dt (if signed then Nx.maximum_s q (-1.) else q)
  in
  let f32 = Nx.Dtype.equal dt Nx.float32 in
  if (s <= 8 || (s <= 16 && f32)) && not (Nx.Dtype.equal dt Nx.float64) then
    quotient Nx.float32
  else quotient Nx.float64

(* Vulkan's float to normalized fixed point: clamp, multiply by [m], round to
   nearest even. The product is exact in float32 for the narrow floats and 8-bit
   codes. *)
let to_code (type a b d r) (dt : (a, b) Nx.dtype) ~signed ~s
    (x : (float, r, d) Nx.t) : (a, b, d) Nx.t =
  let code (type w) (wide : (float, w) Nx.dtype) =
    let f = Nx.cast wide x in
    let f = Nx.where (Nx.isnan f) (Nx.zeros_like f) f in
    let f = Nx.clamp ~min:(if signed then -1. else 0.) ~max:1. f in
    let c = Nx.cast dt (Nx.roundeven (Nx.mul_s f (largest s))) in
    (* At 64 bits [-m] is not a float64: the product of [-1] is [-m - 1], which
       Vulkan maps to [-m]. The cast saturates [m] itself, so [-m] is one
       negation away. *)
    if signed && s > 53 then
      Nx.maximum c (Nx.neg (Nx.cast dt (Nx.scalar_like f (largest s))))
    else c
  in
  if Nx.Dtype.equal (Nx.dtype x) Nx.float64 || s > 24 then code Nx.float64
  else code Nx.float32

(* [rescale ~s ~s' a] is the code of [s'] bits nearest [a n / m] for a code [a]
   of [s] bits, [m = 2ˢ - 1] and [n = 2ˢ' - 1], in uint64. [a / m] is [a]'s [s]
   bits repeated forever after the binary point, so [a 2ˢ' / m] is the integer
   [I] of its first [s'] bits plus [a' / m], [a'] the next [s] bits, [a] rotated
   left by [s' mod s]. The fraction [a n / m] is [I + (a' - a) / m], and [m]
   being odd no such value lies halfway between integers. [a = m] is [n]. *)
let rescale ~s ~s' a =
  let m = Nx.scalar_like a (mask s) in
  let r = s' mod s in
  let i = ref (if r = 0 then Nx.zeros_like a else Nx.rshift a (s - r)) in
  for k = 1 to s' / s do
    i := Nx.add !i (Nx.lshift a (s' - (k * s)))
  done;
  let a' =
    if r = 0 then a
    else Nx.bitwise_or (Nx.bitwise_and (Nx.lshift a r) m) (Nx.rshift a (s - r))
  in
  let half = Nx.scalar_like a (mask (s - 1)) in
  let up = Nx.logical_and (Nx.greater a' a) (Nx.greater (Nx.sub a' a) half) in
  let down = Nx.logical_and (Nx.greater a a') (Nx.greater (Nx.sub a a') half) in
  let i = Nx.where up (Nx.add_s !i 1L) (Nx.where down (Nx.sub_s !i 1L) !i) in
  Nx.where (Nx.equal a m) (Nx.full_like a (mask s')) i

(* Codes to codes through their magnitudes and signs, in uint64 and int64. *)
let recode (type a b d v r) (dt : (a, b) Nx.dtype) ~src:(signed, s)
    ~dst:(signed', s') (x : (v, r, d) Nx.t) : (a, b, d) Nx.t =
  let neg, a =
    if signed then
      let c = Nx.maximum_s (Nx.cast Nx.int64 x) (Int64.neg (mask s)) in
      (Some (Nx.less_s c 0L), Nx.cast Nx.uint64 (Nx.abs c))
    else (None, Nx.cast Nx.uint64 x)
  in
  let a = if s = s' then a else rescale ~s ~s' a in
  match neg with
  | None -> Nx.cast dt a
  | Some neg when signed' ->
      let c = Nx.cast Nx.int64 a in
      Nx.cast dt (Nx.where neg (Nx.neg c) c)
  | Some neg -> Nx.cast dt (Nx.where neg (Nx.zeros_like a) a)

let convert (type a b v r d) (dt : (a, b) Nx.dtype) (x : (v, r, d) Nx.t) :
    (a, b, d) Nx.t =
  let fn = "Mynd.convert" in
  let src = num fn (Nx.dtype x) and dst = num fn dt in
  match Nx.Dtype.equal_witness (Nx.dtype x) dt with
  | Some Equal -> x
  | None -> (
      match (src, dst, Nx.Dtype.kind (Nx.dtype x), Nx.Dtype.kind dt) with
      | Fraction, Fraction, _, _ -> Nx.cast dt x
      | Code { signed; s }, Fraction, _, Float -> of_code dt ~signed ~s x
      | Fraction, Code { signed; s }, Float, _ -> to_code dt ~signed ~s x
      | Code c, Code c', _, _ ->
          recode dt ~src:(c.signed, c.s) ~dst:(c'.signed, c'.s) x
      | _ -> assert false)

(* Transfer functions *)

(* CSS Color 4's sRGB constants: the knees of the linear segment in each
   direction, its slope, and the power curve's offset, scale and exponent. *)
let encoded_knee = 0.04045
let linear_knee = 0.0031308
let slope = 12.92
let offset = 0.055
let scale = 1.055
let gamma = 2.4

(* [odd x curve] is [sign x · curve |x|]: the curve reflected for negative
   values, as CSS extends it. *)
let odd x curve = Nx.where (Nx.less_s x 0.) (Nx.neg curve) curve

let srgb_to_linear x =
  let a = Nx.abs x in
  let near = Nx.less_equal_s a encoded_knee in
  (* [a] where the power is taken, so its derivative is finite elsewhere. *)
  let far = Nx.where near (Nx.ones_like a) a in
  let curve = Nx.pow_s (Nx.div_s (Nx.add_s far offset) scale) gamma in
  Nx.where near (Nx.div_s x slope) (odd x curve)

let linear_to_srgb x =
  let a = Nx.abs x in
  let near = Nx.less_equal_s a linear_knee in
  let far = Nx.where near (Nx.ones_like a) a in
  let curve = Nx.sub_s (Nx.mul_s (Nx.pow_s far (1. /. gamma)) scale) offset in
  Nx.where near (Nx.mul_s x slope) (odd x curve)

(* Alpha *)

let split fn x =
  if Nx.ndim x = 0 then invalid_argf "%s: a scalar has no channels" fn;
  let c = Nx.dim (-1) x in
  if c < 1 then invalid_argf "%s: no alpha channel" fn;
  match Nx.split_at ~axis:(-1) [ c - 1 ] x with
  | [ colour; alpha ] -> (colour, alpha)
  | _ -> assert false

let premultiply x =
  let colour, alpha = split "Mynd.premultiply" x in
  Nx.concatenate ~axis:(-1) [ Nx.mul colour alpha; alpha ]

let unpremultiply x =
  let colour, alpha = split "Mynd.unpremultiply" x in
  let zero = Nx.equal_s alpha 0. in
  let safe = Nx.where zero (Nx.ones_like alpha) alpha in
  let colour = Nx.div colour safe in
  Nx.concatenate ~axis:(-1)
    [ Nx.where zero (Nx.zeros_like colour) colour; alpha ]
