(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

(* Values *)

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

(* Shapes *)

(* [broadcast fn a b] is the broadcast of two batch shapes of one rank. *)
let broadcast fn a b =
  Array.map2
    (fun x y ->
      if x = y || y = 1 then x
      else if x = 1 then y
      else
        invalid_argf "%s: batch axes [%s] and [%s] do not broadcast" fn
          (String.concat "; " (Array.to_list (Array.map string_of_int a)))
          (String.concat "; " (Array.to_list (Array.map string_of_int b))))
    a b

let image fn x =
  if Nx.ndim x < 3 then
    invalid_argf "%s: an image has axes [...; h; w; c], not %d" fn (Nx.ndim x)

(* [points fn ~batch p] splits positions [[b…; q…; 2]] into their batch shape,
   their points' shape and the flat [[b…; q; 2]]. *)
let points fn ~batch p =
  let r = Nx.ndim p in
  if r < batch + 1 || Nx.dim (-1) p <> 2 then
    invalid_argf "%s: positions have axes [b...; q...; 2] with %d batch axes" fn
      batch;
  let s = Nx.shape p in
  let b = Array.sub s 0 batch and q = Array.sub s batch (r - batch - 1) in
  let n = Array.fold_left ( * ) 1 q in
  (b, q, Nx.reshape (Array.concat [ b; [| n; 2 |] ]) p)

(* [coordinate p k] is part [k] of positions or parameters along their last
   axis. *)
let coordinate p k =
  Nx.slice (List.init (Nx.ndim p - 1) (fun _ -> Nx.A) @ [ Nx.I k ]) p

(* Colour *)

let colours fn x =
  if Nx.ndim x < 1 || Nx.dim (-1) x <> 3 then
    invalid_argf "%s: colours have axes [...; 3], not [%s]" fn
      (String.concat "; "
         (Array.to_list (Array.map string_of_int (Nx.shape x))))

(* [rgb x] is the channels of the colours [x], each [[…]]. *)
let rgb x = (coordinate x 0, coordinate x 1, coordinate x 2)

(* [weigh (wr, wg, wb) (r, g, b)] is [wr r + wg g + wb b] for weights summing to
   [1], computed as [g + wr (r - g) + wb (b - g)], which a grey [(v, v, v)]
   gives exactly as [v]: both products are [0]. Where that form is not finite,
   as for [r = g = ∞] or [r - g] past the dtype's range, it is the sum. *)
let weigh (wr, wg, wb) (r, g, b) =
  let form =
    Nx.add (Nx.add g (Nx.mul_s (Nx.sub r g) wr)) (Nx.mul_s (Nx.sub b g) wb)
  in
  let sum = Nx.add (Nx.add (Nx.mul_s r wr) (Nx.mul_s g wg)) (Nx.mul_s b wb) in
  Nx.where (Nx.isfinite form) form sum

(* BT.601 §2.5.1's luma weights, and CSS Color 4's middle row of [M]. *)
let luma_weights = (0.299, 0.587, 0.114)

let luminance_weights =
  (87098. /. 409605., 175762. /. 245763., 12673. /. 175545.)

let luma x =
  colours "Mynd.luma" x;
  Nx.unsqueeze ~axes:[ -1 ] (weigh luma_weights (rgb x))

let luminance x =
  colours "Mynd.luminance" x;
  Nx.unsqueeze ~axes:[ -1 ] (weigh luminance_weights (rgb x))

(* Each branch divides by its own channel less [m], so that at a tie the
   channel's whole tangent reaches the hue, as at its neighbours; a shared [v -
   m] would take half of it from [Nx.max]. Outside its branch, and at a grey, a
   denominator is [1], so that no [0 / 0] reaches the derivative. The wrap
   subtracts [1], which keeps the hue's derivative where [h] rounds to [1]. *)
let rgb_to_hsv x =
  colours "Mynd.rgb_to_hsv" x;
  let r, g, b = rgb x in
  let v = Nx.max ~axes:[ -1 ] x and m = Nx.min ~axes:[ -1 ] x in
  let grey = Nx.equal v m in
  let on_r = Nx.equal r v in
  let on_g = Nx.logical_and (Nx.equal g v) (Nx.logical_not on_r) in
  let on_b = Nx.logical_not (Nx.logical_or on_r on_g) in
  let branch on c base num =
    let live = Nx.logical_and on (Nx.logical_not grey) in
    let den = Nx.where live (Nx.mul_s (Nx.sub c m) 6.) (Nx.ones_like c) in
    Nx.add_s (Nx.div num den) base
  in
  let h =
    Nx.where on_r
      (branch on_r r 0. (Nx.sub g b))
      (Nx.where on_g
         (branch on_g g (1. /. 3.) (Nx.sub b r))
         (branch on_b b (2. /. 3.) (Nx.sub r g)))
  in
  let h = Nx.where grey (Nx.zeros_like h) h in
  let h = Nx.where (Nx.less_s h 0.) (Nx.add_s h 1.) h in
  let h = Nx.where (Nx.greater_equal_s h 1.) (Nx.sub_s h 1.) h in
  let safe = Nx.where grey (Nx.ones_like v) v in
  let s = Nx.where grey (Nx.zeros_like v) (Nx.div (Nx.sub v m) safe) in
  (* An infinite channel leaves hue and saturation without a value, an infinite
     grey included. *)
  let finite = Nx.logical_and (Nx.isfinite v) (Nx.isfinite m) in
  let nan x = Nx.where finite x (Nx.full_like x Float.nan) in
  Nx.stack ~axis:(-1) [ nan h; nan s; v ]

(* CSS Color 4's continuous form, [c = clamp (min k (4 - k)) 0 1] with [k] the
   remainder of [n + 6h] by [6]. Each ramp is taken as a difference between [t =
   6h mod 6] and an integer: [k] is [t + s], [s] being [n], or [n - 6] where [t
   >= 6 - n], and [c = 1 + clamp (min (k - 1) (3 - k)) (-1) 0]. Near a kink each
   difference is exact (Sterbenz), so a kink is reached only where [t] is an
   integer; [n + 6h] would round at [n]'s magnitude onto one. [Nx.mod_] keeps
   the dividend's sign, so a negative [t] takes [6]. *)
let hsv_to_rgb y =
  colours "Mynd.hsv_to_rgb" y;
  let h, s, v = rgb y in
  let t = Nx.mod_s (Nx.mul_s h 6.) 6. in
  let t = Nx.where (Nx.less_s t 0.) (Nx.add_s t 6.) t in
  let channel n =
    let wrap = Nx.greater_equal_s t (6. -. n) in
    let shift a b = Nx.where wrap (Nx.full_like t b) (Nx.full_like t a) in
    let up = Nx.add t (shift (n -. 1.) (n -. 7.)) in
    let down = Nx.sub (shift (3. -. n) (9. -. n)) t in
    let c = Nx.add_s (Nx.clamp ~min:(-1.) ~max:0. (Nx.minimum up down)) 1. in
    Nx.sub v (Nx.mul (Nx.mul v s) c)
  in
  let out = Nx.stack ~axis:(-1) [ channel 5.; channel 3.; channel 1. ] in
  let finite =
    Nx.logical_and (Nx.isfinite h)
      (Nx.logical_and (Nx.isfinite s) (Nx.isfinite v))
  in
  Nx.where (Nx.unsqueeze ~axes:[ -1 ] finite) out (Nx.full_like out Float.nan)

(* CIE's exact constants: [f] meets its cube root at [ε = (6/29)³], where both
   pieces are [6/29] and both slopes [κ / 116]; [L*] is [κ ε = 8] there. *)
let epsilon = 216. /. 24389.
let kappa = 24389. /. 27.
let knee = 6. /. 29.

(* CSS Color 4's [M] and its inverse, in rationals, with [M]'s rows divided by
   D65's [X = 3127/3290] and [Z = 3583/3290] and the inverse's columns
   multiplied by them. Each row then sums to [1], so the rows take [weigh]'s
   form: a grey gives [X = Y = Z], white [1] and black [0], exactly. *)
let to_xyz =
  [|
    (1013504. /. 2335869., 878810. /. 2335869., 443555. /. 2335869.);
    luminance_weights;
    (15836. /. 892167., 878810. /. 8029503., 7008169. /. 8029503.);
  |]

let of_xyz =
  [|
    (121953. /. 39590., -329. /. 214., -10749. /. 19795.);
    (-8095803. /. 8788100., 1648619. /. 878810., 397713. /. 8788100.);
    (9381. /. 177422., -2585. /. 12673., 10749. /. 9338.);
  |]

let product rows c =
  Nx.stack ~axis:(-1) (Array.to_list (Array.map (fun w -> weigh w c) rows))

(* [cbrt t] where [t > ε], its argument made [1] elsewhere so that the branch
   not taken carries no infinite derivative. *)
let cube_root t =
  let above = Nx.greater_s t epsilon in
  (above, Nx.pow_s (Nx.where above t (Nx.ones_like t)) (1. /. 3.))

(* [g t] is [f t - 4/29]: [t^⅓ - 4/29] above [ε], and [κ t / 116] below, which
   is [f]'s linear piece without its [16 / 116]. [a*] and [b*] are differences
   of [g], equal to those of [f], so that near black, where both pieces are
   linear, no [16 / 116] is added and taken back, and a dark colour keeps its
   relative accuracy. *)
let lab_g t =
  let above, root = cube_root t in
  Nx.where above (Nx.sub_s root (4. /. 29.)) (Nx.mul_s t (kappa /. 116.))

(* [L*] by its pieces, [116 t^⅓ - 16] and [κ t], so that white is [100] and
   black [0] exactly. *)
let linear_to_lab x =
  colours "Mynd.linear_to_lab" x;
  let xyz = product to_xyz (rgb x) in
  let gx = lab_g (coordinate xyz 0) and gz = lab_g (coordinate xyz 2) in
  let y = coordinate xyz 1 in
  let gy = lab_g y in
  let above, root = cube_root y in
  let l =
    Nx.where above (Nx.sub_s (Nx.mul_s root 116.) 16.) (Nx.mul_s y kappa)
  in
  Nx.stack ~axis:(-1)
    [ l; Nx.mul_s (Nx.sub gx gy) 500.; Nx.mul_s (Nx.sub gy gz) 200. ]

(* [f]'s inverse at [u = (L + 16) / 116 + d]: [u³] above [6/29], else [(116 u -
   16) / κ], computed as [(L + 116 d) / κ], CSS Color 4's form, so that near
   black no [16] is added and taken back, black is [0] exactly, and a grey, [d =
   0], gives one value for [X], [Y] and [Z]. *)
let lab_to_linear y =
  colours "Mynd.lab_to_linear" y;
  let l, a, b = rgb y in
  let fy = Nx.div_s (Nx.add_s l 16.) 116. in
  let inverse d =
    let u = Nx.add fy d in
    let linear = Nx.div_s (Nx.add l (Nx.mul_s d 116.)) kappa in
    Nx.where (Nx.greater_s u knee) (Nx.mul (Nx.square u) u) linear
  in
  let x = inverse (Nx.div_s a 500.)
  and z = inverse (Nx.neg (Nx.div_s b 200.)) in
  let y = inverse (Nx.zeros_like l) in
  product of_xyz (x, y, z)

(* Kernels *)

type kernel = [ `Box | `Triangle | `Cubic of float | `Lanczos of int ]

let half_width = function
  | `Box -> 0.5
  | `Triangle -> 1.
  | `Cubic _ -> 2.
  | `Lanczos n -> Float.of_int n

let check_kernel fn = function
  | `Cubic a when not (Float.is_finite a) ->
      invalid_argf "%s: Keys' parameter %g is not finite" fn a
  | `Lanczos n when n < 1 -> invalid_argf "%s: Lanczos of %d lobes" fn n
  | `Box | `Triangle | `Cubic _ | `Lanczos _ -> ()

(* [eval k t] is [k] at the arguments [t], each in its support [[-h, h)], in
   [t]'s dtype: the formulas of mynd.mli, Keys' in powers of [|t|]. *)
let eval kernel t =
  match kernel with
  | `Box -> Nx.ones_like t
  | `Triangle -> Nx.rsub_s 1. (Nx.abs t)
  | `Cubic a ->
      let t = Nx.abs t in
      let t2 = Nx.square t in
      let t3 = Nx.mul t2 t in
      let near =
        Nx.add_s (Nx.sub (Nx.mul_s t3 (a +. 2.)) (Nx.mul_s t2 (a +. 3.))) 1.
      in
      let far =
        Nx.sub_s
          (Nx.add
             (Nx.sub (Nx.mul_s t3 a) (Nx.mul_s t2 (5. *. a)))
             (Nx.mul_s t (8. *. a)))
          (4. *. a)
      in
      Nx.where (Nx.less_equal_s t 1.) near far
  | `Lanczos n -> Nx.mul (Nx.sinc t) (Nx.sinc (Nx.div_s t (Float.of_int n)))

(* Sampling *)

type extension =
  [ `Zero | `Replicate | `Symmetric_whole | `Symmetric_half | `Periodic ]

type edge = [ extension | `Renormalise ]

(* The index of the pixel a tap at [t] reads, [-1] where it reads zero: under
   [`Renormalise] a tap outside reads zero, and weighs nothing. *)
let fold edge n t =
  match edge with
  | `Renormalise -> Nx.extension_index `Zero n t
  | #extension as e -> Nx.extension_index e n t

(* Positions past ±2⁶² read as if there: Zero and Replicate exactly, the folding
   rules from a far position. It keeps [⌊u⌋] and its taps in int64. *)
let far = 0x1p62

(* [below dt h] is [h] in [dt] and the largest value of [dt] below it, host
   floats so that they are constants under any trace. *)
let below dt h =
  let store = Nx.Dtype.of_float dt in
  let h = store h in
  let eps = (Nx.Dtype.float_format dt).epsilon in
  let rec refine lo =
    let mid = store ((lo +. h) /. 2.) in
    if mid > lo && mid < h then refine mid else lo
  in
  (h, refine (store (h -. (Float.abs h *. eps))))

(* [taps k u] is, along an axis, the [2h] taps reading coordinates [u]: each a
   pixel [t] in [(u - ½ - h, u - ½ + h]], not yet folded, and the kernel's
   argument [τ = u - ½ - t] there. They start at [⌊u⌋ + 1 - ⌊h + ½⌋], one less
   where [u]'s fraction lies below [h + ½]'s, [0] or [½].

   [τ] is the exact difference rounded once: [f - (t - ⌊u⌋ + ½)] with [f = u -
   ⌊u⌋], exact but for [u] in [(-1, 0)], where [u - (t + ½)] is, [t + ½] being a
   small half-integer. Rounding can take a [τ] just below [h] to [h], outside
   the support of the pixel it belongs to: it is held at the value below. It
   cannot take one to below [-h], which [u]'s dtype holds. *)
let taps kernel u =
  let h = half_width kernel in
  let m = Float.floor (h +. 0.5) in
  let fl = Nx.floor u in
  let f = Nx.sub u fl in
  let small = Nx.logical_and (Nx.less_s u 0.) (Nx.greater_s u (-1.)) in
  let base = Nx.cast Nx.int64 (Nx.clamp ~min:(-.far) ~max:far fl) in
  let r = h +. 0.5 -. m in
  let shift = if r = 0. then None else Some (Nx.less_s f r) in
  let h_dt, under = below (Nx.dtype u) h in
  List.init
    (Float.to_int (2. *. h))
    (fun d ->
      let delta = 1. -. m +. Float.of_int d in
      let t = Nx.add_s base (Int64.of_float delta) in
      let off = Nx.full_like f delta in
      let t, off =
        match shift with
        | None -> (t, off)
        | Some s ->
            (Nx.sub t (Nx.cast Nx.int64 s), Nx.sub off (Nx.cast (Nx.dtype f) s))
      in
      let half = Nx.add_s off 0.5 in
      let tau = Nx.where small (Nx.sub u (Nx.add fl half)) (Nx.sub f half) in
      let tau =
        Nx.where (Nx.greater_equal_s tau h_dt) (Nx.full_like tau under) tau
      in
      (t, tau))

(* [weights ~edge k n u] is the taps of an axis of [n] pixels at the coordinates
   [u]: each the index it reads and its weight, the kernel's values divided by
   their sum, added in order, and [0] where that sum is [0]. There the division
   is by [1], so that no [0 / 0] reaches the derivative. *)
let weights ~edge kernel n u =
  let taps =
    List.map
      (fun (t, tau) ->
        let i = fold edge n t in
        let k = eval kernel tau in
        match edge with
        | `Renormalise ->
            (i, Nx.where (Nx.greater_equal_s i 0L) k (Nx.zeros_like k))
        | #extension -> (i, k))
      (taps kernel u)
  in
  let sum =
    List.fold_left
      (fun s (_, k) -> Nx.add s k)
      (snd (List.hd taps))
      (List.tl taps)
  in
  let zero = Nx.equal_s sum 0. in
  let safe = Nx.where zero (Nx.ones_like sum) sum in
  List.map
    (fun (i, k) -> (i, Nx.where zero (Nx.zeros_like k) (Nx.div k safe)))
    taps

(* [flat ~w i j] is the flat index [i w + j] of a tap, [-1] where either reads
   zero, since [-1] in a flat index names the previous row's last pixel. *)
let flat ~w i j =
  let out = Nx.logical_or (Nx.less_s i 0L) (Nx.less_s j 0L) in
  Nx.where out (Nx.full_like i (-1L)) (Nx.add (Nx.mul_s i (Int64.of_int w)) j)

(* [gather x idx] reads the images [x], [[b…; h w; c]], at the flat indices
   [idx], [[b…; q]]: [[b…; q; c]]. *)
let gather x idx =
  let batch = Nx.ndim x - 2 in
  Nx.take_along_axis ~axis:batch (Nx.unsqueeze ~axes:[ -1 ] idx) x

let check_edge fn ~edge x =
  let h = Nx.dim (-3) x and w = Nx.dim (-2) x in
  match edge with
  | `Zero | `Renormalise -> ()
  | `Replicate | `Symmetric_whole | `Symmetric_half | `Periodic ->
      if h = 0 || w = 0 then
        invalid_argf
          "%s: an image of %d by %d pixels reads nothing past its edge" fn h w

(* [prepare fn x p] is the batch shape, the points' shape, [x] with its pixels
   flat and the coordinates [u] and [v], [[b…; q]]. *)
let prepare fn ~edge x p =
  image fn x;
  check_edge fn ~edge x;
  let batch = Nx.ndim x - 3 in
  let pb, q, p = points fn ~batch p in
  let s = Nx.shape x in
  let xb = Array.sub s 0 batch in
  let b = broadcast fn xb pb in
  let h = s.(batch) and w = s.(batch + 1) and c = s.(batch + 2) in
  let xf = Nx.reshape (Array.concat [ xb; [| h * w; c |] ]) x in
  (b, q, (h, w, c), xf, coordinate p 0, coordinate p 1)

let finish ~b ~q ~c y = Nx.reshape (Array.concat [ b; q; [| c |] ]) y

let sample ~edge kernel x p =
  let fn = "Mynd.sample" in
  check_kernel fn kernel;
  let edge = (edge :> edge) in
  let b, q, (h, w, c), xf, u, v = prepare fn ~edge x p in
  let rows = weights ~edge kernel h v and cols = weights ~edge kernel w u in
  let sum = ref None in
  List.iter
    (fun (i, wi) ->
      List.iter
        (fun (j, wj) ->
          let weight = Nx.unsqueeze ~axes:[ -1 ] (Nx.mul wj wi) in
          let term = Nx.mul weight (gather xf (flat ~w i j)) in
          sum := Some (match !sum with None -> term | Some s -> Nx.add s term))
        cols)
    rows;
  let y = Option.get !sum in
  let finite =
    Nx.unsqueeze ~axes:[ -1 ] (Nx.logical_and (Nx.isfinite u) (Nx.isfinite v))
  in
  let y = Nx.where finite y (Nx.full_like y Float.nan) in
  finish ~b ~q ~c y

let nearest ~edge x p =
  let fn = "Mynd.nearest" in
  let edge = (edge :> extension) in
  let b, q, (h, w, c), xf, u, v = prepare fn ~edge:(edge :> edge) x p in
  let index n u =
    let i = Nx.cast Nx.int64 (Nx.clamp ~min:(-.far) ~max:far (Nx.floor u)) in
    let i = Nx.extension_index edge n i in
    Nx.where (Nx.isfinite u) i (Nx.full_like i (-1L))
  in
  finish ~b ~q ~c (gather xf (flat ~w (index h v) (index w u)))

let centres dt (h, w) =
  if h < 0 || w < 0 then invalid_argf "Mynd.centres: %d by %d pixels" h w;
  let half n = Nx.add_s (Nx.arange_f Nx.float64 0. (Float.of_int n) 1.) 0.5 in
  let x = Nx.broadcast_to [| h; w |] (Nx.reshape [| 1; w |] (half w)) in
  let y = Nx.broadcast_to [| h; w |] (Nx.reshape [| h; 1 |] (half h)) in
  Nx.cast dt (Nx.stack ~axis:(-1) [ x; y ])

(* Transforms *)

module Transform = struct
  let entry m i j =
    Nx.slice (List.init (Nx.ndim m - 2) (fun _ -> Nx.A) @ [ Nx.I i; Nx.I j ]) m

  let apply m p =
    let fn = "Mynd.Transform.apply" in
    if Nx.ndim m < 2 || Nx.dim (-1) m <> 3 || Nx.dim (-2) m <> 3 then
      invalid_argf "%s: a transform has axes [b...; 3; 3]" fn;
    let batch = Nx.ndim m - 2 in
    let pb, q, p = points fn ~batch p in
    let b = broadcast fn (Array.sub (Nx.shape m) 0 batch) pb in
    let e i j = Nx.unsqueeze ~axes:[ -1 ] (entry m i j) in
    let x = coordinate p 0 and y = coordinate p 1 in
    let row i = Nx.add (Nx.add (Nx.mul (e i 0) x) (Nx.mul (e i 1) y)) (e i 2) in
    let d = row 2 in
    let r = Nx.stack ~axis:(-1) [ Nx.div (row 0) d; Nx.div (row 1) d ] in
    Nx.reshape (Array.concat [ b; q; [| 2 |] ]) r

  let matrix rows =
    Nx.stack ~axis:(-2) (List.map (fun r -> Nx.stack ~axis:(-1) r) rows)

  let pair fn t =
    if Nx.ndim t < 1 || Nx.dim (-1) t <> 2 then
      invalid_argf "%s: parameters have axes [...; 2]" fn;
    (coordinate t 0, coordinate t 1)

  let translation t =
    let tx, ty = pair "Mynd.Transform.translation" t in
    let o = Nx.ones_like tx and z = Nx.zeros_like tx in
    matrix [ [ o; z; tx ]; [ z; o; ty ]; [ z; z; o ] ]

  let scaling s =
    let sx, sy = pair "Mynd.Transform.scaling" s in
    let o = Nx.ones_like sx and z = Nx.zeros_like sx in
    matrix [ [ sx; z; z ]; [ z; sy; z ]; [ z; z; o ] ]

  let rotation a =
    let c = Nx.cos a and s = Nx.sin a in
    let o = Nx.ones_like a and z = Nx.zeros_like a in
    matrix [ [ c; Nx.neg s; z ]; [ s; c; z ]; [ z; z; o ] ]

  let shear s =
    let sx, sy = pair "Mynd.Transform.shear" s in
    let o = Nx.ones_like sx and z = Nx.zeros_like sx in
    matrix [ [ o; sx; z ]; [ sy; o; z ]; [ z; z; o ] ]
end

(* Resizing *)

(* [fdiv a b] is [⌊a / b⌋] for [b > 0]. *)
let fdiv a b = if a >= 0 then a / b else -((b - 1 - a) / b)

(* [spans ~edge k n n'] is each new pixel's taps, in increasing [i], with their
   arguments. New pixel [j]'s [τ = (c - i - ½) / s] is [d / e] for the integers
   [d = (2j + 1) n - (2i + 1) n'] and [e], [2n] when shrinking and [2n']
   otherwise, so one division rounds it. [h e] is an integer, [2h] being whole
   and [e] even, so the support test [-h e <= d < h e] is exact. Under
   [`Renormalise] the taps outside are dropped here. *)
let spans ~edge kernel n n' =
  let h = half_width kernel in
  let e = if n > n' then 2 * n else 2 * n' in
  let he = Float.to_int (h *. Float.of_int e) in
  Array.init n' (fun j ->
      let num = ((2 * j) + 1) * n in
      let lo = fdiv (num - he - n') (2 * n') + 1 in
      let hi = fdiv (num + he - n') (2 * n') in
      let lo, hi =
        match edge with
        | `Renormalise -> (Int.max lo 0, Int.min hi (n - 1))
        | #extension -> (lo, hi)
      in
      Array.init
        (Int.max 0 (hi - lo + 1))
        (fun k ->
          let i = lo + k in
          let d = num - (((2 * i) + 1) * n') in
          (i, Float.of_int d /. Float.of_int e)))

(* [sinc x] at float64 on the host, for weights: [sin (πr) / (πx)] signed by the
   parity of [x]'s nearest integer [k], [r = x - k] exact, so it is [0] at every
   other integer. A weight needs no derivative, so the quotient serves near [0]
   too. *)
let sinc x =
  if x = 0. then 1.
  else
    let k = Float.round x in
    let q = Float.sin (Float.pi *. (x -. k)) /. (Float.pi *. x) in
    if Float.rem k 2. = 0. then q else -.q

(* [value k t] is [k] at [t] in its support, at float64 on the host, by the
   formulas [eval] stages. *)
let value kernel t =
  match kernel with
  | `Box -> 1.
  | `Triangle -> 1. -. Float.abs t
  | `Cubic a ->
      let t = Float.abs t in
      let t2 = t *. t in
      let t3 = t2 *. t in
      if t <= 1. then (t3 *. (a +. 2.)) -. (t2 *. (a +. 3.)) +. 1.
      else (t3 *. a) -. (t2 *. (5. *. a)) +. (t *. (8. *. a)) -. (4. *. a)
  | `Lanczos n -> sinc t *. sinc (t /. Float.of_int n)

(* [table fn ~edge k n n'] is each new pixel's taps that read a pixel with a
   nonzero weight, unfolded, and their weights: the kernel's float64 values
   divided by the row's float64 sum, added in order. A tap outside reads zero
   under [`Zero], and [`Renormalise] dropped it before the sum. A row whose sum
   is [0] has no tap. OCaml floats and integers alone, so that a trace stages
   none of it. *)
let table fn ~edge kernel n n' =
  check_kernel fn kernel;
  if n < 1 || n' < 1 then invalid_argf "%s: %d pixels to %d" fn n n';
  let reads i =
    match edge with
    | `Zero -> i >= 0 && i < n
    | `Renormalise | `Replicate | `Symmetric_whole | `Symmetric_half | `Periodic
      ->
        true
  in
  Array.map
    (fun span ->
      let values = Array.map (fun (_, t) -> value kernel t) span in
      let sum = Array.fold_left ( +. ) 0. values in
      if sum = 0. then [||]
      else
        let taps = Array.mapi (fun k (i, _) -> (i, values.(k) /. sum)) span in
        Array.of_list
          (List.filter (fun (i, w) -> reads i && w <> 0.) (Array.to_list taps)))
    (spans ~edge kernel n n')

(* [indices ~edge n shape is] is the pixels the unfolded positions [is] read,
   folded by {!Nx.extension_index}, and [-1] for [None], the padding. *)
let indices ~edge n shape is =
  let raw f = Nx.create Nx.int64 shape (Array.map f is) in
  match edge with
  | `Zero | `Renormalise ->
      raw (function Some i -> Int64.of_int i | None -> -1L)
  | #extension as e ->
      let pad = Nx.create Nx.bool shape (Array.map Option.is_none is) in
      let i = raw (function Some i -> Int64.of_int i | None -> 0L) in
      Nx.where pad (Nx.full_like i (-1L)) (Nx.extension_index e n i)

(* [slot rows k] is slot [k] of each row, [None] past its taps. *)
let slot rows k =
  Array.map (fun r -> if k < Array.length r then Some r.(k) else None) rows

let resize_weights ~edge kernel n n' =
  let edge = (edge :> edge) in
  let rows = table "Mynd.resize_weights" ~edge kernel n n' in
  let t = Array.fold_left (fun t r -> Int.max t (Array.length r)) 0 rows in
  let taps = Array.concat (List.init t (slot rows)) in
  (* [taps] is slot-major: transposed into rows. *)
  let rowwise a = Nx.copy (Nx.transpose (Nx.reshape [| t; n' |] a)) in
  ( rowwise (indices ~edge n [| t * n' |] (Array.map (Option.map fst) taps)),
    rowwise
      (Nx.create Nx.float64
         [| t * n' |]
         (Array.map (function Some (_, w) -> w | None -> 0.) taps)) )

(* [along fn ~edge k axis n' x] resizes axis [axis] of [x] to [n'] pixels: for
   each slot of the table, one gather of its pixels times its weights, rounded
   once into [x]'s dtype, the slots added in order. A padding slot reads zero
   with weight zero. A table of no slot has one of padding. *)
let along fn ~edge kernel axis n' x =
  let n = Nx.dim axis x in
  if n = n' then x
  else
    let rows = table fn ~edge kernel n n' in
    let t = Array.fold_left (fun t r -> Int.max t (Array.length r)) 1 rows in
    let at = Nx.beside x in
    let shape = Array.init (-axis) (fun k -> if k = 0 then n' else 1) in
    let term k =
      let taps = slot rows k in
      let i = indices ~edge n [| n' |] (Array.map (Option.map fst) taps) in
      (* Stored straight into [x]'s dtype, which rounds each float64 weight
         once, as a cast would, and stages no operation. *)
      let w =
        Array.map (function Some (_, w) -> w | None -> 0.) taps
        |> Nx.create (Nx.dtype x) shape
      in
      Nx.mul (Nx.take ~axis (Nx.place at i) x) (Nx.place at w)
    in
    let sum = ref (term 0) in
    for k = 1 to t - 1 do
      sum := Nx.add !sum (term k)
    done;
    !sum

let resize ~edge kernel (h', w') x =
  let fn = "Mynd.resize" in
  image fn x;
  if h' < 1 || w' < 1 then invalid_argf "%s: %d by %d pixels" fn h' w';
  if Nx.dim (-3) x < 1 || Nx.dim (-2) x < 1 then
    invalid_argf "%s: an image of %d by %d pixels" fn (Nx.dim (-3) x)
      (Nx.dim (-2) x);
  check_kernel fn kernel;
  let edge = (edge :> edge) in
  let go x =
    x |> along fn ~edge kernel (-2) w' |> along fn ~edge kernel (-3) h'
  in
  (* The formats narrower than 32 bits compute at float32 and round once. *)
  let dt = Nx.dtype x in
  if Nx.Dtype.bits dt < 32 then Nx.cast dt (go (Nx.cast Nx.float32 x)) else go x

(* Masks are boolean images of one channel. *)
let mask_shape fn m =
  if Nx.ndim m < 3 || Nx.dim (-1) m <> 1 then
    invalid_argf "%s: a mask has axes [...; h; w; 1]" fn

(* Boxes *)

module Box = struct
  let corners fn b =
    if Nx.ndim b < 1 || Nx.dim (-1) b <> 4 then
      invalid_argf "%s: boxes have axes [...; 4]" fn;
    (coordinate b 0, coordinate b 1, coordinate b 2, coordinate b 3)

  let pack a b c d = Nx.stack ~axis:(-1) [ a; b; c; d ]

  let of_xywh b =
    let x, y, w, h = corners "Mynd.Box.of_xywh" b in
    pack x y (Nx.add x w) (Nx.add y h)

  let to_xywh b =
    let x0, y0, x1, y1 = corners "Mynd.Box.to_xywh" b in
    pack x0 y0 (Nx.sub x1 x0) (Nx.sub y1 y0)

  let of_cxcywh b =
    let cx, cy, w, h = corners "Mynd.Box.of_cxcywh" b in
    let hw = Nx.mul_s w 0.5 and hh = Nx.mul_s h 0.5 in
    pack (Nx.sub cx hw) (Nx.sub cy hh) (Nx.add cx hw) (Nx.add cy hh)

  let to_cxcywh b =
    let x0, y0, x1, y1 = corners "Mynd.Box.to_cxcywh" b in
    pack
      (Nx.mul_s (Nx.add x0 x1) 0.5)
      (Nx.mul_s (Nx.add y0 y1) 0.5)
      (Nx.sub x1 x0) (Nx.sub y1 y0)

  (* An empty box, [x₁ <= x₀] or [y₁ <= y₀], has area [0]: a difference's sign
     is exact. *)
  let area_of (x0, y0, x1, y1) =
    let w = Nx.sub x1 x0 and h = Nx.sub y1 y0 in
    let empty = Nx.logical_or (Nx.less_equal_s w 0.) (Nx.less_equal_s h 0.) in
    Nx.where empty (Nx.zeros_like w) (Nx.mul w h)

  let area b = area_of (corners "Mynd.Box.area" b)

  (* torchvision's arithmetic: the overlap's sides clamped at [0], their
     product, and [(area a + area b) - i]. *)
  let overlap a b =
    let ((ax0, ay0, ax1, ay1) as ca) = a and ((bx0, by0, bx1, by1) as cb) = b in
    let row x = Nx.unsqueeze ~axes:[ -1 ] x
    and col x = Nx.unsqueeze ~axes:[ -2 ] x in
    let side lo lo' hi hi' =
      Nx.maximum_s
        (Nx.sub (Nx.minimum (row hi) (col hi')) (Nx.maximum (row lo) (col lo')))
        0.
    in
    let i = Nx.mul (side ax0 bx0 ax1 bx1) (side ay0 by0 ay1 by1) in
    let union = Nx.sub (Nx.add (row (area_of ca)) (col (area_of cb))) i in
    let empty = Nx.equal_s union 0. in
    let safe = Nx.where empty (Nx.ones_like union) union in
    Nx.where empty (Nx.zeros_like i) (Nx.div i safe)

  let iou a b =
    let fn = "Mynd.Box.iou" in
    if Nx.ndim a < 2 || Nx.ndim b < 2 then
      invalid_argf "%s: boxes have axes [...; n; 4]" fn;
    overlap (corners fn a) (corners fn b)

  let of_mask dt m =
    let fn = "Mynd.Box.of_mask" in
    mask_shape fn m;
    let h = Nx.dim (-3) m and w = Nx.dim (-2) m in
    if h = 0 || w = 0 then
      (* No pixel: [any] of nothing is [false], [0] once cast. *)
      let none = Nx.cast dt (Nx.any ~axes:[ -3; -2; -1 ] ~keepdims:true m) in
      let s = Nx.shape m in
      let lead = Array.sub s 0 (Array.length s - 3) in
      Nx.copy
        (Nx.broadcast_to
           (Array.append lead [| 4 |])
           (Nx.reshape (Array.append lead [| 1 |]) none))
    else
      (* [first a n] is the first index where [a] holds along its last axis;
         [last] one past the last. *)
      let first a = Nx.argmax ~axis:(-1) (Nx.cast Nx.uint8 a) in
      let past a n =
        Nx.rsub_s (Int64.of_int n) (first (Nx.flip ~axes:[ -1 ] a))
      in
      let rows = Nx.any ~axes:[ -2; -1 ] m
      and cols = Nx.any ~axes:[ -3; -1 ] m in
      let b =
        Nx.stack ~axis:(-1) [ first cols; first rows; past cols w; past rows h ]
      in
      let some = Nx.unsqueeze ~axes:[ -1 ] (Nx.any ~axes:[ -1 ] cols) in
      Nx.cast dt (Nx.where some b (Nx.zeros_like b))
end

(* Masks *)

module Mask = struct
  let iou dt a b =
    let fn = "Mynd.Mask.iou" in
    mask_shape fn a;
    mask_shape fn b;
    if Nx.ndim a < 4 || Nx.ndim b < 4 then
      invalid_argf "%s: masks have axes [...; n; h; w; 1]" fn;
    let flat m =
      let s = Nx.shape m in
      let r = Array.length s in
      Nx.reshape
        (Array.append (Array.sub s 0 (r - 3)) [| s.(r - 3) * s.(r - 2) |])
        (Nx.cast Nx.int64 m)
    in
    let a = flat a and b = flat b in
    let both = Nx.matmul a (Nx.swapaxes (-1) (-2) b) in
    let count m = Nx.sum ~axes:[ -1 ] m in
    let either =
      Nx.sub
        (Nx.add
           (Nx.unsqueeze ~axes:[ -1 ] (count a))
           (Nx.unsqueeze ~axes:[ -2 ] (count b)))
        both
    in
    let empty = Nx.equal_s either 0L in
    let safe = Nx.where empty (Nx.ones_like either) either in
    (* The exact counts' quotient, rounded once into [dt]. float32 rounds it
       once for counts below 2²⁴. Through float64, a quotient of counts below 2ᵏ
       lies at least 2⁻ᵏ⁻ᵖ⁻¹ relatively from a midpoint of a float of [p] bits,
       beyond float64's rounding while [k + p + 1 < 53]: below 2²⁸ pixels for
       every float. *)
    let quotient (type w) (wide : (float, w) Nx.dtype) =
      Nx.cast dt (Nx.div (Nx.cast wide both) (Nx.cast wide safe))
    in
    let pixels = Nx.dim (-1) a in
    let r =
      if Nx.Dtype.equal dt Nx.float32 && pixels < 1 lsl 24 then
        quotient Nx.float32
      else quotient Nx.float64
    in
    Nx.where empty (Nx.zeros_like r) r
end

(* Run-length encoding *)

module Rle = struct
  type t = { height : int; width : int; counts : int array }

  let make fn ~height ~width counts =
    if height < 0 || width < 0 then
      invalid_argf "%s: %d by %d pixels" fn height width;
    let sum =
      Array.fold_left
        (fun s c ->
          if c < 0 then invalid_argf "%s: a negative count, %d" fn c;
          s + c)
        0 counts
    in
    if sum <> height * width then
      invalid_argf "%s: counts sum to %d, not %d by %d" fn sum height width;
    { height; width; counts }

  let v ~height ~width counts = make "Mynd.Rle.v" ~height ~width counts

  (* Runs in column-major order, the first of [false] pixels. *)
  let encode m =
    let fn = "Mynd.Rle.encode" in
    if Nx.ndim m <> 3 || Nx.dim (-1) m <> 1 then
      invalid_argf "%s: a mask has axes [h; w; 1]" fn;
    let height = Nx.dim 0 m and width = Nx.dim 1 m in
    let px =
      Nx.to_array
        (Nx.reshape
           [| height * width |]
           (Nx.copy (Nx.transpose ~axes:[ 1; 0; 2 ] m)))
    in
    let runs = ref [] and run = ref 0 and cur = ref false in
    Array.iter
      (fun p ->
        if p = !cur then incr run
        else (
          runs := !run :: !runs;
          run := 1;
          cur := p))
      px;
    { height; width; counts = Array.of_list (List.rev (!run :: !runs)) }

  let decode r =
    let px = Array.make (r.height * r.width) false in
    let at = ref 0 in
    Array.iteri
      (fun i c ->
        if i land 1 = 1 then Array.fill px !at c true;
        at := !at + c)
      r.counts;
    Nx.transpose ~axes:[ 1; 0; 2 ]
      (Nx.create Nx.bool [| r.width; r.height; 1 |] px)
    |> Nx.copy

  let area r =
    let a = ref 0 in
    Array.iteri (fun i c -> if i land 1 = 1 then a := !a + c) r.counts;
    !a

  (* pycocotools' rleToString: each count after the third less the count two
     before, then five bits a character from the low end, [0x20] marking one
     more, [0x10] the sign of the last, plus 48. *)
  let to_string r =
    let b = Buffer.create (Array.length r.counts * 2) in
    Array.iteri
      (fun i c ->
        let x = ref (if i > 2 then c - r.counts.(i - 2) else c) in
        let more = ref true in
        while !more do
          let c = !x land 0x1f in
          x := !x asr 5;
          more := if c land 0x10 <> 0 then !x <> -1 else !x <> 0;
          Buffer.add_char b (Char.chr (48 + if !more then c lor 0x20 else c))
        done)
      r.counts;
    Buffer.contents b

  (* A count of more than 12 characters passes 60 bits. *)
  let max_chars = 12

  let of_string ~height ~width s =
    let fn = "Mynd.Rle.of_string" in
    let n = String.length s in
    let counts = ref [] and m = ref 0 and p = ref 0 in
    while !p < n do
      let x = ref 0 and k = ref 0 and more = ref true in
      while !more do
        if !p >= n then invalid_argf "%s: %S ends inside a count" fn s;
        if !k >= max_chars then invalid_argf "%s: a count of %S overflows" fn s;
        let c = Char.code s.[!p] - 48 in
        if c < 0 || c > 63 then
          invalid_argf "%s: %C is not a count's character" fn s.[!p];
        x := !x lor ((c land 0x1f) lsl (5 * !k));
        more := c land 0x20 <> 0;
        incr p;
        incr k;
        if (not !more) && c land 0x10 <> 0 then x := !x lor (-1 lsl (5 * !k))
      done;
      let x = if !m > 2 then !x + List.nth !counts 1 else !x in
      counts := x :: !counts;
      incr m
    done;
    make fn ~height ~width (Array.of_list (List.rev !counts))
end

(* Components *)

(* Liu and Tarjan's algorithm R with two shortcuts a round over the pixel graph.
   Every pixel starts as its own root. A round reads the parents [p] once: for
   each edge [{v, w}] between true pixels whose parent [p v] is a root and [p w
   < p v], the root takes the least such [p w] as its parent
   (parent-root-connect, a scatter of [Min]); then each pixel's parent becomes
   its grandparent, twice. The rounds end when none changes a parent, every tree
   then flat and rooted at its least pixel. Their §4.3: with two shortcuts Lemma
   23 holds for [b = 2^(1/3)], so no tree is active once [b^(k - 5) > 3n / 2],
   at most [6 + ⌊3 log₂ (3n / 2)⌋] rounds with the last, which changes
   nothing. *)

let neighbours = function
  | `Four -> [ (-1, 0); (0, -1); (0, 1); (1, 0) ]
  | `Eight ->
      [ (-1, -1); (-1, 0); (-1, 1); (0, -1); (0, 1); (1, -1); (1, 0); (1, 1) ]

let component_rounds n =
  6
  + Float.to_int
      (Float.ceil (3. *. Float.log2 (1.5 *. Float.of_int (Int.max 1 n))))

let label conn m =
  let fn = "Mynd.components" in
  mask_shape fn m;
  let s = Nx.shape m in
  let r = Array.length s in
  let b = Array.sub s 0 (r - 3) and h = s.(r - 3) and w = s.(r - 2) in
  let n = h * w in
  let fg = Nx.reshape (Array.append b [| h; w |]) m in
  let flat x = Nx.reshape (Array.append b [| n |]) x in
  let grid x = Nx.reshape (Array.append b [| h; w |]) x in
  let none = Int64.max_int in
  let pads =
    Array.append (Array.map (fun _ -> (0, 0)) b) [| (1, 1); (1, 1) |]
  in
  let offsets = neighbours conn in
  let at_offset q (dy, dx) =
    Nx.slice
      (List.init (Array.length b) (fun _ -> Nx.A)
      @ [ Nx.R (1 + dy, 1 + dy + h); Nx.R (1 + dx, 1 + dx + w) ])
      q
  in
  let step (p, (_, k)) =
    (* The least parent of a true neighbour, [none] where there is none. *)
    let q = Nx.where fg (grid p) (Nx.full_like (grid p) none) in
    let q = Nx.pad pads none q in
    let least =
      List.fold_left
        (fun acc o -> Nx.minimum acc (at_offset q o))
        (at_offset q (List.hd offsets))
        (List.tl offsets)
    in
    let least = flat (Nx.where fg least (Nx.full_like least none)) in
    let root = Nx.equal (Nx.take_along_axis ~axis:(-1) p p) p in
    let hook = Nx.logical_and root (Nx.less least p) in
    let target = Nx.where hook p (Nx.full_like p (-1L)) in
    let p' = Nx.scatter ~combine:Nx.Min ~axis:(-1) target least p in
    let shortcut p = Nx.take_along_axis ~axis:(-1) p p in
    let p' = shortcut (shortcut p') in
    (p', (Nx.any (Nx.not_equal p' p), Nx.add_s k 1l))
  in
  let at = Nx.placement m in
  let start =
    Nx.copy
      (Nx.broadcast_to (Array.append b [| n |])
         (Nx.arange_at at Nx.int64 0 n 1))
  in
  let carry = Nx.Ptree.(pair tensor (pair tensor tensor)) in
  let p, (_, rounds) =
    Rune.iterate carry ~max:(component_rounds n)
      ~until:(fun (_, (changed, _)) -> Nx.logical_not changed)
      step
      (start, (Nx.scalar_at at Nx.bool true, Nx.scalar_at at Nx.int32 0l))
  in
  let labels = Nx.where (flat fg) p (Nx.full_like p (-1L)) in
  (Nx.reshape s labels, rounds)

let components conn m = fst (label conn m)
let components_rounds = label

(* Non-maximum suppression *)

(* [above x t] is [x > t] for the real [t]: with [c] the value of [x]'s dtype
   nearest [t], [x > t] is [x >= c] where [c > t], since no value lies in [(t,
   c)], and [x > c] otherwise, since none lies in [(c, t]]. *)
let above x t =
  let c = Nx.Dtype.of_float (Nx.dtype x) t in
  if c > t then Nx.greater_equal_s x c else Nx.greater_s x c

(* The keep mask is the fixed point of [keep i = ¬∃ j earlier than i: keep j ∧
   conflict i j] from all true. A box ranked [r] is final after round [r + 1],
   so [n + 1] rounds suffice, the last changing nothing; a box Blelloch, Fineman
   and Shun's parallel greedy step decides in its step [s] is final by round
   [2s]. *)
let suppress ?classes ~iou b s =
  let fn = "Mynd.nms" in
  if Nx.ndim b < 2 || Nx.dim (-1) b <> 4 then
    invalid_argf "%s: boxes have axes [...; n; 4]" fn;
  let lead = Array.sub (Nx.shape b) 0 (Nx.ndim b - 1) in
  let same name x =
    if Nx.shape x <> lead then
      invalid_argf "%s: %s's axes differ from the boxes' [...; n]" fn name
  in
  same "the scores" s;
  Option.iter (same "the classes") classes;
  let n = lead.(Array.length lead - 1) in
  (* The sort orders [0.] before [-0.]; adding [0.] makes every zero [0.], so
     equal scores tie by position. *)
  let order = Nx.argsort ~descending:true ~axis:(-1) (Nx.add_s s 0.) in
  let rank = Nx.argsort ~axis:(-1) order in
  let earlier =
    Nx.less (Nx.unsqueeze ~axes:[ -2 ] rank) (Nx.unsqueeze ~axes:[ -1 ] rank)
  in
  let c = Box.corners fn b in
  let conflict = Nx.logical_and earlier (above (Box.overlap c c) iou) in
  let conflict =
    match classes with
    | None -> conflict
    | Some k ->
        Nx.logical_and conflict
          (Nx.equal (Nx.unsqueeze ~axes:[ -1 ] k) (Nx.unsqueeze ~axes:[ -2 ] k))
  in
  let step (keep, (_, k)) =
    let hit = Nx.logical_and conflict (Nx.unsqueeze ~axes:[ -2 ] keep) in
    let keep' = Nx.logical_not (Nx.any ~axes:[ -1 ] hit) in
    (keep', (Nx.any (Nx.not_equal keep' keep), Nx.add_s k 1l))
  in
  let at = Nx.placement b in
  let carry = Nx.Ptree.(pair tensor (pair tensor tensor)) in
  let keep, (_, rounds) =
    Rune.iterate carry ~max:(n + 1)
      ~until:(fun (_, (changed, _)) -> Nx.logical_not changed)
      step
      ( Nx.copy (Nx.broadcast_to lead (Nx.scalar_at at Nx.bool true)),
        (Nx.scalar_at at Nx.bool true, Nx.scalar_at at Nx.int32 0l) )
  in
  (keep, rounds)

let nms ?classes ~iou b s = fst (suppress ?classes ~iou b s)
let nms_rounds = suppress
