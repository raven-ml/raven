(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Image

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
  colours "Mynd.Colour.luma" x;
  Nx.unsqueeze ~axes:[ -1 ] (weigh luma_weights (rgb x))

let luminance x =
  colours "Mynd.Colour.luminance" x;
  Nx.unsqueeze ~axes:[ -1 ] (weigh luminance_weights (rgb x))

(* Each branch divides by its own channel less [m], so that at a tie the
   channel's whole tangent reaches the hue, as at its neighbours; a shared [v -
   m] would take half of it from [Nx.max]. Outside its branch, and at a grey, a
   denominator is [1], so that no [0 / 0] reaches the derivative. The wrap
   subtracts [1], which keeps the hue's derivative where [h] rounds to [1]. *)
let to_hsv x =
  colours "Mynd.Colour.to_hsv" x;
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
let of_hsv y =
  colours "Mynd.Colour.of_hsv" y;
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
  colours "Mynd.Colour.linear_to_lab" x;
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
  colours "Mynd.Colour.lab_to_linear" y;
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
