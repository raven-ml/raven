(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Quantity = Ymir_units.Quantity
module Unit = Ymir_units.Unit

let strf = Printf.sprintf

type 'd vectors = (float, Nx.float64_elt, 'd) Nx.t
type ('f, 'd) t = { frame : 'f Frame.t; xyz : 'd vectors }

let frame d = d.frame
let xyz d = d.xyz
let radians x = Quantity.v Unit.radian x
let two_pi = 2. *. Float.pi

(* Rows *)

(* [at i] places a message at the batch index [i]: [" at [3; 1]"], or nothing
   for a single vector. *)
let at i =
  if Array.length i = 0 then ""
  else
    strf " at [%s]"
      (String.concat "; " (Array.to_list (Array.map string_of_int i)))

let last v = Nx.ndim v - 1
let component v k = Nx.slice (List.init (last v) (fun _ -> Nx.A) @ [ Nx.I k ]) v
let components v = (component v 0, component v 1, component v 2)
let sub3 (ax, ay, az) (bx, by, bz) = Nx.(sub ax bx, sub ay by, sub az bz)

let dot (ax, ay, az) (bx, by, bz) =
  Nx.(add (add (mul ax bx) (mul ay by)) (mul az bz))

let norm2 v = dot v v

let cross (ax, ay, az) (bx, by, bz) =
  Nx.
    ( sub (mul ay bz) (mul az by),
      sub (mul az bx) (mul ax bz),
      sub (mul ax by) (mul ay bx) )

let is_zero (x, y, z) =
  Nx.(logical_and (logical_and (equal_s x 0.) (equal_s y 0.)) (equal_s z 0.))

let check_finite what v =
  let finite = Nx.logical_not (Nx.any ~axes:[ last v ] (Nx.isinf v)) in
  Nx.check Nx.Ptree.unit finite () (fun i () ->
      Invalid_argument (strf "%s%s is infinite" what (at i)))

let sign_bit_off = 0x7FFF_FFFF_FFFF_FFFFL
let mantissa_bits = 52
let exponent_all_ones = 2047L
let least_normal = 0x1p-1022
let subnormal_lift = 0x1p54

(* [scale v] is [v] with each row times the power of two that brings its largest
   component into [2, 4). No one power of two reaches from the least subnormal,
   2^-1074, to 2, so a row whose largest component is subnormal is first lifted
   by 2^54, which makes each of its nonzero components normal. Both factors are
   read from the bits, so the scaling is exact and carries no derivative: each
   reader is homogeneous of degree 0 in a row, and holding the scale constant
   gives its exact gradient. Squares and cross products of a scaled row stay in
   range for every finite row, a zero row stays zero, and a row of NaN or
   infinities scales by 0. *)
let scale v =
  let magnitude =
    let bits = Nx.bitcast Nx.int64 v in
    Nx.bitwise_and bits (Nx.scalar_like bits sign_bit_off)
  in
  let largest =
    Nx.bitcast Nx.float64 (Nx.max ~axes:[ last v ] ~keepdims:true magnitude)
  in
  let lift =
    Nx.where
      (Nx.less_s largest least_normal)
      (Nx.full_like largest subnormal_lift)
      (Nx.full_like largest 1.)
  in
  let exponent =
    Nx.rshift (Nx.bitcast Nx.int64 (Nx.mul largest lift)) mantissa_bits
  in
  let exponent = Nx.maximum_s exponent 1L in
  let biased = Nx.rsub_s exponent_all_ones exponent in
  Nx.mul (Nx.mul v lift)
    (Nx.bitcast Nx.float64 (Nx.lshift biased mantissa_bits))

(* [rows what v] is [scale v]. Raises [Invalid_argument] where a row has an
   infinite component. *)
let rows what v =
  check_finite what v;
  scale v

(* [wrap a] maps an angle of [[-π, π]] into [[+0, 2π)]. An angle at or below
   zero gains 2π; a result of 2π, from a zero or a tiny negative angle, loses it
   again and is +0. *)
let wrap a =
  let a = Nx.where (Nx.less_equal_s a 0.) (Nx.add_s a two_pi) a in
  Nx.where (Nx.greater_equal_s a two_pi) (Nx.sub_s a two_pi) a

(* Constructors *)

let check_angle what x =
  Nx.check Nx.Ptree.unit
    (Nx.logical_not (Nx.isinf x))
    ()
    (fun i () ->
      Invalid_argument (strf "Direction.lonlat: %s%s is infinite" what (at i)))

let lonlat frame ~lon ~lat =
  let l = Quantity.value Unit.radian lon
  and b = Quantity.value Unit.radian lat in
  check_angle "lon" l;
  check_angle "lat" b;
  let l, b =
    match Nx.broadcast_arrays [ l; b ] with
    | [ l; b ] -> (l, b)
    | _ -> assert false
  in
  let cb = Nx.cos b in
  let xyz =
    Nx.stack ~axis:(-1) [ Nx.mul cb (Nx.cos l); Nx.mul cb (Nx.sin l); Nx.sin b ]
  in
  { frame; xyz }

let of_xyz (type d) frame (v : (float, Nx.float64_elt, d) Nx.t) : (_, d) t =
  let shape = Nx.shape v in
  let n = Array.length shape in
  if n = 0 || shape.(n - 1) <> 3 then
    invalid_arg
      (strf
         "Direction.of_xyz: expected vectors on a last axis of 3, got shape \
          [%s]"
         (String.concat "; " (Array.to_list (Array.map string_of_int shape))));
  check_finite "Direction.of_xyz: the vector" v;
  Nx.check Nx.Ptree.unit
    (Nx.logical_not (is_zero (components v)))
    ()
    (fun i () ->
      Invalid_argument
        (strf "Direction.of_xyz: the vector%s is zero and names no direction"
           (at i)));
  let s = scale v in
  let norm = Nx.sqrt (Nx.sum ~axes:[ n - 1 ] ~keepdims:true (Nx.square s)) in
  { frame; xyz = Nx.div s norm }

(* Readers *)

let lon d =
  let x, y, _ = components (rows "Direction.lon: the vector" d.xyz) in
  let pole = Nx.logical_and (Nx.equal_s x 0.) (Nx.equal_s y 0.) in
  radians (wrap (Guard.atan2 pole (Nx.full_like x 0.) y x))

let lat d =
  let x, y, z = components (rows "Direction.lat: the vector" d.xyz) in
  let rho2 = Nx.add (Nx.square x) (Nx.square y) in
  let pole = Nx.equal_s rho2 0. in
  let half_pi = Float.pi /. 2. in
  let at_pole =
    Nx.where (Nx.greater_s z 0.) (Nx.full_like z half_pi)
      (Nx.where (Nx.less_s z 0.)
         (Nx.full_like z (-.half_pi))
         (Nx.full_like z 0.))
  in
  let rho = Guard.sqrt pole (Nx.full_like rho2 0.) rho2 in
  radians (Guard.atan2 pole at_pole z rho)

(* [pair fn a b] is [a]'s and [b]'s scaled rows. *)
let pair fn a b =
  let a = rows (fn ^ ": the first direction's vector") a.xyz in
  let b = rows (fn ^ ": the second direction's vector") b.xyz in
  (components a, components b)

(* [difference p q r s] is [p q - r s] within 2 ulp whatever the cancellation
   (Kahan): [e] is the rounding error of [r s], exact through a fused
   multiply-add, and restores it. *)
let difference p q r s =
  let w = Nx.mul r s in
  let e = Nx.fma (Nx.neg r) s w in
  Nx.add (Nx.fma p q (Nx.neg w)) e

(* [fused_cross a b] is [a × b], each component within 2 ulp. Each keeps its
   relative accuracy however close the rays and whatever the rows' norms. *)
let fused_cross (ax, ay, az) (bx, by, bz) =
  (difference ay bz az by, difference az bx ax bz, difference ax by ay bx)

(* [|a × b|]'s square underflows below 2^-1022 and loses its relative accuracy
   there: where it is below [tiny_square], [a × b] is first lifted by [lift],
   which brings any nonzero cross product of scaled rows, at least 2^-1074,
   to a normal square, and any below [√tiny_square] to one in range. *)
let tiny_square = 0x1p-960
let lift = 0x1p600

(* [atan2 |a × b| (a · b)], [|a × b|] read from [a × b] lifted where its square
   is tiny, and divided by the exact lift. *)
let separation a b =
  let a, b = pair "Direction.separation" a b in
  let ((cx, cy, cz) as c) = fused_cross a b in
  let tiny = Nx.less_s (norm2 c) tiny_square in
  let k = Nx.where tiny (Nx.full_like cx lift) (Nx.full_like cx 1.) in
  let c2 = norm2 (Nx.mul cx k, Nx.mul cy k, Nx.mul cz k) in
  let singular = Nx.equal_s c2 0. in
  let cos = dot a b in
  let flat =
    Nx.where (Nx.less_s cos 0.)
      (Nx.full_like cos Float.pi)
      (Nx.full_like cos 0.)
  in
  let length = Guard.sqrt singular (Nx.full_like c2 0.) c2 in
  radians (Guard.atan2 singular flat (Nx.div length k) cos)

(* For [a = (x, y, z)], [atan2 (b · e) (b · n)] with north [n = (-x z, -y z, x²
   + y²)] and east [e = |a| (-y, x, 0)], both of norm [|a| ρ]. With [c = a × b],
   [b · e = |a| c_z] and [b · n = |a|² b_z - z (a · b) = (c × a)_z]: both are
   read from [c], which keeps their accuracy however close the rays and
   whatever the rows' norms. At a pole
   north is the limit along longitude 0, [(-sign z, 0, 0)], and east is [(0, 1,
   0)]; these are constant, so the result's derivative with respect to [a] is 0
   there. Where [a × b = 0] the bearing is 0. *)
let position_angle a b =
  let ((x, y, z) as a), ((bx, by, _) as b) =
    pair "Direction.position_angle" a b
  in
  let ((cx, cy, cz) as c) = fused_cross a b in
  let pole = Nx.logical_and (Nx.equal_s x 0.) (Nx.equal_s y 0.) in
  let length = Guard.sqrt pole (Nx.full_like x 0.) (norm2 a) in
  let east = Nx.mul length cz and north = difference cx y cy x in
  let pole_north =
    Nx.where (Nx.less_s z 0.) (Nx.full_like z 1.) (Nx.full_like z (-1.))
  in
  let east = Nx.where pole by east
  and north = Nx.where pole (Nx.mul pole_north bx) north in
  let singular = is_zero c in
  radians (wrap (Guard.atan2 singular (Nx.full_like east 0.) east north))

(* Rotations *)

(* Each output component is [fma m2 z (fma m1 y (m0 x))]: rounded the same way
   eagerly and compiled, with the entries constants of the program. *)
let rotate g c =
  if Frame.name g = Frame.name c.frame then { frame = g; xyz = c.xyz }
  else
    let m = Frame.entries c.frame g in
    let x, y, z = components c.xyz in
    let row i =
      let k v j = Nx.full_like v m.((3 * i) + j) in
      Nx.fma (k z 2) z (Nx.fma (k y 1) y (Nx.mul_s x m.(3 * i)))
    in
    { frame = g; xyz = Nx.stack ~axis:(-1) [ row 0; row 1; row 2 ] }

(* Conversions at instants *)

(* The Sun's Schwarzschild radius 2GM/c² in au (SOFA's ERFA_SRS), and the
   light time of one au in days. *)
let sun_radius = 1.97412574336e-8
let au_light_days = 149597870700. /. 299792458. /. 86400.

let map3 f (x, y, z) = (f x, f y, f z)
let add3 (ax, ay, az) (bx, by, bz) = Nx.(add ax bx, add ay by, add az bz)
let scale3 k v = map3 (Nx.mul k) v
let norm v = Nx.sqrt (norm2 v)
let unit3 v = scale3 (Nx.recip (norm v)) v

(* What the light from a source meets at the geocentre: the Earth's
   heliocentric direction [eh] and distance [em] in au, its barycentric
   velocity [v] in units of c and [bm1 = √(1 − v²)]. *)
type 'd geocentre = {
  eh : 'd vectors * 'd vectors * 'd vectors;
  em : 'd vectors;
  v : 'd vectors * 'd vectors * 'd vectors;
  bm1 : 'd vectors;
}

let geocentre e =
  let tdb = Time.convert Time.tdb (Earth.tt e) in
  let ph, _, _, vb = Ephemeris.earth tdb in
  let ph = components ph in
  let v = components (Nx.mul_s vb au_light_days) in
  let em = norm ph in
  {
    eh = scale3 (Nx.recip em) ph;
    em;
    v;
    bm1 = Nx.sqrt (Nx.rsub_s 1. (norm2 v));
  }

(* [deflect g p] is the natural direction of light that arrives from the
   direction [p] after the Sun has deflected it (eraLdsun, Klioner 2003
   (70)). Within about 5′ of the Sun's centre, [p · (p + eh)] below
   [1e-6 / max(em², 1)], the deflection falls linearly to 0.

   For unit [p] and [eh], [p · (p + eh)] is [|p + eh|² / 2]. Near the Sun,
   where [p ≈ −eh], the dot product cancels to [1 − cos θ] and an ulp of
   either norm changes it by 1e-10 relative at the limiter; the sum is
   exact there and its square keeps its digits. *)
let deflect g p =
  let limit = Nx.mul_s (Nx.recip (Nx.maximum_s (Nx.square g.em) 1.)) 1e-6 in
  let qpe = Nx.mul_s (norm2 (add3 p g.eh)) 0.5 in
  let w =
    Nx.div
      (Nx.full_like g.em sun_radius)
      (Nx.mul g.em (Nx.maximum qpe limit))
  in
  add3 p (scale3 w (cross p (cross g.eh p)))

(* [aberrate g p] is the proper direction, as the moving geocentre sees it,
   of the natural direction [p], with the Sun's potential at the Earth
   (eraAb: the Explanatory Supplement (2013) (7.40) and Klioner 2003 (7)). *)
let aberrate g p =
  let pdv = dot p g.v in
  let w1 = Nx.add_s (Nx.div pdv (Nx.add_s g.bm1 1.)) 1. in
  let w2 = Nx.div (Nx.full_like g.em sun_radius) g.em in
  let px, py, pz = p and vx, vy, vz = g.v in
  let term p v =
    Nx.add
      (Nx.add (Nx.mul p g.bm1) (Nx.mul w1 v))
      (Nx.mul w2 (Nx.sub v (Nx.mul pdv p)))
  in
  unit3 (term px vx, term py vy, term pz vz)

let apparent g p = aberrate g (deflect g p)

(* [astrometric g q] is the [p] with [apparent g p = q], from [steps]
   steps of [p ← unit (q − (apparent g p − p))] from [p = q], as eraAticq
   steps aberration and deflection apart. Taken together, the two invert
   the forward model exactly: [deflect]'s result is not a unit vector, and
   [aberrate] reads its norm. Each step multiplies the error, at most the
   20″ of aberration, by the rate of the displacement: [|v|], 1e-4, plus
   the deflection's, which peaks at 0.02 at the edge of the limiter, so 8
   steps leave it below 1e-17. *)
let steps = 8

let astrometric g q =
  let rec go p k =
    if k = 0 then p
    else go (unit3 (sub3 q (sub3 (apparent g p) p))) (k - 1)
  in
  go q steps

(* [apply m v] is [m v] for matrices [[...; 3; 3]] and vectors [[...; 3]],
   their batch axes broadcast. *)
let apply m (x, y, z) =
  let row i =
    let e j = Earth.entry m i j in
    Nx.add (Nx.add (Nx.mul (e 0) x) (Nx.mul (e 1) y)) (Nx.mul (e 2) z)
  in
  (row 0, row 1, row 2)

let stack3 (x, y, z) = Nx.stack ~axis:(-1) [ x; y; z ]

(* [ray d] is [d]'s rows scaled as {!rows} scales them, read as rays. *)
let ray d = components (scale d.xyz)

let convert : type f g d. d Earth.t -> g Frame.t -> (f, d) t -> (g, d) t =
 fun e g d ->
  check_finite "Direction.convert: the vector" d.xyz;
  let shape =
    Nx.broadcast_shapes
      [ Earth.shape e; Array.sub (Nx.shape d.xyz) 0 (Nx.ndim d.xyz - 1) ]
  in
  let out v =
    { frame = g; xyz = Nx.broadcast_to (Array.append shape [| 3 |]) v }
  in
  match (Frame.kind d.frame, Frame.kind g) with
  | Frame.Fixed, Frame.Fixed -> out (rotate g d).xyz
  | Frame.Geocentric, Frame.Geocentric ->
      out (stack3 (apply (Frame.matrix_at e d.frame g) (components d.xyz)))
  | Frame.Fixed, Frame.Geocentric ->
      let p = unit3 (ray (rotate Frame.icrs d)) in
      let q = apparent (geocentre e) p in
      out (stack3 (apply (Frame.matrix_at e Frame.gcrs g) q))
  | Frame.Geocentric, Frame.Fixed ->
      let m = Frame.matrix_at e d.frame Frame.gcrs in
      let p = apply m (ray d) in
      let q = astrometric (geocentre e) (unit3 p) in
      out (rotate g { frame = Frame.icrs; xyz = stack3 q }).xyz

(* Structure *)


let walk c d =
  let frame = Frame.walk c d.frame in
  { frame; xyz = Nx.Ptree.Walk.tensor c d.xyz }

let ptree () = Nx.Ptree.v walk
