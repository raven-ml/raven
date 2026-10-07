(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Quantity = Ymir_units.Quantity
module Unit = Ymir_units.Unit

let strf = Printf.sprintf

type vectors = (float, Nx.float64_elt) Nx.t
type 'f t = { frame : 'f Frame.t; xyz : vectors }

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
    Nx.(bitwise_and (bitcast int64 v) (scalar int64 sign_bit_off))
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
  let exponent = Nx.maximum exponent (Nx.scalar Nx.int64 1L) in
  let biased = Nx.sub (Nx.scalar Nx.int64 exponent_all_ones) exponent in
  Nx.mul (Nx.mul v lift)
    (Nx.bitcast Nx.float64 (Nx.lshift biased mantissa_bits))

(* [rows what v] is [scale v]. Raises [Invalid_argument] where a row has an
   infinite component. *)
let rows what v =
  check_finite what v;
  scale v

(* [near a b] is [b], scaled rows, times 2, 1 or 1/2, whichever brings its
   largest component within a factor √2 of [a]'s. Two close rows then have close
   components, and their difference is exact, even where their largest
   components fall on either side of a power of two, as a row next to [(1, 0,
   0)] does. *)
let near a b =
  let root_two = Float.sqrt 2. in
  let largest v = Nx.max ~axes:[ last v ] ~keepdims:true (Nx.abs v) in
  let ma = largest a and mb = largest b in
  let up = Nx.greater ma (Nx.mul_s mb root_two) in
  let down = Nx.greater mb (Nx.mul_s ma root_two) in
  let factor =
    Nx.where up (Nx.full_like ma 2.)
      (Nx.where down (Nx.full_like ma 0.5) (Nx.full_like ma 1.))
  in
  Nx.mul b factor

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
  let l, b = Nx.broadcasted l b in
  let cb = Nx.cos b in
  let xyz =
    Nx.stack ~axis:(-1) [ Nx.mul cb (Nx.cos l); Nx.mul cb (Nx.sin l); Nx.sin b ]
  in
  { frame; xyz }

let of_xyz frame v =
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

let pair fn a b =
  let a = rows (fn ^ ": the first direction's vector") a.xyz in
  let b = rows (fn ^ ": the second direction's vector") b.xyz in
  (a, near a b)

(* For [a] and [d = b - a], [atan2 |a × d| (a · b)]. In exact arithmetic [a × d
   = a × b]; for close rows [d] is exact, so the result keeps its relative
   accuracy at any separation. *)
let separation a b =
  let a, b = pair "Direction.separation" a b in
  let a = components a and b = components b in
  let c2 = norm2 (cross a (sub3 b a)) in
  let singular = Nx.equal_s c2 0. in
  let cos = dot a b in
  let flat =
    Nx.where (Nx.less_s cos 0.)
      (Nx.full_like cos Float.pi)
      (Nx.full_like cos 0.)
  in
  radians
    (Guard.atan2 singular flat
       (Guard.sqrt singular (Nx.full_like c2 0.) c2)
       cos)

(* For [a = (x, y, z)], [atan2 (w · e) (w · n)] with north [n = (-x z, -y z, x²
   + y²)] and east [e = |a| (-y, x, 0)], both of norm [|a| ρ]. Since [a] is
   orthogonal to both, [w] may be [b], [b - a] or [b + a]: it is [b - a] for
   rows on one side, [b + a] for rows on opposite sides, each exact for rows
   close to [a] or to [-a], so the bearing keeps its accuracy there. At a pole
   north is the limit along longitude 0, [(-sign z, 0, 0)], and east is [(0, 1,
   0)]; these are constant, so the result's derivative with respect to [a] is 0
   there. Where [a × w = 0] the bearing is 0. *)
let position_angle a b =
  let a, b = pair "Direction.position_angle" a b in
  let ((x, y, z) as a) = components a and ((bx, by, bz) as b) = components b in
  let opposite = Nx.less_s (dot a b) 0. in
  let side u v = Nx.where opposite (Nx.add u v) (Nx.sub u v) in
  let ((wx, wy, wz) as w) = (side bx x, side by y, side bz z) in
  let pole = Nx.logical_and (Nx.equal_s x 0.) (Nx.equal_s y 0.) in
  let length = Guard.sqrt pole (Nx.full_like x 0.) (norm2 a) in
  let east = Nx.(mul length (add (mul (neg y) wx) (mul x wy))) in
  let north =
    Nx.(
      add
        (neg (add (mul (mul x z) wx) (mul (mul y z) wy)))
        (mul (add (square x) (square y)) wz))
  in
  let pole_north =
    Nx.where (Nx.less_s z 0.) (Nx.full_like z 1.) (Nx.full_like z (-1.))
  in
  let east = Nx.where pole by east
  and north = Nx.where pole (Nx.mul pole_north bx) north in
  let singular = is_zero (cross a w) in
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

(* Structure *)

type 'f direction = 'f t

let ptree (type f) () : f t Nx.Ptree.t =
  Nx.Ptree.instantiate
    (module struct
      type _ t = f direction

      let walk c d =
        let frame = Frame.walk c d.frame in
        { frame; xyz = Nx.Ptree.Walk.tensor c d.xyz }
    end)
