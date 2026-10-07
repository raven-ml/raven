(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Projections between the native sphere and the plane.

   Every map works on native unit vectors [u = (cos θ cos φ, cos θ sin φ, sin
   θ)] and plane points [(x, y)] in radians, so no projection takes a longitude
   at a pole. A zenithal projection puts the native pole at the origin, with [x
   = R sin φ] and [y = -R cos φ] at distance [R] from it: [x = k u₂] and [y =
   -k u₁] with [k = R / sin Z], [Z = 90° - θ] the angle from the pole. A
   cylindrical one puts [(φ, θ) = (0, 0)] at the origin, [x] along [φ].

   Each map returns where its point is in the projection's domain: the
   connected region about the reference point where the map is a bijection.
   Outside it, the map returns finite values that no caller keeps. Formulas
   follow Calabretta and Greisen (2002), "Representations of celestial
   coordinates in FITS", A&A 395, 1077. *)

type code =
  | Azp
  | Szp
  | Tan
  | Stg
  | Sin
  | Arc
  | Zpn
  | Zea
  | Air
  | Cyp
  | Cea
  | Car
  | Mer

type v = (float, Nx.float64_elt) Nx.t

let name = function
  | Azp -> "AZP"
  | Szp -> "SZP"
  | Tan -> "TAN"
  | Stg -> "STG"
  | Sin -> "SIN"
  | Arc -> "ARC"
  | Zpn -> "ZPN"
  | Zea -> "ZEA"
  | Air -> "AIR"
  | Cyp -> "CYP"
  | Cea -> "CEA"
  | Car -> "CAR"
  | Mer -> "MER"

let codes = [ Azp; Szp; Tan; Stg; Sin; Arc; Zpn; Zea; Air; Cyp; Cea; Car; Mer ]
let of_name s = List.find_opt (fun c -> name c = s) codes

(* ZPN's polynomial has degree up to 29: FITS gives it [PVi_0] to [PVi_29]. *)
let zpn_terms = 30

(* [count c] is the number of [c]'s parameters, and [first c] the FITS index
   [m] of [PVi_m] its first one is. *)
let count = function
  | Tan | Stg | Arc | Zea | Car | Mer -> 0
  | Air | Cea -> 1
  | Azp | Sin | Cyp -> 2
  | Szp -> 3
  | Zpn -> zpn_terms

let first = function Zpn -> 0 | _ -> 1

(* The parameters' names, by index. *)
let parameter_names = function
  | Azp -> [| "μ"; "γ" |]
  | Szp -> [| "μ"; "φc"; "θc" |]
  | Sin -> [| "ξ"; "η" |]
  | Air -> [| "θb" |]
  | Cyp -> [| "μ"; "λ" |]
  | Cea -> [| "λ" |]
  | Zpn -> Array.init zpn_terms (Printf.sprintf "P%d")
  | Tan | Stg | Arc | Zea | Car | Mer -> [||]

(* The value FITS gives a parameter the file leaves out, degrees for angles. *)
let defaults = function
  | Azp | Sin -> [| 0.; 0. |]
  | Szp -> [| 0.; 0.; 90. |]
  | Air -> [| 90. |]
  | Cyp -> [| 1.; 1. |]
  | Cea -> [| 1. |]
  | Zpn -> Array.make zpn_terms 0.
  | Tan | Stg | Arc | Zea | Car | Mer -> [||]

let cylindrical = function Cyp | Cea | Car | Mer -> true | _ -> false

(* The native latitude θ₀ of the reference point FITS assumes, degrees. *)
let theta0 c = if cylindrical c then 0. else 90.

(* Tensor helpers *)

let pi = Float.pi
let half_pi = pi /. 2.
let radian_per_degree = pi /. 180.
let last v = Nx.ndim v - 1
let component v k = Nx.slice (List.init (last v) (fun _ -> Nx.A) @ [ Nx.I k ]) v
let param pv k = component pv k
let angle pv k = Nx.mul_s (param pv k) radian_per_degree
let ones x = Nx.ones_like x
let zeros x = Nx.zeros_like x
let ( &&& ) = Nx.logical_and

(* [safe_div singular a b] is [a / b], and 0 with derivative 0 where
   [singular]. *)
let safe_div singular a b =
  Nx.where singular (zeros a) (Nx.div a (Nx.where singular (ones b) b))

(* [roots ~half_b ~a ~c] is the larger root of [a t² + 2 half_b t + c = 0], [a
   > 0], without cancellation, and where it is real. *)
let larger_root ~a ~half_b ~c =
  let d = Nx.sub (Nx.square half_b) (Nx.mul a c) in
  let real = Nx.greater_equal_s d 0. in
  let sq = Guard.sqrt (Nx.logical_not (Nx.greater_s d 0.)) (zeros d) d in
  (* With [q = -half_b + sign sq], the roots are [q / a] and [c / q]. *)
  let nonneg = Nx.greater_equal_s (Nx.neg half_b) 0. in
  let big = Nx.add (Nx.abs half_b) sq in
  let flat = Nx.equal_s big 0. in
  let root =
    Nx.where nonneg (Nx.div big a)
      (safe_div flat (Nx.neg c) big)
  in
  (root, real)

(* Zenithal maps *)

(* [axis u] is [ρ = sin Z] and [Z] for the unit vector [u], with [Z] the angle
   from the native pole, and where [u] is on the polar axis. *)
let polar (u1, u2, u3) =
  let rho2 = Nx.add (Nx.square u1) (Nx.square u2) in
  let axis = Nx.equal_s rho2 0. in
  let rho = Guard.sqrt axis (zeros rho2) rho2 in
  let z = Guard.atan2 (axis &&& Nx.equal_s u3 0.) (zeros u3) rho u3 in
  (rho, z, axis)

(* [radial ~r ~k0 u] is the zenithal plane point of [u] for the radial function
   [r Z], whose ratio [r Z / sin Z] tends to [k0] at the pole. On the axis
   behind the pole, at the antipode, it is [(0, -r π)], native longitude 0. *)
let radial ~r ~k0 ((u1, u2, u3) as u) =
  let rho, z, axis = polar u in
  let k = safe_div axis (r z) rho in
  let k = Nx.where axis (Nx.broadcast_to (Nx.shape k) k0) k in
  let x = Nx.mul k u2 and y = Nx.neg (Nx.mul k u1) in
  let antipode = axis &&& Nx.less_s u3 0. in
  let y = Nx.where antipode (Nx.neg (r (Nx.full_like z pi))) y in
  (x, y)

(* [unradial ~z ~s0 x y] is the native vector of the zenithal plane point [(x,
   y)] whose angle from the pole is [z ρ], [ρ] the distance from the origin;
   [s0] is [sin Z / ρ] at the origin. *)
let unradial ~z ~s0 x y =
  let rho2 = Nx.add (Nx.square x) (Nx.square y) in
  let origin = Nx.equal_s rho2 0. in
  let rho = Guard.sqrt origin (zeros rho2) rho2 in
  let zr = z rho in
  let s = safe_div origin (Nx.sin zr) rho in
  let s = Nx.where origin (Nx.broadcast_to (Nx.shape s) s0) s in
  ((Nx.neg (Nx.mul y s), Nx.mul x s, Nx.cos zr), rho)

(* ZPN's polynomial [R ξ = Σ Pₘ ξᵐ] and its derivative, by Horner's rule. *)
let zpn_r pv xi =
  let acc = ref (param pv (zpn_terms - 1)) in
  for m = zpn_terms - 2 downto 0 do
    acc := Nx.fma !acc xi (param pv m)
  done;
  !acc

let zpn_slope pv xi =
  let acc = ref (Nx.mul_s (param pv (zpn_terms - 1)) (float (zpn_terms - 1))) in
  for m = zpn_terms - 2 downto 1 do
    acc := Nx.fma !acc xi (Nx.mul_s (param pv m) (float m))
  done;
  !acc

(* ZPN's domain ends at the first zero of [R'] in [0, π], or at π: [R'] is
   sampled at each degree, and the first sign change refined. *)
let zpn_samples = 181

let zpn_limit pv =
  let p1 = param pv 1 in
  let expand t = Nx.reshape (Array.append (Nx.shape t) [| 1 |]) t in
  let coefficients = Nx.stack ~axis:(-1)
      (List.init zpn_terms (fun m -> expand (param pv m))) in
  let grid =
    Nx.mul_s
      (Nx.cast Nx.float64 (Nx.arange Nx.int32 0 zpn_samples 1))
      radian_per_degree
  in
  let slope =
    let acc = ref (Nx.mul_s (component coefficients (zpn_terms - 1))
                     (float (zpn_terms - 1))) in
    for m = zpn_terms - 2 downto 1 do
      acc := Nx.fma !acc grid (Nx.mul_s (component coefficients m) (float m))
    done;
    !acc
  in
  let falls = Nx.less_equal_s slope 0. in
  let shape = Nx.shape falls in
  let sentinel =
    Nx.ones Nx.bool (Array.append (Array.sub shape 0 (Array.length shape - 1)) [| 1 |])
  in
  let j = Nx.argmax ~axis:(-1) (Nx.cast Nx.int32 (Nx.concatenate ~axis:(-1) [ falls; sentinel ])) in
  let none = Nx.equal_s j (Int64.of_int zpn_samples) in
  let j = Nx.cast Nx.float64 (Nx.maximum j (Nx.scalar Nx.int64 1L)) in
  let hi = Nx.where none (Nx.full_like j pi) (Nx.mul_s j radian_per_degree) in
  let lo = Nx.where none (Nx.full_like j pi) (Nx.mul_s (Nx.sub_s j 1.) radian_per_degree) in
  let rising = Nx.greater_s p1 0. in
  let lo = Nx.where rising lo (zeros lo) and hi = Nx.where rising hi (zeros hi) in
  let turn, ok = Solve.bracket (zpn_slope pv) ~lo ~hi in
  let zd = Nx.where none (Nx.full_like turn pi) turn in
  (zd, ok |> Nx.logical_or none)

(* AIR's [R ξ = -2 (ln cos ξ / tan ξ + w ln … tan ξ)], [ξ = Z / 2], with [w =
   ln cos ξb / tan² ξb] and [ξb = (90° - θb) / 2]; [w] is [-1/2] at θb = 90°. *)
let log_cos xi = Nx.log1p (Nx.mul_s (Nx.square (Nx.sin (Nx.div_s xi 2.))) (-2.))

let air_w pv =
  let xib = Nx.div_s (Nx.rsub_s half_pi (angle pv 0)) 2. in
  let zero = Nx.equal_s xib 0. in
  let w = safe_div zero (log_cos xib) (Nx.square (Nx.tan xib)) in
  Nx.where zero (Nx.full_like w (-0.5)) w

let air_r w xi =
  let zero = Nx.equal_s xi 0. in
  let t = Nx.tan xi in
  Nx.mul_s
    (Nx.add (safe_div zero (log_cos xi) t) (Nx.mul w t))
    (-2.)

(* Cylindrical maps *)

(* [lonlat u] is the native [(φ, θ)] of the unit vector [u], [φ] 0 at the
   poles, and [cos θ]. *)
let lonlat (u1, u2, u3) =
  let rho2 = Nx.add (Nx.square u1) (Nx.square u2) in
  let pole = Nx.equal_s rho2 0. in
  let rho = Guard.sqrt pole (zeros rho2) rho2 in
  let phi = Guard.atan2 pole (zeros u1) u2 u1 in
  let theta = Guard.atan2 (pole &&& Nx.equal_s u3 0.) (zeros u3) u3 rho in
  (phi, theta, rho)

let of_lonlat phi ~cos_theta ~sin_theta =
  (Nx.mul cos_theta (Nx.cos phi), Nx.mul cos_theta (Nx.sin phi), sin_theta)

let in_lon phi = Nx.less_equal_s (Nx.abs phi) pi

(* CYP's domain about θ = 0: [μ + cos θ] and [1 + μ cos θ] keep their signs at
   the reference point, so the map neither crosses its pole nor folds. *)
let cyp_domain mu c =
  let same a b = Nx.greater_s (Nx.mul a b) 0. in
  same (Nx.add mu c) (Nx.add_s mu 1.)
  &&& same (Nx.add_s (Nx.mul mu c) 1.) (Nx.add_s mu 1.)

(* Deprojection *)

(* [deproject c pv x y] is the native unit vector of the plane point [(x, y)],
   radians, and where it is in [c]'s domain. *)
let deproject code pv x y =
  match code with
  | Tan ->
      let s = Nx.rsqrt (Nx.add_s (Nx.add (Nx.square x) (Nx.square y)) 1.) in
      ((Nx.neg (Nx.mul y s), Nx.mul x s, s), Nx.ones Nx.bool [||])
  | Stg ->
      let r2 = Nx.add (Nx.square x) (Nx.square y) in
      let d = Nx.recip (Nx.add_s r2 4.) in
      ( ( Nx.mul (Nx.mul_s y (-4.)) d,
          Nx.mul (Nx.mul_s x 4.) d,
          Nx.mul (Nx.rsub_s 4. r2) d ),
        Nx.ones Nx.bool [||] )
  | Arc ->
      let u, rho = unradial ~z:Fun.id ~s0:(Nx.scalar Nx.float64 1.) x y in
      (u, Nx.less_equal_s rho pi)
  | Zea ->
      let r2 = Nx.add (Nx.square x) (Nx.square y) in
      let k2 = Nx.rsub_s 1. (Nx.div_s r2 4.) in
      let inside = Nx.greater_equal_s k2 0. in
      let k = Guard.sqrt (Nx.logical_not (Nx.greater_s k2 0.)) (zeros k2) k2 in
      ((Nx.neg (Nx.mul y k), Nx.mul x k, Nx.rsub_s 1. (Nx.div_s r2 2.)), inside)
  | Sin ->
      (* The point is [u = (−y + η z, x − ξ z, 1 − z)], [z] the smaller root
         of [(1 + ξ² + η²) z² − 2 (1 + ξ x + η y) z + x² + y² = 0]: the
         intersection that faces the direction of projection. *)
      let xi = param pv 0 and eta = param pv 1 in
      let a = Nx.add_s (Nx.add (Nx.square xi) (Nx.square eta)) 1. in
      let b = Nx.add_s (Nx.add (Nx.mul xi x) (Nx.mul eta y)) 1. in
      let r2 = Nx.add (Nx.square x) (Nx.square y) in
      let d = Nx.sub (Nx.square b) (Nx.mul a r2) in
      let inside = Nx.greater_equal_s d 0. &&& Nx.greater_s b 0. in
      let sq = Guard.sqrt (Nx.logical_not (Nx.greater_s d 0.)) (zeros d) d in
      let den = Nx.add b sq in
      let z = safe_div (Nx.logical_not (Nx.greater_s den 0.)) r2 den in
      ( ( Nx.add (Nx.neg y) (Nx.mul eta z),
          Nx.sub x (Nx.mul xi z),
          Nx.rsub_s 1. z ),
        inside )
  | Azp ->
      (* [u = D (b, a, c) − μ e₃], [D] the larger root of
         [|w|² D² − 2 μ c D + μ² − 1 = 0]: the intersection, seen from the
         point of projection, beyond the horizon's tangent. *)
      let mu = param pv 0 and gamma = angle pv 1 in
      let m1 = Nx.add_s mu 1. in
      let a = Nx.div x m1 and b = Nx.div (Nx.neg (Nx.mul y (Nx.cos gamma))) m1 in
      let c = Nx.rsub_s 1. (Nx.mul b (Nx.tan gamma)) in
      let w2 = Nx.add (Nx.add (Nx.square a) (Nx.square b)) (Nx.square c) in
      let d, real =
        larger_root ~a:w2 ~half_b:(Nx.neg (Nx.mul mu c))
          ~c:(Nx.sub_s (Nx.square mu) 1.)
      in
      ( (Nx.mul d b, Nx.mul d a, Nx.sub (Nx.mul d c) mu),
        real &&& Nx.greater_s d 0. )
  | Szp ->
      let mu = param pv 0 and phic = angle pv 1 and thetac = angle pv 2 in
      let ct = Nx.cos thetac in
      let s1 = Nx.neg (Nx.mul mu (Nx.mul ct (Nx.cos phic)))
      and s2 = Nx.neg (Nx.mul mu (Nx.mul ct (Nx.sin phic)))
      and s3 = Nx.neg (Nx.mul mu (Nx.sin thetac)) in
      let d1 = Nx.sub (Nx.neg y) s1 and d2 = Nx.sub x s2
      and d3 = Nx.rsub_s 1. s3 in
      let dd = Nx.add (Nx.add (Nx.square d1) (Nx.square d2)) (Nx.square d3) in
      let sd = Nx.add (Nx.add (Nx.mul s1 d1) (Nx.mul s2 d2)) (Nx.mul s3 d3) in
      let ss = Nx.add (Nx.add (Nx.square s1) (Nx.square s2)) (Nx.square s3) in
      let t, real = larger_root ~a:dd ~half_b:sd ~c:(Nx.sub_s ss 1.) in
      ( (Nx.fma t d1 s1, Nx.fma t d2 s2, Nx.fma t d3 s3),
        real &&& Nx.greater_s t 0. )
  | Zpn ->
      let zd, limit_ok = zpn_limit pv in
      let p0 = param pv 0 in
      let rmax = zpn_r pv zd in
      let r2 = Nx.add (Nx.square x) (Nx.square y) in
      let origin = Nx.equal_s r2 0. in
      let rho = Guard.sqrt origin (zeros r2) r2 in
      let inside = Nx.greater_equal rho p0 &&& Nx.less_equal rho rmax in
      let target = Nx.minimum (Nx.maximum rho p0) rmax in
      let zr, solved =
        Solve.bracket
          (fun xi -> Nx.sub (zpn_r pv xi) target)
          ~lo:(Nx.zeros_like target) ~hi:(Nx.broadcast_to (Nx.shape target) zd)
      in
      let s = safe_div origin (Nx.sin zr) rho in
      let s = Nx.where origin (Nx.recip (param pv 1) |> Nx.broadcast_to (Nx.shape s)) s in
      ( (Nx.neg (Nx.mul y s), Nx.mul x s, Nx.cos zr),
        inside &&& solved &&& limit_ok )
  | Air ->
      let w = air_w pv in
      let r2 = Nx.add (Nx.square x) (Nx.square y) in
      let origin = Nx.equal_s r2 0. in
      let rho = Guard.sqrt origin (zeros r2) r2 in
      (* [R ξ ≥ −2 w tan ξ], so the zero lies below [atan (ρ / (−2 w))]. *)
      let target = Nx.where origin (ones rho) rho in
      let hi = Nx.atan (Nx.div target (Nx.mul_s w (-2.))) in
      let xi, solved =
        Solve.bracket
          (fun xi -> Nx.sub (air_r w xi) target)
          ~lo:(Nx.zeros_like hi) ~hi
      in
      let z = Nx.where origin (zeros xi) (Nx.mul_s xi 2.) in
      let s = safe_div origin (Nx.sin z) rho in
      let s0 = Nx.recip (Nx.rsub_s 0.5 w) in
      let s = Nx.where origin (Nx.broadcast_to (Nx.shape s) s0) s in
      ((Nx.neg (Nx.mul y s), Nx.mul x s, Nx.cos z), solved)
  | Car ->
      let inside = in_lon x &&& Nx.less_equal_s (Nx.abs y) half_pi in
      (of_lonlat x ~cos_theta:(Nx.cos y) ~sin_theta:(Nx.sin y), inside)
  | Mer ->
      ( of_lonlat x ~cos_theta:(Nx.recip (Nx.cosh y)) ~sin_theta:(Nx.tanh y),
        in_lon x )
  | Cea ->
      let s = Nx.mul (param pv 0) y in
      let c2 = Nx.rsub_s 1. (Nx.square s) in
      let inside = in_lon x &&& Nx.greater_equal_s c2 0. in
      let c = Guard.sqrt (Nx.logical_not (Nx.greater_s c2 0.)) (zeros c2) c2 in
      (of_lonlat x ~cos_theta:c ~sin_theta:s, inside)
  | Cyp ->
      let mu = param pv 0 and lambda = param pv 1 in
      let phi = Nx.div x lambda in
      let eta = Nx.div y (Nx.add mu lambda) in
      let t = Nx.div (Nx.mul eta mu) (Nx.sqrt (Nx.add_s (Nx.square eta) 1.)) in
      let real = Nx.less_equal_s (Nx.abs t) 1. in
      let t = Nx.clamp ~min:(-1.) ~max:1. t in
      let theta = Nx.add (Nx.atan eta) (Nx.asin t) in
      let inside =
        real &&& in_lon phi
        &&& Nx.less_equal_s (Nx.abs theta) half_pi
        &&& cyp_domain mu (Nx.cos theta)
      in
      (of_lonlat phi ~cos_theta:(Nx.cos theta) ~sin_theta:(Nx.sin theta), inside)

(* Projection *)

(* [project c pv u] is the plane point, radians, of the native unit vector [u],
   and where [u] is in [c]'s domain. *)
let project code pv ((u1, u2, u3) as u) =
  match code with
  | Tan ->
      let inside = Nx.greater_s u3 0. in
      let w = Nx.where inside u3 (ones u3) in
      ((Nx.div u2 w, Nx.neg (Nx.div u1 w)), inside)
  | Stg ->
      let inside = Nx.greater_s (Nx.add_s u3 1.) 0. in
      let k = Nx.div (Nx.full_like u3 2.) (Nx.where inside (Nx.add_s u3 1.) (ones u3)) in
      ((Nx.mul k u2, Nx.neg (Nx.mul k u1)), inside)
  | Arc ->
      (radial ~r:Fun.id ~k0:(Nx.scalar Nx.float64 1.) u, Nx.ones Nx.bool [||])
  | Zea ->
      (* [R = 2 sin (Z / 2)]. *)
      let r z = Nx.mul_s (Nx.sin (Nx.div_s z 2.)) 2. in
      (radial ~r ~k0:(Nx.scalar Nx.float64 1.) u, Nx.ones Nx.bool [||])
  | Sin ->
      let xi = param pv 0 and eta = param pv 1 in
      let z = Nx.rsub_s 1. u3 in
      let facing =
        Nx.greater_equal_s
          (Nx.add u3 (Nx.sub (Nx.mul xi u2) (Nx.mul eta u1)))
          0.
      in
      ((Nx.fma xi z u2, Nx.fma eta z (Nx.neg u1)), facing)
  | Azp ->
      let mu = param pv 0 and gamma = angle pv 1 in
      let d = Nx.add (Nx.add mu u3) (Nx.mul u1 (Nx.tan gamma)) in
      let inside =
        Nx.greater_s d 0. &&& Nx.greater_equal_s (Nx.add_s (Nx.mul mu u3) 1.) 0.
      in
      let k = Nx.div (Nx.add_s mu 1.) (Nx.where inside d (ones d)) in
      ((Nx.mul k u2, Nx.neg (Nx.div (Nx.mul k u1) (Nx.cos gamma))), inside)
  | Szp ->
      let mu = param pv 0 and phic = angle pv 1 and thetac = angle pv 2 in
      let ct = Nx.cos thetac in
      let s1 = Nx.neg (Nx.mul mu (Nx.mul ct (Nx.cos phic)))
      and s2 = Nx.neg (Nx.mul mu (Nx.mul ct (Nx.sin phic)))
      and s3 = Nx.neg (Nx.mul mu (Nx.sin thetac)) in
      let zp = Nx.rsub_s 1. s3 in
      let den = Nx.sub u3 s3 in
      let su = Nx.add (Nx.add (Nx.mul s1 u1) (Nx.mul s2 u2)) (Nx.mul s3 u3) in
      let inside =
        Nx.greater_s (Nx.mul den zp) 0. &&& Nx.greater_equal_s (Nx.rsub_s 1. su) 0.
      in
      let t = Nx.div zp (Nx.where inside den (ones den)) in
      let q1 = Nx.fma t (Nx.sub u1 s1) s1 and q2 = Nx.fma t (Nx.sub u2 s2) s2 in
      ((q2, Nx.neg q1), inside)
  | Zpn ->
      let zd, limit_ok = zpn_limit pv in
      let _, z, axis = polar u in
      let p0 = param pv 0 in
      let x, y = radial ~r:(zpn_r pv) ~k0:(param pv 1) u in
      (* A pole away from the origin maps to [(0, -P₀)], native longitude 0. *)
      let ahead = axis &&& Nx.greater_s u3 0. &&& Nx.not_equal_s p0 0. in
      let y = Nx.where ahead (Nx.broadcast_to (Nx.shape y) (Nx.neg p0)) y in
      ((x, y), Nx.less_equal z zd &&& limit_ok)
  | Air ->
      let w = air_w pv in
      let _, z, _ = polar u in
      let r z = air_r w (Nx.div_s z 2.) in
      (radial ~r ~k0:(Nx.rsub_s 0.5 w) u, Nx.less_s z pi)
  | Car ->
      let phi, theta, _ = lonlat u in
      ((phi, theta), Nx.ones Nx.bool [||])
  | Mer ->
      let phi, _, rho = lonlat u in
      let inside = Nx.greater_s rho 0. in
      let y = Nx.asinh (Nx.div u3 (Nx.where inside rho (ones rho))) in
      ((phi, y), inside)
  | Cea ->
      let phi, _, _ = lonlat u in
      ((phi, Nx.div u3 (param pv 0)), Nx.ones Nx.bool [||])
  | Cyp ->
      let mu = param pv 0 and lambda = param pv 1 in
      let phi, _, rho = lonlat u in
      let inside = cyp_domain mu rho in
      let den = Nx.add mu rho in
      let y =
        Nx.div (Nx.mul (Nx.add mu lambda) u3) (Nx.where inside den (ones den))
      in
      ((Nx.mul lambda phi, y), inside)

(* Parameters *)

(* [valid c pv] is where [pv] defines a projection, and the condition's text. *)
let valid code pv =
  let t = Nx.ones Nx.bool [||] in
  match code with
  | Tan | Stg | Arc | Zea | Car | Mer | Sin -> (t, "")
  | Azp ->
      ( Nx.not_equal_s (param pv 0) (-1.)
        &&& Nx.less_s (Nx.abs (param pv 1)) 90.,
        "μ ≠ -1 and |γ| < 90 deg" )
  | Szp ->
      ( Nx.not_equal_s
          (Nx.add_s (Nx.mul (param pv 0) (Nx.sin (angle pv 2))) 1.)
          0.,
        "μ sin θc ≠ -1" )
  | Zpn -> (Nx.greater_s (param pv 1) 0., "P1 > 0")
  | Air ->
      ( Nx.greater_s (param pv 0) (-90.) &&& Nx.less_equal_s (param pv 0) 90.,
        "-90 < θb ≤ 90 deg" )
  | Cyp ->
      ( Nx.not_equal_s (param pv 1) 0.
        &&& Nx.not_equal_s (Nx.add (param pv 0) (param pv 1)) 0.
        &&& Nx.not_equal_s (param pv 0) (-1.),
        "λ ≠ 0, μ + λ ≠ 0 and μ ≠ -1" )
  | Cea ->
      ( Nx.greater_s (param pv 0) 0. &&& Nx.less_equal_s (param pv 0) 1.,
        "0 < λ ≤ 1" )
