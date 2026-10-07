(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Quantity = Ymir_units.Quantity
module Unit = Ymir_units.Unit

let strf = Printf.sprintf

type plane = (float, Nx.float64_elt) Nx.t Quantity.t
type vectors = (float, Nx.float64_elt) Nx.t

type code = Projection.code =
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

type sense = Forward | Inverse

type planar =
  | Axes of { perm : int array; origin : int }
  | Shift of plane
  | Linear of plane
  | Scale of plane
  | Sip of { a : vectors; b : vectors; seed : (vectors * vectors) option }
  | Tpv of { pv : vectors; stated : int array }

(* A projection and the rotation to its frame. *)
type 'f celestial = {
  code : code;
  frame : 'f Frame.t;
  stated : int array;
  pv : vectors;
  native : plane;
  crval : plane;
  lonpole : plane;
  latpole : plane;
}

(* A stage maps points of one type to another. [Deproject] is a celestial
   stage's forward map, from the plane to directions, and [Project] its
   inverse. A rotation's inverse is the rotation between its frames swapped. *)
type (_, _) stage =
  | Plane : planar * sense -> (plane, plane) stage
  | Deproject : 'f celestial -> (plane, 'f Direction.t) stage
  | Project : 'f celestial -> ('f Direction.t, plane) stage
  | Rotate :
      'a Frame.fixed Frame.t * 'b Frame.fixed Frame.t
      -> ('a Frame.fixed Direction.t, 'b Frame.fixed Direction.t) stage

type (_, _) t =
  | Id : ('a, 'a) t
  | Stage : ('a, 'b) stage * ('b, 'c) t -> ('a, 'c) t

let one s = Stage (s, Id)

(* Messages *)

let code_name = Projection.name
let sense_name = function Forward -> "forward" | Inverse -> "inverse"

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

(* [point i] names the point at batch index [i]: ["point 12"], ["point [3; 1]"],
   or ["the point"] for a single one. *)
let index = function
  | [| i |] -> string_of_int i
  | i -> Format.asprintf "%a" pp_ints i

let point = function [||] -> "the point" | i -> "point " ^ index i
let shape_text s = Format.asprintf "%a" pp_ints s

let planar_name = function
  | Axes _ -> "axes"
  | Shift _ -> "shift"
  | Linear _ -> "linear"
  | Scale _ -> "scale"
  | Sip _ -> "sip"
  | Tpv _ -> "tpv"

let stage_name : type a b. (a, b) stage -> string = function
  | Plane (p, Forward) -> planar_name p
  | Plane (p, Inverse) -> strf "inverse %s" (planar_name p)
  | Deproject c -> strf "celestial %s" (code_name c.code)
  | Project c -> strf "inverse celestial %s" (code_name c.code)
  | Rotate (f, g) -> strf "rotation %s %s" (Frame.name f) (Frame.name g)

(* Components *)

let last v = Nx.ndim v - 1
let component v k = Nx.slice (List.init (last v) (fun _ -> Nx.A) @ [ Nx.I k ]) v

(* [broadcast a b] is the shape [a] and [b] broadcast to. *)
let broadcast a b =
  let n = max (Array.length a) (Array.length b) in
  let dim s i =
    let k = i - (n - Array.length s) in
    if k < 0 then 1 else s.(k)
  in
  Array.init n (fun i ->
      let x = dim a i and y = dim b i in
      if x = y || y = 1 then x
      else if x = 1 then y
      else
        invalid_arg
          (strf "Transform: shapes %s and %s do not broadcast" (shape_text a)
             (shape_text b)))

let stack vs =
  let shape = List.fold_left (fun s v -> broadcast s (Nx.shape v)) [||] vs in
  Nx.stack ~axis:(-1) (List.map (Nx.broadcast_to shape) vs)

let check_rank fn name n x =
  let s = Nx.shape x in
  if Array.length s = 0 || s.(Array.length s - 1) <> n then
    invalid_arg
      (strf "%s: %s takes points on a last axis of %d, got shape %s" fn name n
         (shape_text s))

(* How a run treats points outside a stage's domain: [Check] raises through
   [Nx.check] at a finite one, [Cover] computes where each stage is defined. NaN
   maps to NaN, outside every domain. *)
type mode = Check | Cover

let numbers v = Nx.logical_not (Nx.any ~axes:[ last v ] (Nx.isnan v))

(* [nan_where bad v] is [v] with each point NaN where [bad]. *)
let nan_where bad v =
  let bad = Nx.reshape (Array.append (Nx.shape bad) [| 1 |]) bad in
  Nx.where bad (Nx.full_like v Float.nan) v

(* [domain mode ok data fail] is [ok] under [Cover], and under [Check] raises
   [fail i d] where [ok] fails at a point of numbers. *)
let domain mode ~numbers s ok data fail =
  match mode with
  | Cover -> Some (Nx.logical_and numbers ok)
  | Check ->
      Nx.check s (Nx.logical_or ok (Nx.logical_not numbers)) data fail;
      None

(* [floats t] is [t]'s values as ["a, b, c"]. *)
let floats t =
  String.concat ", "
    (Array.to_list (Array.map (strf "%g") (Nx.to_array (Nx.reshape [| -1 |] t))))

(* Planar families *)

(* [axes] maps [x] to [x.(perm.(k)) + origin] in component [k]; its inverse maps
   [y] to [y.(inv.(j)) - origin] in component [j]. *)
let apply_axes fn ~perm ~origin sense (x : plane) =
  let n = Array.length perm in
  let unit = if origin = 0 then Quantity.unit x else Unit.one in
  let v = Quantity.value unit x in
  check_rank fn "axes" n v;
  let take p = stack (Array.to_list (Array.map (component v) p)) in
  let o = float_of_int origin in
  match sense with
  | Forward ->
      let y = take perm in
      Quantity.v unit (if origin = 0 then y else Nx.add_s y o)
  | Inverse ->
      let inv = Array.make n 0 in
      Array.iteri (fun k p -> inv.(p) <- k) perm;
      let y = take inv in
      Quantity.v unit (if origin = 0 then y else Nx.sub_s y o)

(* [product m x] is [m · x] over the last axes, [m] [[...; n; n]] and [x] [[...;
   n]]: each component one [fma] chain in a fixed order, so it rounds the same
   at every batch shape. *)
let product m x =
  let n = Nx.dim (-1) x in
  let entry i j =
    Nx.slice (List.init (Nx.ndim m - 2) (fun _ -> Nx.A) @ [ Nx.I i; Nx.I j ]) m
  in
  let row i =
    let acc = ref (Nx.mul (entry i 0) (component x 0)) in
    for j = 1 to n - 1 do
      acc := Nx.fma (entry i j) (component x j) !acc
    done;
    !acc
  in
  stack (List.init n row)

(* [distortion mode fn name unit ~forward ~jacobian ~inverse sense x] applies a
   distortion family to [x], read in [unit]. Its domain is where its forward
   map's Jacobian keeps its sign at the origin, positive, and, inverted, where
   Newton's method converged. *)
let distortion mode fn name unit ~forward ~jacobian ~inverse sense (x : plane) =
  let v = Quantity.value unit x in
  check_rank fn name 2 v;
  let finite = numbers v in
  let safe = Nx.where (Nx.reshape (Array.append (Nx.shape finite) [| 1 |]) finite)
      v (Nx.zeros_like v) in
  let y, solved, at =
    match sense with
    | Forward -> (forward safe, None, safe)
    | Inverse ->
        let y, ok = inverse safe in
        (y, Some ok, y)
  in
  let det = Distortion.det2 (jacobian at) in
  let unfolded = Nx.greater_s det 0. in
  let ok = match solved with None -> unfolded | Some s -> Nx.logical_and s unfolded in
  let residual =
    match sense with
    | Forward -> Nx.zeros_like det
    | Inverse ->
        let r = Nx.sub (forward y) safe in
        Nx.sqrt (Nx.sum ~axes:[ -1 ] (Nx.square r))
  in
  let mask =
    domain mode ~numbers:finite
      Nx.Ptree.(pair tensor tensor)
      ok (det, residual) (fun i (det, r) ->
        Invalid_argument
          (match sense with
          | Forward ->
              strf
                "%s: %s is where the %s distortion folds (its Jacobian's \
                 determinant is %g), outside its domain; Transform.covers \
                 gives the mask"
                fn (point i) name (Nx.item [] det)
          | Inverse ->
              strf
                "%s: %s does not invert through the %s distortion (Newton's \
                 residual is %g %s, the Jacobian's determinant %g), outside \
                 its domain; Transform.covers gives the mask"
                fn (point i) name (Nx.item [] r)
                (if Unit.equal unit Unit.one then "pixel" else Unit.to_string unit)
                (Nx.item [] det)))
  in
  (Quantity.v unit (nan_where (Nx.logical_not finite) y), mask)

let apply_planar mode fn p sense (x : plane) : plane * Nx.bool_t option =
  let plain y =
    let ok =
      match mode with
      | Check -> None
      | Cover -> Some (numbers (Quantity.value (Quantity.unit x) x))
    in
    (y, ok)
  in
  match (p, sense) with
  | Axes { perm; origin }, _ -> plain (apply_axes fn ~perm ~origin sense x)
  | Shift r, _ ->
      let u = Quantity.unit r and r = Quantity.value (Quantity.unit r) r in
      let v = Quantity.value u x in
      check_rank fn "shift" (Nx.dim (-1) r) v;
      plain
        (Quantity.v u
           (match sense with Forward -> Nx.sub v r | Inverse -> Nx.add v r))
  | Linear m, Forward ->
      let v = Quantity.value Unit.one x in
      check_rank fn "linear"
        (Nx.dim (-1) (Quantity.value (Quantity.unit m) m))
        v;
      plain
        (Quantity.v (Quantity.unit m)
           (product (Quantity.value (Quantity.unit m) m) v))
  | Linear m, Inverse ->
      let u = Quantity.unit m in
      let m = Quantity.value u m in
      let v = Quantity.value u x in
      check_rank fn "inverse linear" (Nx.dim (-1) m) v;
      plain (Quantity.v Unit.one (product (Nx.inv m) v))
  | Scale d, Forward ->
      let u = Quantity.unit d and d = Quantity.value (Quantity.unit d) d in
      let v = Quantity.value Unit.one x in
      check_rank fn "scale" (Nx.dim (-1) d) v;
      plain (Quantity.v u (Nx.mul v d))
  | Scale d, Inverse ->
      let u = Quantity.unit d and d = Quantity.value (Quantity.unit d) d in
      let v = Quantity.value u x in
      check_rank fn "inverse scale" (Nx.dim (-1) d) v;
      plain (Quantity.v Unit.one (Nx.div v d))
  | Sip { a; b; seed }, _ ->
      distortion mode fn "SIP" Unit.one
        ~forward:(Distortion.sip_forward a b)
        ~jacobian:(Distortion.sip_jacobian a b)
        ~inverse:(Distortion.sip_inverse ?seed a b)
        sense x
  | Tpv { pv; _ }, _ ->
      distortion mode fn "TPV" Unit.degree
        ~forward:(Distortion.tpv_forward pv)
        ~jacobian:(Distortion.tpv_jacobian pv)
        ~inverse:(Distortion.tpv_inverse pv)
        sense x

(* Celestial stages

   The projection maps plane points to native unit vectors, and the rotation
   [M] takes native vectors to the frame's: [v = M · u]. [M] is [Rz(α_p) ·
   Ry(90° - δ_p) · Rz(180° - φ_p)] for the celestial pole (α_p, δ_p) of the
   native sphere and LONPOLE φ_p. Its pole solves FITS WCS Paper II's
   equations from CRVAL, the native reference point (φ₀, θ₀), LONPOLE and
   LATPOLE, as WCSLIB's [celset] does: with θ₀ = 90° it is CRVAL; otherwise
   two latitudes solve and LATPOLE picks between them. Every branch is a
   [where], so the rotation is differentiable in the header's numbers. *)

let degree_per_radian = 180. /. Float.pi
let radian_per_degree = Float.pi /. 180.

(* [sincosd d] is the sine and cosine of [d] degrees, exact at multiples of
   90°: the argument is reduced to [[-45°, 45°]] about its quadrant. *)
let sincosd d =
  let q = Nx.round (Nx.div_s d 90.) in
  let e = Nx.mul_s (Nx.sub d (Nx.mul_s q 90.)) radian_per_degree in
  let s = Nx.sin e and c = Nx.cos e in
  let k = Nx.sub q (Nx.mul_s (Nx.floor (Nx.div_s q 4.)) 4.) in
  let is n = Nx.equal_s k n in
  let pick a b cc d = Nx.where (is 0.) a (Nx.where (is 1.) b (Nx.where (is 2.) cc d)) in
  (pick s c (Nx.neg s) (Nx.neg c), pick c (Nx.neg s) (Nx.neg c) s)

let atan2d y x = Nx.mul_s (Nx.atan2 y x) degree_per_radian

(* WCSLIB's tolerance on degrees and on products of cosines. *)
let pole_tol = 1e-10

let degrees_of q = Quantity.value Unit.degree q

(* [wrap180 a] maps degrees in [(-540, 540)] into [[-180, 180]]. *)
let wrap180 a =
  Nx.where (Nx.greater_s a 180.) (Nx.sub_s a 360.)
    (Nx.where (Nx.less_s a (-180.)) (Nx.add_s a 360.) a)

type pole = {
  lngp : vectors;  (** α_p, degrees. *)
  latp : vectors;  (** δ_p, degrees. *)
  phip : vectors;  (** φ_p, degrees. *)
  exists : Nx.bool_t;
}

let pole c =
  let crval = degrees_of c.crval and native = degrees_of c.native in
  let lng0 = component crval 0 and lat0 = component crval 1 in
  let phi0 = component native 0 and theta0 = component native 1 in
  let phip = degrees_of c.lonpole and latpole = degrees_of c.latpole in
  let zenith = Nx.equal_s theta0 90. in
  let slat0, clat0 = sincosd lat0 and sthe0, cthe0 = sincosd theta0 in
  let same = Nx.equal phip phi0 in
  let sphip, cphip = sincosd (Nx.sub phip phi0) in
  let x = Nx.mul cthe0 cphip and y = sthe0 in
  let z2 = Nx.add (Nx.square x) (Nx.square y) in
  let flat = Nx.equal_s z2 0. in
  let z = Guard.sqrt flat (Nx.zeros_like z2) z2 in
  let slz = Nx.div slat0 (Nx.where flat (Nx.ones_like z) z) in
  let over = Nx.greater_s (Nx.abs slz) 1. in
  let slz =
    Nx.where (Nx.logical_and over (Nx.less_s (Nx.sub_s (Nx.abs slz) 1.) pole_tol))
      (Nx.sign slz) slz
  in
  let solvable =
    Nx.where flat (Nx.equal_s slat0 0.) (Nx.less_equal_s (Nx.abs slz) 1.)
  in
  (* [acos s] as [atan2 (√(1 - s²)) s], whose derivative stays finite at ±1
     through the guard. *)
  let c2 = Nx.rsub_s 1. (Nx.square slz) in
  let edge = Nx.less_equal_s c2 0. in
  let v = atan2d (Guard.sqrt edge (Nx.zeros_like c2) c2) slz in
  let u = atan2d (Nx.broadcast_to (Nx.shape x) y) (Nx.where flat (Nx.ones_like x) x) in
  let u = Nx.where same theta0 u and v = Nx.where same (Nx.rsub_s 90. lat0) v in
  let latp1 = wrap180 (Nx.add u v) and latp2 = wrap180 (Nx.sub u v) in
  let valid l = Nx.less_s (Nx.abs l) (90. +. pole_tol) in
  let first =
    Nx.where
      (Nx.less (Nx.abs (Nx.sub latpole latp1)) (Nx.abs (Nx.sub latpole latp2)))
      (valid latp1)
      (Nx.logical_not (valid latp2))
  in
  let latp = Nx.where first latp1 latp2 in
  let latp = Nx.where (Nx.logical_and flat (Nx.logical_not same))
      (Nx.clamp ~min:(-90.) ~max:90. (Nx.broadcast_to (Nx.shape latp) latpole))
      latp in
  let exists = Nx.logical_and solvable (valid latp) in
  let latp = Nx.clamp ~min:(-90.) ~max:90. latp in
  let slatp, clatp = sincosd latp in
  let zz = Nx.mul clatp clat0 in
  let polar = Nx.less_s (Nx.abs zz) pole_tol in
  let at_pole = Nx.less_s (Nx.abs clat0) pole_tol in
  let lngp_polar =
    Nx.where at_pole lng0
      (Nx.where (Nx.greater_s latp 0.)
         (Nx.sub_s (Nx.sub (Nx.add lng0 phip) phi0) 180.)
         (Nx.add (Nx.sub lng0 phip) phi0))
  in
  let xx = Nx.div (Nx.sub sthe0 (Nx.mul slatp slat0)) (Nx.where polar (Nx.ones_like zz) zz) in
  let yy = Nx.div (Nx.mul sphip cthe0) (Nx.where at_pole (Nx.ones_like clat0) clat0) in
  let still = Nx.logical_or polar (Nx.logical_and (Nx.equal_s xx 0.) (Nx.equal_s yy 0.)) in
  let lngp_general =
    Nx.sub lng0
      (Nx.mul_s (Guard.atan2 still (Nx.zeros_like xx) yy xx) degree_per_radian)
  in
  let lngp = Nx.where polar lngp_polar lngp_general in
  {
    lngp = Nx.where zenith lng0 lngp;
    latp = Nx.where zenith lat0 latp;
    phip;
    exists = Nx.logical_or zenith exists;
  }

(* [rotation p] is [M]'s nine entries, row-major, each of the pole's batch
   shape. *)
let rotation p =
  let sa, ca = sincosd p.lngp and sd, cd = sincosd p.latp in
  let sp, cp = sincosd p.phip in
  let open Nx in
  [|
    sub (neg (mul (mul ca sd) cp)) (mul sa sp);
    add (neg (mul (mul ca sd) sp)) (mul sa cp);
    mul ca cd;
    add (neg (mul (mul sa sd) cp)) (mul ca sp);
    sub (neg (mul (mul sa sd) sp)) (mul ca cp);
    mul sa cd;
    mul cd cp;
    mul cd sp;
    sd;
  |]

let rotate m (x, y, z) =
  let row i =
    Nx.(
      add
        (add (mul m.(3 * i) x) (mul m.((3 * i) + 1) y))
        (mul m.((3 * i) + 2) z))
  in
  (row 0, row 1, row 2)

let rotate_back m (x, y, z) =
  let col j =
    Nx.(add (add (mul m.(j) x) (mul m.(3 + j) y)) (mul m.(6 + j) z))
  in
  (col 0, col 1, col 2)

let reference c =
  let crval = degrees_of c.crval in
  (component crval 0, component crval 1)

let degrees_of_radians r = Nx.mul_s r degree_per_radian

(* [setup mode fn c] is the stage's rotation, and under [Cover]
   where its parameters define it; under [Check] it raises where they do
   not. *)
let setup mode fn c =
  let name = code_name c.code in
  let p = pole c in
  let valid, condition = Projection.valid c.code c.pv in
  let ok =
    match mode with
    | Cover -> Some (Nx.logical_and p.exists valid)
    | Check ->
        let pv = List.init (Nx.dim (-1) c.pv) (component c.pv) in
        Nx.check Nx.Ptree.(list tensor) valid pv (fun i pv ->
            Invalid_argument
              (strf
                 "%s: the %s stage's parameters%s (%s) do not define a \
                  projection: it needs %s"
                 fn name
                 (if Array.length i = 0 then "" else " " ^ index i)
                 (String.concat ", " (List.map floats pv))
                 condition));
        let lon, lat = reference c in
        let native = degrees_of c.native in
        Nx.check
          Nx.Ptree.(pair (pair tensor tensor) (pair tensor tensor))
          p.exists
          ((p.phip, lon), (lat, component native 1))
          (fun i ((phip, lon), (lat, theta0)) ->
            Invalid_argument
              (strf
                 "%s: the %s stage's LONPOLE %g deg%s and CRVAL (%g, %g) deg \
                  place no celestial pole for the native reference latitude \
                  %g deg"
                 fn name (Nx.item [] phip)
                 (if Array.length i = 0 then "" else " at " ^ index i)
                 (Nx.item [] lon) (Nx.item [] lat) (Nx.item [] theta0)));
        None
  in
  (rotation p, ok)

let and_mask a b =
  match (a, b) with
  | None, o | o, None -> o
  | Some a, Some b -> Some (Nx.logical_and a b)

let deproject mode fn c (x : plane) =
  let v = Quantity.value Unit.radian x in
  check_rank fn (strf "celestial %s" (code_name c.code)) 2 v;
  let m, setup_ok = setup mode fn c in
  let u, inside =
    Projection.deproject c.code c.pv (component v 0) (component v 1)
  in
  let finite = numbers v in
  let distance = degrees_of_radians (Nx.hypot (component v 0) (component v 1)) in
  let ok =
    domain mode ~numbers:finite Nx.Ptree.tensor inside distance (fun i r ->
        Invalid_argument
          (strf
             "%s: %s is %.6g deg from the %s plane's origin, outside the \
              projection's domain; Transform.covers gives the mask"
             fn (point i) (Nx.item [] r) (code_name c.code)))
  in
  let x, y, z = rotate m u in
  let xyz = nan_where (Nx.logical_not finite) (stack [ x; y; z ]) in
  ({ Direction.frame = c.frame; xyz }, and_mask setup_ok ok)

let project mode fn c (d : _ Direction.t) =
  let v = Direction.rows (fn ^ ": the direction's vector") d.xyz in
  let m, setup_ok = setup mode fn c in
  let u1, u2, u3 = rotate_back m (Direction.components v) in
  let n2 = Nx.add (Nx.add (Nx.square u1) (Nx.square u2)) (Nx.square u3) in
  let n = Nx.sqrt n2 in
  let u = (Nx.div u1 n, Nx.div u2 n, Nx.div u3 n) in
  let (x, y), inside = Projection.project c.code c.pv u in
  let finite = numbers v in
  let angle =
    (* The angle from CRVAL's direction, for the message. *)
    let lon, lat = reference c in
    let sl, cl = sincosd lon and sb, cb = sincosd lat in
    let a1, a2, a3 = Direction.components v in
    let rx = Nx.mul cb cl and ry = Nx.mul cb sl in
    let dot = Nx.add (Nx.add (Nx.mul a1 rx) (Nx.mul a2 ry)) (Nx.mul a3 sb) in
    degrees_of_radians (Nx.acos (Nx.clamp ~min:(-1.) ~max:1. (Nx.div dot n)))
  in
  let lon, lat = reference c in
  let ok =
    domain mode ~numbers:finite
      Nx.Ptree.(pair tensor (pair tensor tensor))
      inside (angle, (lon, lat))
      (fun i (r, (lon, lat)) ->
        Invalid_argument
          (strf
             "%s: %s is %.6g deg from the %s reference (%.10g, %.10g) deg, \
              outside the projection's domain; Transform.covers gives the mask"
             fn (point i) (Nx.item [] r) (code_name c.code) (Nx.item [] lon)
             (Nx.item [] lat)))
  in
  ( Quantity.v Unit.radian (nan_where (Nx.logical_not finite) (stack [ x; y ])),
    and_mask setup_ok ok )

(* Application *)

(* [stage mode fn s x] is [s] applied to [x] and, under [Cover], where [x] is in
   [s]'s domain. *)
let stage : type a b.
    mode -> string -> (a, b) stage -> a -> b * Nx.bool_t option =
 fun mode fn s x ->
  match s with
  | Plane (p, sense) -> apply_planar mode fn p sense x
  | Deproject c -> deproject mode fn c x
  | Project c -> project mode fn c x
  | Rotate (_, g) ->
      let ok =
        match mode with
        | Check -> None
        | Cover -> Some (numbers x.Direction.xyz)
      in
      (Direction.rotate g x, ok)

let rec run : type a b. mode -> string -> (a, b) t -> a -> b * Nx.bool_t option
    =
 fun mode fn t x ->
  match t with
  | Id -> (x, None)
  | Stage (s, rest) ->
      let y, ok = stage mode fn s x in
      let z, ok' = run mode fn rest y in
      (z, and_mask ok ok')

let apply t x = fst (run Check "Transform.apply" t x)

(* Constructors *)

let id = Id

let axes perm ~origin =
  let n = Array.length perm in
  let seen = Array.make n false in
  Array.iter
    (fun p ->
      if p < 0 || p >= n || seen.(p) then
        invalid_arg
          (Format.asprintf "Transform.axes: %a is not a permutation" pp_ints
             perm);
      seen.(p) <- true)
    perm;
  one (Plane (Axes { perm = Array.copy perm; origin }, Forward))

let vector fn what q =
  let v = Quantity.value (Quantity.unit q) q in
  if Nx.ndim v = 0 then
    invalid_arg (strf "%s: %s is a scalar, expected a vector [...; n]" fn what)

let shift r =
  vector "Transform.shift" "the offset" r;
  one (Plane (Shift r, Forward))

let scale d =
  vector "Transform.scale" "the scale" d;
  one (Plane (Scale d, Forward))

let check_square fn what s =
  let k = Array.length s in
  if k < 2 || s.(k - 1) <> s.(k - 2) then
    invalid_arg
      (strf "%s: expected %s [...; n; n], got shape %s" fn what (shape_text s))

let linear m =
  check_square "Transform.linear" "a matrix"
    (Nx.shape (Quantity.value (Quantity.unit m) m));
  one (Plane (Linear m, Forward))

let sip ?seed (a, b) =
  let fn = "Transform.sip" in
  let square what m = check_square fn what (Nx.shape m) in
  square "A" a;
  square "B" b;
  Option.iter
    (fun (ap, bp) ->
      square "AP" ap;
      square "BP" bp)
    seed;
  one (Plane (Sip { a; b; seed }, Forward))

(* [check_stated fn what stated m] checks that [stated] holds ascending indices
   below [m], and is all of them by default. *)
let check_stated fn what stated m =
  let stated =
    match stated with None -> Array.init m Fun.id | Some a -> Array.copy a
  in
  Array.iteri
    (fun k t ->
      if t < 0 || t >= m || (k > 0 && t <= stated.(k - 1)) then
        invalid_arg
          (Format.asprintf
             "%s: stated terms %a are not ascending indices of %s's %d terms" fn
             pp_ints stated what m))
    stated;
  stated

let tpv ?stated pv =
  let fn = "Transform.tpv" in
  let s = Nx.shape pv in
  let k = Array.length s in
  let count = Distortion.tpv_count in
  if k < 2 || s.(k - 2) <> 2 || s.(k - 1) <> count then
    invalid_arg
      (strf "%s: expected parameters [...; 2; %d], got shape %s" fn count
         (shape_text s));
  let stated = check_stated fn "TPV" stated (2 * count) in
  one (Plane (Tpv { pv; stated }, Forward))

let check_angle fn what q =
  if not (Unit.convertible (Quantity.unit q) Unit.radian) then
    invalid_arg
      (Format.asprintf "%s: %s is in %a, which is not an angle" fn what Unit.pp
         (Quantity.unit q))

let check_pair fn what q =
  check_angle fn what q;
  let s = Nx.shape (Quantity.value (Quantity.unit q) q) in
  let k = Array.length s in
  if k = 0 || s.(k - 1) <> 2 then
    invalid_arg
      (strf "%s: %s takes [...; 2] angles, got shape %s" fn what (shape_text s))

let celestial ?stated code frame ~pv ~native ~crval ~lonpole ~latpole =
  let fn = "Transform.celestial" in
  let m = Projection.count code in
  let s = Nx.shape pv in
  if Array.length s = 0 || s.(Array.length s - 1) <> m then
    invalid_arg
      (strf "%s: %s takes %d parameters on pv's last axis, got shape %s" fn
         (code_name code) m (shape_text s));
  let stated = check_stated fn (code_name code) stated m in
  check_pair fn "native" native;
  check_pair fn "crval" crval;
  check_angle fn "lonpole" lonpole;
  check_angle fn "latpole" latpole;
  one (Deproject { code; frame; stated; pv; native; crval; lonpole; latpole })

let rotation f g = one (Rotate (f, g))
let degrees x = Quantity.v Unit.degree (Nx.scalar Nx.float64 x)

(* The inverse of the zenithal stage [code] at [c] with LONPOLE 180°: x east, y
   north. *)
let zenithal code (c : _ Direction.t) =
  let crval =
    Quantity.v Unit.radian
      (stack
         [
           Quantity.value Unit.radian (Direction.lon c);
           Quantity.value Unit.radian (Direction.lat c);
         ])
  in
  let native =
    Quantity.v Unit.degree (Nx.create Nx.float64 [| 2 |] [| 0.; 90. |])
  in
  one
    (Project
       {
         code;
         frame = c.frame;
         stated = [||];
         pv = Nx.zeros Nx.float64 [| 0 |];
         native;
         crval;
         lonpole = degrees 180.;
         latpole = degrees 90.;
       })

let about c = zenithal Arc c
let gnomonic c = zenithal Tan c

(* Composition *)

let rec ( >> ) : type a b c. (a, b) t -> (b, c) t -> (a, c) t =
 fun t u -> match t with Id -> u | Stage (s, rest) -> Stage (s, rest >> u)

let flip : type a b. (a, b) stage -> (b, a) stage = function
  | Plane (p, Forward) -> Plane (p, Inverse)
  | Plane (p, Inverse) -> Plane (p, Forward)
  | Deproject c -> Project c
  | Project c -> Deproject c
  | Rotate (f, g) -> Rotate (g, f)

let rec inverse : type a b. (a, b) t -> (b, a) t = function
  | Id -> Id
  | Stage (s, rest) -> inverse rest >> one (flip s)

(* [covers t x] is a scalar [true] for a transform with no stage, whose input
   type is unknown. *)
let covers t x =
  match run Cover "Transform.covers" t x with
  | _, Some ok -> ok
  | _, None -> Nx.scalar Nx.bool true

(* Printing *)

let pp_planar ppf = function
  | Axes { perm; origin } ->
      Format.fprintf ppf "axes %a ~origin:%d" pp_ints perm origin
  | Shift r -> Format.fprintf ppf "shift %a" Quantity.pp r
  | Linear m -> Format.fprintf ppf "linear %a" Quantity.pp m
  | Scale d -> Format.fprintf ppf "scale %a" Quantity.pp d
  | Sip { a; b; seed } ->
      Format.fprintf ppf "sip%s (%a, %a)"
        (if Option.is_some seed then " ~seed" else "")
        Nx.pp a Nx.pp b
  | Tpv { pv; _ } -> Format.fprintf ppf "tpv %a" Nx.pp pv

let pp_celestial ppf c =
  Format.fprintf ppf "celestial %s %a ~crval:%a ~lonpole:%a" (code_name c.code)
    Frame.pp c.frame Quantity.pp c.crval Quantity.pp c.lonpole

let pp_stage : type a b. Format.formatter -> (a, b) stage -> unit =
 fun ppf -> function
  | Plane (p, Forward) -> pp_planar ppf p
  | Plane (p, Inverse) -> Format.fprintf ppf "inverse (%a)" pp_planar p
  | Deproject c -> pp_celestial ppf c
  | Project c -> Format.fprintf ppf "inverse (%a)" pp_celestial c
  | Rotate (f, g) -> Format.fprintf ppf "rotation %a %a" Frame.pp f Frame.pp g

let pp ppf t =
  let rec stages : type a b. bool -> (a, b) t -> unit =
   fun first -> function
     | Id -> if first then Format.pp_print_string ppf "id"
     | Stage (s, rest) ->
         if not first then Format.fprintf ppf "@ >> ";
         pp_stage ppf s;
         stages false rest
  in
  Format.fprintf ppf "@[<hov 2>";
  stages true t;
  Format.fprintf ppf "@]"

(* Structure *)

module W = Nx.Ptree.Walk

let quantity c q = W.structure (Nx.Ptree.instantiate (module Quantity)) c q

let ints c a =
  ignore (W.int c (Array.length a));
  Array.map (W.int c) a

let walk_planar c = function
  | Axes { perm; origin } ->
      let perm = W.field c "perm" ints perm in
      let origin = W.field c "origin" W.int origin in
      Axes { perm; origin }
  | Shift r -> Shift (W.field c "offset" quantity r)
  | Linear m -> Linear (W.field c "matrix" quantity m)
  | Scale d -> Scale (W.field c "scale" quantity d)
  | Sip { a; b; seed } ->
      let a = W.field c "a" W.tensor a in
      let b = W.field c "b" W.tensor b in
      let seed = W.field c "seed" (W.structure Nx.Ptree.(option (pair tensor tensor))) seed in
      Sip { a; b; seed }
  | Tpv { pv; stated } ->
      let stated = W.field c "stated" ints stated in
      let pv = W.field c "pv" W.tensor pv in
      Tpv { pv; stated }

let walk_celestial c cel =
  W.case c (code_name cel.code);
  let frame = W.field c "frame" Frame.walk cel.frame in
  let stated = W.field c "stated" ints cel.stated in
  let pv = W.field c "pv" W.tensor cel.pv in
  let native = W.field c "native" quantity cel.native in
  let crval = W.field c "crval" quantity cel.crval in
  let lonpole = W.field c "lonpole" quantity cel.lonpole in
  let latpole = W.field c "latpole" quantity cel.latpole in
  { cel with frame; stated; pv; native; crval; lonpole; latpole }

let walk_stage : type a b. ('x, 'y) W.cursor -> (a, b) stage -> (a, b) stage =
 fun c -> function
  | Plane (p, sense) ->
      W.case c (planar_name p);
      W.case c (sense_name sense);
      Plane (walk_planar c p, sense)
  | Deproject cel ->
      W.case c "celestial";
      W.case c "forward";
      Deproject (walk_celestial c cel)
  | Project cel ->
      W.case c "celestial";
      W.case c "inverse";
      Project (walk_celestial c cel)
  | Rotate (f, g) ->
      W.case c "rotation";
      W.case c "forward";
      let f = W.field c "from" Frame.walk f in
      let g = W.field c "to" Frame.walk g in
      Rotate (f, g)

let rec length : type a b. (a, b) t -> int = function
  | Id -> 0
  | Stage (_, rest) -> 1 + length rest

let walk c t =
  let rec stages : type a b. int -> (a, b) t -> (a, b) t =
   fun i -> function
     | Id -> Id
     | Stage (s, rest) ->
         let s = W.index c i walk_stage s in
         Stage (s, stages (i + 1) rest)
  in
  ignore (W.int c (length t));
  stages 0 t

type ('a, 'b) transform = ('a, 'b) t

let ptree (type a b) () : (a, b) t Nx.Ptree.t =
  Nx.Ptree.instantiate
    (module struct
      type _ t = (a, b) transform

      let walk = walk
    end)

(* Geometry on cells

   Grids and regions map cell corners with [cells] axes of cells after the
   batch axes of the transform's parameters: a parameter of batch shape [b]
   meets points of shape [b @ cells @ [n]]. *)

type value = P of plane | D : 'f Direction.t -> value

(* [expand k core q] inserts [k] axes of 1 before [q]'s last [core] axes. *)
let expand_tensor k core v =
  if k = 0 then v
  else
    let s = Nx.shape v in
    let b = Array.length s - core in
    Nx.reshape
      (Array.concat [ Array.sub s 0 b; Array.make k 1; Array.sub s b core ])
      v

let expand k core q = Quantity.map (expand_tensor k core) q

let expand_planar k = function
  | Axes a -> Axes a
  | Shift r -> Shift (expand k 1 r)
  | Linear m -> Linear (expand k 2 m)
  | Scale d -> Scale (expand k 1 d)
  | Sip { a; b; seed } ->
      let e = expand_tensor k 2 in
      Sip { a = e a; b = e b; seed = Option.map (fun (p, q) -> (e p, e q)) seed }
  | Tpv { pv; stated } -> Tpv { pv = expand_tensor k 2 pv; stated }

let expand_celestial k c =
  {
    c with
    pv = expand_tensor k 1 c.pv;
    native = expand k 1 c.native;
    crval = expand k 1 c.crval;
    lonpole = expand k 0 c.lonpole;
    latpole = expand k 0 c.latpole;
  }

(* [run_cells ~cells t x] is [t] applied to [x] and where each stage is
   defined, under [Cover]: no point raises. *)
let run_cells ~cells t x =
  let fn = "Transform.apply" in
  let rec go : type a b. (a, b) t -> value -> value * Nx.bool_t option =
   fun t x ->
    match (t, x) with
    | Id, x -> (x, None)
    | Stage (Plane (p, sense), rest), P v ->
        let y, ok = apply_planar Cover fn (expand_planar cells p) sense v in
        let y, ok' = go rest (P y) in
        (y, and_mask ok ok')
    | Stage (Deproject c, rest), P v ->
        let d, ok = deproject Cover fn (expand_celestial cells c) v in
        let y, ok' = go rest (D d) in
        (y, and_mask ok ok')
    | Stage (Project c, rest), D d ->
        let d = { Direction.frame = c.frame; xyz = d.xyz } in
        let v, ok = project Cover fn (expand_celestial cells c) d in
        let y, ok' = go rest (P v) in
        (y, and_mask ok ok')
    | Stage (Rotate (f, g), rest), D d ->
        let d = Direction.rotate g { Direction.frame = f; xyz = d.xyz } in
        let y, ok' = go rest (D d) in
        (y, and_mask (Some (numbers d.xyz)) ok')
    | Stage _, _ ->
        invalid_arg "Transform.run_cells: a stage met a point of another type"
  in
  go t x

(* Endpoints

   A grid knows its transform's input, a plane, and reads the output's kind
   from the stages. *)

type _ endpoint =
  | Planar : plane endpoint
  | Sky : 'f Frame.t -> 'f Direction.t endpoint

let rec target : type a b. a endpoint -> (a, b) t -> b endpoint =
 fun e -> function
  | Id -> e
  | Stage (Plane _, rest) -> target e rest
  | Stage (Deproject c, rest) -> target (Sky c.frame) rest
  | Stage (Project _, rest) -> target Planar rest
  | Stage (Rotate (_, g), rest) -> target (Sky g) rest

let value : type a. a endpoint -> a -> value =
 fun e x -> match e with Planar -> P x | Sky _ -> D x

(* [map_points e f x] applies [f] to [x]'s tensor: a plane's payload or a
   direction's vectors. *)
type fn = { f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t }

let map_points : type a. a endpoint -> fn -> a -> a =
 fun e fn x ->
  match e with
  | Planar -> Quantity.map fn.f x
  | Sky _ -> { x with Direction.xyz = fn.f x.Direction.xyz }
