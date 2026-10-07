(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Quantity = Ymir_units.Quantity
module Unit = Ymir_units.Unit

let strf = Printf.sprintf

type 'e plane = (float, 'e) Nx.t Quantity.t
type angles = Nx.float64_elt plane
type vectors = (float, Nx.float64_elt) Nx.t
type code = Tan | Arc
type sense = Forward | Inverse

(* Planar families hold their leaves at the plane's dtype. *)
type 'e planar =
  | Axes of { perm : int array; origin : int }
  | Shift of 'e plane
  | Linear of 'e plane
  | Scale of 'e plane

(* A projection and the rotation to its frame. [dtype] is the dtype of the
   plane: the stage computes in float64 and its inverse returns planes at
   [dtype]. *)
type ('e, 'f) celestial = {
  code : code;
  frame : 'f Frame.t;
  dtype : (float, 'e) Nx.dtype;
  stated : int array;
  pv : vectors;
  native : angles;
  crval : angles;
  lonpole : angles;
  latpole : angles;
}

(* A stage maps points of one type to another. [Deproject] is a celestial
   stage's forward map, from the plane to directions, and [Project] its
   inverse. *)
type (_, _) stage =
  | Plane : 'e planar * sense -> ('e plane, 'e plane) stage
  | Deproject : ('e, 'f) celestial -> ('e plane, 'f Direction.t) stage
  | Project : ('e, 'f) celestial -> ('f Direction.t, 'e plane) stage

type (_, _) t =
  | Id : ('a, 'a) t
  | Stage : ('a, 'b) stage * ('b, 'c) t -> ('a, 'c) t

let one s = Stage (s, Id)

(* Messages *)

let code_name = function Tan -> "TAN" | Arc -> "ARC"
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

let planar_name : type e. e planar -> string = function
  | Axes _ -> "axes"
  | Shift _ -> "shift"
  | Linear _ -> "linear"
  | Scale _ -> "scale"

let stage_name : type a b. (a, b) stage -> string = function
  | Plane (p, Forward) -> planar_name p
  | Plane (p, Inverse) -> strf "inverse %s" (planar_name p)
  | Deproject c -> strf "celestial %s" (code_name c.code)
  | Project c -> strf "inverse celestial %s" (code_name c.code)

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

(* Planar families *)

(* [axes] maps [x] to [x.(perm.(k)) + origin] in component [k]; its inverse maps
   [y] to [y.(inv.(j)) - origin] in component [j]. *)
let apply_axes fn ~perm ~origin sense (x : _ plane) =
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

let apply_planar : type e. string -> e planar -> sense -> e plane -> e plane =
 fun fn p sense x ->
  match (p, sense) with
  | Axes { perm; origin }, _ -> apply_axes fn ~perm ~origin sense x
  | Shift r, _ ->
      let u = Quantity.unit r and r = Quantity.value (Quantity.unit r) r in
      let v = Quantity.value u x in
      check_rank fn "shift" (Nx.dim (-1) r) v;
      Quantity.v u
        (match sense with Forward -> Nx.sub v r | Inverse -> Nx.add v r)
  | Linear m, Forward ->
      let v = Quantity.value Unit.one x in
      check_rank fn "linear"
        (Nx.dim (-1) (Quantity.value (Quantity.unit m) m))
        v;
      Quantity.v (Quantity.unit m)
        (product (Quantity.value (Quantity.unit m) m) v)
  | Linear m, Inverse ->
      let u = Quantity.unit m in
      let m = Quantity.value u m in
      let v = Quantity.value u x in
      check_rank fn "inverse linear" (Nx.dim (-1) m) v;
      Quantity.v Unit.one (product (Nx.inv m) v)
  | Scale d, Forward ->
      let u = Quantity.unit d and d = Quantity.value (Quantity.unit d) d in
      let v = Quantity.value Unit.one x in
      check_rank fn "scale" (Nx.dim (-1) d) v;
      Quantity.v u (Nx.mul v d)
  | Scale d, Inverse ->
      let u = Quantity.unit d and d = Quantity.value (Quantity.unit d) d in
      let v = Quantity.value u x in
      check_rank fn "inverse scale" (Nx.dim (-1) d) v;
      Quantity.v Unit.one (Nx.div v d)

(* Celestial stages

   A zenithal projection's native sphere has its pole at the reference point.
   The native vector of a plane point [(x, y)] at native longitude [φ] and
   distance [R] from the pole is [(sin R cos φ, sin R sin φ, cos R)], with [x =
   R sin φ] and [y = -R cos φ] for ARC and [R] replaced by [tan R] for TAN. The
   rotation [M] takes native vectors to the frame's: [v = M · u]. *)

let radians q = Quantity.value Unit.radian q
let half_pi = Float.pi /. 2.

(* [rotation c] is [M]'s nine entries, row-major, each of [c]'s batch shape:
   [Rz(α₀) · Ry(π/2 - δ₀) · Rz(π - φ_p)] for CRVAL [(α₀, δ₀)] and LONPOLE
   [φ_p]. *)
let rotation c =
  let crval = radians c.crval in
  let a = component crval 0 and d = component crval 1 in
  let p = radians c.lonpole in
  let ca = Nx.cos a and sa = Nx.sin a in
  let cd = Nx.cos d and sd = Nx.sin d in
  let cp = Nx.cos p and sp = Nx.sin p in
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

(* [zenith c] is where [c]'s native θ₀ is 90°. *)
let zenith c =
  Nx.equal_s (component (Quantity.value Unit.degree c.native) 1) 90.

(* Native θ₀ must be 90° until the general pole solution arrives with the
   projections whose θ₀ differs. *)
let check_native fn c =
  let theta = component (Quantity.value Unit.degree c.native) 1 in
  Nx.check Nx.Ptree.tensor (zenith c) theta (fun i t ->
      Invalid_argument
        (strf
           "%s: the %s stage's native latitude θ₀%s is %g deg; zenithal \
            projections take θ₀ = 90 deg"
           fn (code_name c.code)
           (if Array.length i = 0 then ""
            else Format.asprintf " at %a" pp_ints i)
           (Nx.item [] t)))

(* [native_of_plane c x y] is the native vector of the plane point [(x, y)] in
   radians, and where it is in the projection's domain. *)
let native_of_plane c x y =
  match c.code with
  | Tan ->
      let s = Nx.rsqrt (Nx.add_s (Nx.add (Nx.square x) (Nx.square y)) 1.) in
      ((Nx.neg (Nx.mul y s), Nx.mul x s, s), None)
  | Arc ->
      let r2 = Nx.add (Nx.square x) (Nx.square y) in
      let pole = Nx.equal_s r2 0. in
      let r = Guard.sqrt pole (Nx.zeros_like r2) r2 in
      let safe = Nx.where pole (Nx.ones_like r) r in
      let sinc = Nx.where pole (Nx.ones_like r) (Nx.div (Nx.sin safe) safe) in
      ( (Nx.neg (Nx.mul y sinc), Nx.mul x sinc, Nx.cos r),
        Some (r, Nx.less_equal_s r Float.pi) )

(* [plane_of_native c u] is the plane point in radians of the native vector [u],
   read as the ray it spans, and where it is in the projection's domain with the
   angle from the reference point. *)
let plane_of_native c (u1, u2, u3) =
  let rho2 = Nx.add (Nx.square u1) (Nx.square u2) in
  let axis = Nx.equal_s rho2 0. in
  let rho = Guard.sqrt axis (Nx.zeros_like rho2) rho2 in
  let zero_row = Nx.logical_and axis (Nx.equal_s u3 0.) in
  let angle = Guard.atan2 zero_row (Nx.zeros_like u3) rho u3 in
  match c.code with
  | Tan ->
      let inside = Nx.greater_s u3 0. in
      let w = Nx.where inside u3 (Nx.ones_like u3) in
      let nan = Nx.full_like u3 Float.nan in
      let x = Nx.where inside (Nx.div u2 w) nan
      and y = Nx.where inside (Nx.neg (Nx.div u1 w)) nan in
      (x, y, inside, angle)
  | Arc ->
      (* [x = R u2 / ρ], [y = -R u1 / ρ]; on the axis [R / ρ] is [1 / u3] at the
         reference point, and the antipode maps to [(0, -π)], native longitude
         0. *)
      let ahead = Nx.greater_s u3 0. in
      let k_axis =
        Nx.where ahead
          (Nx.recip (Nx.where ahead u3 (Nx.ones_like u3)))
          (Nx.zeros_like u3)
      in
      let k =
        Nx.where axis k_axis
          (Nx.div angle (Nx.where axis (Nx.ones_like rho) rho))
      in
      let x = Nx.mul k u2 in
      let y = Nx.neg (Nx.mul k u1) in
      let antipode = Nx.logical_and axis (Nx.less_s u3 0.) in
      let y = Nx.where antipode (Nx.full_like y (-.Float.pi)) y in
      (x, y, Nx.logical_not (Nx.isnan angle), angle)

let degrees_of r = Nx.mul_s r (180. /. Float.pi)

let reference c =
  let crval = Quantity.value Unit.degree c.crval in
  (component crval 0, component crval 1)

(* How a run treats points outside a stage's domain: [Check] raises through
   [Nx.check] at a finite one, [Cover] computes where each stage is defined. NaN
   maps to NaN, outside every domain. *)
type mode = Check | Cover

let numbers v = Nx.logical_not (Nx.any ~axes:[ last v ] (Nx.isnan v))

let deproject mode fn c (x : _ plane) =
  let v = Nx.cast Nx.float64 (radians x) in
  check_rank fn (strf "celestial %s" (code_name c.code)) 2 v;
  let u, domain = native_of_plane c (component v 0) (component v 1) in
  let ok =
    match mode with
    | Cover ->
        let ok = Nx.logical_and (numbers v) (zenith c) in
        Some
          (match domain with
          | None -> ok
          | Some (_, inside) -> Nx.logical_and ok inside)
    | Check ->
        check_native fn c;
        (match domain with
        | None -> ()
        | Some (r, inside) ->
            Nx.check Nx.Ptree.tensor
              (Nx.logical_or inside (Nx.isnan r))
              (degrees_of r)
              (fun i r ->
                Invalid_argument
                  (strf
                     "%s: %s is %.6g deg from the %s reference, beyond 180 \
                      deg, outside the projection's domain; Transform.covers \
                      gives the mask"
                     fn (point i) (Nx.item [] r) (code_name c.code))));
        None
  in
  let x, y, z = rotate (rotation c) u in
  ({ Direction.frame = c.frame; xyz = stack [ x; y; z ] }, ok)

let project mode fn c (d : _ Direction.t) =
  let v = Direction.rows (fn ^ ": the direction's vector") d.xyz in
  let u = rotate_back (rotation c) (Direction.components v) in
  let x, y, inside, angle = plane_of_native c u in
  let ok =
    match mode with
    | Cover -> Some Nx.(logical_and (logical_and (numbers v) (zenith c)) inside)
    | Check ->
        check_native fn c;
        let lon, lat = reference c in
        Nx.check
          Nx.Ptree.(pair tensor (pair tensor tensor))
          (Nx.logical_or inside (Nx.isnan angle))
          (degrees_of angle, (lon, lat))
          (fun i (r, (lon, lat)) ->
            Invalid_argument
              (strf
                 "%s: %s is %.6g deg from the %s reference (%.10g, %.10g) deg, \
                  outside the projection's domain; Transform.covers gives the \
                  mask"
                 fn (point i) (Nx.item [] r) (code_name c.code) (Nx.item [] lon)
                 (Nx.item [] lat)));
        None
  in
  (Quantity.v Unit.radian (Nx.cast c.dtype (stack [ x; y ])), ok)

(* Application *)

(* [stage mode fn s x] is [s] applied to [x] and, under [Cover], where [x] is in
   [s]'s domain. *)
let stage : type a b.
    mode -> string -> (a, b) stage -> a -> b * Nx.bool_t option =
 fun mode fn s x ->
  match s with
  | Plane (p, sense) ->
      let y = apply_planar fn p sense x in
      let ok =
        match mode with
        | Check -> None
        | Cover -> Some (numbers (Quantity.value (Quantity.unit x) x))
      in
      (y, ok)
  | Deproject c -> deproject mode fn c x
  | Project c -> project mode fn c x

let rec run : type a b. mode -> string -> (a, b) t -> a -> b * Nx.bool_t option
    =
 fun mode fn t x ->
  match t with
  | Id -> (x, None)
  | Stage (s, rest) ->
      let y, ok = stage mode fn s x in
      let z, ok' = run mode fn rest y in
      let ok =
        match (ok, ok') with
        | None, o | o, None -> o
        | Some a, Some b -> Some (Nx.logical_and a b)
      in
      (z, ok)

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

let linear m =
  let s = Nx.shape (Quantity.value (Quantity.unit m) m) in
  let k = Array.length s in
  if k < 2 || s.(k - 1) <> s.(k - 2) then
    invalid_arg
      (strf "Transform.linear: expected a matrix [...; n; n], got shape %s"
         (shape_text s));
  one (Plane (Linear m, Forward))

let parameters = function Tan | Arc -> 0

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

let celestial ?stated code frame dtype ~pv ~native ~crval ~lonpole ~latpole =
  let fn = "Transform.celestial" in
  let m = parameters code in
  let s = Nx.shape pv in
  if Array.length s = 0 || s.(Array.length s - 1) <> m then
    invalid_arg
      (strf "%s: %s takes %d parameters on pv's last axis, got shape %s" fn
         (code_name code) m (shape_text s));
  let stated =
    match stated with None -> Array.init m Fun.id | Some a -> Array.copy a
  in
  Array.iteri
    (fun k t ->
      if t < 0 || t >= m || (k > 0 && t <= stated.(k - 1)) then
        invalid_arg
          (Format.asprintf
             "%s: stated terms %a are not ascending indices of %s's %d \
              parameters"
             fn pp_ints stated (code_name code) m))
    stated;
  check_pair fn "native" native;
  check_pair fn "crval" crval;
  check_angle fn "lonpole" lonpole;
  check_angle fn "latpole" latpole;
  one
    (Deproject
       { code; frame; dtype; stated; pv; native; crval; lonpole; latpole })

let degrees x = Quantity.v Unit.degree (Nx.scalar Nx.float64 x)

(* The inverse of the zenithal stage [code] at [c] with LONPOLE 180°: x east, y
   north. *)
let zenithal code (c : _ Direction.t) =
  let crval =
    Quantity.v Unit.radian
      (stack [ radians (Direction.lon c); radians (Direction.lat c) ])
  in
  let native =
    Quantity.v Unit.degree (Nx.create Nx.float64 [| 2 |] [| 0.; 90. |])
  in
  one
    (Project
       {
         code;
         frame = c.frame;
         dtype = Nx.float64;
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

let pp_planar : type e. Format.formatter -> e planar -> unit =
 fun ppf -> function
  | Axes { perm; origin } ->
      Format.fprintf ppf "axes %a ~origin:%d" pp_ints perm origin
  | Shift r -> Format.fprintf ppf "shift %a" Quantity.pp r
  | Linear m -> Format.fprintf ppf "linear %a" Quantity.pp m
  | Scale d -> Format.fprintf ppf "scale %a" Quantity.pp d

let pp_celestial ppf c =
  Format.fprintf ppf "celestial %s %a %a ~crval:%a ~lonpole:%a"
    (code_name c.code) Frame.pp c.frame Nx.pp_dtype c.dtype Quantity.pp c.crval
    Quantity.pp c.lonpole

let pp_stage : type a b. Format.formatter -> (a, b) stage -> unit =
 fun ppf -> function
  | Plane (p, Forward) -> pp_planar ppf p
  | Plane (p, Inverse) -> Format.fprintf ppf "inverse (%a)" pp_planar p
  | Deproject c -> pp_celestial ppf c
  | Project c -> Format.fprintf ppf "inverse (%a)" pp_celestial c

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

let walk_planar : type e. ('x, 'y) W.cursor -> e planar -> e planar =
 fun c -> function
  | Axes { perm; origin } ->
      let perm = W.field c "perm" ints perm in
      let origin = W.field c "origin" W.int origin in
      Axes { perm; origin }
  | Shift r -> Shift (W.field c "offset" quantity r)
  | Linear m -> Linear (W.field c "matrix" quantity m)
  | Scale d -> Scale (W.field c "scale" quantity d)

let walk_celestial c cel =
  W.case c (code_name cel.code);
  let frame = W.field c "frame" Frame.walk cel.frame in
  W.case c (Format.asprintf "%a" Nx.pp_dtype cel.dtype);
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

(* Geometry in float64

   Grids and regions map cell corners in float64 whatever their dtype, with
   [cells] axes of cells after the batch axes of the transform's parameters: a
   parameter of batch shape [b] meets points of shape [b @ cells @ [n]]. *)

type value = P of Nx.float64_elt plane | D : 'f Direction.t -> value

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

let planar64 : type e. int -> e planar -> Nx.float64_elt planar =
 fun k -> function
  | Axes a -> Axes a
  | Shift r -> Shift (expand k 1 (Quantity.map (Nx.cast Nx.float64) r))
  | Linear m -> Linear (expand k 2 (Quantity.map (Nx.cast Nx.float64) m))
  | Scale d -> Scale (expand k 1 (Quantity.map (Nx.cast Nx.float64) d))

let celestial64 k c =
  {
    c with
    dtype = Nx.float64;
    pv = expand_tensor k 1 c.pv;
    native = expand k 1 c.native;
    crval = expand k 1 c.crval;
    lonpole = expand k 0 c.lonpole;
    latpole = expand k 0 c.latpole;
  }

let and_mask a b =
  match (a, b) with
  | None, o | o, None -> o
  | Some a, Some b -> Some (Nx.logical_and a b)

(* [run64 ~cells t x] is [t] applied to [x] in float64 and where each stage is
   defined, under [Cover]: no point raises. *)
let run64 ~cells t x =
  let fn = "Transform.apply" in
  let rec go : type a b. (a, b) t -> value -> value * Nx.bool_t option =
   fun t x ->
    match (t, x) with
    | Id, x -> (x, None)
    | Stage (Plane (p, sense), rest), P v ->
        let ok = numbers (Quantity.value (Quantity.unit v) v) in
        let y, ok' = go rest (P (apply_planar fn (planar64 cells p) sense v)) in
        (y, and_mask (Some ok) ok')
    | Stage (Deproject c, rest), P v ->
        let d, ok = deproject Cover fn (celestial64 cells c) v in
        let y, ok' = go rest (D d) in
        (y, and_mask ok ok')
    | Stage (Project c, rest), D d ->
        let d = { Direction.frame = c.frame; xyz = d.xyz } in
        let v, ok = project Cover fn (celestial64 cells c) d in
        let y, ok' = go rest (P v) in
        (y, and_mask ok ok')
    | Stage _, _ ->
        invalid_arg "Transform.run64: a stage met a point of another type"
  in
  go t x

(* Endpoints

   A grid knows its transform's input, planes at its dtype, and reads the
   output's kind from the stages. *)

type _ endpoint =
  | Plane_at : (float, 'e) Nx.dtype -> 'e plane endpoint
  | Sky : 'f Frame.t -> 'f Direction.t endpoint

let rec target : type a b. a endpoint -> (a, b) t -> b endpoint =
 fun e -> function
  | Id -> e
  | Stage (Plane _, rest) -> target e rest
  | Stage (Deproject c, rest) -> target (Sky c.frame) rest
  | Stage (Project c, rest) -> target (Plane_at c.dtype) rest

let value : type a. a endpoint -> a -> value =
 fun e x ->
  match e with
  | Plane_at _ -> P (Quantity.map (Nx.cast Nx.float64) x)
  | Sky _ -> D x

(* [map_points e f x] applies [f] to [x]'s tensor: a plane's payload or a
   direction's vectors. *)
type fn = { f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t }

let map_points : type a. a endpoint -> fn -> a -> a =
 fun e fn x ->
  match e with
  | Plane_at _ -> Quantity.map fn.f x
  | Sky _ -> { x with Direction.xyz = fn.f x.Direction.xyz }
