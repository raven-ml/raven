(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'a fixed = Fixed_frame of 'a
type icrs = [ `Icrs ] fixed
type fk5_j2000 = [ `Fk5_j2000 ] fixed
type galactic = [ `Galactic ] fixed
type ecliptic_j2000 = [ `Ecliptic_j2000 ] fixed
type supergalactic = [ `Supergalactic ] fixed

type 'a geocentric = Geocentric_frame of 'a
type gcrs = [ `Gcrs ] geocentric
type cirs = [ `Cirs ] geocentric
type true_of_date = [ `True_of_date ] geocentric
type tirs = [ `Tirs ] geocentric

type _ t =
  | Icrs : icrs t
  | Fk5_j2000 : fk5_j2000 t
  | Galactic : galactic t
  | Ecliptic_j2000 : ecliptic_j2000 t
  | Supergalactic : supergalactic t
  | Gcrs : gcrs t
  | Cirs : cirs t
  | True_of_date : true_of_date t
  | Tirs : tirs t

let icrs = Icrs
let fk5_j2000 = Fk5_j2000
let galactic = Galactic
let ecliptic_j2000 = Ecliptic_j2000
let supergalactic = Supergalactic
let gcrs = Gcrs
let cirs = Cirs
let true_of_date = True_of_date
let tirs = Tirs

let name : type f. f t -> string = function
  | Icrs -> "icrs"
  | Fk5_j2000 -> "fk5_j2000"
  | Galactic -> "galactic"
  | Ecliptic_j2000 -> "ecliptic_j2000"
  | Supergalactic -> "supergalactic"
  | Gcrs -> "gcrs"
  | Cirs -> "cirs"
  | True_of_date -> "true_of_date"
  | Tirs -> "tirs"

(* [kind f] says which of the two families [f] is in, refining its type. *)
type _ kind = Fixed : 'a fixed kind | Geocentric : 'a geocentric kind

let kind : type f. f t -> f kind = function
  | Icrs -> Fixed
  | Fk5_j2000 -> Fixed
  | Galactic -> Fixed
  | Ecliptic_j2000 -> Fixed
  | Supergalactic -> Fixed
  | Gcrs -> Geocentric
  | Cirs -> Geocentric
  | True_of_date -> Geocentric
  | Tirs -> Geocentric

let pp ppf f = Format.pp_print_string ppf (name f)

let walk c f =
  Nx.Ptree.Walk.case c (name f);
  f

(* Orientations *)

let identity = [| 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1. |]

(* [orientation f] is [R_f], row-major, with [v_f = R_f · v_icrs]. *)
let orientation : type a. a fixed t -> float array = function
  | Icrs -> identity
  | Fk5_j2000 -> Orientation.fk5_j2000
  | Galactic -> Orientation.galactic
  | Ecliptic_j2000 -> Orientation.ecliptic_j2000
  | Supergalactic -> Orientation.supergalactic

let transpose m = Array.init 9 (fun k -> m.((k mod 3 * 3) + (k / 3)))

(* [product rb ra] is [R_b · R_aᵀ], each entry one [Float.fma] chain over the
   inner index. Swapping [ra] and [rb] swaps the factors of every product and
   keeps their order, so [product ra rb] is the exact transpose of [product rb
   ra]. *)
let product rb ra =
  Array.init 9 (fun k ->
      let i = k / 3 and j = k mod 3 in
      let p q = (rb.((3 * i) + q), ra.((3 * j) + q)) in
      let b0, a0 = p 0 and b1, a1 = p 1 and b2, a2 = p 2 in
      Float.fma b2 a2 (Float.fma b1 a1 (b0 *. a0)))

(* [entries a b] is [matrix a b] as a row-major array. *)
let entries : type a b. a fixed t -> b fixed t -> float array =
 fun a b ->
  match (a, b) with
  | _ when name a = name b -> identity
  | Icrs, _ -> orientation b
  | _, Icrs -> transpose (orientation a)
  | _ -> product (orientation b) (orientation a)

let matrix a b = Nx.create Nx.float64 [| 3; 3 |] (entries a b)

(* Orientations at instants *)

(* [constant e m] is the row-major entries [m] as [[...; 3; 3]] beside and
   across [e]'s instants. *)
let constant e m =
  Nx.broadcast_to
    (Array.append (Earth.shape e) [| 3; 3 |])
    (Nx.place (Earth.beside e) (Nx.create Nx.float64 [| 3; 3 |] m))

(* The intermediate frames, CIRS, TIRS and the true equator and equinox of
   date, share the CIP as their z-axis and differ by the angle θ of their
   x-axis from the CIO about it, [v_f = R3(θ_f) v_cirs]: 0 for CIRS, the ERA
   for TIRS (IERS Conventions 2010, 5.5) and the equation of the origins for
   the equinox (5.30). Between two of them the matrix is [R3(θ_b − θ_a)],
   which needs the series only for the equinox. From ICRS's axes, an
   intermediate frame's orientation is [R3(θ_f) C2I] and GCRS's the
   identity. *)
let matrix_at : type a b. 'd Earth.t -> a t -> b t -> 'd Earth.tensor =
 fun e a b ->
  let series = lazy (Earth.intermediate e) in
  let theta : type f. f t -> 'd Earth.tensor option = function
    | Cirs -> Some (Nx.zeros_like e.Earth.ut1_second)
    | Tirs -> Some (Earth.era_radians e)
    | True_of_date -> Some (snd (Lazy.force series))
    | _ -> None
  in
  let orientation : type f. f t -> 'd Earth.tensor =
   fun f ->
    match (kind f, theta f) with
    | Fixed, _ -> constant e (entries icrs f)
    | Geocentric, None -> constant e identity
    | Geocentric, Some t -> Nx.matmul (Earth.rz t) (fst (Lazy.force series))
  in
  match (kind a, kind b) with
  | Fixed, Fixed -> constant e (entries a b)
  | _ when name a = name b -> constant e identity
  | _ -> (
      match (theta a, theta b) with
      | Some ta, Some tb -> Earth.rz (Nx.sub tb ta)
      | _ ->
          Nx.matmul (orientation b) (Nx.matrix_transpose (orientation a)))
