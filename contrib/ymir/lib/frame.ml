(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type !'a fixed
type icrs = [ `Icrs ] fixed
type fk5_j2000 = [ `Fk5_j2000 ] fixed
type galactic = [ `Galactic ] fixed
type ecliptic_j2000 = [ `Ecliptic_j2000 ] fixed
type supergalactic = [ `Supergalactic ] fixed

type _ t =
  | Icrs : icrs t
  | Fk5_j2000 : fk5_j2000 t
  | Galactic : galactic t
  | Ecliptic_j2000 : ecliptic_j2000 t
  | Supergalactic : supergalactic t

let icrs = Icrs
let fk5_j2000 = Fk5_j2000
let galactic = Galactic
let ecliptic_j2000 = Ecliptic_j2000
let supergalactic = Supergalactic

let name : type f. f t -> string = function
  | Icrs -> "icrs"
  | Fk5_j2000 -> "fk5_j2000"
  | Galactic -> "galactic"
  | Ecliptic_j2000 -> "ecliptic_j2000"
  | Supergalactic -> "supergalactic"

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
