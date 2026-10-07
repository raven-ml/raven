(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Frames and directions for tests: the fixed frames as a list, directions with
   arbitrary vectors, and double-double references. *)

open Ymir

type fixed = F : 'a Frame.fixed Frame.t -> fixed

let fixed =
  [
    F Frame.icrs;
    F Frame.fk5_j2000;
    F Frame.galactic;
    F Frame.ecliptic_j2000;
    F Frame.supergalactic;
  ]

let name (F f) = Frame.name f
let pp_fixed ppf (F f) = Frame.pp ppf f
let vectors xs = Nx.create Nx.float64 [| Array.length xs / 3; 3 |] xs
let vector x y z = Nx.create Nx.float64 [| 3 |] [| x; y; z |]
let scalar x = Nx.scalar Nx.float64 x
let radians x = Quantity.v Unit.radian x
let degrees x = Quantity.v Unit.degree x
let in_radians q = Quantity.value Unit.radian q

(* [raw f v] is the direction of frame [f] whose vectors are [v] itself, as
   [Nx.Ptree.map] builds one: no constructor normalises it. *)
let raw f v =
  Nx.Ptree.map (Direction.ptree ())
    (fun _ x -> Nx.cast (Nx.dtype x) v)
    (Direction.of_xyz f (vector 1. 0. 0.))

let lon d = Nx.to_array (in_radians (Direction.lon d))
let lat d = Nx.to_array (in_radians (Direction.lat d))
let separation a b = Nx.to_array (in_radians (Direction.separation a b))
let position_angle a b = Nx.to_array (in_radians (Direction.position_angle a b))

let one x =
  match x with [| v |] -> v | _ -> invalid_arg "one: not one element"

(* Double-double arithmetic, accurate to about 1e-32 relative: enough to judge
   results rounded to float64. *)

let quick_two_sum a b =
  let s = a +. b in
  (s, b -. (s -. a))

let two_sum a b =
  let s = a +. b in
  let bb = s -. a in
  (s, a -. (s -. bb) +. (b -. bb))

let dd_add (ah, al) (bh, bl) =
  let s, e = two_sum ah bh in
  quick_two_sum s (e +. al +. bl)

let dd_mul (ah, al) (bh, bl) =
  let p = ah *. bh in
  let e = Float.fma ah bh (-.p) in
  quick_two_sum p (e +. Float.fma ah bl (al *. bh))

let dd x = (x, 0.)

let identity =
  Array.init 9 (fun k -> if k mod 4 = 0 then (1., 0.) else (0., 0.))

(* [orientation f] is [R_f] at 60 digits as double-doubles, row-major. *)
let orientation (F f) =
  match Frame.name f with
  | "icrs" -> identity
  | n -> List.assoc n Frames_reference.orientations

(* [exact_matrix a b] is [R_b · R_aᵀ] as double-doubles. *)
let exact_matrix a b =
  let ra = orientation a and rb = orientation b in
  Array.init 9 (fun k ->
      let i = k / 3 and j = k mod 3 in
      List.fold_left
        (fun acc q -> dd_add acc (dd_mul rb.((3 * i) + q) ra.((3 * j) + q)))
        (dd 0.) [ 0; 1; 2 ])

(* [exact_rotate a b v] is [R_b · R_aᵀ · v] as double-doubles, [v] three
   floats. *)
let exact_rotate a b v =
  let m = exact_matrix a b in
  Array.init 3 (fun i ->
      List.fold_left
        (fun acc j -> dd_add acc (dd_mul m.((3 * i) + j) (dd v.(j))))
        (dd 0.) [ 0; 1; 2 ])
