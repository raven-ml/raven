(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Exact winding numbers: the oracle of the isoband laws.

   [ring pt r] is the winding number of [r] around [pt], and [mem pt p] is
   [true] iff the rings of [p] wind around [pt] a number of times that is not
   [0] and [pt] is finite. A point on a ring is decided as the point moved from
   it by an infinitely small distance towards positive x, and then by an
   infinitely smaller one towards positive y: a box-shaped surface holds the
   points of its left and top edges and not those of its right and bottom edges,
   as a pixel does. The decision is exact unless products of coordinate
   differences overflow or underflow, so a point is in at most one of two
   polygons whose surfaces do not overlap, such as two adjacent isobands. *)

open Hugin_gg
open Hugin_gg_kit

(* Orientation

   The sign of [(ux - lx)(py - ly) - (px - lx)(uy - ly)]. The rounded value
   decides when it exceeds a bound on its error, a generous multiple of the
   bound of Shewchuk's orient2d. Otherwise the sign is that of the exact value:
   each difference and each product is split exactly into its rounded value and
   its error, and the sixteen terms are summed as an expansion whose components
   do not overlap, so that the largest non-zero one has the sign of the sum.
   Products are kept from fusing with the sums that follow them, which would
   lose their error terms. *)

let two_sum a b =
  let s = a +. b in
  let bv = s -. a in
  let av = s -. bv in
  (s, a -. av +. (b -. bv))

let two_diff a b =
  let d = a -. b in
  let bv = a -. d in
  let av = d +. bv in
  (d, a -. av +. (bv -. b))

let exact_orientation lx ly ux uy px py =
  let e = Array.make 16 0. and n = ref 0 in
  let grow b =
    let q = ref b in
    for i = 0 to !n - 1 do
      let s, r = two_sum !q e.(i) in
      e.(i) <- r;
      q := s
    done;
    e.(!n) <- !q;
    incr n
  in
  let product a b =
    let p = Sys.opaque_identity (a *. b) in
    grow p;
    grow (Float.fma a b (-.p))
  in
  let dux, dux' = two_diff ux lx and duy, duy' = two_diff uy ly in
  let dpx, dpx' = two_diff px lx and dpy, dpy' = two_diff py ly in
  product dux dpy;
  product dux dpy';
  product dux' dpy;
  product dux' dpy';
  product (-.dpx) duy;
  product (-.dpx) duy';
  product (-.dpx') duy;
  product (-.dpx') duy';
  let rec sign i =
    if i < 0 then 0
    else if e.(i) > 0. then 1
    else if e.(i) < 0. then -1
    else sign (i - 1)
  in
  sign (!n - 1)

let orientation lx ly ux uy px py =
  let left = Sys.opaque_identity ((ux -. lx) *. (py -. ly)) in
  let right = Sys.opaque_identity ((px -. lx) *. (uy -. ly)) in
  let d = left -. right in
  let bound = 4. *. epsilon_float *. (Float.abs left +. Float.abs right) in
  if d > bound then 1
  else if -.d > bound then -1
  else exact_orientation lx ly ux uy px py

(* A point on the ring is decided as the point moved from it towards positive x,
   then infinitely less towards positive y. For that point an endpoint is above
   iff its y exceeds [py], and a segment that straddles the horizontal through
   it counts iff it passes on its right: the orientation of the point from the
   segment's lower to its upper endpoint is positive. *)
let ring pt r =
  let px = P2.x pt and py = P2.y pt and n = Ring2.length r in
  let w = ref 0 in
  for i = 0 to n - 1 do
    let j = if i = n - 1 then 0 else i + 1 in
    let ax = Ring2.x r i and ay = Ring2.y r i in
    let bx = Ring2.x r j and by = Ring2.y r j in
    let a_above = ay > py and b_above = by > py in
    if b_above && (not a_above) && orientation ax ay bx by px py > 0 then incr w
    else if a_above && (not b_above) && orientation bx by ax ay px py > 0 then
      decr w
  done;
  !w

let mem pt p =
  Float.is_finite (P2.x pt)
  && Float.is_finite (P2.y pt)
  && List.fold_left (fun w r -> w + ring pt r) 0 (Pgon2.rings p) <> 0
