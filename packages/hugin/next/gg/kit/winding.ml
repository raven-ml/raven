(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

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
let number n x y px py =
  let w = ref 0 in
  for i = 0 to n - 1 do
    let j = if i = n - 1 then 0 else i + 1 in
    let ax = x i and ay = y i and bx = x j and by = y j in
    let a_above = ay > py and b_above = by > py in
    if b_above && (not a_above) && orientation ax ay bx by px py > 0 then incr w
    else if a_above && (not b_above) && orientation bx by ax ay px py > 0 then
      decr w
  done;
  !w
