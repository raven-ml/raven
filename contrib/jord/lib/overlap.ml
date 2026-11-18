(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* [v] clamped to [lo, hi], with derivative 1 on [lo, hi) and 0 elsewhere: at
   a window's end it moves as the side after the end does, so the area's
   derivative is the right-hand one there, as [Warp.bilinear]'s is on its
   centre lines. *)
let clip ~lo ~hi v =
  Nx.where (Nx.greater_equal v hi) (Nx.broadcast_to (Nx.shape v) hi)
    (Nx.where (Nx.less v lo) (Nx.broadcast_to (Nx.shape v) lo) v)

(* The mean of max(y, 0) along a segment from height u to height v: half their
   sum where neither is negative, 0 where neither is positive, and else the
   share of the segment above 0, max(u, v) / (|u| + |v|), times half its top.
   Its value and first derivatives agree across both changes of sign, but at u =
   v = 0, where the first branch gives the right-hand derivative. *)
let positive_mean u v =
  let above = Nx.logical_and (Nx.greater_equal_s u 0.) (Nx.greater_equal_s v 0.)
  and below = Nx.logical_and (Nx.less_equal_s u 0.) (Nx.less_equal_s v 0.) in
  let mixed = Nx.logical_not (Nx.logical_or above below) in
  let hi = Nx.maximum u v in
  let span = Nx.where mixed (Nx.add (Nx.abs u) (Nx.abs v)) (Nx.ones_like u) in
  let part = Nx.div (Nx.mul hi hi) (Nx.mul_s span 2.) in
  Nx.where above
    (Nx.mul_s (Nx.add u v) 0.5)
    (Nx.where mixed part (Nx.zeros_like u))

let term ~lo ~hi (yp, xp) (yq, xq) =
  let a = clip ~lo ~hi xp and b = clip ~lo ~hi xq in
  let dx = Nx.sub xq xp in
  let vertical = Nx.equal_s dx 0. in
  let safe = Nx.where vertical (Nx.ones_like dx) dx in
  (* The edge's parameter at [x], clamped to [0, 1] by selection rather than
     by [Nx.clamp], whose derivative at a bound rune splits between the tied
     operands. At a bound the parameter keeps its derivative, the right-hand
     one: it reaches a bound at an unclipped end, where it does not move, or
     at an end the window clips at [lo] or [hi], where moving the edge's
     other end to the right moves it inward. Adding 0 turns the -0 of [x =
     xp] on a leftward edge into 0, as [Nx.clamp] does. *)
  let along x at =
    let t = Nx.div (Nx.sub x xp) safe in
    let t =
      Nx.where (Nx.less_s t 0.) (Nx.zeros_like t)
        (Nx.where (Nx.greater_s t 1.) (Nx.ones_like t) t)
    in
    Nx.where vertical (Nx.full_like dx at) (Nx.add_s t 0.)
  in
  let dy = Nx.sub yq yp in
  let ya = Nx.add yp (Nx.mul (along a 0.) dy)
  and yb = Nx.add yp (Nx.mul (along b 1.) dy) in
  let height =
    Nx.sub (positive_mean ya yb)
      (positive_mean (Nx.sub_s ya 1.) (Nx.sub_s yb 1.))
  in
  let w = Nx.sub b a in
  (Nx.mul w height, Nx.abs w)
