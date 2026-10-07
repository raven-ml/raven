(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The exact area of a convex quadrilateral cell inside a disc centred on the
   origin, with fixed shapes and no data-dependent branch.

   Each cell edge from [a] to [b] is clipped to the circle at the parameters [t1
   <= t2] of [a + t (b - a)], clamped to [0, 1]. Its contribution to the area
   (Green's theorem) is the segment from [p1] to [p2], inside the disc, and an
   arc of the circle for each part outside: from [a]'s radial projection onto
   the circle to [p1]'s, and from [p2]'s to [b]'s. The pieces of all four edges
   join into one closed curve, which bounds the cell's part inside the disc.

   Each term is taken about the cell's first corner [c0], so its size is the
   cell's: a segment from [p] to [q] adds [(p - c0) × (q - c0) / 2], and an arc
   from [s] to [e] of angle [θ] adds the chord's term plus the circular segment
   [r² (θ - sin θ) / 2]. The terms dropped by the change of origin, [c0 × (q -
   p)], sum to zero over the closed curve.

   A cell with every corner inside weighs exactly 1. A cell whose nearest point
   is at least [r] from the centre and that does not hold the centre weighs
   exactly 0. Other weights are clamped to [0, 1]. The weight is the area inside
   over the cell's area, both signed by the cell's orientation, so either
   orientation gives the same weight. *)

type 'e t = {
  weight : (float, 'e) Nx.t;
  area : (float, 'e) Nx.t;  (** The cell's shoelace area. *)
  outside : Nx.bool_t;  (** Where the weight is exactly 0 by the predicate. *)
  full : Nx.bool_t;  (** Where the weight is exactly 1 by the predicate. *)
}

let cross (ax, ay) (bx, by) = Nx.sub (Nx.mul ax by) (Nx.mul ay bx)
let sub (ax, ay) (bx, by) = (Nx.sub ax bx, Nx.sub ay by)
let dot (ax, ay) (bx, by) = Nx.add (Nx.mul ax bx) (Nx.mul ay by)
let along (ax, ay) t (dx, dy) = (Nx.fma t dx ax, Nx.fma t dy ay)
let clamp01 t = Nx.clamp ~min:0. ~max:1. t

(* [θ - sin θ] without cancellation: its Taylor series below 0.5, where seven
   terms reach float64's rounding, and the difference above. *)
let segment_angle theta =
  let small = Nx.less_s (Nx.abs theta) 0.5 in
  let t2 = Nx.square theta in
  let coefficients =
    [
      1. /. 6.;
      -1. /. 120.;
      1. /. 5040.;
      -1. /. 362880.;
      1. /. 39916800.;
      -1. /. 6227020800.;
      1. /. 1307674368000.;
    ]
  in
  let series =
    List.fold_right
      (fun c acc -> Nx.add_s (Nx.mul t2 acc) c)
      coefficients (Nx.zeros_like theta)
  in
  let series = Nx.mul (Nx.mul t2 theta) series in
  Nx.where small series (Nx.sub theta (Nx.sin theta))

(* [project r p] is [p] moved radially onto the circle of radius [r]; the centre
   stays at the centre. *)
let project r (x, y) =
  let n2 = Nx.add (Nx.square x) (Nx.square y) in
  let centre = Nx.equal_s n2 0. in
  let n = Guard.sqrt centre (Nx.ones_like n2) n2 in
  let k = Nx.div r (Nx.where centre (Nx.ones_like n) n) in
  let k = Nx.where centre (Nx.zeros_like k) k in
  (Nx.mul k x, Nx.mul k y)

(* [arc r c0 s e] is the term of the arc from [s] to [e] on the circle, both on
   it, about [c0]. *)
let arc r c0 s e =
  let sin_r2 = cross s e and cos_r2 = dot s e in
  let still = Nx.logical_and (Nx.equal_s sin_r2 0.) (Nx.equal_s cos_r2 0.) in
  let theta = Guard.atan2 still (Nx.zeros_like sin_r2) sin_r2 cos_r2 in
  let chord = cross (sub s c0) (sub e c0) in
  Nx.add chord (Nx.mul (Nx.square r) (segment_angle theta))

(* [edge r c0 a b] is twice the edge's term, and the squared distance from the
   centre to the edge. *)
let edge r c0 a b =
  let d = sub b a in
  let qa = dot d d and qb = dot a d in
  let qc = Nx.sub (dot a a) (Nx.square r) in
  let disc = Nx.sub (Nx.square qb) (Nx.mul qa qc) in
  let degenerate = Nx.equal_s qa 0. in
  let qa = Nx.where degenerate (Nx.ones_like qa) qa in
  let crossing =
    Nx.logical_and (Nx.greater_s disc 0.) (Nx.logical_not degenerate)
  in
  let root = Guard.sqrt (Nx.logical_not crossing) (Nx.zeros_like disc) disc in
  let nearest = clamp01 (Nx.div (Nx.neg qb) qa) in
  let t1 =
    Nx.where crossing (clamp01 (Nx.div (Nx.sub (Nx.neg qb) root) qa)) nearest
  in
  let t2 = Nx.where crossing (clamp01 (Nx.div (Nx.sub root qb) qa)) nearest in
  let p1 = along a t1 d and p2 = along a t2 d in
  let term =
    Nx.add
      (cross (sub p1 c0) (sub p2 c0))
      (Nx.add
         (arc r c0 (project r a) (project r p1))
         (arc r c0 (project r p2) (project r b)))
  in
  let q = along a nearest d in
  (term, dot q q)

(* [disc r xs ys] weighs the cells whose corners are [(xs.(k), ys.(k))], each
   [[...]], against the disc of radius [r] about the origin. [r] broadcasts
   against the corners; a zero radius weighs 0. *)
let disc r xs ys =
  let corner k = (xs.(k), ys.(k)) in
  let c0 = corner 0 in
  let area =
    Nx.add
      (cross (sub (corner 1) c0) (sub (corner 2) c0))
      (cross (sub (corner 2) c0) (sub (corner 3) c0))
  in
  let r2 = Nx.square r in
  let terms =
    List.init 4 (fun k -> edge r c0 (corner k) (corner ((k + 1) mod 4)))
  in
  let inside =
    List.fold_left (fun acc (t, _) -> Nx.add acc t) (Nx.zeros_like area) terms
  in
  let nearest =
    List.fold_left
      (fun acc (_, d2) -> Nx.minimum acc d2)
      (snd (List.hd terms))
      (List.tl terms)
  in
  let within =
    List.init 4 (fun k ->
        let x, y = corner k in
        Nx.less_equal (Nx.add (Nx.square x) (Nx.square y)) r2)
  in
  let all_in =
    List.fold_left Nx.logical_and (List.hd within) (List.tl within)
  in
  (* The cell holds the centre when the centre is on the inner side of every
     edge: [a × b] has the sign of the cell's area. *)
  let holds =
    List.init 4 (fun k ->
        Nx.greater_equal_s
          (Nx.mul (cross (corner k) (corner ((k + 1) mod 4))) area)
          0.)
  in
  let holds = List.fold_left Nx.logical_and (List.hd holds) (List.tl holds) in
  let zero_radius = Nx.equal_s r 0. in
  let outside =
    Nx.logical_or zero_radius
      (Nx.logical_and (Nx.greater_equal nearest r2) (Nx.logical_not holds))
  in
  let flat = Nx.equal_s area 0. in
  let ratio = Nx.div inside (Nx.where flat (Nx.ones_like area) area) in
  let weight =
    Nx.where outside (Nx.zeros_like ratio)
      (Nx.where all_in (Nx.ones_like ratio)
         (Nx.where flat (Nx.zeros_like ratio) (clamp01 ratio)))
  in
  let full = Nx.logical_and all_in (Nx.logical_not outside) in
  { weight; area = Nx.mul_s area 0.5; outside; full }

(* [annulus inner outer xs ys] is the outer disc's weight minus the inner's. A
   cell is exactly outside where it is outside the outer disc or wholly inside
   the inner one. *)
let annulus inner outer xs ys =
  let o = disc outer xs ys and i = disc inner xs ys in
  let outside = Nx.logical_or o.outside (Nx.logical_and i.full o.full) in
  let full = Nx.logical_and o.full i.outside in
  let weight =
    Nx.where outside (Nx.zeros_like o.weight)
      (Nx.where full (Nx.ones_like o.weight)
         (clamp01 (Nx.sub o.weight i.weight)))
  in
  { weight; area = o.area; outside; full }
