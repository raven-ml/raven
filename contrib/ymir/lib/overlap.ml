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
  (* [qb² − qa (|a|² − r²)] is [qa r² − (a × d)²]: written so, it keeps its
     sign where [r²] is below the rounding of [|a|²], as for a small disc
     near a long edge. *)
  let disc = Nx.sub (Nx.mul qa (Nx.square r)) (Nx.square (cross a d)) in
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

(* [shoelace xs ys] is twice the signed area of the quadrilateral of corners
   [(xs.(k), ys.(k))], about its first corner. *)
let shoelace xs ys =
  let corner k = (xs.(k), ys.(k)) in
  let c0 = corner 0 in
  Nx.add
    (cross (sub (corner 1) c0) (sub (corner 2) c0))
    (cross (sub (corner 2) c0) (sub (corner 3) c0))

(* Ellipses

   An ellipse of semi-axes [a] along the direction at [angle] from +y toward
   +x and [b] across it is the disc of radius [m = min a b] after the affine
   map that shrinks the axis along the direction by [m / a] and the one
   across by [m / b]. The map scales every area alike, so a cell's weight is
   the disc's weight on its mapped corners; shrinking keeps every coordinate
   within the cell's size. A cell maps to a sliver [a / b] times thinner, and
   the weights' rounding grows as the square of that ratio: 1e-14 of the
   ellipse's area at a ratio of 100, 1e-12 at 1000. *)
let ellipse a b angle xs ys =
  let s = Nx.sin angle and c = Nx.cos angle in
  let empty = Nx.logical_or (Nx.equal_s a 0.) (Nx.equal_s b 0.) in
  let a' = Nx.where empty (Nx.ones_like a) a
  and b' = Nx.where empty (Nx.ones_like b) b in
  let m = Nx.minimum a' b' in
  let ka = Nx.div m a' and kb = Nx.div m b' in
  let across k = Nx.mul (Nx.sub (Nx.mul xs.(k) c) (Nx.mul ys.(k) s)) kb
  and along k = Nx.mul (Nx.add (Nx.mul xs.(k) s) (Nx.mul ys.(k) c)) ka in
  let o = disc m (Array.init 4 across) (Array.init 4 along) in
  let weight = Nx.where empty (Nx.zeros_like o.weight) o.weight in
  {
    weight;
    area = Nx.mul_s (shoelace xs ys) 0.5;
    outside = Nx.logical_or empty o.outside;
    full = Nx.logical_and (Nx.logical_not empty) o.full;
  }

(* Polygons

   The polygon is the signed sum of the triangles [(c₀, Vⱼ, Vⱼ₊₁)] fanned from
   the cell's first corner [c₀], and a convex cell's intersection with each
   triangle is bounded by the parts of the triangle's edges inside the cell
   and of the cell's edges inside the triangle. Taken about [c₀], the terms of
   the fan's edges and of the cell's two edges through [c₀] vanish, which
   leaves, for each polygon edge, the edge clipped to the cell and the cell's
   edges [c₁c₂] and [c₂c₃] clipped to its triangle, each clip an interval of
   one parameter (Liang and Barsky). The cell and the polygon are first
   oriented counter-clockwise. A segment that lies along a boundary counts
   once in each triangle: a polygon edge along a cell edge counts where the
   triangle's interior and the cell's meet, and a cell edge along a polygon
   edge never. *)

(* [clip planes p q] is the interval [[t0, t1]] of [p + t (q - p)], [t] in [0,
   1], on the inner side of every half-plane: [planes] lists, for each, its
   side function's values at [p] and [q] (inner where positive), and where the
   segment lies on its line, whether that counts as inside. *)
let clip planes =
  List.fold_left
    (fun (t0, t1) (f0, f1, on_line) ->
      let line = Nx.logical_and (Nx.equal_s f0 0.) (Nx.equal_s f1 0.) in
      let both_in =
        Nx.logical_and (Nx.greater_equal_s f0 0.) (Nx.greater_equal_s f1 0.)
      in
      let both_out = Nx.logical_and (Nx.less_s f0 0.) (Nx.less_s f1 0.) in
      let span = Nx.sub f0 f1 in
      let flat = Nx.equal_s span 0. in
      let t = Nx.div f0 (Nx.where flat (Nx.ones_like span) span) in
      let entering = Nx.less_s f0 0. in
      let t0' = Nx.where entering (Nx.maximum t0 t) t0
      and t1' = Nx.where entering t1 (Nx.minimum t1 t) in
      let crossing = Nx.logical_not (Nx.logical_or both_in both_out) in
      let t0' = Nx.where crossing t0' t0 and t1' = Nx.where crossing t1' t1 in
      let empty =
        Nx.logical_or
          (Nx.logical_and line (Nx.logical_not on_line))
          (Nx.logical_and both_out (Nx.logical_not line))
      in
      ( Nx.where empty (Nx.ones_like t0') t0',
        Nx.where empty (Nx.zeros_like t1') t1' ))
    (Nx.zeros_like (let f0, _, _ = List.hd planes in f0),
     Nx.ones_like (let f0, _, _ = List.hd planes in f0))
    planes

(* [segment c0 p q (t0, t1)] is twice the term of [p + t (q - p)] for [t] in
   [[t0, t1]], about [c0], and 0 for an empty interval. *)
let segment c0 p q (t0, t1) =
  let d = sub q p in
  let a = along p t0 d and b = along p t1 d in
  let term = cross (sub a c0) (sub b c0) in
  Nx.where (Nx.greater t1 t0) term (Nx.zeros_like term)

(* [side a b p] is [(b - a) × (p - a)], positive left of [a → b]. *)
let side a b p = cross (sub b a) (sub p a)

(* [winding vs p] is whether [p] is inside the polygon [vs], by its winding
   number: crossings of the ray toward +x, upward ones counted where [p] is
   left of the edge and downward ones where it is right. *)
let winding vs (px, py) =
  let n = Array.length vs in
  let count = ref (Nx.zeros_like (Nx.add px py)) in
  for j = 0 to n - 1 do
    let ((_, ay) as a) = vs.(j) and ((_, by) as b) = vs.((j + 1) mod n) in
    let s = side a b (px, py) in
    let up =
      Nx.logical_and
        (Nx.logical_and (Nx.less_equal ay py) (Nx.less py by))
        (Nx.greater_s s 0.)
    and down =
      Nx.logical_and
        (Nx.logical_and (Nx.less_equal by py) (Nx.less py ay))
        (Nx.less_s s 0.)
    in
    let one = Nx.ones_like !count in
    count :=
      Nx.add !count
        (Nx.sub
           (Nx.where up one (Nx.zeros_like one))
           (Nx.where down one (Nx.zeros_like one)))
  done;
  Nx.not_equal_s !count 0.

(* [crosses a b c d] is whether the segments [a b] and [c d] cross at a point
   inside both. *)
let crosses a b c d =
  let s1 = side a b c and s2 = side a b d in
  let s3 = side c d a and s4 = side c d b in
  Nx.logical_and
    (Nx.less_s (Nx.mul s1 s2) 0.)
    (Nx.less_s (Nx.mul s3 s4) 0.)

(* [polygon vs xs ys] weighs the cells whose corners are [(xs.(k), ys.(k))]
   against the polygon of vertices [vs], each coordinate broadcasting against
   the corners. *)
let polygon vs xs ys =
  let n = Array.length vs in
  (* Reflect a clockwise cell, and with it the polygon, so the cell turns
     counter-clockwise. *)
  let twice = shoelace xs ys in
  let flip = Nx.less_s twice 0. in
  let mirror x = Nx.where flip (Nx.neg x) x in
  let xs = Array.map mirror xs in
  let vs = Array.map (fun (x, y) -> (mirror x, y)) vs in
  (* Orient the polygon counter-clockwise. *)
  let twice_own =
    let acc = ref (Nx.zeros_like (fst vs.(0))) in
    for j = 0 to n - 1 do
      acc := Nx.add !acc (cross vs.(j) vs.((j + 1) mod n))
    done;
    !acc
  in
  let reversed = Nx.less_s twice_own 0. in
  let edge j =
    let a = vs.(j) and b = vs.((j + 1) mod n) in
    let pick (ax, ay) (bx, by) = (Nx.where reversed bx ax, Nx.where reversed by ay) in
    (pick a b, pick b a)
  in
  let corner k = (xs.(k), ys.(k)) in
  let c0 = corner 0 in
  (* A polygon edge along a cell edge bounds the triangle's part of the cell
     where the triangle and the cell lie on one side of it: where the edges
     run the same way, for a counter-clockwise triangle. *)
  let cell_planes orientation p q =
    List.init 4 (fun k ->
        let a = corner k and b = corner ((k + 1) mod 4) in
        let same =
          Nx.greater_s (Nx.mul orientation (dot (sub q p) (sub b a))) 0.
        in
        (side a b p, side a b q, same))
  in
  let total = ref (Nx.zeros_like (Nx.add twice (fst vs.(0)))) in
  let never = Nx.zeros Nx.bool [||] in
  for j = 0 to n - 1 do
    let a, b = edge j in
    let orientation = Nx.sign (side c0 a b) in
    (* The polygon edge inside the cell. *)
    total :=
      Nx.add !total (segment c0 a b (clip (cell_planes orientation a b)));
    (* The cell's far edges inside the triangle (c₀, a, b), oriented as it. *)
    let tri_planes p q =
      List.map
        (fun (u, v) ->
          ( Nx.mul orientation (side u v p),
            Nx.mul orientation (side u v q),
            never ))
        [ (c0, a); (a, b); (b, c0) ]
    in
    List.iter
      (fun k ->
        let p = corner k and q = corner (k + 1) in
        let term = segment c0 p q (clip (tri_planes p q)) in
        total := Nx.add !total (Nx.mul orientation term))
      [ 1; 2 ]
  done;
  let area = Nx.abs twice in
  (* Predicates: every corner inside and no edge crossing, or no corner
     inside, no edge crossing and no vertex inside the cell. *)
  let inside = Array.init 4 (fun k -> winding vs (corner k)) in
  let all_in = Array.fold_left Nx.logical_and inside.(0) (Array.sub inside 1 3) in
  let any_in = Array.fold_left Nx.logical_or inside.(0) (Array.sub inside 1 3) in
  let cross_any = ref (Nx.zeros Nx.bool [||]) in
  let vertex_in = ref (Nx.zeros Nx.bool [||]) in
  for j = 0 to n - 1 do
    let a = vs.(j) and b = vs.((j + 1) mod n) in
    for k = 0 to 3 do
      cross_any :=
        Nx.logical_or !cross_any (crosses a b (corner k) (corner ((k + 1) mod 4)))
    done;
    let within =
      List.fold_left
        (fun acc k ->
          Nx.logical_and acc
            (Nx.greater_equal_s (side (corner k) (corner ((k + 1) mod 4)) a) 0.))
        (Nx.ones Nx.bool [||]) [ 0; 1; 2; 3 ]
    in
    vertex_in := Nx.logical_or !vertex_in within
  done;
  let full = Nx.logical_and all_in (Nx.logical_not !cross_any) in
  let outside =
    Nx.logical_not (Nx.logical_or (Nx.logical_or any_in !cross_any) !vertex_in)
  in
  let flat = Nx.equal_s area 0. in
  let ratio = Nx.div !total (Nx.where flat (Nx.ones_like area) area) in
  let weight =
    Nx.where outside (Nx.zeros_like ratio)
      (Nx.where full (Nx.ones_like ratio)
         (Nx.where flat (Nx.zeros_like ratio) (clamp01 ratio)))
  in
  { weight; area = Nx.mul_s twice 0.5; outside; full }
