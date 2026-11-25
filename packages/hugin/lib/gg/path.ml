(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_float_text

type op =
  | Move of float * float
  | Line of float * float
  | Cubic of float * float * float * float * float * float
  | Close
  | Poly of { xs : float array; ys : float array; closed : bool }

(* Operations in reverse drawing order, so that building is a cons. The arrays
   of a [Poly] have equal lengths of at least two and belong to the path. *)
type t = op list

let empty = []
let is_empty = function [] -> true | _ :: _ -> false

(* Building *)

let has_current_point = function
  | [] | Close :: _ | Poly { closed = true; _ } :: _ -> false
  | (Move _ | Line _ | Cubic _ | Poly { closed = false; _ }) :: _ -> true

let move_to pt p = Move (P2.x pt, P2.y pt) :: p

let line_to pt p =
  if has_current_point p then Line (P2.x pt, P2.y pt) :: p
  else Move (P2.x pt, P2.y pt) :: p

let cubic_to c c' pt p =
  if has_current_point p then
    Cubic (P2.x c, P2.y c, P2.x c', P2.y c', P2.x pt, P2.y pt) :: p
  else Move (P2.x pt, P2.y pt) :: p

(* The control points of the cubic of a quadratic are two thirds of the way from
   each end point to the quadratic's control point. Stepping from the end point
   keeps a control point that coincides with it exactly there; where the step
   overflows, summing thirds of the coordinates cannot. A non-finite start makes
   the cubic's first control point meaningless, and the gap rule ignores the
   start, so [c] stands in for it there. *)
let quad_to c pt p =
  let elevate x0 y0 =
    let cx = P2.x c and cy = P2.y c and x = P2.x pt and y = P2.y pt in
    let two_thirds a c =
      let d = c -. a in
      if Float.is_finite d then a +. (2. *. (d /. 3.))
      else (a /. 3.) +. (2. *. (c /. 3.))
    in
    let c1x, c1y =
      if Float.is_finite x0 && Float.is_finite y0 then
        (two_thirds x0 cx, two_thirds y0 cy)
      else (cx, cy)
    in
    Cubic (c1x, c1y, two_thirds x cx, two_thirds y cy, x, y) :: p
  in
  match p with
  | [] | Close :: _ | Poly { closed = true; _ } :: _ ->
      Move (P2.x pt, P2.y pt) :: p
  | (Move (x, y) | Line (x, y) | Cubic (_, _, _, _, x, y)) :: _ -> elevate x y
  | Poly { xs; ys; closed = false } :: _ ->
      let n = Array.length xs - 1 in
      elevate xs.(n) ys.(n)

let max_sweep = 2000. *. Float.pi

(* Each cubic spans [da <= pi/2] radians, with its control points on the
   tangents at distance [4/3 tan (da/4)] times the radius, which departs from
   the circle by at most 0.027% of the radius. *)
let arc c r ~start ~sweep p =
  if not (r >= 0. && Float.is_finite r) then
    invalid_arg (Printf.sprintf "Path.arc: invalid radius %s" (Float_print.g r));
  if not (Float.is_finite start && Float.is_finite sweep) then
    invalid_arg
      (Printf.sprintf "Path.arc: angles not finite (%s, %s)"
         (Float_print.g start) (Float_print.g sweep));
  if Float.abs sweep > max_sweep then
    invalid_arg
      (Printf.sprintf "Path.arc: sweep %s exceeds 1000 turns"
         (Float_print.g sweep));
  let cx = P2.x c and cy = P2.y c in
  let cos0 = Float.cos start and sin0 = Float.sin start in
  let x0 = cx +. (r *. cos0) and y0 = cy +. (r *. sin0) in
  let p =
    if has_current_point p then Line (x0, y0) :: p else Move (x0, y0) :: p
  in
  let n = int_of_float (Float.ceil (Float.abs sweep /. (Float.pi /. 2.))) in
  let da = sweep /. float n in
  let k = r *. (4. /. 3.) *. Float.tan (da /. 4.) in
  let rec segments i cos0 sin0 x0 y0 p =
    if i > n then p
    else
      let a1 = if i = n then start +. sweep else start +. (float i *. da) in
      let cos1 = Float.cos a1 and sin1 = Float.sin a1 in
      let x1 = cx +. (r *. cos1) and y1 = cy +. (r *. sin1) in
      let seg =
        Cubic
          ( x0 -. (k *. sin0),
            y0 +. (k *. cos0),
            x1 +. (k *. sin1),
            y1 -. (k *. cos1),
            x1,
            y1 )
      in
      segments (i + 1) cos1 sin1 x1 y1 (seg :: p)
  in
  segments 1 cos0 sin0 x0 y0 p

let close p =
  match p with
  | [] | Close :: _ | Move _ :: _ | Poly { closed = true; _ } :: _ -> p
  | Poly ({ closed = false; _ } as poly) :: rest ->
      Poly { poly with closed = true } :: rest
  | (Line _ | Cubic _) :: _ -> Close :: p

let append q p = q @ p

let transform (m : Affine.t) p =
  let[@inline] tx x y = (m.xx *. x) +. (m.xy *. y) +. m.x0 in
  let[@inline] ty x y = (m.yx *. x) +. (m.yy *. y) +. m.y0 in
  let op = function
    | Move (x, y) -> Move (tx x y, ty x y)
    | Line (x, y) -> Line (tx x y, ty x y)
    | Cubic (c1x, c1y, c2x, c2y, x, y) ->
        Cubic (tx c1x c1y, ty c1x c1y, tx c2x c2y, ty c2x c2y, tx x y, ty x y)
    | Close -> Close
    | Poly { xs; ys; closed } ->
        let n = Array.length xs in
        let xs' = Array.create_float n and ys' = Array.create_float n in
        for i = 0 to n - 1 do
          let x = Array.unsafe_get xs i and y = Array.unsafe_get ys i in
          Array.unsafe_set xs' i (tx x y);
          Array.unsafe_set ys' i (ty x y)
        done;
        Poly { xs = xs'; ys = ys'; closed }
  in
  List.map op p

(* Shapes *)

let rect b =
  let x0 = Box2.minx b and y0 = Box2.miny b in
  let x1 = Box2.maxx b and y1 = Box2.maxy b in
  [
    Poly { xs = [| x0; x1; x1; x0 |]; ys = [| y0; y0; y1; y1 |]; closed = true };
  ]

let circle c r =
  if not (r >= 0. && Float.is_finite r) then
    invalid_arg
      (Printf.sprintf "Path.circle: invalid radius %s" (Float_print.g r));
  close (arc c r ~start:0. ~sweep:(2. *. Float.pi) empty)

let poly fn ~closed xs ys =
  let n = Array.length xs in
  if n <> Array.length ys then
    invalid_arg
      (Printf.sprintf "Path.%s: %d xs but %d ys" fn n (Array.length ys));
  if n < 2 then []
  else [ Poly { xs = Array.copy xs; ys = Array.copy ys; closed } ]

let polyline xs ys = poly "polyline" ~closed:false xs ys
let polygon xs ys = poly "polygon" ~closed:true xs ys

(* Traversing *)

(* [fold_built] folds over the segments [p] was built from, in drawing order,
   with polylines expanded point by point. *)
let fold_built ~move ~line ~cubic ~close acc p =
  let step acc = function
    | Move (x, y) -> move acc x y
    | Line (x, y) -> line acc x y
    | Cubic (c1x, c1y, c2x, c2y, x, y) -> cubic acc c1x c1y c2x c2y x y
    | Close -> close acc
    | Poly { xs; ys; closed } ->
        let acc =
          ref (move acc (Array.unsafe_get xs 0) (Array.unsafe_get ys 0))
        in
        for i = 1 to Array.length xs - 1 do
          acc := line !acc (Array.unsafe_get xs i) (Array.unsafe_get ys i)
        done;
        if closed then close !acc else !acc
  in
  List.fold_left step acc (List.rev p)

(* [fold_drawn m] is [fold] over [transform m p] without building it: every
   built point is mapped through [m], then the gap rule applies, and [gap acc]
   is called at each gap. [started] is whether a subpath is current and [drawn]
   whether it has a segment. *)
let fold_drawn (m : Affine.t) ~move ~line ~cubic ~close ~gap acc p =
  let started = ref false and drawn = ref false in
  let[@inline] tx x y = (m.xx *. x) +. (m.xy *. y) +. m.x0 in
  let[@inline] ty x y = (m.yx *. x) +. (m.yy *. y) +. m.y0 in
  let finite x y = Float.is_finite x && Float.is_finite y in
  let start acc x y =
    started := true;
    drawn := false;
    move acc x y
  in
  let gap acc =
    started := false;
    gap acc
  in
  (* The built point [(x, y)] starts a subpath if [starts] or none is current,
     and otherwise ends a line. Inlined, a polyline's points are boxed only for
     the callbacks. In OCaml 5 a store to a ref is a release store, which costs
     more than the rest of a point, hence the test of [!drawn]. *)
  let[@inline] point acc starts x y =
    let x' = tx x y and y' = ty x y in
    if not (Float.is_finite x' && Float.is_finite y') then gap acc
    else if !started && not starts then begin
      if not !drawn then drawn := true;
      line acc x' y'
    end
    else start acc x' y'
  in
  let close_subpath acc =
    let closes = !started && !drawn in
    started := false;
    if closes then close acc else acc
  in
  let step acc = function
    | Move (x, y) -> point acc true x y
    | Line (x, y) -> point acc false x y
    | Cubic (c1x, c1y, c2x, c2y, x, y) ->
        let c1x' = tx c1x c1y and c1y' = ty c1x c1y in
        let c2x' = tx c2x c2y and c2y' = ty c2x c2y in
        let x' = tx x y and y' = ty x y in
        if not (finite c1x' c1y' && finite c2x' c2y' && finite x' y') then
          gap acc
        else if !started then begin
          drawn := true;
          cubic acc c1x' c1y' c2x' c2y' x' y'
        end
        else start acc x' y'
    | Close -> close_subpath acc
    | Poly { xs; ys; closed } ->
        let acc =
          ref (point acc true (Array.unsafe_get xs 0) (Array.unsafe_get ys 0))
        in
        for i = 1 to Array.length xs - 1 do
          acc :=
            point !acc false (Array.unsafe_get xs i) (Array.unsafe_get ys i)
        done;
        if closed then close_subpath !acc else !acc
  in
  List.fold_left step acc (List.rev p)

let fold ~move ~line ~cubic ~close acc p =
  fold_drawn Affine.id ~move ~line ~cubic ~close ~gap:Fun.id acc p

let bezier t a b c d =
  let u = 1. -. t in
  (u *. u *. u *. a)
  +. (3. *. u *. u *. t *. b)
  +. (3. *. u *. t *. t *. c)
  +. (t *. t *. t *. d)

let max_chords = 65536.

(* Wang's bound: n chords at uniform parameters keep a cubic within the
   tolerance if n² is at least 0.75 d / tolerance, where d is the largest norm
   of the second differences of its control points. *)
let flatten ?(tolerance = 0.1) m ~move ~line ~close acc p =
  if not (tolerance > 0.) then
    invalid_arg
      (Printf.sprintf "Path.flatten: tolerance %s not positive"
         (Float_print.g tolerance));
  (* The current point after [m]. *)
  let cur = Array.create_float 2 in
  let at acc x y f =
    Array.unsafe_set cur 0 x;
    Array.unsafe_set cur 1 y;
    f acc x y
  in
  fold_drawn m
    ~move:(fun acc x y -> at acc x y move)
    ~line:(fun acc x y -> at acc x y line)
    ~cubic:(fun acc x1 y1 x2 y2 x3 y3 ->
      let x0 = Array.unsafe_get cur 0 and y0 = Array.unsafe_get cur 1 in
      let dd ax ay bx by cx cy =
        Float.hypot (ax -. (2. *. bx) +. cx) (ay -. (2. *. by) +. cy)
      in
      let d = Float.max (dd x0 y0 x1 y1 x2 y2) (dd x1 y1 x2 y2 x3 y3) in
      let k = Float.ceil (Float.sqrt (0.75 *. d /. tolerance)) in
      (* [k] is NaN for an infinite curve under an infinite tolerance. *)
      let n =
        if Float.is_nan k then 1
        else int_of_float (Float.min max_chords (Float.max 1. k))
      in
      let acc = ref acc in
      for i = 1 to n - 1 do
        let t = float i /. float n in
        acc := line !acc (bezier t x0 x1 x2 x3) (bezier t y0 y1 y2 y3)
      done;
      at !acc x3 y3 line)
    ~close ~gap:Fun.id acc p

(* [turns a b c d] is the parameters in ]0;1[ at which the cubic Bézier
   coordinate with control values [a], [b], [c] and [d] turns, the roots of its
   derivative, in increasing order. The values are first scaled by a power of
   two to at most 1, which moves no root, so that the derivative's coefficients
   and its discriminant are floats however large the values. Without real roots
   the square root is NaN, and so are the roots, which are dropped. *)
let turns a b c d =
  let top =
    Float.max
      (Float.max (Float.abs a) (Float.abs b))
      (Float.max (Float.abs c) (Float.abs d))
  in
  if not (top > 0. && Float.is_finite top) then []
  else
    let e = -snd (Float.frexp top) in
    let a = Float.ldexp a e and b = Float.ldexp b e in
    let c = Float.ldexp c e and d = Float.ldexp d e in
    let qa = -.a +. (3. *. b) -. (3. *. c) +. d in
    let qb = 2. *. (a -. (2. *. b) +. c) in
    let qc = b -. a in
    let disc = (qb *. qb) -. (4. *. qa *. qc) in
    let q = -0.5 *. (qb +. Float.copy_sign (Float.sqrt disc) qb) in
    List.filter (fun t -> 0. < t && t < 1.) [ q /. qa; qc /. q ]
    |> List.sort_uniq Float.compare

(* [widen bounds k a b c d] widens the interval of [bounds] at [k] and [k + 2]
   by the extremes of the cubic Bézier coordinate with control values [a], [b],
   [c] and [d], its values where it turns. An extreme is clamped to the control
   values, which bound the curve, so rounding never leaves them. *)
let widen bounds k a b c d =
  let lo = Float.min (Float.min a b) (Float.min c d) in
  let hi = Float.max (Float.max a b) (Float.max c d) in
  List.iter
    (fun t ->
      let v = Float.min hi (Float.max lo (bezier t a b c d)) in
      bounds.(k) <- Float.min bounds.(k) v;
      bounds.(k + 2) <- Float.max bounds.(k + 2) v)
    (turns a b c d)

(* [extent m p] is [bounds (transform m p)] and whether that path has a gap,
   from one traversal. *)
let extent m p =
  (* [minx; miny; maxx; maxy; current x; current y] *)
  let b = [| infinity; infinity; neg_infinity; neg_infinity; 0.; 0. |] in
  let add () x y =
    b.(0) <- Float.min b.(0) x;
    b.(1) <- Float.min b.(1) y;
    b.(2) <- Float.max b.(2) x;
    b.(3) <- Float.max b.(3) y;
    b.(4) <- x;
    b.(5) <- y
  in
  let gaps = ref false in
  fold_drawn m ~move:add ~line:add
    ~cubic:(fun () c1x c1y c2x c2y x y ->
      widen b 0 b.(4) c1x c2x x;
      widen b 1 b.(5) c1y c2y y;
      add () x y)
    ~close:Fun.id
    ~gap:(fun () -> gaps := true)
    () p;
  let box =
    if b.(0) <= b.(2) then
      Some (Box2.of_pts (P2.v b.(0) b.(1)) (P2.v b.(2) b.(3)))
    else None
  in
  (box, !gaps)

let bounds p = fst (extent Affine.id p)

(* Cropping

   A box is the intersection of four half-planes, and [crop] cuts each subpath
   by them in turn. Under a map, each is the half-plane of the points the map
   takes to one side of an edge, and a cut in the path's own coordinates is the
   cut of the mapped path, since an affine map keeps lines and the order of
   their points: the points of a cut are found without mapping them, so the
   image may lie beyond the floats. A cut keeps the pieces of an open subpath
   within the half-plane as subpaths of their own. A closed subpath stays one
   closed subpath: from where it leaves the half-plane to where it comes back,
   it runs straight along the edge instead. That segment and the piece it
   replaces bound a region outside the half-plane, so the winding number of
   every point inside is unchanged, and so is the region under either fill rule.
   Cut for stroking ([cut]), a region of no area is kept: a stroke draws it. *)

type seg =
  | S_line of float * float
  | S_cubic of float * float * float * float * float * float

(* A subpath as [fold] visits it: its start, its segments in drawing order and
   whether it is closed, its closing segment not among [segs]. *)
type sub = { sx : float; sy : float; segs : seg list; closed : bool }

let subpaths p =
  let subs = ref [] and cur = ref None in
  let push closed =
    match !cur with
    | None -> ()
    | Some (sx, sy, segs) ->
        subs := { sx; sy; segs = List.rev segs; closed } :: !subs;
        cur := None
  in
  let add s =
    match !cur with
    | Some (sx, sy, segs) -> cur := Some (sx, sy, s :: segs)
    | None -> assert false (* [fold] starts every subpath with [move]. *)
  in
  fold
    ~move:(fun () x y ->
      push false;
      cur := Some (x, y, []))
    ~line:(fun () x y -> add (S_line (x, y)))
    ~cubic:(fun () a b c d x y -> add (S_cubic (a, b, c, d, x, y)))
    ~close:(fun () -> push true)
    () p;
  push false;
  List.rev !subs

(* The half-plane of the points whose value [a x + b y] is at least [c] if
   [lower], and at most [c] otherwise. [a] and [b] are a row of a map scaled by
   a power of two so that the larger is at least 1/8 and below 1/4: the value of
   a point is a float, and so is the difference of two. A row along an axis has
   its edge at [edge] on that axis, where a cut puts its points exactly. *)
type half = {
  a : float;
  b : float;
  c : float;
  lower : bool;
  edge : [ `X of float | `Y of float | `None ];
}

(* [half (ra, rb, rc) bound ~lower] is the half-plane of the points that the row
   [ra x + rb y + rc] takes to at least [bound] if [lower], at most
   otherwise. *)
let half (ra, rb, rc) bound ~lower =
  let gap = bound -. rc in
  let e =
    let top = Float.max (Float.abs ra) (Float.abs rb) in
    if top = 0. then 0 else -snd (Float.frexp top) - 2
  in
  let c =
    if Float.is_finite gap then Float.ldexp gap e
    else Float.ldexp ((0.5 *. bound) -. (0.5 *. rc)) (e + 1)
  in
  let edge =
    if rb = 0. && ra <> 0. then `X (gap /. ra)
    else if ra = 0. && rb <> 0. then `Y (gap /. rb)
    else `None
  in
  { a = Float.ldexp ra e; b = Float.ldexp rb e; c; lower; edge }

let value h x y = (h.a *. x) +. (h.b *. y)
let holds h v = if h.lower then v >= h.c else v <= h.c
let within h x y = holds h (value h x y)

let on_edge h x y =
  match h.edge with `X e -> (e, y) | `Y e -> (x, e) | `None -> (x, y)

(* [lerp u v t] is [u +. t *. (v -. u)], halving where the difference
   overflows. *)
let lerp u v t =
  let d = v -. u in
  if Float.is_finite d then u +. (t *. d)
  else 2. *. ((0.5 *. u) +. (t *. ((0.5 *. v) -. (0.5 *. u))))

(* [crossing h x0 y0 x1 y1] is where the line from [(x0, y0)] to [(x1, y1)],
   which [h]'s edge separates, crosses it: measured from the end nearer the
   edge, so that it keeps that end's precision however far the other lies. *)
let crossing h x0 y0 x1 y1 =
  let v0 = value h x0 y0 and v1 = value h x1 y1 in
  let d = v1 -. v0 in
  if Float.abs (h.c -. v0) <= Float.abs (v1 -. h.c) then
    let t = (h.c -. v0) /. d in
    on_edge h (lerp x0 x1 t) (lerp y0 y1 t)
  else
    let t = (v1 -. h.c) /. d in
    on_edge h (lerp x1 x0 t) (lerp y1 y0 t)

(* [split t a b c d] is the control values of the parts of the cubic Bézier
   coordinate [a], [b], [c], [d] before and after [t], by de Casteljau. *)
let split t a b c d =
  let lerp u v = lerp u v t in
  let ab = lerp a b and bc = lerp b c and cd = lerp c d in
  let abc = lerp ab bc and bcd = lerp bc cd in
  let m = lerp abc bcd in
  ((a, ab, abc, m), (m, bcd, cd, d))

(* [section t0 t1 a b c d] is the control values of the part of the cubic
   coordinate between [t0] and [t1]. *)
let section t0 t1 a b c d =
  let (a, b, c, d), _ = split t1 a b c d in
  snd (split (t0 /. t1) a b c d)

(* [bisect inside lo hi] is the parameter in \[[lo];[hi]\] where [inside]
   changes, from that of [lo], to within adjacent floats: bisected on the bits
   of the non-negative parameters, whose order is theirs, so that it takes at
   most 64 steps however small the parameter. *)
let bisect inside lo hi =
  let side = inside lo in
  let lo = ref (Int64.bits_of_float lo) and hi = ref (Int64.bits_of_float hi) in
  while Int64.sub !hi !lo > 1L do
    let mid = Int64.add !lo (Int64.div (Int64.sub !hi !lo) 2L) in
    if inside (Int64.float_of_bits mid) = side then lo := mid else hi := mid
  done;
  (Int64.float_of_bits !lo +. Int64.float_of_bits !hi) /. 2.

(* [crossings h a b c d] is the parameters in ]0;1[ at which the cubic
   coordinate of values [a], [b], [c], [d] under [h] enters or leaves [h], in
   increasing order. It is monotone between the roots of its derivative, so each
   such interval holds at most one crossing, found by bisection. *)
let crossings h a b c d =
  let inside t = holds h (bezier t a b c d) in
  let rec cross acc lo = function
    | [] -> List.rev acc
    | hi :: rest ->
        let acc =
          if inside lo = inside hi then acc else bisect inside lo hi :: acc
        in
        cross acc hi rest
  in
  cross [] 0. (turns a b c d @ [ 1. ])

(* [contained h s] is [true] iff every point [s] was built with is within [h],
   which then holds all of [s]. *)
let contained h s =
  let ok x y = within h x y in
  ok s.sx s.sy
  && List.for_all
       (function
         | S_line (x, y) -> ok x y
         | S_cubic (a, b, c, d, x, y) -> ok a b && ok c d && ok x y)
       s.segs

(* [degenerate x y seg] is [true] iff [seg] from [(x, y)] draws nothing: all its
   points are [(x, y)]. *)
let degenerate x y = function
  | S_line (x', y') -> x' = x && y' = y
  | S_cubic (a, b, c, d, x', y') ->
      a = x && b = y && c = x && d = y && x' = x && y' = y

(* [flat s] is [true] iff every point of [s] lies on one line through its start,
   as computed: as a region, [s] has no area. Points on an edge of the box are
   found so exactly. The two products are compared, each rounded on its own:
   ocamlopt on arm64 fuses [a *. b -. c *. d] into one multiply-add, whose
   rounding can put a point off the line through itself. The differences are
   halved and scaled by one power of two to at most 1, which rounds them as they
   round unscaled, so that the products stay floats however far the points
   lie. *)
let flat s =
  let pts =
    List.concat_map
      (function
        | S_line (x, y) -> [ (x, y) ]
        | S_cubic (a, b, c, d, x, y) -> [ (a, b); (c, d); (x, y) ])
      s.segs
  in
  match List.find_opt (fun (x, y) -> x <> s.sx || y <> s.sy) pts with
  | None -> true
  | Some (dx, dy) ->
      let half v v0 = (0.5 *. v) -. (0.5 *. v0) in
      let top =
        List.fold_left
          (fun t (x, y) ->
            Float.max t
              (Float.max (Float.abs (half x s.sx)) (Float.abs (half y s.sy))))
          0. pts
      in
      let e = -snd (Float.frexp top) in
      let d v v0 = Float.ldexp (half v v0) e in
      let ux = d dx s.sx and uy = d dy s.sy in
      List.for_all (fun (x, y) -> d x s.sx *. uy = d y s.sy *. ux) pts

(* [cut_by h ~fill s acc] adds to [acc] the subpaths of [s] within [h], latest
   first: [s]'s region within [h] as one closed subpath if [s] is closed, and
   its pieces within [h] otherwise. A cut adds no segment of zero length, a
   region ends with its close and no line back to its start, and a region with
   no area within [h] is dropped if [fill]. *)
let cut_by h ~fill s acc =
  let region = s.closed in
  if contained h s then s :: acc
  else
    let out = ref acc and start = ref None and segs = ref [] in
    (* The point the subpath being built has reached. *)
    let cx = ref 0. and cy = ref 0. in
    let finish ~closed =
      (match (!start, !segs) with
      | Some (sx, sy), _ :: _ -> (
          let segs =
            match !segs with
            | S_line (x, y) :: rest when closed && x = sx && y = sy -> rest
            | segs -> segs
          in
          let sub = { sx; sy; segs = List.rev segs; closed } in
          match segs with
          | [] -> ()
          | _ :: _ when closed && fill && flat sub -> ()
          | _ :: _ -> out := sub :: !out)
      | _ -> ());
      start := None;
      segs := []
    in
    let add seg =
      segs := seg :: !segs;
      match seg with
      | S_line (x, y) | S_cubic (_, _, _, _, x, y) ->
          cx := x;
          cy := y
    in
    (* The subpath enters [h] at [(x, y)], on its edge. *)
    let enter x y =
      match !start with
      | None ->
          start := Some (x, y);
          cx := x;
          cy := y
      | Some _ -> if x <> !cx || y <> !cy then add (S_line (x, y))
    in
    let leave () = if not region then finish ~closed:false in
    let inside = ref (within h s.sx s.sy) in
    if !inside then begin
      start := Some (s.sx, s.sy);
      cx := s.sx;
      cy := s.sy
    end;
    (* [piece seg ~enters ~ends ~whole] adds [seg], the part of a segment within
       [h], which starts at [enters], entering [h] there if the subpath was
       outside, and whose end is put on the edge if [ends]. A part that draws
       nothing is dropped, unless it is the [whole] segment. *)
    let piece seg ~enters ~ends ~whole =
      let snap x y = if ends then on_edge h x y else (x, y) in
      let seg =
        match seg with
        | S_line (x, y) ->
            let x, y = snap x y in
            S_line (x, y)
        | S_cubic (a, b, c, d, x, y) ->
            let x, y = snap x y in
            S_cubic (a, b, c, d, x, y)
      in
      if not !inside then begin
        let x, y = enters in
        enter x y
      end;
      if whole || not (degenerate !cx !cy seg) then add seg;
      inside := true
    in
    let outside () =
      if !inside then leave ();
      inside := false
    in
    let rec go x0 y0 = function
      | [] -> ()
      | (S_line (x, y) as seg) :: rest ->
          (* Where the segment crosses the edge, if it does. *)
          let cross () = crossing h x0 y0 x y in
          (match (within h x0 y0, within h x y) with
          | true, true -> piece seg ~enters:(x0, y0) ~ends:false ~whole:true
          | false, false -> outside ()
          | true, false ->
              let ex, ey = cross () in
              piece (S_line (ex, ey)) ~enters:(x0, y0) ~ends:false ~whole:false;
              outside ()
          | false, true -> piece seg ~enters:(cross ()) ~ends:false ~whole:false);
          go x y rest
      | (S_cubic (c1x, c1y, c2x, c2y, x, y) as seg) :: rest ->
          let a = value h x0 y0 and b = value h c1x c1y in
          let c = value h c2x c2y and d = value h x y in
          let all p = p a && p b && p c && p d in
          (if all (holds h) then
             piece seg ~enters:(x0, y0) ~ends:false ~whole:true
           else if all (fun v -> not (holds h v)) then outside ()
           else
             let ts = (0. :: crossings h a b c d) @ [ 1. ] in
             let rec pieces = function
               | t0 :: (t1 :: _ as rest) ->
                   let xs = section t0 t1 x0 c1x c2x x
                   and ys = section t0 t1 y0 c1y c2y y in
                   let (px0, px1, px2, px3), (py0, py1, py2, py3) = (xs, ys) in
                   let mid =
                     within h
                       (bezier 0.5 px0 px1 px2 px3)
                       (bezier 0.5 py0 py1 py2 py3)
                   in
                   (if not mid then outside ()
                    else
                      let enters =
                        if t0 = 0. then (x0, y0) else on_edge h px0 py0
                      in
                      piece
                        (S_cubic (px1, py1, px2, py2, px3, py3))
                        ~enters ~ends:(t1 < 1.) ~whole:false);
                   pieces rest
               | [ _ ] | [] -> ()
             in
             pieces ts);
          go x y rest
    in
    let segs_in =
      if s.closed then s.segs @ [ S_line (s.sx, s.sy) ] else s.segs
    in
    go s.sx s.sy segs_in;
    finish ~closed:region;
    !out

(* [maps_within m b p] is [true] iff [m] maps every point [p] was built with
   into [b], control points included, which then holds all [p] draws. *)
let maps_within (m : Affine.t) b p =
  let x0 = Box2.minx b and y0 = Box2.miny b in
  let x1 = Box2.maxx b and y1 = Box2.maxy b in
  let[@inline] inside x y =
    let x' = (m.xx *. x) +. (m.xy *. y) +. m.x0 in
    let y' = (m.yx *. x) +. (m.yy *. y) +. m.y0 in
    x0 <= x' && x' <= x1 && y0 <= y' && y' <= y1
  in
  let rec poly xs ys i =
    i >= Array.length xs
    || inside (Array.unsafe_get xs i) (Array.unsafe_get ys i)
       && poly xs ys (i + 1)
  in
  List.for_all
    (function
      | Move (x, y) | Line (x, y) -> inside x y
      | Cubic (a, b, c, d, x, y) -> inside a b && inside c d && inside x y
      | Close -> true
      | Poly { xs; ys; _ } -> poly xs ys 0)
    p

(* [cropped ~fill m b p] cuts [p] to the points [m] maps into [b], dropping
   regions left with no area if [fill]. A path with a gap is cut even when it
   lies within [b], so that the result holds only the finite points [fold]
   visits. *)
let cropped ~fill (m : Affine.t) b p =
  let fits () =
    match extent m p with
    | None, false -> true
    | Some pb, false ->
        Box2.minx b <= Box2.minx pb
        && Box2.maxx pb <= Box2.maxx b
        && Box2.miny b <= Box2.miny pb
        && Box2.maxy pb <= Box2.maxy b
    | _, true -> false
  in
  if maps_within m b p || fits () then p
  else
    let row_x = (m.xx, m.xy, m.x0) and row_y = (m.yx, m.yy, m.y0) in
    let halves =
      [
        half row_x (Box2.minx b) ~lower:true;
        half row_x (Box2.maxx b) ~lower:false;
        half row_y (Box2.miny b) ~lower:true;
        half row_y (Box2.maxy b) ~lower:false;
      ]
    in
    let cut_all subs h =
      List.rev (List.fold_left (fun acc s -> cut_by h ~fill s acc) [] subs)
    in
    let subs = List.fold_left cut_all (subpaths p) halves in
    let seg acc = function
      | S_line (x, y) -> Line (x, y) :: acc
      | S_cubic (a, b, c, d, x, y) -> Cubic (a, b, c, d, x, y) :: acc
    in
    List.fold_left
      (fun acc s ->
        let acc = List.fold_left seg (Move (s.sx, s.sy) :: acc) s.segs in
        if s.closed then Close :: acc else acc)
      [] subs

let crop ?(m = Affine.id) b p = cropped ~fill:true m b p
let cut ?(m = Affine.id) b p = cropped ~fill:false m b p

(* Comparing and formatting *)

(* A cursor reads the segments a path was built from one at a time: [next]
   writes the points of the next one in [pts] and returns its kind. *)
type cursor = { mutable ops : op list; mutable i : int; pts : float array }
type kind = K_end | K_move | K_line | K_cubic | K_close

let rec next c =
  match c.ops with
  | [] -> K_end
  | Move (x, y) :: ops ->
      c.ops <- ops;
      c.pts.(0) <- x;
      c.pts.(1) <- y;
      K_move
  | Line (x, y) :: ops ->
      c.ops <- ops;
      c.pts.(0) <- x;
      c.pts.(1) <- y;
      K_line
  | Cubic (c1x, c1y, c2x, c2y, x, y) :: ops ->
      c.ops <- ops;
      c.pts.(0) <- c1x;
      c.pts.(1) <- c1y;
      c.pts.(2) <- c2x;
      c.pts.(3) <- c2y;
      c.pts.(4) <- x;
      c.pts.(5) <- y;
      K_cubic
  | Close :: ops ->
      c.ops <- ops;
      K_close
  | Poly { xs; ys; closed } :: ops ->
      let i = c.i in
      if i < Array.length xs then begin
        c.i <- i + 1;
        c.pts.(0) <- xs.(i);
        c.pts.(1) <- ys.(i);
        if i = 0 then K_move else K_line
      end
      else begin
        c.i <- 0;
        c.ops <- ops;
        if closed then K_close else next c
      end

let rec same_pts a b n i =
  i >= n || (Float.equal a.(i) b.(i) && same_pts a b n (i + 1))

let equal p q =
  let a = { ops = List.rev p; i = 0; pts = Array.create_float 6 } in
  let b = { ops = List.rev q; i = 0; pts = Array.create_float 6 } in
  let rec loop () =
    let k = next a in
    k = next b
    &&
    match k with
    | K_end -> true
    | K_close -> loop ()
    | K_move | K_line -> same_pts a.pts b.pts 2 0 && loop ()
    | K_cubic -> same_pts a.pts b.pts 6 0 && loop ()
  in
  p == q || loop ()

let pp ppf p =
  let first = ref true in
  let sep () =
    if !first then first := false else Format.pp_print_space ppf ()
  in
  Format.pp_open_box ppf 0;
  fold_built
    ~move:(fun () x y ->
      sep ();
      Format.fprintf ppf "M %a %a" Float_print.pp x Float_print.pp y)
    ~line:(fun () x y ->
      sep ();
      Format.fprintf ppf "L %a %a" Float_print.pp x Float_print.pp y)
    ~cubic:(fun () c1x c1y c2x c2y x y ->
      sep ();
      let pp = Float_print.pp in
      Format.fprintf ppf "C %a %a %a %a %a %a" pp c1x pp c1y pp c2x pp c2y pp x
        pp y)
    ~close:(fun () ->
      sep ();
      Format.pp_print_string ppf "Z")
    () p;
  Format.pp_close_box ppf ()
