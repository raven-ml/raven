(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

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
    invalid_arg (Printf.sprintf "Path.arc: invalid radius %g" r);
  if not (Float.is_finite start && Float.is_finite sweep) then
    invalid_arg
      (Printf.sprintf "Path.arc: angles not finite (%g, %g)" start sweep);
  if Float.abs sweep > max_sweep then
    invalid_arg (Printf.sprintf "Path.arc: sweep %g exceeds 1000 turns" sweep);
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
    invalid_arg (Printf.sprintf "Path.circle: invalid radius %g" r);
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
   built point is mapped through [m], then the gap rule applies. [started] is
   whether a subpath is current and [drawn] whether it has a segment. *)
let fold_drawn (m : Affine.t) ~move ~line ~cubic ~close acc p =
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
    acc
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
  fold_drawn Affine.id ~move ~line ~cubic ~close acc p

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
      (Printf.sprintf "Path.flatten: tolerance %g not positive" tolerance);
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
    ~close acc p

(* [widen bounds k a b c d] widens the interval of [bounds] at [k] and [k + 2]
   by the extremes of the cubic Bézier coordinate with control values [a], [b],
   [c] and [d]: its values at the roots in ]0;1[ of its derivative, the
   quadratic [qa], [qb], [qc]. Without real roots the square root is NaN, and so
   are the roots. An extreme is clamped to the control values, which bound the
   curve, so rounding never leaves them. *)
let widen bounds k a b c d =
  let qa = -.a +. (3. *. b) -. (3. *. c) +. d in
  let qb = 2. *. (a -. (2. *. b) +. c) in
  let qc = b -. a in
  let lo = Float.min (Float.min a b) (Float.min c d) in
  let hi = Float.max (Float.max a b) (Float.max c d) in
  let extreme t =
    if 0. < t && t < 1. then begin
      let v = Float.min hi (Float.max lo (bezier t a b c d)) in
      bounds.(k) <- Float.min bounds.(k) v;
      bounds.(k + 2) <- Float.max bounds.(k + 2) v
    end
  in
  let disc = (qb *. qb) -. (4. *. qa *. qc) in
  let q = -0.5 *. (qb +. Float.copy_sign (Float.sqrt disc) qb) in
  extreme (q /. qa);
  extreme (qc /. q)

let bounds p =
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
  fold ~move:add ~line:add
    ~cubic:(fun () c1x c1y c2x c2y x y ->
      widen b 0 b.(4) c1x c2x x;
      widen b 1 b.(5) c1y c2y y;
      add () x y)
    ~close:Fun.id () p;
  if b.(0) <= b.(2) then
    Some (Box2.of_pts (P2.v b.(0) b.(1)) (P2.v b.(2) b.(3)))
  else None

(* Cropping

   A box is the intersection of four half-planes, and [crop] cuts each subpath
   by them in turn. A cut keeps the pieces of an open subpath within the
   half-plane as subpaths of their own. A closed subpath stays one closed
   subpath: from where it leaves the half-plane to where it comes back, it runs
   straight along the edge instead. That segment and the piece it replaces bound
   a region outside the half-plane, so the winding number of every point inside
   is unchanged, and so is the region under either fill rule. *)

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

(* The half-plane of the points whose coordinate along [axis] is at least [c] if
   [lower], and at most [c] otherwise. *)
type half = { axis : [ `X | `Y ]; c : float; lower : bool }

let within h v = if h.lower then v >= h.c else v <= h.c
let along h x y = match h.axis with `X -> x | `Y -> y
let on_edge h x y = match h.axis with `X -> (h.c, y) | `Y -> (x, h.c)

(* [split t a b c d] is the control values of the parts of the cubic Bézier
   coordinate [a], [b], [c], [d] before and after [t], by de Casteljau. *)
let split t a b c d =
  let lerp u v = u +. (t *. (v -. u)) in
  let ab = lerp a b and bc = lerp b c and cd = lerp c d in
  let abc = lerp ab bc and bcd = lerp bc cd in
  let m = lerp abc bcd in
  ((a, ab, abc, m), (m, bcd, cd, d))

(* [section t0 t1 a b c d] is the control values of the part of the cubic
   coordinate between [t0] and [t1]. *)
let section t0 t1 a b c d =
  let (a, b, c, d), _ = split t1 a b c d in
  snd (split (t0 /. t1) a b c d)

let bisections = 64

(* [crossings h a b c d] is the parameters in ]0;1[ at which the cubic
   coordinate [a], [b], [c], [d] enters or leaves [h], in increasing order. It
   is monotone between the roots of its derivative, so each such interval holds
   at most one crossing, found by bisection. *)
let crossings h a b c d =
  let d0 = b -. a and d1 = c -. b and d2 = d -. c in
  let qa = d0 -. (2. *. d1) +. d2 and qb = 2. *. (d1 -. d0) and qc = d0 in
  let disc = (qb *. qb) -. (4. *. qa *. qc) in
  let q = -0.5 *. (qb +. Float.copy_sign (Float.sqrt disc) qb) in
  let turns =
    List.filter (fun t -> 0. < t && t < 1.) [ q /. qa; qc /. q ]
    |> List.sort_uniq Float.compare
  in
  let inside t = within h (bezier t a b c d) in
  let rec cross acc lo = function
    | [] -> List.rev acc
    | hi :: rest ->
        let acc =
          if inside lo = inside hi then acc
          else begin
            let lo' = ref lo and hi' = ref hi and side = inside lo in
            for _ = 1 to bisections do
              let mid = (!lo' +. !hi') /. 2. in
              if inside mid = side then lo' := mid else hi' := mid
            done;
            ((!lo' +. !hi') /. 2.) :: acc
          end
        in
        cross acc hi rest
  in
  cross [] 0. (turns @ [ 1. ])

(* [contained h s] is [true] iff every point [s] was built with is within [h],
   which then holds all of [s]. *)
let contained h s =
  let ok x y = within h (along h x y) in
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
   found so exactly. *)
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
      let ux = dx -. s.sx and uy = dy -. s.sy in
      List.for_all
        (fun (x, y) -> ((x -. s.sx) *. uy) -. ((y -. s.sy) *. ux) = 0.)
        pts

(* [cut h ~region s acc] adds to [acc] the subpaths of [s] within [h], latest
   first: [s]'s region within [h] as one closed subpath if [region], and its
   pieces within [h] otherwise. A cut adds no segment of zero length, a region
   ends with its close and no line back to its start, and a region with no area
   within [h] is dropped. *)
let cut h ~region s acc =
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
          | _ :: _ when closed && flat sub -> ()
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
    let inside = ref (within h (along h s.sx s.sy)) in
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
          let a = along h x0 y0 and b = along h x y in
          (* Where the segment crosses the edge, if it does. *)
          let cross () =
            let t = (h.c -. a) /. (b -. a) in
            on_edge h (x0 +. (t *. (x -. x0))) (y0 +. (t *. (y -. y0)))
          in
          (match (within h a, within h b) with
          | true, true -> piece seg ~enters:(x0, y0) ~ends:false ~whole:true
          | false, false -> outside ()
          | true, false ->
              let ex, ey = cross () in
              piece (S_line (ex, ey)) ~enters:(x0, y0) ~ends:false ~whole:false;
              outside ()
          | false, true -> piece seg ~enters:(cross ()) ~ends:false ~whole:false);
          go x y rest
      | (S_cubic (c1x, c1y, c2x, c2y, x, y) as seg) :: rest ->
          let a = along h x0 y0 and b = along h c1x c1y in
          let c = along h c2x c2y and d = along h x y in
          let all p = p a && p b && p c && p d in
          (if all (within h) then
             piece seg ~enters:(x0, y0) ~ends:false ~whole:true
           else if all (fun v -> not (within h v)) then outside ()
           else
             let ts = (0. :: crossings h a b c d) @ [ 1. ] in
             let rec pieces = function
               | t0 :: (t1 :: _ as rest) ->
                   let xs = section t0 t1 x0 c1x c2x x
                   and ys = section t0 t1 y0 c1y c2y y in
                   let (px0, px1, px2, px3), (py0, py1, py2, py3) = (xs, ys) in
                   let mid =
                     along h
                       (bezier 0.5 px0 px1 px2 px3)
                       (bezier 0.5 py0 py1 py2 py3)
                   in
                   (if not (within h mid) then outside ()
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

let crop b p =
  match bounds p with
  | None -> p
  | Some pb
    when Box2.minx b <= Box2.minx pb
         && Box2.maxx pb <= Box2.maxx b
         && Box2.miny b <= Box2.miny pb
         && Box2.maxy pb <= Box2.maxy b ->
      p
  | Some _ ->
      let halves =
        [
          { axis = `X; c = Box2.minx b; lower = true };
          { axis = `X; c = Box2.maxx b; lower = false };
          { axis = `Y; c = Box2.miny b; lower = true };
          { axis = `Y; c = Box2.maxy b; lower = false };
        ]
      in
      let cut_all subs h =
        List.rev
          (List.fold_left (fun acc s -> cut h ~region:s.closed s acc) [] subs)
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
      Format.fprintf ppf "M %g %g" x y)
    ~line:(fun () x y ->
      sep ();
      Format.fprintf ppf "L %g %g" x y)
    ~cubic:(fun () c1x c1y c2x c2y x y ->
      sep ();
      Format.fprintf ppf "C %g %g %g %g %g %g" c1x c1y c2x c2y x y)
    ~close:(fun () ->
      sep ();
      Format.pp_print_string ppf "Z")
    () p;
  Format.pp_close_box ppf ()
