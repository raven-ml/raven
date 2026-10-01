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
