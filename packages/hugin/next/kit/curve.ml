(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_next_gg

(* Curves drawn with lines only, as polylines, and curves drawn with cubics. *)
type lines = Linear | Step_after | Step_before | Step_mid
type smooth = Monotone_x | Monotone_y | Natural | Catmull_rom | Basis
type t = Lines of lines | Smooth of smooth

let linear = Lines Linear
let step_after = Lines Step_after
let step_before = Lines Step_before
let step_mid = Lines Step_mid
let monotone_x = Smooth Monotone_x
let monotone_y = Smooth Monotone_y
let natural = Smooth Natural
let catmull_rom = Smooth Catmull_rom
let basis = Smooth Basis

(* Runs *)

(* [fold_runs missing n f acc] folds [f] over the runs of at least two points
   among the points [0] to [n - 1], as [f first length acc]. *)
let fold_runs missing n f acc =
  let rec skip i acc =
    if i >= n then acc
    else if missing i then skip (i + 1) acc
    else run i (i + 1) acc
  and run first i acc =
    if i < n && not (missing i) then run first (i + 1) acc
    else
      let acc = if i - first >= 2 then f first (i - first) acc else acc in
      skip i acc
  in
  skip 0 acc

(* Curves of lines *)

(* [vertices c xs ys i len] is the points of the polyline that [c] draws through
   the [len] points from [i]. *)
let vertices c xs ys i len =
  let between =
    match c with Linear -> 0 | Step_after | Step_before -> 1 | Step_mid -> 2
  in
  let m = 1 + ((len - 1) * (between + 1)) in
  let vx = Array.create_float m and vy = Array.create_float m in
  vx.(0) <- xs.(i);
  vy.(0) <- ys.(i);
  for k = 0 to len - 2 do
    let x0 = xs.(i + k) and y0 = ys.(i + k) in
    let x1 = xs.(i + k + 1) and y1 = ys.(i + k + 1) in
    let j = 1 + (k * (between + 1)) in
    (match c with
    | Linear -> ()
    | Step_after ->
        vx.(j) <- x1;
        vy.(j) <- y0
    | Step_before ->
        vx.(j) <- x0;
        vy.(j) <- y1
    | Step_mid ->
        let xm = (0.5 *. x0) +. (0.5 *. x1) in
        vx.(j) <- xm;
        vy.(j) <- y0;
        vx.(j + 1) <- xm;
        vy.(j + 1) <- y1);
    vx.(j + between) <- x1;
    vy.(j + between) <- y1
  done;
  (vx, vy)

(* Smooth curves *)

(* Each smooth curve adds to a path, from the run's first point, a line or a
   cubic per segment, in order. *)
let line x y p = Path.line_to (P2.v x y) p

let cubic ax ay bx by x y p =
  Path.cubic_to (P2.v ax ay) (P2.v bx by) (P2.v x y) p

(* [monotone_segments ~swap px py i len p] is [p] with the Steffen interpolant
   of [py] as a function of [px] through the [len] points from [i], its segments
   mapped back to x and y by exchanging the coordinates if [swap]. *)
let monotone_segments ~swap px py i len p =
  let line x y = if swap then line y x else line x y in
  let cubic ax ay bx by x y =
    if swap then cubic ay ax by bx y x else cubic ax ay bx by x y
  in
  let x k = px.(i + k) and y k = py.(i + k) in
  let sign v = if v > 0. then 1. else if v < 0. then -1. else 0. in
  let dir k =
    if x (k + 1) > x k then 1 else if x (k + 1) < x k then -1 else 0
  in
  (* The slope at an end of a piece from its end segment, of width [h] and slope
     [s], and the next one inwards, [h'] and [s']. *)
  let end_slope h s h' s' =
    let p = (s *. (1. +. (h /. (h +. h')))) -. (s' *. h /. (h +. h')) in
    if p *. s <= 0. then 0.
    else if Float.abs p > 2. *. Float.abs s then 2. *. s
    else p
  in
  (* The piece from point [a] to point [b], [b > a + 1]. *)
  let piece a b p =
    let m = b - a in
    let h = Array.init m (fun k -> x (a + k + 1) -. x (a + k)) in
    let s = Array.init m (fun k -> (y (a + k + 1) -. y (a + k)) /. h.(k)) in
    let t = Array.create_float (m + 1) in
    t.(0) <- end_slope h.(0) s.(0) h.(1) s.(1);
    t.(m) <- end_slope h.(m - 1) s.(m - 1) h.(m - 2) s.(m - 2);
    for k = 1 to m - 1 do
      let p =
        ((s.(k - 1) *. h.(k)) +. (s.(k) *. h.(k - 1))) /. (h.(k - 1) +. h.(k))
      in
      t.(k) <-
        (sign s.(k - 1) +. sign s.(k))
        *. Float.min
             (Float.min (Float.abs s.(k - 1)) (Float.abs s.(k)))
             (0.5 *. Float.abs p)
    done;
    let p = ref p in
    for k = 0 to m - 1 do
      let x0 = x (a + k) and y0 = y (a + k) in
      let x1 = x (a + k + 1) and y1 = y (a + k + 1) in
      let dx = h.(k) /. 3. in
      p :=
        cubic (x0 +. dx)
          (y0 +. (dx *. t.(k)))
          (x1 -. dx)
          (y1 -. (dx *. t.(k + 1)))
          x1 y1 !p
    done;
    !p
  in
  (* Pieces are maximal runs of segments of one direction; a segment between
     equal x is a line, and so is a piece of one segment. *)
  let rec pieces a p =
    if a >= len - 1 then p
    else
      let d = dir a in
      if d = 0 then pieces (a + 1) (line (x (a + 1)) (y (a + 1)) p)
      else
        let rec stop b = if b < len - 1 && dir b = d then stop (b + 1) else b in
        let b = stop (a + 1) in
        pieces b (if b = a + 1 then line (x b) (y b) p else piece a b p)
  in
  pieces 0 p

(* [natural_controls v] is the first and the second control points of each
   segment of the natural cubic spline through [v], [Array.length v >= 3],
   solved as a tridiagonal system. *)
let natural_controls v =
  let n = Array.length v - 1 in
  let a = Array.make n 1.
  and b = Array.make n 4.
  and r = Array.create_float n in
  a.(0) <- 0.;
  b.(0) <- 2.;
  r.(0) <- v.(0) +. (2. *. v.(1));
  for i = 1 to n - 2 do
    r.(i) <- (4. *. v.(i)) +. (2. *. v.(i + 1))
  done;
  a.(n - 1) <- 2.;
  b.(n - 1) <- 7.;
  r.(n - 1) <- (8. *. v.(n - 1)) +. v.(n);
  for i = 1 to n - 1 do
    let m = a.(i) /. b.(i - 1) in
    b.(i) <- b.(i) -. m;
    r.(i) <- r.(i) -. (m *. r.(i - 1))
  done;
  a.(n - 1) <- r.(n - 1) /. b.(n - 1);
  for i = n - 2 downto 0 do
    a.(i) <- (r.(i) -. a.(i + 1)) /. b.(i)
  done;
  b.(n - 1) <- (v.(n) +. a.(n - 1)) /. 2.;
  for i = 0 to n - 2 do
    b.(i) <- (2. *. v.(i + 1)) -. a.(i + 1)
  done;
  (a, b)

let natural_segments xs ys i len p =
  if len = 2 then line xs.(i + 1) ys.(i + 1) p
  else
    let ax, bx = natural_controls (Array.sub xs i len) in
    let ay, by = natural_controls (Array.sub ys i len) in
    let p = ref p in
    for k = 0 to len - 2 do
      p := cubic ax.(k) ay.(k) bx.(k) by.(k) xs.(i + k + 1) ys.(i + k + 1) !p
    done;
    !p

(* A centripetal Catmull–Rom segment from [p1] to [p2] puts its control points
   by the square roots of the distances between [p0], [p1], [p2] and [p3]. A
   missing neighbour, at an end, counts as at distance [0.], which puts the
   control point on the end point. Once consecutive repeats are removed, every
   other distance is positive. *)
let catmull_rom_segments xs ys i len p =
  (* The run without consecutive repeats. *)
  let px = Array.create_float len and py = Array.create_float len in
  let m = ref 0 in
  for k = i to i + len - 1 do
    if !m = 0 || xs.(k) <> px.(!m - 1) || ys.(k) <> py.(!m - 1) then begin
      px.(!m) <- xs.(k);
      py.(!m) <- ys.(k);
      incr m
    end
  done;
  let m = !m in
  if m = 1 then (
    let p = ref p in
    for _ = 2 to len do
      p := line px.(0) py.(0) !p
    done;
    !p)
  else if m = 2 then line px.(1) py.(1) p
  else
    let d k = Float.hypot (px.(k + 1) -. px.(k)) (py.(k + 1) -. py.(k)) in
    let p = ref p in
    for k = 0 to m - 2 do
      let l12_2a = d k in
      let l12_a = Float.sqrt l12_2a in
      let x1 = px.(k) and y1 = py.(k) and x2 = px.(k + 1) and y2 = py.(k + 1) in
      let c1x, c1y =
        if k > 0 then
          let l01_2a = d (k - 1) in
          let l01_a = Float.sqrt l01_2a in
          let a = (2. *. l01_2a) +. (3. *. l01_a *. l12_a) +. l12_2a in
          let n = 3. *. l01_a *. (l01_a +. l12_a) in
          ( ((x1 *. a) -. (px.(k - 1) *. l12_2a) +. (x2 *. l01_2a)) /. n,
            ((y1 *. a) -. (py.(k - 1) *. l12_2a) +. (y2 *. l01_2a)) /. n )
        else (x1, y1)
      in
      let c2x, c2y =
        if k < m - 2 then
          let l23_2a = d (k + 1) in
          let l23_a = Float.sqrt l23_2a in
          let b = (2. *. l23_2a) +. (3. *. l23_a *. l12_a) +. l12_2a in
          let n = 3. *. l23_a *. (l23_a +. l12_a) in
          ( ((x2 *. b) +. (x1 *. l23_2a) -. (px.(k + 2) *. l12_2a)) /. n,
            ((y2 *. b) +. (y1 *. l23_2a) -. (py.(k + 2) *. l12_2a)) /. n )
        else (x2, y2)
      in
      p := cubic c1x c1y c2x c2y x2 y2 !p
    done;
    !p

(* The uniform cubic B-spline with the end points counted three times: a line to
   the first knot, a cubic per interior point and per end, and a line to the
   last point. *)
let basis_segments xs ys i len p =
  if len = 2 then line xs.(i + 1) ys.(i + 1) p
  else
    let x k = xs.(i + Int.min k (len - 1))
    and y k = ys.(i + Int.min k (len - 1)) in
    let p =
      ref (line (((5. *. x 0) +. x 1) /. 6.) (((5. *. y 0) +. y 1) /. 6.) p)
    in
    (* The cubic from the knot of points [k - 2], [k - 1] and [k]; the last one
       repeats the last point as [k]. *)
    for k = 2 to len do
      let x0 = x (k - 2)
      and y0 = y (k - 2)
      and x1 = x (k - 1)
      and y1 = y (k - 1) in
      p :=
        cubic
          (((2. *. x0) +. x1) /. 3.)
          (((2. *. y0) +. y1) /. 3.)
          ((x0 +. (2. *. x1)) /. 3.)
          ((y0 +. (2. *. y1)) /. 3.)
          ((x0 +. (4. *. x1) +. x k) /. 6.)
          ((y0 +. (4. *. y1) +. y k) /. 6.)
          !p
    done;
    line (x (len - 1)) (y (len - 1)) !p

let segments c xs ys i len p =
  match c with
  | Monotone_x -> monotone_segments ~swap:false xs ys i len p
  | Monotone_y -> monotone_segments ~swap:true ys xs i len p
  | Natural -> natural_segments xs ys i len p
  | Catmull_rom -> catmull_rom_segments xs ys i len p
  | Basis -> basis_segments xs ys i len p

(* Drawing *)

let finite x = Float.is_finite x

let path c xs ys =
  let n = Array.length xs in
  if Array.length ys <> n then
    invalid_arg
      (Printf.sprintf "Curve.path: %d xs but %d ys" n (Array.length ys));
  let missing i = not (finite xs.(i) && finite ys.(i)) in
  fold_runs missing n
    (fun i len p ->
      match c with
      | Lines c ->
          let vx, vy = vertices c xs ys i len in
          Path.append (Path.polyline vx vy) p
      | Smooth c ->
          Path.move_to (P2.v xs.(i) ys.(i)) p |> segments c xs ys i len)
    Path.empty

(* Comparing and formatting *)

let equal (c : t) c' = c = c'

let pp ppf c =
  Format.pp_print_string ppf
    (match c with
    | Lines Linear -> "linear"
    | Lines Step_after -> "step_after"
    | Lines Step_before -> "step_before"
    | Lines Step_mid -> "step_mid"
    | Smooth Monotone_x -> "monotone_x"
    | Smooth Monotone_y -> "monotone_y"
    | Smooth Natural -> "natural"
    | Smooth Catmull_rom -> "catmull_rom"
    | Smooth Basis -> "basis")
