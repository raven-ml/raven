(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_gg

(* Strokes as coverage. A subpath is stroked in its own coordinates, where the
   pen is round, as the polygon of its outline: along the left of the subpath
   with a join at each corner, around its end cap, back along its right, and
   around its start cap, or, for a closed subpath, one loop along each side in
   opposite directions. A join on the inner side of a corner goes through the
   corner's point, so that the outline winds the same way around everything it
   covers, and the nonzero rule covers it exactly once. Its vertices are mapped
   to the device as they are made. *)

type t = {
  cover : Cover.t;
  m : Affine.t;  (** From the stroke's coordinates to the device. *)
  hw : float;  (** Half the line width. *)
  cap : Stroke.cap;
  join : Stroke.join;
  miter_limit : float;
  tol : float;
      (** The flattening tolerance, a tenth of a pixel, in the stroke's
          coordinates. *)
  step : float;  (** The largest angle of an arc's chord within [tol]. *)
  wx0 : float;
      (** The window grown by the reach of the pen: what lies outside it paints
          nothing visible. *)
  wy0 : float;
  wx1 : float;
  wy1 : float;
  cur : float array;  (** The outline's first and last device points. *)
}

let create cover (clip : Surface.clip) m s =
  let hw = 0.5 *. Stroke.width s in
  let scale = Affine.stretch m in
  let reach = (Stroke.reach s *. scale) +. 1. in
  let r = hw *. scale in
  {
    cover;
    m;
    hw;
    cap = Stroke.cap s;
    join = Stroke.join s;
    miter_limit = Stroke.miter_limit s;
    tol = 0.1 /. scale;
    step =
      (if r <= 0.1 then Float.pi /. 2. else 2. *. Float.acos (1. -. (0.1 /. r)));
    wx0 = float clip.x0 -. reach;
    wy0 = float clip.y0 -. reach;
    wx1 = float clip.x1 +. reach;
    wy1 = float clip.y1 +. reach;
    cur = Array.make 4 0.;
  }

let[@inline] dev_x p x y = (p.m.xx *. x) +. (p.m.xy *. y) +. p.m.x0
let[@inline] dev_y p x y = (p.m.yx *. x) +. (p.m.yy *. y) +. p.m.y0

(* The outline *)

let[@inline] move_to p x y =
  let dx = dev_x p x y and dy = dev_y p x y in
  p.cur.(0) <- dx;
  p.cur.(1) <- dy;
  p.cur.(2) <- dx;
  p.cur.(3) <- dy

let[@inline] line_to p x y =
  let dx = dev_x p x y and dy = dev_y p x y in
  Cover.line p.cover (Array.unsafe_get p.cur 2) (Array.unsafe_get p.cur 3) dx dy;
  Array.unsafe_set p.cur 2 dx;
  Array.unsafe_set p.cur 3 dy

let close p = Cover.line p.cover p.cur.(2) p.cur.(3) p.cur.(0) p.cur.(1)

(* [arc p x y nx ny sweep] continues the outline along the arc about [(x, y)]
   from [(x + nx, y + ny)], turning by [sweep] radians. *)
let[@inline] arc p x y nx ny sweep =
  let n = Int.max 1 (int_of_float (Float.ceil (Float.abs sweep /. p.step))) in
  let da = sweep /. float n in
  let c = Float.cos da and s = Float.sin da in
  let nx = ref nx and ny = ref ny in
  for _ = 1 to n do
    let rx = (!nx *. c) -. (!ny *. s) and ry = (!nx *. s) +. (!ny *. c) in
    nx := rx;
    ny := ry;
    line_to p (x +. rx) (y +. ry)
  done

(* [join p x y ax ay bx by] continues the outline, on the left of a line
   arriving at [(x, y)] along the unit vector [(ax, ay)], to the left of one
   leaving along [(bx, by)]. *)
let[@inline] join p x y ax ay bx by =
  let hw = p.hw in
  let nax = -.ay *. hw and nay = ax *. hw in
  let nbx = -.by *. hw and nby = bx *. hw in
  let cross = (ax *. by) -. (ay *. bx) and dot = (ax *. bx) +. (ay *. by) in
  if dot > 0. && Float.abs cross *. hw < p.tol then
    line_to p (x +. nbx) (y +. nby)
  else if cross > 0. then begin
    (* The inner side of the corner. *)
    line_to p x y;
    line_to p (x +. nbx) (y +. nby)
  end
  else
    match p.join with
    | `Round ->
        (* A reversal turns about the side of the arriving line's end. *)
        let sweep = if cross = 0. then -.Float.pi else Float.atan2 cross dot in
        arc p x y nax nay sweep
    | `Miter
      when dot > -1. && 1. /. Float.sqrt ((1. +. dot) /. 2.) <= p.miter_limit ->
        let k = 1. /. (1. +. dot) in
        line_to p (x +. ((nax +. nbx) *. k)) (y +. ((nay +. nby) *. k));
        line_to p (x +. nbx) (y +. nby)
    | `Miter | `Bevel -> line_to p (x +. nbx) (y +. nby)

(* [cap p x y dx dy] continues the outline from the left of a line ending at
   [(x, y)] along the unit vector [(dx, dy)] around its end to its right. *)
let cap p x y dx dy =
  let nx = -.dy *. p.hw and ny = dx *. p.hw in
  match p.cap with
  | `Butt -> line_to p (x -. nx) (y -. ny)
  | `Round -> arc p x y nx ny (-.Float.pi)
  | `Square ->
      let ex = dx *. p.hw and ey = dy *. p.hw in
      line_to p (x +. nx +. ex) (y +. ny +. ey);
      line_to p (x -. nx +. ex) (y -. ny +. ey);
      line_to p (x -. nx) (y -. ny)

(* [polyline p xs ys n ~closed ~dx ~dy] strokes the subpath of the [n] points of
   [xs] and [ys], no two consecutive ones equal. A single point has the
   direction [(dx, dy)], or none if both are [0.]. *)
let polyline p xs ys n ~closed ~dx ~dy =
  if n = 1 then begin
    let x = xs.(0) and y = ys.(0) in
    match p.cap with
    | `Butt -> ()
    | `Round ->
        move_to p (x +. p.hw) y;
        arc p x y p.hw 0. (2. *. Float.pi);
        close p
    | `Square ->
        if dx <> 0. || dy <> 0. then begin
          move_to p (x -. (dy *. p.hw)) (y +. (dx *. p.hw));
          cap p x y dx dy;
          cap p x y (-.dx) (-.dy);
          close p
        end
  end
  else begin
    let segs = if closed then n else n - 1 in
    let ux = Array.make segs 0. and uy = Array.make segs 0. in
    for i = 0 to segs - 1 do
      let j = if i = n - 1 then 0 else i + 1 in
      let ddx = xs.(j) -. xs.(i) and ddy = ys.(j) -. ys.(i) in
      let l = Float.hypot ddx ddy in
      ux.(i) <- ddx /. l;
      uy.(i) <- ddy /. l
    done;
    let hw = p.hw in
    (* Forward along the left. *)
    move_to p (xs.(0) -. (uy.(0) *. hw)) (ys.(0) +. (ux.(0) *. hw));
    for i = 1 to segs - 1 do
      line_to p (xs.(i) -. (uy.(i - 1) *. hw)) (ys.(i) +. (ux.(i - 1) *. hw));
      join p xs.(i) ys.(i) ux.(i - 1) uy.(i - 1) ux.(i) uy.(i)
    done;
    let l = segs - 1 in
    if closed then begin
      line_to p (xs.(0) -. (uy.(l) *. hw)) (ys.(0) +. (ux.(l) *. hw));
      join p xs.(0) ys.(0) ux.(l) uy.(l) ux.(0) uy.(0);
      close p;
      (* Backward along the right, the left of the way back. *)
      move_to p (xs.(0) +. (uy.(l) *. hw)) (ys.(0) -. (ux.(l) *. hw))
    end
    else begin
      line_to p (xs.(n - 1) -. (uy.(l) *. hw)) (ys.(n - 1) +. (ux.(l) *. hw));
      cap p xs.(n - 1) ys.(n - 1) ux.(l) uy.(l)
    end;
    for i = l downto 1 do
      line_to p (xs.(i) +. (uy.(i) *. hw)) (ys.(i) -. (ux.(i) *. hw));
      join p xs.(i) ys.(i) (-.ux.(i)) (-.uy.(i)) (-.ux.(i - 1)) (-.uy.(i - 1))
    done;
    if closed then begin
      line_to p (xs.(0) +. (uy.(0) *. hw)) (ys.(0) -. (ux.(0) *. hw));
      join p xs.(0) ys.(0) (-.ux.(0)) (-.uy.(0)) (-.ux.(l)) (-.uy.(l))
    end
    else begin
      line_to p (xs.(0) +. (uy.(0) *. hw)) (ys.(0) -. (ux.(0) *. hw));
      cap p xs.(0) ys.(0) (-.ux.(0)) (-.uy.(0))
    end;
    close p
  end

(* Points *)

(* A growable buffer of points. *)
type points = {
  mutable xs : float array;
  mutable ys : float array;
  mutable n : int;
}

let points () = { xs = Array.make 64 0.; ys = Array.make 64 0.; n = 0 }

let grow b =
  b.xs <- Array.append b.xs (Array.make (Array.length b.xs) 0.);
  b.ys <- Array.append b.ys (Array.make (Array.length b.ys) 0.)

let[@inline] push b x y =
  if b.n = Array.length b.xs then grow b;
  Array.unsafe_set b.xs b.n x;
  Array.unsafe_set b.ys b.n y;
  b.n <- b.n + 1

(* [far m dx dy] is [true] iff [m] maps the vector [(dx, dy)] more than a
   twentieth of a pixel along an axis. *)
let[@inline] far (m : Affine.t) dx dy =
  Float.abs ((m.xx *. dx) +. (m.xy *. dy)) > 0.05
  || Float.abs ((m.yx *. dx) +. (m.yy *. dy)) > 0.05

(* [strip p src kept ~closed] sets [kept] to the points of [src] without those
   within a twentieth of a pixel of the point kept before them, and, for a
   closed subpath, of its first point. A closed subpath that would keep fewer
   than three points keeps all of them but exact repeats instead, since its
   outline has no caps to stand for it. It is the direction of the first segment
   of [src] of positive length, or [(0., 0.)] if there is none. *)
let strip p src kept ~closed =
  let m = p.m in
  kept.n <- 0;
  let ux = ref 0. and uy = ref 0. in
  for i = 0 to src.n - 1 do
    let x = Array.unsafe_get src.xs i and y = Array.unsafe_get src.ys i in
    if kept.n = 0 then push kept x y
    else begin
      let dx = x -. kept.xs.(kept.n - 1) and dy = y -. kept.ys.(kept.n - 1) in
      if !ux = 0. && !uy = 0. && (dx <> 0. || dy <> 0.) then begin
        let d = Float.hypot dx dy in
        ux := dx /. d;
        uy := dy /. d
      end;
      if far m dx dy then push kept x y
    end
  done;
  if closed && kept.n > 1 then begin
    let l = kept.n - 1 in
    if not (far m (kept.xs.(l) -. kept.xs.(0)) (kept.ys.(l) -. kept.ys.(0)))
    then kept.n <- l
  end;
  if closed && kept.n < 3 then begin
    kept.n <- 0;
    for i = 0 to src.n - 1 do
      let x = Array.unsafe_get src.xs i and y = Array.unsafe_get src.ys i in
      if kept.n = 0 || x <> kept.xs.(kept.n - 1) || y <> kept.ys.(kept.n - 1)
      then push kept x y
    done;
    let l = kept.n - 1 in
    if l > 0 && kept.xs.(l) = kept.xs.(0) && kept.ys.(l) = kept.ys.(0) then
      kept.n <- l
  end;
  (!ux, !uy)

(* Dashes *)

type dashes = {
  pattern : float array;  (** Of even length. *)
  period : float;
  offset : float;
  mutable index : int;  (** The pattern's current length: a dash if even. *)
  mutable rest : float;  (** What remains of it. *)
}

(* [dashes p s] is the dashes of [s] stroked by [p], or [None] for a solid
   stroke. A pattern whose period is below a hundredth of [p]'s tolerance, a
   thousandth of a pixel, is solid: pixels show nothing of its dashes, and
   walking them would take a step for each. *)
let dashes p s =
  match Stroke.dash s with
  | [] -> None
  | d ->
      let d = Array.of_list d in
      let pattern = if Array.length d mod 2 = 1 then Array.append d d else d in
      let period = Array.fold_left ( +. ) 0. pattern in
      if period < 0.01 *. p.tol then None
      else
        Some
          {
            pattern;
            period;
            offset = Stroke.dash_offset s;
            index = 0;
            rest = 0.;
          }

let next d =
  d.index <- (if d.index + 1 = Array.length d.pattern then 0 else d.index + 1);
  d.rest <- d.pattern.(d.index)

(* [advance d len] moves [d] [len] along the pattern. Arriving where a length of
   the pattern ends, it moves to the next, so that a dash of zero length there
   is still to be drawn. *)
let advance d len =
  if len > 0. then
    if len < d.rest then d.rest <- d.rest -. len
    else begin
      let len = ref (Float.rem (len -. d.rest) d.period) in
      next d;
      while !len > 0. && !len >= d.rest do
        len := !len -. d.rest;
        next d
      done;
      d.rest <- d.rest -. !len
    end

let restart d =
  d.index <- 0;
  d.rest <- d.pattern.(0);
  advance d d.offset

let is_dash d = d.index land 1 = 0

(* Subpaths *)

type subpaths = {
  pen : t;
  dashes : dashes option;
  raw : points;  (** The subpath being flattened. *)
  piece : points;  (** The dash being cut. *)
  kept : points;
  last : float array;  (** The direction of the last segment dashed. *)
}

let stroke_points s pts ~closed ~dx ~dy =
  let p = s.pen in
  let ddx, ddy = strip p pts s.kept ~closed in
  let dx, dy = if ddx = 0. && ddy = 0. then (dx, dy) else (ddx, ddy) in
  polyline p s.kept.xs s.kept.ys s.kept.n
    ~closed:(closed && s.kept.n > 1)
    ~dx ~dy

let flush_piece s ~dx ~dy =
  if s.piece.n > 0 then begin
    stroke_points s s.piece ~closed:false ~dx ~dy;
    s.piece.n <- 0
  end

(* [dash_segment s d ax ay bx by] cuts into dashes the segment from [(ax, ay)]
   to [(bx, by)], walking only the part that the grown window reaches. *)
let dash_segment s d ax ay bx by =
  let p = s.pen in
  let len = Float.hypot (bx -. ax) (by -. ay) in
  if len > 0. then begin
    let ux = (bx -. ax) /. len and uy = (by -. ay) /. len in
    s.last.(0) <- ux;
    s.last.(1) <- uy;
    (* The parameters of the segment within the grown window, by Liang and
       Barsky's clipping of the device segment. *)
    let sx = dev_x p ax ay and sy = dev_y p ax ay in
    let ex = dev_x p bx by -. sx and ey = dev_y p bx by -. sy in
    let t0 = ref 0. and t1 = ref 1. in
    let cut q r =
      if q = 0. then (if r < 0. then t1 := -1.)
      else
        let t = r /. q in
        if q < 0. then (if t > !t0 then t0 := t) else if t < !t1 then t1 := t
    in
    cut (-.ex) (sx -. p.wx0);
    cut ex (p.wx1 -. sx);
    cut (-.ey) (sy -. p.wy0);
    cut ey (p.wy1 -. sy);
    if !t0 >= !t1 then begin
      flush_piece s ~dx:ux ~dy:uy;
      advance d len
    end
    else begin
      let start = !t0 *. len and stop = !t1 *. len in
      if start > 0. then begin
        flush_piece s ~dx:ux ~dy:uy;
        advance d start
      end;
      let pos = ref start in
      let continue = ref true in
      while !continue do
        if is_dash d && s.piece.n = 0 then
          push s.piece (ax +. (ux *. !pos)) (ay +. (uy *. !pos));
        if !pos +. d.rest <= stop then begin
          pos := !pos +. d.rest;
          if is_dash d then begin
            push s.piece (ax +. (ux *. !pos)) (ay +. (uy *. !pos));
            flush_piece s ~dx:ux ~dy:uy
          end;
          next d
        end
        else begin
          d.rest <- d.rest -. (stop -. !pos);
          if is_dash d then
            push s.piece (ax +. (ux *. stop)) (ay +. (uy *. stop));
          continue := false
        end
      done;
      if stop < len then begin
        flush_piece s ~dx:ux ~dy:uy;
        advance d (len -. stop)
      end
    end
  end

(* [subpath s ~closed] strokes the flattened subpath in [s.raw]. *)
let subpath s ~closed =
  let raw = s.raw in
  if raw.n > 0 then begin
    let moves = ref false in
    for i = 1 to raw.n - 1 do
      if raw.xs.(i) <> raw.xs.(0) || raw.ys.(i) <> raw.ys.(0) then moves := true
    done;
    match s.dashes with
    | Some d when !moves ->
        restart d;
        for i = 0 to raw.n - 2 do
          dash_segment s d raw.xs.(i) raw.ys.(i) raw.xs.(i + 1) raw.ys.(i + 1)
        done;
        let l = raw.n - 1 in
        if closed then
          dash_segment s d raw.xs.(l) raw.ys.(l) raw.xs.(0) raw.ys.(0);
        flush_piece s ~dx:s.last.(0) ~dy:s.last.(1)
    | _ -> stroke_points s raw ~closed:(closed && !moves) ~dx:0. ~dy:0.
  end;
  raw.n <- 0

(* [stroke cover clip m ~pen s path] deposits in [cover] the outline of [path]
   stroked with [s], its lengths multiplied by [pen], in the coordinates that
   [m] maps to the device. The path is stroked in coordinates divided by [pen],
   where [s] is the pen as given, so that a pen a stamp's tiny scale makes huge
   stays finite. *)
let stroke cover clip m ~pen:k s path =
  let m, shrink =
    if k = 1. then (m, Affine.id)
    else (Affine.(m * scale k k), Affine.scale (1. /. k) (1. /. k))
  in
  let pen = create cover clip m s in
  (* Transforms within a stamp that magnify as much as its scale shrinks map the
     pen beyond the range of floats, where it draws nothing. *)
  if Float.is_finite pen.tol && pen.tol > 0. then begin
    let s =
      {
        pen;
        dashes = dashes pen s;
        raw = points ();
        piece = points ();
        kept = points ();
        last = [| 0.; 0. |];
      }
    in
    Path.flatten ~tolerance:pen.tol shrink
      ~move:(fun () x y ->
        subpath s ~closed:false;
        push s.raw x y)
      ~line:(fun () x y -> push s.raw x y)
      ~close:(fun () -> subpath s ~closed:true)
      () path;
    subpath s ~closed:false
  end
