(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_gg
open Hugin_vg_instances

(* Strokes as coverage. A subpath is stroked in its own coordinates, where the
   pen is round, as the polygon of its outline: along the left of the subpath
   with a join at each corner, around its end cap, back along its right, and
   around its start cap, or, for a closed subpath, one loop along each side in
   opposite directions. A join on the inner side of a corner goes through the
   corner's point, so that the outline winds the same way around everything it
   covers, and the nonzero rule covers it exactly once. Every subpath's outline
   winds that way, so that where the outlines of two subpaths overlap their
   windings add instead of cancelling. Its vertices are mapped to the device as
   they are made. *)

type t = {
  cover : Cover.t;
  m : Affine.t;  (** From the stroke's coordinates to the device. *)
  inv : Affine.t;
      (** Its inverse; strokes are drawn only under maps that have one. *)
  hw : float;  (** Half the line width. *)
  cap : Stroke.cap;
  join : Stroke.join;
  miter_limit : float;
  tol : float;
      (** The flattening tolerance, a tenth of a pixel, in the stroke's
          coordinates. *)
  step : float;  (** The largest angle of an arc's chord within [tol]. *)
  reach : float;  (** How far the pen paints from the path, in pixels. *)
  pt : float array;  (** A point [crossing] finds. *)
  wx0 : float;
      (** The window grown by the reach of the pen: what lies outside it paints
          nothing visible. *)
  wy0 : float;
  wx1 : float;
  wy1 : float;
  page : float array;
      (** The window itself, [[|x0; y0; x1; y1|]]: where a dash's ink must lie
          to show. *)
  cur : float array;  (** The outline's first and last device points. *)
}

let create cover (clip : Surface.clip) m k s =
  let hw = 0.5 *. Stroke.width s *. k in
  let scale = Affine.stretch m in
  let reach = (Stroke.reach s *. (k *. scale)) +. 1. in
  let r = 0.5 *. Stroke.width s *. (k *. scale) in
  {
    cover;
    m;
    inv = Option.value (Affine.invert m) ~default:Affine.id;
    hw;
    cap = Stroke.cap s;
    join = Stroke.join s;
    miter_limit = Stroke.miter_limit s;
    tol = 0.1 /. scale;
    step =
      (if r <= 0.1 then Float.pi /. 2. else 2. *. Float.acos (1. -. (0.1 /. r)));
    reach;
    pt = Array.make 2 0.;
    wx0 = float clip.x0 -. reach;
    wy0 = float clip.y0 -. reach;
    wx1 = float clip.x1 +. reach;
    wy1 = float clip.y1 +. reach;
    page = [| float clip.x0; float clip.y0; float clip.x1; float clip.y1 |];
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
        (* Turning as a cap turns, the disc winds as every other outline. *)
        move_to p (x +. p.hw) y;
        arc p x y p.hw 0. (-2. *. Float.pi);
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
      let ddx = ref (xs.(j) -. xs.(i)) and ddy = ref (ys.(j) -. ys.(i)) in
      (* Halved where the difference overflows: only its direction counts. *)
      if not (Float.is_finite !ddx && Float.is_finite !ddy) then begin
        ddx := (0.5 *. xs.(j)) -. (0.5 *. xs.(i));
        ddy := (0.5 *. ys.(j)) -. (0.5 *. ys.(i))
      end;
      let ddx = !ddx and ddy = !ddy in
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

(* [far m x0 y0 x1 y1] is [true] iff [m] maps the vector from [(x0, y0)] to
   [(x1, y1)] more than a twentieth of a pixel along an axis. A vector that
   overflows is far, as is one that [m] maps beyond the floats. *)
let[@inline] far (m : Affine.t) x0 y0 x1 y1 =
  let dx = x1 -. x0 and dy = y1 -. y0 in
  (not (Float.abs dx < Float.infinity && Float.abs dy < Float.infinity))
  || (not (Float.abs ((m.xx *. dx) +. (m.xy *. dy)) <= 0.05))
  || not (Float.abs ((m.yx *. dx) +. (m.yy *. dy)) <= 0.05)

(* [strip p src kept ~closed ~ends] sets [kept] to the points of [src] without
   those within a twentieth of a pixel of the point kept before them, and, for a
   closed subpath, of its first point. With [ends], [src] is a dash, whose first
   point gives way to a point of the subpath within that distance. It is the
   direction of the first segment of [src] of positive length, or [(0., 0.)] if
   there is none. *)
let strip p src kept ~closed ~ends =
  let m = p.m in
  kept.n <- 0;
  let ux = ref 0. and uy = ref 0. in
  for i = 0 to src.n - 1 do
    let x = Array.unsafe_get src.xs i and y = Array.unsafe_get src.ys i in
    if kept.n = 0 then push kept x y
    else begin
      let px = kept.xs.(kept.n - 1) and py = kept.ys.(kept.n - 1) in
      if !ux = 0. && !uy = 0. && (x <> px || y <> py) then begin
        (* Halved, so that a difference beyond the floats keeps its
           direction. *)
        let dx = (0.5 *. x) -. (0.5 *. px) and dy = (0.5 *. y) -. (0.5 *. py) in
        let d = Float.hypot dx dy in
        ux := dx /. d;
        uy := dy /. d
      end;
      if far m px py x y then push kept x y
      else if ends && kept.n = 1 && i < src.n - 1 then begin
        kept.xs.(0) <- x;
        kept.ys.(0) <- y
      end
    end
  done;
  if closed && kept.n > 1 then begin
    let l = kept.n - 1 in
    if not (far m kept.xs.(0) kept.ys.(0) kept.xs.(l) kept.ys.(l)) then
      kept.n <- l
  end;
  (!ux, !uy)

(* [unstripped src kept] sets [kept] to the points of the closed subpath [src]
   but exact repeats. A closed subpath that would keep fewer than three points
   strokes these instead, since its outline has no caps to stand for it. *)
let unstripped src kept =
  kept.n <- 0;
  for i = 0 to src.n - 1 do
    let x = Array.unsafe_get src.xs i and y = Array.unsafe_get src.ys i in
    if kept.n = 0 || x <> kept.xs.(kept.n - 1) || y <> kept.ys.(kept.n - 1) then
      push kept x y
  done;
  let l = kept.n - 1 in
  if l > 0 && kept.xs.(l) = kept.xs.(0) && kept.ys.(l) = kept.ys.(0) then
    kept.n <- l

(* Dashes *)

type dashes = {
  pattern : float array;  (** Of even length. *)
  period : float;
  offset : float;
  mutable index : int;  (** The pattern's current length: a dash if even. *)
  mutable rest : float;  (** What remains of it. *)
}

(* [dashes p k s] is the dashes of [s], their lengths multiplied by [k], stroked
   by [p], or [None] for a solid stroke. A pattern whose period is below a
   hundredth of [p]'s tolerance, a thousandth of a pixel, is solid: pixels show
   nothing of its dashes, and walking them would take a step for each. *)
let dashes p k s =
  match Stroke.dash s with
  | [] -> None
  | d ->
      let d = Array.of_list (List.map (fun l -> l *. k) d) in
      let pattern = if Array.length d mod 2 = 1 then Array.append d d else d in
      let period = Array.fold_left ( +. ) 0. pattern in
      if period < 0.01 *. p.tol then None
      else
        Some
          {
            pattern;
            period;
            offset = Stroke.dash_offset s *. k;
            index = 0;
            rest = 0.;
          }

let next d =
  d.index <- (if d.index + 1 = Array.length d.pattern then 0 else d.index + 1);
  d.rest <- d.pattern.(d.index)

(* [advance d len] moves [d] [len] along the pattern. Arriving where a length of
   the pattern ends, it moves to the next, so that a dash of zero length there
   is still to be drawn. A length beyond the floats leaves [d] where it is: no
   phase is known that far. *)
let advance d len =
  if len > 0. && len < Float.infinity then
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

(* [back_to_end d] is how far back from where [d] stands the last dash ended,
   and [ended d] that dash's length. *)
let back_to_end d =
  let n = Array.length d.pattern and i = d.index in
  let used = d.pattern.(i) -. d.rest in
  if i land 1 = 1 then used else used +. d.pattern.((i + n - 1) mod n)

let ended d =
  let n = Array.length d.pattern and i = d.index in
  d.pattern.(if i land 1 = 1 then i - 1 else (i + n - 2) mod n)

(* [on_to_start d] is how far on from where [d] stands the next dash starts, and
   [starting d] that dash's length. *)
let on_to_start d =
  let n = Array.length d.pattern and i = d.index in
  if i land 1 = 1 then d.rest else d.rest +. d.pattern.((i + 1) mod n)

let starting d =
  let n = Array.length d.pattern and i = d.index in
  d.pattern.(if i land 1 = 1 then (i + 1) mod n else (i + 2) mod n)

(* Subpaths *)

type subpaths = {
  pen : t;
  dashes : dashes option;
  raw : points;  (** The subpath being flattened. *)
  piece : points;  (** The dash being cut. *)
  spine : points;
      (** The points of the subpath that stripping it keeps: dashes are cut from
          the polyline of these alone. *)
  kept : points;
  last : float array;  (** The direction of the last segment dashed. *)
  walked : float array;
      (** How far the walk of a segment has gone, along it from the point where
          its device line crosses the page's middle, then the origin and
          position of the run it left. *)
}

let stroke_points s pts ~closed ~ends ~dx ~dy =
  let p = s.pen in
  let ddx, ddy = strip p pts s.kept ~closed ~ends in
  if closed && s.kept.n < 3 then unstripped pts s.kept;
  let dx, dy = if ddx = 0. && ddy = 0. then (dx, dy) else (ddx, ddy) in
  polyline p s.kept.xs s.kept.ys s.kept.n
    ~closed:(closed && s.kept.n > 1)
    ~dx ~dy

let flush_piece s ~dx ~dy =
  if s.piece.n > 0 then begin
    stroke_points s s.piece ~closed:false ~ends:true ~dx ~dy;
    s.piece.n <- 0
  end

(* [crossing p ~x v sx sy tx ty] sets [p.pt] to the point of the segment from
   the device point [(sx, sy)] to [(tx, ty)] on the line [x = v] if [x], and [y
   = v] otherwise, mapped back to the stroke's coordinates: its other device
   coordinate is measured from the end nearer it, as [Cover] cuts its edges. *)
let crossing p ~x v sx sy tx ty =
  let dx = if x then v else Cover.at v sy ty sx tx in
  let dy = if x then Cover.at v sx tx sy ty else v in
  let i = p.inv in
  p.pt.(0) <- (i.xx *. dx) +. (i.xy *. dy) +. i.x0;
  p.pt.(1) <- (i.yx *. dx) +. (i.yy *. dy) +. i.y0

(* [lone s ox oy ux uy a b] strokes, as a piece of its own, the part from [a] to
   [b] of the line from [(ox, oy)] along [(ux, uy)]. *)
let[@inline] lone s ox oy ux uy a b =
  push s.piece (ox +. (ux *. a)) (oy +. (uy *. a));
  push s.piece (ox +. (ux *. b)) (oy +. (uy *. b));
  flush_piece s ~dx:ux ~dy:uy

(* The walk of a segment's dashes is a sequence of runs along it, each walked
   from an origin near it, so that its points keep the precision of their place:
   positions in a run are measured from its origin, and runs are placed by their
   distance along the segment from the point [z] where its device line crosses
   the page's middle ([s.walked] holds how far from [z] the walk has gone, then
   the origin and position it left the last run at). Between runs, the pattern
   advances by the length skipped. *)

(* [skip s d ~page ~ux ~uy ~gap nx ny nl] moves the walk on by [gap], not
   walked, to the position [nl] from the origin [(nx, ny)]. A dash in progress
   ends at its own end, and one in progress where the walk resumes starts at its
   own start, so that every piece is a dash of the pattern, with its caps: a
   dash across the whole gap stays one piece. Of the dashes wholly within the
   gap, ends included, only a cap can reach the page, and with caps that reach
   along the line the one that reaches nearest is drawn: the last to end if the
   page lies ahead, the first to start if it lies behind. The caps of the others
   reach less far onto it. *)
let[@inline] skip s d ~page ~ux ~uy ~gap nx ny nl =
  let ox = s.walked.(1) and oy = s.walked.(2) and l = s.walked.(3) in
  let caps = match s.pen.cap with `Butt -> false | `Round | `Square -> true in
  let across = is_dash d && d.rest > gap in
  if is_dash d && not across then begin
    if s.piece.n = 0 then push s.piece (ox +. (ux *. l)) (oy +. (uy *. l));
    let e = l +. d.rest in
    push s.piece (ox +. (ux *. e)) (oy +. (uy *. e))
  end;
  if not across then flush_piece s ~dx:ux ~dy:uy;
  if caps && page = `Behind then begin
    let on = on_to_start d and len = starting d in
    if on +. len <= gap then lone s ox oy ux uy (l +. on) (l +. on +. len)
  end;
  advance d gap;
  if caps && page = `Ahead then begin
    let back = back_to_end d and len = ended d in
    if back +. len < gap then lone s nx ny ux uy (nl -. back -. len) (nl -. back)
  end;
  if is_dash d && not across then begin
    let a = nl -. (d.pattern.(d.index) -. d.rest) in
    push s.piece (nx +. (ux *. a)) (ny +. (uy *. a))
  end;
  s.walked.(0) <- s.walked.(0) +. gap

(* [run s d ~ox ~oy ~zo ~l0 ~l1 ~ux ~uy ~hi] walks the dashes from [l0] to [l1]
   along the segment from the origin [(ox, oy)], which lies [zo] from [z],
   skipping first to its start what the walk has not reached; [hi] is the end of
   the page's run, from [z]. A run the walk has gone past is left out. The piece
   of a dash still in progress at its end stays open there, to go on in a run
   that follows on. *)
let[@inline] run s d ~ox ~oy ~zo ~l0 ~l1 ~ux ~uy ~hi =
  let cur = s.walked.(0) in
  if cur <= zo +. l1 then begin
    (* A run that the walk has reached goes on from there, with no skip, so that
       its piece runs on across the two. *)
    let l0 =
      if cur < zo +. l0 then begin
        let gap = zo +. l0 -. cur in
        let page = if hi <= cur then `Behind else `Ahead in
        skip s d ~page ~ux ~uy ~gap ox oy l0;
        l0
      end
      else Float.min l1 (cur -. zo)
    in
    s.walked.(0) <- zo +. l1;
    s.walked.(1) <- ox;
    s.walked.(2) <- oy;
    s.walked.(3) <- l1;
    let pos = ref l0 and continue = ref true in
    while !continue do
      if is_dash d && s.piece.n = 0 then
        push s.piece (ox +. (ux *. !pos)) (oy +. (uy *. !pos));
      if !pos +. d.rest <= l1 then begin
        pos := !pos +. d.rest;
        if is_dash d then begin
          push s.piece (ox +. (ux *. !pos)) (oy +. (uy *. !pos));
          flush_piece s ~dx:ux ~dy:uy
        end;
        next d
      end
      else begin
        d.rest <- d.rest -. (l1 -. !pos);
        if is_dash d then push s.piece (ox +. (ux *. l1)) (oy +. (uy *. l1));
        continue := false
      end
    done
  end

(* [dash_segment s d ax ay bx by] cuts into dashes the segment from [(ax, ay)]
   to [(bx, by)], walking only the parts that can put ink on the page.

   A dash's ink is the band of its path out to half the pen's width, its caps at
   its ends and the joins where it passes a vertex. A point of the band at arc
   length [t] shows only if [t] lies within the page's projection onto the
   segment, and a cap reaches no farther along it than the pen's half width. So
   the walk covers the page's projection, grown by a pixel to absorb rounding,
   and a pixel at each end of the segment that lies within the pen's reach of
   the page, where its joins are; it skips the rest. Of the dashes it skips,
   only the one whose cap reaches nearest the page can show, on each side, and
   only through its cap: it is drawn alone. The cost is the page's extent and
   the segment's ends in dashes, whatever the pen's width. A pattern whose
   period, along the segment, is below a thousandth of a pixel is drawn solid
   there. *)
let dash_segment s d ax ay bx by =
  let p = s.pen in
  (* Halved, so that a segment longer than the floats keeps its direction. *)
  let hx = (0.5 *. bx) -. (0.5 *. ax) and hy = (0.5 *. by) -. (0.5 *. ay) in
  let half = Float.hypot hx hy in
  let len = 2. *. half in
  if len > 0. then begin
    let ux = hx /. half and uy = hy /. half in
    s.last.(0) <- ux;
    s.last.(1) <- uy;
    (* A pixel along the segment, in the stroke's coordinates. *)
    let m = p.m in
    let px_len =
      Float.hypot ((m.xx *. ux) +. (m.xy *. uy)) ((m.yx *. ux) +. (m.yy *. uy))
    in
    let dd = 1. /. px_len in
    let sx = dev_x p ax ay and sy = dev_y p ax ay in
    let tx = dev_x p bx by and ty = dev_y p bx by in
    (* Where the segment enters and leaves the grown window, by Liang and
       Barsky: the last of its entries and the first of its exits along it. *)
    let px = ref ax and py = ref ay and qx = ref bx and qy = ref by in
    let missed = ref false in
    for k = 0 to 1 do
      let x = k = 0 in
      let a = if x then sx else sy and b = if x then tx else ty in
      let lo = if x then p.wx0 else p.wy0 and hi = if x then p.wx1 else p.wy1 in
      if a = b then (if a < lo || a > hi then missed := true)
      else begin
        if (a < b && a < lo) || (a > b && a > hi) then begin
          crossing p ~x (if a < b then lo else hi) sx sy tx ty;
          let ex = p.pt.(0) and ey = p.pt.(1) in
          if ((ex -. !px) *. ux) +. ((ey -. !py) *. uy) > 0. then begin
            px := ex;
            py := ey
          end
        end;
        if (a < b && b > hi) || (a > b && b < lo) then begin
          crossing p ~x (if a < b then hi else lo) sx sy tx ty;
          let lx = p.pt.(0) and ly = p.pt.(1) in
          if ((lx -. !qx) *. ux) +. ((ly -. !qy) *. uy) < 0. then begin
            qx := lx;
            qy := ly
          end
        end
      end
    done;
    let px = !px and py = !py and qx = !qx and qy = !qy in
    let starts = px = ax && py = ay and ends = qx = bx && qy = by in
    if !missed || not (((qx -. px) *. ux) +. ((qy -. py) *. uy) > 0.) then begin
      flush_piece s ~dx:ux ~dy:uy;
      advance d len
    end
    else if d.period *. px_len < 0.001 then begin
      (* Solid: the piece in progress goes on through it. *)
      if s.piece.n = 0 then push s.piece px py;
      push s.piece qx qy;
      advance d len;
      if not ends then flush_piece s ~dx:ux ~dy:uy
    end
    else begin
      (* A dash ended where the segment starts. *)
      if not (is_dash d) then flush_piece s ~dx:ux ~dy:uy;
      advance d (Float.hypot (px -. ax) (py -. ay));
      (* [z], and the page's projection from it, a pixel wider each way. *)
      let page = p.page in
      let mx = Float.abs (tx -. sx) >= Float.abs (ty -. sy) in
      crossing p ~x:mx
        (if mx then 0.5 *. (page.(0) +. page.(2))
         else 0.5 *. (page.(1) +. page.(3)))
        sx sy tx ty;
      let zx = p.pt.(0) and zy = p.pt.(1) in
      let i = p.inv in
      let lo = ref Float.infinity and hi = ref Float.neg_infinity in
      for c = 0 to 3 do
        let dx = page.(if c = 0 || c = 3 then 0 else 2) in
        let dy = page.(if c < 2 then 1 else 3) in
        let cx = (i.xx *. dx) +. (i.xy *. dy) +. i.x0 in
        let cy = (i.yx *. dx) +. (i.yy *. dy) +. i.y0 in
        let a = ((cx -. zx) *. ux) +. ((cy -. zy) *. uy) in
        lo := Float.min !lo a;
        hi := Float.max !hi a
      done;
      let lo = !lo -. dd and hi = !hi +. dd in
      let pz = ((px -. zx) *. ux) +. ((py -. zy) *. uy) in
      let qz = ((qx -. zx) *. ux) +. ((qy -. zy) *. uy) in
      s.walked.(0) <- pz;
      s.walked.(1) <- px;
      s.walked.(2) <- py;
      s.walked.(3) <- 0.;
      if starts then
        run s d ~ox:px ~oy:py ~zo:pz ~l0:0.
          ~l1:(Float.min dd (qz -. pz))
          ~ux ~uy ~hi;
      if Float.max lo pz < Float.min hi qz then
        run s d ~ox:zx ~oy:zy ~zo:0. ~l0:(Float.max lo pz) ~l1:(Float.min hi qz)
          ~ux ~uy ~hi;
      if ends then
        run s d ~ox:qx ~oy:qy ~zo:qz
          ~l0:(Float.max (-.dd) (pz -. qz))
          ~l1:0. ~ux ~uy ~hi
      else begin
        let gap = qz -. s.walked.(0) in
        if gap > 0. then begin
          let page = if hi <= s.walked.(0) then `Behind else `Ahead in
          skip s d ~page ~ux ~uy ~gap qx qy 0.
        end;
        (* A dash in progress runs on to where the segment leaves the grown
           window, beyond which nothing of it shows. *)
        if is_dash d then push s.piece qx qy;
        flush_piece s ~dx:ux ~dy:uy;
        advance d (Float.hypot (bx -. qx) (by -. qy))
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
        let spine = s.spine in
        let ux, uy = strip s.pen raw spine ~closed ~ends:false in
        s.last.(0) <- ux;
        s.last.(1) <- uy;
        if spine.n = 1 && is_dash d then push s.piece spine.xs.(0) spine.ys.(0);
        for i = 0 to spine.n - 2 do
          dash_segment s d spine.xs.(i) spine.ys.(i) spine.xs.(i + 1)
            spine.ys.(i + 1)
        done;
        let l = spine.n - 1 in
        if closed && l > 0 then
          dash_segment s d spine.xs.(l) spine.ys.(l) spine.xs.(0) spine.ys.(0);
        flush_piece s ~dx:s.last.(0) ~dy:s.last.(1)
    | _ ->
        stroke_points s raw ~closed:(closed && !moves) ~ends:false ~dx:0. ~dy:0.
  end;
  raw.n <- 0

(* [in_range p k s] is [true] iff [p], the pen of [s] with its lengths
   multiplied by [k], stays within the range of floats, in the stroke's
   coordinates and, with its reach, on the device, twice over from the edge of
   [Instances.far], where its path is cut. Transforms within a stamp that
   magnify as much as its scale shrinks take it beyond, where it draws
   nothing. *)
let in_range p k s =
  let fits l = Float.is_finite (l *. k) in
  Float.is_finite p.tol && p.tol > 0. && Float.is_finite p.hw
  && Float.is_finite p.wx0
  && Float.is_finite (Box2.maxx Instances.far +. (2. *. p.reach))
  && fits (Stroke.dash_offset s)
  && List.for_all fits (Stroke.dash s)

(* [stroke cover clip m ~pen s path] deposits in [cover] the outline of [path]
   stroked with [s], its lengths multiplied by [pen], in the coordinates that
   [m] maps to the device. The points of [path] are used as they are: as an
   instance of a stamp is the picture with its pens divided by its scale, only
   its pen's lengths change, so a stamp strokes what the instance strokes. *)
let stroke cover clip m ~pen:k s path =
  let pen = create cover clip m k s in
  if in_range pen k s then begin
    let dashes = dashes pen k s and kept = points () in
    let s =
      {
        pen;
        dashes;
        raw = points ();
        piece = points ();
        (* A solid stroke reads no spine. *)
        spine = (if Option.is_none dashes then kept else points ());
        kept;
        last = [| 0.; 0. |];
        walked = Array.make 4 0.;
      }
    in
    (* A cut point lies a reach beyond [far], so its cap stays outside it. *)
    let far = Box2.grow pen.reach Instances.far in
    Path.flatten ~tolerance:pen.tol Affine.id
      ~move:(fun () x y ->
        subpath s ~closed:false;
        push s.raw x y)
      ~line:(fun () x y -> push s.raw x y)
      ~close:(fun () -> subpath s ~closed:true)
      () (Path.cut ~m far path);
    subpath s ~closed:false
  end
