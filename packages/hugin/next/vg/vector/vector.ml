(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_next_gg
open Hugin_next_font
open Hugin_next_vg

(* Numbers *)

let decimals k =
  if not (k > 1.) then 3
  else if k >= 1e14 then 17
  else 3 + int_of_float (Float.ceil (Float.log10 k))

(* [limit v] is [v] within [±1e15], NaN taken as [1e15]. *)
let limit v = if v < -1e15 then -1e15 else if v <= 1e15 then v else 1e15
let pow10 = Array.init 18 (fun i -> 10. ** float i)
let ipow10 = Array.init 18 (fun i -> int_of_float pow10.(i))

let trim_zeros s =
  if not (String.contains s '.') then s
  else begin
    let n = ref (String.length s) in
    while s.[!n - 1] = '0' do
      decr n
    done;
    if s.[!n - 1] = '.' then decr n;
    String.sub s 0 !n
  end

(* [add_digits b n k] adds the [k] last decimal digits of [n >= 0]. *)
let add_digits b n k =
  for i = k - 1 downto 0 do
    Buffer.add_char b (Char.unsafe_chr (48 + (n / ipow10.(i) mod 10)))
  done

(* [width n] is the number of decimal digits of [0 <= n < 10{^17}]. *)
let width n =
  let rec loop k = if k < 17 && n >= ipow10.(k) then loop (k + 1) else k in
  loop 1

let finer k d = Int.min 17 (d + k)

let add_fixed b d v =
  let v = limit v in
  let scaled = Float.round (v *. pow10.(d)) in
  if Float.abs scaled < 1e17 then begin
    let n = int_of_float scaled in
    if n = 0 then Buffer.add_char b '0'
    else begin
      if n < 0 then Buffer.add_char b '-';
      let n = Int.abs n and p = ipow10.(d) in
      let q = n / p in
      add_digits b q (width q);
      let f = ref (n mod p) in
      if !f <> 0 then begin
        Buffer.add_char b '.';
        let digits = ref d in
        while !f mod 10 = 0 do
          f := !f / 10;
          decr digits
        done;
        add_digits b !f !digits
      end
    end
  end
  else Buffer.add_string b (trim_zeros (Printf.sprintf "%.*f" d v))

let add_exact b v =
  if v = 0. then Buffer.add_char b '0'
  else
    let s = Printf.sprintf "%.15g" v in
    let s =
      if float_of_string s = v then s
      else
        let s = Printf.sprintf "%.16g" v in
        if float_of_string s = v then s else Printf.sprintf "%.17g" v
    in
    Buffer.add_string b s

(* Maps *)

let linear (m : Affine.t) = { m with x0 = 0.; y0 = 0. }

(* The coefficients are divided by the largest of their magnitudes [k] first, so
   that their squares neither overflow nor underflow. *)
let stretch (m : Affine.t) =
  let k =
    Float.max
      (Float.max (Float.abs m.xx) (Float.abs m.yx))
      (Float.max (Float.abs m.xy) (Float.abs m.yy))
  in
  let xx = m.xx /. k and yx = m.yx /. k and xy = m.xy /. k and yy = m.yy /. k in
  let a = (xx *. xx) +. (yx *. yx) and b = (xy *. xy) +. (yy *. yy) in
  let c = (xx *. xy) +. (yx *. yy) in
  k
  *. Float.sqrt
       ((0.5 *. (a +. b))
       +. Float.sqrt ((0.25 *. (a -. b) *. (a -. b)) +. (c *. c)))

let unit (m : Affine.t) =
  let s = stretch m in
  {
    Affine.xx = m.xx /. s;
    yx = m.yx /. s;
    xy = m.xy /. s;
    yy = m.yy /. s;
    x0 = 0.;
    y0 = 0.;
  }

let near eps a b = Float.abs (a -. b) <= eps

let is_similar (m : Affine.t) =
  let eps = 1e-9 *. stretch m in
  (near eps m.xx m.yy && near eps m.yx (-.m.xy))
  || (near eps m.xx (-.m.yy) && near eps m.yx m.xy)

let is_axial (m : Affine.t) =
  let eps = 1e-9 *. stretch m in
  m.xx > 0. && m.yy > 0. && near eps m.yx 0. && near eps m.xy 0.

let is_even (m : Affine.t) = is_axial m && near (1e-9 *. stretch m) m.xx m.yy

(* Paths *)

type sink = {
  move : float -> float -> unit;
  line : float -> float -> unit;
  cubic : float -> float -> float -> float -> float -> float -> unit;
  close : unit -> unit;
}

(* A subpath mapped to the frame: its points, the start first, then each
   segment's, one for a line and three for a cubic, as [kinds] says. *)
type sub = {
  mutable xs : float array;
  mutable ys : float array;
  mutable np : int;
  mutable kinds : Bytes.t;  (** ['L'] or ['C'] per segment. *)
  mutable nk : int;
  mutable closed : bool;
}

let sub () =
  {
    xs = Array.make 16 0.;
    ys = Array.make 16 0.;
    np = 0;
    kinds = Bytes.make 16 'L';
    nk = 0;
    closed = false;
  }

let push s x y =
  if s.np = Array.length s.xs then begin
    s.xs <- Array.append s.xs (Array.make s.np 0.);
    s.ys <- Array.append s.ys (Array.make s.np 0.)
  end;
  s.xs.(s.np) <- x;
  s.ys.(s.np) <- y;
  s.np <- s.np + 1

let kind s k =
  if s.nk = Bytes.length s.kinds then s.kinds <- Bytes.extend s.kinds 0 s.nk;
  Bytes.set s.kinds s.nk k;
  s.nk <- s.nk + 1

(* [iter_subs m q f] calls [f] on each subpath of [q] mapped through [m]. The
   subpath given is reused. *)
let iter_subs (m : Affine.t) q f =
  let s = sub () in
  let point x y =
    push s
      (limit ((m.xx *. x) +. (m.xy *. y) +. m.x0))
      (limit ((m.yx *. x) +. (m.yy *. y) +. m.y0))
  in
  let flush () = if s.np > 0 then f s in
  Path.fold
    ~move:(fun () x y ->
      flush ();
      s.np <- 0;
      s.nk <- 0;
      s.closed <- false;
      point x y)
    ~line:(fun () x y ->
      kind s 'L';
      point x y)
    ~cubic:(fun () x1 y1 x2 y2 x y ->
      kind s 'C';
      point x1 y1;
      point x2 y2;
      point x y)
    ~close:(fun () -> s.closed <- true)
    () q;
  flush ()

(* Points on the edges of [r] count as inside, as they would outside: nothing
   paints along a line. *)
let inside (r : Box2.t) x y =
  x >= Box2.minx r && x <= Box2.maxx r && y >= Box2.miny r && y <= Box2.maxy r

let all_inside r s =
  let rec loop i = i >= s.np || (inside r s.xs.(i) s.ys.(i) && loop (i + 1)) in
  loop 0

let is_point s =
  let rec loop i =
    i >= s.np || (s.xs.(i) = s.xs.(0) && s.ys.(i) = s.ys.(0) && loop (i + 1))
  in
  loop 1

(* [mapped out sink] is [sink] receiving its points mapped through the linear
   map [out]. *)
let mapped out sink =
  match out with
  | None -> sink
  | Some (o : Affine.t) ->
      let ox x y = (o.xx *. x) +. (o.xy *. y) in
      let oy x y = (o.yx *. x) +. (o.yy *. y) in
      {
        move = (fun x y -> sink.move (ox x y) (oy x y));
        line = (fun x y -> sink.line (ox x y) (oy x y));
        cubic =
          (fun x1 y1 x2 y2 x y ->
            sink.cubic (ox x1 y1) (oy x1 y1) (ox x2 y2) (oy x2 y2) (ox x y)
              (oy x y));
        close = sink.close;
      }

let emit_whole sink s ~close =
  sink.move s.xs.(0) s.ys.(0);
  let i = ref 1 in
  for k = 0 to s.nk - 1 do
    if Bytes.get s.kinds k = 'L' then begin
      sink.line s.xs.(!i) s.ys.(!i);
      incr i
    end
    else begin
      sink.cubic s.xs.(!i) s.ys.(!i)
        s.xs.(!i + 1)
        s.ys.(!i + 1)
        s.xs.(!i + 2)
        s.ys.(!i + 2);
      i := !i + 3
    end
  done;
  if close then sink.close ()

(* Polygons: vertices, each with the edge that arrives at it from the previous
   one, cyclically: a line, or a cubic through two control points. *)
type poly = {
  mutable px : float array;
  mutable py : float array;
  mutable cubic : bool array;
  mutable c : float array;  (** Four control coordinates per vertex. *)
  mutable n : int;
}

let poly () =
  {
    px = Array.make 16 0.;
    py = Array.make 16 0.;
    cubic = Array.make 16 false;
    c = Array.make 64 0.;
    n = 0;
  }

let vertex p ?(c = [||]) x y =
  if p.n = Array.length p.px then begin
    let k = p.n in
    p.px <- Array.append p.px (Array.make k 0.);
    p.py <- Array.append p.py (Array.make k 0.);
    p.cubic <- Array.append p.cubic (Array.make k false);
    p.c <- Array.append p.c (Array.make (4 * k) 0.)
  end;
  p.px.(p.n) <- x;
  p.py.(p.n) <- y;
  p.cubic.(p.n) <- Array.length c = 4;
  if Array.length c = 4 then Array.blit c 0 p.c (4 * p.n) 4;
  p.n <- p.n + 1

let flatness = 0.0005

(* [chords x0 y0 x1 y1 x2 y2 x3 y3 f] calls [f] on the end of each chord of the
   cubic within [flatness] of it, as [Path.flatten] cuts it. *)
let chords x0 y0 x1 y1 x2 y2 x3 y3 f =
  let dd ax ay bx by cx cy =
    Float.hypot (ax -. (2. *. bx) +. cx) (ay -. (2. *. by) +. cy)
  in
  let d = Float.max (dd x0 y0 x1 y1 x2 y2) (dd x1 y1 x2 y2 x3 y3) in
  let k = Float.ceil (Float.sqrt (0.75 *. d /. flatness)) in
  let n =
    if Float.is_nan k then 1
    else int_of_float (Float.min 65536. (Float.max 1. k))
  in
  let at t a b c d =
    let u = 1. -. t in
    (u *. u *. u *. a)
    +. (3. *. u *. u *. t *. b)
    +. (3. *. u *. t *. t *. c)
    +. (t *. t *. t *. d)
  in
  for i = 1 to n - 1 do
    let t = float i /. float n in
    f (at t x0 x1 x2 x3) (at t y0 y1 y2 y3)
  done;
  f x3 y3

(* [polygon r s p] sets [p] to the polygon of [s], its curves that do not lie
   within [r] flattened. *)
let polygon r s p =
  p.n <- 0;
  vertex p s.xs.(0) s.ys.(0);
  let i = ref 1 in
  for k = 0 to s.nk - 1 do
    if Bytes.get s.kinds k = 'L' then begin
      vertex p s.xs.(!i) s.ys.(!i);
      incr i
    end
    else begin
      let j = !i in
      let x0 = s.xs.(j - 1) and y0 = s.ys.(j - 1) in
      if
        inside r s.xs.(j) s.ys.(j)
        && inside r s.xs.(j + 1) s.ys.(j + 1)
        && inside r s.xs.(j + 2) s.ys.(j + 2)
        && inside r x0 y0
      then
        vertex p
          ~c:[| s.xs.(j); s.ys.(j); s.xs.(j + 1); s.ys.(j + 1) |]
          s.xs.(j + 2)
          s.ys.(j + 2)
      else
        chords x0 y0 s.xs.(j) s.ys.(j)
          s.xs.(j + 1)
          s.ys.(j + 1)
          s.xs.(j + 2)
          s.ys.(j + 2)
          (fun x y -> vertex p x y);
      i := !i + 3
    end
  done

(* [clip_half src dst ~y ~ge v] clips the polygon [src] into [dst] to the
   half-plane where the y coordinate if [y], else the x one, is at least [v] if
   [ge], else at most [v]. Its cubics lie within the box being clipped to, so on
   the kept side. *)
let clip_half src dst ~y ~ge v =
  dst.n <- 0;
  let coord i = if y then src.py.(i) else src.px.(i) in
  let keep i = if ge then coord i >= v else coord i <= v in
  let cross a b =
    let t = (v -. coord a) /. (coord b -. coord a) in
    if y then vertex dst (src.px.(a) +. (t *. (src.px.(b) -. src.px.(a)))) v
    else vertex dst v (src.py.(a) +. (t *. (src.py.(b) -. src.py.(a))))
  in
  for i = 0 to src.n - 1 do
    let prev = if i = 0 then src.n - 1 else i - 1 in
    if keep i then begin
      if not (keep prev) then cross prev i;
      if src.cubic.(i) then
        vertex dst ~c:(Array.sub src.c (4 * i) 4) src.px.(i) src.py.(i)
      else vertex dst src.px.(i) src.py.(i)
    end
    else if keep prev then cross prev i
  done

let emit_poly sink p =
  (* Lines of zero length are dropped; a polygon left with fewer than three
     vertices and no curve has no area. *)
  let q = poly () in
  for i = 0 to p.n - 1 do
    if
      p.cubic.(i) || q.n = 0
      || p.px.(i) <> q.px.(q.n - 1)
      || p.py.(i) <> q.py.(q.n - 1)
    then
      if p.cubic.(i) then
        vertex q ~c:(Array.sub p.c (4 * i) 4) p.px.(i) p.py.(i)
      else vertex q p.px.(i) p.py.(i)
  done;
  if q.n >= 3 || Array.exists Fun.id (Array.sub q.cubic 0 q.n) then begin
    sink.move q.px.(0) q.py.(0);
    for i = 1 to q.n - 1 do
      if q.cubic.(i) then
        sink.cubic
          q.c.(4 * i)
          q.c.((4 * i) + 1)
          q.c.((4 * i) + 2)
          q.c.((4 * i) + 3)
          q.px.(i) q.py.(i)
      else sink.line q.px.(i) q.py.(i)
    done;
    sink.close ()
  end

let area ?out m r q sink =
  let sink = mapped out sink in
  let a = poly () and b = poly () in
  iter_subs m q (fun s ->
      if all_inside r s then emit_whole sink s ~close:true
      else begin
        polygon r s a;
        clip_half a b ~y:false ~ge:true (Box2.minx r);
        clip_half b a ~y:false ~ge:false (Box2.maxx r);
        clip_half a b ~y:true ~ge:true (Box2.miny r);
        clip_half b a ~y:true ~ge:false (Box2.maxy r);
        emit_poly sink a
      end)

(* Outlines cut to a box: the pieces of a polyline within it, each a run of
   commands. *)

type cmd =
  | Move of float * float
  | Line of float * float
  | Cubic of float array
  | Close

let length out x0 y0 x1 y1 =
  match out with
  | None -> Float.hypot (x1 -. x0) (y1 -. y0)
  | Some (o : Affine.t) ->
      let dx = x1 -. x0 and dy = y1 -. y0 in
      Float.hypot ((o.xx *. dx) +. (o.xy *. dy)) ((o.yx *. dx) +. (o.yy *. dy))

let cubic_length out x0 y0 c =
  let len = ref 0. and px = ref x0 and py = ref y0 in
  chords x0 y0 c.(0) c.(1) c.(2) c.(3) c.(4) c.(5) (fun x y ->
      len := !len +. length out !px !py x y;
      px := x;
      py := y);
  !len

(* [cut_outline out r ~dashed s] is the pieces of [s] within [r], each with the
   length before it, in order. *)
let cut_outline out r ~dashed s =
  let p = poly () in
  polygon r s p;
  if s.closed then vertex p p.px.(0) p.py.(0);
  let pieces = ref [] and current = ref [] and start = ref 0. in
  let dist = ref 0. in
  let close_piece () =
    if !current <> [] then begin
      pieces := (!start, List.rev !current) :: !pieces;
      current := []
    end
  in
  for i = 1 to p.n - 1 do
    let ax = p.px.(i - 1) and ay = p.py.(i - 1) in
    let bx = p.px.(i) and by = p.py.(i) in
    if p.cubic.(i) then begin
      let c = Array.append (Array.sub p.c (4 * i) 4) [| bx; by |] in
      if !current = [] then begin
        start := !dist;
        current := [ Move (ax, ay) ]
      end;
      current := Cubic c :: !current;
      if dashed then dist := !dist +. cubic_length out ax ay c
    end
    else begin
      let ex = bx -. ax and ey = by -. ay in
      let t0 = ref 0. and t1 = ref 1. in
      (* Touching the box, along an edge or at a corner, counts as meeting it,
         as it would not: nothing paints along a line or at a point. *)
      let edge q d =
        if q = 0. then (if d < 0. then t1 := -1.)
        else
          let t = d /. q in
          if q < 0. then (if t > !t0 then t0 := t) else if t < !t1 then t1 := t
      in
      edge (-.ex) (ax -. Box2.minx r);
      edge ex (Box2.maxx r -. ax);
      edge (-.ey) (ay -. Box2.miny r);
      edge ey (Box2.maxy r -. ay);
      let len = if dashed then length out ax ay bx by else 0. in
      if !t0 > !t1 then close_piece ()
      else begin
        if !t0 > 0. || !current = [] then begin
          close_piece ();
          start := !dist +. (!t0 *. len);
          current := [ Move (ax +. (!t0 *. ex), ay +. (!t0 *. ey)) ]
        end;
        (* The end exactly, where it is kept, so that a closed subpath returns
           to its start. *)
        let x, y =
          if !t1 = 1. then (bx, by) else (ax +. (!t1 *. ex), ay +. (!t1 *. ey))
        in
        current := Line (x, y) :: !current;
        if !t1 < 1. then close_piece ()
      end;
      dist := !dist +. len
    end
  done;
  close_piece ();
  let pieces = List.rev !pieces in
  (* A closed solid subpath whose start lies within [r] joins there, its last
     piece ending at its start: that piece runs on into the first, or closes if
     it is the first. *)
  let starts (_, cmds) =
    match cmds with
    | Move (x, y) :: _ -> x = p.px.(0) && y = p.py.(0)
    | _ -> false
  in
  if dashed || not s.closed then pieces
  else
    match pieces with
    | [ ((d, cmds) as only) ] when starts only -> [ (d, cmds @ [ Close ]) ]
    | first :: (_ :: _ as rest) when starts first ->
        let rev = List.rev rest in
        let d, cmds = List.hd rev in
        List.rev (List.tl rev) @ [ (d, cmds @ List.tl (snd first)) ]
    | _ -> pieces

let outline ?out m r ~points ~dashed q ~piece sink =
  let sink = mapped out sink in
  iter_subs m q (fun s ->
      if is_point s then
        begin if points && inside r s.xs.(0) s.ys.(0) then begin
          piece 0.;
          (* A subpath of a start alone gets a segment, without which viewers
             stroke nothing. *)
          if s.np = 1 then begin
            sink.move s.xs.(0) s.ys.(0);
            sink.line s.xs.(0) s.ys.(0)
          end
          else emit_whole sink s ~close:s.closed
        end
        end
      else if all_inside r s then begin
        piece 0.;
        emit_whole sink s ~close:s.closed
      end
      else
        List.iter
          (fun (d, cmds) ->
            piece (if dashed then d else 0.);
            List.iter
              (function
                | Move (x, y) -> sink.move x y
                | Line (x, y) -> sink.line x y
                | Cubic c -> sink.cubic c.(0) c.(1) c.(2) c.(3) c.(4) c.(5)
                | Close -> sink.close ())
              cmds)
          (cut_outline out r ~dashed s))

(* Leaves *)

let all = Box2.v (-1e15) (-1e15) 2e15 2e15

let grown r k =
  let k = if k <= 1e15 then k else 1e15 in
  Box2.of_pts
    (P2.v (Box2.minx r -. k) (Box2.miny r -. k))
    (P2.v (Box2.maxx r +. k) (Box2.maxy r +. k))

let overlaps r minx miny maxx maxy =
  minx <= Box2.maxx r
  && maxx >= Box2.minx r
  && miny <= Box2.maxy r
  && maxy >= Box2.miny r

let meets r b =
  overlaps r (Box2.minx b) (Box2.miny b) (Box2.maxx b) (Box2.maxy b)

let run_meets r m run =
  match Run.bounds run with
  | None ->
      let o = P2.transform m (P2.v 0. 0.) in
      inside r (P2.x o) (P2.y o)
  | Some b -> (
      try meets r (Box2.transform m b) with Invalid_argument _ -> false)
  | exception Invalid_argument _ -> false

let crop r m box w h =
  let pixel =
    Affine.(
      m
      * translate (Box2.minx box) (Box2.miny box)
      * scale (Box2.w box /. float w) (Box2.h box /. float h))
  in
  match Affine.invert pixel with
  | None -> None
  | Some inv ->
      let corner x y = P2.transform inv (P2.v x y) in
      let cs =
        [
          corner (Box2.minx r) (Box2.miny r);
          corner (Box2.maxx r) (Box2.miny r);
          corner (Box2.maxx r) (Box2.maxy r);
          corner (Box2.minx r) (Box2.maxy r);
        ]
      in
      let lo f = List.fold_left (fun a p -> Float.min a (f p)) infinity cs in
      let hi f =
        List.fold_left (fun a p -> Float.max a (f p)) neg_infinity cs
      in
      (* A corner beyond the range of floats, NaN, gives [0]. *)
      let within n v =
        if v > 0. then if v < float n then int_of_float v else n else 0
      in
      let c0 = within w (Float.floor (lo P2.x))
      and c1 = within w (Float.ceil (hi P2.x)) in
      let r0 = within h (Float.floor (lo P2.y))
      and r1 = within h (Float.ceil (hi P2.y)) in
      if c0 < c1 && r0 < r1 then Some (c0, r0, c1, r1) else None

(* Stamps *)

type pen = { width : float; dash : float list; offset : float }

let pen m k s =
  let f = k *. stretch m in
  {
    width = limit (Stroke.width s *. f);
    dash = List.map (fun d -> limit (d *. f)) (Stroke.dash s);
    offset = limit (Stroke.dash_offset s *. f);
  }

let pen_equal a b =
  Float.equal a.width b.width
  && List.equal Float.equal a.dash b.dash
  && Float.equal a.offset b.offset

let pens m k p =
  let found = ref [] in
  let rec walk m (p : Picture.t) =
    match p with
    | Stroke { stroke; _ } ->
        let q = pen m k stroke in
        if not (List.exists (pen_equal q) !found) then found := q :: !found
    | Empty | Fill _ | Glyphs _ | Image _ -> ()
    | Group ps -> List.iter (walk m) ps
    | Transform { m = m'; picture } -> walk Affine.(m * m') picture
    | Clip { picture; _ }
    | Opacity { picture; _ }
    | Tag { picture; _ }
    | Stamp { picture; _ } ->
        walk m picture
  in
  walk (linear m) p;
  List.rev !found

let rec scales_within (p : Picture.t) =
  match p with
  | Stamp { scales = Some _; _ } -> true
  | Empty | Fill _ | Stroke _ | Glyphs _ | Image _ -> false
  | Group ps -> List.exists scales_within ps
  | Transform { picture; _ }
  | Clip { picture; _ }
  | Opacity { picture; _ }
  | Tag { picture; _ }
  | Stamp { picture; _ } ->
      scales_within picture

let reach pen s =
  let k =
    match (Stroke.join s, Stroke.cap s) with
    | `Miter, `Square -> Float.max (Stroke.miter_limit s) (Float.sqrt 2.)
    | `Miter, _ -> Stroke.miter_limit s
    | _, `Square -> Float.sqrt 2.
    | _ -> 1.
  in
  0.5 *. pen.width *. k
