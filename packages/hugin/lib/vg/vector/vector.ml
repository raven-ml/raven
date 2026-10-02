(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_gg
open Hugin_font
open Hugin_vg

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

let rounded d v = Float.round (v *. pow10.(d)) /. pow10.(d)

(* Maps *)

let unit (m : Affine.t) =
  let s = Affine.stretch m in
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
  let eps = 1e-9 *. Affine.stretch m in
  (near eps m.xx m.yy && near eps m.yx (-.m.xy))
  || (near eps m.xx (-.m.yy) && near eps m.yx m.xy)

let is_axial (m : Affine.t) =
  let eps = 1e-9 *. Affine.stretch m in
  m.xx > 0. && m.yy > 0. && near eps m.yx 0. && near eps m.xy 0.

let is_even (m : Affine.t) =
  is_axial m && near (1e-9 *. Affine.stretch m) m.xx m.yy

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

(* [cropped r s ~close sink] gives [sink] the subpath [s], closed if [close], as
   [Path.crop r] cuts it. *)
let cropped r s ~close sink =
  let path = ref Path.empty in
  emit_whole
    {
      move = (fun x y -> path := Path.move_to (P2.v x y) !path);
      line = (fun x y -> path := Path.line_to (P2.v x y) !path);
      cubic =
        (fun x1 y1 x2 y2 x y ->
          path := Path.cubic_to (P2.v x1 y1) (P2.v x2 y2) (P2.v x y) !path);
      close = (fun () -> path := Path.close !path);
    }
    s ~close;
  Path.fold
    ~move:(fun () -> sink.move)
    ~line:(fun () -> sink.line)
    ~cubic:(fun () -> sink.cubic)
    ~close:sink.close () (Path.crop r !path)

let area m r q sink =
  iter_subs m q (fun s ->
      if all_inside r s then emit_whole sink s ~close:true
      else cropped r s ~close:true sink)

(* Dashed outlines cut to a box: the pieces of a polyline within it, each a run
   of commands with the length along the subpath before it. *)

(* Polygons: vertices, each with the edge that arrives at it from the previous
   one: a line, or a cubic through two control points. *)
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

let length out x0 y0 x1 y1 =
  match out with
  | None -> Float.hypot (x1 -. x0) (y1 -. y0)
  | Some (o : Affine.t) ->
      let dx = x1 -. x0 and dy = y1 -. y0 in
      Float.hypot ((o.xx *. dx) +. (o.xy *. dy)) ((o.yx *. dx) +. (o.yy *. dy))

(* [dashes out r s ~piece sink] gives [sink] the pieces of [s] within [r], each
   preceded by [piece] of the length before it, measured in [out]'s
   coordinates. *)
let dashes out r s ~piece sink =
  let p = poly () in
  polygon r s p;
  if s.closed then vertex p p.px.(0) p.py.(0);
  let dist = ref 0. and open_ = ref false in
  let start d x y =
    if not !open_ then begin
      piece d;
      sink.move x y;
      open_ := true
    end
  in
  for i = 1 to p.n - 1 do
    let ax = p.px.(i - 1) and ay = p.py.(i - 1) in
    let bx = p.px.(i) and by = p.py.(i) in
    if p.cubic.(i) then begin
      let c = p.c and k = 4 * i in
      start !dist ax ay;
      sink.cubic c.(k) c.(k + 1) c.(k + 2) c.(k + 3) bx by;
      let px = ref ax and py = ref ay in
      chords ax ay c.(k)
        c.(k + 1)
        c.(k + 2)
        c.(k + 3)
        bx by
        (fun x y ->
          dist := !dist +. length out !px !py x y;
          px := x;
          py := y)
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
      let len = length out ax ay bx by in
      if !t0 > !t1 then open_ := false
      else begin
        if !t0 > 0. then open_ := false;
        start (!dist +. (!t0 *. len)) (ax +. (!t0 *. ex)) (ay +. (!t0 *. ey));
        (* The end exactly, where it is kept, so that a closed subpath returns
           to its start. *)
        if !t1 = 1. then sink.line bx by
        else begin
          sink.line (ax +. (!t1 *. ex)) (ay +. (!t1 *. ey));
          open_ := false
        end
      end;
      dist := !dist +. len
    end
  done

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
      else if dashed then dashes out r s ~piece sink
      else begin
        piece 0.;
        cropped r s ~close:s.closed sink
      end)

(* Leaves *)

let grown r k = Box2.grow (if k <= 1e15 then k else 1e15) r

let overlaps r minx miny maxx maxy =
  minx <= Box2.maxx r
  && maxx >= Box2.minx r
  && miny <= Box2.maxy r
  && maxy >= Box2.miny r

let run_meets r m run =
  match Run.bounds run with
  | None ->
      let o = P2.transform m (P2.v 0. 0.) in
      inside r (P2.x o) (P2.y o)
  | Some b -> (
      match Box2.transform m b with
      | b -> overlaps r (Box2.minx b) (Box2.miny b) (Box2.maxx b) (Box2.maxy b)
      | exception Invalid_argument _ -> false)

let ems font =
  let b = Font.bounds font in
  Float.max
    (Float.max (Float.abs (Box2.minx b)) (Float.abs (Box2.maxx b)))
    (Float.max (Float.abs (Box2.miny b)) (Float.abs (Box2.maxy b)))

type pen = { width : float; dash : float list; offset : float }

let pen m k s =
  let f = k *. Affine.stretch m in
  {
    width = limit (Stroke.width s *. f);
    dash = List.map (fun d -> limit (d *. f)) (Stroke.dash s);
    offset = limit (Stroke.dash_offset s *. f);
  }

let dash_decimals d = finer 3 d

let is_dashed d pen =
  let d = dash_decimals d in
  List.exists (fun v -> rounded d v <> 0.) pen.dash

let phase pen offset =
  let period = List.fold_left ( +. ) 0. pen.dash in
  let period =
    if List.length pen.dash mod 2 = 1 then 2. *. period else period
  in
  Float.rem (pen.offset +. offset) period

(* [window cut m box w h] is the columns from [c0] to [c1] and rows from [r0] to
   [r1], exclusive, of an image of [w] by [h] pixels over [box] whose cells,
   mapped through [m], may meet [cut], if any does. *)
let window r m box w h =
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

(* The walk *)

type paint = Own | Inherit | Fixed of Color.t

type ctx = {
  m : Affine.t;
  cut : Box2.t;
  mag : float;
  d : int;
  fills : paint;
  strokes : paint;
  fill_set : bool;
  stroke_set : bool;
  pen : float;
  pen_set : bool;
}

let page r =
  let w = Renderable.w r and h = Renderable.h r in
  let margin = Float.max w h in
  {
    m = Affine.id;
    cut =
      Box2.v (-.margin) (-.margin) (w +. (2. *. margin)) (h +. (2. *. margin));
    mag = 1.;
    d = decimals 1.;
    fills = Own;
    strokes = Own;
    fill_set = false;
    stroke_set = false;
    pen = 1.;
    pen_set = false;
  }

let box ctx p =
  match Instances.bounds (Picture.transform ctx.m p) with
  | None -> None
  | Some b ->
      Box2.inter (grown b (Instances.reach ctx.m ctx.pen p +. 1.)) ctx.cut

type stroke = {
  style : Stroke.t;
  pen : pen;
  frame : Affine.t option;
  dashed : bool;
}

type image = {
  pixels : Nx.uint8_t;
  window : int * int * int * int;
  x : float;
  y : float;
  w : float;
  h : float;
}

type stamp = {
  picture : Picture.t;
  extent : Box2.t;
  pens : pen list;
  decimals : int;
  translucent : bool;
  scaled : bool;
  fills : Color.t array option;
  strokes : Color.t array option;
  rows : int array option;
}

let scale_decimals ctx s = finer 1 (decimals (ctx.mag *. s))

let scale_pen k pen =
  {
    width = pen.width *. k;
    dash = List.map (fun d -> d *. k) pen.dash;
    offset = pen.offset *. k;
  }

type ('b, 'd) target = {
  fill : 'b -> ctx -> Picture.rule -> Color.t -> Path.t -> unit;
  stroke :
    'b ->
    ctx ->
    stroke ->
    Color.t ->
    (piece:(float -> unit) -> sink -> unit) ->
    unit;
  glyphs : 'b -> ctx -> Color.t -> P2.t -> Run.t -> unit;
  image : 'b -> ctx -> image -> unit;
  clip : 'b -> ctx -> Picture.rule -> Path.t -> ('b -> unit) -> unit;
  opacity : 'b -> ctx -> float -> Picture.t -> ('b -> unit) -> unit;
  tag : 'b -> ctx -> Picture.tag -> ('b -> unit) -> unit;
  carry : ctx -> stamp -> float -> float;
  define : ctx -> Box2.t -> ('b -> unit) -> 'd;
  use : 'b -> ctx -> stamp -> 'd -> int -> P2.t -> float -> unit;
  instance : 'b -> stamp -> int -> ('b -> unit) -> unit;
}

let pen_equal a b =
  Float.equal a.width b.width
  && List.equal Float.equal a.dash b.dash
  && Float.equal a.offset b.offset

(* [pens m k p] is the distinct pens of the strokes of [p], as [pen] writes them
   under [m] and the transforms within [p]. *)
let pens m k p =
  List.rev
    (Instances.fold_strokes
       (fun m s found ->
         let q = pen m k s in
         if List.exists (pen_equal q) found then found else q :: found)
       (Affine.linear m) p [])

let scales_within =
  Instances.exists (function
    | Stamp { scales = Some _; _ } -> true
    | _ -> false)

let stroke t b ctx s color path =
  let lin = Affine.linear ctx.m in
  let pen = pen ctx.m ctx.pen s in
  (* A pen the frame stretches unevenly is written under [u], the frame's linear
     part scaled to stretch nothing more than the page does, rounded as written,
     and its points under the inverse of [u]. *)
  let frame =
    if is_similar lin then None
    else
      let u = unit lin in
      Some
        {
          u with
          xx = rounded 15 u.xx;
          yx = rounded 15 u.yx;
          xy = rounded 15 u.xy;
          yy = rounded 15 u.yy;
        }
  in
  match Option.map Affine.invert frame with
  | Some None -> ()
  | out ->
      let out = Option.join out in
      let dashed = is_dashed ctx.d pen in
      let reach = Stroke.reach s *. (ctx.pen *. Affine.stretch ctx.m) in
      let cut = grown ctx.cut reach in
      t.stroke b ctx { style = s; pen; frame; dashed } color (fun ~piece sink ->
          outline ?out ctx.m cut
            ~points:(Stroke.cap s <> `Square)
            ~dashed path ~piece sink)

let image t b ctx box pixels =
  let shape = Nx.shape pixels in
  let h = shape.(0) and w = shape.(1) in
  match window ctx.cut ctx.m box w h with
  | None -> ()
  | Some ((c0, r0, c1, r1) as window) ->
      let cw = Box2.w box /. float w and ch = Box2.h box /. float h in
      let x = Box2.minx box +. (float c0 *. cw)
      and y = Box2.miny box +. (float r0 *. ch) in
      let w = float (c1 - c0) *. cw and h = float (r1 - r0) *. ch in
      t.image b ctx { pixels; window; x; y; w; h }

let fixed c = Fixed c

let rec walk t ctx b (p : Picture.t) =
  match p with
  | Empty -> ()
  | Fill { rule; color; path } -> t.fill b ctx rule color path
  | Stroke { stroke = s; color; path } -> stroke t b ctx s color path
  | Glyphs { color; at; run } -> t.glyphs b ctx color at run
  | Image { box; pixels } -> image t b ctx box pixels
  | Group ps -> List.iter (walk t ctx b) ps
  | Clip { rule; path; picture } ->
      t.clip b ctx rule path (fun b -> walk t ctx b picture)
  | Transform { m; picture } ->
      (* Transforms that compose beyond the range of floats, or to a map with no
         inverse, leave nothing to write. *)
      let m = Affine.(ctx.m * m) in
      if Affine.invert m <> None then walk t { ctx with m } b picture
  | Opacity { opacity; picture } ->
      t.opacity b ctx opacity picture (fun b -> walk t ctx b picture)
  | Tag { tag = { rows = Rows a; _ } as tag; picture = Stamp s } ->
      t.tag b ctx { tag with rows = Rows [||] } (fun b ->
          stamp t ctx b ~rows:(Some a) s.picture s.xs s.ys s.scales s.fills
            s.strokes)
  | Tag { tag; picture } -> t.tag b ctx tag (fun b -> walk t ctx b picture)
  | Stamp s ->
      stamp t ctx b ~rows:None s.picture s.xs s.ys s.scales s.fills s.strokes

and stamp t ctx b ~rows p xs ys scales fills strokes =
  let lin = Affine.linear ctx.m in
  match Instances.bounds (Picture.transform lin p) with
  | None -> ()
  | Some extent ->
      let n = Array.length xs in
      let scale i = match scales with None -> 1. | Some a -> a.(i) in
      let shown i =
        let s = scale i in
        Float.is_finite xs.(i)
        && Float.is_finite ys.(i)
        && Float.is_finite s && s > 0.
      in
      let scaled = Option.is_some scales in
      let pens = if scaled then pens lin ctx.pen p else [] in
      (* Instances keep the pens of [p], which reach this far around them. *)
      let kept = Instances.reach ctx.m ctx.pen p in
      let far =
        Float.max
          (Float.max
             (Float.abs (Box2.minx extent))
             (Float.abs (Box2.maxx extent)))
          (Float.max
             (Float.abs (Box2.miny extent))
             (Float.abs (Box2.maxy extent)))
      in
      let translucent =
        List.exists
          (Array.exists (fun c -> Color.alpha c < 1.))
          (List.filter_map Fun.id [ fills; strokes ])
      in
      let st =
        {
          picture = p;
          extent;
          pens;
          decimals = finer 1 (decimals (ctx.mag *. far));
          translucent;
          scaled;
          fills;
          strokes;
          rows;
        }
      in
      (* Instance [i] in full: its picture mapped to the frame, with its own
         colours and pens. *)
      let whole i =
        let s = scale i in
        let m = Affine.(ctx.m * translate xs.(i) ys.(i) * scale s s) in
        let at = P2.transform ctx.m (P2.v xs.(i) ys.(i)) in
        if
          Instances.shows ctx.cut ~reach:kept at s extent
          && Affine.invert m <> None
        then
          let ictx =
            {
              ctx with
              m;
              fills = Instances.color fills i ~own:fixed ctx.fills;
              strokes = Instances.color strokes i ~own:fixed ctx.strokes;
              pen = ctx.pen /. s;
            }
          in
          t.instance b st i (fun b -> walk t ictx b p)
      in
      (* NaN where the definition cannot carry an instance's scale. Instances of
         a stamp that does not scale them all carry the same. *)
      let carried = Array.make n Float.nan in
      if not (scaled && (scales_within p || List.length pens > 1)) then begin
        let carry = t.carry ctx st in
        let one = if scaled then Float.nan else carry 1. in
        for i = 0 to n - 1 do
          if shown i then
            carried.(i) <- (if scaled then carry (scale i) else one)
        done
      end;
      let largest = ref 1. and smallest = ref 1. in
      Array.iter
        (fun s ->
          if not (Float.is_nan s) then begin
            largest := Float.max !largest s;
            smallest := Float.min !smallest s
          end)
        carried;
      let mag = ctx.mag *. !largest in
      let dctx =
        {
          m = lin;
          cut = Instances.everywhere;
          mag;
          d = finer 1 (decimals mag);
          fills = (if fills = None then ctx.fills else Inherit);
          strokes = (if strokes = None then ctx.strokes else Inherit);
          fill_set = ctx.fill_set || fills <> None;
          stroke_set = ctx.stroke_set || strokes <> None;
          (* The widest pen an instance sets: pens kept while instances shrink
             reach beyond the picture's box, and the boxes of the definition and
             of the groups within it hold them. *)
          pen = ctx.pen /. !smallest;
          pen_set = ctx.pen_set || scaled;
        }
      in
      let def =
        if Array.for_all Float.is_nan carried then None
        else
          Option.map
            (fun bx -> t.define dctx bx (fun b -> walk t dctx b p))
            (box dctx p)
      in
      for i = 0 to n - 1 do
        if shown i then
          (* Boxed once for the two calls that take it. *)
          let sw = Sys.opaque_identity carried.(i) in
          if Float.is_nan sw then whole i
          else
            match def with
            | None -> ()
            | Some def ->
                let at = P2.transform ctx.m (P2.v xs.(i) ys.(i)) in
                if Instances.shows ctx.cut ~reach:kept at sw extent then
                  t.use b ctx st def i at sw
      done
