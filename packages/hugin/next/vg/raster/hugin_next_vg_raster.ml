(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_next_gg
open Hugin_next_font
open Hugin_next_vg

type ctx = {
  cover : Cover.t;
  mutable outlines : (Font.t * (int, Path.t) Hashtbl.t) list;
}

(* What a stamp replaces in the leaves below it: their fill and stroke colours,
   if given, and their pens, multiplied by [pen]. *)
type style = { fills : Color.t option; strokes : Color.t option; pen : float }

let plain = { fills = None; strokes = None; pen = 1. }

let premultiplied c =
  let a = Color.alpha c in
  ( Color.r c *. a *. 255.,
    Color.g c *. a *. 255.,
    Color.b c *. a *. 255.,
    a *. 255. )

let outline ctx font g =
  let table =
    match List.assq_opt font ctx.outlines with
    | Some t -> t
    | None ->
        let t = Hashtbl.create 64 in
        ctx.outlines <- (font, t) :: ctx.outlines;
        t
  in
  match Hashtbl.find_opt table g with
  | Some p -> p
  | None ->
      let p = Font.outline font g in
      Hashtbl.add table g p;
      p

(* Coverage of leaves *)

(* [area cover m path] deposits the area of [path] mapped through [m], its
   subpaths closed. *)
let area cover m path =
  (* [start x; start y; current x; current y] *)
  let pt = Array.make 4 0. and open_ = ref false in
  let close () =
    if !open_ then Cover.line cover pt.(2) pt.(3) pt.(0) pt.(1);
    open_ := false
  in
  Path.flatten m
    ~move:(fun () x y ->
      close ();
      pt.(0) <- x;
      pt.(1) <- y;
      pt.(2) <- x;
      pt.(3) <- y;
      open_ := true)
    ~line:(fun () x y ->
      Cover.line cover pt.(2) pt.(3) x y;
      pt.(2) <- x;
      pt.(3) <- y)
    ~close () path;
  close ()

let glyph_map m at run i =
  let s = Run.size run in
  Affine.(
    m * translate (P2.x at +. Run.x run i) (P2.y at +. Run.y run i) * scale s s)

(* Images *)

(* [samples clip m box pixels] is the window [(x0, y0, w, h)] of the device
   pixels within [clip] that the image [pixels] over [box], mapped through [m],
   reaches, and the premultiplied colour of each, four floats in \[0;255\] per
   pixel, row by row. A pixel takes the image pixel under its centre, or, where
   the image is shown smaller than its pixels, the average of up to [4] by [4]
   samples. *)
let samples (clip : Surface.clip) m box pixels =
  let shape = Nx.shape pixels in
  let rows = shape.(0) and cols = shape.(1) and chans = shape.(2) in
  let corners =
    List.map
      (fun (x, y) -> P2.transform m (P2.v x y))
      [
        (Box2.minx box, Box2.miny box);
        (Box2.maxx box, Box2.miny box);
        (Box2.maxx box, Box2.maxy box);
        (Box2.minx box, Box2.maxy box);
      ]
  in
  let fold f init = List.fold_left (fun acc p -> f acc p) init corners in
  let minx = fold (fun a p -> Float.min a (P2.x p)) infinity in
  let maxx = fold (fun a p -> Float.max a (P2.x p)) neg_infinity in
  let miny = fold (fun a p -> Float.min a (P2.y p)) infinity in
  let maxy = fold (fun a p -> Float.max a (P2.y p)) neg_infinity in
  (* A corner mapped beyond [max_float] is infinite or NaN, which clamps to
     [lo], and the window is then empty or whole. *)
  let clamp lo hi v =
    if not (v >= float lo) then lo
    else if v > float hi then hi
    else int_of_float v
  in
  let x0 = clamp clip.x0 clip.x1 (Float.floor minx)
  and x1 = clamp clip.x0 clip.x1 (Float.ceil maxx) in
  let y0 = clamp clip.y0 clip.y1 (Float.floor miny)
  and y1 = clamp clip.y0 clip.y1 (Float.ceil maxy) in
  let w = x1 - x0 and h = y1 - y0 in
  match Affine.invert m with
  | None -> (x0, y0, 0, 0, [||])
  | Some inv ->
      let data =
        Bigarray.reshape_1 (Nx.to_bigarray pixels) (rows * cols * chans)
      in
      let bw = Box2.w box and bh = Box2.h box in
      let shown_w = Float.hypot (m.xx *. bw) (m.yx *. bw) in
      let shown_h = Float.hypot (m.xy *. bh) (m.yy *. bh) in
      let ratio =
        Float.max
          (float cols /. Float.max 1. shown_w)
          (float rows /. Float.max 1. shown_h)
      in
      let k = Int.max 1 (Int.min 4 (int_of_float (Float.ceil ratio))) in
      let n = float (k * k) in
      let buf = Array.make (w * h * 4) 0. in
      let get i = float (Bigarray.Array1.unsafe_get data i) in
      for py = y0 to y1 - 1 do
        for px = x0 to x1 - 1 do
          let r = ref 0. and g = ref 0. and b = ref 0. and a = ref 0. in
          for j = 0 to k - 1 do
            for i = 0 to k - 1 do
              let dx = float px +. ((float i +. 0.5) /. float k) in
              let dy = float py +. ((float j +. 0.5) /. float k) in
              let ux = (inv.xx *. dx) +. (inv.xy *. dy) +. inv.x0 in
              let uy = (inv.yx *. dx) +. (inv.yy *. dy) +. inv.y0 in
              let u = (ux -. Box2.minx box) /. bw
              and v = (uy -. Box2.miny box) /. bh in
              if u >= 0. && u <= 1. && v >= 0. && v <= 1. then begin
                let col = Int.min (cols - 1) (int_of_float (u *. float cols)) in
                let row = Int.min (rows - 1) (int_of_float (v *. float rows)) in
                let at = ((row * cols) + col) * chans in
                match chans with
                | 1 ->
                    let l = get at in
                    r := !r +. l;
                    g := !g +. l;
                    b := !b +. l;
                    a := !a +. 255.
                | 3 ->
                    r := !r +. get at;
                    g := !g +. get (at + 1);
                    b := !b +. get (at + 2);
                    a := !a +. 255.
                | _ ->
                    let al = get (at + 3) in
                    let f = al /. 255. in
                    r := !r +. (get at *. f);
                    g := !g +. (get (at + 1) *. f);
                    b := !b +. (get (at + 2) *. f);
                    a := !a +. al
              end
            done
          done;
          let o = (((py - y0) * w) + (px - x0)) * 4 in
          buf.(o) <- !r /. n;
          buf.(o + 1) <- !g /. n;
          buf.(o + 2) <- !b /. n;
          buf.(o + 3) <- !a /. n
        done
      done;
      (x0, y0, w, h, buf)

(* Clips *)

(* [pixel_rect m path] is the pixel rectangle of [path] mapped through [m] if it
   is one closed subpath around a rectangle on pixel boundaries. *)
let pixel_rect m path =
  let pts = ref [] and moves = ref 0 and closes = ref 0 in
  Path.flatten m
    ~move:(fun () x y ->
      incr moves;
      pts := (x, y) :: !pts)
    ~line:(fun () x y -> pts := (x, y) :: !pts)
    ~close:(fun () -> incr closes)
    () path;
  match (!moves, !closes, !pts) with
  | 1, 1, [ (x0, y0); (x1, y1); (x2, y2); (x3, y3) ] ->
      (* Each side is horizontal or vertical, and opposite sides of one kind:
         the two kinds alternate around a rectangle, or all sides are of one
         kind and the rectangle is flat. The corners are integers that an [int]
         holds. *)
      let along ax ay bx by = ax = bx <> (ay = by) in
      let integral v = Float.is_integer v && Float.abs v <= 1e15 in
      if
        List.for_all integral [ x0; y0; x1; y1; x2; y2; x3; y3 ]
        && along x0 y0 x1 y1 && along x1 y1 x2 y2 && along x2 y2 x3 y3
        && along x3 y3 x0 y0
        && x0 = x1 = (x2 = x3)
      then
        let lo a b = int_of_float (Float.min a b)
        and hi a b = int_of_float (Float.max a b) in
        Some (lo x0 x2, lo y0 y2, hi x0 x2, hi y0 y2)
      else None
  | _ -> None

let clip_by ctx ~w ~h (clip : Surface.clip) m rule path : Surface.clip =
  match pixel_rect m path with
  | Some (rx0, ry0, rx1, ry1) ->
      let x0 = Int.max clip.x0 rx0 and y0 = Int.max clip.y0 ry0 in
      let x1 = Int.min clip.x1 rx1 and y1 = Int.min clip.y1 ry1 in
      if x1 <= x0 || y1 <= y0 then Surface.shut
      else
        let mask =
          Option.map
            (fun _ ->
              let cw = x1 - x0 and ch = y1 - y0 in
              Array.init (cw * ch) (fun i ->
                  Surface.mask_at clip (x0 + (i mod cw)) (y0 + (i / cw))))
            clip.mask
        in
        { x0; y0; x1; y1; mask }
  | None ->
      Cover.start ctx.cover ~w ~h clip;
      area ctx.cover m path;
      let x0, y0, x1, y1 = Cover.touched ctx.cover in
      if x1 <= x0 || y1 <= y0 then begin
        Cover.clear ctx.cover;
        Surface.shut
      end
      else begin
        let mask = Array.make ((x1 - x0) * (y1 - y0)) 0. in
        Cover.take ctx.cover clip rule mask ~ox:x0 ~oy:y0 ~dw:(x1 - x0);
        { x0; y0; x1; y1; mask = Some mask }
      end

(* [extent clip m ~pen p] is the device box of [p] under [m] within [clip], with
   a pixel of margin, its pens' widths multiplied by [pen]. *)
let extent (clip : Surface.clip) m ~pen p =
  match Instances.bounds (Picture.transform m p) with
  | None -> None
  | Some b ->
      let within lo hi v =
        int_of_float (Float.min (float hi) (Float.max (float lo) v))
      in
      (* [bounds] counts each pen once: grow by what [pen] adds to it. *)
      let g =
        1. +. if pen > 1. then (pen -. 1.) *. Instances.reach m 1. p else 0.
      in
      let x0 = within clip.x0 clip.x1 (Float.floor (Box2.minx b -. g)) in
      let x1 = within clip.x0 clip.x1 (Float.ceil (Box2.maxx b +. g)) in
      let y0 = within clip.y0 clip.y1 (Float.floor (Box2.miny b -. g)) in
      let y1 = within clip.y0 clip.y1 (Float.ceil (Box2.maxy b +. g)) in
      if x1 <= x0 || y1 <= y0 then None else Some (x0, y0, x1, y1)

(* Planes

   A stamp draws its picture once per quarter-pixel phase, as planes: the
   coverage of each leaf, or the pixels of each image, in order, on a tile. Each
   instance then composites the planes at its position, with its own colours, as
   drawing the picture there would. *)

type paint = Fills | Strokes | Fixed of (float * float * float * float)

type plane =
  | Coverage of { paint : paint; cov : float array }
  | Pixels of float array  (** Premultiplied, four floats per pixel. *)
  | Layer of { alpha : float; planes : plane list; scratch : Surface.t }

(* [planar p] is [true] iff [p] holds no stamp, which planes cannot hold. *)
let rec planar (p : Picture.t) =
  match p with
  | Stamp _ -> false
  | Group ps -> List.for_all planar ps
  | Clip { picture; _ }
  | Transform { picture; _ }
  | Opacity { picture; _ }
  | Tag { picture; _ } ->
      planar picture
  | Empty | Fill _ | Stroke _ | Glyphs _ | Image _ -> true

(* [planes ctx ~w ~h clip m ~fills ~strokes style p acc] adds to [acc], in
   reverse order, the planes of [p] on a tile of [w] by [h] pixels. Fills and
   strokes take the instance's colours if [fills] and [strokes]. *)
let rec planes ctx ~w ~h clip m ~fills ~strokes style p acc =
  let coverage paint rule acc deposit =
    Cover.start ctx.cover ~w ~h clip;
    deposit ();
    let cov = Array.make (w * h) 0. in
    Cover.take ctx.cover clip rule cov ~ox:0 ~oy:0 ~dw:w;
    Coverage { paint; cov } :: acc
  in
  let fill_paint color =
    if fills then Fills
    else Fixed (premultiplied (Option.value style.fills ~default:color))
  in
  if Surface.is_shut clip then acc
  else
    match (p : Picture.t) with
    | Empty -> acc
    | Fill { rule; color; path } ->
        coverage (fill_paint color) rule acc (fun () -> area ctx.cover m path)
    | Stroke { stroke; color; path } ->
        let paint =
          if strokes then Strokes
          else Fixed (premultiplied (Option.value style.strokes ~default:color))
        in
        coverage paint `Nonzero acc (fun () ->
            Pen.stroke ctx.cover clip m ~pen:style.pen stroke path)
    | Glyphs { color; at; run } ->
        let paint = fill_paint color in
        let acc = ref acc in
        for i = 0 to Run.length run - 1 do
          let path = outline ctx (Run.font run) (Run.glyph run i) in
          acc :=
            coverage paint `Nonzero !acc (fun () ->
                area ctx.cover (glyph_map m at run i) path)
        done;
        !acc
    | Image { box; pixels } ->
        let x0, y0, iw, ih, buf = samples clip m box pixels in
        let px = Array.make (w * h * 4) 0. in
        for y = 0 to ih - 1 do
          for x = 0 to iw - 1 do
            let k = Surface.mask_at clip (x0 + x) (y0 + y) in
            let s = ((y * iw) + x) * 4 and d = (((y0 + y) * w) + x0 + x) * 4 in
            for c = 0 to 3 do
              px.(d + c) <- buf.(s + c) *. k
            done
          done
        done;
        Pixels px :: acc
    | Group ps ->
        List.fold_left
          (fun acc p -> planes ctx ~w ~h clip m ~fills ~strokes style p acc)
          acc ps
    | Clip { rule; path; picture } ->
        let clip = clip_by ctx ~w ~h clip m rule path in
        planes ctx ~w ~h clip m ~fills ~strokes style picture acc
    | Transform { m = m'; picture } -> (
        (* As [Picture.transform], a map with no inverse paints nothing, and so
           does one that overflows the range of floats. *)
        let m = Affine.(m * m') in
        match Affine.invert m with
        | None -> acc
        | Some _ -> planes ctx ~w ~h clip m ~fills ~strokes style picture acc)
    | Opacity { opacity = 0.; _ } -> acc
    | Opacity { opacity; picture } ->
        let inner = planes ctx ~w ~h clip m ~fills ~strokes style picture [] in
        Layer
          {
            alpha = opacity;
            planes = List.rev inner;
            scratch = Surface.create w h;
          }
        :: acc
    | Tag { picture; _ } ->
        planes ctx ~w ~h clip m ~fills ~strokes style picture acc
    | Stamp _ -> assert false

(* [composite t clip ox oy ~tw ~th planes fill stroke] composites [planes], a
   tile of [tw] by [th] pixels with its top left at [(ox, oy)] of [t], with the
   premultiplied colours [fill] and [stroke] for the leaves that take them. *)
let rec composite (t : Surface.t) (clip : Surface.clip) ox oy ~tw ~th planes
    fill stroke =
  let x0 = Int.max clip.x0 ox and y0 = Int.max clip.y0 oy in
  let x1 = Int.min clip.x1 (ox + tw) and y1 = Int.min clip.y1 (oy + th) in
  let plane = function
    | Coverage { paint; cov } ->
        let sr, sg, sb, sa =
          match paint with Fills -> fill | Strokes -> stroke | Fixed c -> c
        in
        for y = y0 to y1 - 1 do
          let row = (y - oy) * tw in
          for x = x0 to x1 - 1 do
            let k = Array.unsafe_get cov (row + x - ox) in
            if k > 0.0005 then
              let k = k *. Surface.mask_at clip x y in
              if k > 0. then
                Surface.blend t.px (((y * t.w) + x) * 4) sr sg sb sa k
          done
        done
    | Pixels px ->
        for y = y0 to y1 - 1 do
          for x = x0 to x1 - 1 do
            let i = (((y - oy) * tw) + x - ox) * 4 in
            let sa = Array.unsafe_get px (i + 3) in
            if sa > 0. then
              let k = Surface.mask_at clip x y in
              if k > 0. then
                Surface.blend t.px
                  (((y * t.w) + x) * 4)
                  (Array.unsafe_get px i)
                  (Array.unsafe_get px (i + 1))
                  (Array.unsafe_get px (i + 2))
                  sa k
          done
        done
    | Layer { alpha; planes; scratch } ->
        Surface.clear scratch;
        composite scratch (Surface.whole scratch) 0 0 ~tw ~th planes fill stroke;
        Surface.composite t clip scratch ox oy alpha
  in
  if x0 < x1 && y0 < y1 then List.iter plane planes

(* Drawing *)

(* Tiles larger than this are not worth drawing in sixteen phases. *)
let max_tile = 1 lsl 16

let rec draw ctx (t : Surface.t) clip m style (p : Picture.t) =
  if not (Surface.is_shut clip) then
    match p with
    | Empty -> ()
    | Fill { rule; color; path } ->
        Cover.start ctx.cover ~w:t.w ~h:t.h clip;
        area ctx.cover m path;
        paint ctx t clip rule (Option.value style.fills ~default:color)
    | Stroke { stroke; color; path } ->
        Cover.start ctx.cover ~w:t.w ~h:t.h clip;
        Pen.stroke ctx.cover clip m ~pen:style.pen stroke path;
        paint ctx t clip `Nonzero (Option.value style.strokes ~default:color)
    | Glyphs { color; at; run } ->
        let color = Option.value style.fills ~default:color in
        for i = 0 to Run.length run - 1 do
          let path = outline ctx (Run.font run) (Run.glyph run i) in
          Cover.start ctx.cover ~w:t.w ~h:t.h clip;
          area ctx.cover (glyph_map m at run i) path;
          paint ctx t clip `Nonzero color
        done
    | Image { box; pixels } ->
        let x0, y0, iw, ih, buf = samples clip m box pixels in
        for y = 0 to ih - 1 do
          for x = 0 to iw - 1 do
            let i = ((y * iw) + x) * 4 in
            let sa = buf.(i + 3) in
            if sa > 0. then
              let k = Surface.mask_at clip (x0 + x) (y0 + y) in
              if k > 0. then
                Surface.blend t.px
                  ((((y0 + y) * t.w) + x0 + x) * 4)
                  buf.(i)
                  buf.(i + 1)
                  buf.(i + 2)
                  sa k
          done
        done
    | Group ps -> List.iter (draw ctx t clip m style) ps
    | Clip { rule; path; picture } ->
        draw ctx t (clip_by ctx ~w:t.w ~h:t.h clip m rule path) m style picture
    | Transform { m = m'; picture } -> (
        let m = Affine.(m * m') in
        match Affine.invert m with
        | None -> ()
        | Some _ -> draw ctx t clip m style picture)
    | Opacity { opacity = 0.; _ } -> ()
    | Opacity { opacity; picture } -> (
        match extent clip m ~pen:style.pen picture with
        | None -> ()
        | Some (x0, y0, x1, y1) ->
            let layer = Surface.create (x1 - x0) (y1 - y0) in
            let m = Affine.(translate (float (-x0)) (float (-y0)) * m) in
            draw ctx layer (Surface.whole layer) m style picture;
            Surface.composite t clip layer x0 y0 opacity)
    | Stamp { picture; xs; ys; scales; fills; strokes } ->
        stamp ctx t clip m style picture xs ys scales fills strokes
    | Tag { picture; _ } -> draw ctx t clip m style picture

and paint ctx t clip rule color =
  let sr, sg, sb, sa = premultiplied color in
  Cover.paint ctx.cover t clip rule sr sg sb sa

and stamp ctx t clip m style picture xs ys scales fills strokes =
  let lin = Affine.linear m in
  match Instances.bounds (Picture.transform lin picture) with
  | None -> ()
  | Some extent -> (
      let instance i s =
        {
          fills = Instances.color fills i ~own:Option.some style.fills;
          strokes = Instances.color strokes i ~own:Option.some style.strokes;
          pen = style.pen /. s;
        }
      in
      (* An instance's device position, rounded to a quarter of a pixel, in
         quarters, or [None] if it is not finite or far beyond the window. *)
      let quarters i =
        let x = xs.(i) and y = ys.(i) in
        let dx = (m.xx *. x) +. (m.xy *. y) +. m.x0 in
        let dy = (m.yx *. x) +. (m.yy *. y) +. m.y0 in
        if Float.abs dx < 1e15 && Float.abs dy < 1e15 then
          Some
            ( int_of_float (Float.round (dx *. 4.)),
              int_of_float (Float.round (dy *. 4.)) )
        else None
      in
      let window =
        Box2.v (float clip.x0) (float clip.y0)
          (float (clip.x1 - clip.x0))
          (float (clip.y1 - clip.y0))
      in
      (* Instances keep the pens of the picture, which reach this far around
         them. *)
      let kept = Instances.reach lin style.pen picture in
      let at qx qy = P2.v (float qx /. 4.) (float qy /. 4.) in
      let in_full () =
        for i = 0 to Array.length xs - 1 do
          let s = match scales with None -> 1. | Some a -> a.(i) in
          match quarters i with
          | Some (qx, qy)
            when Instances.shows window ~reach:kept (at qx qy) s extent -> (
              let at = Affine.translate (float qx /. 4.) (float qy /. 4.) in
              let m = Affine.(at * lin * scale s s) in
              (* As [Picture.transform], a map with no inverse paints nothing:
                 that of a scale that is not finite or is 0 among them. *)
              match Affine.invert m with
              | None -> ()
              | Some _ -> draw ctx t clip m (instance i s) picture)
          | _ -> ()
        done
      in
      (* A tile is bounded by the picture's pens, which those of an instance
         shrunk by an enclosing stamp's scale outgrow. It is at least three
         pixels wider and higher than the extent, which is tested in floats
         first, since the extent's size may overflow an [int]. *)
      let tile =
        match scales with
        | Some _ -> None
        | None when style.pen > 1. || not (planar picture) -> None
        | None
          when (Box2.w extent +. 3.) *. (Box2.h extent +. 3.) > float max_tile
          ->
            None
        | None ->
            let bx0 = int_of_float (Float.floor (Box2.minx extent)) - 1 in
            let by0 = int_of_float (Float.floor (Box2.miny extent)) - 1 in
            let tw = int_of_float (Float.ceil (Box2.maxx extent)) + 2 - bx0 in
            let th = int_of_float (Float.ceil (Box2.maxy extent)) + 2 - by0 in
            if tw * th <= max_tile then Some (bx0, by0, tw, th) else None
      in
      match tile with
      | None -> in_full ()
      | Some (bx0, by0, tw, th) ->
          let phases = Array.make 16 None in
          let phase fx fy =
            let k = (fy * 4) + fx in
            match phases.(k) with
            | Some planes -> planes
            | None ->
                let tile =
                  { Surface.x0 = 0; y0 = 0; x1 = tw; y1 = th; mask = None }
                in
                let m =
                  Affine.(
                    translate
                      ((float fx /. 4.) -. float bx0)
                      ((float fy /. 4.) -. float by0)
                    * lin)
                in
                let planes =
                  List.rev
                    (planes ctx ~w:tw ~h:th tile m ~fills:(fills <> None)
                       ~strokes:(strokes <> None) style picture [])
                in
                phases.(k) <- Some planes;
                planes
          in
          let rgba a i =
            Instances.color a i ~own:premultiplied (0., 0., 0., 0.)
          in
          (* Positions beyond floats show everywhere: the window clips them. *)
          let ps =
            Option.value ~default:Instances.everywhere
              (Instances.positions window ~reach:kept extent)
          in
          let px0 = Box2.minx ps and px1 = Box2.maxx ps in
          let py0 = Box2.miny ps and py1 = Box2.maxy ps in
          let shows qx qy =
            let x = float qx /. 4. and y = float qy /. 4. in
            x >= px0 && x <= px1 && y >= py0 && y <= py1
          in
          for i = 0 to Array.length xs - 1 do
            match quarters i with
            | Some (qx, qy) when shows qx qy ->
                composite t clip
                  ((qx asr 2) + bx0)
                  ((qy asr 2) + by0)
                  ~tw ~th
                  (phase (qx land 3) (qy land 3))
                  (rgba fills i) (rgba strokes i)
            | _ -> ()
          done)

(* Rendering *)

let err fmt = Printf.ksprintf invalid_arg ("Hugin_next_vg_raster." ^^ fmt)
let max_pixels = 2147483647.

let render ~density r =
  if not (density > 0. && Float.is_finite density) then
    err "render: invalid density %g" density;
  let w = Float.round (Renderable.w r *. density) in
  let h = Float.round (Renderable.h r *. density) in
  if not (w >= 1. && w <= max_pixels && h >= 1. && h <= max_pixels) then
    err "render: a page of %g by %g pixels" w h;
  let s = Surface.create (int_of_float w) (int_of_float h) in
  let ctx = { cover = Cover.create (); outlines = [] } in
  draw ctx s (Surface.whole s)
    (Affine.scale density density)
    plain (Renderable.picture r);
  Surface.to_straight s

let png ~density r =
  let ppm = Float.round (density *. 72. /. 0.0254) in
  if ppm > max_pixels || ppm < 1. then
    err "png: density %g is beyond a PNG's resolution" density;
  Nx_io.encode_png ~dpi:(density *. 72.) ~srgb:true (render ~density r)
