(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_vg
open Bigarray

(* Pixels are premultiplied RGBA8 in row-major order. *)
type pixels = (int, int8_unsigned_elt, c_layout) Array1.t
type canvas = { w : int; h : int; px : pixels }

(* A clip is a pixel rectangle, [x1] and [y1] exclusive, and an optional
   per-pixel coverage over that rectangle. *)
type clip = {
  x0 : int;
  y0 : int;
  x1 : int;
  y1 : int;
  mask : float array option;
}

let clip_is_empty c = c.x1 <= c.x0 || c.y1 <= c.y0

let mask_at c x y =
  match c.mask with
  | None -> 1.
  | Some m -> Array.unsafe_get m (((y - c.y0) * (c.x1 - c.x0)) + (x - c.x0))

(* Compositing *)

(* [blend px i sr sg sb sa k] composites the premultiplied source [(sr, sg, sb,
   sa)], in 0..255, scaled by coverage [k] over pixel [i]. *)
let blend (px : pixels) i sr sg sb sa k =
  let inv = 1. -. (sa /. 255. *. k) in
  let mix s d = truncate ((s *. k) +. (float d *. inv) +. 0.5) in
  Array1.unsafe_set px i (mix sr (Array1.unsafe_get px i));
  Array1.unsafe_set px (i + 1) (mix sg (Array1.unsafe_get px (i + 1)));
  Array1.unsafe_set px (i + 2) (mix sb (Array1.unsafe_get px (i + 2)));
  Array1.unsafe_set px (i + 3) (mix sa (Array1.unsafe_get px (i + 3)))

let premultiplied (c : Color.t) =
  let a = Float.min 1. (Float.max 0. c.a) in
  let ch v = Float.min 1. (Float.max 0. v) *. a *. 255. in
  (ch c.r, ch c.g, ch c.b, a *. 255.)

(* Coverage *)

(* Signed-area accumulation: each edge deposits the area it sweeps into an
   accumulator whose running sum along a row is the winding number, with
   fractional values at the edge's pixels. *)
let draw_line acc aw h x0 y0 x1 y1 =
  if y0 <> y1 then begin
    let dir, x0, y0, x1, y1 =
      if y0 < y1 then (1., x0, y0, x1, y1) else (-1., x1, y1, x0, y0)
    in
    let dxdy = (x1 -. x0) /. (y1 -. y0) in
    let x = ref x0 in
    let ystart = if y0 < 0. then 0 else int_of_float y0 in
    if y0 < 0. then x := !x -. (y0 *. dxdy);
    let ystop = Int.min h (int_of_float (Float.ceil y1)) in
    for y = ystart to ystop - 1 do
      let row = y * aw in
      let dy = Float.min (float (y + 1)) y1 -. Float.max (float y) y0 in
      let xnext = !x +. (dxdy *. dy) in
      let d = dy *. dir in
      let xa, xb = if !x < xnext then (!x, xnext) else (xnext, !x) in
      let xa_floor = Float.floor xa in
      let xa_i = int_of_float xa_floor in
      let xb_ceil = Float.ceil xb in
      let xb_i = int_of_float xb_ceil in
      if xb_i <= xa_i + 1 then begin
        let xmf = (0.5 *. (!x +. xnext)) -. xa_floor in
        let i = row + xa_i in
        Array.unsafe_set acc i (Array.unsafe_get acc i +. d -. (d *. xmf));
        Array.unsafe_set acc (i + 1) (Array.unsafe_get acc (i + 1) +. (d *. xmf))
      end
      else begin
        let s = 1. /. (xb -. xa) in
        let xaf = xa -. xa_floor in
        let a0 = 0.5 *. s *. (1. -. xaf) *. (1. -. xaf) in
        let xbf = xb -. xb_ceil +. 1. in
        let am = 0.5 *. s *. xbf *. xbf in
        let i = row + xa_i in
        Array.unsafe_set acc i (Array.unsafe_get acc i +. (d *. a0));
        if xb_i = xa_i + 2 then
          Array.unsafe_set acc (i + 1)
            (Array.unsafe_get acc (i + 1) +. (d *. (1. -. a0 -. am)))
        else begin
          let a1 = s *. (1.5 -. xaf) in
          Array.unsafe_set acc (i + 1)
            (Array.unsafe_get acc (i + 1) +. (d *. (a1 -. a0)));
          for xi = xa_i + 2 to xb_i - 2 do
            Array.unsafe_set acc (row + xi)
              (Array.unsafe_get acc (row + xi) +. (d *. s))
          done;
          let a2 = a1 +. (float (xb_i - xa_i - 3) *. s) in
          let j = row + xb_i - 1 in
          Array.unsafe_set acc j
            (Array.unsafe_get acc j +. (d *. (1. -. a2 -. am)))
        end;
        let j = row + xb_i in
        Array.unsafe_set acc j (Array.unsafe_get acc j +. (d *. am))
      end;
      x := xnext
    done
  end

(* [coverage rule ~x0 ~y0 ~w ~h polys] is the per-pixel coverage of the closed
   polygons [polys] over the pixel rectangle, row-major. *)
let coverage rule ~x0 ~y0 ~w ~h (polys : Flatten.polyline list) =
  let aw = w + 2 in
  let acc = Array.make (aw * h) 0. in
  let fw = float w in
  let clampx x = if x < 0. then 0. else if x > fw then fw else x in
  List.iter
    (fun (p : Flatten.polyline) ->
      let n = Array.length p.xs in
      if n >= 2 then
        for i = 0 to n - 1 do
          let j = if i = n - 1 then 0 else i + 1 in
          draw_line acc aw h
            (clampx (p.xs.(i) -. float x0))
            (p.ys.(i) -. float y0)
            (clampx (p.xs.(j) -. float x0))
            (p.ys.(j) -. float y0)
        done)
    polys;
  let cov = Array.make (w * h) 0. in
  for y = 0 to h - 1 do
    let sum = ref 0. in
    let row = y * aw and out = y * w in
    for x = 0 to w - 1 do
      sum := !sum +. Array.unsafe_get acc (row + x);
      let c = Float.abs !sum in
      let c =
        match rule with
        | `Nonzero -> if c > 1. then 1. else c
        | `Evenodd ->
            let c = Float.rem c 2. in
            if c > 1. then 2. -. c else c
      in
      Array.unsafe_set cov (out + x) c
    done
  done;
  cov

(* [pixel_bounds clip polys] is the pixel rectangle of [polys] within [clip], or
   [None] when nothing is visible. *)
let pixel_bounds clip polys =
  match Flatten.bounds polys with
  | None -> None
  | Some (bx0, by0, bx1, by1) ->
      let x0 = Int.max clip.x0 (int_of_float (Float.floor bx0)) in
      let y0 = Int.max clip.y0 (int_of_float (Float.floor by0)) in
      let x1 = Int.min clip.x1 (int_of_float (Float.ceil bx1)) in
      let y1 = Int.min clip.y1 (int_of_float (Float.ceil by1)) in
      if x1 <= x0 || y1 <= y0 then None else Some (x0, y0, x1, y1)

let fill_polys canvas clip color rule polys =
  match pixel_bounds clip polys with
  | None -> ()
  | Some (x0, y0, x1, y1) ->
      let w = x1 - x0 and h = y1 - y0 in
      let cov = coverage rule ~x0 ~y0 ~w ~h polys in
      let sr, sg, sb, sa = premultiplied color in
      for y = 0 to h - 1 do
        for x = 0 to w - 1 do
          let c = Array.unsafe_get cov ((y * w) + x) in
          if c > 0.0005 then begin
            let px = x0 + x and py = y0 + y in
            let k = c *. mask_at clip px py in
            if k > 0. then
              blend canvas.px (((py * canvas.w) + px) * 4) sr sg sb sa k
          end
        done
      done

(* Clipping *)

(* [as_pixel_rect polys] is the pixel rectangle [polys] draws when it is a
   single axis-aligned rectangle on pixel boundaries. *)
let as_pixel_rect (polys : Flatten.polyline list) =
  match polys with
  | [ { xs; ys; _ } ] when Array.length xs = 4 || Array.length xs = 5 ->
      let n = Array.length xs in
      let integral v = Float.abs (v -. Float.round v) < 1e-6 in
      let aligned = ref true in
      for i = 0 to n - 1 do
        let j = (i + 1) mod n in
        if not (integral xs.(i) && integral ys.(i)) then aligned := false;
        if xs.(i) <> xs.(j) && ys.(i) <> ys.(j) then aligned := false
      done;
      if n = 5 && (xs.(0) <> xs.(4) || ys.(0) <> ys.(4)) then aligned := false;
      if !aligned then
        let x0 = Array.fold_left Float.min infinity xs
        and x1 = Array.fold_left Float.max neg_infinity xs in
        let y0 = Array.fold_left Float.min infinity ys
        and y1 = Array.fold_left Float.max neg_infinity ys in
        Some
          ( int_of_float (Float.round x0),
            int_of_float (Float.round y0),
            int_of_float (Float.round x1),
            int_of_float (Float.round y1) )
      else None
  | _ -> None

let intersect_clip clip polys =
  match as_pixel_rect polys with
  | Some (x0, y0, x1, y1) ->
      let nx0 = Int.max clip.x0 x0 and ny0 = Int.max clip.y0 y0 in
      let nx1 = Int.min clip.x1 x1 and ny1 = Int.min clip.y1 y1 in
      if nx1 <= nx0 || ny1 <= ny0 then
        { x0 = 0; y0 = 0; x1 = 0; y1 = 0; mask = None }
      else begin
        let mask =
          match clip.mask with
          | None -> None
          | Some _ ->
              let w = nx1 - nx0 in
              Some
                (Array.init
                   (w * (ny1 - ny0))
                   (fun i -> mask_at clip (nx0 + (i mod w)) (ny0 + (i / w))))
        in
        { x0 = nx0; y0 = ny0; x1 = nx1; y1 = ny1; mask }
      end
  | None -> (
      match pixel_bounds clip polys with
      | None -> { x0 = 0; y0 = 0; x1 = 0; y1 = 0; mask = None }
      | Some (x0, y0, x1, y1) ->
          let w = x1 - x0 and h = y1 - y0 in
          let cov = coverage `Nonzero ~x0 ~y0 ~w ~h polys in
          (match clip.mask with
          | None -> ()
          | Some _ ->
              for i = 0 to (w * h) - 1 do
                cov.(i) <-
                  cov.(i) *. mask_at clip (x0 + (i mod w)) (y0 + (i / w))
              done);
          { x0; y0; x1; y1; mask = Some cov })

(* Images *)

let draw_image canvas clip m ~x ~y ~w ~h data =
  let data = Nx.contiguous data in
  let shape = Nx.shape data in
  let rows = shape.(0) and cols = shape.(1) in
  let channels = if Array.length shape = 3 then shape.(2) else 1 in
  let buf = Nx.data data and off = Nx.offset data in
  let corners =
    Flatten.path m
      (Path.polygon [| x; x +. w; x +. w; x |] [| y; y; y +. h; y +. h |])
  in
  match
    (pixel_bounds clip corners, w > 0. && h > 0. && rows > 0 && cols > 0)
  with
  | None, _ | _, false -> ()
  | Some (px0, py0, px1, py1), true -> (
      match Affine.invert m with
      | exception Invalid_argument _ -> ()
      | inv ->
          (* Supersample when the image is shown smaller than its pixels. *)
          let dw = Float.hypot (m.xx *. w) (m.yx *. w)
          and dh = Float.hypot (m.xy *. h) (m.yy *. h) in
          let ratio =
            Float.max
              (float cols /. Float.max 1. dw)
              (float rows /. Float.max 1. dh)
          in
          let s = Int.max 1 (Int.min 4 (int_of_float (Float.ceil ratio))) in
          let samples = float (s * s) in
          let sample col row c =
            Nx_buffer.unsafe_get buf
              (off + (((row * cols) + col) * channels) + c)
          in
          for py = py0 to py1 - 1 do
            for px = px0 to px1 - 1 do
              let sr = ref 0. and sg = ref 0. and sb = ref 0. and sa = ref 0. in
              for j = 0 to s - 1 do
                for i = 0 to s - 1 do
                  let dx = float px +. ((float i +. 0.5) /. float s)
                  and dy = float py +. ((float j +. 0.5) /. float s) in
                  let ux, uy = Affine.apply inv dx dy in
                  let u = (ux -. x) /. w and v = (uy -. y) /. h in
                  if u >= 0. && u < 1. && v >= 0. && v < 1. then begin
                    let col = int_of_float (u *. float cols)
                    and row = int_of_float (v *. float rows) in
                    let r, g, b, a =
                      match channels with
                      | 1 ->
                          let v = sample col row 0 in
                          (v, v, v, 255)
                      | 3 ->
                          ( sample col row 0,
                            sample col row 1,
                            sample col row 2,
                            255 )
                      | _ ->
                          ( sample col row 0,
                            sample col row 1,
                            sample col row 2,
                            sample col row 3 )
                    in
                    let fa = float a /. 255. in
                    sr := !sr +. (float r *. fa);
                    sg := !sg +. (float g *. fa);
                    sb := !sb +. (float b *. fa);
                    sa := !sa +. float a
                  end
                done
              done;
              if !sa > 0. then begin
                let k = mask_at clip px py in
                if k > 0. then
                  blend canvas.px
                    (((py * canvas.w) + px) * 4)
                    (!sr /. samples) (!sg /. samples) (!sb /. samples)
                    (!sa /. samples) k
              end
            done
          done)

(* Bounds of a picture in device space, for stamping. *)

let union a b =
  match (a, b) with
  | None, x | x, None -> x
  | Some (ax0, ay0, ax1, ay1), Some (bx0, by0, bx1, by1) ->
      Some
        ( Float.min ax0 bx0,
          Float.min ay0 by0,
          Float.max ax1 bx1,
          Float.max ay1 by1 )

let corners_bounds m x0 y0 x1 y1 =
  Flatten.bounds
    (Flatten.path m (Path.polygon [| x0; x1; x1; x0 |] [| y0; y0; y1; y1 |]))

let linear_scale (m : Affine.t) =
  Float.sqrt (Float.abs ((m.xx *. m.yy) -. (m.xy *. m.yx)))

let rec bounds m (p : Picture.t) =
  match p with
  | Empty -> None
  | Fill { path; _ } -> Flatten.bounds (Flatten.path m path)
  | Stroke { stroke; path; _ } ->
      Flatten.bounds
        (Stroker.outline stroke ~scale:(linear_scale m) (Flatten.path m path))
  | Text { font; size; x; y; text; _ } ->
      let bx0, by0, bx1, by1 = Font.bounds font ~size text in
      if bx0 = bx1 && by0 = by1 then None
      else corners_bounds m (x +. bx0) (y +. by0) (x +. bx1) (y +. by1)
  | Image { x; y; w; h; _ } -> corners_bounds m x y (x +. w) (y +. h)
  | Group ps -> List.fold_left (fun acc p -> union acc (bounds m p)) None ps
  | Clip { picture; _ } -> bounds m picture
  | Transform { m = m'; picture } -> bounds Affine.(m * m') picture
  | Stamp { picture; xs; ys } ->
      let acc = ref None in
      Array.iteri
        (fun i x ->
          acc := union !acc (bounds Affine.(m * translate x ys.(i)) picture))
        xs;
      !acc

(* Drawing *)

let draw_text canvas clip m ~font ~size ~color ~x ~y text =
  List.iter
    (fun (g, gx) ->
      let path = Font.glyph_path font ~size g in
      let polys = Flatten.path Affine.(m * translate (x +. gx) y) path in
      fill_polys canvas clip color `Nonzero polys)
    (Font.glyphs font ~size text)

(* [blit canvas clip tile ox oy] composites [tile] with its corner at [(ox,
   oy)]. *)
let blit canvas clip (tile : canvas) ox oy =
  let x0 = Int.max clip.x0 ox and y0 = Int.max clip.y0 oy in
  let x1 = Int.min clip.x1 (ox + tile.w)
  and y1 = Int.min clip.y1 (oy + tile.h) in
  for py = y0 to y1 - 1 do
    for px = x0 to x1 - 1 do
      let ti = (((py - oy) * tile.w) + (px - ox)) * 4 in
      let sa = Array1.unsafe_get tile.px (ti + 3) in
      if sa > 0 then begin
        let k = mask_at clip px py in
        if k > 0. then
          blend canvas.px
            (((py * canvas.w) + px) * 4)
            (float (Array1.unsafe_get tile.px ti))
            (float (Array1.unsafe_get tile.px (ti + 1)))
            (float (Array1.unsafe_get tile.px (ti + 2)))
            (float sa) k
      end
    done
  done

let create w h =
  let px = Array1.create int8_unsigned c_layout (w * h * 4) in
  Array1.fill px 0;
  { w; h; px }

let rec draw canvas clip m (p : Picture.t) =
  if not (clip_is_empty clip) then
    match p with
    | Empty -> ()
    | Fill { rule; color; path } ->
        fill_polys canvas clip color rule (Flatten.path m path)
    | Stroke { stroke; color; path } ->
        let polys =
          Stroker.outline stroke ~scale:(linear_scale m) (Flatten.path m path)
        in
        fill_polys canvas clip color `Nonzero polys
    | Text { font; size; color; x; y; text } ->
        draw_text canvas clip m ~font ~size ~color ~x ~y text
    | Image { x; y; w; h; data } -> draw_image canvas clip m ~x ~y ~w ~h data
    | Group ps -> List.iter (draw canvas clip m) ps
    | Clip { path; picture } ->
        draw canvas (intersect_clip clip (Flatten.path m path)) m picture
    | Transform { m = m'; picture } -> draw canvas clip Affine.(m * m') picture
    | Stamp { picture; xs; ys } -> (
        (* Draw the stamp once at the origin under the linear part of [m], then
           blit it at every position rounded to the pixel grid. *)
        let lin = { m with x0 = 0.; y0 = 0. } in
        match bounds lin picture with
        | None -> ()
        | Some (bx0, by0, bx1, by1) ->
            let tx0 = int_of_float (Float.floor bx0) - 1
            and ty0 = int_of_float (Float.floor by0) - 1 in
            let tw = int_of_float (Float.ceil bx1) + 1 - tx0
            and th = int_of_float (Float.ceil by1) + 1 - ty0 in
            let tile = create tw th in
            let tile_clip = { x0 = 0; y0 = 0; x1 = tw; y1 = th; mask = None } in
            draw tile tile_clip
              Affine.(translate (float (-tx0)) (float (-ty0)) * lin)
              picture;
            Array.iteri
              (fun i x ->
                let dx, dy = Affine.apply m x ys.(i) in
                if Float.is_finite dx && Float.is_finite dy then
                  blit canvas clip tile
                    (int_of_float (Float.round dx) + tx0)
                    (int_of_float (Float.round dy) + ty0))
              xs)

let render ?(background = Color.transparent) ~width ~height picture =
  if width <= 0 || height <= 0 then
    invalid_arg "Hugin_vg_raster.render: width and height must be positive";
  let canvas = create width height in
  let br, bg, bb, ba = premultiplied background in
  if ba > 0. then
    for i = 0 to (width * height) - 1 do
      Array1.unsafe_set canvas.px (4 * i) (truncate (br +. 0.5));
      Array1.unsafe_set canvas.px ((4 * i) + 1) (truncate (bg +. 0.5));
      Array1.unsafe_set canvas.px ((4 * i) + 2) (truncate (bb +. 0.5));
      Array1.unsafe_set canvas.px ((4 * i) + 3) (truncate (ba +. 0.5))
    done;
  draw canvas
    { x0 = 0; y0 = 0; x1 = width; y1 = height; mask = None }
    Affine.id picture;
  (* Back to straight alpha. *)
  for i = 0 to (width * height) - 1 do
    let a = Array1.unsafe_get canvas.px ((4 * i) + 3) in
    if a > 0 && a < 255 then
      for c = 0 to 2 do
        let v = Array1.unsafe_get canvas.px ((4 * i) + c) in
        Array1.unsafe_set canvas.px
          ((4 * i) + c)
          (Int.min 255 (((v * 255) + (a / 2)) / a))
      done
  done;
  Nx.of_bigarray (reshape (genarray_of_array1 canvas.px) [| height; width; 4 |])
