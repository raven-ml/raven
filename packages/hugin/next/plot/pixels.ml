(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Box2 = Hugin_next_gg.Box2
module Picture = Hugin_next_vg.Picture

(* Gathering *)

type plan = { window : Box2.t; rows : int array; cols : int array }

(* The raster renderer shows an image smaller than its cells by averaging 4 by 4
   samples per device pixel, at [(p + (i + 0.5) / 4)] for pixel [p] and sample
   [i], each reading the cell under it. A gathered image holds exactly those
   cells, one per sample, over the device pixels the box reaches, so that it is
   shown at 4 cells per pixel and each sample reads its own. *)
let samples = 4

let sampled ~density lo hi n =
  let p0 = Float.to_int (Float.floor (lo *. density))
  and p1 = Float.to_int (Float.ceil (hi *. density)) in
  let inv = 1. /. density and len = hi -. lo in
  let at k =
    let d =
      float (p0 + (k / samples))
      +. ((float (k mod samples) +. 0.5) /. float samples)
    in
    let u = ((d *. inv) -. lo) /. len in
    if u >= 0. && u <= 1. then Int.min (n - 1) (Float.to_int (u *. float n))
    else -1
  in
  (p0, p1, Array.init ((p1 - p0) * samples) at)

let plan ~density box ~rows ~cols =
  let shown l = Float.max 1. (l *. density) in
  let ratio =
    Float.max
      (float cols /. shown (Box2.w box))
      (float rows /. shown (Box2.h box))
  in
  if ratio <= float samples then None
  else
    let x0, x1, cs = sampled ~density (Box2.minx box) (Box2.maxx box) cols in
    let y0, y1, rs = sampled ~density (Box2.miny box) (Box2.maxy box) rows in
    let px p = float p /. density in
    let window = Box2.v (px x0) (px y0) (px (x1 - x0)) (px (y1 - y0)) in
    Some { window; rows = rs; cols = cs }

(* [rgba px] is the [[|h; w; c|]] image [px], [c] being 1, 3 or 4, as RGBA. *)
let rgba px =
  let s = Nx.shape px in
  match s.(2) with
  | 4 -> px
  | c ->
      let opaque = Nx.full Nx.uint8 [| s.(0); s.(1); 1 |] 255 in
      let rgb = if c = 1 then [ px; px; px ] else [ px ] in
      Nx.concatenate ~axis:2 (rgb @ [ opaque ])

let gather p px =
  let px = rgba px in
  let h = (Nx.shape px).(0) and w = (Nx.shape px).(1) in
  (* A last transparent row and column stand for the samples outside. *)
  let padded = Nx.pad [| (0, 1); (0, 1); (0, 0) |] 0 px in
  let indices outside a =
    Nx.create Nx.int64
      [| Array.length a |]
      (Array.map (fun i -> Int64.of_int (if i < 0 then outside else i)) a)
  in
  padded
  |> Nx.take ~axis:0 ~indices:(indices h p.rows)
  |> Nx.take ~axis:1 ~indices:(indices w p.cols)

(* Images under a transform or a stamp are not on the page's device pixels, and
   are left as they are. *)
let rec gathered ~density (p : Picture.t) =
  match p with
  | Image { box; pixels } -> (
      let shape = Nx.shape pixels in
      match plan ~density box ~rows:shape.(0) ~cols:shape.(1) with
      | None -> p
      | Some plan -> Picture.image plan.window (gather plan pixels))
  | Group ps -> Picture.group (List.map (gathered ~density) ps)
  | Clip { rule; path; picture } ->
      Picture.clip ~rule path (gathered ~density picture)
  | Opacity { opacity; picture } ->
      Picture.opacity opacity (gathered ~density picture)
  | Tag { tag; picture } -> Picture.tag tag (gathered ~density picture)
  | Empty | Fill _ | Stroke _ | Glyphs _ | Transform _ | Stamp _ -> p
