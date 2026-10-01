(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The renderers' budgets: 100,000 stamped markers and a stroked polyline of a
   million points on a 640 by 480 page at density 1, and the scatter of the
   first release's budget, 100,000 dots on a 360 by 240 page, to PNG at density
   2. *)

open Hugin_next_gg
open Hugin_next_vg

let () = Random.init 42
let n = 100_000
let xs = Array.init n (fun _ -> Random.float 640.)
let ys = Array.init n (fun _ -> Random.float 480.)
let disc = Path.circle (P2.v 0. 0.) 3.

let marker =
  Picture.group
    [
      Picture.fill (Color.v 0.2 0.4 0.8) disc;
      Picture.stroke (Stroke.v 0.5) Color.white disc;
    ]

let page p = Renderable.v 640. 480. p
let stamps = page (Picture.stamp xs ys marker)

let coloured =
  let fills =
    Array.init n (fun i -> if i mod 2 = 0 then Color.red else Color.blue)
  in
  page (Picture.stamp ~fills xs ys marker)

let polyline =
  let m = 1_000_000 in
  let px = Array.init m (fun i -> 640. *. float i /. float m) in
  let py = Array.make m 240. in
  for i = 1 to m - 1 do
    py.(i) <-
      Float.max 0. (Float.min 480. (py.(i - 1) +. Random.float 2. -. 1.))
  done;
  page (Picture.stroke (Stroke.v 1.) Color.black (Path.polyline px py))

let scatter =
  let sx = Array.map (fun x -> x *. 360. /. 640.) xs in
  let sy = Array.map (fun y -> y *. 240. /. 480.) ys in
  let dot = Path.circle (P2.v 0. 0.) 1.5 in
  Renderable.v 360. 240.
    (Picture.group
       [
         Picture.fill Color.white (Path.rect (Box2.v 0. 0. 360. 240.));
         Picture.stamp sx sy (Picture.fill (Color.v 0.2 0.4 0.8) dot);
       ])

(* Scaled instances of an opacity, each drawn in a layer of its own. *)
let faded =
  let scales = Array.init n (fun _ -> 0.5 +. Random.float 1.5) in
  page (Picture.stamp ~scales xs ys (Picture.opacity 0.5 marker))

let raster =
  let render r () = Hugin_next_vg_raster.render ~density:1. r in
  Thumper.group "raster"
    [
      Thumper.bench "100k stamps" (render stamps);
      Thumper.bench "100k coloured stamps" (render coloured);
      Thumper.bench "100k scaled faded stamps" (render faded);
      Thumper.bench "1M-point polyline" (render polyline);
      Thumper.bench "100k-dot scatter to PNG" (fun () ->
          Hugin_next_vg_raster.png ~density:2. scatter);
    ]

let () = Thumper.run "hugin_next_vg" [ raster ]
