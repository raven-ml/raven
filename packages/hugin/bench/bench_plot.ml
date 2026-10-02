(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A live training curve: a dashboard of a run planned for 3,000 steps, its loss
   over epoch boundaries above its validation accuracy, sharing x. At step 2,000
   the loss grows by one step and the figure is redrawn through [resolve],
   [layout] and [draw], each given the stage's output of the step before. The
   boundaries and the accuracy are the same tensors in both. *)

open Hugin

let steps = 3_000
let epoch = 500
let at = 2_000
let size = Size.figure 420. 320.
let step = Nx.linspace Nx.float64 0. (float (steps - 1)) steps

let loss =
  let noise = Nx.Rng.normal (Nx.Rng.key 11) Nx.float64 [| steps |] in
  Nx.mul
    (Nx.add_s (Nx.exp (Nx.mul_s step (-0.0015))) 0.05)
    (Nx.exp (Nx.mul_s noise 0.25))

let epochs = Nx.linspace Nx.float64 0. (float steps) ((steps / epoch) + 1)
let vstep = Nx.linspace Nx.float64 (float epoch) (float at) (at / epoch)
let acc = Nx.create Nx.float64 [| at / epoch |] [| 0.61; 0.74; 0.81; 0.84 |]

let accuracy =
  layer
    [
      line ~x:(num vstep) ~y:(num ~title:"val. accuracy" acc) ();
      dot ~x:(num vstep) ~y:(num acc) ();
    ]

let dashboard n =
  let upto t = Nx.(slice [ R (0, n) ] t) in
  let losses =
    layer
      [
        rule ~x:(num epochs) ~opacity:(const 0.15) ();
        line
          ~x:(num ~title:"step" (upto step))
          ~y:(num ~scale:(Scale.log ()) ~title:"loss" (upto loss))
          ();
      ]
  in
  grid [ [ losses ]; [ accuracy ] ] |> share [ ("x", `Shared) ]

type shown = { r : Resolved.t; l : Layout.t; d : Drawing.t }

let show fig =
  let r = resolve fig in
  let l = layout size r in
  { r; l; d = draw ~density:2. l }

let redraw prev fig =
  let r = resolve ~prev:prev.r fig in
  let l = layout ~prev:prev.l size r in
  { r; l; d = draw ~prev:prev.d ~density:2. l }

let live =
  Thumper.bench_with_setup "live curve redraw"
    ~setup:(fun () -> show (dashboard at))
    (fun prev -> redraw prev (dashboard (at + 1)))

(* A 3 × 3 grid of cells of magnitudes from 10^-4 to 10^4, each a pair of facet
   panels with a colour legend, under a title: grids nested three deep, whose
   guides the layout stacks and aligns. *)
let facet_pairs =
  let cell k =
    let v = 10. ** float k in
    let x = Nx.create Nx.float64 [| 2 |] [| -.v; v |] in
    dot
      ~fx:(strings [| "left"; "right" |])
      ~fill:(strings [| "alpha"; "b" |])
      ~x:(num x)
      ~y:(num ~title:(Text.v "value") x)
      ()
  in
  grid (List.init 3 (fun r -> List.init 3 (fun c -> cell ((3 * r) + c - 4))))
  |> title (Text.v "A figure")

let nested =
  let r = resolve facet_pairs in
  Thumper.bench "layout of nested grids" (fun () -> layout size r)

(* The budgets of large data: each figure is drawn from its tensors to a PNG
   file at density 2, every stage included. *)

let png size f =
  Hugin_vg_raster.png ~density:2. (Drawing.renderable (render size f))

let walk = Nx.cumsum (Nx.Rng.normal (Nx.Rng.key 1) Nx.float32 [| 10_000_000 |])

(* A million steps of the walk with one in a thousand missing. A gap costs the
   same at any length, and the 10M-step line holds the length's budget. *)
let gappy =
  let walk = Nx.slice [ R (0, 1_000_000) ] walk in
  let missing =
    Nx.less_s (Nx.Rng.uniform (Nx.Rng.key 4) Nx.float32 [| 1_000_000 |]) 0.001
  in
  Nx.where missing (Nx.full_like walk Float.nan) walk

(* A line whose x is in no order changes column at almost every row, so that M4
   keeps almost every row. *)
let shuffled = Nx.Rng.uniform (Nx.Rng.key 5) Nx.float32 [| 1_000_000 |]
let shuffled_walk = Nx.slice [ R (0, 1_000_000) ] walk
let cloud = Nx.Rng.normal (Nx.Rng.key 2) Nx.float32 [| 100_000; 2 |]

(* Attention weights of 12 layers by 12 heads over 128 tokens. *)
let weights = Nx.Rng.uniform (Nx.Rng.key 3) Nx.float32 [| 12; 12; 128; 128 |]

let budgets =
  Thumper.group "budgets"
    [
      Thumper.bench "10M-step line to PNG" (fun () ->
          png (Size.figure 360. 240.) (line ~y:(num walk) ()));
      Thumper.bench "1M-step line with gaps to PNG" (fun () ->
          png (Size.figure 360. 240.) (line ~y:(num gappy) ()));
      Thumper.bench "1M-step line over unsorted x to PNG" (fun () ->
          png (Size.figure 360. 240.)
            (line ~x:(num shuffled) ~y:(num shuffled_walk) ()));
      Thumper.bench "100k dots to PNG" (fun () ->
          png (Size.figure 360. 240.)
            (dot
               ~x:(num Nx.(slice [ A; I 0 ] cloud))
               ~y:(num Nx.(slice [ A; I 1 ] cloud))
               ()));
      Thumper.bench "attention at T = 128 to PNG" (fun () ->
          png (Size.figure 800. 800.)
            (rect ~fy:(dim 0) ~fx:(dim 1) ~y:(dim 2) ~x:(dim 3)
               ~fill:(num weights) ()));
    ]

let config = Thumper.Config.(default |> deadline 60.)

let () =
  Thumper.run ~config "hugin_next_plot"
    [ Thumper.group "stages" [ live; nested ]; budgets ]
  Thumper.run ~config "hugin_plot" [ Thumper.group "stages" [ live ]; budgets ]
  |> exit
