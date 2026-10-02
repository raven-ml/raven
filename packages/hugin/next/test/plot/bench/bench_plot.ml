(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A live training curve: a dashboard of a run planned for 3,000 steps, its loss
   over epoch boundaries above its validation accuracy, sharing x. At step 2,000
   the loss grows by one step and the figure is redrawn through [resolve],
   [layout] and [draw], each given the stage's output of the step before. The
   boundaries and the accuracy are the same tensors in both. *)

open Hugin_next

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
      line ~x:(num vstep) ~y:(num ~title:(Text.v "val. accuracy") acc) ();
      dot ~x:(num vstep) ~y:(num acc) ();
    ]

let dashboard n =
  let upto t = Nx.(slice [ R (0, n) ] t) in
  let losses =
    layer
      [
        rule ~x:(num epochs) ~opacity:(const 0.15) ();
        line
          ~x:(num ~title:(Text.v "step") (upto step))
          ~y:(num ~scale:(Scale.log ()) ~title:(Text.v "loss") (upto loss))
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

let () = Thumper.run "hugin_next_plot" [ Thumper.group "stages" [ live ] ]
