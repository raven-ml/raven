(* Line styles: three optimisers told apart by colour and dash pattern from one
   scale, so one legend, set inside the panel's empty corner; a dotted target
   line and a frame around the panel. *)

open Hugin_next

let () =
  let steps = 120 in
  let t = Nx.linspace Nx.float64 0. 1. steps in
  let rates = Nx.create Nx.float64 [| 3; 1 |] [| 2.; 3.5; 6. |] in
  let floor = Nx.create Nx.float64 [| 3; 1 |] [| 0.3; 0.2; 0.25 |] in
  let loss =
    Nx.add floor (Nx.exp (Nx.mul (Nx.neg t) rates))
    |> Nx.add
         (Nx.mul_s
            (Nx.Rng.normal (Nx.Rng.key 16) Nx.float64 [| 3; steps |])
            0.01)
  in
  let optimiser = Scale.band ~name:"optimiser" () in
  let run () =
    dim ~scale:optimiser ~title:(Text.v "optimiser")
      ~labels:[| "SGD"; "Adam"; "Lion" |]
      0
  in
  layer
    [
      rule ~y:(floats [| 0.25 |]) ~dash:(const Dash.dotted) ();
      line
        ~x:(num ~title:(Text.v "progress") t)
        ~y:(num ~title:(Text.v "loss") loss)
        ~stroke:(run ()) ~dash:(run ()) ();
      frame ();
      legend ~side:(`Inside `Top_right) "optimiser";
    ]
  |> title (Text.v "Validation loss")
  |> save "styles.png"
