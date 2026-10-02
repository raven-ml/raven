(* Calibration: the observed frequency of the positives against the predicted
   probability, binned, for two models, over the diagonal [y = x] of a perfectly
   calibrated one. *)

open Hugin_next

let () =
  let bins = 10 in
  let p = Nx.linspace Nx.float64 0.05 0.95 bins in
  (* An overconfident model and an underconfident one. *)
  let over = Nx.add_s (Nx.mul_s (Nx.sub_s p 0.5) 0.7) 0.5 in
  let under = Nx.pow_s p 1.4 in
  let observed = Nx.stack [ over; under ] in
  let unit = Scale.linear ~domain:(0., 1.) ~notation:Percent in
  let x = num ~scale:(unit ~name:"x" ()) ~title:(Text.v "predicted") p in
  let y = num ~scale:(unit ~name:"y" ()) ~title:(Text.v "observed") observed in
  let model = dim ~title:(Text.v "model") ~labels:[| "A"; "B" |] 0 in
  layer
    [
      abline ~slope:(const 1.) ~intercept:(const 0.) ~dash:(const Dash.dashed)
        ();
      line ~x ~y ~stroke:model ();
      dot ~x ~y ~fill:model ();
    ]
  |> coord (Coord.cartesian ~aspect:1. ())
  |> title (Text.v "Calibration")
  |> save ~size:(Size.figure 320. 260.) "calibration.png"
