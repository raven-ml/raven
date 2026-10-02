(* A training dashboard: the loss, raw and smoothed, over epoch boundaries, and
   the validation accuracy with its best value written out. The run's history
   stands in for what a run monitor records. *)

open Hugin

let steps = 3_000
let epoch = 500

let history () =
  let step = Nx.linspace Nx.float64 0. (float (steps - 1)) steps in
  let noise = Nx.Rng.normal (Nx.Rng.key 11) Nx.float64 [| steps |] in
  let loss =
    Nx.mul
      (Nx.add_s (Nx.exp (Nx.mul_s step (-0.0015))) 0.05)
      (Nx.exp (Nx.mul_s noise 0.25))
  in
  let vstep =
    Nx.linspace Nx.float64 (float epoch) (float steps) (steps / epoch)
  in
  let acc =
    Nx.create Nx.float64
      [| steps / epoch |]
      [| 0.61; 0.74; 0.81; 0.84; 0.835; 0.851 |]
  in
  let epochs = Nx.linspace Nx.float64 0. (float steps) ((steps / epoch) + 1) in
  (step, loss, vstep, acc, epochs)

let () =
  let step, loss, vstep, acc, epochs = history () in
  let at t = Nx.take ~indices:(Nx.reshape [| 1 |] (Nx.argmax acc)) t in
  let losses =
    layer
      [
        rule ~x:(num epochs) ~opacity:(const 0.15) ();
        line ~x:(num ~title:"step" step)
          ~y:(num ~scale:(Scale.log ()) ~title:"loss" loss)
          ~opacity:(const 0.3) ();
        line ~x:(num step) ~y:(num (Nx.ewma ~alpha:0.02 loss)) ();
      ]
  and accuracy =
    layer
      [
        line ~x:(num vstep) ~y:(num ~title:"val. accuracy" acc) ();
        dot ~x:(num vstep) ~y:(num acc) ();
        text
          ~x:(num (at vstep))
          ~y:(num (at acc))
          ~text:(num (at acc))
          ~dy:8. ();
      ]
  in
  grid [ [ losses ]; [ accuracy ] ]
  |> share [ ("x", `Shared) ]
  |> save ~size:(Size.figure 420. 320.) "dashboard.png"
