(* Areas: the validation loss of eight seeds over training, the band between the
   lowest and the highest seed at each step, and the mean as a line. *)

open Hugin

let () =
  let seeds = 8 and steps = 120 in
  let t = Nx.linspace Nx.float64 0. 1. steps in
  let rates =
    Nx.add_s
      (Nx.mul_s (Nx.Rng.uniform (Nx.Rng.key 14) Nx.float64 [| seeds; 1 |]) 2.)
      3.
  in
  let noise =
    Nx.mul_s (Nx.Rng.normal (Nx.Rng.key 15) Nx.float64 [| seeds; steps |]) 0.02
  in
  let losses = Nx.add (Nx.exp (Nx.mul (Nx.neg t) rates)) (Nx.add_s noise 0.1) in
  let progress = num ~title:(Text.v "progress") t in
  layer
    [
      area ~x:progress
        ~y:(num ~title:(Text.v "validation loss") (Nx.max ~axes:[ 0 ] losses))
        ~y2:(num (Nx.min ~axes:[ 0 ] losses))
        ~opacity:(const 0.3) ();
      line ~x:progress ~y:(num (Nx.mean ~axes:[ 0 ] losses)) ();
    ]
  |> title (Text.v "Validation loss over eight seeds")
  |> save "area.png"
