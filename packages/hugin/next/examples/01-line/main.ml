(* Loss curves: one line per seed, coloured by the seed, in the default theme
   and in the dark one. *)

open Hugin_next

let () =
  let steps = 200 in
  let t = Nx.linspace Nx.float64 0. 1. steps in
  let seeds =
    Nx.reshape [| 3; 1 |] (Nx.create Nx.float64 [| 3 |] [| 1.; 1.3; 1.7 |])
  in
  let noise =
    Nx.mul_s (Nx.Rng.normal (Nx.Rng.key 1) Nx.float64 [| 3; steps |]) 0.03
  in
  let losses = Nx.add (Nx.exp (Nx.mul (Nx.neg t) (Nx.mul_s seeds 4.))) noise in
  let f =
    line
      ~x:(num ~title:(Text.v "progress") t)
      ~y:(num ~title:(Text.v "loss") losses)
      ~stroke:(dim ~title:(Text.v "seed") 0)
      ()
    |> title (Text.v "Training loss")
  in
  save "line.png" f;
  save ~theme:Theme.dark "line-dark.png" f
