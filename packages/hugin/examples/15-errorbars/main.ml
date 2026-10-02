(* Error bars: the mean accuracy of four methods over five seeds, each a dot
   over a rule from one standard deviation below the mean to one above. *)

open Hugin

let () =
  let methods = [| "baseline"; "dropout"; "mixup"; "ours" |] in
  let base = Nx.create Nx.float64 [| 4; 1 |] [| 0.71; 0.74; 0.76; 0.8 |] in
  let runs =
    Nx.add base
      (Nx.mul_s (Nx.Rng.normal (Nx.Rng.key 15) Nx.float64 [| 4; 5 |]) 0.02)
  in
  let mean = Nx.mean ~axes:[ 1 ] runs and std = Nx.std ~axes:[ 1 ] runs in
  let x = strings ~title:(Text.v "method") methods in
  layer
    [
      rule ~x ~y:(num (Nx.sub mean std)) ~y2:(num (Nx.add mean std)) ();
      dot ~x ~y:(num ~title:(Text.v "accuracy") mean) ();
    ]
  |> title (Text.v "Accuracy over five seeds")
  |> save "errorbars.png"
