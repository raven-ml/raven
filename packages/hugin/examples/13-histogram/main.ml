(* Histograms: the weights of a layer at initialisation and after training, one
   histogram per row over shared bins, drawn as rects from their edges. *)

open Hugin

let weights n =
  let keys = Nx.Rng.split ~n:3 (Nx.Rng.key 13) in
  let init = Nx.mul_s (Nx.Rng.normal keys.(0) Nx.float32 [| 2 * n |]) 0.05 in
  let narrow = Nx.mul_s (Nx.Rng.normal keys.(1) Nx.float32 [| n |]) 0.03 in
  let wide = Nx.mul_s (Nx.Rng.normal keys.(2) Nx.float32 [| n |]) 0.12 in
  Nx.stack ~axis:0 [ init; Nx.concatenate ~axis:0 [ narrow; wide ] ]

let () =
  let h = Stats.histogram ~bins:40 (weights 2_500) in
  rect
    ~x:(num ~title:(Text.v "weight") h.x)
    ~x2:(num h.x2)
    ~y:(num ~title:(Text.v "density") h.density)
    ~fill:(dim ~labels:[| "initialisation"; "trained" |] 0)
    ~opacity:(const 0.6) ()
  |> save "histogram.png"
