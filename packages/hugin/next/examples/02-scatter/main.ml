(* A scatter of three clusters, coloured by cluster, sized by weight. *)

open Hugin_next

let () =
  let n = 300 in
  let key = Nx.Rng.key 2 in
  let keys = Nx.Rng.split ~n:3 key in
  let cluster = Nx.cast Nx.int32 (Nx.div_s (Nx.arange Nx.int32 0 n 1) 100l) in
  let centre = Nx.create Nx.float64 [| 3; 2 |] [| 0.; 0.; 3.; 1.; 1.; 3. |] in
  let at = Nx.take ~axis:0 ~indices:(Nx.cast Nx.int64 cluster) centre in
  let p = Nx.add at (Nx.Rng.normal keys.(0) Nx.float64 [| n; 2 |]) in
  let weight = Nx.Rng.uniform keys.(1) Nx.float64 [| n |] in
  dot
    ~x:(num ~title:(Text.v "x") Nx.(slice [ A; I 0 ] p))
    ~y:(num ~title:(Text.v "y") Nx.(slice [ A; I 1 ] p))
    ~fill:(cat ~title:(Text.v "cluster") ~labels:[| "a"; "b"; "c" |] cluster)
    ~size:(num ~title:(Text.v "weight") weight)
    ~opacity:(const 0.8) ()
  |> save "scatter.png"
