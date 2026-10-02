(* A heatmap: a matrix drawn as cells, its rows and columns by index. *)

open Hugin

let () =
  let x = Nx.linspace Nx.float64 (-3.) 3. 40 in
  let y = Nx.reshape [| 30; 1 |] (Nx.linspace Nx.float64 (-2.) 2. 30) in
  let z = Nx.add (Nx.sin x) (Nx.cos (Nx.mul_s y 1.5)) in
  rect
    ~x:(dim ~title:(Text.v "column") 1)
    ~y:(dim ~title:(Text.v "row") 0)
    ~fill:(num ~title:(Text.v "value") z)
    ()
  |> save "heatmap.png"
