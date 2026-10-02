(* Grouped scores: a bar per class, coloured by metric, side by side. *)

open Hugin

let () =
  let classes = [| "cat"; "dog"; "bird"; "fish" |] in
  let scores = Nx.create Nx.float64 [| 4 |] [| 0.92; 0.88; 0.71; 0.64 |] in
  rect
    ~x:(strings ~title:(Text.v "class") classes)
    ~y:(num ~title:(Text.v "accuracy") scores)
    ~fill:(strings classes) ()
  |> save "bars.png"
