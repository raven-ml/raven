(* A two-column paper figure: four panels labelled (a) to (d), set at 8 points
   in the bundled faces, 180 mm wide. *)

open Hugin

let panel s f = title ~align:`Left (Text.bold (Text.v s)) f

let () =
  let t = Nx.linspace Nx.float64 0. 6. 120 in
  let k =
    Nx.reshape [| 3; 1 |] (Nx.create Nx.float64 [| 3 |] [| 1.; 2.; 3. |])
  in
  let waves = Nx.sin (Nx.mul k t) in
  let a =
    line
      ~x:(num ~title:(Text.v "time (s)") t)
      ~y:(num ~title:(Text.v "amplitude") waves)
      ~stroke:(dim ~title:(Text.v "harmonic") ~labels:[| "1"; "2"; "3" |] 0)
      ()
  in
  let n = Nx.linspace Nx.float64 1. 4. 30 in
  let b =
    dot
      ~x:
        (num ~scale:(Scale.log ()) ~title:(Text.v "parameters")
           (Nx.init Nx.float64 [| 30 |] (fun i ->
                Float.pow 10. (1. +. (3. *. float i.(0) /. 29.)))))
      ~y:(num ~title:(Text.v "error") (Nx.div (Nx.ones_like n) n))
      ()
  in
  let methods = [| "ours"; "base"; "prior" |] in
  let c =
    rect
      ~x:(strings ~title:(Text.v "method") methods)
      ~y:
        (num ~title:(Text.v "score")
           (Nx.create Nx.float64 [| 3 |] [| 0.82; 0.71; 0.76 |]))
      ~fill:(strings ~scale:(Scale.band ~name:"method" ()) methods)
      ()
  in
  let z =
    Nx.mul
      (Nx.sin (Nx.reshape [| 20; 1 |] (Nx.linspace Nx.float64 0. 3. 20)))
      (Nx.cos (Nx.linspace Nx.float64 0. 3. 24))
  in
  let d =
    rect ~x:(dim 1) ~y:(dim 0) ~fill:(num ~title:(Text.v "response") z) ()
  in
  grid [ [ panel "(a)" a; panel "(b)" b ]; [ panel "(c)" c; panel "(d)" d ] ]
  |> save ~theme:(Theme.v ~size:8. ())
       ~size:(Size.figure (Size.mm 180.) (Size.mm 110.))
       "paper.png"
