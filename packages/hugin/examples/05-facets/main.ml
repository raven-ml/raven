(* Small multiples: one panel per run, sharing every scale. *)

open Hugin

let () =
  let steps = 100 in
  let t = Nx.linspace Nx.float64 0. 10. steps in
  let rate =
    Nx.reshape [| 4; 1 |]
      (Nx.create Nx.float64 [| 4 |] [| 0.2; 0.4; 0.6; 0.8 |])
  in
  let y = Nx.mul (Nx.exp (Nx.neg (Nx.mul_s (Nx.mul rate t) 0.5))) (Nx.cos t) in
  line ~x:(num ~title:"time" t) ~y:(num ~title:"signal" y)
    ~fx:(dim ~title:"damping" ~labels:[| "0.2"; "0.4"; "0.6"; "0.8" |] 0)
    ()
  |> save ~size:(Size.figure 480. 200.) "facets.png"
