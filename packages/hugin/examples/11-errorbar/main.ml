(* Error bars.

   Errorbar marks show measurement uncertainty. Use `Symmetric for equal +/-
   errors or `Asymmetric for independent lower and upper bounds. *)

open Hugin

let () =
  let x = Nx.create Nx.float32 [| 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let y = Nx.create Nx.float32 [| 6 |] [| 2.1; 3.8; 3.2; 5.1; 4.5; 6.3 |] in
  let err = Nx.create Nx.float32 [| 6 |] [| 0.3; 0.5; 0.2; 0.6; 0.4; 0.3 |] in
  errorbar ~x ~y ~yerr:(`Symmetric err) ~cap_size:6. ~color:Color.blue ()
  |> title "Measurements" |> xlabel "Trial" |> ylabel "Value"
  |> render_png "errorbar.png"
