(* Grid layout.

   Arrange independent plots in a grid. Each cell is a standalone specification
   with its own axes and decorations. *)

open Hugin

let () =
  let x = Nx.linspace Nx.float32 0. (2. *. Float.pi) 100 in
  let p1 = line ~x ~y:(Nx.sin x) () |> title "sin" in
  let p2 = line ~x ~y:(Nx.cos x) () |> title "cos" in
  let p3 = line ~x ~y:(Nx.tan (Nx.mul_s x 0.3)) () |> title "tan(0.3x)" in
  let p4 =
    point ~x ~y:(Nx.sin x) ~color:Color.vermillion ~marker:Plus ()
    |> title "sin (scatter)"
  in
  grid [ [ p1; p2 ]; [ p3; p4 ] ] |> render_png "grid_layout.png"
