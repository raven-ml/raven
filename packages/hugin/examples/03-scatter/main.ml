(* Scatter plots.

   Point marks place individual markers at data coordinates. Use color_by to map
   a third variable through the theme's sequential colormap. *)

open Hugin

let () =
  let x = Nx.rand Nx.float32 [| 200 |] in
  let y = Nx.rand Nx.float32 [| 200 |] in
  let c = Nx.add x y in
  point ~x ~y ~color_by:c ~size:8. ~marker:Circle ()
  |> title "Random Scatter" |> render_png "scatter.png"
