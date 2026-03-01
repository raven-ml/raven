(* Axis decorations.

   Decorations control axes limits, scales, and grid visibility. They compose
   naturally with the |> pipeline. *)

open Hugin

let () =
  let x = Nx.linspace Nx.float32 1. 1000. 100 in
  let y = Nx.log x in
  line ~x ~y () |> title "Logarithmic Scale" |> xlabel "x" |> ylabel "ln(x)"
  |> xscale `Log |> xlim 1. 1000. |> ylim 0. 8.
  |> xtick_format (Printf.sprintf "%.0f")
  |> grid_lines true
  |> render_png "decorations.png"
