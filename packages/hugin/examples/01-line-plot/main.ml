(* Your first plot.

   The simplest possible visualization: create data with Nx, build a line plot,
   and render to PNG. *)

open Hugin

let () =
  let x = Nx.linspace Nx.float32 0. (2. *. Float.pi) 100 in
  let y = Nx.sin x in
  line ~x ~y () |> render_png "line_plot.png"
