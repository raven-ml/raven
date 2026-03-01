(* Layers and legends.

   Use layers to overlay different mark types on shared axes. Any mark with a
   ~label automatically appears in the legend. *)

open Hugin

let () =
  let x = Nx.linspace Nx.float32 0. 10. 100 in
  let y = Nx.sin x in
  layers
    [
      fill_between ~x ~y1:(Nx.sub_s y 0.3) ~y2:(Nx.add_s y 0.3) ~label:"± 0.3"
        ();
      line ~x ~y ~label:"sin(x)" ();
      hline ~y:0. ~line_style:`Dashed ~color:Color.gray ~label:"baseline" ();
    ]
  |> title "Sine with Confidence Band"
  |> xlabel "x" |> ylabel "y" |> legend |> render_png "layers.png"
