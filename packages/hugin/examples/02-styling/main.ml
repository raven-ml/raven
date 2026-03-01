(* Styling.

   Every mark constructor accepts optional visual properties as labeled
   arguments. This example shows how to set color, line style, width, and marker
   shape. *)

open Hugin

let () =
  let x = Nx.linspace Nx.float32 0. (2. *. Float.pi) 50 in
  let y = Nx.sin x in
  line ~x ~y ~color:Color.vermillion ~line_style:`Dashed ~line_width:2.5
    ~marker:Triangle ~alpha:0.7 ()
  |> render_png "styling.png"
