(* Bar charts.

   Bar marks draw vertical bars centered at x positions. Height is measured from
   bottom (default 0). *)

open Hugin

let () =
  let x = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let h = Nx.create Nx.float32 [| 5 |] [| 4.2; 7.1; 3.8; 9.0; 5.5 |] in
  bar ~x ~height:h ~color:Color.sky_blue ()
  |> title "Quarterly Revenue" |> xlabel "Quarter" |> ylabel "Revenue ($M)"
  |> xticks [ (1., "Q1"); (2., "Q2"); (3., "Q3"); (4., "Q4"); (5., "Q5") ]
  |> render_png "bar_chart.png"
