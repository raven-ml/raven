(* Histograms.

   Histogram marks bin continuous data. Use ~density:true to normalize so the
   total area equals 1. *)

open Hugin

let () =
  let samples = Nx.rand Nx.float32 [| 1000 |] in
  hist ~x:samples ~bins:(`Num 25) ~density:true ~color:Color.green ()
  |> title "Distribution" |> xlabel "Value" |> render_png "histogram.png"
