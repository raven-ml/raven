(* Histograms.

   Histogram marks bin continuous data. Use ~density:true to normalize so the
   total area equals 1. *)

open Hugin

let () =
  (* Seeded: the build regenerates the committed image, so the data has to be
     the same on every run. *)
  Nx.Rng.with_key (Nx.Rng.key 42) @@ fun () ->
  let samples = Nx.rand Nx.float32 [| 1000 |] in
  hist ~x:samples ~bins:(`Num 25) ~density:true ~color:Color.green ()
  |> title "Distribution" |> xlabel "Value" |> render_png "histogram.png"
