(* Full showcase.

   Demonstrates multiple mark types, layouts, themes, and output formats in a
   single example. *)

open Hugin

let () =
  let x = Nx.linspace Nx.float32 0. 10. 100 in

  let p1 =
    layers
      [
        line ~x ~y:(Nx.sin x) ~label:"sin" ~color:Color.blue ();
        point
          ~x:(Nx.mul_s (Nx.rand Nx.float32 [| 30 |]) 10.)
          ~y:(Nx.sub_s (Nx.mul_s (Nx.rand Nx.float32 [| 30 |]) 2.) 1.)
          ~color:Color.vermillion ~marker:Star ~label:"noise" ();
      ]
    |> title "Lines & Scatter" |> legend
  in

  let p2 =
    let xb = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
    let h = Nx.create Nx.float32 [| 4 |] [| 3.; 7.; 2.; 5. |] in
    bar ~x:xb ~height:h ~color:Color.orange () |> title "Bar Chart"
  in

  let p3 =
    hist ~x:(Nx.rand Nx.float32 [| 500 |]) ~bins:(`Num 20) ~color:Color.green ()
    |> title "Histogram"
  in

  let p4 =
    let xs = Nx.rand Nx.float32 [| 50 |] in
    let ys = Nx.rand Nx.float32 [| 50 |] in
    let cb = Nx.mul_s xs 100. in
    let sb = Nx.mul_s ys 40. in
    point ~x:xs ~y:ys ~color_by:cb ~size_by:sb ~marker:Circle ()
    |> title "color_by + size_by" |> xlabel "x" |> ylabel "y"
  in

  let p5 =
    let xl = Nx.linspace Nx.float32 1. 100. 50 in
    line ~x:xl ~y:(Nx.mul xl xl) ~color:Color.purple ()
    |> title "Quadratic (log y)" |> yscale `Log
  in

  let p6 =
    let data =
      Nx.init Nx.float32 [| 8; 10 |] (fun idx ->
          let i = Float.of_int idx.(0) and j = Float.of_int idx.(1) in
          Float.sin (i *. 0.5) *. Float.cos (j *. 0.4))
    in
    heatmap ~data ~cmap:Cmap.viridis () |> title "Heatmap"
  in

  let spec = grid [ [ p1; p2 ]; [ p3; p4 ]; [ p5; p6 ] ] in
  spec |> render_png "showcase.png";
  spec |> render_svg "showcase.svg"
